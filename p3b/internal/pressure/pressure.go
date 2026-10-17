// Package pressure measures what a machine is actually doing right now.
//
// Admission needs this because its own accounting is a record of reservations
// rather than of load: a node that granted 32 GB of requests whose sandboxes are
// each using 200 MB has room, and a node that granted 8 GB to sandboxes that are
// all at their limit does not. Ranking or admitting on the reservation alone
// gets both cases wrong in the expensive direction -- it refuses work the
// machine could take, and it accepts work that pushes the host into swap.
//
// The reading comes from the cgroup hierarchy rather than from the host's /proc,
// and the difference matters. /proc/meminfo describes the machine including
// everything outside this service -- a trainer's model weights, a page cache the
// kernel will drop under pressure -- so a sandbox service reading it refuses work
// because of memory it does not control and cannot release. Reading this
// service's own cgroup subtree measures the sandboxes, which is the quantity its
// ceilings are about.
//
// Both cgroup versions are supported, because which one a host runs is not a
// choice this service gets to make and a v1 host is not a host whose ceilings
// may silently stop being enforced. v2 is tried first and v1 is the fallback;
// they differ only in which files hold the two numbers, so one Reader reads
// either through a source.
//
// CPU is a rate rather than a level, so it cannot be read from one sample. Two
// samples of the cumulative counter are differenced over the wall time between
// them, which is why a Reader is stateful and why the first call reports zero
// rather than guessing.
package pressure

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"psrl.dev/sandboxd/internal/node"
)

// cgroupRoot is where a cgroup hierarchy is mounted on Linux, in either version.
const cgroupRoot = "/sys/fs/cgroup"

// source reads the two cumulative numbers a pressure reading is derived from.
//
// It exists because cgroup v1 and v2 hold the same quantities in different files
// with different units, and the arithmetic over them -- a rate from two samples,
// a fraction against a limit -- is identical. Keeping the version difference to
// this interface means the ceilings cannot behave differently per host.
type source interface {
	// cpuUsage is cumulative CPU time consumed by this cgroup.
	cpuUsage() (time.Duration, error)
	// memoryUsage is bytes currently charged to this cgroup.
	memoryUsage() (int64, error)
	// memoryLimit is the ceiling a usage reading is a fraction of. A cgroup with
	// no limit of its own reports the machine's memory, which is the real one.
	memoryLimit() (int64, error)
	// describe names the version and path, for the log line that says which is in use.
	describe() string
	// effectiveCores is the capacity a CPU fraction is relative to.
	//
	// It is the domain's own quota when it has one, because the ceilings these
	// fractions feed are about what this node promised to run: a subtree quota'd to
	// 180 cores on a 384-core machine is at 100% when it uses all 180, not at 47%.
	// A domain without a quota falls back to its CPU set and then to the machine.
	effectiveCores() (float64, error)
}

// Reader samples one cgroup subtree.
//
// It is safe for concurrent use: the node agent reads it from its admission path
// and from its report loop, and a torn CPU sample would be a wrong refusal
// rather than a race the race detector would catch.
type Reader struct {
	src source

	// limit is the memory ceiling, resolved once at construction. It is a property
	// of the cgroup rather than of the moment, and re-reading it per sample would
	// spend a syscall on a number that does not move.
	limit int64

	mu       sync.Mutex
	lastCPU  time.Duration
	lastRead time.Time
	cores    float64
}

// NewReader returns a reader for this process's own cgroup.
//
// cgroup v2 is tried first because a host running both has v2 as the one its
// own tooling reads. A host with neither is reported as an error rather than as
// zero pressure: zero is a meaningful reading -- an idle node -- and returning it
// for an unreadable cgroup would make every node look idle and every ceiling
// unreachable, which is the failure that fills a host.
func NewReader() (*Reader, error) {
	v2, v2Err := newV2Source()
	if v2Err == nil {
		return newReaderFrom(v2)
	}
	v1, v1Err := newV1Source()
	if v1Err == nil {
		return newReaderFrom(v1)
	}
	return nil, fmt.Errorf(
		"pressure: no readable cgroup hierarchy (v2: %v; v1: %v); this node reports no measurable pressure",
		v2Err, v1Err)
}

// NewDomainReader returns a reader for a named subtree of this process's cgroup root.
//
// Unlike NewReader it does not fall back to the root after joining the subtree. A
// configured domain that does not exist is a configuration error, because the root is
// the over-broad domain this function exists to stop reading: measured on one of these
// nodes, the root held 1100 GiB of which the containers were 109 GiB, so a ceiling
// applied to it was mostly about memory no sandbox had touched.
func NewDomainReader(subtree string) (*Reader, error) {
	if subtree == "" {
		return nil, fmt.Errorf("pressure: a domain reader needs a subtree name")
	}
	v2, v2Err := newV2DomainSource(subtree)
	if v2Err == nil {
		reader, err := newReaderFrom(v2)
		if err == nil {
			return reader, nil
		}
		v2Err = err
	}
	v1, v1Err := newV1DomainSource(subtree)
	if v1Err != nil {
		return nil, fmt.Errorf(
			"pressure: the domain %q is not readable (v2: %v; v1: %v)", subtree, v2Err, v1Err)
	}
	return newReaderFrom(v1)
}

// newV2DomainSource resolves a subtree under the unified hierarchy.
func newV2DomainSource(subtree string) (*v2Source, error) {
	root, err := v2Dir()
	if err != nil {
		return nil, err
	}
	return v2Domain(root, subtree)
}

// v2Domain joins a subtree under a given root, without ever falling back to it.
func v2Domain(root, subtree string) (*v2Source, error) {
	dir := filepath.Join(root, subtree)
	if !fileReadable(filepath.Join(dir, "cpu.stat")) || !fileReadable(filepath.Join(dir, "memory.current")) {
		return nil, fmt.Errorf("%s is not a readable cgroup v2 subtree under %s", subtree, root)
	}
	return &v2Source{dir: dir}, nil
}

// newV1DomainSource resolves a subtree in each controller's separate hierarchy.
func newV1DomainSource(subtree string) (*v1Source, error) {
	cpuRoot, err := v1ControllerDir(
		[]string{"cpuacct", "cpu,cpuacct", "cpu"},
		[]string{"cpu,cpuacct", "cpuacct"},
		"cpuacct.usage",
	)
	if err != nil {
		return nil, err
	}
	memoryRoot, err := v1ControllerDir([]string{"memory"}, []string{"memory"}, "memory.usage_in_bytes")
	if err != nil {
		return nil, err
	}
	cpusetRoot := ""
	if root, cpusetErr := v1ControllerDir([]string{"cpuset"}, []string{"cpuset"}, "cpuset.cpus"); cpusetErr == nil {
		cpusetRoot = root
	}
	return v1Domain(cpuRoot, memoryRoot, cpusetRoot, subtree)
}

// v1Domain joins a subtree under resolved controller roots, without ever falling
// back to a root: a domain that does not exist is a configuration error.
func v1Domain(cpuRoot, memoryRoot, cpusetRoot, subtree string) (*v1Source, error) {
	cpuDir := filepath.Join(cpuRoot, subtree)
	memoryDir := filepath.Join(memoryRoot, subtree)
	if !fileReadable(filepath.Join(cpuDir, "cpuacct.usage")) {
		return nil, fmt.Errorf("%s has no cpuacct.usage for the domain %q", cpuRoot, subtree)
	}
	if !fileReadable(filepath.Join(memoryDir, "memory.usage_in_bytes")) {
		return nil, fmt.Errorf("%s has no memory.usage_in_bytes for the domain %q", memoryRoot, subtree)
	}
	cpusetDir := ""
	if cpusetRoot != "" {
		candidate := filepath.Join(cpusetRoot, subtree)
		if fileReadable(filepath.Join(candidate, "cpuset.cpus")) {
			cpusetDir = candidate
		}
	}
	return &v1Source{cpuDir: cpuDir, memoryDir: memoryDir, cpusetDir: cpusetDir}, nil
}

// Describe names the hierarchy this reader is using, so a deployment can confirm
// which one was detected rather than inferring it from behaviour.
func (r *Reader) Describe() string { return r.src.describe() }

// newReaderFrom builds a reader over one source, resolving what does not change.
// MeasuredBytes returns the memory the measured domain is using and the ceiling
// that reading is a fraction of.
//
// A fraction alone does not say what was measured: the same 0.48 means "this
// node's sandboxes hold half its memory" in one deployment and "the outer cgroup
// is half full, for reasons this node does not control" in another. This is what
// Read's MemUsedPct is a fraction of, so a caller comparing it against a ceiling
// can say which domain the ceiling was applied to.
//
// The second result is false when the usage file cannot be read, which is the
// one case a caller must not read as an idle machine.
func (r *Reader) MeasuredBytes() (used, limit int64, ok bool) {
	used, err := r.src.memoryUsage()
	if err != nil {
		return 0, r.limit, false
	}
	return used, r.limit, true
}

func newReaderFrom(src source) (*Reader, error) {
	limit, err := src.memoryLimit()
	if err != nil {
		return nil, fmt.Errorf("pressure: %s has no readable memory limit: %w", src.describe(), err)
	}
	if limit <= 0 {
		return nil, fmt.Errorf("pressure: %s reports a memory limit of %d", src.describe(), limit)
	}
	// Probe the CPU counter at construction: a source whose files appear but
	// cannot be parsed fails here rather than silently reporting zero load on
	// every later sample.
	if _, err := src.cpuUsage(); err != nil {
		return nil, fmt.Errorf("pressure: %s has no readable cpu counter: %w", src.describe(), err)
	}
	cores, err := src.effectiveCores()
	if err != nil || cores <= 0 {
		cores = float64(hostCores())
	}
	if cores <= 0 {
		return nil, fmt.Errorf("pressure: the measured domain and the machine report no usable cores")
	}
	return &Reader{src: src, limit: limit, cores: cores}, nil
}

// Read returns the node's current pressure.
//
// The CPU figure is the rate since the previous call, so the first call after
// construction reports zero: there is no interval to divide by yet. That is
// deliberately not an estimate -- a fabricated first sample would be read by
// admission as a real measurement and could refuse the node's first request.
func (r *Reader) Read() node.Pressure {
	// A sample carries when it was taken and whether it can be trusted. A caller
	// that cannot tell "idle" from "unreadable" decides the same way for both, and
	// for a pressure ceiling that means admitting into a machine nobody can see.
	out := node.Pressure{SampledAt: time.Now(), Valid: true}

	used, err := r.src.memoryUsage()
	if err != nil {
		out.Valid = false
		return out
	}
	out.MemUsedPct = fraction(float64(used), float64(r.limit))

	total, err := r.src.cpuUsage()
	if err != nil {
		out.Valid = false
		return out
	}
	now := time.Now()

	r.mu.Lock()
	defer r.mu.Unlock()
	if r.lastRead.IsZero() {
		r.lastCPU, r.lastRead = total, now
		return out
	}
	elapsed := now.Sub(r.lastRead)
	if elapsed <= 0 {
		return out
	}
	consumed := total - r.lastCPU
	r.lastCPU, r.lastRead = total, now
	if consumed < 0 {
		// The counter reset, which happens when the cgroup is recreated. One
		// sample is lost rather than reported as negative load.
		return out
	}
	// Consumed CPU time over elapsed wall time is cores busy; dividing by the
	// cores available makes it the fraction the ceilings are stated in.
	out.CPUUsedPct = fraction(float64(consumed), float64(elapsed)*r.cores)
	return out
}

// Pressure returns this reader as the function the node agent takes.
func (r *Reader) Pressure() func() node.Pressure {
	return r.Read
}

// fraction divides and clamps into [0, 1].
//
// Clamping matters rather than being defensive: a cgroup can briefly exceed a
// soft memory limit, and the configuration contract states ceilings as fractions,
// so a value above one would be compared against a ceiling that can never be
// reached.
func fraction(part, whole float64) float64 {
	if whole <= 0 {
		return 0
	}
	value := part / whole
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

// -- cgroup v2 -----------------------------------------------------------------

// v2Source reads the unified hierarchy, where one directory holds every
// controller's files.
type v2Source struct{ dir string }

func newV2Source() (*v2Source, error) {
	dir, err := v2Dir()
	if err != nil {
		return nil, err
	}
	return &v2Source{dir: dir}, nil
}

func (s *v2Source) describe() string { return "cgroup v2 at " + s.dir }

// effectiveCores is the quota, then the CPU set, then the machine.
func (s *v2Source) effectiveCores() (float64, error) {
	if cores, ok := quotaCoresV2(s.dir); ok {
		return cores, nil
	}
	if cores, ok := cpusetCores(s.dir, "cpuset.cpus.effective", "cpuset.cpus"); ok {
		return cores, nil
	}
	return float64(hostCores()), nil
}

// cpuUsage reads cpu.stat, whose usage_usec is the only cumulative field a rate
// can be taken from.
func (s *v2Source) cpuUsage() (time.Duration, error) {
	micros, err := fieldFromFile(filepath.Join(s.dir, "cpu.stat"), "usage_usec")
	if err != nil {
		return 0, err
	}
	return time.Duration(micros) * time.Microsecond, nil
}

func (s *v2Source) memoryUsage() (int64, error) {
	return intFromFile(filepath.Join(s.dir, "memory.current"))
}

// memoryLimit reads memory.max, which is the literal "max" when the cgroup sets
// no limit of its own. The machine's memory is then the real ceiling.
func (s *v2Source) memoryLimit() (int64, error) {
	limit, err := intFromFile(filepath.Join(s.dir, "memory.max"))
	if err == nil && limit > 0 {
		return limit, nil
	}
	return hostMemoryTotal()
}

// v2Dir resolves the unified cgroup this process belongs to.
//
// On v2 /proc/self/cgroup has one entry whose first field is 0 and whose second
// is empty. The third is the path below the mount point.
func v2Dir() (string, error) {
	lines, err := cgroupLines()
	if err != nil {
		return "", err
	}
	for _, parts := range lines {
		if parts[0] != "0" || parts[1] != "" {
			continue
		}
		dir := filepath.Join(cgroupRoot, parts[2])
		if fileReadable(filepath.Join(dir, "cpu.stat")) {
			return dir, nil
		}
		// The path exists in the hierarchy but not on this filesystem, which is
		// what a container sees when its cgroup is namespaced. The mount point is
		// then the process's own root.
		if fileReadable(filepath.Join(cgroupRoot, "cpu.stat")) {
			return cgroupRoot, nil
		}
		return "", fmt.Errorf("cgroup v2 path %s has no readable cpu.stat", dir)
	}
	return "", fmt.Errorf("no cgroup v2 entry in /proc/self/cgroup")
}

// -- cgroup v1 -----------------------------------------------------------------

// v1Source reads the split hierarchy, where each controller is its own mount and
// a process can sit at a different path in each.
//
// The two controllers are resolved separately rather than assumed to share a
// path, because they legitimately differ: a container runtime can place a
// process under one path for cpu and another for memory, and reading memory from
// the cpu path would report the parent's usage.
type v1Source struct {
	cpuDir    string
	memoryDir string
	// cpusetDir is where this domain's CPU set lives, when the host mounts one.
	// Only consulted when the domain carries no quota, and empty when the host has
	// no cpuset mount.
	cpusetDir string
}

func newV1Source() (*v1Source, error) {
	// cpuacct carries the cumulative counter. It is usually mounted together with
	// cpu as "cpu,cpuacct", so both spellings are tried.
	cpuDir, err := v1ControllerDir(
		[]string{"cpuacct", "cpu,cpuacct", "cpu"},
		[]string{"cpu,cpuacct", "cpuacct"},
		"cpuacct.usage",
	)
	if err != nil {
		return nil, err
	}
	memoryDir, err := v1ControllerDir(
		[]string{"memory"},
		[]string{"memory"},
		"memory.usage_in_bytes",
	)
	if err != nil {
		return nil, err
	}
	// The CPU set is optional: a host without a cpuset mount has no set to read, and
	// the quota or the machine's count then answers the same question.
	cpusetDir, cpusetErr := v1ControllerDir([]string{"cpuset"}, []string{"cpuset"}, "cpuset.cpus")
	if cpusetErr != nil {
		cpusetDir = ""
	}
	return &v1Source{cpuDir: cpuDir, memoryDir: memoryDir, cpusetDir: cpusetDir}, nil
}

// effectiveCores is the quota, then the CPU set, then the machine.
func (s *v1Source) effectiveCores() (float64, error) {
	if cores, ok := quotaCores(s.cpuDir); ok {
		return cores, nil
	}
	if cores, ok := cpusetCores(s.cpusetDir, "cpuset.effective_cpus", "cpuset.cpus"); ok {
		return cores, nil
	}
	return float64(hostCores()), nil
}

func (s *v1Source) describe() string {
	return "cgroup v1 at " + s.cpuDir + " (cpu) and " + s.memoryDir + " (memory)"
}

// cpuUsage reads cpuacct.usage, which is nanoseconds rather than the
// microseconds v2 reports.
func (s *v1Source) cpuUsage() (time.Duration, error) {
	nanos, err := intFromFile(filepath.Join(s.cpuDir, "cpuacct.usage"))
	if err != nil {
		return 0, err
	}
	return time.Duration(nanos) * time.Nanosecond, nil
}

func (s *v1Source) memoryUsage() (int64, error) {
	return intFromFile(filepath.Join(s.memoryDir, "memory.usage_in_bytes"))
}

// memoryLimit reads memory.limit_in_bytes.
//
// An unlimited v1 cgroup does not say "max": it reports a sentinel near the
// maximum int64, which differs by kernel and page size. A limit at or above the
// machine's memory is therefore treated as no limit, which is what it means.
func (s *v1Source) memoryLimit() (int64, error) {
	total, totalErr := hostMemoryTotal()
	limit, err := intFromFile(filepath.Join(s.memoryDir, "memory.limit_in_bytes"))
	if err != nil || limit <= 0 {
		return total, totalErr
	}
	if totalErr == nil && limit >= total {
		return total, nil
	}
	return limit, nil
}

// v1ControllerDir resolves where this process sits for one controller.
//
// names are the controller spellings to match in /proc/self/cgroup, mounts are
// the directory names to try under the cgroup root, and probe is a file that
// must be readable for the directory to be the right one.
func v1ControllerDir(names, mounts []string, probe string) (string, error) {
	lines, err := cgroupLines()
	if err != nil {
		return "", err
	}
	var path string
	for _, parts := range lines {
		if parts[1] == "" {
			continue
		}
		for _, controller := range strings.Split(parts[1], ",") {
			for _, name := range names {
				// A controller list "cpu,cpuacct" matches either member, and the
				// list itself matches when a caller names it that way.
				if controller == name || parts[1] == name {
					path = parts[2]
				}
			}
		}
		if path != "" {
			break
		}
	}
	if path == "" {
		return "", fmt.Errorf("no cgroup v1 entry for %v in /proc/self/cgroup", names)
	}
	for _, mount := range mounts {
		// The process's own path first, then the mount root. A namespaced cgroup
		// reports a path it cannot see, and the root is what it actually reads.
		for _, candidate := range []string{
			filepath.Join(cgroupRoot, mount, path),
			filepath.Join(cgroupRoot, mount),
		} {
			if fileReadable(filepath.Join(candidate, probe)) {
				return candidate, nil
			}
		}
	}
	return "", fmt.Errorf("no readable %s for %v under %s", probe, names, cgroupRoot)
}

// -- shared file reading -------------------------------------------------------

// cgroupLines parses /proc/self/cgroup into its three fields per entry.
func cgroupLines() ([][3]string, error) {
	raw, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return nil, fmt.Errorf("reading /proc/self/cgroup: %w", err)
	}
	var out [][3]string
	for _, line := range strings.Split(string(raw), "\n") {
		parts := strings.SplitN(strings.TrimSpace(line), ":", 3)
		if len(parts) != 3 {
			continue
		}
		out = append(out, [3]string{parts[0], parts[1], parts[2]})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("/proc/self/cgroup has no usable entries")
	}
	return out, nil
}

func fileReadable(path string) bool {
	file, err := os.Open(path)
	if err != nil {
		return false
	}
	_ = file.Close()
	return true
}

// intFromFile reads a file holding one integer.
func intFromFile(path string) (int64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	value := strings.TrimSpace(string(raw))
	if value == "max" {
		// v2 states an absent limit this way. The caller decides what the real
		// ceiling is, so this is not an error.
		return 0, nil
	}
	return strconv.ParseInt(value, 10, 64)
}

// fieldFromFile reads one named field from a "key value" file.
func fieldFromFile(path, field string) (int64, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != 2 || fields[0] != field {
			continue
		}
		return strconv.ParseInt(fields[1], 10, 64)
	}
	if err := scanner.Err(); err != nil {
		return 0, err
	}
	return 0, fmt.Errorf("%s has no %s", path, field)
}

// hostMemoryTotal reads the machine's memory, which is the ceiling for a cgroup
// that sets no limit of its own.
func hostMemoryTotal() (int64, error) {
	kilobytes, err := fieldFromFile("/proc/meminfo", "MemTotal:")
	if err != nil {
		// MemTotal's line is "MemTotal:  N kB", so the two-field match above
		// fails on the unit. Parse it directly.
		file, openErr := os.Open("/proc/meminfo")
		if openErr != nil {
			return 0, fmt.Errorf("reading /proc/meminfo: %w", openErr)
		}
		defer file.Close()
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			fields := strings.Fields(scanner.Text())
			if len(fields) < 2 || fields[0] != "MemTotal:" {
				continue
			}
			parsed, convErr := strconv.ParseInt(fields[1], 10, 64)
			if convErr != nil {
				return 0, convErr
			}
			return parsed * 1024, nil
		}
		return 0, fmt.Errorf("/proc/meminfo has no MemTotal")
	}
	return kilobytes * 1024, nil
}
