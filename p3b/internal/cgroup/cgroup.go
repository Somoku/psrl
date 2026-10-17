// Package cgroup gives the sandbox service one cgroup subtree to measure and bound.
//
// Admission's ceilings are fractions of a domain, and a fraction is only useful if
// the domain is the thing being managed. Reading this process's own cgroup is not
// that: on a node that shares its cgroup with a trainer, the fraction describes the
// trainer's page cache as much as the sandboxes, so "the node is 49% full" can be
// true while every sandbox together holds 5% of it. The fix is a subtree that
// contains sandboxes and nothing else, created before Docker creates any container
// and pointed at by both the sampler and the container runtime.
//
// Three things are not obvious and are the reason this is a package rather than a
// few lines in main:
//
//   - A v1 host mounts one hierarchy per controller, and each hierarchy has its own
//     cgroup path. The subtree and this process's leaf must exist in every hierarchy
//     that matters, or a container placed by Docker ends up in a parent that has no
//     limit.
//   - Under a cgroup namespace the process's own path is not visible under the mount;
//     the mount root *is* the process's cgroup. Both cases have to resolve, and only
//     the root may fall back — a configured subtree that does not exist is a
//     configuration error, because silently reading the root is the over-broad domain
//     this package exists to stop using.
//   - A v1 memory subtree reports its descendants' usage only when
//     memory.use_hierarchy is on. With it off the sampler reads zero and every
//     ceiling becomes unreachable, so it is verified rather than assumed.
package cgroup

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	// SandboxParent is the subtree Docker creates sandboxes into and the sampler
	// reads. It is a sibling of Docker's own "docker" default parent, so the domain
	// excludes every container this service did not create.
	SandboxParent = "psrl-sandboxes"

	// HostLeaf holds this service's own process. The design asks for it so the
	// parent has no process of its own: cgroup v2 forbids enabling controllers on a
	// cgroup that has both processes and children ("no internal processes"), and on
	// v1 it simply keeps the measured domain free of the service.
	HostLeaf = "psrl-host"

	// MiB is the unit the API states memory in.
	MiB = int64(1024 * 1024)

	// cpuPeriodMicros matches the period the container body requests, so a quota
	// written here means the same thing to the subtree and to each container.
	cpuPeriodMicros = int64(100000)

	// setupLimitTolerance is how far an existing subtree's limit may differ from the
	// resolved one before it is reported as a configuration change. The resolved value
	// moves with measured memory, so an exact comparison reports drift as drift.
	setupLimitTolerance = 0.05
)

// knownControllers is what a v1 mount's option list can name. The list only has to
// contain what a deployment can mount; an unknown name is ignored rather than
// treated as an error, because a kernel may add one.
var knownControllers = map[string]struct{}{
	"blkio": {}, "cpu": {}, "cpuacct": {}, "cpuset": {}, "devices": {},
	"freezer": {}, "hugetlb": {}, "memory": {}, "misc": {}, "net_cls": {},
	"net_prio": {}, "perf_event": {}, "pids": {}, "rdma": {},
}

// Hierarchy is one mounted cgroup hierarchy and the cgroup this process belongs to
// in it.
type Hierarchy struct {
	// Controllers are the controller names the mount carries, e.g. ["cpu", "cpuacct"].
	Controllers []string
	// Mount is the mount point, which is the ancestor of Root in the non-namespaced
	// case and the same directory as Root in the namespaced one.
	Mount string
	// Root is this process's own cgroup: where the subtree is created and where the
	// ancestor's limits are read from.
	Root string
}

// Has reports whether the hierarchy carries a controller.
func (h Hierarchy) Has(controller string) bool {
	for _, name := range h.Controllers {
		if name == controller {
			return true
		}
	}
	return false
}

// Domain is every cgroup hierarchy this process can reach.
type Domain struct {
	// Version is 2 for a unified host and 1 for a split one.
	Version int
	// Hierarchies is one entry per mount, in mount order.
	Hierarchies []Hierarchy
}

// Request is what the node asks the tree to enforce.
type Request struct {
	// MemoryMB is the node's declared envelope: what it lends to the fleet.
	MemoryMB int64
	// Overcommit is the dynamic density multiple. The subtree limit is the most
	// admission may grant (envelope x overcommit), so the kernel bound and the
	// admission bound are the same number and an admitted burst cannot meet an
	// unexpected kernel limit.
	Overcommit float64
	// CPUMillis is the CPU the node sells. The subtree's quota, so a CPU fraction is
	// of what this node promised rather than of the machine.
	CPUMillis int64
	// Safety is the share of the ancestor's limit kept free for page cache growth,
	// kernel memory and work that is not a sandbox. Zero means the default.
	Safety float64
}

// Budget is what the tree can actually enforce.
type Budget struct {
	// SubtreeLimitBytes is what is written to the subtree's memory limit.
	SubtreeLimitBytes int64
	// DeclaredMB is the envelope admission must use, so that
	// DeclaredMB x Overcommit == SubtreeLimitBytes. It is lower than Request.MemoryMB
	// when the ancestor cannot hold what the node declared.
	DeclaredMB int64
	// AncestorLimitBytes and UnreclaimableBytes explain the cap in the startup log.
	AncestorLimitBytes int64
	UnreclaimableBytes int64
	// CPUMillis is the quota written to the subtree, so a CPU fraction is of what
	// this node sells.
	CPUMillis int64
	// RequestedMB is the envelope the node asked for, for the log line.
	RequestedMB int64
}

// Capped reports whether the ancestor reduced what the node asked for.
func (b Budget) Capped() bool { return b.DeclaredMB < b.RequestedMB }

// Resolve discovers the hierarchies this process can reach.
//
// cgroup v2 is tried first because a unified host is the one its own tooling reads,
// and a host carrying both is a host migrating to v2. A host with neither is an
// error rather than an empty domain: an empty domain would read as zero pressure,
// which is the state that fills a machine.
func Resolve() (Domain, error) {
	return resolveFrom("/proc/self/mounts", "/proc/self/cgroup")
}

func resolveFrom(mountsPath, cgroupPath string) (Domain, error) {
	mounts, err := readMounts(mountsPath)
	if err != nil {
		return Domain{}, err
	}
	paths, err := readProcessPaths(cgroupPath)
	if err != nil {
		return Domain{}, err
	}
	var v1 []Hierarchy
	for _, mount := range mounts {
		hierarchy, ok := resolveHierarchy(mount, paths)
		if !ok {
			continue
		}
		if mount.filesystem == "cgroup2" {
			return Domain{Version: 2, Hierarchies: []Hierarchy{hierarchy}}, nil
		}
		v1 = append(v1, hierarchy)
	}
	if len(v1) == 0 {
		return Domain{}, fmt.Errorf("cgroup: no readable cgroup hierarchy under %s", mountsPath)
	}
	return Domain{Version: 1, Hierarchies: v1}, nil
}

// String describes the domain for a startup log, naming the subtree's ancestor.
func (d Domain) String() string {
	parts := make([]string, 0, len(d.Hierarchies))
	for _, h := range d.Hierarchies {
		// A named hierarchy ("name=systemd") carries no controller, so it is labelled
		// by its mount point rather than by an empty list.
		if len(h.Controllers) == 0 {
			parts = append(parts, fmt.Sprintf("%s=%s", filepath.Base(h.Mount), h.Root))
			continue
		}
		if h.Root == h.Mount {
			parts = append(parts, fmt.Sprintf("%s=%s (mount root is the process cgroup)", strings.Join(h.Controllers, ","), h.Root))
			continue
		}
		parts = append(parts, fmt.Sprintf("%s=%s", strings.Join(h.Controllers, ","), h.Root))
	}
	return fmt.Sprintf("cgroup v%d: %s", d.Version, strings.Join(parts, "; "))
}

// Budget resolves how much memory the subtree may hold without asking the ancestor
// for memory it does not have.
//
// The ancestor is the process's own cgroup, which on a containerised deployment is
// the pod. Page cache charged to it is reclaimable and is not counted; resident
// anonymous memory, kernel memory and a safety share are, because the sandbox
// subtree cannot take them.
func (d Domain) Budget(req Request) (Budget, error) {
	memory, ok := d.memory()
	if !ok {
		return Budget{}, fmt.Errorf("cgroup: %s has no memory hierarchy", d.String())
	}
	if req.MemoryMB <= 0 {
		return Budget{}, fmt.Errorf("cgroup: the sandbox envelope must be positive, got %dMB", req.MemoryMB)
	}
	if req.Overcommit < 1 {
		return Budget{}, fmt.Errorf("cgroup: overcommit %.3f would make the subtree smaller than the envelope", req.Overcommit)
	}
	safety := req.Safety
	if safety <= 0 {
		safety = 0.05
	}
	if safety >= 1 {
		return Budget{}, fmt.Errorf("cgroup: safety %.3f must leave room for the sandboxes", safety)
	}

	ancestorLimit, err := memoryLimit(memory.Root)
	if err != nil {
		return Budget{}, fmt.Errorf("cgroup: reading the ancestor's memory limit: %w", err)
	}
	unreclaimable, err := unreclaimableBytes(memory.Root)
	if err != nil {
		return Budget{}, fmt.Errorf("cgroup: reading the ancestor's unreclaimable memory: %w", err)
	}
	capacity := ancestorLimit - unreclaimable - int64(float64(ancestorLimit)*safety)
	if capacity <= 0 {
		return Budget{}, fmt.Errorf(
			"cgroup: the ancestor at %s can hold no sandbox memory: limit %dMiB, unreclaimable %dMiB, "+
				"safety %.0f%% leaves %dMiB", memory.Root, ancestorLimit/MiB, unreclaimable/MiB, safety*100, capacity/MiB)
	}

	declared := req.MemoryMB * MiB
	if declared > capacity {
		return Budget{}, fmt.Errorf(
			"cgroup: this node declares %dMiB but only %dMiB of %s remains after %dMiB of unreclaimable "+
				"memory and a %.0f%% safety margin; lower node.memory_mb or overcommit, or move the other "+
				"workloads out of this cgroup",
			req.MemoryMB, capacity/MiB, memory.Root, unreclaimable/MiB, safety*100)
	}

	want := int64(float64(declared) * req.Overcommit)
	if want > capacity {
		want = capacity
	}
	budget := Budget{
		SubtreeLimitBytes:  want,
		DeclaredMB:         int64(float64(want) / req.Overcommit / float64(MiB)),
		AncestorLimitBytes: ancestorLimit,
		UnreclaimableBytes: unreclaimable,
		CPUMillis:          req.CPUMillis,
		RequestedMB:        req.MemoryMB,
	}
	if budget.DeclaredMB <= 0 {
		return Budget{}, fmt.Errorf("cgroup: the resolved subtree limit %d bytes is smaller than one MiB", want)
	}
	return budget, nil
}

// Setup creates the sandbox subtree and this service's leaf, writes the limits, and
// moves this process into the leaf.
//
// The order is deliberate: the budget is resolved before anything is created, so an
// impossible request fails without leaving a half-built tree; the limits are written
// before the process moves, so a failure to move does not leave the service in a leaf
// whose sandbox domain does not exist.
func (d Domain) Setup(parent, leaf string, req Request) (Budget, error) {
	budget, err := d.Budget(req)
	if err != nil {
		return Budget{}, err
	}
	for _, hierarchy := range d.Hierarchies {
		if err := d.create(hierarchy, parent, budget); err != nil {
			return Budget{}, err
		}
		if err := createLeaf(hierarchy, leaf); err != nil {
			return Budget{}, err
		}
	}
	if err := d.moveTo(leaf); err != nil {
		return Budget{}, err
	}
	return budget, nil
}

// SetupReadOnly resolves the budget without creating anything or moving this
// process, for a preflight that must not have side effects.
func (d Domain) SetupReadOnly(parent string, req Request) (Budget, error) {
	budget, err := d.Budget(req)
	if err != nil {
		return Budget{}, err
	}
	memory, ok := d.memory()
	if !ok {
		return Budget{}, fmt.Errorf("cgroup: %s has no memory hierarchy", d.String())
	}
	dir := filepath.Join(memory.Root, parent)
	if _, statErr := os.Stat(dir); statErr == nil {
		existing, readErr := memoryLimit(dir)
		// The limit is derived from measured memory that moves every second, so two
		// resolutions of the same configuration legitimately differ by a few MiB. What this
		// check is for is a *configuration* change behind a running node, which moves the
		// limit by whole percent: a band separates the two without reporting drift.
		if readErr == nil && existing > 0 {
			drift := math.Abs(float64(existing-budget.SubtreeLimitBytes)) / float64(budget.SubtreeLimitBytes)
			if drift > setupLimitTolerance {
				return Budget{}, fmt.Errorf(
					"cgroup: %s already exists with a %dMiB limit but this configuration resolves to %dMiB; "+
						"a running node must be restarted to change it",
					dir, existing/MiB, budget.SubtreeLimitBytes/MiB)
			}
		}
	}
	return budget, nil
}

// create makes one hierarchy's subtree and writes its limits.
func (d Domain) create(hierarchy Hierarchy, parent string, budget Budget) error {
	dir := filepath.Join(hierarchy.Root, parent)
	if _, err := os.Stat(dir); err != nil {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			if hierarchy.Has("memory") {
				return fmt.Errorf("cgroup: creating the sandbox subtree %s: %w", dir, err)
			}
			// A hierarchy we cannot create is one Docker will create without a limit.
			// That is tolerable for a controller nothing measures; it is not
			// tolerable for memory, which is handled above.
			return nil
		}
	}
	if hierarchy.Has("memory") {
		if err := ensureHierarchicalAccounting(hierarchy.Root); err != nil {
			return err
		}
		if err := writeMemoryLimits(dir, budget.SubtreeLimitBytes); err != nil {
			return err
		}
	}
	if hierarchy.Has("cpu") {
		if err := writeCPUQuota(dir, budget, d.Version); err != nil {
			return err
		}
	}
	if hierarchy.Has("cpuset") {
		if err := inheritCPUSet(hierarchy.Root, dir); err != nil {
			return err
		}
	}
	return nil
}

// memory returns the memory hierarchy.
func (d Domain) memory() (Hierarchy, bool) {
	for _, hierarchy := range d.Hierarchies {
		if hierarchy.Has("memory") {
			return hierarchy, true
		}
	}
	return Hierarchy{}, false
}

// createLeaf makes this service's own cgroup in one hierarchy.
func createLeaf(hierarchy Hierarchy, leaf string) error {
	dir := filepath.Join(hierarchy.Root, leaf)
	if _, err := os.Stat(dir); err == nil {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		if hierarchy.Has("memory") || hierarchy.Has("cpu") {
			return fmt.Errorf("cgroup: creating the service leaf %s: %w", dir, err)
		}
		return nil
	}
	return nil
}

// moveTo moves this process into the leaf in every hierarchy that has one.
//
// cgroup.procs is written rather than tasks because it moves the whole thread group in
// one operation; tasks moves one thread, and a thread forked concurrently in the Go
// runtime would be left behind in the parent.
func (d Domain) moveTo(leaf string) error {
	pid := strconv.Itoa(os.Getpid())
	for _, hierarchy := range d.Hierarchies {
		dir := filepath.Join(hierarchy.Root, leaf)
		if _, err := os.Stat(dir); err != nil {
			continue
		}
		if err := os.WriteFile(filepath.Join(dir, "cgroup.procs"), []byte(pid), 0); err != nil {
			if !hierarchy.Has("memory") && !hierarchy.Has("cpu") {
				continue
			}
			return fmt.Errorf("cgroup: moving this process into %s: %w", dir, err)
		}
		if !holdsProcess(dir, pid) {
			return fmt.Errorf("cgroup: this process did not land in %s after writing cgroup.procs", dir)
		}
	}
	return nil
}

// holdsProcess reports whether pid is listed in the cgroup's cgroup.procs.
func holdsProcess(dir, pid string) bool {
	raw, err := os.ReadFile(filepath.Join(dir, "cgroup.procs"))
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.TrimSpace(line) == pid {
			return true
		}
	}
	return false
}

// ensureHierarchicalAccounting makes a v1 memory hierarchy aggregate descendants.
//
// With memory.use_hierarchy off, a parent's usage_in_bytes excludes its children, so
// the subtree would read zero no matter how much the sandboxes hold and every ceiling
// derived from it would be unreachable. The kernel accepts the write only on a cgroup
// that has no children yet, which is why this runs before the subtree gets any.
func ensureHierarchicalAccounting(root string) error {
	path := filepath.Join(root, "memory.use_hierarchy")
	current, err := os.ReadFile(path)
	if err != nil {
		// v2 has no such knob: hierarchical accounting is the only mode.
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("cgroup: reading %s: %w", path, err)
	}
	if strings.TrimSpace(string(current)) == "1" {
		return nil
	}
	if err := os.WriteFile(path, []byte("1"), 0); err != nil {
		return fmt.Errorf(
			"cgroup: %s reports hierarchical accounting off and does not accept turning it on: %w; "+
				"a subtree measured there would always read zero", path, err)
	}
	raw, err := os.ReadFile(path)
	if err != nil || strings.TrimSpace(string(raw)) != "1" {
		return fmt.Errorf("cgroup: %s still reports hierarchical accounting off after writing 1", path)
	}
	return nil
}

// writeMemoryLimits writes the subtree's memory bound.
//
// On a v1 host with swap accounting the kernel requires
// memory.limit_in_bytes <= memory.memsw.limit_in_bytes. A fresh cgroup is unlimited in
// both, so the pair moves in the order that keeps the invariant at every step: memsw is
// raised first if it is somehow below the target, memory is set, and memsw is then
// brought down to match. Writing memsw down first fails with EINVAL, which is how this
// was found.
func writeMemoryLimits(dir string, bytes int64) error {
	memsw := filepath.Join(dir, "memory.memsw.limit_in_bytes")
	hasMemsw := readable(memsw)
	if hasMemsw {
		if current, err := readIntFile(memsw); err == nil && current >= 0 && current < bytes {
			if err := writeInt(memsw, bytes); err != nil {
				return fmt.Errorf("cgroup: raising %s: %w", memsw, err)
			}
		}
	}
	limit := filepath.Join(dir, "memory.limit_in_bytes")
	if err := writeInt(limit, bytes); err != nil {
		return fmt.Errorf("cgroup: writing %s: %w", limit, err)
	}
	got, err := readIntFile(limit)
	if err != nil {
		return fmt.Errorf("cgroup: re-reading %s: %w", limit, err)
	}
	if got != bytes {
		return fmt.Errorf("cgroup: %s reports %d after writing %d", limit, got, bytes)
	}
	if hasMemsw {
		if err := writeInt(memsw, bytes); err != nil {
			return fmt.Errorf("cgroup: writing %s: %w", memsw, err)
		}
	}
	return nil
}

// writeCPUQuota bounds the subtree's CPU at what the node sells.
func writeCPUQuota(dir string, budget Budget, version int) error {
	if budget.CPUMillis <= 0 {
		return nil
	}
	if version == 2 {
		path := filepath.Join(dir, "cpu.max")
		if err := os.WriteFile(path, []byte(fmt.Sprintf("%d %d", budget.CPUMillis*100, cpuPeriodMicros)), 0); err != nil {
			return fmt.Errorf("cgroup: writing %s: %w", path, err)
		}
		return nil
	}
	period := filepath.Join(dir, "cpu.cfs_period_us")
	if err := writeInt(period, cpuPeriodMicros); err != nil {
		return fmt.Errorf("cgroup: writing %s: %w", period, err)
	}
	quota := filepath.Join(dir, "cpu.cfs_quota_us")
	if err := writeInt(quota, budget.CPUMillis*100); err != nil {
		return fmt.Errorf("cgroup: writing %s: %w", quota, err)
	}
	return nil
}

// inheritCPUSet copies the parent's CPU and memory sets into a new v1 cpuset cgroup.
//
// A freshly created v1 cpuset cgroup has empty cpuset.cpus and cpuset.mems and cannot
// accept a process or a container until they are set, which would fail every create
// under the subtree.
func inheritCPUSet(parent, dir string) error {
	for _, pair := range [][2]string{{"cpuset.cpus", "cpuset.effective_cpus"}, {"cpuset.mems", "cpuset.effective_mems"}} {
		field, effective := pair[0], pair[1]
		target := filepath.Join(dir, field)
		if raw, err := os.ReadFile(target); err == nil && strings.TrimSpace(string(raw)) != "" {
			continue
		}
		value := ""
		if raw, err := os.ReadFile(filepath.Join(parent, field)); err == nil {
			value = strings.TrimSpace(string(raw))
		}
		if value == "" {
			if raw, err := os.ReadFile(filepath.Join(parent, effective)); err == nil {
				value = strings.TrimSpace(string(raw))
			}
		}
		if value == "" {
			// The kernel always publishes the effective set of an online CPU, so an
			// empty one means the hierarchy is not the one scheduling this machine.
			return fmt.Errorf("cgroup: %s has no CPU set to inherit for %s", parent, field)
		}
		if err := os.WriteFile(target, []byte(value), 0); err != nil {
			return fmt.Errorf("cgroup: writing %s: %w", target, err)
		}
	}
	return nil
}

// memoryLimit reads a cgroup's memory limit in bytes.
//
// An unlimited v1 cgroup reports a sentinel near the maximum int64 rather than "max",
// and an unlimited v2 cgroup reports "max". Both mean the machine, which is the real
// ceiling.
func memoryLimit(dir string) (int64, error) {
	for _, name := range []string{"memory.limit_in_bytes", "memory.max"} {
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			continue
		}
		value := strings.TrimSpace(string(raw))
		if value == "max" {
			return machineMemory()
		}
		limit, convErr := strconv.ParseInt(value, 10, 64)
		if convErr != nil {
			return 0, fmt.Errorf("parsing %s: %w", name, convErr)
		}
		total, totalErr := machineMemory()
		if totalErr == nil && limit >= total {
			return total, nil
		}
		if limit <= 0 {
			return 0, fmt.Errorf("%s reports %d", name, limit)
		}
		return limit, nil
	}
	return 0, fmt.Errorf("%s has neither memory.limit_in_bytes nor memory.max", dir)
}

// unreclaimableBytes is what the ancestor holds that the sandbox subtree cannot take.
//
// Page cache is excluded on purpose: it is charged to the cgroup but the kernel drops
// it under pressure, so counting it would cap the subtree at a fraction of what the
// machine can actually run. Resident anonymous memory and kernel memory are not
// reclaimable and do bound it.
func unreclaimableBytes(dir string) (int64, error) {
	if raw, err := os.ReadFile(filepath.Join(dir, "memory.stat")); err == nil {
		fields := parseStat(string(raw))
		if v2, ok := fields["anon"]; ok {
			return v2 + fields["slab_unreclaimable"] + fields["kernel_stack"], nil
		}
		if v1, ok := fields["total_rss"]; ok {
			return v1 + readIntOrZero(filepath.Join(dir, "memory.kmem.usage_in_bytes")), nil
		}
	}
	return readIntOrZero(filepath.Join(dir, "memory.kmem.usage_in_bytes")), nil
}

// parseStat reads a memory.stat "key value" file into a map of bytes.
func parseStat(raw string) map[string]int64 {
	out := map[string]int64{}
	for _, line := range strings.Split(raw, "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			continue
		}
		value, err := strconv.ParseInt(fields[1], 10, 64)
		if err != nil {
			continue
		}
		out[fields[0]] = value
	}
	return out
}

func readIntOrZero(path string) int64 {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	value, err := strconv.ParseInt(strings.TrimSpace(string(raw)), 10, 64)
	if err != nil || value < 0 {
		return 0
	}
	return value
}

// readable reports whether a file can be opened.
func readable(path string) bool {
	file, err := os.Open(path)
	if err != nil {
		return false
	}
	_ = file.Close()
	return true
}

// parseInt reads an integer field from a "key value" file.
func parseInt(value string) (int64, error) {
	return strconv.ParseInt(strings.TrimSpace(value), 10, 64)
}

// readIntFile reads one integer from a one-value file.
func readIntFile(path string) (int64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(strings.TrimSpace(string(raw)), 10, 64)
}

func writeInt(path string, value int64) error {
	return os.WriteFile(path, []byte(strconv.FormatInt(value, 10)), 0)
}

// machineMemory is the ceiling of a cgroup that sets none.
func machineMemory() (int64, error) {
	raw, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, err
	}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 || fields[0] != "MemTotal:" {
			continue
		}
		kilobytes, convErr := strconv.ParseInt(fields[1], 10, 64)
		if convErr != nil {
			return 0, convErr
		}
		return kilobytes * 1024, nil
	}
	return 0, fmt.Errorf("/proc/meminfo has no MemTotal")
}
