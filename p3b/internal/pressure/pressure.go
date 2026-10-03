// Package pressure measures what a machine is actually doing right now.
//
// Admission needs this because its own accounting is a record of reservations
// rather than of load: a node that granted 32 GB of requests whose sandboxes are
// each using 200 MB has room, and a node that granted 8 GB to sandboxes that are
// all at their limit does not. Ranking or admitting on the reservation alone
// gets both cases wrong in the expensive direction -- it refuses work the
// machine could take, and it accepts work that pushes the host into swap.
//
// The reading comes from cgroup v2 rather than from the host's /proc, and the
// difference matters. /proc/meminfo describes the machine including everything
// outside this service -- a trainer's model weights, a page cache the kernel will
// drop under pressure -- so a sandbox service reading it refuses work because of
// memory it does not control and cannot release. Reading this service's own
// cgroup subtree measures the sandboxes, which is the quantity its ceilings are
// about.
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

// cgroupRoot is where a cgroup v2 hierarchy is mounted on Linux.
const cgroupRoot = "/sys/fs/cgroup"

// Reader samples one cgroup subtree.
//
// It is safe for concurrent use: the node agent reads it from its admission path
// and from its report loop, and a torn CPU sample would be a wrong refusal
// rather than a race the race detector would catch.
type Reader struct {
	// dir is the cgroup directory this reader measures. It is the service's own
	// cgroup by default, so the numbers describe the sandboxes this node owns
	// rather than everything on the machine.
	dir string

	// memoryLimit is the ceiling the percentage is taken against, resolved once.
	// A cgroup with no limit of its own inherits the machine's memory, which is
	// what makes "used percent" meaningful in both shapes.
	memoryLimit int64

	mu       sync.Mutex
	lastCPU  time.Duration
	lastRead time.Time
	cores    float64
}

// NewReader returns a reader for this process's own cgroup.
//
// A machine with no cgroup v2 hierarchy, or a cgroup whose files cannot be read,
// is reported as an error rather than as zero pressure. Zero is a meaningful
// reading -- an idle node -- and returning it for an unreadable cgroup would make
// every node look idle and every ceiling unreachable, which is the failure that
// fills a host.
func NewReader() (*Reader, error) {
	dir, err := selfCgroupDir()
	if err != nil {
		return nil, err
	}
	return newReaderAt(dir)
}

// newReaderAt returns a reader for one cgroup directory, for a test that
// supplies its own files.
func newReaderAt(dir string) (*Reader, error) {
	limit, err := readMemoryLimit(dir)
	if err != nil {
		return nil, err
	}
	// Probe cpu.stat at construction: fail fast on a misconfigured or missing
	// cgroup rather than on the first Read call, where the error would be silent.
	if _, err := readCPUUsage(dir); err != nil {
		return nil, fmt.Errorf("pressure: cgroup at %s is not usable: %w", dir, err)
	}
	cores := float64(hostCores())
	if cores <= 0 {
		return nil, fmt.Errorf("pressure: the machine reports no usable cores")
	}
	return &Reader{dir: dir, memoryLimit: limit, cores: cores}, nil
}

// Read returns the node's current pressure.
//
// The CPU figure is the rate since the previous call, so the first call after
// construction reports zero: there is no interval to divide by yet. That is
// deliberately not an estimate -- a fabricated first sample would be read by
// admission as a real measurement and could refuse the node's first request.
func (r *Reader) Read() node.Pressure {
	var out node.Pressure

	if used, err := readMemoryCurrent(r.dir); err == nil && r.memoryLimit > 0 {
		out.MemUsedPct = float64(used) / float64(r.memoryLimit)
		if out.MemUsedPct > 1 {
			// A cgroup can briefly exceed a soft limit. Reporting above one would
			// put a fraction where a fraction is expected and fail the config
			// contract's range check downstream.
			out.MemUsedPct = 1
		}
	}

	total, err := readCPUUsage(r.dir)
	if err != nil {
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
	out.CPUUsedPct = float64(consumed) / float64(elapsed) / r.cores
	if out.CPUUsedPct > 1 {
		out.CPUUsedPct = 1
	}
	return out
}

// Pressure returns this reader as the function the node agent takes, so a
// deployment without a readable cgroup can pass nil and get the zero reading
// rather than failing to start.
func (r *Reader) Pressure() func() node.Pressure {
	return r.Read
}

// selfCgroupDir resolves the cgroup this process belongs to.
//
// On cgroup v2 /proc/self/cgroup has one entry whose third field is the path
// below the mount point. A process in the root cgroup reports "/", which
// resolves to the mount point itself.
func selfCgroupDir() (string, error) {
	raw, err := os.ReadFile("/proc/self/cgroup")
	if err != nil {
		return "", fmt.Errorf("pressure: reading /proc/self/cgroup: %w", err)
	}
	for _, line := range strings.Split(string(raw), "\n") {
		parts := strings.SplitN(strings.TrimSpace(line), ":", 3)
		if len(parts) != 3 {
			continue
		}
		// A v2 entry has an empty controller list and an empty subsystem name.
		if parts[0] != "0" || parts[1] != "" {
			continue
		}
		dir := filepath.Join(cgroupRoot, parts[2])
		if _, err := os.Stat(filepath.Join(dir, "cpu.stat")); err != nil {
			// The path exists in the hierarchy but not on this filesystem, which
			// is what a container sees when its cgroup is namespaced. The mount
			// point is then the process's own root.
			return cgroupRoot, nil
		}
		return dir, nil
	}
	return "", fmt.Errorf(
		"pressure: no cgroup v2 entry in /proc/self/cgroup; this node reports no measurable pressure")
}

// readMemoryLimit resolves the ceiling a memory reading is a fraction of.
//
// memory.max is "max" when the cgroup sets no limit of its own, in which case
// the machine's total memory is the real ceiling.
func readMemoryLimit(dir string) (int64, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "memory.max"))
	if err == nil {
		value := strings.TrimSpace(string(raw))
		if value != "max" {
			limit, convErr := strconv.ParseInt(value, 10, 64)
			if convErr == nil && limit > 0 {
				return limit, nil
			}
		}
	}
	total, err := hostMemoryTotal()
	if err != nil {
		return 0, err
	}
	return total, nil
}

func readMemoryCurrent(dir string) (int64, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "memory.current"))
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(strings.TrimSpace(string(raw)), 10, 64)
}

// readCPUUsage returns the cgroup's cumulative CPU time.
//
// cpu.stat states usage_usec in microseconds, which is the only field here that
// a rate can be taken from.
func readCPUUsage(dir string) (time.Duration, error) {
	file, err := os.Open(filepath.Join(dir, "cpu.stat"))
	if err != nil {
		return 0, err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != 2 || fields[0] != "usage_usec" {
			continue
		}
		micros, convErr := strconv.ParseInt(fields[1], 10, 64)
		if convErr != nil {
			return 0, convErr
		}
		return time.Duration(micros) * time.Microsecond, nil
	}
	if err := scanner.Err(); err != nil {
		return 0, err
	}
	return 0, fmt.Errorf("pressure: cpu.stat in %s has no usage_usec", dir)
}

// hostMemoryTotal reads the machine's memory, which is the ceiling for a cgroup
// that sets no limit of its own.
func hostMemoryTotal() (int64, error) {
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return 0, fmt.Errorf("pressure: reading /proc/meminfo: %w", err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 || fields[0] != "MemTotal:" {
			continue
		}
		kilobytes, convErr := strconv.ParseInt(fields[1], 10, 64)
		if convErr != nil {
			return 0, convErr
		}
		return kilobytes * 1024, nil
	}
	return 0, fmt.Errorf("pressure: /proc/meminfo has no MemTotal")
}
