package pressure

import (
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// hostCores is the last-resort CPU denominator: the machine's count.
//
// It is only reached when the measured domain declares neither a quota nor a CPU set,
// which in practice means a host with no cgroup at all. The domain's own capacity is
// the right denominator when it has one, because the ceilings these fractions feed are
// about what this node promised: a subtree quota'd to 180 cores on a 384-core machine
// is saturated at 100%, not at 47%, and reading it as 47% means the ceiling it feeds
// can never be reached.
func hostCores() int {
	return runtime.NumCPU()
}

// quotaCores returns the cores a v1 domain's CPU quota allows.
func quotaCores(dir string) (float64, bool) {
	if dir == "" {
		return 0, false
	}
	quota, err := readInt(filepath.Join(dir, "cpu.cfs_quota_us"))
	if err != nil || quota <= 0 {
		return 0, false
	}
	period, err := readInt(filepath.Join(dir, "cpu.cfs_period_us"))
	if err != nil || period <= 0 {
		period = 100000
	}
	return float64(quota) / float64(period), true
}

// quotaCoresV2 returns the cores a unified domain's CPU quota allows.
//
// cpu.max holds "<quota> <period>", with "max" in place of the quota when the domain is
// unbounded.
func quotaCoresV2(dir string) (float64, bool) {
	if dir == "" {
		return 0, false
	}
	raw, err := os.ReadFile(filepath.Join(dir, "cpu.max"))
	if err != nil {
		return 0, false
	}
	fields := strings.Fields(string(raw))
	if len(fields) != 2 || fields[0] == "max" {
		return 0, false
	}
	quota, convErr := strconv.ParseInt(fields[0], 10, 64)
	period, periodErr := strconv.ParseInt(fields[1], 10, 64)
	if convErr != nil || periodErr != nil || quota <= 0 || period <= 0 {
		return 0, false
	}
	return float64(quota) / float64(period), true
}

// cpusetCores counts the CPUs a domain may run on.
//
// The effective file is preferred over the configured one: in a namespace the
// configured set can be stated relative to the parent and reads empty, while the
// effective set is what the kernel will actually schedule on.
func cpusetCores(dir string, fields ...string) (float64, bool) {
	if dir == "" {
		return 0, false
	}
	for _, field := range fields {
		raw, err := os.ReadFile(filepath.Join(dir, field))
		if err != nil {
			continue
		}
		if count, ok := countCPUs(strings.TrimSpace(string(raw))); ok {
			return float64(count), true
		}
	}
	return 0, false
}

// countCPUs counts the CPUs in a cgroup CPU list such as "0-7,10,12-13".
func countCPUs(list string) (int, bool) {
	if list == "" {
		return 0, false
	}
	total := 0
	for _, part := range strings.Split(list, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		low, high, ranged := strings.Cut(part, "-")
		first, err := strconv.Atoi(low)
		if err != nil {
			return 0, false
		}
		if !ranged {
			total++
			continue
		}
		last, err := strconv.Atoi(high)
		if err != nil || last < first {
			return 0, false
		}
		total += last - first + 1
	}
	if total == 0 {
		return 0, false
	}
	return total, true
}

// readInt reads one integer from a one-value file.
func readInt(path string) (int64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(strings.TrimSpace(string(raw)), 10, 64)
}
