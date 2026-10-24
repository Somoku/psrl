// Per-sandbox reads and writes inside the sandbox subtree.
//
// The domain's aggregate usage is what admission is bounded by, but a controller that
// only sees the aggregate can only make aggregate decisions: it cannot tell an idle
// sandbox holding a large page cache from a busy one, and it cannot hand back the memory
// of one without touching the other. These are the per-child operations that make the
// difference visible and actionable.
package cgroup

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Child is one sandbox's own cgroup.
type Child struct {
	// ID is the directory name, which is the container id Docker created it under.
	ID string
	// Usage is everything charged to the sandbox: its pages, its page cache and its
	// kernel memory.
	Usage int64
	// Resident is what the sandbox cannot give back: anonymous pages plus kernel memory.
	// A limit below this cannot be met without killing the sandbox, which is the floor
	// every shrink respects.
	Resident int64
	// Cache is the reclaimable part, and the only part a shrink actually frees.
	Cache int64
	// Limit is the sandbox's own memory bound.
	Limit int64
	// OOMKills is how many times the kernel has killed a process in this sandbox.
	OOMKills int64
	// CPUUsageNanos is the CPU time this sandbox has consumed since it was created.
	//
	// It is cumulative, so what a caller wants -- a rate -- needs two readings and the
	// time between them. A single reading says nothing about load, and reading it as if
	// it did would price an old sandbox as busy forever.
	CPUUsageNanos int64
	// CPUQuotaMicros is the sandbox's own CPU bound within the kernel's period, and
	// zero when the host bounds it nowhere.
	CPUQuotaMicros int64
	// CPUThrottledNanos is how long the sandbox has been stopped because it used its
	// whole quota for a period.
	//
	// It is the only signal of *unmet* demand. cpuacct usage is censored: a sandbox that
	// wants more CPU than its quota is given only reaches its quota, so the measured rate
	// saturates there and says nothing about how much more was wanted. The throttling
	// counters are the kernel saying the sandbox asked and was refused, which is exactly
	// the signal a pricing controller needs and the only one that can reveal it.
	CPUThrottledNanos int64
	// MemoryFailCount counts how many times the sandbox hit its memory limit and had to
	// reclaim or was killed. It is the memory-side equivalent: usage sits at the limit,
	// and the counter is what says the limit was actually in the way.
	MemoryFailCount int64
}

// Children reads every sandbox cgroup under a parent.
//
// A child that disappeared between the listing and the read is skipped rather than
// reported: a sandbox being released is the normal case at this cadence, not an error.
//
// CPU is read from the accounting hierarchy, which on a v1 host is a mount of its own,
// so a sandbox's two halves are named by the same directory in two different trees. A
// domain with no accounting hierarchy still reports every child, with the CPU fields
// left zero: the caller prices CPU from the request instead of from measurement, which
// is the behaviour it had before CPU was measured at all.
func (d Domain) Children(parent string) ([]Child, error) {
	memory, ok := d.memory()
	if !ok {
		return nil, fmt.Errorf("cgroup: %s has no memory hierarchy", d.String())
	}
	dir := filepath.Join(memory.Root, parent)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("cgroup: listing %s: %w", dir, err)
	}
	accounting, hasAccounting := d.accounting()
	out := make([]Child, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		child, ok := readChild(filepath.Join(dir, entry.Name()), entry.Name())
		if !ok {
			continue
		}
		if hasAccounting {
			readCPU(filepath.Join(accounting.Root, parent, entry.Name()), &child)
		}
		out = append(out, child)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

// readCPU fills one child's CPU counters from its cgroup in the accounting hierarchy.
//
// Throttling is read from the same file in both versions, under a name that changed:
// v1 reports nanoseconds in `throttled_time`, v2 microseconds in `throttled_usec`. A
// host that publishes neither reports zero, which is read as "not throttled" rather
// than as "no counter" -- the same convention as the OOM counter, and for the same
// reason: a missing counter must never invent pressure, and it must never hide it
// either, so the caller compares deltas and a counter that never moves is simply one
// that never fired.
func readCPU(dir string, child *Child) {
	if nanos, err := readIntFile(filepath.Join(dir, "cpuacct.usage")); err == nil {
		// v1 counts nanoseconds in cpuacct.usage.
		child.CPUUsageNanos = nanos
	} else if raw, err := os.ReadFile(filepath.Join(dir, "cpu.stat")); err == nil {
		// v2 counts microseconds under a different name in a keyed file.
		fields := parseStat(string(raw))
		if usec, ok := fields["usage_usec"]; ok {
			child.CPUUsageNanos = usec * 1000
		}
	}
	if raw, err := os.ReadFile(filepath.Join(dir, "cpu.stat")); err == nil {
		fields := parseStat(string(raw))
		if nanos, ok := fields["throttled_time"]; ok {
			child.CPUThrottledNanos = nanos
		} else if usec, ok := fields["throttled_usec"]; ok {
			child.CPUThrottledNanos = usec * 1000
		}
	}
	if quota, err := readIntFile(filepath.Join(dir, "cpu.cfs_quota_us")); err == nil {
		child.CPUQuotaMicros = quota
		return
	}
	if raw, err := os.ReadFile(filepath.Join(dir, "cpu.max")); err == nil {
		fields := strings.Fields(string(raw))
		if len(fields) == 2 {
			if quota, err := parseInt(fields[0]); err == nil {
				child.CPUQuotaMicros = quota
			}
		}
	}
}

// readChild reads one sandbox's counters, reporting false when it is already gone.
func readChild(dir, id string) (Child, bool) {
	usage, err := readIntFile(filepath.Join(dir, "memory.usage_in_bytes"))
	if err != nil {
		// v2 names it differently.
		usage, err = readIntFile(filepath.Join(dir, "memory.current"))
		if err != nil {
			return Child{}, false
		}
	}
	child := Child{ID: id, Usage: usage}
	if raw, err := os.ReadFile(filepath.Join(dir, "memory.stat")); err == nil {
		fields := parseStat(string(raw))
		if v2, ok := fields["anon"]; ok {
			child.Resident = v2 + fields["slab_unreclaimable"] + fields["kernel_stack"]
			child.Cache = fields["file"]
		} else {
			child.Resident = fields["total_rss"] + readIntOrZero(filepath.Join(dir, "memory.kmem.usage_in_bytes"))
			child.Cache = fields["total_cache"]
		}
	}
	if limit, err := memoryLimit(dir); err == nil {
		child.Limit = limit
	}
	child.OOMKills = oomKills(dir)
	if fails, err := readIntFile(filepath.Join(dir, "memory.failcnt")); err == nil {
		child.MemoryFailCount = fails
	} else if raw, err := os.ReadFile(filepath.Join(dir, "memory.events")); err == nil {
		// v2 splits the counter by which limit was hit; they all mean the sandbox was
		// bounded by memory it could not have, so the sum is the signal.
		for name, value := range parseStat(string(raw)) {
			switch name {
			case "low", "high", "max", "oom_kill":
				child.MemoryFailCount += value
			}
		}
	}
	return child, true
}

// oomKills reads the kernel's per-cgroup kill counter.
//
// v1 exposes it in memory.oom_control as a plain field; v2 exposes it in
// memory.events. A host that has neither reports zero, which reads as "no evidence"
// rather than as "no kills" -- the caller compares it against a previous reading, so a
// missing counter never invents a kill.
func oomKills(dir string) int64 {
	if raw, err := os.ReadFile(filepath.Join(dir, "memory.oom_control")); err == nil {
		for _, line := range strings.Split(string(raw), "\n") {
			fields := strings.Fields(line)
			if len(fields) == 2 && fields[0] == "oom_kill" {
				if value, convErr := parseInt(fields[1]); convErr == nil {
					return value
				}
			}
		}
	}
	if raw, err := os.ReadFile(filepath.Join(dir, "memory.events")); err == nil {
		for _, line := range strings.Split(string(raw), "\n") {
			fields := strings.Fields(line)
			if len(fields) == 2 && fields[0] == "oom_kill" {
				if value, convErr := parseInt(fields[1]); convErr == nil {
					return value
				}
			}
		}
	}
	return 0
}

// RemoveChild deletes one sandbox's cgroup from every hierarchy that carries one.
//
// Nothing here decides *whether* a cgroup may go -- that is the caller's, and the guards
// belong there because only the caller knows what the cgroup is for. This only carries the
// removal out and reports which hierarchies let it happen, because a cgroup removed from
// the memory hierarchy but not the accounting one is a worse state than one left alone:
// it is invisible to half the controller and still charged by the other half.
//
// The kernel is the last guard and it is the only one that cannot be wrong: rmdir on a
// cgroup that still holds a task fails, whatever the caller believed about it.
func (d Domain) RemoveChild(parent, id string) error {
	if parent == "" || id == "" {
		return fmt.Errorf("cgroup: removing a child needs a parent and an id")
	}
	var removed []string
	var failed []string
	for _, hierarchy := range d.Hierarchies {
		dir := filepath.Join(hierarchy.Root, parent, id)
		if _, err := os.Stat(dir); err != nil {
			continue
		}
		if err := os.Remove(dir); err != nil {
			failed = append(failed, fmt.Sprintf("%s: %v", hierarchy.Controllers, err))
			continue
		}
		removed = append(removed, strings.Join(hierarchy.Controllers, ","))
	}
	if len(failed) > 0 {
		return fmt.Errorf("cgroup: %s could not be removed from %s", id, strings.Join(failed, "; "))
	}
	if len(removed) == 0 {
		return fmt.Errorf("cgroup: %s is in no hierarchy under %s", id, parent)
	}
	return nil
}

// SetChildLimit rewrites one sandbox's memory bound.
//
// Lowering it is the only way a v1 host can make a sandbox give memory back: v1 has no
// memory.high, and memory.soft_limit_in_bytes is a hint the kernel consults only when the
// whole host is under pressure. Raising it back above memory.memsw.limit_in_bytes fails,
// so the pair is moved by the same routine that created it.
func (d Domain) SetChildLimit(parent, id string, bytes int64) error {
	memory, ok := d.memory()
	if !ok {
		return fmt.Errorf("cgroup: %s has no memory hierarchy", d.String())
	}
	if bytes <= 0 {
		return fmt.Errorf("cgroup: a sandbox limit must be positive, got %d", bytes)
	}
	return writeMemoryLimits(filepath.Join(memory.Root, parent, id), bytes)
}

// QuotaToMillis converts a kernel CPU quota into the millicores the API charges.
//
// A quota is microseconds of CPU per period, so the conversion is a ratio rather than a
// constant, and the period is the one this package writes. A quota of -1 -- the kernel's
// "unbounded" -- is not a number a ledger can charge and converts to zero, which reads
// as "no bound stated".
func QuotaToMillis(quotaMicros int64) int64 {
	if quotaMicros <= 0 {
		return 0
	}
	return quotaMicros * 1000 / cpuPeriodMicros
}

// SetChildCPUQuota rewrites one sandbox's CPU bound.
//
// CPU is the compressible dimension: unlike memory, a bound below what the sandbox is
// running costs it latency rather than its life, so this is the knob a node can move on
// a sandbox that is still running. It is still only moved on an idle one, because a
// throttled tool call is a step that takes longer.
func (d Domain) SetChildCPUQuota(parent, id string, millis int64) error {
	accounting, ok := d.accounting()
	if !ok {
		return fmt.Errorf("cgroup: %s has no CPU accounting hierarchy", d.String())
	}
	if millis <= 0 {
		return fmt.Errorf("cgroup: a sandbox CPU quota must be positive, got %d millis", millis)
	}
	dir := filepath.Join(accounting.Root, parent, id)
	quota := millis * 100
	if d.Version == 2 {
		path := filepath.Join(dir, "cpu.max")
		if err := os.WriteFile(path, []byte(fmt.Sprintf("%d %d", quota, cpuPeriodMicros)), 0); err != nil {
			return fmt.Errorf("cgroup: writing %s: %w", path, err)
		}
		return nil
	}
	period := filepath.Join(dir, "cpu.cfs_period_us")
	if err := writeInt(period, cpuPeriodMicros); err != nil {
		return fmt.Errorf("cgroup: writing %s: %w", period, err)
	}
	path := filepath.Join(dir, "cpu.cfs_quota_us")
	if err := writeInt(path, quota); err != nil {
		return fmt.Errorf("cgroup: writing %s: %w", path, err)
	}
	return nil
}
