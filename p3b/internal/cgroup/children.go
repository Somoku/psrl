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
}

// Children reads every sandbox cgroup under a parent.
//
// A child that disappeared between the listing and the read is skipped rather than
// reported: a sandbox being released is the normal case at this cadence, not an error.
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
	out := make([]Child, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		child, ok := readChild(filepath.Join(dir, entry.Name()), entry.Name())
		if !ok {
			continue
		}
		out = append(out, child)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
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
