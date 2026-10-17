// The elastic side of admission: what a sandbox is charged, and what it gives back.
//
// The node used to charge every sandbox its declared maximum. That maximum is a
// protection bound -- the kernel limit that stops one sandbox eating the node -- not a
// cost: a step that admits 128 sandboxes of 30 GB reserves 3.8 TB on a node with 2.3 TB,
// so most of the batch waits for a reservation nobody is using, and the memory that
// "fills" the node is a number rather than pages.
//
// Two things change here, and they need each other:
//
//   - The charge becomes an estimate of what a sandbox of its class actually costs, taken
//     from the per-sandbox resident memory the kernel reports. It follows an observed peak
//     up at once and comes down slowly, and the measured aggregate is a floor under it.
//   - A sandbox holding reclaimable memory while the node is near its budget is asked to
//     give it back, by lowering its own kernel limit to what it cannot release. v1 has no
//     memory.high -- memory.soft_limit_in_bytes is a hint the kernel consults only when the
//     whole host is under pressure -- so the limit is the only knob that makes a sandbox
//     return memory. It is raised again when the node has room.
package server

import (
	"context"
	"log/slog"

	"psrl.dev/sandboxd/internal/cgroup"
	"psrl.dev/sandboxd/internal/node"
)

// ElasticConfig states how a node prices a sandbox and when it asks one for memory back.
type ElasticConfig struct {
	// Enabled leaves admission on declared maxima when false, which is the shape every
	// deployment had before.
	Enabled bool
	// Footprint is how a class's per-sandbox charge is estimated.
	Footprint node.FootprintConfig
	// ShrinkAbove is the measured fraction of the sandbox budget above which idle sandboxes
	// are asked to release reclaimable memory.
	ShrinkAbove float64
	// RestoreBelow is the fraction below which their limits are raised again. The gap
	// between the two is the hysteresis, so a node sitting at the threshold does not shrink
	// and restore on alternate samples.
	RestoreBelow float64
	// ReleaseMarginMB is what is left above a sandbox's resident memory when its limit is
	// lowered: enough to run a command without immediately reclaiming.
	ReleaseMarginMB int64
	// MinLimitMB is the lowest a limit may be lowered to, whatever the measurement says.
	MinLimitMB int64
}

// ElasticConfigWithDefaults fills in what a deployment left unset.
func ElasticConfigWithDefaults(cfg ElasticConfig) ElasticConfig {
	if cfg.ShrinkAbove <= 0 || cfg.ShrinkAbove > 1 {
		cfg.ShrinkAbove = 0.85
	}
	if cfg.RestoreBelow <= 0 || cfg.RestoreBelow > cfg.ShrinkAbove {
		cfg.RestoreBelow = cfg.ShrinkAbove * 0.7
	}
	if cfg.ReleaseMarginMB <= 0 {
		cfg.ReleaseMarginMB = 256
	}
	if cfg.MinLimitMB <= 0 {
		cfg.MinLimitMB = 256
	}
	cfg.Footprint = node.FootprintConfigWithDefaults(cfg.Footprint)
	return cfg
}

// ElasticReport is one tick's decisions, for the metric hook and for the log line that
// explains a change in how much the node will admit.
type ElasticReport struct {
	// ObservedMB is what the sandboxes were measured using.
	ObservedMB int64
	// BudgetMB is the subtree's limit.
	BudgetMB int64
	// ReservedMB is what the ledger charges in total after this tick.
	ReservedMB int64
	// Reserves are the per-class per-sandbox charges this tick settled on.
	Reserves map[string]int64
	// Shrunk and Restored name the sandboxes whose kernel limit changed.
	Shrunk   []string
	Restored []string
	// OOMKills is the total the kernel reports across these sandboxes.
	OOMKills int64
}

// elastic is one node's controller state.
type elastic struct {
	cfg ElasticConfig
	// footprints are per class, because that is the unit the ledger charges and the unit a
	// class guarantee is stated in.
	footprints map[string]*node.Footprint
	// maxBySandbox is the declared maximum of each running sandbox, which is what its
	// kernel limit is restored to.
	maxBySandbox map[string]int64
	// classBySandbox is what the sandbox was admitted as.
	classBySandbox map[string]string
	// shrunk records which sandboxes are currently below their declared maximum.
	shrunk map[string]bool
}

// ElasticTick applies one sample of per-sandbox usage.
//
// It records the measurement as a floor under the ledger, reprices each class from what its
// sandboxes are actually resident, and -- when the budget is nearly spent -- asks the idle
// ones to release what they can. The order matters: the charge is settled before the shrink,
// so a tick can never hand out capacity it has not accounted for.
func (n *Node) ElasticTick(ctx context.Context) ElasticReport {
	report := ElasticReport{Reserves: map[string]int64{}}
	if n.elastic == nil || !n.elastic.cfg.Enabled || len(n.elasticDomain.Hierarchies) == 0 {
		return report
	}
	cfg := n.elastic.cfg
	children, err := n.elasticDomain.Children(n.elasticParent)
	if err != nil {
		// An unreadable domain is not an empty node: leave the ledger on its reservations
		// and change no limits.
		slog.Warn("elastic admission could not read the sandbox domain", "error", err)
		return report
	}

	byClass := map[string][]int64{}
	var observedBytes, oomKills int64
	countByClass := map[string]int{}
	for _, child := range children {
		observedBytes += child.Usage
		oomKills += child.OOMKills
		class, known := n.elastic.classBySandbox[child.ID]
		if !known {
			continue
		}
		byClass[class] = append(byClass[class], child.Resident/(1024*1024))
		countByClass[class]++
	}
	report.ObservedMB = observedBytes / (1024 * 1024)
	report.OOMKills = oomKills
	report.BudgetMB = n.elasticBudgetMB()

	// The measurement is a floor under the charge, so a stale reservation can never make the
	// node believe it has room the pages have already taken.
	n.admission.SetObserved(node.Resources{MemoryMB: report.ObservedMB}, true)

	// Reprice each class. A class with no running sandbox keeps its last estimate, which is
	// what seeds the next batch: an empty class would otherwise fall back to the minimum
	// between two steps and under-reserve the batch that follows.
	for class, residents := range byClass {
		footprint := n.footprintFor(class)
		before := footprint.ReserveMB()
		footprint.Observe(residents)
		if after := footprint.ReserveMB(); after != before {
			n.admission.Adjust(class, node.Resources{MemoryMB: int64(len(residents)) * (after - before)})
		}
	}
	for class := range n.elastic.footprints {
		report.Reserves[class] = n.elastic.footprints[class].ReserveMB()
	}
	n.elasticAudit()
	report.ReservedMB = n.admission.Snapshot().Granted.MemoryMB

	fraction := 0.0
	if report.BudgetMB > 0 {
		fraction = float64(report.ObservedMB) / float64(report.BudgetMB)
	}
	for _, child := range children {
		if _, known := n.elastic.classBySandbox[child.ID]; !known {
			continue
		}
		switch {
		case fraction >= cfg.ShrinkAbove:
			report.Shrunk = n.releaseMemory(child, cfg, report.Shrunk)
		case fraction <= cfg.RestoreBelow:
			report.Restored = n.restoreMemory(child, report.Restored)
		}
	}
	return report
}

// releaseMemory lowers one idle sandbox's limit to what it cannot give back.
func (n *Node) releaseMemory(child cgroup.Child, cfg ElasticConfig, shrunk []string) []string {
	if n.elastic.shrunk[child.ID] || n.lifecycle.Busy(child.ID) {
		// A command is running, so its page cache is the code it is executing: shrinking
		// here would reclaim the working set of a live step.
		return shrunk
	}
	targetMB := child.Resident/(1024*1024) + cfg.ReleaseMarginMB
	if targetMB < cfg.MinLimitMB {
		targetMB = cfg.MinLimitMB
	}
	if child.Limit <= targetMB*(1024*1024) {
		return shrunk
	}
	if err := n.elasticDomain.SetChildLimit(n.elasticParent, child.ID, targetMB*(1024*1024)); err != nil {
		slog.Warn("elastic admission could not lower a sandbox limit",
			"sandbox", child.ID, "target_mb", targetMB, "error", err)
		return shrunk
	}
	n.elastic.shrunk[child.ID] = true
	return append(shrunk, child.ID)
}

// restoreMemory raises a sandbox back to its declared maximum.
func (n *Node) restoreMemory(child cgroup.Child, restored []string) []string {
	if !n.elastic.shrunk[child.ID] {
		return restored
	}
	maxMB := n.elastic.maxBySandbox[child.ID]
	if maxMB <= 0 {
		delete(n.elastic.shrunk, child.ID)
		return restored
	}
	if err := n.elasticDomain.SetChildLimit(n.elasticParent, child.ID, maxMB*(1024*1024)); err != nil {
		slog.Warn("elastic admission could not restore a sandbox limit",
			"sandbox", child.ID, "target_mb", maxMB, "error", err)
		return restored
	}
	delete(n.elastic.shrunk, child.ID)
	return append(restored, child.ID)
}

// classReserves is what this node publishes about the cost of a sandbox.
//
// It is reported rather than kept to itself because admission above this node -- the fleet
// quota and placement -- charges a request before this node ever sees it. A node that
// prices a sandbox from measurement but does not say so leaves every caller charging the
// maximum, which is the whole reservation this exists to remove.
func (n *Node) classReserves() map[string]int64 {
	if n.elastic == nil {
		return nil
	}
	reserves := make(map[string]int64, len(n.elastic.footprints))
	for class, footprint := range n.elastic.footprints {
		reserves[class] = footprint.ReserveMB()
	}
	if len(reserves) == 0 {
		return nil
	}
	return reserves
}

// footprintFor returns a class's estimate, creating it on first use.
func (n *Node) footprintFor(class string) *node.Footprint {
	if footprint, ok := n.elastic.footprints[class]; ok {
		return footprint
	}
	footprint := node.NewFootprint(n.elastic.cfg.Footprint, n.maxMemoryForClass(class))
	n.elastic.footprints[class] = footprint
	return footprint
}

// maxMemoryForClass is the largest maximum declared by a sandbox of this class, which caps
// what the class's reservation may charge.
func (n *Node) maxMemoryForClass(class string) int64 {
	var largest int64
	for sandboxID, sandboxClass := range n.elastic.classBySandbox {
		if sandboxClass != class {
			continue
		}
		if candidate := n.elastic.maxBySandbox[sandboxID]; candidate > largest {
			largest = candidate
		}
	}
	return largest
}

// elasticBudgetMB is the sandbox budget the controller measures against.
func (n *Node) elasticBudgetMB() int64 {
	if n.elasticBudget > 0 {
		return n.elasticBudget
	}
	return n.admission.Snapshot().Envelope.MemoryMB
}

// elasticAudit drops controller state for sandboxes this node no longer holds.
func (n *Node) elasticAudit() {
	for sandboxID := range n.elastic.maxBySandbox {
		if n.lifecycle.Running(sandboxID) {
			continue
		}
		delete(n.elastic.maxBySandbox, sandboxID)
		delete(n.elastic.classBySandbox, sandboxID)
		delete(n.elastic.shrunk, sandboxID)
	}
}
