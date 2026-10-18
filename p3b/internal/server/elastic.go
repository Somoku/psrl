// The elastic side of admission: what a sandbox is charged, and what it gives back.
//
// The node used to charge every sandbox its declared maximum. That maximum is a
// protection bound -- the kernel limit that stops one sandbox eating the node -- not a
// cost: a step that admits 128 sandboxes of 30 GB reserves 3.8 TB on a node with 2.3 TB,
// so most of the batch waits for a reservation nobody is using, and the memory that
// "fills" the node is a number rather than pages.
//
// The same is true of CPU, and it is the dimension that actually binds a burst. Each
// sandbox declares two cores because a tool call may want them; between tool calls it
// uses a hundredth of one. Charging the declaration makes the fleet's CPU the ceiling --
// a fleet selling 360 cores admits 180 sandboxes -- while the machine is idle.
//
// Two things change here, and they need each other:
//
//   - The charge becomes an estimate of what a sandbox of its class actually costs, taken
//     from what the kernel reports per sandbox: resident memory, and CPU time converted to
//     a rate. It follows an observed peak up at once and comes down slowly, and the
//     measured aggregate is a floor under it.
//   - A sandbox holding reclaimable memory while the node is near its budget is asked to
//     give it back, by lowering its own kernel limit to what it cannot release. v1 has no
//     memory.high -- memory.soft_limit_in_bytes is a hint the kernel consults only when the
//     whole host is under pressure -- so the limit is the only knob that makes a sandbox
//     return memory. It is raised again when the node has room.
//
// The CPU side has no release of its own to ask for: CPU is compressible, so a bound below
// what a sandbox is using costs it latency rather than its life, and the knob is a quota
// the node may move on a running sandbox. It is still only moved on an idle one, because a
// throttled tool call is a step that takes longer.
package server

import (
	"context"
	"log/slog"
	"math"
	"sync"
	"time"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/cgroup"
	"psrl.dev/sandboxd/internal/node"
)

// ElasticConfig states how a node prices a sandbox and when it asks one for memory back.
type ElasticConfig struct {
	// Enabled leaves admission on declared maxima when false, which is the shape every
	// deployment had before.
	Enabled bool
	// Footprint is how a class's per-sandbox memory charge is estimated, in MiB.
	Footprint node.FootprintConfig
	// CPUFootprint is how its CPU charge is estimated, in millicores. It is separate
	// because the two dimensions are not the same number: a sandbox that fits in 500 MiB
	// may still need two cores for a compile, and one estimator over both would price
	// each by the other's unit.
	CPUFootprint node.FootprintConfig
	// ShrinkAbove is the measured fraction of the sandbox budget above which idle sandboxes
	// are asked to release reclaimable memory, and above which their CPU quota is lowered.
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
	// ReleaseMarginCPUMillis is what an idle sandbox's quota is left above what it measured
	// using, so a throttled sandbox can still answer its next tool call.
	ReleaseMarginCPUMillis int64
	// MinCPUMillis is the lowest a sandbox's quota may be lowered to. A sandbox throttled
	// to nothing cannot finish the step it is part of, so the floor keeps it alive.
	MinCPUMillis int64
	// GrowthHorizonS is how far ahead the growth reservation looks. A sandbox's cost is a
	// claim about the future, and this is how much of it: the observed growth rate per
	// sandbox is reserved over this window, which is what stops a fleet admitting a burst
	// at a price that stops being true before the burst has finished arriving. Zero takes
	// DefaultGrowthHorizonS.
	GrowthHorizonS float64
	// ThrottledAbove is the share of an interval a sandbox must spend throttled before the
	// controller stops believing its measured rate. Usage is censored by the quota, so
	// without this a class can be priced at its idle rate while every sandbox in it is
	// being throttled. Zero takes the default.
	ThrottledAbove float64
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
	if cfg.ReleaseMarginCPUMillis <= 0 {
		cfg.ReleaseMarginCPUMillis = 250
	}
	if cfg.MinCPUMillis <= 0 {
		cfg.MinCPUMillis = 100
	}
	// Ten percent of the interval spent throttled. It is deliberately low: the question is
	// not "is this sandbox slow" but "did the quota refuse it work", and any throttling at
	// all is evidence that the measured rate is truncated.
	if cfg.ThrottledAbove <= 0 {
		cfg.ThrottledAbove = DefaultThrottledAbove
	}
	if cfg.GrowthHorizonS <= 0 {
		cfg.GrowthHorizonS = DefaultGrowthHorizonS
	}
	// A quarter of a core is what a sandbox holding a shell and an agent costs before the
	// class has any history. It is deliberately not zero: the first batch after a restart
	// is admitted against this, and zero would say a sandbox is free.
	if cfg.CPUFootprint.MinReserve <= 0 {
		cfg.CPUFootprint.MinReserve = 250
	}
	if cfg.Footprint.MinReserve <= 0 {
		cfg.Footprint.MinReserve = 512
	}
	cfg.Footprint = node.FootprintConfigWithDefaults(cfg.Footprint)
	cfg.CPUFootprint = node.FootprintConfigWithDefaults(cfg.CPUFootprint)
	return cfg
}

// DefaultThrottledAbove is the share of an interval a sandbox spends throttled before the
// controller treats its measured CPU rate as truncated by the quota.
const DefaultThrottledAbove = 0.10

// DefaultGrowthHorizonS is how far ahead admission reserves a class's growth. One minute
// covers a sampling interval, a fleet report and the time a batch takes to arrive at that
// price -- long enough that the reservation is real, short enough that it does not price a
// sandbox at its peak before it has any history.
const DefaultGrowthHorizonS = 60.0

// ElasticReport is one tick's decisions, for the metric hook and for the log line that
// explains a change in how much the node will admit.
type ElasticReport struct {
	// ObservedMB is what the sandboxes were measured using.
	ObservedMB int64
	// ObservedCPUMillis is the CPU they were measured using, as a rate over the interval
	// since the last tick.
	ObservedCPUMillis int64
	// BudgetMB is the subtree's limit.
	BudgetMB int64
	// BudgetCPUMillis is the CPU the node sells, which is the number the CPU fraction is
	// taken of.
	BudgetCPUMillis int64
	// ReservedMB is what the ledger charges in total after this tick.
	ReservedMB int64
	// ReservedCPUMillis is the same for CPU.
	ReservedCPUMillis int64
	// ReservesMB are the per-class per-sandbox memory charges this tick settled on.
	ReservesMB map[string]int64
	// ReservesCPUMillis are the per-class per-sandbox CPU charges, in millicores.
	ReservesCPUMillis map[string]int64
	// Shrunk and Restored name the sandboxes whose kernel limit changed.
	Shrunk   []string
	Restored []string
	// Throttled and Released name the sandboxes whose CPU quota changed.
	Throttled []string
	Released  []string
	// OOMKills is the total the kernel reports across these sandboxes.
	OOMKills int64
	// ThrottledPct is the share of the interval the sandboxes spent stopped for exceeding
	// their CPU quota, averaged over the ones being measured. It is the number that says
	// the node admitted more CPU demand than it can serve, and it is invisible in usage:
	// usage saturates at the quota and looks like a full but healthy node.
	ThrottledPct float64
}

// elastic is one node's controller state.
//
// Its maps are read by the node's report, which a caller drives from the fleet poll, and
// written by the controller's own tick on another goroutine, so the whole controller is
// behind one lock. A report that raced a tick would not merely read a stale price -- an
// unsynchronised map read against a map write is a fatal error in Go, which would take
// the node down.
type elastic struct {
	mu  sync.Mutex
	cfg ElasticConfig
	// footprints are per class, because that is the unit the ledger charges and the unit a
	// class guarantee is stated in.
	memoryFootprints map[string]*node.Footprint
	// cpuFootprints are the same classes' CPU estimates.
	cpuFootprints map[string]*node.Footprint
	// maxBySandbox is the declared maximum of each running sandbox, which is what its
	// kernel limit is restored to.
	maxMemoryMBBySandbox map[string]int64
	// maxCPUMillisBySandbox is the CPU each running sandbox declared, which is what its
	// quota is restored to.
	maxCPUMillisBySandbox map[string]int64
	// classBySandbox is what the sandbox was admitted as.
	classBySandbox map[string]string
	// shrunk records which sandboxes are currently below their declared maximum.
	shrunk map[string]bool
	// throttled records which sandboxes are currently below their declared CPU quota.
	throttled map[string]bool
	// cpuUsage is the last cumulative CPU reading per sandbox. A rate needs the previous
	// reading and the interval between the two, so the first sighting of a sandbox is
	// recorded without pricing it.
	cpuUsageNanos map[string]int64
	// cpuRateMillis is the rate each sandbox was last measured at, which is what its quota
	// is throttled to.
	cpuRateMillis map[string]int64
	// cpuThrottledNanos is the previous throttling reading per sandbox, and cpuThrottle the
	// share of the last interval it spent throttled. Kept because throttling is the only
	// signal of demand the quota refused, and it decides both the price and whether the
	// sandbox may be shrunk.
	cpuThrottledNanos map[string]int64
	cpuThrottle       map[string]float64
	// lastSample is when the previous tick read the domain, which is the interval a rate
	// is taken over.
	lastSample time.Time
	// missed counts ticks that could not read the domain, so a domain that stops being
	// readable is stated rather than inferred from a ledger that quietly stopped moving.
	missed int
	// previousPrice is each class's per-sandbox estimate at the previous tick, and
	// classCount how many sandboxes the class had. Together they give the rate the
	// growth reservation is computed from.
	previousPrice map[string]node.Resources
	classCount    map[string]int
}

// ElasticTick applies one sample of per-sandbox usage.
//
// It records the measurement as a floor under the ledger, reprices each class from what its
// sandboxes are actually resident, and -- when the budget is nearly spent -- asks the idle
// ones to release what they can. The order matters: the charge is settled before the shrink,
// so a tick can never hand out capacity it has not accounted for.
func (n *Node) ElasticTick(ctx context.Context) ElasticReport {
	report := ElasticReport{ReservesMB: map[string]int64{}, ReservesCPUMillis: map[string]int64{}}
	if n.elastic == nil || !n.elastic.cfg.Enabled || len(n.elasticDomain.Hierarchies) == 0 {
		return report
	}
	n.elastic.mu.Lock()
	defer n.elastic.mu.Unlock()
	cfg := n.elastic.cfg
	children, err := n.elasticDomain.Children(n.elasticParent)
	if err != nil {
		// An unreadable domain is not an empty node, and it is not an idle one either.
		// Reservations stay as the charge and no limit moves; what changes is that the
		// measurement stops being fresh, so once it is old enough the ledger stops using
		// it as a floor instead of charging against a number nobody can confirm.
		n.elastic.missed++
		slog.Warn("elastic admission could not read the sandbox domain",
			"error", err, "consecutive_misses", n.elastic.missed)
		return report
	}
	if n.elastic.missed > 0 {
		slog.Warn("elastic admission is reading the sandbox domain again",
			"missed_samples", n.elastic.missed)
		n.elastic.missed = 0
	}

	now := time.Now()
	elapsed := now.Sub(n.elastic.lastSample)
	n.elastic.lastSample = now

	memoryByClass := map[string][]int64{}
	cpuMillisByClass := map[string][]int64{}
	var observedBytes, oomKills, observedCPU int64
	var throttledTotal float64
	measured := 0
	for class := range n.elastic.classCount {
		n.elastic.classCount[class] = 0
	}
	for _, child := range children {
		observedBytes += child.Usage
		oomKills += child.OOMKills
		class, known := n.elastic.classBySandbox[child.ID]
		if !known {
			continue
		}
		n.elastic.classCount[class]++
		memoryByClass[class] = append(memoryByClass[class], child.Resident/(1024*1024))
		if rate, ok := n.cpuDemand(child, elapsed); ok {
			cpuMillisByClass[class] = append(cpuMillisByClass[class], rate)
			observedCPU += rate
			measured++
			throttledTotal += n.elastic.cpuThrottle[child.ID]
		}
	}
	report.ObservedMB = observedBytes / (1024 * 1024)
	report.ObservedCPUMillis = observedCPU
	report.OOMKills = oomKills
	if measured > 0 {
		report.ThrottledPct = throttledTotal / float64(measured)
	}
	report.BudgetMB = n.elasticBudgetMB()
	report.BudgetCPUMillis = n.elasticBudgetCPUMillis()

	// The measurement is a floor under the charge, so a stale reservation can never make the
	// node believe it has room the pages have already taken.
	n.admission.SetObserved(node.Resources{MemoryMB: report.ObservedMB, CPUMillis: report.ObservedCPUMillis}, true)

	// Reserve the growth that is already under way. This is the piece that was missing
	// when a burst was admitted at a price that stopped being true before it finished
	// arriving: the samples below say how fast each class is climbing, and admission
	// charges that climb over the next horizon rather than the present alone.
	n.reserveGrowth(elapsed)

	// Reprice each class. A class with no running sandbox keeps its last estimate, which is
	// what seeds the next batch: an empty class would otherwise fall back to the minimum
	// between two steps and under-reserve the batch that follows.
	for class, residents := range memoryByClass {
		footprint := n.memoryFootprintForLocked(class)
		before := footprint.Reserve()
		footprint.Observe(residents)
		if after := footprint.Reserve(); after != before {
			n.admission.Adjust(class, node.Resources{MemoryMB: int64(len(residents)) * (after - before)})
		}
	}
	for class, rates := range cpuMillisByClass {
		footprint := n.cpuFootprintForLocked(class)
		before := footprint.Reserve()
		footprint.Observe(rates)
		if after := footprint.Reserve(); after != before {
			n.admission.Adjust(class, node.Resources{CPUMillis: int64(len(rates)) * (after - before)})
		}
	}
	for class := range n.elastic.memoryFootprints {
		report.ReservesMB[class] = n.elastic.memoryFootprints[class].Reserve()
	}
	for class := range n.elastic.cpuFootprints {
		report.ReservesCPUMillis[class] = n.elastic.cpuFootprints[class].Reserve()
	}
	n.elasticAudit()
	granted := n.admission.Snapshot().Granted
	report.ReservedMB = granted.MemoryMB
	report.ReservedCPUMillis = granted.CPUMillis

	memFraction := 0.0
	if report.BudgetMB > 0 {
		memFraction = float64(report.ObservedMB) / float64(report.BudgetMB)
	}
	cpuFraction := 0.0
	if report.BudgetCPUMillis > 0 {
		cpuFraction = float64(report.ObservedCPUMillis) / float64(report.BudgetCPUMillis)
	}
	for _, child := range children {
		if _, known := n.elastic.classBySandbox[child.ID]; !known {
			continue
		}
		switch {
		case memFraction >= cfg.ShrinkAbove:
			report.Shrunk = n.releaseMemory(child, cfg, report.Shrunk)
		case memFraction <= cfg.RestoreBelow:
			report.Restored = n.restoreMemory(child, report.Restored)
		}
		switch {
		case cpuFraction >= cfg.ShrinkAbove:
			report.Throttled = n.throttleCPU(child, cfg, report.Throttled)
		case cpuFraction <= cfg.RestoreBelow:
			report.Released = n.releaseCPU(child, report.Released)
		}
	}
	return report
}

// cpuRate converts one sandbox's cumulative CPU time into the rate it used over an
// interval, in millicores.
//
// The second result is false when there is nothing to divide: the first sighting of a
// sandbox has no previous reading, and an interval of zero would divide by nothing. The
// reading is recorded either way, so the next tick prices it. A sandbox that is destroyed
// before its second sample is never priced, which is the honest answer -- one reading of a
// cumulative counter says nothing about load.
func (n *Node) cpuDemand(child cgroup.Child, elapsed time.Duration) (int64, bool) {
	previous, known := n.elastic.cpuUsageNanos[child.ID]
	n.elastic.cpuUsageNanos[child.ID] = child.CPUUsageNanos
	throttled := n.throttledFraction(child, elapsed)
	n.elastic.cpuThrottle[child.ID] = throttled
	if !known || elapsed <= 0 {
		return 0, false
	}
	delta := child.CPUUsageNanos - previous
	if delta <= 0 {
		// An idle sandbox is a real measurement, not an absent one: it is what lets a
		// class's price come down.
		n.elastic.cpuRateMillis[child.ID] = 0
		return 0, true
	}
	// Nanoseconds of CPU over nanoseconds of wall clock is cores; a thousandth of a core
	// is the unit the ledger charges.
	rate := int64(math.Round(1000 * float64(delta) / float64(elapsed.Nanoseconds())))

	// The measured rate is censored from above by the sandbox's quota: a sandbox that
	// wanted more than it was given only ever reaches the quota, so `rate` saturates
	// there and stops describing demand. Throttling is the kernel saying the sandbox
	// asked and was refused, and what that establishes is a lower bound -- the demand is
	// at least the whole quota. Substituting it replaces a number we know is truncated
	// with one we know is true, which is what lets a class's price follow a burst instead
	// of sitting at the idle rate while every sandbox in it is being throttled.
	if throttled >= n.elastic.cfg.ThrottledAbove {
		if quota := cgroup.QuotaToMillis(child.CPUQuotaMicros); quota > rate {
			rate = quota
		}
	}
	n.elastic.cpuRateMillis[child.ID] = rate
	return rate, true
}

// throttledFraction is the share of the interval this sandbox spent stopped for exceeding
// its CPU quota.
func (n *Node) throttledFraction(child cgroup.Child, elapsed time.Duration) float64 {
	previous, known := n.elastic.cpuThrottledNanos[child.ID]
	n.elastic.cpuThrottledNanos[child.ID] = child.CPUThrottledNanos
	if !known || elapsed <= 0 {
		return 0
	}
	delta := child.CPUThrottledNanos - previous
	if delta <= 0 {
		return 0
	}
	return float64(delta) / float64(elapsed.Nanoseconds())
}

// sandboxBusy reports whether work is happening in a sandbox.
//
// A command in flight is not the only way, and relying on it alone is how a controller
// shrinks a sandbox whose background process is mid-compile. A sandbox being throttled is
// definitionally trying to run, and one hitting its memory limit is being held by memory it
// needs, so both read as busy whether or not an RPC happens to be outstanding.
func (n *Node) sandboxBusy(child cgroup.Child) bool {
	if n.lifecycle.Busy(child.ID) {
		return true
	}
	if n.elastic.cpuThrottle[child.ID] >= n.elastic.cfg.ThrottledAbove {
		return true
	}
	return child.MemoryFailCount > 0 && child.Usage >= child.Limit && child.Limit > 0
}

// reserveGrowth charges each class with the growth it is already committed to.
//
// A sandbox's cost is not what it uses now but what it will have used by the time the next
// sample can correct the price -- and by then a whole batch may have been admitted at the
// old number. So the controller watches how fast each class's per-sandbox estimate is
// climbing and reserves that rate over a horizon, both for the sandboxes already running
// and for the one being admitted.
//
// This is the same reservation a scheduler makes when it charges a pod its requests rather
// than its current usage: the request is a claim about the future, and admission is a
// decision about the future.
func (n *Node) reserveGrowth(elapsed time.Duration) {
	if elapsed <= 0 {
		return
	}
	cfg := n.elastic.cfg
	for class := range n.elastic.cpuFootprints {
		price := node.Resources{
			MemoryMB:  n.memoryFootprintForLocked(class).Reserve(),
			CPUMillis: n.cpuFootprintForLocked(class).Reserve(),
		}
		previous, seen := n.elastic.previousPrice[class]
		n.elastic.previousPrice[class] = price

		var perSandbox node.Resources
		if seen {
			// Only the rising component: a class that is shrinking needs no headroom, and
			// reserving one would charge twice for memory that is on its way back.
			perSandbox.MemoryMB = growthOver(price.MemoryMB, previous.MemoryMB, elapsed, cfg.GrowthHorizonS)
			perSandbox.CPUMillis = growthOver(price.CPUMillis, previous.CPUMillis, elapsed, cfg.GrowthHorizonS)
		}
		count := int64(n.elastic.classCount[class])
		n.admission.SetGrowth(class, perSandbox, node.Resources{
			MemoryMB:  perSandbox.MemoryMB * count,
			CPUMillis: perSandbox.CPUMillis * count,
		})
	}
}

// growthOver turns an observed rate into a reservation over a horizon.
func growthOver(now, before int64, elapsed time.Duration, horizonS float64) int64 {
	delta := now - before
	if delta <= 0 || horizonS <= 0 {
		return 0
	}
	ratePerSecond := float64(delta) / elapsed.Seconds()
	return int64(ratePerSecond * horizonS)
}

// releaseMemory lowers one idle sandbox's limit to what it cannot give back.
func (n *Node) releaseMemory(child cgroup.Child, cfg ElasticConfig, shrunk []string) []string {
	if n.elastic.shrunk[child.ID] || n.sandboxBusy(child) {
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
	maxMB := n.elastic.maxMemoryMBBySandbox[child.ID]
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

// throttleCPU lowers one idle sandbox's quota to what it measured using, plus a margin.
//
// CPU is the compressible dimension, so unlike a memory shrink this cannot kill the
// sandbox -- it can only make it slower. That is why the quota may be moved at all while
// the sandbox runs, and why it is still only moved on an idle one: throttling a tool call
// mid-flight lengthens the step that call belongs to.
func (n *Node) throttleCPU(child cgroup.Child, cfg ElasticConfig, throttled []string) []string {
	if n.elastic.throttled[child.ID] || n.sandboxBusy(child) {
		return throttled
	}
	quota := cgroup.QuotaToMillis(child.CPUQuotaMicros)
	target := n.elastic.cpuRateMillis[child.ID] + cfg.ReleaseMarginCPUMillis
	if target < cfg.MinCPUMillis {
		target = cfg.MinCPUMillis
	}
	if quota > 0 && quota <= target {
		// Already at or below the target: the quota is not the bound that matters.
		return throttled
	}
	if err := n.elasticDomain.SetChildCPUQuota(n.elasticParent, child.ID, target); err != nil {
		slog.Warn("elastic admission could not lower a sandbox CPU quota",
			"sandbox", child.ID, "target_millis", target, "error", err)
		return throttled
	}
	n.elastic.throttled[child.ID] = true
	return append(throttled, child.ID)
}

// releaseCPU raises a sandbox back to the CPU it declared.
func (n *Node) releaseCPU(child cgroup.Child, released []string) []string {
	if !n.elastic.throttled[child.ID] {
		return released
	}
	declared := n.elastic.maxCPUMillisBySandbox[child.ID]
	if declared <= 0 {
		delete(n.elastic.throttled, child.ID)
		return released
	}
	if err := n.elasticDomain.SetChildCPUQuota(n.elasticParent, child.ID, declared); err != nil {
		slog.Warn("elastic admission could not restore a sandbox CPU quota",
			"sandbox", child.ID, "target_millis", declared, "error", err)
		return released
	}
	delete(n.elastic.throttled, child.ID)
	return append(released, child.ID)
}

// classReserves is what this node publishes about the memory cost of a sandbox.
//
// It is reported rather than kept to itself because admission above this node -- the fleet
// quota and placement -- charges a request before this node ever sees it. A node that
// prices a sandbox from measurement but does not say so leaves every caller charging the
// maximum, which is the whole reservation this exists to remove.
func (n *Node) classMemoryReservesMB() map[string]int64 {
	if n.elastic == nil {
		return nil
	}
	n.elastic.mu.Lock()
	defer n.elastic.mu.Unlock()
	reserves := make(map[string]int64, len(n.elastic.memoryFootprints))
	for class, footprint := range n.elastic.memoryFootprints {
		reserves[class] = footprint.Reserve()
	}
	if len(reserves) == 0 {
		return nil
	}
	return reserves
}

// classCPUReserves is the same publication for CPU, in millicores.
//
// It is the one that decides how large a burst the fleet admits: a fleet that sells 360
// cores admits 180 sandboxes when each is charged the two it declared, and the sandboxes
// together were measured using a fraction of one.
func (n *Node) classCPUReservesMillis() map[string]int64 {
	if n.elastic == nil {
		return nil
	}
	n.elastic.mu.Lock()
	defer n.elastic.mu.Unlock()
	reserves := make(map[string]int64, len(n.elastic.cpuFootprints))
	for class, footprint := range n.elastic.cpuFootprints {
		reserves[class] = footprint.Reserve()
	}
	if len(reserves) == 0 {
		return nil
	}
	return reserves
}

// elasticCharge is what admission charges one request when the controller is on: the
// class's measured cost in each dimension, and the declared maximum when that is lower.
//
// The maximum is never raised by this -- a class that measures larger than it declared is
// charged what it declared, because the declared bound is also the kernel's and the node
// cannot honour more.
func (n *Node) elasticCharge(spec backend.Spec) node.Resources {
	charge := nodeResources(spec)
	if n.elastic == nil {
		return charge
	}
	n.elastic.mu.Lock()
	defer n.elastic.mu.Unlock()
	if reserve := n.memoryFootprintForLocked(spec.ResourceClass).Reserve(); reserve > 0 && reserve < charge.MemoryMB {
		charge.MemoryMB = reserve
	}
	if reserve := n.cpuFootprintForLocked(spec.ResourceClass).Reserve(); reserve > 0 && reserve < charge.CPUMillis {
		charge.CPUMillis = reserve
	}
	return charge
}

// elasticRemember records what a sandbox declared, which is what its kernel limits are
// restored to after the controller has taken them down.
func (n *Node) elasticRemember(sandboxID string, spec backend.Spec) {
	if n.elastic == nil {
		return
	}
	n.elastic.mu.Lock()
	defer n.elastic.mu.Unlock()
	n.elastic.maxMemoryMBBySandbox[sandboxID] = spec.Resources.MemoryMB
	n.elastic.maxCPUMillisBySandbox[sandboxID] = int64(spec.Resources.CPUCount * 1000)
	n.elastic.classBySandbox[sandboxID] = spec.ResourceClass
}

// footprintFor returns a class's memory estimate, creating it on first use.
func (n *Node) memoryFootprintFor(class string) *node.Footprint {
	n.elastic.mu.Lock()
	defer n.elastic.mu.Unlock()
	return n.memoryFootprintForLocked(class)
}

// memoryFootprintForLocked is footprintFor for a caller that already holds the lock.
func (n *Node) memoryFootprintForLocked(class string) *node.Footprint {
	if footprint, ok := n.elastic.memoryFootprints[class]; ok {
		return footprint
	}
	footprint := node.NewFootprint(n.elastic.cfg.Footprint, n.maxMemoryMBForClass(class))
	n.elastic.memoryFootprints[class] = footprint
	return footprint
}

// cpuFootprintFor returns a class's CPU estimate, creating it on first use.
func (n *Node) cpuFootprintFor(class string) *node.Footprint {
	n.elastic.mu.Lock()
	defer n.elastic.mu.Unlock()
	return n.cpuFootprintForLocked(class)
}

// cpuFootprintForLocked is cpuFootprintFor for a caller that already holds the lock.
func (n *Node) cpuFootprintForLocked(class string) *node.Footprint {
	if footprint, ok := n.elastic.cpuFootprints[class]; ok {
		return footprint
	}
	footprint := node.NewFootprint(n.elastic.cfg.CPUFootprint, n.maxCPUMillisForClass(class))
	n.elastic.cpuFootprints[class] = footprint
	return footprint
}

// maxMemoryMBForClass is the largest maximum declared by a sandbox of this class, which caps
// what the class's reservation may charge.
func (n *Node) maxMemoryMBForClass(class string) int64 {
	var largest int64
	for sandboxID, sandboxClass := range n.elastic.classBySandbox {
		if sandboxClass != class {
			continue
		}
		if candidate := n.elastic.maxMemoryMBBySandbox[sandboxID]; candidate > largest {
			largest = candidate
		}
	}
	return largest
}

// maxCPUMillisForClass is the largest CPU declared by a sandbox of this class.
func (n *Node) maxCPUMillisForClass(class string) int64 {
	var largest int64
	for sandboxID, sandboxClass := range n.elastic.classBySandbox {
		if sandboxClass != class {
			continue
		}
		if candidate := n.elastic.maxCPUMillisBySandbox[sandboxID]; candidate > largest {
			largest = candidate
		}
	}
	return largest
}

// elasticBudgetMB is the sandbox memory budget the controller measures against.
func (n *Node) elasticBudgetMB() int64 {
	if n.elasticBudget > 0 {
		return n.elasticBudget
	}
	return n.admission.Snapshot().Envelope.MemoryMB
}

// elasticBudgetCPUMillis is the CPU the node sells, which the CPU fraction is taken of.
//
// The envelope rather than the subtree's quota, because that is what admission grants
// against: a sandbox is admitted while the subtree is throttled nowhere, and the quota is
// only moved afterwards.
func (n *Node) elasticBudgetCPUMillis() int64 {
	return n.admission.Snapshot().Envelope.CPUMillis
}

// elasticAudit drops controller state for sandboxes this node no longer holds.
func (n *Node) elasticAudit() {
	for sandboxID := range n.elastic.maxMemoryMBBySandbox {
		if n.lifecycle.Running(sandboxID) {
			continue
		}
		delete(n.elastic.maxMemoryMBBySandbox, sandboxID)
		delete(n.elastic.maxCPUMillisBySandbox, sandboxID)
		delete(n.elastic.classBySandbox, sandboxID)
		delete(n.elastic.shrunk, sandboxID)
		delete(n.elastic.throttled, sandboxID)
		delete(n.elastic.cpuUsageNanos, sandboxID)
		delete(n.elastic.cpuRateMillis, sandboxID)
	}
}
