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
	"sort"
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
	// ShrinkAbove is the measured fraction of the CPU budget above which idle sandboxes'
	// quotas are lowered. It is a CPU threshold only: memory bounds are decided per
	// sandbox, on their own cadence and against that sandbox's own usage, because an
	// aggregate fraction says the node is busy and not which sandbox should pay.
	ShrinkAbove float64
	// RestoreBelow is the fraction below which their quotas are raised again. The gap
	// between the two is the hysteresis, so a node sitting at the threshold does not
	// throttle and release on alternate samples.
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
	// ThrottleDwell is how many consecutive samples a sandbox must be throttled before the
	// ledger charges it its whole quota rather than the rate it measured. One is too eager:
	// a single tool call that wants more than its share would set the class's price for the
	// rest of the episode.
	ThrottleDwell int

	// StartLimitMB is the kernel limit a sandbox is created under: its guarantee, and the
	// level its limit is never lowered below while it lives.
	//
	// Creating a sandbox at its declared maximum and then lowering it is the worst of both
	// worlds: the node pays the risk of every shrink and gets none of the density, because
	// the reservation was already made at the maximum. A small start with room to grow
	// costs nothing a sandbox does not ask for, and it is what keeps the sum of the
	// kernel's hard bounds -- which the machine has to be able to back, whatever the
	// ledger charged -- proportional to what is actually in use.
	//
	// Zero means "create at the declared maximum", which is the behaviour this had before.
	StartLimitMB int64
	// CeilingLimitMB is the highest a sandbox's limit may be raised to. Zero means the
	// maximum the sandbox declared. Either way the ceiling is protection, not entitlement:
	// growth stops there, and a sandbox that OOMs there has no tier left to raise.
	CeilingLimitMB int64
	// GrowAbove is the fraction of its current limit a sandbox's measured usage reaches
	// before the limit is raised. It is deliberately well below one: by the time usage is
	// at the limit the kernel is already refusing allocations, and on a host with no
	// memory.high there is no soft band between "reclaiming" and "killing".
	GrowAbove float64
	// GrowFactor is how much the limit is multiplied by on ordinary pressure.
	GrowFactor float64
	// EmergencyGrowFactor is how much it is multiplied by on an OOM. An OOM means the
	// controller was already too late, so the correction has to be larger than the one a
	// threshold crossing gets.
	EmergencyGrowFactor float64
	// ShrinkBelow is the fraction of its limit a sandbox's usage must fall under before
	// the limit may come down, and ShrinkDwellS is how long it must stay there.
	//
	// The gap between this and GrowAbove is the hysteresis, and it is wide on purpose.
	// A wrong shrink is not a lost optimisation: it kills the sandbox and discards every
	// turn of model interaction it accumulated. A wrong grow costs memory that was
	// otherwise idle.
	ShrinkBelow  float64
	ShrinkDwellS float64
	// ShrinkFactor is what the limit is multiplied by when it comes down. Halving rather
	// than fitting to the measurement is what keeps a shrink from landing on the working
	// set: the measurement is one sample, and the working set is not.
	ShrinkFactor float64
	// SafetyFraction is the share of the sandbox budget growth may not consume. It is the
	// physical counterpart of the ledger's admission reserve: the ledger keeps capacity for
	// admissions, this keeps it for sandboxes that have not asked yet.
	SafetyFraction float64
	// ReapForeign removes cgroups in the sandbox subtree that belong to nothing: no
	// allocation, no tasks, and holding under ReapUsageMB.
	//
	// A destroyed container leaves its cgroup behind when the kernel still holds a
	// reference to it, and the directory then outlives the container for as long as the
	// node runs. What that costs is the accounting rather than the memory -- they are
	// counted in the observed total, and their cumulative OOM counters are counted with
	// everybody else's, so a node that has run for a week reports faults nobody owns.
	ReapForeign bool
	// ReapUsageMB is the most an orphan may be holding before it is left alone. A cgroup
	// with real pages in them is real memory, and removing the directory would not free
	// them -- it would only stop anyone from being able to say whose they are.
	ReapUsageMB int64
	// ReapDwellS is how long a cgroup must have been foreign before it is a candidate.
	//
	// This is the guard that makes the difference between an orphan and a container that
	// has not been registered yet. Docker creates a container's cgroups in every
	// hierarchy at create time and does not write its process into them until start; in
	// between, the directory has no tasks and no usage and is indistinguishable from
	// garbage. Removing it makes the start fail with ENODEV on a cgroup that no longer
	// exists. A runtime owns its container in milliseconds, and this waits far longer.
	ReapDwellS float64
	// LimitSampleIntervalS is how often the limit controller reads the domain. It is much
	// shorter than the pricing cadence because the two answer different questions: a price
	// is a statistic over a class and is noise at this resolution, while a limit is a
	// bound on one sandbox that a burst can outrun inside a single slow interval.
	LimitSampleIntervalS float64
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
	if cfg.ThrottleDwell <= 0 {
		cfg.ThrottleDwell = DefaultThrottleDwell
	}
	if cfg.GrowthHorizonS <= 0 {
		cfg.GrowthHorizonS = DefaultGrowthHorizonS
	}
	// Sixty percent. Usage at the limit already means the kernel is refusing allocations;
	// on a v1 host there is no memory.high to warn first, so the threshold has to leave
	// room for the interval it takes to notice.
	if cfg.GrowAbove <= 0 || cfg.GrowAbove >= 1 {
		cfg.GrowAbove = 0.60
	}
	if cfg.GrowFactor <= 1 {
		cfg.GrowFactor = 2.0
	}
	if cfg.EmergencyGrowFactor <= 1 {
		cfg.EmergencyGrowFactor = 4.0
	}
	if cfg.ShrinkBelow <= 0 {
		cfg.ShrinkBelow = 0.20
	}
	if cfg.ShrinkDwellS <= 0 {
		cfg.ShrinkDwellS = DefaultShrinkDwellS
	}
	if cfg.ShrinkFactor <= 0 || cfg.ShrinkFactor >= 1 {
		cfg.ShrinkFactor = 0.5
	}
	if cfg.SafetyFraction <= 0 || cfg.SafetyFraction >= 1 {
		cfg.SafetyFraction = 0.05
	}
	if cfg.LimitSampleIntervalS <= 0 {
		cfg.LimitSampleIntervalS = DefaultLimitSampleIntervalS
	}
	if cfg.ReapUsageMB <= 0 {
		cfg.ReapUsageMB = DefaultReapUsageMB
	}
	if cfg.ReapDwellS <= 0 {
		cfg.ReapDwellS = DefaultReapDwellS
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

// DefaultThrottleDwell is how many consecutive samples a sandbox must be throttled before
// the ledger charges it its whole quota.
//
// Three: at the 200ms cadence that is 600ms of being held down, which a real burst
// clears easily and an idle sandbox never reaches. It is the same asymmetry the
// controller applies to bounds -- quick to believe a sandbox needs more, slow to keep
// believing it.
const DefaultThrottleDwell = 3

// DefaultGrowthHorizonS is how far ahead admission reserves a class's growth. One minute
// covers a sampling interval, a fleet report and the time a batch takes to arrive at that
// price -- long enough that the reservation is real, short enough that it does not price a
// sandbox at its peak before it has any history.
const DefaultGrowthHorizonS = 60.0

// DefaultShrinkDwellS is how long a sandbox's measured usage must stay below ShrinkBelow
// before its limit comes down.
//
// A minute, against ten seconds for the CPU quota: memory is the dimension where a wrong
// move is unrecoverable. The dwell is not caution about the measurement, which is instant,
// but about the workload -- an agent between tool calls looks idle for seconds at a time
// and is about to allocate again.
const DefaultShrinkDwellS = 60.0

// DefaultLimitSampleIntervalS is how often the limit controller reads the sandbox domain.
//
// Two hundred milliseconds, because that is the interval at which a burst is visible at
// all: one tool call can move a sandbox's demand by an order of magnitude inside a second,
// and the pricing cadence of five seconds sees only the result. The cost of the read is
// linear in the number of sandboxes and is measured in the benchmark, not assumed.
const DefaultLimitSampleIntervalS = 0.2

// DefaultReapUsageMB is the most an orphaned cgroup may be holding before the node leaves
// it alone.
//
// A few tens of MiB: enough to cover the residue the kernel keeps for a container that has
// gone, and far below anything a sandbox that is doing anything would be charged. The
// point is not to free the memory -- rmdir does not -- but to be able to say, before
// removing a directory, that no work is in it.
const DefaultReapUsageMB = 64

// DefaultReapDwellS is how long a cgroup must have been foreign before the node will
// consider removing it.
//
// A minute, for the same reason the shrink dwell is a minute: the observation that makes a
// cgroup look disposable is also true of a container at the wrong moment of its own life,
// and the cost of being wrong is an episode rather than an optimisation.
const DefaultReapDwellS = 60.0

// daemonOwnedTTL is how long the runtime's own inventory is reused.
//
// Enumerating a daemon with hundreds of containers is not a thing to do on every tick, and
// the answer only ever loses entries -- a container that exists now existed a second ago.
const daemonOwnedTTL = 30 * time.Second

// DefaultLimitSampleInterval is DefaultLimitSampleIntervalS as a duration, for the
// deployment's loop, which needs a ticker rather than a number.
func DefaultLimitSampleInterval() time.Duration {
	return time.Duration(DefaultLimitSampleIntervalS * float64(time.Second))
}

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
	// Throttled and Released name the sandboxes whose CPU quota changed. Memory bounds
	// moved in a tick are reported by LimitReport, which runs on its own cadence.
	Throttled []string
	Released  []string
	// OOMKills is the total the kernel reports across the sandboxes this node owns.
	//
	// Deliberately not the total across the domain. The counter is cumulative and a
	// cgroup outlives the container that made it, so a domain total counts kills that
	// belong to sandboxes that no longer exist -- which is a number that never goes down,
	// points at nothing, and reads as an ongoing fault. A node that restarted shows zero
	// not because anything was fixed but because the memory of it was.
	OOMKills int64
	// Reaped names the cgroups this tick removed because nothing owned them.
	Reaped []string
	// Foreign is how many cgroups in the domain are not sandboxes this node owns, and
	// ForeignMB what they are charged. They are counted apart because they are real
	// memory and real directories -- the machine carries them -- but they are not
	// anybody's episode, so nothing in the controller may act on them and no alarm may
	// be raised in their name.
	Foreign   int
	ForeignMB int64
	// ThrottledPct is the share of the interval the sandboxes spent stopped for exceeding
	// their CPU quota, averaged over the ones being measured. It is the number that says
	// the node admitted more CPU demand than it can serve, and it is invisible in usage:
	// usage saturates at the quota and looks like a full but healthy node.
	ThrottledPct float64
}

// LimitReport is one limit tick's decisions.
//
// It is separate from ElasticReport because the two run at different cadences and answer
// different questions: this one is about the kernel's hard bounds on individual sandboxes
// over the next fraction of a second, the other about what a class costs over the next
// minute.
type LimitReport struct {
	// Grown names the sandboxes whose limit was raised, and Shrunk the ones whose limit
	// came down.
	Grown  []string
	Shrunk []string
	// Preempted names the sandboxes that gave back a loan so another could grow.
	Preempted []string
	// OOMed names the sandboxes the kernel killed a process in since the last tick.
	OOMed []string
	// Ended names the sandboxes this tick ended because they were killed again with their
	// limit already at the ceiling.
	Ended []string
	// LimitSumMB is the sum of the kernel's hard bounds across the domain, which is the
	// number the machine has to be able to back regardless of what the ledger charged.
	LimitSumMB int64
	// SpareMB is what is left of the sandbox budget after those bounds and the safety
	// share, which is what growth may spend.
	SpareMB int64
}

// ElasticLimitsTick moves each sandbox's kernel memory bound one step.
//
// It runs far more often than ElasticTick, and that is the point: a bound is a promise
// about the next allocation, and a tool call can outrun it inside one slow interval. The
// pricing half of the controller is a statistic over a class and would be noise at this
// resolution, so the two are deliberately separate loops rather than one loop at a
// compromise cadence.
//
// The rules are asymmetric on purpose. Growing is cheap -- memory that is not asked for is
// not spent -- so it happens on a single sample, and on an OOM it happens by a larger
// factor still, because an OOM means this tick was already too late. Shrinking is
// expensive: a bound below a sandbox's working set does not slow it down, it kills it and
// discards every turn of model interaction it had accumulated. So a shrink needs a fraction
// of the limit that no sane workload sits at by accident, and it needs that to hold for a
// dwell, because an agent between tool calls looks idle for seconds and then allocates.
func (n *Node) ElasticLimitsTick(ctx context.Context) LimitReport {
	report := LimitReport{}
	if n.elastic == nil || !n.elastic.cfg.Enabled || len(n.elasticDomain.Hierarchies) == 0 {
		return report
	}
	n.elastic.mu.Lock()
	defer n.elastic.mu.Unlock()
	cfg := n.elastic.cfg
	children, err := n.elasticDomain.Children(n.elasticParent)
	if err != nil {
		// A domain that cannot be read is not a domain that is idle. No bound moves.
		return report
	}

	// Everything below is in bytes, which is what the kernel speaks; the budget is
	// published in MiB, and mixing the two here would make "spare" a number from another
	// unit system rather than a wrong number -- the kind that fails only at fleet scale.
	budget := n.elasticBudgetMB() * mib
	// The domain is read once. Every decision in this tick works from this slice, including
	// preemption, which needs to know what every *other* sandbox's bound is: re-reading the
	// domain per sandbox is one directory listing and a handful of files per sandbox per
	// sandbox, which is a tick that costs seconds at two hundred sandboxes rather than
	// milliseconds. The entries are mutable so a write made here is seen by the decisions
	// that follow it in the same tick.
	tracked := make([]boundState, 0, len(children))
	var limitSum int64
	for _, child := range children {
		if _, known := n.elastic.classBySandbox[child.ID]; !known {
			continue
		}
		if child.Limit <= 0 {
			continue
		}
		oomDelta, oomKnown := n.oomDeltaLocked(child)
		failDelta, failKnown := n.failDeltaLocked(child)
		tracked = append(tracked, boundState{
			id: child.ID, usage: child.Usage, resident: child.Resident, limit: child.Limit,
			start: n.elastic.startMBBySandbox[child.ID] * mib, ceiling: n.ceilingLocked(child.ID),
			oomDelta: oomDelta, oomKnown: oomKnown, failDelta: failDelta, failKnown: failKnown,
		})
		limitSum += child.Limit
	}
	// Growth spends from what the machine can actually back, not from what the ledger
	// charged. The two are different numbers by design -- the ledger sells the average --
	// and the kernel's bounds are the ones that have to hold when several sandboxes are
	// at their ceiling at once.
	spare := budget - int64(float64(budget)*cfg.SafetyFraction) - limitSum
	initialSpare := spare

	// The preemptible loans, ranked largest first and computed once. Reclaiming one big
	// loan is one write and one reclaim; several small ones are several of each.
	loans := make([]*boundState, 0, len(tracked))
	for i := range tracked {
		e := &tracked[i]
		if e.limit > e.start && e.usage < e.start {
			loans = append(loans, e)
		}
	}
	sort.Slice(loans, func(i, j int) bool { return loans[i].limit-loans[i].start > loans[j].limit-loans[j].start })

	now := time.Now()
	var ended []string
	next := 0
	for i := range tracked {
		e := &tracked[i]
		switch {
		case e.oomKnown && e.oomDelta > 0:
			// The kernel killed something in this sandbox. It is the emergency signal: the
			// threshold was crossed and then some, so the correction is larger than the one
			// a threshold crossing gets.
			report.OOMed = append(report.OOMed, e.id)
			if atCeiling(e.limit, e.ceiling) {
				if n.elastic.exhausted[e.id] {
					// Killed again with nowhere left to grow. This trajectory is already
					// corrupted by starvation; ending it and saying so lets the trainer
					// resample the task instead of learning from it.
					ended = append(ended, e.id)
					continue
				}
				// The first one is not enough to condemn it: the kill may have been of a
				// runaway child, and growth may still be possible once a loan is taken back.
				n.elastic.exhausted[e.id] = true
				slog.Error("a sandbox was killed for memory with its limit already at the ceiling",
					"sandbox", e.id, "class", n.elastic.classBySandbox[e.id],
					"limit_mb", e.limit/mib, "usage_mb", e.usage/mib, "rss_mb", e.resident/mib)
				break
			}
			if grew := n.raise(e, e.limit*int64(cfg.EmergencyGrowFactor), &spare, loans, &next); grew {
				report.Grown = append(report.Grown, e.id)
			}
		case float64(e.usage) >= float64(e.limit)*cfg.GrowAbove || (e.failKnown && e.failDelta > 0):
			// Usage approaching the bound, or the kernel reporting that the bound was
			// already in the way. failcnt is the early signal and OOM is the late one: on
			// a host with no memory.high there is no soft band between them, so the
			// threshold has to be crossed well before the ceiling.
			if e.limit < e.ceiling {
				if grew := n.raise(e, e.limit*int64(cfg.GrowFactor), &spare, loans, &next); grew {
					report.Grown = append(report.Grown, e.id)
				}
			}
		case float64(e.usage) < float64(e.limit)*cfg.ShrinkBelow:
			if started, running := n.elastic.lowSince[e.id]; !running {
				n.elastic.lowSince[e.id] = now
			} else if now.Sub(started) >= time.Duration(cfg.ShrinkDwellS*float64(time.Second)) {
				if target := n.shrinkTarget(e, cfg); target > 0 && target < e.limit {
					if err := n.elasticDomain.SetChildLimit(n.elasticParent, e.id, target); err == nil {
						report.Shrunk = append(report.Shrunk, e.id)
						spare += e.limit - target
						e.limit = target
						delete(n.elastic.lowSince, e.id)
					}
				}
			}
		default:
			// Inside the hysteresis band: one sample neither grows nor shrinks, which is
			// what stops the limit oscillating around the threshold.
			delete(n.elastic.lowSince, e.id)
		}
	}

	// Ended outside the loop and outside this lock's other work, so the lifecycle is never
	// re-entered from under the controller's own mutex.
	for _, id := range ended {
		if err := n.lifecycle.Release(ctx, id, node.ExitOOM); err == nil {
			report.Ended = append(report.Ended, id)
			slog.Error("ended a sandbox that was killed for memory with no tier left to raise",
				"sandbox", id, "class", n.elastic.classBySandbox[id])
			delete(n.elastic.exhausted, id)
			delete(n.elastic.startMBBySandbox, id)
			delete(n.elastic.maxMemoryMBBySandbox, id)
		}
	}
	// The bounds the machine now has to back are the ones it had at the start plus
	// everything this tick handed out, and "spare" already carries that difference
	// exactly: growth subtracts from it and reclamation adds to it. Deriving the sum from
	// it is cheaper and cannot disagree with it.
	report.LimitSumMB = (limitSum + (initialSpare - spare)) / mib
	report.SpareMB = spare / mib
	return report
}

// boundState is one sandbox's bounds, as this tick sees them.
//
// It is a working copy rather than the reading itself because a tick makes writes, and a
// decision taken after a write has to see it: two sandboxes may draw from the same spare
// budget in one tick, and a preemption has to know that the loan it is about to take back
// was not already given to someone else in the tick before.
type boundState struct {
	id        string
	usage     int64
	resident  int64
	limit     int64
	start     int64
	ceiling   int64
	oomDelta  int64
	oomKnown  bool
	failDelta int64
	failKnown bool
}

// mib is the unit the configuration speaks and the kernel does not.
const mib = 1024 * 1024

// atCeiling reports whether a limit has nowhere left to grow. A limit within a MiB of the
// ceiling is at it: the difference is rounding in the kernel's own accounting, and treating
// it as headroom would raise the bound by a MiB and call that help.
func atCeiling(limit, ceiling int64) bool {
	return ceiling <= 0 || limit+mib > ceiling
}

// oomDeltaLocked and failDeltaLocked report how many times the kernel killed a process, or
// refused an allocation, since the previous reading.
//
// Both counters are cumulative and both survive the container that owned them, so the
// first sighting is recorded without being acted on: a cgroup that already says four
// kills belongs to a sandbox that no longer exists, and reading that as fresh evidence
// would end an episode for something it never did.
func (n *Node) oomDeltaLocked(child cgroup.Child) (int64, bool) {
	previous, known := n.elastic.oomCountBySandbox[child.ID]
	n.elastic.oomCountBySandbox[child.ID] = child.OOMKills
	if !known {
		return 0, false
	}
	return child.OOMKills - previous, true
}

func (n *Node) failDeltaLocked(child cgroup.Child) (int64, bool) {
	previous, known := n.elastic.failCountBySandbox[child.ID]
	n.elastic.failCountBySandbox[child.ID] = child.MemoryFailCount
	if !known {
		return 0, false
	}
	return child.MemoryFailCount - previous, true
}

// ceilingLocked is the highest this sandbox's limit may be raised to: what it declared,
// capped by what the node will allow any one sandbox to hold.
func (n *Node) ceilingLocked(sandboxID string) int64 {
	ceiling := n.elastic.maxMemoryMBBySandbox[sandboxID] * mib
	if cap := n.elastic.cfg.CeilingLimitMB * mib; cap > 0 && (ceiling <= 0 || ceiling > cap) {
		ceiling = cap
	}
	if start := n.elastic.startMBBySandbox[sandboxID] * mib; start > ceiling {
		ceiling = start
	}
	return ceiling
}

// raise moves one sandbox's bound up towards target, spending from what the machine has
// spare and taking back unused loans when it does not have enough.
//
// Callers hold the controller's lock. loans is this tick's ranked list of preemptible
// loans and next is how far into it the tick has already got, so one loan is taken back at
// most once even though every sandbox in a burst asks in the same tick.
func (n *Node) raise(e *boundState, target int64, spare *int64, loans []*boundState, next *int) bool {
	if target > e.ceiling {
		target = e.ceiling
	}
	need := target - e.limit
	if need <= 0 {
		return false
	}
	for need > *spare && *next < len(loans) {
		l := loans[*next]
		*next++
		if l == e || l.limit <= l.start {
			continue
		}
		// Still not using its guarantee? The reading is from this tick's single pass and
		// nothing has changed it since, so the answer is still the one it was.
		if l.usage >= l.start {
			continue
		}
		if err := n.elasticDomain.SetChildLimit(n.elasticParent, l.id, l.start); err != nil {
			continue
		}
		*spare += l.limit - l.start
		l.limit = l.start
	}
	if need > *spare {
		// Take what there is rather than nothing. A partial growth still buys time, and the
		// next tick is one interval away.
		need = *spare
		if need <= 0 {
			return false
		}
		target = e.limit + need
	}
	if err := n.elasticDomain.SetChildLimit(n.elasticParent, e.id, target); err != nil {
		slog.Warn("elastic admission could not raise a sandbox limit",
			"sandbox", e.id, "target_mb", target/mib, "error", err)
		return false
	}
	*spare -= need
	e.limit = target
	// A sandbox that just grew is not idle, whatever the sample said before it grew.
	delete(n.elastic.lowSince, e.id)
	return true
}

// shrinkTarget is where one sandbox's bound may come down to, or zero when it may not come
// down at all.
//
// The floors, in order of what they protect: half the current bound, because fitting the
// bound to one sample is how it lands on a working set; the guarantee, because that was
// promised and only preemption of a loan may take it back; and what the sandbox has
// charged plus a margin, because a bound below the pages is not a slowdown.
func (n *Node) shrinkTarget(e *boundState, cfg ElasticConfig) int64 {
	target := int64(float64(e.limit) * cfg.ShrinkFactor)
	floor := e.start
	if min := cfg.MinLimitMB * mib; floor < min {
		floor = min
	}
	// Usage, not resident: the bound charges page cache too, so a floor taken from
	// anonymous pages alone would sit below what the sandbox already has charged and force
	// a reclaim of the very files the next command is about to read.
	if used := e.usage + cfg.ReleaseMarginMB*mib; floor < used {
		floor = used
	}
	if target < floor {
		target = floor
	}
	if target >= e.limit {
		return 0
	}
	return target
}

// applyStartLimit returns the spec a sandbox is created under: its memory bound is the
// guarantee rather than the declared maximum.
//
// The ceiling is remembered separately and is what growth may reach, so lowering the
// creation bound does not lower what the sandbox may become -- only what it must be given
// before it has asked for anything.
func (n *Node) applyStartLimit(spec backend.Spec) backend.Spec {
	if n.elastic == nil || !n.elastic.cfg.Enabled {
		return spec
	}
	n.elastic.mu.Lock()
	start := n.startLimitLocked(spec)
	n.elastic.mu.Unlock()
	if start <= 0 || start >= spec.Resources.MemoryMB {
		return spec
	}
	out := spec
	out.Resources.MemoryMB = start
	return out
}

// startLimitLocked is the memory bound a new sandbox is created under, in MiB.
//
// Unset, it is the declared maximum and nothing moves -- which is the shape this had
// before, and which a deployment that has not measured its workloads should keep. Set, it
// is the guarantee: the level the sandbox's bound is never lowered below, and the level
// growth starts from. It is never more than what was declared, because a caller asking for
// a smaller sandbox is asking for a smaller sandbox.
func (n *Node) startLimitLocked(spec backend.Spec) int64 {
	cfg := n.elastic.cfg
	declared := spec.Resources.MemoryMB
	if declared <= 0 || cfg.StartLimitMB <= 0 {
		return declared
	}
	start := cfg.StartLimitMB
	if min := cfg.MinLimitMB; start < min {
		start = min
	}
	if start > declared {
		start = declared
	}
	return start
}

// elasticRemember records what a sandbox declared, which is the ceiling its kernel limit
// may be raised to, and the guarantee it was created under, which is the floor it may not
// be lowered below.
func (n *Node) elasticRemember(sandboxID string, spec backend.Spec) {
	if n.elastic == nil {
		return
	}
	n.elastic.mu.Lock()
	defer n.elastic.mu.Unlock()
	n.elastic.maxMemoryMBBySandbox[sandboxID] = spec.Resources.MemoryMB
	n.elastic.maxCPUMillisBySandbox[sandboxID] = int64(spec.Resources.CPUCount * 1000)
	n.elastic.classBySandbox[sandboxID] = spec.ResourceClass
	n.elastic.startMBBySandbox[sandboxID] = n.startLimitLocked(spec)
}

// reapForeign removes the cgroups in this node's subtree that belong to nothing.
//
// A container that is destroyed leaves its cgroup behind when the kernel still holds a
// reference to it -- a charged page, a zombie -- and the directory then outlives the
// container for as long as the node runs. They accumulate one wave at a time, and what
// they cost is the accounting rather than the memory: they are counted in the domain's
// observed usage, and before this change their cumulative OOM counters were counted with
// everybody else's, so a node that had run for a while reported faults nobody owned.
//
// The guards, in order of how much they protect:
//
//   - The node must not own it, and neither may the runtime. A cgroup the controller has
//     registered, one with a live allocation, or one the daemon says is a container of
//     its own is somebody's sandbox -- including a warm-pool entry, which has no lease
//     and no tasks and looks exactly like garbage.
//   - It must have been foreign for ReapDwellS. This is the one that separates an orphan
//     from a container that has simply not been registered yet: Docker creates the
//     cgroups in every hierarchy at create time and writes the process into them at
//     start, and in between the directory is empty and indistinguishable. Removing it
//     makes the start fail with ENODEV against a cgroup that no longer exists.
//   - It must be holding almost nothing. A cgroup with real pages in it is real memory,
//     and removing the directory does not free them -- it only stops anyone from being
//     able to say whose they are.
//   - The kernel must agree. rmdir on a cgroup that still holds a task fails, and that is
//     the only test for "is anything in there" that cannot be wrong.
func (n *Node) reapForeign(children []cgroup.Child, now time.Time) []string {
	if !n.elastic.cfg.ReapForeign || len(n.elasticDomain.Hierarchies) == 0 {
		return nil
	}
	dwell := time.Duration(n.elastic.cfg.ReapDwellS * float64(time.Second))
	limit := n.elastic.cfg.ReapUsageMB * mib
	if limit <= 0 {
		limit = DefaultReapUsageMB * mib
	}

	// First-seen times, refreshed against this tick's reading. A cgroup that has gone is
	// forgotten, so a later one that reuses the name is not treated as old.
	present := make(map[string]struct{}, len(children))
	var candidates []cgroup.Child
	for _, child := range children {
		if _, known := n.elastic.classBySandbox[child.ID]; known {
			continue
		}
		if n.lifecycle.Running(child.ID) {
			continue
		}
		present[child.ID] = struct{}{}
		if _, seen := n.elastic.foreignSince[child.ID]; !seen {
			n.elastic.foreignSince[child.ID] = now
			continue
		}
		if now.Sub(n.elastic.foreignSince[child.ID]) < dwell {
			continue
		}
		if child.Usage > limit {
			continue
		}
		candidates = append(candidates, child)
	}
	for id := range n.elastic.foreignSince {
		if _, here := present[id]; !here {
			delete(n.elastic.foreignSince, id)
		}
	}
	if len(candidates) == 0 {
		return nil
	}

	owned := n.daemonOwned(now)
	var reaped []string
	for _, child := range candidates {
		if _, theirs := owned[child.ID]; theirs {
			// The runtime says this is a container of its own. It may be stopped, or idle
			// in a pool, or mid-create: none of those are states a node may delete.
			continue
		}
		if err := n.elasticDomain.RemoveChild(n.elasticParent, child.ID); err != nil {
			if !n.elastic.reapFailed[child.ID] {
				n.elastic.reapFailed[child.ID] = true
				slog.Info("elastic admission could not remove an orphaned cgroup",
					"sandbox", child.ID, "error", err)
			}
			continue
		}
		delete(n.elastic.reapFailed, child.ID)
		delete(n.elastic.foreignSince, child.ID)
		reaped = append(reaped, child.ID)
		slog.Info("elastic admission removed an orphaned cgroup",
			"sandbox", child.ID, "usage_mb", child.Usage/mib,
			"foreign_for", now.Sub(n.elastic.foreignSince[child.ID]).String())
	}
	return reaped
}

// daemonOwned asks the runtimes this node hosts which sandboxes they actually have.
//
// The node's own records are not enough to decide that a cgroup belongs to nothing: a
// warm-pool entry has no lease and no tasks, and a container between create and start has
// neither a registration nor a process. Only the runtime knows. The answer is cached
// because enumerating a daemon is not a thing to do on every tick, and because it only
// ever loses entries -- a container that exists now existed a moment ago.
func (n *Node) daemonOwned(now time.Time) map[string]struct{} {
	if !n.elastic.ownedAt.IsZero() && now.Sub(n.elastic.ownedAt) < daemonOwnedTTL {
		return n.elastic.owned
	}
	out := make(map[string]struct{})
	for _, name := range n.order {
		reconciler, can := n.backends[name].(backend.Reconciler)
		if !can {
			// A runtime whose objects live on an external platform has nothing on this
			// machine to protect.
			continue
		}
		found, err := reconciler.Reconcile(context.Background())
		if err != nil {
			slog.Warn("elastic admission could not enumerate a runtime before reaping",
				"backend", name, "error", err)
			// Not knowing is not permission. Leave everything alone until the runtime
			// will answer.
			return n.elastic.owned
		}
		for _, rec := range found {
			out[rec.Handle.SandboxID] = struct{}{}
		}
	}
	n.elastic.owned = out
	n.elastic.ownedAt = now
	return out
}

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
	// startMBBySandbox is the guarantee each sandbox was created under. A bound is never
	// lowered below it by the slow shrink -- that would be taking back what was promised
	// -- but the portion above it is a loan, and a loan is preemptible.
	startMBBySandbox map[string]int64
	// lowSince is when each sandbox's usage first fell under ShrinkBelow, which is what
	// the dwell is measured from. Absent means the clock is not running.
	lowSince map[string]time.Time
	// failCount and oomCount are the previous readings of each sandbox's memory.failcnt
	// and oom_kill counters. Both are cumulative, so the counter is not the signal -- the
	// delta is. An OOM that happened before the sandbox was admitted is not the
	// controller's business and must not be read as one.
	failCountBySandbox map[string]int64
	oomCountBySandbox  map[string]int64
	// exhausted names the sandboxes that OOMed with their bound already at the ceiling,
	// so a second one is an ending rather than another growth attempt.
	exhausted map[string]bool
	// reapFailed names the orphaned cgroups the kernel would not let go, so the node says
	// so once instead of every five seconds for the rest of the run.
	reapFailed map[string]bool
	// foreignSince is when each cgroup in the domain was first seen as belonging to
	// nobody. Nothing is removed on the strength of one observation.
	foreignSince map[string]time.Time
	// owned is what the runtimes last said they have, and ownedAt when. A cgroup the
	// runtime owns is not an orphan whatever it looks like.
	owned   map[string]struct{}
	ownedAt time.Time
	// classBySandbox is what the sandbox was admitted as.
	classBySandbox map[string]string
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
	// throttleRun counts consecutive throttled samples per sandbox, which is what the
	// quota substitution waits on.
	throttleRun map[string]int
	// lastThrottledPct is the most recent throttling measurement, published with the
	// envelope it governs: a report that disagrees with the decision it explains is
	// worse than no report.
	lastThrottledPct float64
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
	var foreignBytes int64
	foreign := 0
	for _, child := range children {
		observedBytes += child.Usage
		class, known := n.elastic.classBySandbox[child.ID]
		if !known {
			// In the domain but not this node's sandbox: an orphaned cgroup, a container
			// belonging to another runtime, or one created by a previous process. Its
			// memory is real and the machine carries it, so it stays in the observed total
			// that protects admission -- but its kills and its failures are not this run's
			// and must not be reported as though they were.
			foreign++
			foreignBytes += child.Usage
			continue
		}
		oomKills += child.OOMKills
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
	report.Foreign = foreign
	report.ForeignMB = foreignBytes / (1024 * 1024)
	if reap := n.reapForeign(children, time.Now()); len(reap) > 0 {
		report.Reaped = reap
	}
	if measured > 0 {
		report.ThrottledPct = throttledTotal / float64(measured)
	}
	// Throttling is the feedback term for CPU packing, in the role that utilization plays
	// for memory: memory asks how full the machine is, CPU asks how much the kernel is
	// already refusing. Without a reading the gate falls back to the declared CPU envelope,
	// because "the reservations are generous" is exactly the claim an unmeasured node
	// cannot make.
	n.admission.SetThrottle(report.ThrottledPct)
	n.elastic.lastThrottledPct = report.ThrottledPct
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

	// Reprice each class from what its sandboxes measured. The price is what the *next*
	// sandbox of that class is charged at admission, and a class with no running sandbox
	// keeps its last estimate, which is what seeds the following batch: an empty class
	// would otherwise fall back to the minimum between two steps and under-reserve the
	// batch that follows.
	//
	// The price is deliberately not pushed back into the ledger's charge for the sandboxes
	// that are already running. The ledger holds what the node promised, and a promise is
	// made once, at admission; moving it afterwards cannot be undone, because the return
	// only knows the amount it was handed. What tracks a sandbox that outgrew its promise
	// is the measurement above, which headroom already treats as a floor, and the growth
	// reservation, which is recomputed from scratch on every tick. Both are self-healing;
	// an adjusted charge is not, and it ratchets until the node refuses everything.
	for class, residents := range memoryByClass {
		n.memoryFootprintForLocked(class).Observe(residents)
	}
	for class, rates := range cpuMillisByClass {
		n.cpuFootprintForLocked(class).Observe(rates)
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

	// Memory bounds are not decided here. They are per-sandbox decisions on a 200ms
	// cadence -- see ElasticLimitsTick -- because a bound is a promise about the next
	// allocation, and an aggregate fraction over the whole domain is not: it says the node
	// is busy, which is neither a reason to take memory from a particular sandbox nor a
	// signal that arrives soon enough to matter.
	cpuFraction := 0.0
	if report.BudgetCPUMillis > 0 {
		cpuFraction = float64(report.ObservedCPUMillis) / float64(report.BudgetCPUMillis)
	}
	for _, child := range children {
		if _, known := n.elastic.classBySandbox[child.ID]; !known {
			continue
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
	//
	// The substitution is gated on throttling that *persists*, for the same reason a
	// scale-up is allowed on one sample but a scale-down is not: one throttled interval is
	// a tool call that wanted more than its share, and pricing the class at the full quota
	// for the rest of the episode on the strength of it is how the charge ends up near the
	// limit while the sandboxes are using a fifth of it. Sustained throttling is the
	// sandbox being held down, which is the thing worth paying for.
	if throttled >= n.elastic.cfg.ThrottledAbove {
		n.elastic.throttleRun[child.ID]++
	} else {
		n.elastic.throttleRun[child.ID] = 0
	}
	if n.elastic.throttleRun[child.ID] >= n.elastic.cfg.ThrottleDwell {
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
	// Clamped to one interval. cgroup v1's `throttled_time` is the sum over the tasks in
	// the cgroup, not wall time: four spinners held down together accumulate four seconds
	// of throttling per second. Divided by the interval that yields values well above one
	// -- measured at 2.0 on this cluster. An unnormalised "share of the interval" above
	// one is not a share at all, and because it is the term that withdraws CPU overcommit,
	// it turned a proportional control law into a switch: any throttling at all read as
	// infinitely far past the target, so the expansion was always fully withdrawn.
	fraction := float64(delta) / float64(elapsed.Nanoseconds())
	if fraction > 1 {
		return 1
	}
	return fraction
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
//
// Every map the controller keeps is indexed by sandbox id, so a sandbox that goes without
// this leaves its start, its dwell clock and its OOM counters behind. The counters are the
// dangerous ones: they are cumulative, so a later sandbox that inherits nothing else still
// inherits a reading that says "this one has already been killed", which is how a fresh
// episode gets ended for an OOM that was never its own.
func (n *Node) elasticAudit() {
	for sandboxID := range n.elastic.maxMemoryMBBySandbox {
		if n.lifecycle.Running(sandboxID) {
			continue
		}
		delete(n.elastic.maxMemoryMBBySandbox, sandboxID)
		delete(n.elastic.maxCPUMillisBySandbox, sandboxID)
		delete(n.elastic.classBySandbox, sandboxID)
		delete(n.elastic.startMBBySandbox, sandboxID)
		delete(n.elastic.lowSince, sandboxID)
		delete(n.elastic.failCountBySandbox, sandboxID)
		delete(n.elastic.oomCountBySandbox, sandboxID)
		delete(n.elastic.exhausted, sandboxID)
		delete(n.elastic.throttled, sandboxID)
		delete(n.elastic.cpuUsageNanos, sandboxID)
		delete(n.elastic.cpuRateMillis, sandboxID)
		delete(n.elastic.cpuThrottledNanos, sandboxID)
		delete(n.elastic.cpuThrottle, sandboxID)
		delete(n.elastic.throttleRun, sandboxID)
	}
}
