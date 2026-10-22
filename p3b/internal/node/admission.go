// Package node is one machine's sandbox authority.
//
// The node is the last word on admission, and it has information the cluster
// does not: placement compares a headroom reported seconds ago, while the node
// knows its pressure now. So a placement decision is a proposal and this is
// where it is accepted. A refusal is not a failure; it tells the gateway to ask
// placement for a different node, which is what stops a stale fleet view from
// overriding a local limit.
//
// Admission is per class, because that is the granularity the fleet is shared
// at. A borrower may not take a class's unmet guarantee, so comparing a request
// against the envelope remainder would overstate what it can get by the whole of
// every other queued guarantee.
package node

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// Resources is what one sandbox asks of a node.
type Resources struct {
	MemoryMB  int64
	CPUMillis int64
	GPUCount  int32
	DiskMB    int64
}

// Sub returns the difference, clamped at zero.
func (r Resources) Sub(o Resources) Resources {
	return Resources{
		MemoryMB:  maxI64(r.MemoryMB-o.MemoryMB, 0),
		CPUMillis: maxI64(r.CPUMillis-o.CPUMillis, 0),
		GPUCount:  maxI32(r.GPUCount-o.GPUCount, 0),
		DiskMB:    maxI64(r.DiskMB-o.DiskMB, 0),
	}
}

// Add returns the sum.
func (r Resources) Add(o Resources) Resources {
	return Resources{r.MemoryMB + o.MemoryMB, r.CPUMillis + o.CPUMillis, r.GPUCount + o.GPUCount, r.DiskMB + o.DiskMB}
}

// Scale returns this vector multiplied by a fraction, rounded down.
func (r Resources) Scale(f float64) Resources {
	return Resources{
		MemoryMB:  int64(float64(r.MemoryMB) * f),
		CPUMillis: int64(float64(r.CPUMillis) * f),
		GPUCount:  int32(float64(r.GPUCount) * f),
		DiskMB:    int64(float64(r.DiskMB) * f),
	}
}

// FitsIn reports whether this vector is covered by another in every dimension a
// caller actually asked for. A zero dimension states no requirement, so it is
// not compared.
func (r Resources) FitsIn(limit Resources) bool {
	if r.MemoryMB > 0 && r.MemoryMB > limit.MemoryMB {
		return false
	}
	if r.CPUMillis > 0 && r.CPUMillis > limit.CPUMillis {
		return false
	}
	if r.GPUCount > 0 && r.GPUCount > limit.GPUCount {
		return false
	}
	if r.DiskMB > 0 && r.DiskMB > limit.DiskMB {
		return false
	}
	return true
}

// ClassShare is one class's guaranteed floor and its ceiling on this node.
type ClassShare struct {
	Guaranteed float64
	Max        float64
}

// Pressure is what the node measures about itself right now.
//
// This is the input placement cannot have: a reported headroom is seconds old,
// and a node under memory pressure has to refuse before it starts swapping.
//
// Valid and SampledAt exist because "zero pressure" and "no reading" are different
// states and must not decide the same way. A reader whose cgroup files vanished, or a
// sampler that stopped, used to look exactly like an idle machine, so the ceilings it
// feeds were silently unenforced. An invalid sample now refuses when a measured
// ceiling is configured, and withdraws overcommit otherwise: the conservative answer
// is not to guess in the permissive direction.
type Pressure struct {
	CPUUsedPct float64
	MemUsedPct float64
	// SampledAt is when the reading was taken. Zero means never.
	SampledAt time.Time
	// Valid is false when the files could not be read or the sample is too old to
	// describe the machine now.
	Valid bool
}

// Config is the node's envelope and the limits it enforces locally.
type Config struct {
	// Envelope is what this node will admit in total, after utilisation is
	// applied. It is declared rather than detected for devices and disk: a device
	// count cannot be inferred from a host.
	Envelope Resources
	Classes  map[string]ClassShare
	// GPUIndices are the devices this node may hand out. Indices rather than a
	// count, because two sandboxes must never be granted the same one.
	GPUIndices []int32
	// LocalCPUCeiling and LocalMemCeiling are measured pressure above which the
	// node refuses regardless of its envelope, because the envelope is an
	// accounting of reservations and this is the machine's actual state.
	LocalCPUCeiling float64
	LocalMemCeiling float64
	// Overcommit is the largest multiple of the declared envelope the node may
	// grant while measurement says the reservations are overstated.
	//
	// A sandbox that reserved 4 GB and is using 500 MB leaves the envelope full
	// and the machine idle. That gap is the whole reason this knob exists: the
	// envelope accounts for what was promised, and this admits against what is
	// actually being used. One means off, which is the default.
	Overcommit float64
	// ObservedMaxAge is how long a measured reading may serve as a floor on the charge
	// without a fresh sample. Zero takes DefaultObservedMaxAge. It exists because a
	// measurement is knowledge about a moment, and a domain that stopped being readable
	// leaves that moment behind: the pages are still there, so the reading stays as the
	// best estimate, but it is no longer a fact about now.
	ObservedMaxAge time.Duration
	// UtilizationTarget is the measured fraction at which the overcommit is fully
	// withdrawn. Below it the node expands proportionally; at it the node grants
	// exactly its declared envelope and nothing more.
	//
	// It sits below the pressure ceilings by construction: expanding admission up
	// to the point where the node starts refusing would mean the overcommit is
	// handing out capacity the next request is rejected for.
	UtilizationTarget float64
	// CPUOvercommit is how far above the declared CPU envelope the node may promise when
	// nothing is being throttled, and ThrottleTarget is the share of the interval spent
	// throttled at which that expansion is fully withdrawn.
	//
	// They exist because CPU is compressible and memory is not. Over-subscribing memory
	// risks an OOM kill, which destroys a sandbox; over-subscribing CPU risks throttling,
	// which slows a tool call. So the two dimensions get the same mechanism -- a declared
	// base that is expanded by measurement -- but different signals: memory asks how full
	// the machine is, CPU asks how much the kernel is already refusing.
	CPUOvercommit float64
	// ThrottleTarget is the throttled share at which CPUOvercommit has been fully spent.
	ThrottleTarget float64
	// AdmissionReserve is a share of the envelope that no running sandbox may consume, so
	// there is always somewhere for a later request to go.
	//
	// Without it the first wave takes everything and the second wave -- grading, or the
	// next batch -- has nowhere to land until a whole episode finishes. It is headroom
	// reserved for admissions rather than for a class: a guarantee protects one class, and
	// this protects the act of admitting at all.
	AdmissionReserve float64
	// DemandTTL bounds how long a refused request holds its class's guarantee. The
	// guarantee exists so a class that is waiting is not starved by the classes that
	// arrived first; a caller that has gone away is not waiting, and holding its share
	// for longer than this keeps that share out of every other class's reach for nothing.
	// Zero takes DefaultDemandTTL.
	DemandTTL time.Duration
	// LeaseTTL bounds a grant whose owner stops renewing.
	LeaseTTL time.Duration
}

// Validate refuses a configuration the node cannot honour.
func (c Config) Validate() error {
	var guaranteed float64
	for name, share := range c.Classes {
		if share.Guaranteed <= 0 || share.Guaranteed > 1 {
			return fmt.Errorf("node class %q guarantee %.3f must be in (0, 1]", name, share.Guaranteed)
		}
		if share.Max != 0 && share.Max < share.Guaranteed {
			return fmt.Errorf("node class %q ceiling %.3f is below its guarantee %.3f", name, share.Max, share.Guaranteed)
		}
		guaranteed += share.Guaranteed
	}
	if guaranteed > 1+1e-9 {
		return fmt.Errorf(
			"node class guarantees sum to %.3f, which exceeds the envelope. Keep the sum below one so every "+
				"guarantee stays satisfiable without preempting a running sandbox", guaranteed)
	}
	if c.LeaseTTL <= 0 {
		return fmt.Errorf("node lease TTL must be greater than zero")
	}
	for _, ceiling := range []float64{c.LocalCPUCeiling, c.LocalMemCeiling} {
		if ceiling < 0 || ceiling > 1 {
			return fmt.Errorf("node pressure ceilings must be fractions in [0, 1]")
		}
	}
	if c.Overcommit != 0 && c.Overcommit < 1 {
		return fmt.Errorf("overcommit factor %.3f must be at least 1 (zero and one both mean off)", c.Overcommit)
	}
	if c.Overcommit > 1 {
		if c.UtilizationTarget <= 0 || c.UtilizationTarget > 1 {
			return fmt.Errorf(
				"overcommit is %.3f but utilization target %.3f is not in (0, 1]: "+
					"the target is the measured fraction at which overcommit is fully withdrawn",
				c.Overcommit, c.UtilizationTarget)
		}
		// The target must sit below the memory pressure ceiling, or the node can
		// hand out overcommitted capacity that the ceiling then refuses — handing
		// out capacity and immediately refusing the next request that wants it.
		if c.LocalMemCeiling > 0 && c.UtilizationTarget >= c.LocalMemCeiling {
			return fmt.Errorf(
				"utilization target %.3f must be below the memory pressure ceiling %.3f, "+
					"or overcommit allocates capacity the ceiling will then refuse",
				c.UtilizationTarget, c.LocalMemCeiling)
		}
	}
	return nil
}

// Refusal says why a node would not take a request, so the gateway can act.
type Refusal string

const (
	// RefusedClassFull means this class's share is spoken for. Another node may
	// still take it.
	RefusedClassFull Refusal = "class_full"
	// RefusedClassCeiling means the class is at its own ceiling.
	RefusedClassCeiling Refusal = "class_ceiling"
	// RefusedPressure means the machine is under measured load. The envelope would
	// have allowed it; the node knows better.
	RefusedPressure Refusal = "node_pressure"
	// RefusedDevices means no free device index of the kind asked for.
	RefusedDevices Refusal = "no_free_devices"
)

// Grant is an accepted admission and the devices it reserved.
type Grant struct {
	LeaseID    string
	Class      string
	Resources  Resources
	GPUIndices []int32
}

// Admission is this node's accounting. Safe for concurrent use.
//
// It holds aggregates and no per-grant table: the lifecycle owns the record for
// each sandbox, and this only knows how much of the envelope is spoken for. Two
// copies of one fact drift, and that drift is what let a grant be returned while
// its container still held memory.
type Admission struct {
	mu  sync.Mutex
	cfg Config
	// grants is the set of leases this gate has issued and not yet had back, keyed by
	// lease id. It is the only record of what the node has promised.
	//
	// Every charge below is derived from it rather than accumulated beside it. A
	// per-class total stored next to the leases it describes can drift from them, and
	// the drift has no upper bound: one that is raised while a class is busy and
	// lowered while it is quiet keeps the difference forever, and a class that empties
	// is never repriced at all. The node then charges more than the machine has and
	// refuses everything until it is restarted. Deriving the totals makes that
	// unrepresentable -- there is no second copy to disagree.
	grants map[string]Grant
	// queued counts demand per class, so a class's guarantee is reserved only
	// while it is actually waiting. Reserving against an idle class leaves the
	// node idle and full at once.
	queued map[string]int
	// queuedAt is when each class last had demand refused, which is what bounds the
	// claim: a guarantee held for a caller that has gone quiet is capacity kept idle.
	queuedAt map[string]time.Time
	// growth is what each class is expected to grow by before the next sample can correct
	// it: growthPerSandbox for the request being admitted, growthTotal for the ones
	// already running.
	//
	// A sandbox's cost is not what it uses now, it is what it will have used by the time
	// anyone notices -- and the noticing is a sample away. Charging only the present is
	// how a fleet admits a burst at a price that stops being true before the burst has
	// finished starting.
	growthPerSandbox map[string]Resources
	growthTotal      map[string]Resources
	owners           map[string]time.Time
	pressure         Pressure
	// throttlePct is the share of the last interval the sandboxes spent stopped for
	// exceeding their CPU quota, and throttleAt when it was read. It is the feedback term
	// for CPU overcommit: throttling is the kernel saying the packing is too tight, and it
	// is the only signal that distinguishes "the reservations are generous" from "they are
	// wrong".

	// An unusable reading withdraws the expansion rather than reading as zero: a node that
	// cannot measure throttling cannot claim there is none.
	throttlePct float64
	throttleAt  time.Time
	// observed is what the measured domain is actually using, and observedValid says
	// whether that reading can be trusted. The ledger charges reservations, which are an
	// estimate; when the estimate is low the measurement is the truth, so headroom is
	// computed against whichever is larger.
	//
	// observedAt is when the reading was taken, and it is what makes "trusted" a
	// statement about now rather than about a moment in the past. A domain that stopped
	// being readable leaves the last measurement behind, and that measurement is still
	// the best estimate of the pages -- memory does not come back on its own -- but it
	// is not a fact about now, so it stops being a floor once it is old enough that the
	// domain could have changed and we would not know.
	observed      Resources
	observedValid bool
	observedAt    time.Time
	// observedMaxAge is how long a measurement may be used as a floor without a fresh
	// reading. Three missed samples is the same bound the pressure sampler uses to call
	// a reading unusable, so the two dimensions stop agreeing at the same moment rather
	// than one trusting a number the other has already discarded.
	observedMaxAge time.Duration
	// pool is the memory held by warm-pool entries that no caller has claimed yet.
	// It is not a class: an entry exists before any episode does, so charging it to
	// a class would let prefetching starve the work it exists to accelerate. It is
	// subtracted from every class's headroom instead, because it is real memory the
	// node cannot also promise to a caller.
	pool Resources
	// draining reports that a container could not be destroyed and is still
	// charged. It is a condition for an operator to read, not a gate: the charge
	// already keeps its memory out of headroom.
	draining bool
	now      func() time.Time

	admitted int64
	refusals map[Refusal]int64
	nextID   uint64
}

// NewAdmission returns an admission gate over a validated configuration.
func NewAdmission(cfg Config) (*Admission, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	// Devices are declared as indices, so the envelope's device count follows from
	// them rather than being configured twice and able to disagree.
	cfg.Envelope.GPUCount = int32(len(cfg.GPUIndices))
	if cfg.ObservedMaxAge <= 0 {
		cfg.ObservedMaxAge = DefaultObservedMaxAge
	}
	if cfg.DemandTTL <= 0 {
		cfg.DemandTTL = DefaultDemandTTL
	}
	if cfg.CPUOvercommit <= 0 {
		cfg.CPUOvercommit = DefaultCPUOvercommit
	}
	if cfg.ThrottleTarget <= 0 {
		cfg.ThrottleTarget = DefaultThrottleTarget
	}
	if cfg.AdmissionReserve < 0 || cfg.AdmissionReserve >= 1 {
		cfg.AdmissionReserve = DefaultAdmissionReserve
	}
	return &Admission{
		cfg:              cfg,
		grants:           map[string]Grant{},
		queued:           map[string]int{},
		queuedAt:         map[string]time.Time{},
		growthPerSandbox: map[string]Resources{},
		growthTotal:      map[string]Resources{},
		owners:           map[string]time.Time{},
		refusals:         map[Refusal]int64{},
		now:              time.Now,
		observedMaxAge:   cfg.ObservedMaxAge,
	}, nil
}

// DefaultObservedMaxAge is how long a measurement may serve as a floor without a fresh
// reading. It is three sampling intervals at the cadence this deployment runs, so a
// controller that has lost its domain for three samples stops being trusted rather than
// keep charging against a number nobody has confirmed.
const DefaultObservedMaxAge = 15 * time.Second

// DefaultDemandTTL is how long a refused request keeps its class's guarantee reserved.
// It is longer than a placement retry, so a request that bounces between nodes keeps its
// claim, and far shorter than an episode, so a caller that gave up does not hold capacity
// idle for the whole rollout.
const DefaultDemandTTL = 2 * time.Minute

// DefaultCPUOvercommit is how far CPU reservations may over-subscribe when nothing is
// being throttled.
//
// Two, against a memory factor of 1.5 on the same nodes: CPU is the compressible
// dimension, so being wrong costs latency rather than episodes, and the agent workload is
// sparse -- a sandbox declares the cores one tool call may want and uses a fraction of one
// between calls. Measured on this cluster, reservations ran about nine times actual use,
// so a factor of two still leaves the packing well inside reality.
const DefaultCPUOvercommit = 2.0

// DefaultThrottleTarget is the throttled share at which that expansion is fully spent.
//
// Ten percent of the interval, matching the threshold at which a sandbox's measured rate
// is treated as truncated by its quota: any sustained throttling at all is the kernel
// saying the reservations are no longer generous.
const DefaultThrottleTarget = 0.10

// DefaultAdmissionReserve is the share of the envelope kept for admissions. Ten percent
// costs a tenth of the density and buys the guarantee that a later wave -- grading, or the
// next batch -- is never locked out until an episode finishes.
const DefaultAdmissionReserve = 0.10

// SetObservedMaxAge overrides the floor's freshness bound, for a test that drives the
// clock or for a deployment whose sampling cadence is not the default five seconds.
func (a *Admission) SetObservedMaxAge(age time.Duration) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.observedMaxAge = age
}

// demandTTL is how long a refused request may hold its class's guarantee.
func (a *Admission) demandTTL() time.Duration {
	if a.cfg.DemandTTL > 0 {
		return a.cfg.DemandTTL
	}
	return DefaultDemandTTL
}

// SetDemandTTL overrides how long a refused request holds its class's guarantee, for a
// test that drives the clock.
func (a *Admission) SetDemandTTL(ttl time.Duration) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.cfg.DemandTTL = ttl
}

// SetGrowth records what a class is expected to grow by before the next sample.
//
// perSandbox is added to the request being admitted, and total to the classes already
// running. They are separate because they answer different questions -- "will this one
// fit once it has grown" and "has the fleet already promised more than the machine can
// hold once they all have" -- and adding one to the other would either double-charge the
// candidate or let the running set grow for free.
func (a *Admission) SetGrowth(class string, perSandbox, total Resources) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.growthPerSandbox == nil {
		a.growthPerSandbox = map[string]Resources{}
		a.growthTotal = map[string]Resources{}
	}
	a.growthPerSandbox[class] = perSandbox
	a.growthTotal[class] = total
}

// totalGrowthLocked is the growth every class is expected to add before the next sample.
func (a *Admission) totalGrowthLocked() Resources {
	var total Resources
	for _, growth := range a.growthTotal {
		total = total.Add(growth)
	}
	return total
}

// admissionReserve is the share of the envelope kept for admissions.
//
// Zero is a configured choice -- "keep nothing back" -- and not an unset field, so it is
// respected rather than replaced by the default. Only a value outside the range, which a
// node could not honour, falls back.
func (a *Admission) admissionReserve() float64 {
	if a.cfg.AdmissionReserve >= 0 && a.cfg.AdmissionReserve < 1 {
		return a.cfg.AdmissionReserve
	}
	return DefaultAdmissionReserve
}

// GrowthPerSandbox is what one more sandbox of this class is expected to grow by, which
// admission charges on top of the class's price.
func (a *Admission) GrowthPerSandbox(class string) Resources {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.growthPerSandbox[class]
}

// ExpireDemand drops claims whose caller has gone quiet, so a node is not left holding
// a share for a request that will never arrive.
func (a *Admission) ExpireDemand() {
	a.mu.Lock()
	defer a.mu.Unlock()
	for name, seen := range a.queuedAt {
		if a.now().Sub(seen) > a.demandTTL() {
			delete(a.queued, name)
			delete(a.queuedAt, name)
		}
	}
}

// observedUsableLocked reports whether the measurement may still be believed.
//
// A node that has never been given a reading is not a node with a stale one: the elastic
// controller may be off, or it may not have ticked yet, and in that case overcommit has
// always been pressure-driven. Requiring a measurement there would withdraw overcommit
// from every deployment that never enabled the controller, which is a regression dressed
// as a safety check. So the gate only applies once something has reported.
func (a *Admission) observedUsableLocked() bool {
	if a.observedAt.IsZero() {
		return true
	}
	if !a.observedValid {
		return false
	}
	return a.now().Sub(a.observedAt) <= a.observedMaxAge
}

// SetClock replaces the gate's clock, for a test that drives time directly.
func (a *Admission) SetClock(now func() time.Time) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.now = now
}

// SetObserved records what the measured domain is actually using.
//
// It is a floor on the charge, not a replacement for it: reservations say what the node
// has promised, and a promise the measurement has already exceeded is no longer the bound
// that matters. An invalid reading leaves the ledger on its reservations, which is the
// conservative direction -- it cannot make the node admit more.
func (a *Admission) SetObserved(used Resources, valid bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.observed = used
	a.observedValid = valid
	if valid {
		a.observedAt = a.now()
	}
}

// chargesLocked is what each class is holding, summed from the leases that are still
// open. Callers hold the lock.
//
// It is a sum over the grant set rather than a stored total. The cost is one pass over
// at most a few hundred leases, on a path that runs once per admission.
func (a *Admission) chargesLocked() map[string]Resources {
	charges := make(map[string]Resources, len(a.cfg.Classes))
	for _, grant := range a.grants {
		charges[grant.Class] = charges[grant.Class].Add(grant.Resources)
	}
	return charges
}

// Held returns the leases this gate has issued and not had back.
//
// It is what lets an operator, or a test, compare the node's charge against the
// sandboxes that are actually supposed to account for it.
func (a *Admission) Held() []Grant {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]Grant, 0, len(a.grants))
	for _, grant := range a.grants {
		out = append(out, grant)
	}
	return out
}

// SetThrottle records the share of the last interval the sandboxes spent throttled.
//
// Zero is a measurement, not an absence of one; `throttleAt` is what says the reading is
// current. A node that has stopped reading the domain must fall back to its declared CPU
// envelope, because the claim that CPU reservations overstate demand is exactly the claim
// a stale or missing measurement cannot support.
func (a *Admission) SetThrottle(pct float64) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.throttlePct = pct
	a.throttleAt = time.Now()
}

// SetPressure records what the node measures about itself.
func (a *Admission) SetPressure(p Pressure) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.pressure = p
}

// MemoryPressure reports the measured memory fraction and whether it is usable.
//
// A warm pool asks this before creating another entry: prefetching is an
// optimization, and one that keeps building containers into a node already at its
// measured ceiling is an optimization that causes the failures it exists to avoid.
func (a *Admission) MemoryPressure() (float64, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.pressure.MemUsedPct, a.pressure.Valid
}

// AcquirePool reserves one warm-pool entry's footprint.
//
// It is checked against the *declared* envelope rather than the overcommitted one: an
// entry is memory that exists before any caller is charged, so measurement must not
// expand what the pool may hold. Overcommit is about reservations overstating use;
// a pool entry has no use to overstate.
func (a *Admission) AcquirePool(want Resources) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	free := a.cfg.Envelope.Sub(totalOf(a.chargesLocked())).Sub(a.pool)
	if !want.FitsIn(free) {
		return false
	}
	a.pool = a.pool.Add(want)
	return true
}

// ReleasePool returns a pool entry's footprint once it is claimed or destroyed.
func (a *Admission) ReleasePool(want Resources) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.pool = a.pool.Sub(want)
	if a.pool.MemoryMB < 0 || a.pool.CPUMillis < 0 || a.pool.DiskMB < 0 {
		// A release without a matching acquire would let the node promise the same
		// memory twice, so it is corrected rather than carried.
		a.pool = Resources{}
	}
}

// AsPoolBudget adapts this gate to the interface a prefetching backend takes.
//
// The two packages speak different units for CPU -- the backend states fractional
// cores and the node charges milliseconds -- so the conversion lives here rather than in
// either of them.
func (a *Admission) AsPoolBudget() backend.PoolBudget { return poolBudget{gate: a} }

type poolBudget struct{ gate *Admission }

func (p poolBudget) AcquirePool(want backend.Resources) bool {
	return p.gate.AcquirePool(nodeResourcesFrom(want))
}

func (p poolBudget) ReleasePool(want backend.Resources) { p.gate.ReleasePool(nodeResourcesFrom(want)) }

func (p poolBudget) MemoryPressure() (float64, bool) { return p.gate.MemoryPressure() }

// Pooled reports what the warm pool currently holds.
func (a *Admission) Pooled() Resources {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.pool
}

// SetDraining marks the node as refusing work because it holds containers it
// could not destroy. Their memory is still in use, so admitting against it would
// over-commit the host.
func (a *Admission) SetDraining(draining bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.draining = draining
}

// Draining reports whether the node is refusing new work.
func (a *Admission) Draining() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.draining
}

// Enqueue records demand that could not be admitted, so this class's guarantee
// is held back while it waits.
//
// It is the only thing that makes a guarantee real at this node. A guarantee says a
// class can always have its share, but the share is only worth something if it is held
// back from everyone else *while the class is waiting*, and the node is the only place
// that can see a class waiting: the control plane sees refusals from whichever node it
// happened to ask, and it does not own the pages. A node that never records refused
// demand has guarantees that are never exercised, and the first class to arrive takes
// the whole envelope -- which at this deployment showed up as grading sandboxes that
// could never be admitted because rollout had already taken the machine.
//
// The demand is stamped, because a refusal is not a reservation forever: a caller that
// went away must not hold a share idle, so the claim lapses (see Config.DemandTTL).
func (a *Admission) Enqueue(class string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.queued[class]++
	a.queuedAt[class] = a.now()
}

// Dequeue withdraws queued demand whose caller gave up or was admitted.
func (a *Admission) Dequeue(class string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.queued[class] > 0 {
		a.queued[class]--
		if a.queued[class] == 0 {
			delete(a.queued, class)
			delete(a.queuedAt, class)
		}
	}
}

// Reserve charges one request against this node's live state.
//
// The caller owns the grant it returns. Budget comes back only through Return,
// which the lifecycle calls once it has confirmed the runtime object is gone.
//
// A container the node failed to destroy is not refused here: its grant was never
// returned, so it is still charged to its class and the headroom below already
// excludes the memory it holds. Refusing the whole node on top of that would cost
// the fleet a node per stuck container and buy nothing.
func (a *Admission) Reserve(class, owner string, want Resources) (Grant, Refusal, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()

	// Measured pressure first: the envelope is an accounting of reservations, and
	// a machine already swapping must refuse even when its accounting says yes.
	//
	// An unusable sample refuses here too, when a ceiling is configured. The
	// alternative is what this used to do -- read zero and admit -- so a node whose
	// cgroup files vanished would accept work until the host filled, which is the
	// one direction a pressure check must never fail in.
	if measured := a.cfg.LocalCPUCeiling > 0 || a.cfg.LocalMemCeiling > 0; measured && !a.pressure.Valid {
		a.refuse(RefusedPressure)
		return Grant{}, RefusedPressure, false
	}
	if a.cfg.LocalCPUCeiling > 0 && a.pressure.CPUUsedPct > a.cfg.LocalCPUCeiling {
		a.refuse(RefusedPressure)
		return Grant{}, RefusedPressure, false
	}
	if a.cfg.LocalMemCeiling > 0 && a.pressure.MemUsedPct > a.cfg.LocalMemCeiling {
		a.refuse(RefusedPressure)
		return Grant{}, RefusedPressure, false
	}
	charges := a.chargesLocked()
	if share, declared := a.cfg.Classes[class]; declared && share.Max > 0 {
		if !want.FitsIn(a.cfg.Envelope.Scale(share.Max).Sub(charges[class])) {
			a.refuse(RefusedClassCeiling)
			return Grant{}, RefusedClassCeiling, false
		}
	}
	// Devices before the class share: an exhausted device pool is a more specific
	// answer than a full class, and reporting it as "class full" would suggest the
	// request will fit once other work drains, which it will not.
	if int(want.GPUCount) > len(a.freeGPUsLocked()) {
		a.refuse(RefusedDevices)
		return Grant{}, RefusedDevices, false
	}
	if !want.FitsIn(a.headroomForLocked(class, charges)) {
		a.refuse(RefusedClassFull)
		return Grant{}, RefusedClassFull, false
	}

	free := a.freeGPUsLocked()
	devices := append([]int32(nil), free[:want.GPUCount]...)
	a.nextID++
	grant := Grant{
		LeaseID: fmt.Sprintf("lease-%d", a.nextID), Class: class,
		Resources: want, GPUIndices: devices,
	}
	a.grants[grant.LeaseID] = grant
	a.owners[owner] = a.now()
	a.admitted++
	return grant, "", true
}

func (a *Admission) refuse(reason Refusal) { a.refusals[reason]++ }

// effectiveEnvelopeLocked is what the node will admit against right now.
//
// Without overcommit this is the declared envelope, and the whole function is a
// single return. With overcommit it is the declared envelope scaled by how far
// measured usage sits below the target, which is the point of the feature: the
// envelope accounts for what sandboxes reserved, and a fleet of sandboxes that
// each reserved four times what they use leaves a node simultaneously full and
// idle.
//
// The scale is withdrawn proportionally rather than switched off at a threshold.
// A step would make the node admit at the full multiple right up to the target
// and then refuse everything, so a node sitting near the target would oscillate
// between over-admitting and refusing as each sandbox started and settled. A
// proportional withdrawal converges instead: every admission raises measured
// usage, which lowers the next admission's headroom.
//
// Memory is the only dimension that expands. CPU is compressible — a cgroup CPU
// limit throttles and the workload runs slower — so overcommitting it trades
// latency for density and is a reasonable thing to do by default. Memory is not:
// exceeding it is an OOM kill, which destroys a sandbox and loses an episode.
// The asymmetry means CPU is already effectively overcommitted by the scheduler
// and memory is the dimension where measurement buys real density. Devices and
// disk never expand: a GPU cannot be shared by two sandboxes that each believe
// they own it, and disk is not reclaimed by the kernel under pressure.
func (a *Admission) effectiveEnvelopeLocked() Resources {
	envelope := a.cfg.Envelope
	if a.cfg.Overcommit <= 1 || a.cfg.UtilizationTarget <= 0 {
		return envelope
	}
	// An unusable sample withdraws the expansion rather than reading as idle: the
	// node then admits exactly its declared envelope, which is the behaviour it had
	// before overcommit existed. Unusable now covers the measurement as well as the
	// pressure: overcommit is the claim that reservations overstate use, and a node
	// that has not read its sandbox domain for a while cannot make that claim. Both
	// signals therefore have to be fresh for the envelope to expand.
	if !a.observedUsableLocked() {
		return envelope
	}
	// A *valid* zero is different and is not special-cased
	// -- it means the measured domain is empty, which is the case the feature is for.
	if !a.pressure.Valid {
		return envelope
	}
	// headroomFraction is how far below the target measurement sits: 1.0 when the
	// node is empty, 0.0 at the target and above.
	headroomFraction := (a.cfg.UtilizationTarget - a.pressure.MemUsedPct) / a.cfg.UtilizationTarget
	if headroomFraction <= 0 {
		return envelope
	}
	if headroomFraction > 1 {
		headroomFraction = 1
	}
	scale := 1 + (a.cfg.Overcommit-1)*headroomFraction
	envelope.MemoryMB = int64(float64(envelope.MemoryMB) * scale)
	envelope.CPUMillis = int64(float64(envelope.CPUMillis) * a.cpuScaleLocked())
	return envelope
}

// cpuScaleLocked is how far the CPU envelope may be expanded right now.
//
// It mirrors the memory term exactly, with throttling in the role that utilization plays
// there: memory is withdrawn as the machine fills, CPU is withdrawn as the kernel starts
// refusing work. Both are proportional rather than a step, so the node converges instead
// of oscillating between over-subscribing and refusing on alternate samples.
//
// Absent, stale or unusable measurements withdraw the expansion completely.
func (a *Admission) cpuScaleLocked() float64 {
	if a.cfg.CPUOvercommit <= 1 || a.cfg.ThrottleTarget <= 0 {
		return 1
	}
	if a.throttleAt.IsZero() || !a.observedUsableLocked() {
		return 1
	}
	if time.Since(a.throttleAt) > a.throttleMaxAge() {
		return 1
	}
	headroom := (a.cfg.ThrottleTarget - a.throttlePct) / a.cfg.ThrottleTarget
	if headroom <= 0 {
		return 1
	}
	if headroom > 1 {
		headroom = 1
	}
	return 1 + (a.cfg.CPUOvercommit-1)*headroom
}

func (a *Admission) throttleMaxAge() time.Duration {
	if a.cfg.ObservedMaxAge > 0 {
		return a.cfg.ObservedMaxAge
	}
	return DefaultObservedMaxAge
}

// Overcommitted reports the memory multiple the node is currently admitting at.
//
// Exposed because an operator tuning the factor has to see what it resolved to:
// the configured maximum is an upper bound, and the number that matters is what
// measurement allowed at the moment a request was refused.
func (a *Admission) Overcommitted() float64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	declared := a.cfg.Envelope.MemoryMB
	if declared <= 0 {
		return 1
	}
	return float64(a.effectiveEnvelopeLocked().MemoryMB) / float64(declared)
}

// Headroom returns what one class could be granted right now.
//
// This is the number placement must compare against, and it is not the envelope
// remainder: a borrower may not take another class's unmet guarantee.
func (a *Admission) Headroom(class string) Resources {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.headroomForLocked(class, a.chargesLocked())
}

// headroomForLocked answers one class's question against charges the caller has already
// summed, so a report that walks every class sums the leases once rather than once per
// class.
func (a *Admission) headroomForLocked(class string, charges map[string]Resources) Resources {
	effective := a.effectiveEnvelopeLocked()
	// The charge is the larger of what was promised and what is measured: reservations
	// that the measurement has outgrown are no longer what bounds the node.
	bound := totalOf(charges)
	// A stale measurement is still a floor, and the reason is the direction of the
	// unknown. Sandboxes released since the last reading would make the true usage
	// lower, but sandboxes that kept growing would make it higher, and of the two only
	// the second can cost an episode. So the last reading stays as the floor -- the node
	// never assumes usage went down because it cannot confirm it -- and what a stale
	// reading costs is the expansion: see effectiveEnvelopeLocked, which withdraws
	// overcommit until the domain is readable again.
	if a.observedValid && a.observed.MemoryMB > bound.MemoryMB {
		bound.MemoryMB = a.observed.MemoryMB
	}
	if a.observedValid && a.observed.CPUMillis > bound.CPUMillis {
		bound.CPUMillis = a.observed.CPUMillis
	}
	// Growth comes next, and it is charged against every class rather than the one
	// asking: the sandboxes that are already running are the ones about to grow, and
	// their growth consumes the same pages and the same cores as a new sandbox would.
	bound = bound.Add(a.totalGrowthLocked())
	// What is left is then reduced by the admission reserve, which no class may spend.
	// This is the capacity a later wave lands in; a node that let the first wave take
	// all of it has to wait for an episode to end before it can admit anything at all.
	free := effective.Sub(bound).Sub(a.pool).Sub(a.reservedForOthersLocked(class, charges)).
		Sub(a.cfg.Envelope.Scale(a.admissionReserve()))
	share, declared := a.cfg.Classes[class]
	if !declared || share.Max == 0 {
		return free
	}
	// Class ceilings are expressed as a fraction of the declared envelope, not the
	// effective one: a ceiling of 0.5 means "half the machine", which is a
	// configuration the operator stated about the machine, not about the current
	// overcommit factor.
	underCeiling := a.cfg.Envelope.Scale(share.Max).Sub(charges[class])
	return Resources{
		MemoryMB:  minI64(free.MemoryMB, underCeiling.MemoryMB),
		CPUMillis: minI64(free.CPUMillis, underCeiling.CPUMillis),
		GPUCount:  minI32(free.GPUCount, underCeiling.GPUCount),
		DiskMB:    minI64(free.DiskMB, underCeiling.DiskMB),
	}
}

// ClassHeadroom returns every declared class's headroom, for the node's report.
//
// Reported per class rather than as one number because that is the granularity
// admission works at: a scheduler given the remainder sends work this node will
// refuse, and the mismatch shows up only as sandboxes that never start.
func (a *Admission) ClassHeadroom() map[string]Resources {
	a.mu.Lock()
	defer a.mu.Unlock()
	names := make([]string, 0, len(a.cfg.Classes)+1)
	for name := range a.cfg.Classes {
		names = append(names, name)
	}
	if _, declared := a.cfg.Classes["default"]; !declared {
		// A request carries a class either way, and one whose class is absent from
		// the report would be compared against nothing.
		names = append(names, "default")
	}
	sort.Strings(names)
	charges := a.chargesLocked()
	out := make(map[string]Resources, len(names))
	for _, name := range names {
		out[name] = a.headroomForLocked(name, charges)
	}
	return out
}

func (a *Admission) reservedForOthersLocked(class string, charges map[string]Resources) Resources {
	var reserved Resources
	for name, share := range a.cfg.Classes {
		if name == class || a.queued[name] == 0 {
			continue
		}
		if seen, ok := a.queuedAt[name]; ok && a.now().Sub(seen) > a.demandTTL() {
			// The demand lapsed. The share goes back to the pool rather than sitting idle
			// for a caller that is no longer asking.
			continue
		}
		reserved = reserved.Add(a.cfg.Envelope.Scale(share.Guaranteed).Sub(charges[name]))
	}
	return reserved
}

// totalOf sums every class's charge. The whole node's obligation is the sum of its
// leases, so this is Granted.
func totalOf(charges map[string]Resources) Resources {
	var total Resources
	for _, amount := range charges {
		total = total.Add(amount)
	}
	return total
}

// freeGPUsLocked is the devices no open lease holds. Derived from the grant set for the
// same reason the charges are: a free list kept beside the leases can be handed back
// twice, or not at all.
func (a *Admission) freeGPUsLocked() []int32 {
	if len(a.cfg.GPUIndices) == 0 {
		return nil
	}
	held := make(map[int32]struct{}, len(a.cfg.GPUIndices))
	for _, grant := range a.grants {
		for _, index := range grant.GPUIndices {
			held[index] = struct{}{}
		}
	}
	free := make([]int32, 0, len(a.cfg.GPUIndices))
	for _, index := range a.cfg.GPUIndices {
		if _, taken := held[index]; !taken {
			free = append(free, index)
		}
	}
	return free
}

// Return gives one grant's budget back, once its runtime object is gone.
//
// It is keyed on the lease rather than on the amount the caller happens to be holding.
// A caller's copy is a snapshot of what the grant cost when it was handed out; the gate's
// record is what the node is actually charging. Returning the snapshot is how a charge
// that moved afterwards leaves a residue that never comes back.
//
// A lease the gate does not hold is returned as a no-op rather than as a subtraction:
// it was already given back, or never issued here, and subtracting again would leave the
// node believing it holds less than it does.
func (a *Admission) Return(grant Grant) {
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.grants, grant.LeaseID)
}

// Recharge charges a grant this process did not reserve, which is what a node
// does for a container that outlived a restart.
//
// It deliberately skips the headroom check: the container exists, so its memory
// is already spent whether the envelope has room or not. A device index that is
// not free means two records claim one device, which is a fault worth refusing.
func (a *Admission) Recharge(grant Grant, owner string) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if missing := a.claimedIndicesLocked(grant.GPUIndices); len(missing) > 0 {
		return fmt.Errorf("device indices %v are already granted", missing)
	}
	a.grants[grant.LeaseID] = grant
	a.owners[owner] = a.now()
	a.admitted++
	return nil
}

// RetainOwners forgets the heartbeats of owners that no longer hold anything, so
// a long run does not accumulate one entry per caller that ever existed.
func (a *Admission) RetainOwners(live map[string]struct{}) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for owner := range a.owners {
		if _, held := live[owner]; !held {
			delete(a.owners, owner)
		}
	}
}

// claimedIndicesLocked reports which of these devices an open lease already holds.
//
// It takes the whole set rather than claiming them one at a time because a partial claim
// is worse than a refused one: two leases that each hold an index the other believes it
// owns is a fault no later check can detect.
func (a *Admission) claimedIndicesLocked(indices []int32) []int32 {
	if len(indices) == 0 {
		return nil
	}
	held := make(map[int32]struct{}, len(a.cfg.GPUIndices))
	for _, grant := range a.grants {
		for _, index := range grant.GPUIndices {
			held[index] = struct{}{}
		}
	}
	var missing []int32
	for _, index := range indices {
		if _, taken := held[index]; taken {
			missing = append(missing, index)
		}
	}
	return missing
}

// Renew records that an owner still holds its grants.
// RegisterOwner records a caller that now holds a record on this node.
//
// Admit already records the owner, but it and Reserve are separate locks. A
// sweep that runs between them sees no allocation, forgets the owner, and then
// ignores every renewal for it; the sandbox is reclaimed as an orphan even
// though its owner is alive. Reserve calls this after the record is visible, so
// the owner is never lost in that window. Unlike Renew, this is allowed to
// create an entry: the caller has a record here, so its liveness matters.
func (a *Admission) RegisterOwner(owner string) {
	if owner == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.owners[owner] = a.now()
}

// Renew records a heartbeat from an owner this node already knows.
func (a *Admission) Renew(owner string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if _, known := a.owners[owner]; known {
		a.owners[owner] = a.now()
	}
}

// OwnerAlive reports whether an owner is known here and has renewed within the
// lease TTL. An owner this node never admitted is not its business, so it answers
// false only for an owner it has forgotten or has not heard from in time. The
// check is against the entry's freshness rather than the sweep's bookkeeping, so
// an owner that has gone silent reads as gone even before the next sweep expires
// its grants.
func (a *Admission) OwnerAlive(owner string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	seen, known := a.owners[owner]
	if !known {
		return false
	}
	return a.now().Sub(seen) <= a.cfg.LeaseTTL
}

// Report is the gate's accounting for the node's own report and metric hook.
type Report struct {
	Envelope      Resources
	ClassHeadroom map[string]Resources
	Granted       Resources
	FreeGPUs      int
	Draining      bool
	Admitted      int64
	Refusals      map[Refusal]int64
	// EffectiveEnvelope is what the node is admitting against now. It differs from
	// Envelope only under overcommit, and the difference is the whole measurement:
	// an operator tuning the factor needs the resolved number, not the configured
	// bound.
	EffectiveEnvelope Resources
	// Overcommit is EffectiveEnvelope's memory over Envelope's. One means the node
	// is admitting at its declared envelope, either because overcommit is off or
	// because measurement has withdrawn all of it.
	Overcommit float64
	// Observed is what the measured domain is using, and ObservedValid says whether the
	// reading can be trusted. A report that omitted them would hide the number admission
	// is bounded by when it exceeds the reservations.
	Observed      Resources
	ObservedValid bool
	// ObservedFresh says the reading is recent enough to still be used as a floor. Valid
	// but stale means the domain stopped being readable and admission has gone back to
	// charging reservations alone, which is the state an operator has to be able to see:
	// without it an unreadable domain looks exactly like an idle one.
	ObservedFresh bool
	// Pooled is the memory held by warm-pool entries that no caller has claimed.
	// It is inside Granted's envelope but outside every class's charge, which is
	// why a report that omitted it would show headroom the node does not have.
	Pooled Resources
}

// Snapshot returns the gate's state. The planes report and never log.
func (a *Admission) Snapshot() Report {
	a.mu.Lock()
	defer a.mu.Unlock()
	refusals := make(map[Refusal]int64, len(a.refusals))
	for reason, count := range a.refusals {
		refusals[reason] = count
	}
	names := make([]string, 0, len(a.cfg.Classes))
	for name := range a.cfg.Classes {
		names = append(names, name)
	}
	charges := a.chargesLocked()
	headroom := make(map[string]Resources, len(names))
	for _, name := range names {
		headroom[name] = a.headroomForLocked(name, charges)
	}
	effective := a.effectiveEnvelopeLocked()
	overcommit := 1.0
	if a.cfg.Envelope.MemoryMB > 0 {
		overcommit = float64(effective.MemoryMB) / float64(a.cfg.Envelope.MemoryMB)
	}
	return Report{
		Envelope:          a.cfg.Envelope,
		EffectiveEnvelope: effective,
		Overcommit:        overcommit,
		ClassHeadroom:     headroom,
		Granted:           totalOf(charges),
		Observed:          a.observed,
		ObservedValid:     a.observedValid,
		ObservedFresh:     a.observedUsableLocked(),
		Pooled:            a.pool,
		FreeGPUs:          len(a.freeGPUsLocked()),
		Draining:          a.draining,
		Admitted:          a.admitted,
		Refusals:          refusals,
	}
}

func maxI64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func minI64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func maxI32(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}

func minI32(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}
