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
type Pressure struct {
	CPUUsedPct float64
	MemUsedPct float64
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
	// RefusedDraining means the node still holds containers it could not destroy,
	// so its returned memory is not actually free.
	RefusedDraining Refusal = "draining"
)

// Grant is an accepted admission and the devices it reserved.
type Grant struct {
	LeaseID    string
	Class      string
	Resources  Resources
	GPUIndices []int32
}

type lease struct {
	grant     Grant
	owner     string
	renewedAt time.Time
}

// Admission is this node's accounting. Safe for concurrent use.
type Admission struct {
	mu   sync.Mutex
	cfg  Config
	used map[string]Resources
	// queued counts demand per class, so a class's guarantee is reserved only
	// while it is actually waiting. Reserving against an idle class leaves the
	// node idle and full at once.
	queued   map[string]int
	leases   map[string]*lease
	owners   map[string]time.Time
	freeGPUs []int32
	pressure Pressure
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
	free := make([]int32, len(cfg.GPUIndices))
	copy(free, cfg.GPUIndices)
	// Devices are declared as indices, so the envelope's device count follows from
	// them rather than being configured twice and able to disagree.
	cfg.Envelope.GPUCount = int32(len(cfg.GPUIndices))
	return &Admission{
		cfg:      cfg,
		used:     map[string]Resources{},
		queued:   map[string]int{},
		leases:   map[string]*lease{},
		owners:   map[string]time.Time{},
		freeGPUs: free,
		refusals: map[Refusal]int64{},
		now:      time.Now,
	}, nil
}

// SetClock replaces the gate's clock, for a test that drives time directly.
func (a *Admission) SetClock(now func() time.Time) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.now = now
}

// SetPressure records what the node measures about itself.
func (a *Admission) SetPressure(p Pressure) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.pressure = p
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
func (a *Admission) Enqueue(class string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.queued[class]++
}

// Dequeue withdraws queued demand whose caller gave up or was admitted.
func (a *Admission) Dequeue(class string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.queued[class] > 0 {
		a.queued[class]--
		if a.queued[class] == 0 {
			delete(a.queued, class)
		}
	}
}

// Admit accepts or refuses one request against this node's live state.
func (a *Admission) Admit(class, owner string, want Resources) (Grant, Refusal, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.expireLocked()

	if a.draining {
		a.refuse(RefusedDraining)
		return Grant{}, RefusedDraining, false
	}
	// Measured pressure first: the envelope is an accounting of reservations, and
	// a machine already swapping must refuse even when its accounting says yes.
	if a.cfg.LocalCPUCeiling > 0 && a.pressure.CPUUsedPct > a.cfg.LocalCPUCeiling {
		a.refuse(RefusedPressure)
		return Grant{}, RefusedPressure, false
	}
	if a.cfg.LocalMemCeiling > 0 && a.pressure.MemUsedPct > a.cfg.LocalMemCeiling {
		a.refuse(RefusedPressure)
		return Grant{}, RefusedPressure, false
	}
	if share, declared := a.cfg.Classes[class]; declared && share.Max > 0 {
		if !want.FitsIn(a.cfg.Envelope.Scale(share.Max).Sub(a.used[class])) {
			a.refuse(RefusedClassCeiling)
			return Grant{}, RefusedClassCeiling, false
		}
	}
	// Devices before the class share: an exhausted device pool is a more specific
	// answer than a full class, and reporting it as "class full" would suggest the
	// request will fit once other work drains, which it will not.
	if int(want.GPUCount) > len(a.freeGPUs) {
		a.refuse(RefusedDevices)
		return Grant{}, RefusedDevices, false
	}
	if !want.FitsIn(a.headroomLocked(class)) {
		a.refuse(RefusedClassFull)
		return Grant{}, RefusedClassFull, false
	}

	devices := a.takeGPUsLocked(want.GPUCount)
	a.nextID++
	id := fmt.Sprintf("lease-%d", a.nextID)
	grant := Grant{LeaseID: id, Class: class, Resources: want, GPUIndices: devices}
	now := a.now()
	a.leases[id] = &lease{grant: grant, owner: owner, renewedAt: now}
	a.used[class] = a.used[class].Add(want)
	a.owners[owner] = now
	a.admitted++
	return grant, "", true
}

func (a *Admission) refuse(reason Refusal) { a.refusals[reason]++ }

// Headroom returns what one class could be granted right now.
//
// This is the number placement must compare against, and it is not the envelope
// remainder: a borrower may not take another class's unmet guarantee.
func (a *Admission) Headroom(class string) Resources {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.headroomLocked(class)
}

func (a *Admission) headroomLocked(class string) Resources {
	free := a.cfg.Envelope.Sub(a.totalUsedLocked()).Sub(a.reservedForOthersLocked(class))
	share, declared := a.cfg.Classes[class]
	if !declared || share.Max == 0 {
		return free
	}
	underCeiling := a.cfg.Envelope.Scale(share.Max).Sub(a.used[class])
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
	out := make(map[string]Resources, len(names))
	for _, name := range names {
		out[name] = a.headroomLocked(name)
	}
	return out
}

func (a *Admission) reservedForOthersLocked(class string) Resources {
	var reserved Resources
	for name, share := range a.cfg.Classes {
		if name == class || a.queued[name] == 0 {
			continue
		}
		reserved = reserved.Add(a.cfg.Envelope.Scale(share.Guaranteed).Sub(a.used[name]))
	}
	return reserved
}

func (a *Admission) totalUsedLocked() Resources {
	var total Resources
	for _, amount := range a.used {
		total = total.Add(amount)
	}
	return total
}

func (a *Admission) takeGPUsLocked(count int32) []int32 {
	if count <= 0 {
		return nil
	}
	taken := make([]int32, count)
	copy(taken, a.freeGPUs[:count])
	a.freeGPUs = a.freeGPUs[count:]
	return taken
}

// Release returns one grant and the devices it held.
func (a *Admission) Release(leaseID string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.releaseLocked(leaseID)
}

func (a *Admission) releaseLocked(leaseID string) {
	held, ok := a.leases[leaseID]
	if !ok {
		return
	}
	delete(a.leases, leaseID)
	a.used[held.grant.Class] = a.used[held.grant.Class].Sub(held.grant.Resources)
	if (a.used[held.grant.Class] == Resources{}) {
		delete(a.used, held.grant.Class)
	}
	a.freeGPUs = append(a.freeGPUs, held.grant.GPUIndices...)
	sort.Slice(a.freeGPUs, func(i, j int) bool { return a.freeGPUs[i] < a.freeGPUs[j] })
}

// Renew records that an owner still holds its grants.
func (a *Admission) Renew(owner string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if _, known := a.owners[owner]; known {
		a.owners[owner] = a.now()
	}
}

// ReleaseOwner returns everything one owner held.
func (a *Admission) ReleaseOwner(owner string) int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.releaseOwnerLocked(owner)
}

func (a *Admission) releaseOwnerLocked(owner string) int {
	var ids []string
	for id, held := range a.leases {
		if held.owner == owner {
			ids = append(ids, id)
		}
	}
	for _, id := range ids {
		a.releaseLocked(id)
	}
	delete(a.owners, owner)
	return len(ids)
}

// expireLocked reclaims the grants of owners that stopped renewing.
//
// Owner silence is the only evidence the node has. Grant age is not: a
// long-lived sandbox is indistinguishable from an abandoned one by age alone.
func (a *Admission) expireLocked() {
	now := a.now()
	for owner, seen := range a.owners {
		if now.Sub(seen) > a.cfg.LeaseTTL {
			a.releaseOwnerLocked(owner)
		}
	}
}

// Expire runs lease reclamation, for a caller driving the gate from a sweep.
func (a *Admission) Expire() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.expireLocked()
}

// Report is the gate's accounting for the node's own report and metric hook.
type Report struct {
	Envelope      Resources
	ClassHeadroom map[string]Resources
	Granted       Resources
	Leases        int
	FreeGPUs      int
	Draining      bool
	Admitted      int64
	Refusals      map[Refusal]int64
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
	headroom := make(map[string]Resources, len(names))
	for _, name := range names {
		headroom[name] = a.headroomLocked(name)
	}
	return Report{
		Envelope:      a.cfg.Envelope,
		ClassHeadroom: headroom,
		Granted:       a.totalUsedLocked(),
		Leases:        len(a.leases),
		FreeGPUs:      len(a.freeGPUs),
		Draining:      a.draining,
		Admitted:      a.admitted,
		Refusals:      refusals,
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
