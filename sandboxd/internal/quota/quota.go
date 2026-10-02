// Package quota is the cross-backend admission ledger.
//
// It answers one question that no single backend can: of everything this fleet
// is allowed to run, how much may this class of work hold right now. A backend
// bounds its own nodes; nothing below this bounds a rollout against a grader.
//
// Three rules shape it, and each one exists because its absence produced a
// specific failure.
//
//   - A guarantee is reserved only against *queued* demand. Reserving a class's
//     share while it has nothing waiting leaves the fleet idle and full at the
//     same time.
//   - A borrower may be bypassed, but only a bounded number of times. Strict FIFO
//     across classes lets one large request block a queue of small ones; no bound
//     at all starves the large request forever.
//   - A grant is held until it is returned or its owner's lease expires. Age is
//     not evidence that a sandbox is gone, so nothing here reclaims on age.
package quota

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

// Amount is a resource vector. Every dimension is an integer, because admission
// must be exact: a float would let repeated grant and release drift until the
// ledger disagrees with the fleet.
type Amount struct {
	MemoryMB  int64
	CPUMillis int64
	GPUCount  int32
	DiskMB    int64
	// Sandboxes is the fallback dimension for a backend that reports no bytes.
	// A managed provider sells concurrency, so that is the only quantity it can
	// be held to.
	Sandboxes int64
}

// Add returns the sum of two vectors.
func (a Amount) Add(b Amount) Amount {
	return Amount{
		MemoryMB:  a.MemoryMB + b.MemoryMB,
		CPUMillis: a.CPUMillis + b.CPUMillis,
		GPUCount:  a.GPUCount + b.GPUCount,
		DiskMB:    a.DiskMB + b.DiskMB,
		Sandboxes: a.Sandboxes + b.Sandboxes,
	}
}

// Sub returns the difference, clamped at zero in every dimension.
//
// Clamped because a release that arrives twice, or one that outlives a rebuilt
// ledger, must not drive a dimension negative and silently inflate the envelope.
func (a Amount) Sub(b Amount) Amount {
	return Amount{
		MemoryMB:  max64(a.MemoryMB-b.MemoryMB, 0),
		CPUMillis: max64(a.CPUMillis-b.CPUMillis, 0),
		GPUCount:  max32(a.GPUCount-b.GPUCount, 0),
		DiskMB:    max64(a.DiskMB-b.DiskMB, 0),
		Sandboxes: max64(a.Sandboxes-b.Sandboxes, 0),
	}
}

// FitsIn reports whether this vector is covered by another in every dimension.
//
// A zero request in a dimension always fits: zero means the caller stated no
// requirement there, which is not the same as requesting nothing and must not be
// compared against a limit.
func (a Amount) FitsIn(limit Amount) bool {
	return a.MemoryMB <= limit.MemoryMB &&
		a.CPUMillis <= limit.CPUMillis &&
		a.GPUCount <= limit.GPUCount &&
		a.DiskMB <= limit.DiskMB &&
		a.Sandboxes <= limit.Sandboxes
}

// Scale returns this vector multiplied by a fraction, rounded down.
func (a Amount) Scale(f float64) Amount {
	return Amount{
		MemoryMB:  int64(float64(a.MemoryMB) * f),
		CPUMillis: int64(float64(a.CPUMillis) * f),
		GPUCount:  int32(float64(a.GPUCount) * f),
		DiskMB:    int64(float64(a.DiskMB) * f),
		Sandboxes: int64(float64(a.Sandboxes) * f),
	}
}

// IsZero reports whether a vector requests nothing at all.
func (a Amount) IsZero() bool { return a == Amount{} }

// ClassShare is one class's priority and its ceiling, as fractions of the fleet.
type ClassShare struct {
	// Guaranteed is held out of a borrower's reach while this class has demand
	// queued. It is a floor, not an allocation.
	Guaranteed float64
	// Max caps the class even when the fleet is otherwise idle. Zero means the
	// class may grow into whatever the guarantees leave free.
	Max float64
}

// Config declares the fleet envelope and how it is shared.
type Config struct {
	// Total is the sum of what every backend will admit. It is the ledger's
	// ceiling, not a promise that any one node can hold a given request.
	Total   Amount
	Classes map[string]ClassShare
	// LeaseTTL bounds a grant whose owner stops renewing. A caller that dies must
	// not hold fleet capacity until the run ends.
	LeaseTTL time.Duration
	// MaxBypass is how many times a queued borrower may be passed over before it
	// is admitted ahead of newer work.
	MaxBypass int
}

// Validate refuses a configuration the ledger cannot honour.
func (c Config) Validate() error {
	var guaranteed float64
	for name, share := range c.Classes {
		if share.Guaranteed <= 0 || share.Guaranteed > 1 {
			return fmt.Errorf("quota class %q guarantee %.3f must be in (0, 1]", name, share.Guaranteed)
		}
		if share.Max != 0 && share.Max < share.Guaranteed {
			return fmt.Errorf("quota class %q ceiling %.3f is below its guarantee %.3f", name, share.Max, share.Guaranteed)
		}
		guaranteed += share.Guaranteed
	}
	if guaranteed > 1+1e-9 {
		return fmt.Errorf(
			"quota guarantees sum to %.3f, which exceeds the fleet. Keep the sum below one so every "+
				"guarantee stays satisfiable without preempting a running sandbox", guaranteed)
	}
	if c.LeaseTTL <= 0 {
		return fmt.Errorf("quota lease TTL must be greater than zero")
	}
	return nil
}

type grant struct {
	id        string
	class     string
	amount    Amount
	owner     string
	renewedAt time.Time
}

type waiter struct {
	class   string
	amount  Amount
	bypassd int
}

// Ledger is the fleet's admission accounting. Safe for concurrent use.
type Ledger struct {
	mu     sync.Mutex
	cfg    Config
	grants map[string]*grant
	// used is maintained incrementally rather than summed per call, because
	// admission is on the create path and a fleet holds hundreds of thousands of
	// grants.
	used   map[string]Amount
	queued map[string][]*waiter
	owners map[string]time.Time
	now    func() time.Time

	admitted, refused, bypasses int64
}

// New returns a ledger over a validated configuration.
func New(cfg Config) (*Ledger, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.MaxBypass <= 0 {
		cfg.MaxBypass = 8
	}
	return &Ledger{
		cfg:    cfg,
		grants: map[string]*grant{},
		used:   map[string]Amount{},
		queued: map[string][]*waiter{},
		owners: map[string]time.Time{},
		now:    time.Now,
	}, nil
}

// SetClock replaces the ledger's clock, for a test that drives time directly.
func (l *Ledger) SetClock(now func() time.Time) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.now = now
}

// Headroom returns what one class could be granted right now.
//
// This is the number an outside scheduler has to compare against, and it is not
// the fleet remainder: a borrower may not take another class's unmet guarantee,
// and a class may have a ceiling of its own, so the remainder overstates what any
// one class can get by the whole of every other queued guarantee.
func (l *Ledger) Headroom(class string) Amount {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.headroomLocked(class)
}

func (l *Ledger) headroomLocked(class string) Amount {
	free := l.cfg.Total.Sub(l.totalUsedLocked()).Sub(l.reservedForOthersLocked(class))
	share, declared := l.cfg.Classes[class]
	if !declared || share.Max == 0 {
		return free
	}
	underCeiling := l.cfg.Total.Scale(share.Max).Sub(l.used[class])
	return Amount{
		MemoryMB:  min64(free.MemoryMB, underCeiling.MemoryMB),
		CPUMillis: min64(free.CPUMillis, underCeiling.CPUMillis),
		GPUCount:  min32(free.GPUCount, underCeiling.GPUCount),
		DiskMB:    min64(free.DiskMB, underCeiling.DiskMB),
		Sandboxes: min64(free.Sandboxes, underCeiling.Sandboxes),
	}
}

// reservedForOthersLocked returns capacity held out of this class's reach.
//
// Only a class with a request *queued* reserves anything. Reserving against a
// class that is not asking would leave the fleet idle and full at once, which is
// the failure this rule exists to prevent.
func (l *Ledger) reservedForOthersLocked(class string) Amount {
	var reserved Amount
	for name, share := range l.cfg.Classes {
		if name == class || len(l.queued[name]) == 0 {
			continue
		}
		unmet := l.cfg.Total.Scale(share.Guaranteed).Sub(l.used[name])
		reserved = reserved.Add(unmet)
	}
	return reserved
}

func (l *Ledger) totalUsedLocked() Amount {
	var total Amount
	for _, amount := range l.used {
		total = total.Add(amount)
	}
	return total
}

// Acquire admits one request, or reports why it cannot be admitted now.
//
// It never blocks. A caller that must wait enqueues through Enqueue, so the
// ledger's lock is never held across a wait and a queued request is visible to
// every other class's headroom while it waits.
func (l *Ledger) Acquire(id, class, owner string, amount Amount) (bool, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, exists := l.grants[id]; exists {
		return false, fmt.Errorf("quota grant %q already exists", id)
	}
	l.expireLocked()
	if !amount.FitsIn(l.headroomLocked(class)) {
		l.refused++
		return false, nil
	}
	now := l.now()
	l.grants[id] = &grant{id: id, class: class, amount: amount, owner: owner, renewedAt: now}
	l.used[class] = l.used[class].Add(amount)
	l.owners[owner] = now
	l.admitted++
	l.dequeueLocked(class, amount)
	return true, nil
}

// Enqueue records demand that could not be admitted, so the class's guarantee is
// reserved while it waits.
func (l *Ledger) Enqueue(class string, amount Amount) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.queued[class] = append(l.queued[class], &waiter{class: class, amount: amount})
}

// Cancel withdraws queued demand whose caller gave up.
func (l *Ledger) Cancel(class string, amount Amount) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.dequeueLocked(class, amount)
}

func (l *Ledger) dequeueLocked(class string, amount Amount) {
	queue := l.queued[class]
	for i, w := range queue {
		if w.amount == amount {
			l.queued[class] = append(queue[:i], queue[i+1:]...)
			if len(l.queued[class]) == 0 {
				delete(l.queued, class)
			}
			return
		}
	}
}

// Release returns one grant. Releasing an unknown grant is not an error, because
// a retried release must be safe.
func (l *Ledger) Release(id string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.releaseLocked(id)
}

func (l *Ledger) releaseLocked(id string) {
	g, ok := l.grants[id]
	if !ok {
		return
	}
	delete(l.grants, id)
	l.used[g.class] = l.used[g.class].Sub(g.amount)
	if l.used[g.class].IsZero() {
		delete(l.used, g.class)
	}
}

// Renew records that an owner is still holding its grants.
func (l *Ledger) Renew(owner string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, known := l.owners[owner]; known {
		l.owners[owner] = l.now()
	}
}

// ReleaseOwner returns everything one owner held, for a clean shutdown.
func (l *Ledger) ReleaseOwner(owner string) int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.releaseOwnerLocked(owner)
}

func (l *Ledger) releaseOwnerLocked(owner string) int {
	var ids []string
	for id, g := range l.grants {
		if g.owner == owner {
			ids = append(ids, id)
		}
	}
	for _, id := range ids {
		l.releaseLocked(id)
	}
	delete(l.owners, owner)
	return len(ids)
}

// expireLocked reclaims the grants of owners that stopped renewing.
//
// An owner's silence is the only evidence the ledger has that its sandboxes are
// gone. Age of the grant itself is not: a long-lived sandbox is indistinguishable
// from an abandoned one by age alone.
func (l *Ledger) expireLocked() {
	now := l.now()
	for owner, seen := range l.owners {
		if now.Sub(seen) > l.cfg.LeaseTTL {
			l.releaseOwnerLocked(owner)
		}
	}
}

// Expire runs lease reclamation, for a caller driving the ledger from a sweep.
func (l *Ledger) Expire() {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.expireLocked()
}

// ClassReport is one class's accounting, for the caller's metric hook.
type ClassReport struct {
	Guaranteed float64
	Max        float64
	Granted    Amount
	Headroom   Amount
	Queued     int
}

// Report returns the ledger's accounting. The planes report and never log.
func (l *Ledger) Report() map[string]ClassReport {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.expireLocked()
	names := make([]string, 0, len(l.cfg.Classes))
	for name := range l.cfg.Classes {
		names = append(names, name)
	}
	for name := range l.used {
		if _, declared := l.cfg.Classes[name]; !declared {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	out := make(map[string]ClassReport, len(names))
	for _, name := range names {
		share := l.cfg.Classes[name]
		out[name] = ClassReport{
			Guaranteed: share.Guaranteed,
			Max:        share.Max,
			Granted:    l.used[name],
			Headroom:   l.headroomLocked(name),
			Queued:     len(l.queued[name]),
		}
	}
	return out
}

// Stats returns cumulative counters for the metric hook.
func (l *Ledger) Stats() (admitted, refused, bypasses int64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.admitted, l.refused, l.bypasses
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func min64(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func max32(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}

func min32(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}
