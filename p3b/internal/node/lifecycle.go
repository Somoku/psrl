package node

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// exitRecordTTL is how long the node remembers why a sandbox left.
//
// A caller that reaches for a sandbox just after it is gone gets an attribution
// rather than silence, and an attribution older than this can explain nothing a
// caller could still be asking about.
const exitRecordTTL = 10 * time.Minute

// DefaultLifetime is the lifetime a sandbox gets when its create states none.
// It is a fallback, not the design: the caller states how long its sandbox may
// live, and this only keeps a caller that omits the field from creating an
// unbounded one.
const DefaultLifetime = 4 * time.Hour

// ExitReason says why a sandbox stopped existing. A post mortem starts here, so
// these must not collapse into one another: an operator acts differently on one
// released by its caller, one reaped for outliving its lifetime, and one whose
// owner died.
type ExitReason string

const (
	ExitReleased        ExitReason = "released"
	ExitReapedLifetime  ExitReason = "reaped_lifetime"
	ExitReclaimedOrphan ExitReason = "reclaimed_orphan"
)

// exitRecord attributes a sandbox that is no longer held.
type exitRecord struct {
	reason ExitReason
	at     time.Time
}

// resident is one sandbox this node holds.
type resident struct {
	handle    backend.Handle
	backend   backend.Backend
	leaseID   string
	owner     string
	execMode  string // "persistent" | "one_shot" | "" (one-shot default)
	createdAt time.Time
	// lifetime is the absolute span this sandbox may exist, stated on its create.
	// It is the backstop behind explicit release and owner death, not an idle
	// window: a sandbox is otherwise reclaimed when its owner says so.
	lifetime time.Duration
	// lastActivity moves on both command boundaries, start and return. A stamp
	// written only when a command returns stays stale for the whole of a long
	// command, so a sweep reading it would report a running test suite as idle.
	lastActivity time.Time
	inFlight     int
}

// Lifecycle owns every sandbox on one node and reclaims them in one sweep.
//
// One loop, one clock, three reasons: released, outlived its lifetime, or its
// owner died. Idle pause and idle reap are gone: a sandbox's lifetime is stated
// on its create, owner death is read from owner liveness, and an explicit release
// is what an idle sandbox normally gets.
type Lifecycle struct {
	mu        sync.Mutex
	nodeID    string
	admission *Admission
	sweep     time.Duration
	residents map[string]*resident
	now       func() time.Time

	// ownerAlive reports whether a caller is still there. Owner silence is the
	// only evidence a node has that a sandbox is abandoned; sandbox age is not.
	ownerAlive func(owner string) bool

	sweeping sync.Mutex
	stop     chan struct{}
	stopped  sync.WaitGroup

	counters map[ExitReason]int64
	failures int64

	// lastExit remembers why each sandbox left, so a caller that reaches for it
	// after it is gone gets an attribution instead of silence. Pruned by the
	// sweep once an entry is older than exitRecordTTL.
	lastExit map[string]exitRecord
}

// NewLifecycle returns the node's sandbox owner.
func NewLifecycle(nodeID string, admission *Admission, sweepInterval time.Duration) (*Lifecycle, error) {
	if sweepInterval <= 0 {
		return nil, fmt.Errorf("node reclamation sweep interval must be greater than zero")
	}
	return &Lifecycle{
		nodeID:     nodeID,
		admission:  admission,
		sweep:      sweepInterval,
		residents:  map[string]*resident{},
		now:        time.Now,
		ownerAlive: func(string) bool { return true },
		counters:   map[ExitReason]int64{},
		lastExit:   map[string]exitRecord{},
	}, nil
}

// SetClock replaces the lifecycle's clock, for a test that drives time directly.
func (l *Lifecycle) SetClock(now func() time.Time) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.now = now
}

// SetOwnerLiveness installs the check that decides whether a caller is still
// there.
func (l *Lifecycle) SetOwnerLiveness(alive func(owner string) bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.ownerAlive = alive
}

// Adopt records a sandbox this node now owns, with the lifetime its create
// stated.
func (l *Lifecycle) Adopt(handle backend.Handle, b backend.Backend, leaseID, owner, execMode string, lifetime time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.now()
	l.residents[handle.SandboxID] = &resident{
		handle: handle, backend: b, leaseID: leaseID, owner: owner, execMode: execMode,
		createdAt: now, lifetime: lifetime, lastActivity: now,
	}
}

// ExecMode returns the exec mode recorded for a sandbox, for the node server to
// select the right execution strategy.
func (l *Lifecycle) ExecMode(sandboxID string) string {
	l.mu.Lock()
	defer l.mu.Unlock()
	if held, ok := l.residents[sandboxID]; ok {
		return held.execMode
	}
	return ""
}

// Lookup returns one owned sandbox's handle and backend.
func (l *Lifecycle) Lookup(sandboxID string) (backend.Handle, backend.Backend, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	held, owned := l.residents[sandboxID]
	if !owned {
		return backend.Handle{}, nil, false
	}
	return held.handle, held.backend, true
}

// Begin marks a command as started, so the sweep cannot read a busy sandbox as
// idle. The returned function marks it finished.
func (l *Lifecycle) Begin(sandboxID string) func() {
	l.mu.Lock()
	held, owned := l.residents[sandboxID]
	if owned {
		held.inFlight++
		held.lastActivity = l.now()
	}
	l.mu.Unlock()
	if !owned {
		return func() {}
	}
	return func() {
		l.mu.Lock()
		defer l.mu.Unlock()
		if held.inFlight > 0 {
			held.inFlight--
		}
		held.lastActivity = l.now()
	}
}

// Release destroys one sandbox and returns its capacity.
//
// The sandbox is destroyed before its reservation is returned, and the
// reservation is held if destruction fails: a container the daemon refused to
// remove still holds its memory, so returning the reservation would over-commit
// the node and turn one stuck container into a host OOM.
func (l *Lifecycle) Release(ctx context.Context, sandboxID string, reason ExitReason) error {
	l.mu.Lock()
	held, owned := l.residents[sandboxID]
	l.mu.Unlock()
	if !owned {
		// Already gone is the outcome the caller wanted.
		return nil
	}
	if err := held.backend.Release(ctx, held.handle); err != nil {
		l.mu.Lock()
		l.failures++
		l.mu.Unlock()
		return fmt.Errorf("release %s: %w", sandboxID, err)
	}
	l.mu.Lock()
	delete(l.residents, sandboxID)
	l.counters[reason]++
	l.lastExit[sandboxID] = exitRecord{reason: reason, at: l.now()}
	l.mu.Unlock()
	l.admission.Release(held.leaseID)
	return nil
}

// ExitReasonOf reports why a sandbox this node used to hold left it.
//
// A caller that reaches for a sandbox after it is gone -- a command against a
// handle the reaper already released -- gets the attribution rather than an
// empty answer. The second result is false when the sandbox was never held here
// or its record has aged out.
func (l *Lifecycle) ExitReasonOf(sandboxID string) (ExitReason, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	rec, ok := l.lastExit[sandboxID]
	return rec.reason, ok
}

// Touch records activity on a sandbox this node holds.
//
// A command that does not pass through Node.Exec -- one driven by the sandbox's
// own in-container agent -- has no other way to tell the node it is busy. An
// unknown id is ignored: a touch racing a release must not resurrect anything.
func (l *Lifecycle) Touch(sandboxID string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if held, ok := l.residents[sandboxID]; ok {
		held.lastActivity = l.now()
	}
}

// Sweep decides once for every sandbox this node holds.
//
// A sandbox goes for exactly one of three reasons, checked in order: its owner is
// gone, it has outlived the lifetime its create stated, or its caller released
// it (which the caller does directly, not through the sweep). The lifetime is an
// absolute backstop, so it reclaims even a busy sandbox; owner loss is read from
// liveness rather than from age.
func (l *Lifecycle) Sweep(ctx context.Context) SweepReport {
	// One sweep at a time: a second pass entering while the first is still
	// releasing would decide twice on the same sandbox.
	l.sweeping.Lock()
	defer l.sweeping.Unlock()

	l.mu.Lock()
	now := l.now()
	// Attribution records older than exitRecordTTL can explain nothing a caller
	// could still be asking about, so the sweep is also where they age out.
	for id, rec := range l.lastExit {
		if now.Sub(rec.at) >= exitRecordTTL {
			delete(l.lastExit, id)
		}
	}
	type decision struct {
		id     string
		held   *resident
		reason ExitReason
	}
	var decisions []decision
	for id, held := range l.residents {
		switch {
		case !l.ownerAlive(held.owner):
			decisions = append(decisions, decision{id, held, ExitReclaimedOrphan})
		case now.Sub(held.createdAt) >= held.lifetime:
			decisions = append(decisions, decision{id, held, ExitReapedLifetime})
		}
	}
	busy := 0
	for _, held := range l.residents {
		if held.inFlight > 0 {
			busy++
		}
	}
	l.mu.Unlock()

	// Stable order, so a sweep's effects are reproducible from its inputs.
	sort.Slice(decisions, func(i, j int) bool { return decisions[i].id < decisions[j].id })

	report := SweepReport{SkippedBusy: busy}
	for _, d := range decisions {
		if err := l.Release(ctx, d.id, d.reason); err != nil {
			report.Failures++
			continue
		}
		switch d.reason {
		case ExitReapedLifetime:
			report.ReapedLifetime = append(report.ReapedLifetime, d.id)
		case ExitReclaimedOrphan:
			report.ReclaimedOrphan = append(report.ReclaimedOrphan, d.id)
		}
	}
	return report
}

// Start runs the sweep on its own cadence until Stop.
func (l *Lifecycle) Start(ctx context.Context) {
	l.mu.Lock()
	if l.stop != nil {
		l.mu.Unlock()
		return
	}
	stop := make(chan struct{})
	l.stop = stop
	l.mu.Unlock()

	l.stopped.Add(1)
	go func() {
		defer l.stopped.Done()
		ticker := time.NewTicker(l.sweep)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				l.Sweep(ctx)
				l.admission.Expire()
			}
		}
	}()
}

// Stop ends the sweep, leaving the sandboxes it was watching in place.
func (l *Lifecycle) Stop() {
	l.mu.Lock()
	stop := l.stop
	l.stop = nil
	l.mu.Unlock()
	if stop == nil {
		return
	}
	close(stop)
	l.stopped.Wait()
}

// SweepReport is what one pass did, by reason.
type SweepReport struct {
	ReapedLifetime  []string
	ReclaimedOrphan []string
	SkippedBusy     int
	Failures        int
}

// Released reports how many sandboxes the pass destroyed, for any reason.
func (r SweepReport) Released() int {
	return len(r.ReapedLifetime) + len(r.ReclaimedOrphan)
}

// LifecycleReport is the node's own accounting for the metric hook.
type LifecycleReport struct {
	Resident int
	Busy     int
	Exits    map[ExitReason]int64
	Failures int64
}

// Snapshot returns what this node holds. The planes report and never log.
func (l *Lifecycle) Snapshot() LifecycleReport {
	l.mu.Lock()
	defer l.mu.Unlock()
	busy := 0
	for _, held := range l.residents {
		if held.inFlight > 0 {
			busy++
		}
	}
	exits := make(map[ExitReason]int64, len(l.counters))
	for reason, count := range l.counters {
		exits[reason] = count
	}
	return LifecycleReport{
		Resident: len(l.residents), Busy: busy,
		Exits: exits, Failures: l.failures,
	}
}
