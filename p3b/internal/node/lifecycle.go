package node

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// Windows are the spans the reclamation sweep enforces.
//
// They arrive resolved from the one timing contract rather than being derived
// here, because the orderings between them are asserted once and must not be
// re-derived anywhere else.
type Windows struct {
	PauseWindow   time.Duration
	ReapWindow    time.Duration
	Lifetime      time.Duration
	SweepInterval time.Duration
}

// Validate refuses windows the sweep cannot honour.
func (w Windows) Validate() error {
	if w.PauseWindow <= 0 || w.ReapWindow <= 0 || w.Lifetime <= 0 || w.SweepInterval <= 0 {
		return fmt.Errorf("node reclamation windows must be greater than zero")
	}
	if w.PauseWindow >= w.ReapWindow {
		return fmt.Errorf("pause window (%s) must be shorter than the reap window (%s), or a sandbox is "+
			"destroyed before it is ever paused", w.PauseWindow, w.ReapWindow)
	}
	if w.ReapWindow >= w.Lifetime {
		return fmt.Errorf("reap window (%s) must be shorter than the lifetime (%s), or the backstop fires "+
			"first and the reaper never runs", w.ReapWindow, w.Lifetime)
	}
	if w.SweepInterval >= w.PauseWindow {
		return fmt.Errorf("sweep interval (%s) must be shorter than the shortest window it enforces (%s)",
			w.SweepInterval, w.PauseWindow)
	}
	return nil
}

// ExitReason says why a sandbox stopped existing. A post mortem starts here, so
// these must not collapse into one another: an operator acts differently on one
// released by its caller, one reaped for silence, and one whose owner died.
type ExitReason string

const (
	ExitReleased        ExitReason = "released"
	ExitReapedIdle      ExitReason = "reaped_idle"
	ExitReapedLifetime  ExitReason = "reaped_lifetime"
	ExitReclaimedOrphan ExitReason = "reclaimed_orphan"
)

// resident is one sandbox this node holds.
type resident struct {
	handle    backend.Handle
	backend   backend.Backend
	leaseID   string
	owner     string
	createdAt time.Time
	// lastActivity moves on both command boundaries, start and return. A stamp
	// written only when a command returns stays stale for the whole of a long
	// command, so a sweep reading it would pause a running test suite.
	lastActivity time.Time
	inFlight     int
	paused       bool
}

// Lifecycle owns every sandbox on one node and reclaims them in one sweep.
//
// One loop, one clock, four reasons. Idle pause and orphan reclamation used to
// be separate mechanisms on separate clocks, so a caller exiting silently
// stopped half the answer to the only question that matters here: which
// sandboxes on this node should go.
type Lifecycle struct {
	mu        sync.Mutex
	nodeID    string
	admission *Admission
	windows   Windows
	residents map[string]*resident
	now       func() time.Time

	// ownerAlive reports whether a caller is still there. Owner silence is the
	// only evidence a node has that a sandbox is abandoned; sandbox age is not.
	ownerAlive func(owner string) bool

	sweeping sync.Mutex
	stop     chan struct{}
	stopped  sync.WaitGroup

	counters map[ExitReason]int64
	paused   int64
	failures int64
}

// NewLifecycle returns the node's sandbox owner.
func NewLifecycle(nodeID string, admission *Admission, windows Windows) (*Lifecycle, error) {
	if err := windows.Validate(); err != nil {
		return nil, err
	}
	return &Lifecycle{
		nodeID:     nodeID,
		admission:  admission,
		windows:    windows,
		residents:  map[string]*resident{},
		now:        time.Now,
		ownerAlive: func(string) bool { return true },
		counters:   map[ExitReason]int64{},
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

// Adopt records a sandbox this node now owns.
func (l *Lifecycle) Adopt(handle backend.Handle, b backend.Backend, leaseID, owner string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.now()
	l.residents[handle.SandboxID] = &resident{
		handle: handle, backend: b, leaseID: leaseID, owner: owner,
		createdAt: now, lastActivity: now,
	}
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
	l.mu.Unlock()
	l.admission.Release(held.leaseID)
	return nil
}

// Sweep decides once for every sandbox this node holds.
//
// The reasons are ordered by how much they preserve, and the order is the
// policy rather than an implementation detail: a sandbox is paused before it is
// destroyed, and the lifetime is a backstop behind the idle windows rather than
// a competitor to them. Each sandbox is decided for exactly one reason per pass.
func (l *Lifecycle) Sweep(ctx context.Context) SweepReport {
	// One sweep at a time: a second pass entering while the first is still
	// releasing would decide twice on the same sandbox.
	l.sweeping.Lock()
	defer l.sweeping.Unlock()

	l.mu.Lock()
	now := l.now()
	type decision struct {
		id     string
		held   *resident
		action string
		reason ExitReason
	}
	var decisions []decision
	for id, held := range l.residents {
		switch {
		case !l.ownerAlive(held.owner):
			decisions = append(decisions, decision{id, held, "release", ExitReclaimedOrphan})
		case now.Sub(held.createdAt) >= l.windows.Lifetime:
			decisions = append(decisions, decision{id, held, "release", ExitReapedLifetime})
		case held.inFlight > 0:
			// Busy. Age alone would read a long command as idle.
		case now.Sub(held.lastActivity) >= l.windows.ReapWindow:
			decisions = append(decisions, decision{id, held, "release", ExitReapedIdle})
		case now.Sub(held.lastActivity) >= l.windows.PauseWindow && !held.paused:
			decisions = append(decisions, decision{id, held, "pause", ""})
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
		if d.action == "pause" {
			l.pauseOne(ctx, d.held, &report)
			continue
		}
		if err := l.Release(ctx, d.id, d.reason); err != nil {
			report.Failures++
			continue
		}
		switch d.reason {
		case ExitReapedIdle:
			report.ReapedIdle = append(report.ReapedIdle, d.id)
		case ExitReapedLifetime:
			report.ReapedLifetime = append(report.ReapedLifetime, d.id)
		case ExitReclaimedOrphan:
			report.ReclaimedOrphan = append(report.ReclaimedOrphan, d.id)
		}
	}
	return report
}

func (l *Lifecycle) pauseOne(ctx context.Context, held *resident, report *SweepReport) {
	stateful, canPause := held.backend.(backend.Stateful)
	if !canPause {
		return
	}
	mode := "hibernate"
	if held.backend.Capabilities().Supports("freeze") {
		// A freeze keeps the sandbox on this host and is cheaper to undo.
		mode = "freeze"
	}
	if err := stateful.Pause(ctx, held.handle, mode); err != nil {
		// Not idle in fact, or no pause to offer. The next sweep decides again,
		// and neither is a failure worth reporting.
		return
	}
	l.mu.Lock()
	held.paused = true
	l.paused++
	l.mu.Unlock()
	report.Paused = append(report.Paused, held.handle.SandboxID)
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
		ticker := time.NewTicker(l.windows.SweepInterval)
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
	Paused          []string
	ReapedIdle      []string
	ReapedLifetime  []string
	ReclaimedOrphan []string
	SkippedBusy     int
	Failures        int
}

// Released reports how many sandboxes the pass destroyed, for any reason.
func (r SweepReport) Released() int {
	return len(r.ReapedIdle) + len(r.ReapedLifetime) + len(r.ReclaimedOrphan)
}

// LifecycleReport is the node's own accounting for the metric hook.
type LifecycleReport struct {
	Resident int
	Busy     int
	Paused   int64
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
		Paused: l.paused, Exits: exits, Failures: l.failures,
	}
}
