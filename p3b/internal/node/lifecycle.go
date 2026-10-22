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

// The runtime prober is the node's continuous reconciliation with the daemon.
// A record is not evidence that a container is alive; it is only evidence that
// the node once asked for one. These spans are how fast that claim is checked
// again, not how long a sandbox runs.
const (
	DefaultRuntimeProbeInterval = 30 * time.Second
	DefaultRuntimeProbeTimeout  = 5 * time.Second
	DefaultRuntimeProbeWorkers  = 8
)

// DefaultLifetime is the lifetime a sandbox gets when its create states none.
// It is a fallback, not the design: the caller states how long its sandbox may
// live, and this only keeps a caller that omits the field from creating an
// unbounded one.
const DefaultLifetime = 4 * time.Hour

// reserveDeadline bounds a grant that never became a container.
//
// A control plane that dies between reserving capacity and asking for a create
// would otherwise hold that budget until the node restarts, and no container
// exists for a sweep to find.
const reserveDeadline = 2 * time.Minute

// ExitReason says why a sandbox stopped existing. A post mortem starts here, so
// these must not collapse into one another: an operator acts differently on one
// released by its caller, one reaped for outliving its lifetime, and one whose
// owner died.
type ExitReason string

const (
	ExitReleased        ExitReason = "released"
	ExitReapedLifetime  ExitReason = "reaped_lifetime"
	ExitReclaimedOrphan ExitReason = "reclaimed_orphan"
	// ExitOOM is a sandbox the kernel kept killing for memory until the node ended it.
	//
	// It is its own reason rather than a failure or a release because what the caller does
	// with it is different: a trajectory whose tool calls were killed for memory is not a
	// trajectory that produced a wrong answer, it is one that never ran, and training on
	// it teaches the policy something about a task it never saw. The caller resamples.
	ExitOOM ExitReason = "oom"
	// ExitExited is a sandbox whose container stopped for a reason other than OOM:
	// its entry process exited, or the runtime reported it dead. The container's
	// filesystem -- and everything the caller built in it -- is gone with it.
	ExitExited ExitReason = "exited"
	// ExitRuntimeGone is a container the runtime no longer has at all. It is
	// distinct from ExitExited because the runtime could not even report an exit:
	// a removed container is a stronger statement than a stopped one.
	ExitRuntimeGone ExitReason = "runtime_gone"
)

// ExitAttribution is why a sandbox left this node and what the runtime reported
// about it. The exit code travels beside the reason because the reason is a
// classification, not evidence: an operator correlating a sandbox loss with a
// node's cgroup or kernel events needs the code and the moment.
type ExitAttribution struct {
	Reason        ExitReason
	ExitCode      int
	ExitCodeKnown bool
	FinishedAt    time.Time
}

// exitCodeForReason states the exit code a reason implies when the runtime did
// not report one. Only an OOM kill has an answer the classification itself
// establishes (SIGKILL, 137); every other reason leaves the code unknown rather
// than claiming a clean exit.
func exitCodeForReason(reason ExitReason) (int, bool) {
	if reason == ExitOOM {
		return 137, true
	}
	return 0, false
}

// exitRecord attributes a sandbox that is no longer held.
type exitRecord struct {
	reason     ExitReason
	exitCode   int
	codeKnown  bool
	at         time.Time
	finishedAt time.Time
}

// Lifecycle owns every sandbox on one node and reclaims them in one sweep.
//
// One loop, one clock, one record per sandbox. The record is the only copy of
// what a sandbox holds: admission keeps envelope arithmetic, this keeps the
// runtime object, and neither can be returned without the other.
type Lifecycle struct {
	mu        sync.Mutex
	nodeID    string
	admission *Admission
	sweep     time.Duration
	// allocations is keyed by the lease id, which exists from the first grant.
	allocations map[string]*allocation
	// bySandbox is the exec and status path, where only the container id is known.
	bySandbox map[string]*allocation
	now       func() time.Time

	// ownerAlive reports whether a caller is still there. Owner silence is the
	// only evidence a node has that a sandbox is abandoned.
	ownerAlive func(owner string) bool

	// cleanup carries out the removals a sweep decides, on its own workers.
	cleanup    *Cleanup
	cleanupCfg CleanupConfig
	// draining is set while at least one record could not be destroyed, because
	// its memory is still in use and admitting against it would over-commit.
	draining bool

	sweeping sync.Mutex
	stop     chan struct{}
	stopped  sync.WaitGroup

	counters map[ExitReason]int64
	failures int64

	// Runtime probing: the continuous check that a resident record still has a
	// live container. It runs on its own goroutine and its own clock, because a
	// slow daemon must not stretch the sweep that enforces owner silence.
	probeInterval time.Duration
	probeTimeout  time.Duration
	probeWorkers  int
	probeStop     chan struct{}
	probeStopped  sync.WaitGroup
	probeChecks   int64
	probeGone     int64
	probeErrors   int64
	probeFailures int64

	// reportSeq advances with every inventory snapshot, so the control plane can
	// tell a report generated before a grant existed from one generated after it.
	reportSeq uint64

	// lastExit remembers why each sandbox left, so a caller that reaches for it
	// after it is gone gets an attribution instead of silence.
	lastExit map[string]exitRecord
}

// NewLifecycle returns the node's sandbox owner.
func NewLifecycle(nodeID string, admission *Admission, sweepInterval time.Duration) (*Lifecycle, error) {
	return NewLifecycleWithCleanup(nodeID, admission, sweepInterval, DefaultCleanupConfig())
}

// NewLifecycleWithCleanup returns the node's sandbox owner with reclamation's own
// concurrency stated, which is what a test drives directly.
func NewLifecycleWithCleanup(
	nodeID string, admission *Admission, sweepInterval time.Duration, cleanupCfg CleanupConfig,
) (*Lifecycle, error) {
	if sweepInterval <= 0 {
		return nil, fmt.Errorf("node reclamation sweep interval must be greater than zero")
	}
	life := &Lifecycle{
		nodeID:      nodeID,
		admission:   admission,
		sweep:       sweepInterval,
		allocations: map[string]*allocation{},
		bySandbox:   map[string]*allocation{},
		now:         time.Now,
		ownerAlive:  func(string) bool { return true },
		cleanupCfg:  cleanupCfg.validated(),
		counters:    map[ExitReason]int64{},
		lastExit:    map[string]exitRecord{},

		probeInterval: DefaultRuntimeProbeInterval,
		probeTimeout:  DefaultRuntimeProbeTimeout,
		probeWorkers:  DefaultRuntimeProbeWorkers,
	}
	life.cleanup = newCleanup(life, life.cleanupCfg)
	return life, nil
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

// SetRuntimeProbeConfig overrides the prober's cadence and bounds.
//
// A non-positive interval or timeout keeps the default, so a deployment cannot
// accidentally configure a probe that runs continuously or waits forever. It is
// mostly for tests: a test that drives the probe directly calls ProbeRuntime and
// does not need the loop at all.
func (l *Lifecycle) SetRuntimeProbeConfig(interval, timeout time.Duration, workers int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if interval > 0 {
		l.probeInterval = interval
	}
	if timeout > 0 {
		l.probeTimeout = timeout
	}
	if workers > 0 {
		l.probeWorkers = workers
	}
}

// Reserve records a grant whose runtime object does not exist yet.
//
// The record exists from the grant, not from the container, so a grant that is
// never turned into one is still swept rather than held forever.
func (l *Lifecycle) Reserve(grant Grant, owner, execMode string, lifetime time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.now()
	l.allocations[grant.LeaseID] = &allocation{
		leaseID: grant.LeaseID, backend: nil, class: grant.Class, owner: owner,
		execMode: execMode, grant: grant, state: allocReserved,
		createdAt: now, lifetime: lifetime, lastActivity: now,
	}
}

// Claim takes a reserved grant for creation, consuming it exactly once.
//
// A false answer means the grant was already spent or already reclaimed, and a
// create against it must be refused rather than become a second container.
func (l *Lifecycle) Claim(leaseID string) (Grant, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	held, known := l.allocations[leaseID]
	if !known || held.state != allocReserved {
		return Grant{}, false
	}
	held.state = allocCreating
	held.lastActivity = l.now()
	return held.grant, true
}

// AdoptCreated records the runtime object a create produced.
//
// A false answer means the record was reclaimed while the create was in flight,
// so the caller is told to destroy what it made rather than leak it.
func (l *Lifecycle) AdoptCreated(leaseID string, handle backend.Handle, b backend.Backend) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	held, known := l.allocations[leaseID]
	if !known || held.state != allocCreating {
		return false
	}
	held.handle = handle
	held.backend = b
	held.sandboxID = handle.SandboxID
	held.state = allocResident
	held.lastActivity = l.now()
	l.bySandbox[handle.SandboxID] = held
	return true
}

// FailCreate drops a record whose create never produced a runtime object, which
// returns its budget without a destroy.
func (l *Lifecycle) FailCreate(leaseID string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	held, known := l.allocations[leaseID]
	if !known || held.holdsRuntime() {
		return
	}
	l.removeLocked(held, ExitReleased)
}

// ExecMode returns the exec mode recorded for a sandbox, for the node server to
// select the right execution strategy.
func (l *Lifecycle) ExecMode(sandboxID string) string {
	l.mu.Lock()
	defer l.mu.Unlock()
	if held, ok := l.bySandbox[sandboxID]; ok {
		return held.execMode
	}
	return ""
}

// Lookup returns one owned sandbox's handle and backend.
func (l *Lifecycle) Lookup(sandboxID string) (backend.Handle, backend.Backend, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	held, owned := l.bySandbox[sandboxID]
	if !owned {
		return backend.Handle{}, nil, false
	}
	return held.handle, held.backend, true
}

// Begin marks a command as started, so the sweep cannot read a busy sandbox as
// idle. The returned function marks it finished.
func (l *Lifecycle) Begin(sandboxID string) func() {
	l.mu.Lock()
	held, owned := l.bySandbox[sandboxID]
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

// BeginCreate marks a create as running, so the sweep cannot read its record as a
// grant that was never claimed. The returned function marks it finished.
//
// A create includes an image pull, which is the slowest thing a node does and the
// one an operator can do least about, so elapsed time says nothing about whether
// the create is alive. Holding the record in flight is what says it.
func (l *Lifecycle) BeginCreate(leaseID string) func() {
	l.mu.Lock()
	held, known := l.allocations[leaseID]
	marked := known && held.state == allocCreating
	if marked {
		held.inFlight++
		held.lastActivity = l.now()
	}
	l.mu.Unlock()
	if !marked {
		return func() {}
	}
	return func() {
		l.mu.Lock()
		defer l.mu.Unlock()
		if held.inFlight > 0 {
			held.inFlight--
		}
	}
}

// Release destroys one sandbox and returns its capacity.
//
// The sandbox is destroyed before its reservation is returned, and the
// reservation is held if destruction fails: a container the daemon refused to
// remove still holds its memory, so returning the reservation would over-commit
// the node and turn one stuck container into a host OOM.
//
// This is the caller-driven path, so it is synchronous: a caller that released a
// sandbox has to be told whether it is gone. Reclamation's own removals go
// through the cleanup workers instead, where nobody is waiting.
func (l *Lifecycle) Release(ctx context.Context, sandboxID string, reason ExitReason) error {
	l.mu.Lock()
	held, owned := l.bySandbox[sandboxID]
	if !owned {
		l.mu.Unlock()
		// Already gone is the outcome the caller wanted.
		return nil
	}
	if held.state == allocDestroying {
		// A destroy is already in flight. A second one would be a second removal
		// of one container, and the first will report the outcome.
		l.mu.Unlock()
		return nil
	}
	if !held.holdsRuntime() {
		// No runtime object exists, so nothing has to be destroyed and the budget
		// comes back now.
		l.removeLocked(held, reason)
		l.mu.Unlock()
		return nil
	}
	held.state = allocDestroying
	held.pendingReason = reason
	held.pendingExitCode, held.pendingCodeKnown = exitCodeForReason(reason)
	held.pendingFinishedAt = l.now()
	leaseID, handle, hosted := held.leaseID, held.handle, held.backend
	l.mu.Unlock()

	if err := hosted.Release(ctx, handle); err != nil {
		l.finishDestroy(leaseID, false, err.Error())
		return fmt.Errorf("release %s: %w", sandboxID, err)
	}
	l.finishDestroy(leaseID, true, "")
	return nil
}

// removeLocked drops a record and returns its budget, which is only ever reached
// once no runtime object is confirmed to exist.
func (l *Lifecycle) removeLocked(held *allocation, reason ExitReason) {
	delete(l.allocations, held.leaseID)
	if held.sandboxID != "" {
		delete(l.bySandbox, held.sandboxID)
		finished := held.pendingFinishedAt
		if finished.IsZero() {
			finished = l.now()
		}
		l.lastExit[held.sandboxID] = exitRecord{
			reason:     reason,
			exitCode:   held.pendingExitCode,
			codeKnown:  held.pendingCodeKnown,
			at:         l.now(),
			finishedAt: finished,
		}
	}
	l.counters[reason]++
	l.admission.Return(held.grant)
}

// ExitAttributionOf reports why and how a sandbox this node used to hold left it.
//
// A caller that reaches for a sandbox after it is gone gets the runtime's
// attribution rather than an empty answer. The second result is false when the
// sandbox was never held here or its record has aged out.
func (l *Lifecycle) ExitAttributionOf(sandboxID string) (ExitAttribution, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if rec, ok := l.lastExit[sandboxID]; ok {
		return ExitAttribution{
			Reason:        rec.reason,
			ExitCode:      rec.exitCode,
			ExitCodeKnown: rec.codeKnown,
			FinishedAt:    rec.finishedAt,
		}, true
	}
	// A record that is already being destroyed has a decided reason but has not
	// reached removeLocked yet. Reporting it now is what lets the command that
	// discovered the loss name the cause instead of waiting a cleanup round.
	if held, ok := l.bySandbox[sandboxID]; ok && held.pendingReason != "" {
		finished := held.pendingFinishedAt
		if finished.IsZero() {
			finished = l.now()
		}
		return ExitAttribution{
			Reason:        held.pendingReason,
			ExitCode:      held.pendingExitCode,
			ExitCodeKnown: held.pendingCodeKnown,
			FinishedAt:    finished,
		}, true
	}
	return ExitAttribution{}, false
}

// MarkLost decides that a resident sandbox's runtime is gone and hands its
// removal to the cleanup workers.
//
// It is the synchronous half of reconciliation: an exec that discovered the
// container is dead must not wait for the next probe to stop charging it, and
// the caller needs an attribution before it returns. The transition is decided
// under the lock and the enqueue happens outside it, exactly as Sweep does, so
// a full cleanup queue restores the record rather than losing it.
func (l *Lifecycle) MarkLost(sandboxID string, attribution ExitAttribution) bool {
	if attribution.Reason == "" {
		attribution.Reason = ExitExited
	}
	if !attribution.ExitCodeKnown {
		attribution.ExitCode, attribution.ExitCodeKnown = exitCodeForReason(attribution.Reason)
	}
	if attribution.FinishedAt.IsZero() {
		attribution.FinishedAt = l.now()
	}
	l.mu.Lock()
	held, ok := l.bySandbox[sandboxID]
	if !ok || held.state != allocResident {
		l.mu.Unlock()
		return false
	}
	held.state = allocDestroying
	held.pendingReason = attribution.Reason
	held.pendingExitCode = attribution.ExitCode
	held.pendingCodeKnown = attribution.ExitCodeKnown
	held.pendingFinishedAt = attribution.FinishedAt
	leaseID := held.leaseID
	l.mu.Unlock()

	if !l.cleanup.enqueue(leaseID) {
		l.mu.Lock()
		if current, still := l.allocations[leaseID]; still && current.state == allocDestroying {
			current.state = allocResident
			current.pendingReason = ""
			current.pendingExitCode = 0
			current.pendingCodeKnown = false
			current.pendingFinishedAt = time.Time{}
		}
		l.failures++
		l.mu.Unlock()
		return false
	}
	return true
}

// OwnerOf reports the owner a node holds a sandbox for.
//
// Data-plane traffic uses it to refresh the owner's liveness on the node that
// actually observes the work, which is a stronger signal than a heartbeat that
// may be travelling through a control plane with other nodes in its path.
func (l *Lifecycle) OwnerOf(sandboxID string) (string, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	held, ok := l.bySandbox[sandboxID]
	if !ok {
		return "", false
	}
	return held.owner, held.owner != ""
}

// Touch records activity on a sandbox this node holds.
//
// A command that does not pass through Node.Exec, one driven by the sandbox's
// own in-container agent, has no other way to tell the node it is busy. An
// unknown id is ignored: a touch racing a release must not resurrect anything.
func (l *Lifecycle) Touch(sandboxID string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if held, ok := l.bySandbox[sandboxID]; ok {
		held.lastActivity = l.now()
	}
}

// Busy reports whether a command is running in this sandbox.
//
// The controller uses it as the phase hint the design asks for: an in-flight command is
// the one moment a sandbox's page cache is certainly its working set, so it must not be
// asked to release memory then.
func (l *Lifecycle) Busy(sandboxID string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	held, known := l.bySandbox[sandboxID]
	return known && held.inFlight > 0
}

// Running reports whether this node still holds an allocation for the sandbox.
func (l *Lifecycle) Running(sandboxID string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	_, known := l.bySandbox[sandboxID]
	return known
}

// Sweep decides once for every record this node holds.
//
// A sandbox goes for exactly one of three reasons, checked in order: its owner
// is gone, it has outlived the lifetime its create stated, or its caller
// released it. A record that never became a container has no lifetime to
// outlive, so it goes when its owner is gone or its reserve deadline passes.
//
// The pass decides and hands removal to the cleanup workers rather than
// destroying inline, so a runtime that has gone slow cannot hold up every other
// reclamation. What the report names is therefore what the pass decided, and a
// removal that later fails shows up as the node's failure count and draining
// rather than in this report.
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
		leaseID string
		// reportID is the sandbox id a caller recognizes. A record that never
		// became a container has none, so it reports the lease id it was
		// reserved under.
		reportID string
		reason   ExitReason
	}
	var decisions []decision
	var retries []string
	busy := 0
	for _, held := range l.allocations {
		if held.inFlight > 0 {
			busy++
		}
		// A create holds its record in flight while it provisions, and provisioning
		// includes an image pull. Reaping it there destroys the container the create is
		// about to hand back. A resident sandbox is not exempt, because ownership and
		// lifetime stay the right reasons to reclaim it even while it works.
		if held.state == allocCreating && held.inFlight > 0 {
			continue
		}
		reportID := held.sandboxID
		if reportID == "" {
			reportID = held.leaseID
		}
		switch held.state {
		case allocReserved, allocCreating:
			// No container exists, so an owner that died or a reserve that was
			// never followed by a create leaves budget held for nothing.
			if !l.ownerAlive(held.owner) || now.Sub(held.createdAt) >= reserveDeadline {
				decisions = append(decisions, decision{held.leaseID, reportID, ExitReclaimedOrphan})
			}
		case allocResident:
			switch {
			// A resident allocation with no owner is not evidence that its owner is
			// gone. A warm-pool entry is claimed without being relabelled -- labels
			// are immutable -- so a sandbox claimed from the pool reaches this node
			// after a restart with no owner label. Reaping it on owner silence would
			// destroy a live sandbox ninety seconds after every restart; the lifetime
			// it does carry is the backstop that actually applies to it.
			case held.owner != "" && !l.ownerAlive(held.owner):
				decisions = append(decisions, decision{held.leaseID, reportID, ExitReclaimedOrphan})
			case now.Sub(held.createdAt) >= held.lifetime:
				decisions = append(decisions, decision{held.leaseID, reportID, ExitReapedLifetime})
			}
		case allocCleanupFailed:
			// A refused removal is retried on the record's own backoff, so a
			// runtime that keeps refusing is not hammered and one that recovers
			// does not leave the node draining until somebody notices.
			if !now.Before(held.nextCleanupAt) {
				held.state = allocDestroying
				retries = append(retries, held.leaseID)
			}
		}
	}
	l.mu.Unlock()

	// Stable order, so a sweep's effects are reproducible from its inputs. The
	// order is by the id the report names, which is what a caller reads.
	sort.Slice(decisions, func(i, j int) bool { return decisions[i].reportID < decisions[j].reportID })

	report := SweepReport{SkippedBusy: busy}
	for _, d := range decisions {
		l.mu.Lock()
		held, known := l.allocations[d.leaseID]
		if !known {
			l.mu.Unlock()
			continue
		}
		if !held.holdsRuntime() {
			l.removeLocked(held, d.reason)
			l.mu.Unlock()
			report.appendDecision(d.reportID, d.reason)
			continue
		}
		held.state = allocDestroying
		held.pendingReason = d.reason
		held.pendingExitCode, held.pendingCodeKnown = exitCodeForReason(d.reason)
		held.pendingFinishedAt = l.now()
		l.mu.Unlock()

		if !l.cleanup.enqueue(d.leaseID) {
			// The queue is full, so this pass does not carry the decision out.
			// Put the record back rather than leave it looking removed.
			l.mu.Lock()
			if current, still := l.allocations[d.leaseID]; still && current.state == allocDestroying {
				current.state = allocResident
				current.pendingReason = ""
				current.pendingExitCode = 0
				current.pendingCodeKnown = false
				current.pendingFinishedAt = time.Time{}
			}
			l.mu.Unlock()
			report.Failures++
			continue
		}
		report.appendDecision(d.reportID, d.reason)
	}

	for _, leaseID := range retries {
		if !l.cleanup.enqueue(leaseID) {
			l.mu.Lock()
			if current, still := l.allocations[leaseID]; still && current.state == allocDestroying {
				current.state = allocCleanupFailed
			}
			l.mu.Unlock()
		}
	}
	l.retainOwners()
	return report
}

// appendDecision records one decided removal in the report, by reason.
func (r *SweepReport) appendDecision(reportID string, reason ExitReason) {
	switch reason {
	case ExitReapedLifetime:
		r.ReapedLifetime = append(r.ReapedLifetime, reportID)
	case ExitReclaimedOrphan:
		r.ReclaimedOrphan = append(r.ReclaimedOrphan, reportID)
	}
}

// retainOwners drops the heartbeats of owners with nothing left, so a long run
// does not keep one entry per caller that ever existed.
func (l *Lifecycle) retainOwners() {
	l.mu.Lock()
	live := make(map[string]struct{}, len(l.allocations))
	for _, held := range l.allocations {
		live[held.owner] = struct{}{}
	}
	l.mu.Unlock()
	l.admission.RetainOwners(live)
}

// ProbeReport is what one runtime-reconciliation pass saw.
type ProbeReport struct {
	Checked  int
	Gone     int
	OOM      int
	Errors   int
	Failures int
}

type probeCandidate struct {
	leaseID   string
	sandboxID string
	handle    backend.Handle
	backend   backend.Backend
}

// ProbeRuntime asks the runtime about every resident sandbox and reclaims the
// ones that are provably gone.
//
// "Provably" is the whole design. A paused container is healthy, a transitional
// one is unknown, and a probe that failed says nothing about the sandbox. Only
// an explicit exited/oom/terminated answer is acted on. Reading a failed probe
// as death would turn one slow daemon into a fleet-wide loss of live work.
func (l *Lifecycle) ProbeRuntime(ctx context.Context) ProbeReport {
	l.mu.Lock()
	timeout := l.probeTimeout
	workers := l.probeWorkers
	candidates := make([]probeCandidate, 0, len(l.bySandbox))
	for _, held := range l.bySandbox {
		if held.state != allocResident || held.inFlight > 0 || held.backend == nil || held.sandboxID == "" {
			continue
		}
		candidates = append(candidates, probeCandidate{
			leaseID:   held.leaseID,
			sandboxID: held.sandboxID,
			handle:    held.handle,
			backend:   held.backend,
		})
	}
	l.mu.Unlock()

	report := ProbeReport{Checked: len(candidates)}
	if len(candidates) == 0 {
		return report
	}
	if workers < 1 {
		workers = 1
	}

	type outcome struct {
		candidate probeCandidate
		status    string
		detail    backend.ExitDetail
		detailOK  bool
		err       error
	}
	results := make(chan outcome, len(candidates))
	jobs := make(chan probeCandidate)
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for candidate := range jobs {
				probeCtx, cancel := context.WithTimeout(ctx, timeout)
				status, err := candidate.backend.Status(probeCtx, candidate.handle)
				out := outcome{candidate: candidate, status: status, err: err}
				// A gone container's exit code and finish time are what an
				// operator correlates with the node's cgroup and kernel logs.
				// Only ask when the status already says the process is gone:
				// a running sandbox has no exit to explain.
				if err == nil && (status == "oom" || status == "exited" || status == "terminated") {
					if detailer, ok := candidate.backend.(backend.ExitDetailer); ok {
						if detail, detailErr := detailer.ExitDetail(probeCtx, candidate.handle); detailErr == nil {
							out.detail, out.detailOK = detail, true
						}
					}
				}
				cancel()
				results <- out
			}
		}()
	}
	go func() {
		for _, candidate := range candidates {
			jobs <- candidate
		}
		close(jobs)
	}()
	go func() {
		wg.Wait()
		close(results)
	}()

	for result := range results {
		if result.err != nil {
			report.Errors++
			continue
		}
		var reason ExitReason
		switch result.status {
		case "oom":
			reason = ExitOOM
		case "exited":
			reason = ExitExited
		case "terminated":
			reason = ExitRuntimeGone
		default:
			// "running", "paused", "unknown" and anything a future backend adds:
			// a state this probe does not recognize is not permission to destroy.
			continue
		}
		attribution := ExitAttribution{Reason: reason, FinishedAt: l.now()}
		if result.detailOK {
			attribution.ExitCode = result.detail.ExitCode
			attribution.ExitCodeKnown = result.detail.ExitCodeKnown
			if !result.detail.FinishedAt.IsZero() {
				attribution.FinishedAt = result.detail.FinishedAt
			}
			if reason == ExitOOM && (!attribution.ExitCodeKnown || attribution.ExitCode == 0) {
				// The daemon can report OOMKilled without a code when the
				// container was killed by the kernel; the signal is SIGKILL.
				attribution.ExitCode = 137
				attribution.ExitCodeKnown = true
			}
		}
		if !attribution.ExitCodeKnown {
			attribution.ExitCode, attribution.ExitCodeKnown = exitCodeForReason(reason)
		}
		gone, failed := l.markProbedGone(result.candidate, attribution)
		if gone {
			report.Gone++
			if reason == ExitOOM {
				report.OOM++
			}
		}
		if failed {
			report.Failures++
		}
	}

	l.mu.Lock()
	l.probeChecks += int64(report.Checked)
	l.probeGone += int64(report.Gone)
	l.probeErrors += int64(report.Errors)
	l.probeFailures += int64(report.Failures)
	l.mu.Unlock()
	return report
}

// markProbedGone transitions one record after the probe, re-checking under the
// lock that a command did not begin while the probe was in flight. A command
// that started mid-probe means the container is alive enough to serve it; the
// next pass will look again.
func (l *Lifecycle) markProbedGone(candidate probeCandidate, attribution ExitAttribution) (gone, failed bool) {
	l.mu.Lock()
	held, ok := l.bySandbox[candidate.sandboxID]
	if !ok || held.leaseID != candidate.leaseID || held.state != allocResident || held.inFlight > 0 {
		l.mu.Unlock()
		return false, false
	}
	held.state = allocDestroying
	held.pendingReason = attribution.Reason
	held.pendingExitCode = attribution.ExitCode
	held.pendingCodeKnown = attribution.ExitCodeKnown
	held.pendingFinishedAt = attribution.FinishedAt
	l.mu.Unlock()

	if !l.cleanup.enqueue(candidate.leaseID) {
		l.mu.Lock()
		if current, still := l.allocations[candidate.leaseID]; still && current.state == allocDestroying {
			current.state = allocResident
			current.pendingReason = ""
			current.pendingExitCode = 0
			current.pendingCodeKnown = false
			current.pendingFinishedAt = time.Time{}
		}
		l.failures++
		l.mu.Unlock()
		return false, true
	}
	return true, false
}

// ProbeCounters returns the prober's cumulative accounting.
func (l *Lifecycle) ProbeCounters() (checks, gone, failures, errors int64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.probeChecks, l.probeGone, l.probeFailures, l.probeErrors
}

// startProber runs ProbeRuntime on its own cadence until the stop channel is
// closed. It is separate from the sweep's goroutine so a daemon that answers
// slowly delays reconciliation, not owner-silence enforcement.
func (l *Lifecycle) startProber(ctx context.Context) {
	l.mu.Lock()
	l.probeStop = make(chan struct{})
	stop := l.probeStop
	interval := l.probeInterval
	l.mu.Unlock()

	l.probeStopped.Add(1)
	go func() {
		defer l.probeStopped.Done()
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				l.ProbeRuntime(ctx)
			}
		}
	}()
}

// Start runs the sweep on its own cadence until Stop, with the cleanup workers
// that carry out what it decides, and the runtime prober that keeps the ledger
// converged with the daemon.
func (l *Lifecycle) Start(ctx context.Context) {
	l.mu.Lock()
	if l.stop != nil {
		l.mu.Unlock()
		return
	}
	stop := make(chan struct{})
	l.stop = stop
	l.mu.Unlock()

	l.cleanup.Start(ctx)
	l.startProber(ctx)
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
			}
		}
	}()
}

// Stop ends the sweep and the cleanup workers, leaving the sandboxes they were
// watching in place.
func (l *Lifecycle) Stop() {
	l.mu.Lock()
	stop := l.stop
	l.stop = nil
	probeStop := l.probeStop
	l.probeStop = nil
	l.mu.Unlock()
	if probeStop != nil {
		close(probeStop)
		l.probeStopped.Wait()
	}
	if stop != nil {
		close(stop)
		l.stopped.Wait()
	}
	l.cleanup.Stop()
}

// WaitCleanup blocks until the removals a sweep decided have been carried out.
//
// It is for a test that has stopped driving the sweep: the decisions are
// already made, so this waits on them rather than racing them.
func (l *Lifecycle) WaitCleanup() {
	l.cleanup.WaitIdle()
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
	// Destroying is removals in flight, and CleanupFailed is removals a runtime
	// refused. Both are still charged, and Draining is set while any of the
	// latter exists.
	Destroying    int
	CleanupFailed int
	Draining      bool
	Exits         map[ExitReason]int64
	Failures      int64
	// Inventory is what the control plane reconciles quota against. The sequence
	// is generated under the same lock as the list, so a report can never claim
	// a newer sequence than the inventory it carries.
	ReportSeq          uint64
	InventoryReported  bool
	InventoryTruncated bool
	LiveSandboxIDs     []string
}

// maxReportedInventory bounds the sandbox list one report carries. A report
// that would exceed it is marked truncated and the control skips reconciliation
// for that node rather than reading an incomplete list as "these are gone".
const maxReportedInventory = 16384

// Snapshot returns what this node holds. The planes report and never log.
func (l *Lifecycle) Snapshot() LifecycleReport {
	l.mu.Lock()
	defer l.mu.Unlock()
	busy, destroying, failed := 0, 0, 0
	for _, held := range l.bySandbox {
		if held.inFlight > 0 {
			busy++
		}
	}
	for _, held := range l.allocations {
		switch held.state {
		case allocDestroying:
			destroying++
		case allocCleanupFailed:
			failed++
		}
	}
	exits := make(map[ExitReason]int64, len(l.counters))
	for reason, count := range l.counters {
		exits[reason] = count
	}
	l.reportSeq++
	live := make([]string, 0, len(l.bySandbox))
	for sandboxID, held := range l.bySandbox {
		if sandboxID == "" {
			continue
		}
		switch held.state {
		case allocResident, allocDestroying, allocCleanupFailed:
			live = append(live, sandboxID)
		}
		if len(live) > maxReportedInventory {
			break
		}
	}
	truncated := len(live) > maxReportedInventory
	if truncated {
		live = live[:maxReportedInventory]
	}
	return LifecycleReport{
		Resident: len(l.bySandbox), Busy: busy,
		Destroying: destroying, CleanupFailed: failed, Draining: l.draining,
		Exits: exits, Failures: l.failures,
		ReportSeq: l.reportSeq, InventoryReported: true,
		InventoryTruncated: truncated, LiveSandboxIDs: live,
	}
}
