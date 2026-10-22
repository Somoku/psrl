package node

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// testLifetime is the lifetime every adopted sandbox states, unless a test wants
// a different one.
const testLifetime = 100 * time.Second

type fakeBackend struct {
	mu         sync.Mutex
	features   []string
	released   []string
	releaseErr error
	// releaseErrs injects a failure for one sandbox id, so a test can leave one
	// destroy stuck while the rest succeed.
	releaseErrs map[string]error
	// releaseBlock, when set, holds every release until it is closed, so a test
	// can observe that a caller did not wait for one.
	releaseBlock chan struct{}
	// attempts counts the releases per sandbox, so a test can see a retry.
	attempts map[string]int
	// found is what Reconcile reports as already present on this node.
	found        []backend.Reconciled
	reconcileErr error
	// statusFn overrides the status the prober sees. Nil means "running".
	statusFn func(backend.Handle) (string, error)
}

func (f *fakeBackend) Name() string                 { return "fake" }
func (f *fakeBackend) Mode() backend.SchedulingMode { return backend.SchedulingDirect }
func (f *fakeBackend) Capabilities() backend.Capabilities {
	return backend.Capabilities{Features: f.features}
}

func (f *fakeBackend) Create(context.Context, string, backend.Spec, string) (backend.Created, error) {
	return backend.Created{}, nil
}

func (f *fakeBackend) Release(_ context.Context, handle backend.Handle) error {
	f.mu.Lock()
	if f.attempts == nil {
		f.attempts = map[string]int{}
	}
	f.attempts[handle.SandboxID]++
	block := f.releaseBlock
	err, injected := f.releaseErrs[handle.SandboxID]
	fallback := f.releaseErr
	f.mu.Unlock()

	if block != nil {
		<-block
	}
	if injected {
		return err
	}
	if fallback != nil {
		return fallback
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.released = append(f.released, handle.SandboxID)
	return nil
}

// failRelease makes the backend refuse to remove one sandbox.
func (f *fakeBackend) failRelease(sandboxID string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.releaseErrs == nil {
		f.releaseErrs = map[string]error{}
	}
	f.releaseErrs[sandboxID] = fmt.Errorf("daemon refused to remove %s", sandboxID)
}

// allowRelease undoes failRelease.
func (f *fakeBackend) allowRelease(sandboxID string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.releaseErrs, sandboxID)
}

// attemptsOn reports how many times the backend was asked to remove a sandbox.
func (f *fakeBackend) attemptsOn(sandboxID string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.attempts[sandboxID]
}

func (f *fakeBackend) Reconcile(context.Context) ([]backend.Reconciled, error) {
	if f.reconcileErr != nil {
		return nil, f.reconcileErr
	}
	return f.found, nil
}

func (f *fakeBackend) FindByIdempotencyKey(context.Context, string) (backend.Reconciled, bool, error) {
	return backend.Reconciled{}, false, nil
}

func (f *fakeBackend) Status(_ context.Context, handle backend.Handle) (string, error) {
	if f.statusFn != nil {
		return f.statusFn(handle)
	}
	return "running", nil
}

func (f *fakeBackend) releasedIDs() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.released...)
}

func adopt(t *testing.T, life *Lifecycle, gate *Admission, b backend.Backend, id string) {
	t.Helper()
	adoptWithLifetime(t, life, gate, b, id, testLifetime)
}

func adoptWithLifetime(t *testing.T, life *Lifecycle, gate *Admission, b backend.Backend, id string, lifetime time.Duration) {
	t.Helper()
	grant, refusal, ok := gate.Reserve("rollout", "worker", Resources{MemoryMB: 10})
	if !ok {
		t.Fatalf("reserve %s: %q", id, refusal)
	}
	// The same three steps a create takes, so a test states one intent.
	life.Reserve(grant, "worker", "", lifetime)
	if _, ok := life.Claim(grant.LeaseID); !ok {
		t.Fatalf("claim %s: the grant was already spent", id)
	}
	handle := backend.Handle{Backend: b.Name(), SandboxID: id, NodeID: "node-a"}
	if !life.AdoptCreated(grant.LeaseID, handle, b) {
		t.Fatalf("adopt %s: the record was reclaimed before it was adopted", id)
	}
}

func TestAZeroSweepIntervalIsRefused(t *testing.T) {
	if _, err := NewLifecycle("n", nil, 0); err == nil {
		t.Fatal("a sweep with no cadence cannot enforce anything")
	}
}

// startCreate reserves and claims a grant, leaving the record where a create that
// is provisioning holds it. The caller brackets it with BeginCreate when the create
// is meant to be running.
func startCreate(t *testing.T, life *Lifecycle, gate *Admission) string {
	t.Helper()
	grant, refusal, ok := gate.Reserve("rollout", "worker", Resources{MemoryMB: 10})
	if !ok {
		t.Fatalf("reserve: %q", refusal)
	}
	life.Reserve(grant, "worker", "", testLifetime)
	if _, ok := life.Claim(grant.LeaseID); !ok {
		t.Fatalf("claim: the grant was already spent")
	}
	return grant.LeaseID
}

func TestACreateInFlightOutlivesItsReservationDeadline(t *testing.T) {
	// The create is provisioning, which includes an image pull. Elapsed time is not
	// evidence that its record is a grant nobody claimed.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	leaseID := startCreate(t, life, gate)
	done := life.BeginCreate(leaseID)

	clock = clock.Add(reserveDeadline + time.Minute)
	report := life.Sweep(context.Background())

	if report.Released() != 0 {
		t.Fatalf("a create in flight was reaped: %+v", report)
	}
	if report.SkippedBusy != 1 {
		t.Fatalf("skipped %d in-flight creates, want 1", report.SkippedBusy)
	}

	// Once the create ends without adopting, the record is a claim again and the
	// backstop clears it.
	done()
	report = life.Sweep(context.Background())
	if len(report.ReclaimedOrphan) != 1 || report.ReclaimedOrphan[0] != leaseID {
		t.Fatalf("the reservation deadline did not clear an abandoned claim: %+v", report)
	}
}

func TestACreateInFlightSurvivesASilentOwner(t *testing.T) {
	// The owner's renewal is the evidence that a sandbox is abandoned, and a create
	// that is mid-flight is not an abandoned one. Waiting for it to end costs one
	// create, and reclaiming it destroys the container it is about to hand back.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	life.SetOwnerLiveness(func(string) bool { return false })
	leaseID := startCreate(t, life, gate)
	done := life.BeginCreate(leaseID)

	report := life.Sweep(context.Background())

	if report.Released() != 0 {
		t.Fatalf("a create in flight was reclaimed: %+v", report)
	}

	done()
	report = life.Sweep(context.Background())
	if len(report.ReclaimedOrphan) != 1 || report.ReclaimedOrphan[0] != leaseID {
		t.Fatalf("a silent owner's finished create must be reclaimed: %+v", report)
	}
}

func TestAGrantClaimedButAbandonedIsStillReaped(t *testing.T) {
	// A create that ends without adopting or failing its record leaves the record
	// behind, and the reservation deadline is what keeps it from holding budget.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	leaseID := startCreate(t, life, gate)

	clock = clock.Add(reserveDeadline + time.Second)
	report := life.Sweep(context.Background())

	if len(report.ReclaimedOrphan) != 1 || report.ReclaimedOrphan[0] != leaseID {
		t.Fatalf("an abandoned claim must be reaped: %+v", report)
	}
}

func TestABusySandboxIsNeverReclaimed(t *testing.T) {
	// A long command runs in flight, and the lifetime backstop has not been
	// reached, so the sweep must leave it alone and report it as busy.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{}
	adopt(t, life, gate, b, "sb-busy")
	done := life.Begin("sb-busy")
	defer done()

	clock = clock.Add(50 * time.Second)
	report := life.Sweep(context.Background())

	if report.Released() != 0 {
		t.Fatalf("a busy sandbox was reclaimed: %+v", report)
	}
	if report.SkippedBusy != 1 {
		t.Fatalf("skipped %d busy sandboxes, want 1", report.SkippedBusy)
	}
}

func TestANeverIdleSandboxStillEndsAtItsLifetime(t *testing.T) {
	// The lifetime is the backstop that bounds a sandbox which keeps running
	// commands, so it reclaims even a busy one.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{}
	adopt(t, life, gate, b, "sb-busy")
	done := life.Begin("sb-busy")
	defer done()

	clock = clock.Add(testLifetime + 20*time.Second)
	report := life.Sweep(context.Background())

	if len(report.ReapedLifetime) != 1 {
		t.Fatalf("the lifetime backstop did not fire: %+v", report)
	}
}

func TestAShortLifetimeReapsSoonerThanALongOne(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{}
	adoptWithLifetime(t, life, gate, b, "sb-short", 30*time.Second)
	adoptWithLifetime(t, life, gate, b, "sb-long", 600*time.Second)

	clock = clock.Add(60 * time.Second)
	report := life.Sweep(context.Background())

	if len(report.ReapedLifetime) != 1 || report.ReapedLifetime[0] != "sb-short" {
		t.Fatalf("only the short-lived sandbox should be reaped: %+v", report)
	}
}

func TestASandboxWhoseOwnerIsGoneIsReclaimedWithoutWaiting(t *testing.T) {
	// Waiting for the lifetime would hold a node slot for a caller that no longer
	// exists.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	life.SetOwnerLiveness(func(string) bool { return false })
	b := &fakeBackend{}
	adopt(t, life, gate, b, "sb-orphan")
	done := life.Begin("sb-orphan")
	defer done()

	report := life.Sweep(context.Background())

	if len(report.ReclaimedOrphan) != 1 {
		t.Fatalf("an orphan must be reclaimed even while busy: %+v", report)
	}
}

func TestEachSandboxIsReclaimedForExactlyOneReason(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{}
	adopt(t, life, gate, b, "sb-1")

	// Past the lifetime.
	clock = clock.Add(testLifetime + time.Second)
	report := life.Sweep(context.Background())
	life.WaitCleanup()

	if report.Released() != 1 {
		t.Fatalf("released %d times for one sandbox: %+v", report.Released(), report)
	}
	if got := len(b.releasedIDs()); got != 1 {
		t.Fatalf("the backend saw %d releases, want 1", got)
	}
}

func TestReleasingReturnsTheCapacityTheSandboxHeld(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	b := &fakeBackend{}
	adopt(t, life, gate, b, "sb-1")
	before := gate.Headroom("rollout").MemoryMB

	if err := life.Release(context.Background(), "sb-1", ExitReleased); err != nil {
		t.Fatalf("release: %v", err)
	}

	if after := gate.Headroom("rollout").MemoryMB; after != before+10 {
		t.Fatalf("headroom went %d -> %d, want +10MB returned", before, after)
	}
}

func TestAFailedDestructionKeepsTheReservationCharged(t *testing.T) {
	// A container the daemon refused to remove still holds its memory, so
	// returning the reservation would over-commit the node and turn one stuck
	// container into a host OOM.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	b := &fakeBackend{releaseErr: fmt.Errorf("daemon refused")}
	adopt(t, life, gate, b, "sb-stuck")
	before := gate.Headroom("rollout").MemoryMB

	err := life.Release(context.Background(), "sb-stuck", ExitReleased)

	if err == nil {
		t.Fatal("a refused destruction must be reported")
	}
	if after := gate.Headroom("rollout").MemoryMB; after != before {
		t.Fatalf("headroom moved %d -> %d; a stuck container's reservation must stay charged", before, after)
	}
}

func TestReleasingAnUnknownSandboxIsANoOp(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)

	if err := life.Release(context.Background(), "ghost", ExitReleased); err != nil {
		t.Fatalf("a retried release must be safe: %v", err)
	}
}

func TestTheSweepIsReproducibleForTheSameInputs(t *testing.T) {
	gate := mustGate(t, Config{
		Envelope: Resources{MemoryMB: 10000},
		Classes:  map[string]ClassShare{"rollout": {Guaranteed: 1.0}},
		LeaseTTL: time.Minute,
	})
	life, _ := NewLifecycle("node-a", gate, time.Second)
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	gate.SetClock(func() time.Time { return clock })
	b := &fakeBackend{}
	for i := 0; i < 10; i++ {
		adopt(t, life, gate, b, fmt.Sprintf("sb-%02d", i))
	}

	clock = clock.Add(testLifetime + time.Second)
	report := life.Sweep(context.Background())

	for i := 1; i < len(report.ReapedLifetime); i++ {
		if report.ReapedLifetime[i-1] >= report.ReapedLifetime[i] {
			t.Fatalf("a sweep's effects must be in a stable order: %v", report.ReapedLifetime)
		}
	}
}

func TestConcurrentSweepsDecideEachSandboxOnce(t *testing.T) {
	gate := mustGate(t, Config{
		Envelope: Resources{MemoryMB: 10000},
		Classes:  map[string]ClassShare{"rollout": {Guaranteed: 1.0}},
		LeaseTTL: time.Minute,
	})
	life, _ := NewLifecycle("node-a", gate, time.Second)
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	gate.SetClock(func() time.Time { return clock })
	b := &fakeBackend{}
	for i := 0; i < 50; i++ {
		adopt(t, life, gate, b, fmt.Sprintf("sb-%02d", i))
	}
	clock = clock.Add(testLifetime + time.Second)

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			life.Sweep(context.Background())
		}()
	}
	wg.Wait()
	life.WaitCleanup()

	if got := len(b.releasedIDs()); got != 50 {
		t.Fatalf("the backend saw %d releases for 50 sandboxes", got)
	}
	if resident := life.Snapshot().Resident; resident != 0 {
		t.Fatalf("%d sandboxes survived", resident)
	}
}

func BenchmarkSweep(b *testing.B) {
	gate, err := NewAdmission(Config{
		Envelope: Resources{MemoryMB: 1 << 40},
		Classes:  map[string]ClassShare{"rollout": {Guaranteed: 1.0}},
		LeaseTTL: time.Hour,
	})
	if err != nil {
		b.Fatal(err)
	}
	life, err := NewLifecycle("node-a", gate, time.Second)
	if err != nil {
		b.Fatal(err)
	}
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	fake := &fakeBackend{}
	// A production node density, so the sweep is measured where it runs.
	for i := 0; i < 3200; i++ {
		grant, _, _ := gate.Reserve("rollout", "w", Resources{MemoryMB: 1})
		id := fmt.Sprintf("sb-%04d", i)
		life.Reserve(grant, "w", "", testLifetime)
		life.Claim(grant.LeaseID)
		life.AdoptCreated(grant.LeaseID, backend.Handle{Backend: "fake", SandboxID: id}, fake)
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		life.Sweep(ctx)
	}
}

// -- runtime probing ----------------------------------------------------------

func TestTheProberReclaimsAnExitedSandboxAndReturnsItsBudget(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	fake := &fakeBackend{statusFn: func(backend.Handle) (string, error) { return "exited", nil }}
	adopt(t, life, gate, fake, "sb-1")
	before := gate.Headroom("rollout").MemoryMB

	report := life.ProbeRuntime(context.Background())
	if report.Checked != 1 || report.Gone != 1 {
		t.Fatalf("probe report %+v, want one checked and one gone", report)
	}
	life.WaitCleanup()

	if attribution, ok := life.ExitAttributionOf("sb-1"); !ok || attribution.Reason != ExitExited {
		t.Fatalf("exit attribution %+v (known=%v), want %q", attribution, ok, ExitExited)
	}
	if after := gate.Headroom("rollout").MemoryMB; after != before+10 {
		t.Fatalf("headroom went %d -> %d, want +10MB returned after the runtime was gone", before, after)
	}
	if len(fake.releasedIDs()) != 1 {
		t.Fatalf("the backend saw %d releases, want 1", len(fake.releasedIDs()))
	}
}

func TestTheProberAttributesAnOom(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	fake := &fakeBackend{statusFn: func(backend.Handle) (string, error) { return "oom", nil }}
	adopt(t, life, gate, fake, "sb-oom")

	report := life.ProbeRuntime(context.Background())
	if report.OOM != 1 {
		t.Fatalf("probe report %+v, want one OOM", report)
	}
	life.WaitCleanup()
	if attribution, ok := life.ExitAttributionOf("sb-oom"); !ok || attribution.Reason != ExitOOM {
		t.Fatalf("exit attribution %+v (known=%v), want %q", attribution, ok, ExitOOM)
	} else if !attribution.ExitCodeKnown || attribution.ExitCode != 137 {
		t.Fatalf("exit code %d (known=%v), want 137", attribution.ExitCode, attribution.ExitCodeKnown)
	}
}

func TestTheProberIgnoresAPausedSandboxAndAFailedProbe(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	paused := &fakeBackend{statusFn: func(backend.Handle) (string, error) { return "paused", nil }}
	broken := &fakeBackend{statusFn: func(backend.Handle) (string, error) {
		return "", fmt.Errorf("daemon did not answer")
	}}
	adopt(t, life, gate, paused, "sb-paused")
	adopt(t, life, gate, broken, "sb-broken")

	report := life.ProbeRuntime(context.Background())
	if report.Gone != 0 || report.Errors != 1 {
		t.Fatalf("probe report %+v, want no reclamation and one probe error", report)
	}
	life.WaitCleanup()
	if len(paused.releasedIDs()) != 0 || len(broken.releasedIDs()) != 0 {
		t.Fatal("a paused or unreadable sandbox must not be reclaimed")
	}
}

func TestTheProberSkipsABusySandbox(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	fake := &fakeBackend{statusFn: func(backend.Handle) (string, error) { return "exited", nil }}
	adopt(t, life, gate, fake, "sb-busy")

	done := life.Begin("sb-busy")
	defer done()
	report := life.ProbeRuntime(context.Background())
	if report.Checked != 0 || report.Gone != 0 {
		t.Fatalf("probe report %+v, want a busy sandbox skipped", report)
	}
}

func TestReserveReRegistersAnOwnerForgottenBetweenAdmitAndReserve(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	grant, refusal, ok := gate.Reserve("rollout", "worker", Resources{MemoryMB: 10})
	if !ok {
		t.Fatalf("reserve: %q", refusal)
	}
	// A sweep between Admit and Reserve sees no allocation and forgets the
	// owner; the reservation must put it back, or every renewal is ignored and
	// the sandbox is reclaimed as an orphan.
	gate.RetainOwners(map[string]struct{}{})
	life.Reserve(grant, "worker", "", testLifetime)

	if !gate.OwnerAlive("worker") {
		t.Fatal("the reservation must re-register its owner")
	}
}

func TestSnapshotReportsTheOwnersItHolds(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	adopt(t, life, gate, &fakeBackend{}, "sb-owner")

	report := life.Snapshot()

	if len(report.LiveOwners) != 1 || report.LiveOwners[0] != "worker" {
		t.Fatalf("live owners %v, want the owner of the resident sandbox", report.LiveOwners)
	}
}

func TestMarkLostIsIdempotentAndAttributedBeforeCleanup(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	fake := &fakeBackend{releaseBlock: make(chan struct{})}
	adopt(t, life, gate, fake, "sb-lost")

	if !life.MarkLost("sb-lost", ExitAttribution{Reason: ExitExited}) {
		t.Fatal("the first mark must win")
	}
	if life.MarkLost("sb-lost", ExitAttribution{Reason: ExitOOM}) {
		t.Fatal("a second mark must not decide twice")
	}
	if attribution, ok := life.ExitAttributionOf("sb-lost"); !ok || attribution.Reason != ExitExited {
		t.Fatalf("attribution %+v (known=%v), want %q while cleanup is in flight", attribution, ok, ExitExited)
	}
	close(fake.releaseBlock)
	life.WaitCleanup()
	if attribution, ok := life.ExitAttributionOf("sb-lost"); !ok || attribution.Reason != ExitExited {
		t.Fatalf("attribution after cleanup %+v (known=%v), want %q", attribution, ok, ExitExited)
	}
}
