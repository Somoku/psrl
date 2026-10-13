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

func (f *fakeBackend) Status(context.Context, backend.Handle) (string, error) { return "running", nil }

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
