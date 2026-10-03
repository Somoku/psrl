package node

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

type fakeBackend struct {
	mu         sync.Mutex
	features   []string
	released   []string
	paused     []string
	releaseErr error
	pauseErr   error
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
	defer f.mu.Unlock()
	if f.releaseErr != nil {
		return f.releaseErr
	}
	f.released = append(f.released, handle.SandboxID)
	return nil
}

func (f *fakeBackend) Status(context.Context, backend.Handle) (string, error) { return "running", nil }

func (f *fakeBackend) Pause(_ context.Context, handle backend.Handle, _ string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.pauseErr != nil {
		return f.pauseErr
	}
	f.paused = append(f.paused, handle.SandboxID)
	return nil
}

func (f *fakeBackend) Resume(context.Context, backend.Handle) error { return nil }
func (f *fakeBackend) Snapshot(context.Context, backend.Handle, string) (string, error) {
	return "snap", nil
}
func (f *fakeBackend) DeleteSnapshot(context.Context, string) error { return nil }

func (f *fakeBackend) releasedIDs() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.released...)
}

func windows() Windows {
	return Windows{
		PauseWindow:   10 * time.Second,
		ReapWindow:    30 * time.Second,
		Lifetime:      100 * time.Second,
		SweepInterval: time.Second,
	}
}

func adopt(t *testing.T, life *Lifecycle, gate *Admission, b backend.Backend, id string) {
	t.Helper()
	grant, refusal, ok := gate.Admit("rollout", "worker", Resources{MemoryMB: 10})
	if !ok {
		t.Fatalf("admit %s: %q", id, refusal)
	}
	life.Adopt(backend.Handle{Backend: "fake", SandboxID: id, NodeID: "node-a"}, b, grant.LeaseID, "worker", "")
}

func TestInvertedWindowsAreRefused(t *testing.T) {
	bad := windows()
	bad.PauseWindow, bad.ReapWindow = bad.ReapWindow, bad.PauseWindow

	if _, err := NewLifecycle("n", nil, bad); err == nil {
		t.Fatal("a pause window after the reap window destroys before pausing")
	}
}

func TestASweepSlowerThanItsShortestWindowIsRefused(t *testing.T) {
	bad := windows()
	bad.SweepInterval = bad.PauseWindow

	if _, err := NewLifecycle("n", nil, bad); err == nil {
		t.Fatal("a sweep must be faster than the window it enforces")
	}
}

func TestABusySandboxIsNeverReclaimed(t *testing.T) {
	// A backend stamps activity when a command returns, so age alone reads a long
	// command as idle. In-flight is the other half of the idle test.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, windows())
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{features: []string{"freeze"}}
	adopt(t, life, gate, b, "sb-busy")
	done := life.Begin("sb-busy")
	defer done()

	clock = clock.Add(50 * time.Second)
	report := life.Sweep(context.Background())

	if report.Released() != 0 || len(report.Paused) != 0 {
		t.Fatalf("a busy sandbox was reclaimed: %+v", report)
	}
	if report.SkippedBusy != 1 {
		t.Fatalf("skipped %d busy sandboxes, want 1", report.SkippedBusy)
	}
}

func TestACommandThatFinishesMakesTheSandboxIdleAgain(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, windows())
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{features: []string{"freeze"}}
	adopt(t, life, gate, b, "sb-1")
	life.Begin("sb-1")()

	clock = clock.Add(15 * time.Second)
	report := life.Sweep(context.Background())

	if len(report.Paused) != 1 {
		t.Fatalf("an idle sandbox must be paused: %+v", report)
	}
}

func TestAnIdleSandboxIsPausedBeforeItIsDestroyed(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, windows())
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{features: []string{"freeze"}}
	adopt(t, life, gate, b, "sb-1")

	clock = clock.Add(15 * time.Second)
	paused := life.Sweep(context.Background())
	clock = clock.Add(20 * time.Second)
	reaped := life.Sweep(context.Background())

	if len(paused.Paused) != 1 {
		t.Fatalf("first sweep must pause: %+v", paused)
	}
	if len(reaped.ReapedIdle) != 1 {
		t.Fatalf("second sweep must reap: %+v", reaped)
	}
}

func TestAPausedSandboxIsNotPausedAgain(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, windows())
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{features: []string{"freeze"}}
	adopt(t, life, gate, b, "sb-1")

	clock = clock.Add(15 * time.Second)
	life.Sweep(context.Background())
	second := life.Sweep(context.Background())

	if len(second.Paused) != 0 {
		t.Fatalf("a paused sandbox was paused again: %+v", second)
	}
}

func TestANeverIdleSandboxStillEndsAtItsLifetime(t *testing.T) {
	// The backstop is what bounds a sandbox that keeps running commands.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, windows())
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{features: []string{"freeze"}}
	adopt(t, life, gate, b, "sb-busy")
	done := life.Begin("sb-busy")
	defer done()

	clock = clock.Add(120 * time.Second)
	report := life.Sweep(context.Background())

	if len(report.ReapedLifetime) != 1 {
		t.Fatalf("the lifetime backstop did not fire: %+v", report)
	}
}

func TestASandboxWhoseOwnerIsGoneIsReclaimedWithoutWaiting(t *testing.T) {
	// Waiting for the idle window would hold a node slot for a caller that no
	// longer exists.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, windows())
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	life.SetOwnerLiveness(func(string) bool { return false })
	b := &fakeBackend{features: []string{"freeze"}}
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
	life, _ := NewLifecycle("node-a", gate, windows())
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{features: []string{"freeze"}}
	adopt(t, life, gate, b, "sb-1")

	// Past the reap window and past the lifetime at once.
	clock = clock.Add(500 * time.Second)
	report := life.Sweep(context.Background())

	if report.Released() != 1 {
		t.Fatalf("released %d times for one sandbox: %+v", report.Released(), report)
	}
	if got := len(b.releasedIDs()); got != 1 {
		t.Fatalf("the backend saw %d releases, want 1", got)
	}
}

func TestReleasingReturnsTheCapacityTheSandboxHeld(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, windows())
	b := &fakeBackend{features: []string{"freeze"}}
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
	life, _ := NewLifecycle("node-a", gate, windows())
	b := &fakeBackend{features: []string{"freeze"}, releaseErr: fmt.Errorf("daemon refused")}
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
	life, _ := NewLifecycle("node-a", gate, windows())

	if err := life.Release(context.Background(), "ghost", ExitReleased); err != nil {
		t.Fatalf("a retried release must be safe: %v", err)
	}
}

func TestABackendThatCannotPauseIsNotAFailure(t *testing.T) {
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, windows())
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{features: []string{"freeze"}, pauseErr: fmt.Errorf("no pause here")}
	adopt(t, life, gate, b, "sb-1")

	clock = clock.Add(15 * time.Second)
	report := life.Sweep(context.Background())

	if report.Failures != 0 {
		t.Fatalf("a refused pause is not a failure: %+v", report)
	}
	if len(report.Paused) != 0 {
		t.Fatalf("a refused pause must not be reported as paused: %+v", report)
	}
}

func TestTheSweepIsReproducibleForTheSameInputs(t *testing.T) {
	gate := mustGate(t, Config{
		Envelope: Resources{MemoryMB: 10000},
		Classes:  map[string]ClassShare{"rollout": {Guaranteed: 1.0}},
		LeaseTTL: time.Minute,
	})
	life, _ := NewLifecycle("node-a", gate, windows())
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	gate.SetClock(func() time.Time { return clock })
	b := &fakeBackend{features: []string{"freeze"}}
	for i := 0; i < 10; i++ {
		adopt(t, life, gate, b, fmt.Sprintf("sb-%02d", i))
	}

	clock = clock.Add(50 * time.Second)
	report := life.Sweep(context.Background())

	for i := 1; i < len(report.ReapedIdle); i++ {
		if report.ReapedIdle[i-1] >= report.ReapedIdle[i] {
			t.Fatalf("a sweep's effects must be in a stable order: %v", report.ReapedIdle)
		}
	}
}

func TestConcurrentSweepsDecideEachSandboxOnce(t *testing.T) {
	gate := mustGate(t, Config{
		Envelope: Resources{MemoryMB: 10000},
		Classes:  map[string]ClassShare{"rollout": {Guaranteed: 1.0}},
		LeaseTTL: time.Minute,
	})
	life, _ := NewLifecycle("node-a", gate, windows())
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	gate.SetClock(func() time.Time { return clock })
	b := &fakeBackend{features: []string{"freeze"}}
	for i := 0; i < 50; i++ {
		adopt(t, life, gate, b, fmt.Sprintf("sb-%02d", i))
	}
	clock = clock.Add(500 * time.Second)

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			life.Sweep(context.Background())
		}()
	}
	wg.Wait()

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
	life, err := NewLifecycle("node-a", gate, windows())
	if err != nil {
		b.Fatal(err)
	}
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	fake := &fakeBackend{features: []string{"freeze"}}
	// A production node density, so the sweep is measured where it runs.
	for i := 0; i < 3200; i++ {
		grant, _, _ := gate.Admit("rollout", "w", Resources{MemoryMB: 1})
		life.Adopt(backend.Handle{SandboxID: fmt.Sprintf("sb-%04d", i)}, fake, grant.LeaseID, "w", "")
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		life.Sweep(ctx)
	}
}
