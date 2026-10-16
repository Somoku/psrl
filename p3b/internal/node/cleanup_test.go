package node

import (
	"context"
	"testing"
	"time"
)

// fastCleanup is a cleanup configuration a test can drive in milliseconds.
func fastCleanup() CleanupConfig {
	return CleanupConfig{
		Workers:        2,
		QueueDepth:     8,
		Backoff:        100 * time.Millisecond,
		MaxBackoff:     200 * time.Millisecond,
		AttemptTimeout: time.Second,
	}
}

func TestTheSweepDoesNotWaitForDestruction(t *testing.T) {
	// A runtime that has gone slow must not hold up every other reclamation, so
	// the sweep hands the removal off and returns while it is still in flight.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycleWithCleanup("node-a", gate, time.Second, fastCleanup())
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{releaseBlock: make(chan struct{})}
	adopt(t, life, gate, b, "sb-slow")

	clock = clock.Add(testLifetime + time.Second)
	report := life.Sweep(context.Background())

	if report.Released() != 1 {
		t.Fatalf("the sweep decided %d removals, want 1: %+v", report.Released(), report)
	}
	if got := len(b.releasedIDs()); got != 0 {
		t.Fatalf("the sweep returned only after %d removals finished, so it waited on the runtime", got)
	}

	close(b.releaseBlock)
	life.WaitCleanup()
	if got := len(b.releasedIDs()); got != 1 {
		t.Fatalf("the backend saw %d removals after the worker was let go, want 1", got)
	}
	if snapshot := gate.Snapshot(); snapshot.Granted.MemoryMB != 0 {
		t.Fatalf("the budget came back only after the removal: %+v", snapshot.Granted)
	}
}

func TestACleanupFailureDrainsAndARetryClearsIt(t *testing.T) {
	// A container the runtime refused to remove still holds its memory, so the
	// record keeps its charge rather than returning budget the host cannot back.
	// The flag is the report of that condition, and it must clear on its own once
	// the runtime will remove the container.
	gate := mustGate(t, envelope())
	cfg := fastCleanup()
	life, _ := NewLifecycleWithCleanup("node-a", gate, time.Second, cfg)
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{}
	b.failRelease("sb-stuck")
	adopt(t, life, gate, b, "sb-stuck")
	charged := gate.Snapshot().Granted.MemoryMB

	clock = clock.Add(testLifetime + time.Second)
	life.Sweep(context.Background())
	life.WaitCleanup()

	if !gate.Draining() {
		t.Fatal("a refused removal must be reported as draining: its memory is still in use")
	}
	if snapshot := life.Snapshot(); snapshot.CleanupFailed != 1 {
		t.Fatalf("cleanup failures %d, want 1: %+v", snapshot.CleanupFailed, snapshot)
	}
	if granted := gate.Snapshot().Granted.MemoryMB; granted != charged {
		t.Fatalf("a refused removal kept %dMB, want %dMB still charged", granted, charged)
	}

	// The backoff has not elapsed, so a sweep in between must not hammer the
	// runtime with a retry it is going to refuse.
	life.Sweep(context.Background())
	life.WaitCleanup()
	if attempts := b.attemptsOn("sb-stuck"); attempts != 1 {
		t.Fatalf("the removal was attempted %d times before its backoff elapsed, want 1", attempts)
	}

	// The runtime recovers, the backoff elapses, and the retry clears it.
	b.allowRelease("sb-stuck")
	clock = clock.Add(cfg.Backoff)
	life.Sweep(context.Background())
	life.WaitCleanup()

	if attempts := b.attemptsOn("sb-stuck"); attempts != 2 {
		t.Fatalf("the removal was attempted %d times, want a retry after the backoff", attempts)
	}
	if gate.Draining() {
		t.Fatal("draining must clear once the container is actually gone")
	}
	if snapshot := gate.Snapshot(); snapshot.Granted.MemoryMB != 0 {
		t.Fatalf("the retry must return the budget, got %+v", snapshot.Granted)
	}
	if resident := life.Snapshot().Resident; resident != 0 {
		t.Fatalf("resident %d after a successful retry, want 0", resident)
	}
}

func TestDestructionIsAttemptedOncePerSandbox(t *testing.T) {
	// Two sweeps racing over the same record, and a caller's release racing them
	// both, must still issue one removal: a second one would be a second
	// destruction of one container.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycleWithCleanup("node-a", gate, time.Second, fastCleanup())
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{}
	adopt(t, life, gate, b, "sb-1")

	clock = clock.Add(testLifetime + time.Second)
	done := make(chan struct{})
	go func() {
		defer close(done)
		life.Sweep(context.Background())
	}()
	life.Sweep(context.Background())
	// A caller releasing what the sweep is already removing is a no-op.
	_ = life.Release(context.Background(), "sb-1", ExitReleased)
	<-done
	life.WaitCleanup()

	if attempts := b.attemptsOn("sb-1"); attempts != 1 {
		t.Fatalf("the backend was asked to remove one sandbox %d times", attempts)
	}
	if snapshot := gate.Snapshot(); snapshot.Granted.MemoryMB != 0 {
		t.Fatalf("the budget must come back exactly once: %+v", snapshot.Granted)
	}
}

func TestTheCleanupQueueIsBounded(t *testing.T) {
	// The queue is a bound rather than a promise: a sweep that finds more work
	// than the queue holds leaves the rest for its next pass instead of growing
	// without limit or dropping the record.
	gate := mustGate(t, envelope())
	cfg := fastCleanup()
	cfg.QueueDepth = 1
	cfg.Workers = 1
	life, _ := NewLifecycleWithCleanup("node-a", gate, time.Second, cfg)
	clock := time.Now()
	life.SetClock(func() time.Time { return clock })
	b := &fakeBackend{releaseBlock: make(chan struct{})}
	adopt(t, life, gate, b, "sb-1")
	adopt(t, life, gate, b, "sb-2")
	adopt(t, life, gate, b, "sb-3")

	clock = clock.Add(testLifetime + time.Second)
	report := life.Sweep(context.Background())

	if report.Failures == 0 {
		t.Fatalf("a queue of one cannot have accepted three removals: %+v", report)
	}
	if report.Released()+report.Failures != 3 {
		t.Fatalf("every decision must be either accepted or counted as a failure: %+v", report)
	}
	close(b.releaseBlock)
	life.WaitCleanup()

	// What the bound held back is still charged and still owned, so the next
	// sweep decides it again rather than the container being forgotten.
	if resident := life.Snapshot().Resident; resident == 0 {
		t.Fatal("a record the queue could not take must stay owned, not vanish")
	}
	life.Sweep(context.Background())
	life.WaitCleanup()
	if resident := life.Snapshot().Resident; resident != 0 {
		t.Fatalf("resident %d after the retry sweep, want 0", resident)
	}
}
