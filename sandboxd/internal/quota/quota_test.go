package quota

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

func fleet() Config {
	return Config{
		Total: Amount{MemoryMB: 1000, CPUMillis: 10000, Sandboxes: 100},
		Classes: map[string]ClassShare{
			"rollout": {Guaranteed: 0.70},
			"grader":  {Guaranteed: 0.20},
			"prepare": {Guaranteed: 0.10},
		},
		LeaseTTL: time.Minute,
	}
}

func mustLedger(t *testing.T, cfg Config) *Ledger {
	t.Helper()
	ledger, err := New(cfg)
	if err != nil {
		t.Fatalf("new ledger: %v", err)
	}
	return ledger
}

func TestGuaranteesAboveTheFleetAreRefused(t *testing.T) {
	cfg := fleet()
	cfg.Classes["extra"] = ClassShare{Guaranteed: 0.5}

	if _, err := New(cfg); err == nil {
		t.Fatal("guarantees summing above one must be refused, or a guarantee is unsatisfiable")
	}
}

func TestACeilingBelowItsGuaranteeIsRefused(t *testing.T) {
	cfg := fleet()
	cfg.Classes["rollout"] = ClassShare{Guaranteed: 0.7, Max: 0.5}

	if _, err := New(cfg); err == nil {
		t.Fatal("a ceiling under the guarantee makes the guarantee unreachable")
	}
}

func TestAnIdleClassReservesNothing(t *testing.T) {
	// Reserving against a class with no queued demand leaves the fleet idle and
	// full at the same time, which is the whole reason the rule exists.
	ledger := mustLedger(t, fleet())

	headroom := ledger.Headroom("rollout")

	if headroom.MemoryMB != 1000 {
		t.Fatalf("an idle fleet offers everything, got %dMB of 1000MB", headroom.MemoryMB)
	}
}

func TestAQueuedClassHoldsItsGuaranteeBackFromABorrower(t *testing.T) {
	ledger := mustLedger(t, fleet())
	ledger.Enqueue("grader", Amount{MemoryMB: 200})

	headroom := ledger.Headroom("rollout")

	// The grader's unmet 20% is out of reach while it waits.
	if headroom.MemoryMB != 800 {
		t.Fatalf("a queued guarantee must be held back: got %dMB, want 800MB", headroom.MemoryMB)
	}
}

func TestAClassCanBorrowWhatNobodyIsWaitingFor(t *testing.T) {
	ledger := mustLedger(t, fleet())

	// 70% guaranteed, but nothing else is queued, so the whole fleet is reachable.
	admitted, err := ledger.Acquire("g1", "rollout", "worker-1", Amount{MemoryMB: 950})
	if err != nil || !admitted {
		t.Fatalf("borrowing idle capacity must be admitted: admitted=%v err=%v", admitted, err)
	}
}

func TestACeilingCapsAClassOnAnIdleFleet(t *testing.T) {
	cfg := fleet()
	cfg.Classes["prepare"] = ClassShare{Guaranteed: 0.10, Max: 0.25}
	ledger := mustLedger(t, cfg)

	if headroom := ledger.Headroom("prepare"); headroom.MemoryMB != 250 {
		t.Fatalf("a ceiling caps an idle fleet: got %dMB, want 250MB", headroom.MemoryMB)
	}
}

func TestADuplicateGrantIdIsRefused(t *testing.T) {
	ledger := mustLedger(t, fleet())
	if _, err := ledger.Acquire("g1", "rollout", "w", Amount{MemoryMB: 10}); err != nil {
		t.Fatalf("first acquire: %v", err)
	}

	if _, err := ledger.Acquire("g1", "rollout", "w", Amount{MemoryMB: 10}); err == nil {
		t.Fatal("a reused grant id would double-count the same sandbox")
	}
}

func TestReleasingTwiceDoesNotInflateTheFleet(t *testing.T) {
	ledger := mustLedger(t, fleet())
	if _, err := ledger.Acquire("g1", "rollout", "w", Amount{MemoryMB: 400}); err != nil {
		t.Fatalf("acquire: %v", err)
	}

	ledger.Release("g1")
	ledger.Release("g1")

	if headroom := ledger.Headroom("rollout"); headroom.MemoryMB != 1000 {
		t.Fatalf("a repeated release must be a no-op: got %dMB, want 1000MB", headroom.MemoryMB)
	}
}

func TestAnOwnerThatStopsRenewingLosesItsGrants(t *testing.T) {
	// Owner silence is the only evidence the ledger has. Age of the grant is not:
	// a long-lived sandbox looks exactly like an abandoned one by age alone.
	ledger := mustLedger(t, fleet())
	clock := time.Now()
	ledger.SetClock(func() time.Time { return clock })
	if _, err := ledger.Acquire("g1", "rollout", "dead-worker", Amount{MemoryMB: 500}); err != nil {
		t.Fatalf("acquire: %v", err)
	}

	clock = clock.Add(2 * time.Minute)
	ledger.Expire()

	if headroom := ledger.Headroom("rollout"); headroom.MemoryMB != 1000 {
		t.Fatalf("a dead owner's grant must be reclaimed: got %dMB, want 1000MB", headroom.MemoryMB)
	}
}

func TestARenewingOwnerKeepsItsGrants(t *testing.T) {
	ledger := mustLedger(t, fleet())
	clock := time.Now()
	ledger.SetClock(func() time.Time { return clock })
	if _, err := ledger.Acquire("g1", "rollout", "live-worker", Amount{MemoryMB: 500}); err != nil {
		t.Fatalf("acquire: %v", err)
	}

	for i := 0; i < 5; i++ {
		clock = clock.Add(30 * time.Second)
		ledger.Renew("live-worker")
	}
	ledger.Expire()

	if headroom := ledger.Headroom("rollout"); headroom.MemoryMB != 500 {
		t.Fatalf("a renewed grant must be held: got %dMB, want 500MB", headroom.MemoryMB)
	}
}

func TestARequestLargerThanTheHeadroomIsRefusedRatherThanQueuedSilently(t *testing.T) {
	ledger := mustLedger(t, fleet())

	admitted, err := ledger.Acquire("g1", "rollout", "w", Amount{MemoryMB: 2000})

	if err != nil {
		t.Fatalf("an oversized request is a refusal, not an error: %v", err)
	}
	if admitted {
		t.Fatal("a request above the fleet must not be admitted")
	}
}

func TestAZeroDimensionIsNotComparedAgainstTheLimit(t *testing.T) {
	// Zero means the caller stated no requirement there, which is not a request
	// for nothing and must not be filtered on.
	ledger := mustLedger(t, fleet())

	admitted, err := ledger.Acquire("g1", "rollout", "w", Amount{MemoryMB: 100})

	if err != nil || !admitted {
		t.Fatalf("a request that names only memory must be admitted: admitted=%v err=%v", admitted, err)
	}
}

func TestReleasingAnOwnerReturnsEverythingItHeld(t *testing.T) {
	ledger := mustLedger(t, fleet())
	for i := 0; i < 3; i++ {
		if _, err := ledger.Acquire(fmt.Sprintf("g%d", i), "rollout", "w", Amount{MemoryMB: 100}); err != nil {
			t.Fatalf("acquire %d: %v", i, err)
		}
	}

	if released := ledger.ReleaseOwner("w"); released != 3 {
		t.Fatalf("released %d grants, want 3", released)
	}
	if headroom := ledger.Headroom("rollout"); headroom.MemoryMB != 1000 {
		t.Fatalf("owner release must return everything: got %dMB", headroom.MemoryMB)
	}
}

func TestTheReportCoversEveryDeclaredClass(t *testing.T) {
	ledger := mustLedger(t, fleet())
	if _, err := ledger.Acquire("g1", "grader", "w", Amount{MemoryMB: 50}); err != nil {
		t.Fatalf("acquire: %v", err)
	}

	report := ledger.Report()

	if len(report) != 3 {
		t.Fatalf("report covers %d classes, want 3", len(report))
	}
	if report["grader"].Granted.MemoryMB != 50 {
		t.Fatalf("grader granted %dMB, want 50MB", report["grader"].Granted.MemoryMB)
	}
}

func TestConcurrentAcquireNeverExceedsTheFleet(t *testing.T) {
	// The ledger is on the create path, so this is the property that matters at
	// burst: whatever interleaving occurs, the sum of grants stays inside Total.
	ledger := mustLedger(t, Config{
		Total:    Amount{MemoryMB: 1000},
		Classes:  map[string]ClassShare{"rollout": {Guaranteed: 1.0}},
		LeaseTTL: time.Minute,
	})

	var wg sync.WaitGroup
	var mu sync.Mutex
	granted := 0
	for i := 0; i < 500; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			ok, err := ledger.Acquire(fmt.Sprintf("g%d", i), "rollout", "w", Amount{MemoryMB: 10})
			if err == nil && ok {
				mu.Lock()
				granted++
				mu.Unlock()
			}
		}(i)
	}
	wg.Wait()

	if granted != 100 {
		t.Fatalf("admitted %d x 10MB against a 1000MB fleet, want exactly 100", granted)
	}
	if used := ledger.Report()["rollout"].Granted.MemoryMB; used != 1000 {
		t.Fatalf("ledger accounts %dMB, want 1000MB", used)
	}
}

func BenchmarkAcquireRelease(b *testing.B) {
	ledger, err := New(Config{
		Total:    Amount{MemoryMB: 1 << 40, CPUMillis: 1 << 40},
		Classes:  map[string]ClassShare{"rollout": {Guaranteed: 1.0}},
		LeaseTTL: time.Hour,
	})
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			id := fmt.Sprintf("g%d-%p", i, pb)
			if ok, err := ledger.Acquire(id, "rollout", "w", Amount{MemoryMB: 1}); err == nil && ok {
				ledger.Release(id)
			}
			i++
		}
	})
}
