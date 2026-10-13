package node

import (
	"sync"
	"testing"
	"time"
)

func envelope() Config {
	return Config{
		Envelope: Resources{MemoryMB: 1000, CPUMillis: 10000, DiskMB: 10000},
		Classes: map[string]ClassShare{
			"rollout": {Guaranteed: 0.70},
			"grader":  {Guaranteed: 0.20},
		},
		GPUIndices:      []int32{0, 1, 2, 3},
		LocalCPUCeiling: 0.95,
		LocalMemCeiling: 0.90,
		LeaseTTL:        time.Minute,
	}
}

func mustGate(t *testing.T, cfg Config) *Admission {
	t.Helper()
	gate, err := NewAdmission(cfg)
	if err != nil {
		t.Fatalf("new admission: %v", err)
	}
	return gate
}

func TestGuaranteesAboveTheEnvelopeAreRefused(t *testing.T) {
	cfg := envelope()
	cfg.Classes["extra"] = ClassShare{Guaranteed: 0.5}

	if _, err := NewAdmission(cfg); err == nil {
		t.Fatal("guarantees above one leave a guarantee unsatisfiable")
	}
}

func TestAPressureCeilingOutsideZeroToOneIsRefused(t *testing.T) {
	cfg := envelope()
	cfg.LocalMemCeiling = 1.5

	if _, err := NewAdmission(cfg); err == nil {
		t.Fatal("a pressure ceiling is a fraction")
	}
}

func TestARequestInsideTheEnvelopeIsAdmitted(t *testing.T) {
	gate := mustGate(t, envelope())

	grant, refusal, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 100})

	if !ok {
		t.Fatalf("refused with %q", refusal)
	}
	if grant.LeaseID == "" {
		t.Fatal("an admitted request must carry a lease")
	}
}

func TestMeasuredPressureRefusesEvenWhenTheEnvelopeWouldAllowIt(t *testing.T) {
	// This is the whole reason the node keeps final authority: placement compares
	// a headroom reported seconds ago, and a machine already swapping must refuse.
	gate := mustGate(t, envelope())
	gate.SetPressure(Pressure{MemUsedPct: 0.97})

	_, refusal, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 1})

	if ok {
		t.Fatal("a node under memory pressure must refuse")
	}
	if refusal != RefusedPressure {
		t.Fatalf("refusal is %q, want %q", refusal, RefusedPressure)
	}
}

func TestADrainingNodeRefusesEverything(t *testing.T) {
	// It holds containers it could not destroy, so its returned memory is not free.
	gate := mustGate(t, envelope())
	gate.SetDraining(true)

	_, refusal, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 1})

	if ok || refusal != RefusedDraining {
		t.Fatalf("a draining node admitted work: ok=%v refusal=%q", ok, refusal)
	}
}

func TestAnIdleClassReservesNothing(t *testing.T) {
	gate := mustGate(t, envelope())

	if headroom := gate.Headroom("rollout"); headroom.MemoryMB != 1000 {
		t.Fatalf("an idle node offers everything: got %dMB", headroom.MemoryMB)
	}
}

func TestAQueuedClassHoldsItsGuaranteeBack(t *testing.T) {
	gate := mustGate(t, envelope())
	gate.Enqueue("grader")

	if headroom := gate.Headroom("rollout"); headroom.MemoryMB != 800 {
		t.Fatalf("a queued guarantee must be held back: got %dMB, want 800MB", headroom.MemoryMB)
	}
}

func TestDequeueingReleasesTheReservation(t *testing.T) {
	gate := mustGate(t, envelope())
	gate.Enqueue("grader")
	gate.Dequeue("grader")

	if headroom := gate.Headroom("rollout"); headroom.MemoryMB != 1000 {
		t.Fatalf("a withdrawn request must stop reserving: got %dMB", headroom.MemoryMB)
	}
}

func TestAClassAtItsCeilingIsRefusedDistinctly(t *testing.T) {
	// An operator acts differently on a class at its own ceiling than on a full node.
	cfg := envelope()
	cfg.Classes["grader"] = ClassShare{Guaranteed: 0.2, Max: 0.25}
	gate := mustGate(t, cfg)
	if _, _, ok := gate.Reserve("grader", "w", Resources{MemoryMB: 250}); !ok {
		t.Fatal("the first request fits the ceiling")
	}

	_, refusal, ok := gate.Reserve("grader", "w", Resources{MemoryMB: 10})

	if ok || refusal != RefusedClassCeiling {
		t.Fatalf("ok=%v refusal=%q, want %q", ok, refusal, RefusedClassCeiling)
	}
}

func TestDevicesAreHandedOutAsDistinctIndices(t *testing.T) {
	// Two sandboxes granted the same index would both drive one device.
	gate := mustGate(t, envelope())

	seen := map[int32]bool{}
	for i := 0; i < 4; i++ {
		grant, refusal, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 1, GPUCount: 1})
		if !ok {
			t.Fatalf("grant %d refused with %q", i, refusal)
		}
		for _, index := range grant.GPUIndices {
			if seen[index] {
				t.Fatalf("device %d handed out twice", index)
			}
			seen[index] = true
		}
	}

	if _, refusal, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 1, GPUCount: 1}); ok {
		t.Fatal("a fifth device request must be refused on a four-device node")
	} else if refusal != RefusedDevices {
		t.Fatalf("refusal is %q, want %q", refusal, RefusedDevices)
	}
}

func TestReturningAGrantReturnsItsDevices(t *testing.T) {
	gate := mustGate(t, envelope())
	grant, _, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 1, GPUCount: 4})
	if !ok {
		t.Fatal("the first request takes every device")
	}

	gate.Return(grant)

	if _, _, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 1, GPUCount: 4}); !ok {
		t.Fatal("returned devices must be reusable")
	}
}

func TestAGrantIsReturnedExactlyOnceByItsOwner(t *testing.T) {
	// Return is not idempotent by itself: the guarantee that a grant comes back
	// once is the lifecycle's, which removes a record once and returns its budget
	// there. This pins what a single Return does.
	gate := mustGate(t, envelope())
	grant, _, _ := gate.Reserve("rollout", "w", Resources{MemoryMB: 400, GPUCount: 2})

	gate.Return(grant)

	report := gate.Snapshot()
	if report.FreeGPUs != 4 {
		t.Fatalf("free devices %d, want 4", report.FreeGPUs)
	}
	if headroom := gate.Headroom("rollout"); headroom.MemoryMB != 1000 {
		t.Fatalf("headroom %dMB, want 1000MB", headroom.MemoryMB)
	}
}

func TestASilentOwnerDoesNotFreeItsOwnBudget(t *testing.T) {
	// Physical budget comes back only through Return, after the lifecycle has
	// confirmed the runtime object is gone. An owner going silent is not that: the
	// container it holds is still consuming memory.
	gate := mustGate(t, envelope())
	clock := time.Now()
	gate.SetClock(func() time.Time { return clock })
	grant, _, ok := gate.Reserve("rollout", "dead", Resources{MemoryMB: 500, GPUCount: 2})
	if !ok {
		t.Fatal("reserve")
	}

	clock = clock.Add(2 * time.Minute)

	report := gate.Snapshot()
	if report.Granted.MemoryMB != 500 || report.FreeGPUs != 2 {
		t.Fatalf("a silent owner must not free budget by itself: %+v", report)
	}
	if gate.OwnerAlive("dead") {
		t.Fatal("a silent owner must read as gone, so the lifecycle reclaims it")
	}

	gate.Return(grant)
	if after := gate.Snapshot(); after.Granted.MemoryMB != 0 || after.FreeGPUs != 4 {
		t.Fatalf("Return must free the grant: %+v", after)
	}
}

func TestARenewedOwnerStaysAliveAndCharged(t *testing.T) {
	gate := mustGate(t, envelope())
	clock := time.Now()
	gate.SetClock(func() time.Time { return clock })
	if _, _, ok := gate.Reserve("rollout", "live", Resources{MemoryMB: 500}); !ok {
		t.Fatal("reserve")
	}

	for i := 0; i < 5; i++ {
		clock = clock.Add(30 * time.Second)
		gate.Renew("live")
	}

	if !gate.OwnerAlive("live") {
		t.Fatal("a renewed owner must read as alive")
	}
	if gate.Snapshot().Granted.MemoryMB != 500 {
		t.Fatal("a renewed owner's grant must stay charged")
	}
}

func TestOwnerAliveTracksReservationAndRenewal(t *testing.T) {
	gate := mustGate(t, envelope())
	clock := time.Now()
	gate.SetClock(func() time.Time { return clock })

	if gate.OwnerAlive("worker") {
		t.Fatal("an owner this node never reserved for is not its business")
	}
	if _, _, ok := gate.Reserve("rollout", "worker", Resources{MemoryMB: 100}); !ok {
		t.Fatal("reserve")
	}
	if !gate.OwnerAlive("worker") {
		t.Fatal("an owner that just reserved must read as alive")
	}

	// Silence past the lease TTL: the node has to treat the owner as gone even
	// before the next sweep reclaims its records.
	clock = clock.Add(2 * time.Minute)
	if gate.OwnerAlive("worker") {
		t.Fatal("an owner silent past the lease TTL must read as gone")
	}

	gate.Renew("worker")
	if !gate.OwnerAlive("worker") {
		t.Fatal("a renewed owner must read as alive again")
	}
}

func TestARenewalCannotResurrectAnOwnerWithNothingLeft(t *testing.T) {
	// RetainOwners forgets the heartbeats of owners with no records left. A
	// heartbeat that arrives afterwards belongs to a caller whose work the node
	// already reclaimed, and it must not put that accounting back.
	gate := mustGate(t, envelope())
	clock := time.Now()
	gate.SetClock(func() time.Time { return clock })
	if _, _, ok := gate.Reserve("rollout", "gone", Resources{MemoryMB: 100}); !ok {
		t.Fatal("reserve")
	}

	gate.RetainOwners(map[string]struct{}{})
	gate.Renew("gone")

	if gate.OwnerAlive("gone") {
		t.Fatal("a renewal must not resurrect an owner with nothing left")
	}
}

func TestEveryClassAppearsInTheReportIncludingDefault(t *testing.T) {
	// A request carries a class either way, and one absent from the report would
	// be compared against nothing by placement.
	gate := mustGate(t, envelope())

	headroom := gate.ClassHeadroom()

	for _, name := range []string{"rollout", "grader", "default"} {
		if _, present := headroom[name]; !present {
			t.Fatalf("class %q is missing from the node report: %v", name, headroom)
		}
	}
}

func TestAZeroDimensionIsNotComparedAgainstTheEnvelope(t *testing.T) {
	gate := mustGate(t, envelope())

	if _, refusal, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 10}); !ok {
		t.Fatalf("a request naming only memory was refused with %q", refusal)
	}
}

func TestConcurrentAdmitNeverOverfillsTheNode(t *testing.T) {
	// Admission is on the create path, so this is the property that matters at
	// burst: whatever the interleaving, grants stay inside the envelope and no
	// device is handed out twice.
	gate := mustGate(t, Config{
		Envelope:   Resources{MemoryMB: 1000},
		Classes:    map[string]ClassShare{"rollout": {Guaranteed: 1.0}},
		GPUIndices: []int32{0, 1, 2, 3},
		LeaseTTL:   time.Minute,
	})

	var wg sync.WaitGroup
	var mu sync.Mutex
	granted := 0
	devices := map[int32]int{}
	for i := 0; i < 400; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			grant, _, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 10, GPUCount: 1})
			if !ok {
				return
			}
			mu.Lock()
			granted++
			for _, index := range grant.GPUIndices {
				devices[index]++
			}
			mu.Unlock()
		}()
	}
	wg.Wait()

	// Four devices bound the run before memory does.
	if granted != 4 {
		t.Fatalf("granted %d, want 4 (one per device)", granted)
	}
	for index, count := range devices {
		if count != 1 {
			t.Fatalf("device %d handed out %d times", index, count)
		}
	}
}

func BenchmarkAdmitRelease(b *testing.B) {
	gate, err := NewAdmission(Config{
		Envelope: Resources{MemoryMB: 1 << 40, CPUMillis: 1 << 40},
		Classes:  map[string]ClassShare{"rollout": {Guaranteed: 1.0}},
		LeaseTTL: time.Hour,
	})
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if grant, _, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 1}); ok {
				gate.Return(grant)
			}
		}
	})
}
