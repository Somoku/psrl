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
		// Off here on purpose. These tests isolate the guarantee, the ceiling and the
		// pressure rules against a whole envelope; the admission reserve takes a share of
		// every one of them, so leaving it on would move every number in the file. It has
		// its own test.
		AdmissionReserve: 0,
	}
}

func mustGate(t *testing.T, cfg Config) *Admission {
	t.Helper()
	gate, err := NewAdmission(cfg)
	if err != nil {
		t.Fatalf("new admission: %v", err)
	}
	// A gate in these tests stands for a node whose sampler is running and whose
	// measured domain is empty. Without a sample every measured ceiling would refuse,
	// which is the production behaviour for a node that cannot measure itself and not
	// what these tests are about.
	gate.SetPressure(Pressure{SampledAt: time.Now(), Valid: true})
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
	gate.SetPressure(Pressure{MemUsedPct: 0.97, Valid: true})

	_, refusal, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 1})

	if ok {
		t.Fatal("a node under memory pressure must refuse")
	}
	if refusal != RefusedPressure {
		t.Fatalf("refusal is %q, want %q", refusal, RefusedPressure)
	}
}

func TestADrainingNodeStillAdmitsWhatFits(t *testing.T) {
	// Draining reports a container the node could not destroy. That container is
	// still charged, so the room read here already excludes its memory, and
	// refusing the whole node would cost the fleet a machine for one container.
	gate := mustGate(t, envelope())
	gate.SetDraining(true)

	_, refusal, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 1})

	if !ok {
		t.Fatalf("a draining node refused work that fits: refusal=%q", refusal)
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

func TestAnInvalidSampleRefusesWhenACeilingIsConfigured(t *testing.T) {
	// A sampler that stopped, or cgroup files that vanished, is not an idle node.
	// The gate has no way to tell those apart from measurement, so it refuses.
	gate := mustGate(t, envelope())
	gate.SetPressure(Pressure{Valid: false})

	_, refusal, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 1})

	if ok {
		t.Fatal("a node that cannot measure itself must refuse rather than assume it is idle")
	}
	if refusal != RefusedPressure {
		t.Fatalf("refusal is %q, want %q", refusal, RefusedPressure)
	}
}

func TestAnInvalidSampleDoesNotRefuseAnUnmeasuredNode(t *testing.T) {
	// Without a configured ceiling nothing is being measured, so the sample cannot
	// make the decision: this is a node that admits on its envelope alone.
	cfg := envelope()
	cfg.LocalCPUCeiling = 0
	cfg.LocalMemCeiling = 0
	gate := mustGate(t, cfg)
	gate.SetPressure(Pressure{Valid: false})

	if _, _, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 1}); !ok {
		t.Fatal("a node with no measured ceiling must admit on its envelope")
	}
}

func TestAPoolReservationReducesEveryClassHeadroom(t *testing.T) {
	// A warm-pool entry is memory the node has already spent, so no class may be
	// promised it. Charging it to one class would let prefetching starve the work it
	// exists to accelerate, so it is subtracted from all of them.
	gate := mustGate(t, envelope())
	before := gate.Headroom("rollout").MemoryMB

	if !gate.AcquirePool(Resources{MemoryMB: 200}) {
		t.Fatal("an entry inside the envelope must be acquired")
	}

	if got := gate.Headroom("rollout").MemoryMB; got != before-200 {
		t.Errorf("rollout headroom: got %d, want %d", got, before-200)
	}
	if got := gate.Headroom("grader").MemoryMB; got != gate.Headroom("rollout").MemoryMB {
		t.Errorf("every class must lose the same reservation, got rollout=%d grader=%d",
			gate.Headroom("rollout").MemoryMB, got)
	}
	if got := gate.Pooled().MemoryMB; got != 200 {
		t.Errorf("pooled: got %d, want 200", got)
	}

	gate.ReleasePool(Resources{MemoryMB: 200})
	if got := gate.Headroom("rollout").MemoryMB; got != before {
		t.Errorf("headroom after release: got %d, want the original %d", got, before)
	}
	if got := gate.Pooled().MemoryMB; got != 0 {
		t.Errorf("pooled after release: got %d, want 0", got)
	}
}

func TestPoolAcquireUsesTheDeclaredEnvelopeNotTheOvercommittedOne(t *testing.T) {
	// Overcommit is about reservations overstating use. A pool entry has no use to
	// overstate: it exists before any caller is charged, so it is bounded by the
	// envelope the operator declared.
	cfg := envelope()
	cfg.Overcommit = 2
	cfg.UtilizationTarget = 0.75
	gate := mustGate(t, cfg)
	gate.SetPressure(Pressure{MemUsedPct: 0.01, Valid: true})

	if gate.Headroom("rollout").MemoryMB <= 1000 {
		t.Fatalf("the test needs an expanded envelope, got %d", gate.Headroom("rollout").MemoryMB)
	}
	if gate.AcquirePool(Resources{MemoryMB: 1200}) {
		t.Error("a pool entry above the declared envelope must be refused even when overcommit allows it")
	}
}

func TestReleasingMorePoolThanWasAcquiredDoesNotCreateHeadroom(t *testing.T) {
	// A double release would let the node promise the same memory twice, so the
	// reservation is corrected rather than carried negative.
	gate := mustGate(t, envelope())
	before := gate.Headroom("rollout").MemoryMB

	gate.ReleasePool(Resources{MemoryMB: 100})

	if got := gate.Pooled().MemoryMB; got != 0 {
		t.Errorf("pooled: got %d, want 0", got)
	}
	if got := gate.Headroom("rollout").MemoryMB; got != before {
		t.Errorf("headroom: got %d, want %d", got, before)
	}
}

func TestMeasuredUsageAboveTheReservationsBoundsTheNode(t *testing.T) {
	// Reservations are an estimate. When the pages have already taken more than the
	// estimate says, the measurement is what the node has left -- not the promise.
	gate := mustGate(t, envelope())
	gate.SetPressure(Pressure{MemUsedPct: 0.1, Valid: true})
	if _, _, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 100}); !ok {
		t.Fatal("the test needs a reservation")
	}
	if got := gate.Headroom("rollout").MemoryMB; got != 900 {
		t.Fatalf("headroom with a 100MB reservation: got %d, want 900", got)
	}

	gate.SetObserved(Resources{MemoryMB: 700}, true)

	if got := gate.Headroom("rollout").MemoryMB; got != 300 {
		t.Errorf("headroom with 700MB measured: got %d, want 300 (the measurement bounds it)", got)
	}
}

func TestAnUnusableObservationLeavesTheLedgerOnItsReservations(t *testing.T) {
	// A reading that cannot be trusted must not make the node admit more, and it must not
	// make it refuse either: the reservations are the only information there is.
	gate := mustGate(t, envelope())

	gate.SetObserved(Resources{MemoryMB: 900}, false)

	if got := gate.Headroom("rollout").MemoryMB; got != 1000 {
		t.Errorf("headroom: got %d, want the full 1000", got)
	}
}

func TestAdjustingAClassChargeMovesItsHeadroom(t *testing.T) {
	// This is how a reservation follows what its sandboxes use: up when one turns out to
	// need more, down when a quiet batch gives some back.
	gate := mustGate(t, envelope())
	if _, _, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 200}); !ok {
		t.Fatal("the test needs a reservation")
	}
	before := gate.Headroom("rollout").MemoryMB

	gate.Adjust("rollout", Resources{MemoryMB: 300})
	if got := gate.Headroom("rollout").MemoryMB; got != before-300 {
		t.Errorf("after growing, headroom is %d, want %d", got, before-300)
	}

	gate.Adjust("rollout", Resources{MemoryMB: -300})
	if got := gate.Headroom("rollout").MemoryMB; got != before {
		t.Errorf("after shrinking back, headroom is %d, want %d", got, before)
	}
}

func TestAShrinkBelowZeroIsCorrectedRatherThanCarried(t *testing.T) {
	// A release without a matching charge would let the node promise the same memory
	// twice, so the class's charge is floored at zero.
	gate := mustGate(t, envelope())
	if _, _, ok := gate.Reserve("rollout", "w", Resources{MemoryMB: 100}); !ok {
		t.Fatal("the test needs a reservation")
	}

	gate.Adjust("rollout", Resources{MemoryMB: -500})

	if got := gate.Snapshot().Granted.MemoryMB; got != 0 {
		t.Errorf("granted: got %d, want 0", got)
	}
}

func TestTheAdmissionReserveIsHeldForAWaveThatHasNotArrived(t *testing.T) {
	// A share of the envelope no running sandbox may spend. Without it the first wave
	// takes the machine and the second wave -- grading, or the next batch -- has nowhere
	// to land until an episode finishes, which is what turned into trajectories kept but
	// ungraded.
	cfg := envelope()
	cfg.AdmissionReserve = 0.10
	gate := mustGate(t, cfg)

	if got := gate.Headroom("rollout").MemoryMB; got != 900 {
		t.Fatalf("headroom with a 10%% reserve: got %dMB, want 900MB", got)
	}
	// The reserve is not a class guarantee: it is taken after every other deduction, so
	// it costs density and nothing else, and it is there for whoever arrives next.
	if got := gate.Headroom("grader").MemoryMB; got != 900 {
		t.Errorf("the reserve is not class-specific: got %dMB for grader", got)
	}
}

func TestExpectedGrowthIsChargedBeforeItHappens(t *testing.T) {
	// The cost of a sandbox is not what it uses now, it is what it will have used by the
	// time anyone notices. Charging only the present is how a fleet admits a burst at a
	// price that stops being true before the burst has finished starting.
	gate := mustGate(t, envelope())
	gate.Reserve("rollout", "w", Resources{MemoryMB: 100, CPUMillis: 500})

	if got := gate.Headroom("rollout").MemoryMB; got != 900 {
		t.Fatalf("headroom before growth is known: got %dMB, want 900MB", got)
	}

	// Ten sandboxes are running and each is growing by 5MB before the next sample.
	gate.SetGrowth("rollout", Resources{MemoryMB: 5}, Resources{MemoryMB: 50})
	if got := gate.Headroom("rollout").MemoryMB; got != 850 {
		t.Errorf("headroom with growth reserved: got %dMB, want 850MB", got)
	}
	if got := gate.GrowthPerSandbox("rollout").MemoryMB; got != 5 {
		t.Errorf("the candidate's own growth: got %dMB, want 5MB", got)
	}
}
