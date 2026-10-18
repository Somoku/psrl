package node

import (
	"testing"
	"time"
)

// observedGate returns a gate whose envelope expands under overcommit, with a clock a
// test drives.
func observedGate(t *testing.T) (*Admission, *time.Time) {
	t.Helper()
	clock := time.Unix(1700000000, 0)
	gate, err := NewAdmission(Config{
		Envelope:          Resources{MemoryMB: 4096, CPUMillis: 100000},
		Classes:           map[string]ClassShare{"rollout": {Guaranteed: 1.0}},
		LeaseTTL:          time.Minute,
		Overcommit:        2,
		UtilizationTarget: 0.75,
		ObservedMaxAge:    15 * time.Second,
	})
	if err != nil {
		t.Fatalf("gate: %v", err)
	}
	gate.SetClock(func() time.Time { return clock })
	return gate, &clock
}

func TestAMeasurementThatGoesStaleStopsExpandingTheEnvelope(t *testing.T) {
	// Overcommit is the claim that reservations overstate use. A node that has not read
	// its sandbox domain for a while cannot make that claim, so the expansion is
	// withdrawn and the node admits exactly its declared envelope -- the same shape as
	// an unusable pressure sample.
	gate, clock := observedGate(t)
	gate.SetPressure(Pressure{MemUsedPct: 0.10, SampledAt: *clock, Valid: true})
	gate.SetObserved(Resources{MemoryMB: 512, CPUMillis: 1000}, true)

	if factor := gate.Overcommitted(); factor <= 1.0 {
		t.Fatalf("a fresh measurement and an idle node must expand: got %.3f", factor)
	}

	*clock = clock.Add(20 * time.Second)
	gate.SetPressure(Pressure{MemUsedPct: 0.10, SampledAt: *clock, Valid: true})

	if factor := gate.Overcommitted(); factor != 1.0 {
		t.Errorf("a measurement 20s old must withdraw the expansion: got %.3f", factor)
	}
	if gate.Snapshot().ObservedFresh {
		t.Error("the report must say the reading is no longer fresh")
	}

	// And a fresh reading restores it, because the domain became readable again.
	gate.SetObserved(Resources{MemoryMB: 512, CPUMillis: 1000}, true)
	if factor := gate.Overcommitted(); factor <= 1.0 {
		t.Errorf("a fresh reading must restore the expansion: got %.3f", factor)
	}
	if !gate.Snapshot().ObservedFresh {
		t.Error("the report must say the reading is fresh again")
	}
}

func TestAStaleMeasurementStaysAFloorRatherThanAssumingUsageFell(t *testing.T) {
	// The unknown points both ways: sandboxes released since the last reading would make
	// usage lower, sandboxes that grew would make it higher, and only the second can cost
	// an episode. So the last reading stays charged.
	gate, clock := observedGate(t)
	gate.SetPressure(Pressure{MemUsedPct: 0.10, SampledAt: *clock, Valid: true})
	gate.SetObserved(Resources{MemoryMB: 3000, CPUMillis: 80000}, true)

	before := gate.Headroom("rollout")
	*clock = clock.Add(20 * time.Second)
	after := gate.Headroom("rollout")

	if after.MemoryMB >= before.MemoryMB {
		t.Errorf("headroom must not grow when the measurement goes stale: %d then %d",
			before.MemoryMB, after.MemoryMB)
	}
	// The floor is still there: reservations alone are 0 here, so headroom is the
	// envelope minus the measurement, not the whole envelope.
	if got := after.MemoryMB; got != 4096-3000 {
		t.Errorf("headroom without the floor would be the whole envelope: got %d", got)
	}
}

func TestAnInvalidReadingIsNotAFloorAndWithdrawsExpansion(t *testing.T) {
	gate, clock := observedGate(t)
	gate.SetPressure(Pressure{MemUsedPct: 0.10, SampledAt: *clock, Valid: true})
	gate.SetObserved(Resources{MemoryMB: 3000}, true)
	gate.SetObserved(Resources{MemoryMB: 3000}, false)

	if gate.Snapshot().ObservedFresh {
		t.Error("an invalid reading must not be reported as fresh")
	}
	if factor := gate.Overcommitted(); factor != 1.0 {
		t.Errorf("an invalid reading must withdraw the expansion: got %.3f", factor)
	}
	if got := gate.Headroom("rollout").MemoryMB; got != 4096 {
		t.Errorf("an invalid reading must not be charged as a floor: got %d", got)
	}
}

func TestANodeThatNeverHadAMeasurementKeepsPressureDrivenOvercommit(t *testing.T) {
	// The elastic controller is optional. Requiring a measurement before expanding would
	// withdraw overcommit from every deployment that never enabled it, which is a
	// regression dressed as a safety check.
	gate, clock := observedGate(t)
	gate.SetPressure(Pressure{MemUsedPct: 0.10, SampledAt: *clock, Valid: true})

	if factor := gate.Overcommitted(); factor <= 1.0 {
		t.Errorf("no measurement yet must not withdraw the expansion: got %.3f", factor)
	}
	if !gate.Snapshot().ObservedFresh {
		t.Error("a node that has never been measured has no stale reading to report")
	}
}
