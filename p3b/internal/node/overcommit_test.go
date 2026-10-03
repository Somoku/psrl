package node

import (
	"testing"
	"time"
)

// Overcommit exists because the envelope accounts for what sandboxes reserved and
// the machine runs on what they actually use. A fleet whose sandboxes each reserve
// four times their working set leaves every node simultaneously full and idle.
// These cases pin the two things that make the feature safe rather than merely
// denser: it never expands a dimension the kernel cannot reclaim, and it withdraws
// itself as measurement rises rather than at a threshold.

func overcommitConfig() Config {
	return Config{
		Envelope:          Resources{MemoryMB: 1000, CPUMillis: 8000, DiskMB: 10000},
		Classes:           map[string]ClassShare{},
		LeaseTTL:          time.Minute,
		LocalMemCeiling:   0.9,
		Overcommit:        2.0,
		UtilizationTarget: 0.8,
	}
}

func newOvercommitGate(t *testing.T, cfg Config) *Admission {
	t.Helper()
	gate, err := NewAdmission(cfg)
	if err != nil {
		t.Fatalf("NewAdmission: %v", err)
	}
	return gate
}

// -- configuration -------------------------------------------------------------

func TestOvercommitBelowOneIsRefused(t *testing.T) {
	// A factor under one would shrink the envelope, which is not overcommit but a
	// silently smaller node than the operator declared.
	cfg := overcommitConfig()
	cfg.Overcommit = 0.5
	if _, err := NewAdmission(cfg); err == nil {
		t.Fatal("an overcommit factor below one must be refused")
	}
}

func TestOvercommitWithoutATargetIsRefused(t *testing.T) {
	// Without a target there is nothing to withdraw against, so the node would
	// admit at the full multiple no matter how loaded it got.
	cfg := overcommitConfig()
	cfg.UtilizationTarget = 0
	if _, err := NewAdmission(cfg); err == nil {
		t.Fatal("overcommit without a utilization target must be refused")
	}
}

func TestATargetAtOrAboveTheCeilingIsRefused(t *testing.T) {
	// The node would expand admission up to the point where the pressure ceiling
	// starts refusing, which hands out capacity and then rejects the request that
	// wants it.
	cfg := overcommitConfig()
	cfg.UtilizationTarget = 0.9 // equal to LocalMemCeiling
	if _, err := NewAdmission(cfg); err == nil {
		t.Fatal("a utilization target at the memory ceiling must be refused")
	}
	cfg.UtilizationTarget = 0.95 // above LocalMemCeiling
	if _, err := NewAdmission(cfg); err == nil {
		t.Fatal("a utilization target above the memory ceiling must be refused")
	}
}

func TestOvercommitOffIsAValidConfiguration(t *testing.T) {
	// Off is the default, and it must not require a target to be set.
	cfg := overcommitConfig()
	cfg.Overcommit = 0
	cfg.UtilizationTarget = 0
	if _, err := NewAdmission(cfg); err != nil {
		t.Fatalf("overcommit off must be valid: %v", err)
	}
	cfg.Overcommit = 1
	if _, err := NewAdmission(cfg); err != nil {
		t.Fatalf("an overcommit factor of exactly one must be valid: %v", err)
	}
}

// -- the envelope ---------------------------------------------------------------

func TestWithoutOvercommitTheEnvelopeIsTheDeclaredOne(t *testing.T) {
	cfg := overcommitConfig()
	cfg.Overcommit = 0
	cfg.UtilizationTarget = 0
	gate := newOvercommitGate(t, cfg)
	gate.SetPressure(Pressure{MemUsedPct: 0.1})

	if got := gate.Headroom("default").MemoryMB; got != 1000 {
		t.Errorf("headroom with overcommit off is %d, want the declared 1000", got)
	}
	if factor := gate.Overcommitted(); factor != 1 {
		t.Errorf("overcommit factor with the feature off is %.3f, want 1", factor)
	}
}

func TestAnIdleNodeExpandsToTheFullMultiple(t *testing.T) {
	// Measurement just above zero means the node is almost empty, so essentially
	// the whole configured multiple is available.
	gate := newOvercommitGate(t, overcommitConfig())
	gate.SetPressure(Pressure{MemUsedPct: 0.01})

	// headroomFraction = (0.8 - 0.01) / 0.8 = 0.9875; scale = 1 + 1.0*0.9875
	if got := gate.Headroom("default").MemoryMB; got < 1950 || got > 1990 {
		t.Errorf("an almost-idle node reports %d MB, want ~1987 (near the 2x bound)", got)
	}
}

func TestAtTheTargetTheExpansionIsFullyWithdrawn(t *testing.T) {
	// This is the property that makes the feature converge: at the target the node
	// admits exactly its declared envelope, so the overcommit cannot push
	// measurement past the point the operator chose.
	gate := newOvercommitGate(t, overcommitConfig())
	gate.SetPressure(Pressure{MemUsedPct: 0.8})

	if got := gate.Headroom("default").MemoryMB; got != 1000 {
		t.Errorf("at the utilization target headroom is %d, want the declared 1000", got)
	}
	if factor := gate.Overcommitted(); factor != 1 {
		t.Errorf("at the target the overcommit factor is %.3f, want 1", factor)
	}
}

func TestAboveTheTargetNeverShrinksBelowTheDeclaredEnvelope(t *testing.T) {
	// Past the target the formula would go negative. The envelope is a floor: the
	// pressure ceiling is what refuses an overloaded node, not a shrinking
	// envelope, because shrinking would also revoke capacity already granted.
	gate := newOvercommitGate(t, overcommitConfig())
	gate.SetPressure(Pressure{MemUsedPct: 0.88})

	if got := gate.Headroom("default").MemoryMB; got != 1000 {
		t.Errorf("above the target headroom is %d, want the declared 1000 as a floor", got)
	}
}

func TestTheExpansionWithdrawsProportionally(t *testing.T) {
	// A step function would make a node near the target oscillate between
	// over-admitting and refusing. Halfway to the target must give half the
	// expansion.
	gate := newOvercommitGate(t, overcommitConfig())
	gate.SetPressure(Pressure{MemUsedPct: 0.4})

	// headroomFraction = (0.8 - 0.4) / 0.8 = 0.5; scale = 1 + 1.0*0.5 = 1.5
	if got := gate.Headroom("default").MemoryMB; got != 1500 {
		t.Errorf("halfway to the target headroom is %d, want 1500 (half the expansion)", got)
	}
}

func TestAnUnreadableMeasurementDoesNotExpand(t *testing.T) {
	// A pressure reader that found no cgroup hierarchy reports zero. Reading that
	// as "idle, expand freely" would overcommit a node whose usage is unknown,
	// which is the one case where the envelope is all the information there is.
	gate := newOvercommitGate(t, overcommitConfig())
	gate.SetPressure(Pressure{MemUsedPct: 0})

	if got := gate.Headroom("default").MemoryMB; got != 1000 {
		t.Errorf("with no measurement headroom is %d, want the declared 1000", got)
	}
	if factor := gate.Overcommitted(); factor != 1 {
		t.Errorf("with no measurement the overcommit factor is %.3f, want 1", factor)
	}
}

// -- what must never expand -----------------------------------------------------

func TestOnlyMemoryExpands(t *testing.T) {
	// CPU is compressible: a cgroup limit throttles and the workload runs slower.
	// Memory is not: exceeding it is an OOM kill. Devices cannot be shared by two
	// sandboxes that each believe they own one, and disk is not reclaimed under
	// pressure. So memory is the only dimension measurement buys density in.
	cfg := overcommitConfig()
	cfg.GPUIndices = []int32{0, 1}
	gate := newOvercommitGate(t, cfg)
	gate.SetPressure(Pressure{MemUsedPct: 0.01})

	headroom := gate.Headroom("default")
	if headroom.MemoryMB <= 1000 {
		t.Errorf("memory did not expand: %d MB", headroom.MemoryMB)
	}
	if headroom.CPUMillis != 8000 {
		t.Errorf("CPU headroom is %d, want the declared 8000: throttling is not an OOM kill, "+
			"but expanding it here would double-count what the scheduler already oversubscribes",
			headroom.CPUMillis)
	}
	if headroom.DiskMB != 10000 {
		t.Errorf("disk headroom is %d, want the declared 10000: disk is not reclaimed under pressure",
			headroom.DiskMB)
	}
	if headroom.GPUCount != 2 {
		t.Errorf("device headroom is %d, want the declared 2: a device cannot be shared", headroom.GPUCount)
	}
}

// -- admission against the expanded envelope ------------------------------------

func TestOvercommitAdmitsWhatTheDeclaredEnvelopeWouldRefuse(t *testing.T) {
	// The whole point: a request that does not fit the declared envelope is
	// admitted when measurement says the reservations ahead of it are overstated.
	gate := newOvercommitGate(t, overcommitConfig())
	gate.SetPressure(Pressure{MemUsedPct: 0.1})

	// Fill the declared envelope exactly.
	if _, _, ok := gate.Admit("default", "owner-a", Resources{MemoryMB: 1000}); !ok {
		t.Fatal("the first request must fit the declared envelope")
	}
	// This one only fits because of the expansion.
	if _, refusal, ok := gate.Admit("default", "owner-b", Resources{MemoryMB: 500}); !ok {
		t.Fatalf("a request beyond the declared envelope was refused (%s) while the node measured 10%% used; "+
			"this is exactly the density overcommit exists to recover", refusal)
	}
}

func TestOvercommitStillRefusesBeyondTheExpandedEnvelope(t *testing.T) {
	// The expansion is a bound, not an absence of one.
	gate := newOvercommitGate(t, overcommitConfig())
	gate.SetPressure(Pressure{MemUsedPct: 0.1})

	if _, refusal, ok := gate.Admit("default", "owner", Resources{MemoryMB: 5000}); ok {
		t.Fatalf("a request of 5000 MB was admitted against a ~1987 MB expanded envelope (refusal %q)", refusal)
	}
}

func TestThePressureCeilingStillRefusesAnOvercommittedNode(t *testing.T) {
	// Overcommit raises the accounting bound; it does not weaken the measured one.
	// A node past its memory ceiling refuses regardless of how much expanded
	// envelope is left, which is what stops the feature from driving a node to OOM.
	gate := newOvercommitGate(t, overcommitConfig())
	gate.SetPressure(Pressure{MemUsedPct: 0.95}) // above LocalMemCeiling of 0.9

	if _, refusal, ok := gate.Admit("default", "owner", Resources{MemoryMB: 1}); ok {
		t.Fatal("a node above its memory pressure ceiling must refuse even under overcommit")
	} else if refusal != RefusedPressure {
		t.Errorf("refusal is %q, want %q", refusal, RefusedPressure)
	}
}

func TestClassCeilingsStayRelativeToTheDeclaredEnvelope(t *testing.T) {
	// A ceiling of 0.5 means "half the machine", which is a statement about the
	// machine rather than about the current overcommit factor. Scaling it with the
	// expansion would let one class take the whole declared envelope while
	// nominally capped at half of it.
	cfg := overcommitConfig()
	cfg.Classes = map[string]ClassShare{"capped": {Guaranteed: 0.25, Max: 0.5}}
	gate := newOvercommitGate(t, cfg)
	gate.SetPressure(Pressure{MemUsedPct: 0.01})

	if got := gate.Headroom("capped").MemoryMB; got != 500 {
		t.Errorf("a class capped at half the machine reports %d MB, want 500 even under a 2x expansion", got)
	}
}

// -- reporting ------------------------------------------------------------------

func TestTheSnapshotReportsBothEnvelopes(t *testing.T) {
	// An operator tuning the factor needs the resolved number, not the configured
	// bound: the bound is in the config file and the resolved value is the only
	// thing that explains a refusal.
	gate := newOvercommitGate(t, overcommitConfig())
	gate.SetPressure(Pressure{MemUsedPct: 0.4})

	report := gate.Snapshot()
	if report.Envelope.MemoryMB != 1000 {
		t.Errorf("declared envelope reports %d, want 1000", report.Envelope.MemoryMB)
	}
	if report.EffectiveEnvelope.MemoryMB != 1500 {
		t.Errorf("effective envelope reports %d, want 1500", report.EffectiveEnvelope.MemoryMB)
	}
	if report.Overcommit < 1.49 || report.Overcommit > 1.51 {
		t.Errorf("overcommit factor reports %.3f, want ~1.5", report.Overcommit)
	}
}

func TestTheReportedFactorTracksMeasurement(t *testing.T) {
	// The factor is not static configuration: it is what measurement allowed at
	// the moment of the report, which is what makes it diagnostic.
	gate := newOvercommitGate(t, overcommitConfig())

	gate.SetPressure(Pressure{MemUsedPct: 0.2})
	loaded := gate.Overcommitted()
	gate.SetPressure(Pressure{MemUsedPct: 0.6})
	heavier := gate.Overcommitted()

	if !(heavier < loaded) {
		t.Errorf("the factor at 60%% used (%.3f) must be below the factor at 20%% used (%.3f): "+
			"the expansion withdraws as the node fills", heavier, loaded)
	}
}
