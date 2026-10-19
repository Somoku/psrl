package node

import (
	"testing"
	"time"
)

// The two dimensions are overcommitted by the same mechanism -- a declared base expanded
// by measurement -- but by different signals, because their failure modes are not the same.
// Memory that is over-subscribed risks an OOM kill, which destroys a sandbox; CPU that is
// over-subscribed risks throttling, which slows a tool call. These tests pin that the CPU
// term answers to throttling and to nothing else.

func gate(t *testing.T, overcommit, target float64) *Admission {
	t.Helper()
	gate, err := NewAdmission(Config{
		Envelope:          Resources{MemoryMB: 100000, CPUMillis: 180000},
		Classes:           map[string]ClassShare{"rollout": {Guaranteed: 0.9}, "grader": {Guaranteed: 0.1}},
		LeaseTTL:          time.Minute,
		Overcommit:        1.5,
		UtilizationTarget: 0.75,
		CPUOvercommit:     overcommit,
		ThrottleTarget:    target,
	})
	if err != nil {
		t.Fatalf("gate: %v", err)
	}
	gate.SetPressure(Pressure{SampledAt: time.Now(), Valid: true, MemUsedPct: 0.10})
	return gate
}

func TestAnIdleNodeExpandsItsCPUPacking(t *testing.T) {
	// Nothing throttled, so the whole expansion is available: 180 cores sells as 360.
	g := gate(t, 2.0, 0.10)
	g.SetThrottle(0)

	if got := g.Snapshot().EffectiveEnvelope.CPUMillis; got != 360000 {
		t.Errorf("effective CPU: got %d millis, want 360000 (2x the 180000 declared)", got)
	}
}

func TestThrottlingWithdrawsTheCPUPackingProportionally(t *testing.T) {
	// Half the target: half the expansion. Proportional rather than a step, so the node
	// converges instead of oscillating between over-subscribing and refusing on alternate
	// samples.
	g := gate(t, 2.0, 0.10)
	g.SetThrottle(0.05)

	if got := g.Snapshot().EffectiveEnvelope.CPUMillis; got != 270000 {
		t.Errorf("effective CPU at half the throttle target: got %d, want 270000", got)
	}
}

func TestThrottlingAtTheTargetLeavesOnlyTheDeclaredEnvelope(t *testing.T) {
	g := gate(t, 2.0, 0.10)
	g.SetThrottle(0.10)

	if got := g.Snapshot().EffectiveEnvelope.CPUMillis; got != 180000 {
		t.Errorf("effective CPU at the throttle target: got %d, want the declared 180000", got)
	}
}

func TestAnUnmeasuredNodeDoesNotClaimThereIsNoThrottling(t *testing.T) {
	// The dangerous direction. A node that has not read its domain, or has stopped
	// reading it, must fall back to the declared envelope: "the reservations overstate
	// demand" is precisely the claim a stale or missing measurement cannot support.
	g := gate(t, 2.0, 0.10)

	if got := g.Snapshot().EffectiveEnvelope.CPUMillis; got != 180000 {
		t.Errorf("effective CPU with no throttling reading: got %d, want the declared 180000", got)
	}
	g.SetThrottle(0)
	g.elasticThrottleStaleForTest()
	if got := g.Snapshot().EffectiveEnvelope.CPUMillis; got != 180000 {
		t.Errorf("effective CPU with a stale reading: got %d, want the declared 180000", got)
	}
}

func TestCPUOvercommitDefaultsToTheDocumentedMultiple(t *testing.T) {
	// The default is on, because the failure mode is throttling and throttling is the
	// feedback that withdraws it. A node that starts throttling converges back to its
	// declared envelope on its own, so there is no state in which the default is unsafe and
	// unobserved.
	g := gate(t, 0, 0)
	g.SetThrottle(0)

	want := int64(float64(180000) * DefaultCPUOvercommit)
	if got := g.Snapshot().EffectiveEnvelope.CPUMillis; got != want {
		t.Errorf("effective CPU: got %d, want %d (%vx by default)", got, want, DefaultCPUOvercommit)
	}
	if DefaultCPUOvercommit <= 1 || DefaultCPUOvercommit > 4 {
		t.Errorf("default CPU overcommit %v is outside a sane range", DefaultCPUOvercommit)
	}
}

// elasticThrottleStaleForTest ages the throttle reading past its freshness bound.
func (a *Admission) elasticThrottleStaleForTest() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.throttleAt = time.Now().Add(-2 * a.throttleMaxAge())
}
