package node

import "testing"

func footprintConfig() FootprintConfig {
	return FootprintConfig{
		MinReserve:      512,
		Percentile:      95,
		GrowthMargin:    0.2,
		ShrinkAfter:     3,
		ShrinkStep:      0.5,
		MaxGrowthFactor: DefaultMaxGrowthFactor,
	}
}

func TestAFreshClassIsChargedTheMinimum(t *testing.T) {
	// The first batch of a step is admitted before anything has been measured, so the
	// minimum is what it is charged. Charging the declared maximum until measurement
	// arrives would reintroduce the reservation this exists to remove.
	footprint := NewFootprint(footprintConfig(), 30000)

	if got := footprint.Reserve(); got != 512 {
		t.Errorf("reserve: got %d, want the 512MiB minimum", got)
	}
}

func TestTheReserveFollowsTheObservedP95Immediately(t *testing.T) {
	// Fast grow: one sample that says the class needs more is enough to move, because
	// under-reserving is the direction that loses sandboxes. What is bounded is the size
	// of the step, not whether it happens.
	footprint := NewFootprint(footprintConfig(), 30000)
	sample := []int64{1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 20000}

	before := footprint.Reserve()
	footprint.Observe(sample)
	first := footprint.Reserve()
	if first <= before {
		t.Fatalf("the reserve must rise on the first sample: %d then %d", before, first)
	}
	if got := float64(first) / float64(before); got > footprintConfig().MaxGrowthFactor+0.01 {
		t.Errorf("first step grew by %.2fx, want at most %.2fx", got, footprintConfig().MaxGrowthFactor)
	}

	// p95 of ten samples is the ninth, 1000, plus the 20% margin: reached within a few
	// samples, which is still fast against an episode.
	for i := 0; i < 8; i++ {
		footprint.Observe(sample)
	}
	if got := footprint.Reserve(); got != 1200 {
		t.Errorf("reserve: got %d, want 1200 (1000 x 1.2)", got)
	}
}

func TestTheReserveComesDownSlowlyAndOnlyAfterQuietSamples(t *testing.T) {
	// Slow shrink: a reservation that followed every quiet sample down would oscillate
	// with each tool call, and every oscillation is a re-admission decision.
	footprint := NewFootprint(footprintConfig(), 30000)
	// Several samples to get up there first: one sample may only double.
	for i := 0; i < 8; i++ {
		footprint.Observe(repeat(4000, 10))
	}
	high := footprint.Reserve()
	if high != 4800 {
		t.Fatalf("the test needs a high reserve, got %d", high)
	}

	// Two quiet samples are not enough.
	footprint.Observe(repeat(500, 10))
	footprint.Observe(repeat(500, 10))
	if got := footprint.Reserve(); got != high {
		t.Errorf("reserve after two quiet samples: got %d, want the unchanged %d", got, high)
	}

	// The third allows a step, and the step is limited rather than jumping to the target.
	footprint.Observe(repeat(500, 10))
	after := footprint.Reserve()
	if after >= high {
		t.Errorf("reserve after three quiet samples: got %d, want below %d", after, high)
	}
	if after < 600 {
		t.Errorf("reserve fell to %d in one step, want no lower than the 500 x 1.2 target", after)
	}
}

func TestTheReserveNeverExceedsTheLargestDeclaredMaximum(t *testing.T) {
	// A reservation above the kernel bound would charge the ledger more than the sandbox
	// can ever be allowed to use.
	footprint := NewFootprint(footprintConfig(), 2048)

	// Ten samples, because a single one may only double: the ceiling is reached, not
	// overshot, and the growth is still fast against an episode.
	for i := 0; i < 10; i++ {
		footprint.Observe(repeat(10000, 10))
	}

	if got := footprint.Reserve(); got != 2048 {
		t.Errorf("reserve: got %d, want the 2048MiB maximum", got)
	}
}

func TestASingleSampleCannotRaiseTheReserveByMoreThanAFactor(t *testing.T) {
	// The bound is on magnitude, not on direction. A class first priced before its
	// sandboxes warmed up can measure orders of magnitude higher on the next sample, and
	// a ledger that follows that in one step has already admitted the previous burst at a
	// price that no longer exists -- which is how a fleet ends up refusing everything
	// until a whole wave of episodes finishes.
	footprint := NewFootprint(footprintConfig(), 30000)

	before := footprint.Reserve()
	footprint.Observe(repeat(30000, 10))

	after := footprint.Reserve()
	if after <= before {
		t.Fatalf("the reserve must rise toward the measurement: %d then %d", before, after)
	}
	if got := float64(after) / float64(before); got > footprintConfig().MaxGrowthFactor+0.01 {
		t.Errorf("grew by %.2fx in one sample, want at most %.2fx",
			got, footprintConfig().MaxGrowthFactor)
	}

	// And it keeps climbing, so the direction is unchanged: a few samples later it is at
	// the measurement rather than stranded at a bounded fraction of it.
	for i := 0; i < 8; i++ {
		footprint.Observe(repeat(30000, 10))
	}
	if got := footprint.Reserve(); got != 30000 {
		t.Errorf("reserve after converging: got %d, want 30000", got)
	}
}

func TestTheReserveNeverFallsBelowTheMinimum(t *testing.T) {
	footprint := NewFootprint(footprintConfig(), 30000)

	for i := 0; i < 20; i++ {
		footprint.Observe([]int64{1, 1, 1})
	}

	if got := footprint.Reserve(); got != 512 {
		t.Errorf("reserve: got %d, want the 512MiB minimum", got)
	}
}

func TestAnEmptySampleChangesNothing(t *testing.T) {
	// A class with no running sandbox is between batches, not idle: dropping to the
	// minimum there would under-reserve the batch that follows.
	footprint := NewFootprint(footprintConfig(), 30000)
	footprint.Observe(repeat(4000, 10))
	before := footprint.Reserve()

	footprint.Observe(nil)

	if got := footprint.Reserve(); got != before {
		t.Errorf("reserve: got %d, want the unchanged %d", got, before)
	}
}

func TestThePercentileIgnoresOneOutlier(t *testing.T) {
	// The maximum would price every sandbox by the one that happened to be compiling. The
	// p95 is what a reservation should track.
	if got := percentile([]int64{100, 100, 100, 100, 100, 100, 100, 100, 100, 10000}, 95); got != 100 {
		t.Errorf("p95: got %d, want 100", got)
	}
	if got := percentile([]int64{5}, 95); got != 5 {
		t.Errorf("p95 of one sample: got %d, want 5", got)
	}
	if got := percentile([]int64{1, 2, 3}, 100); got != 3 {
		t.Errorf("p100: got %d, want 3", got)
	}
}

func TestUnsetFieldsTakeTheirDefaults(t *testing.T) {
	cfg := FootprintConfigWithDefaults(FootprintConfig{})

	if cfg.MinReserve <= 0 || cfg.Percentile != 95 || cfg.ShrinkAfter <= 0 || cfg.ShrinkStep <= 0 {
		t.Errorf("defaults: %+v", cfg)
	}
}

func repeat(value int64, count int) []int64 {
	out := make([]int64, count)
	for i := range out {
		out[i] = value
	}
	return out
}
