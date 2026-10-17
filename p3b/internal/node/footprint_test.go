package node

import "testing"

func footprintConfig() FootprintConfig {
	return FootprintConfig{
		MinReserveMB: 512,
		Percentile:   95,
		GrowthMargin: 0.2,
		ShrinkAfter:  3,
		ShrinkStep:   0.5,
	}
}

func TestAFreshClassIsChargedTheMinimum(t *testing.T) {
	// The first batch of a step is admitted before anything has been measured, so the
	// minimum is what it is charged. Charging the declared maximum until measurement
	// arrives would reintroduce the reservation this exists to remove.
	footprint := NewFootprint(footprintConfig(), 30000)

	if got := footprint.ReserveMB(); got != 512 {
		t.Errorf("reserve: got %d, want the 512MiB minimum", got)
	}
}

func TestTheReserveFollowsTheObservedP95Immediately(t *testing.T) {
	// Fast grow: one sample that says the class needs more is enough, because
	// under-reserving is the direction that loses sandboxes.
	footprint := NewFootprint(footprintConfig(), 30000)

	footprint.Observe([]int64{1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 20000})

	// p95 of ten samples is the ninth, 1000, plus the 20% margin.
	if got := footprint.ReserveMB(); got != 1200 {
		t.Errorf("reserve: got %d, want 1200 (1000 x 1.2)", got)
	}
}

func TestTheReserveComesDownSlowlyAndOnlyAfterQuietSamples(t *testing.T) {
	// Slow shrink: a reservation that followed every quiet sample down would oscillate
	// with each tool call, and every oscillation is a re-admission decision.
	footprint := NewFootprint(footprintConfig(), 30000)
	footprint.Observe(repeat(4000, 10))
	high := footprint.ReserveMB()
	if high != 4800 {
		t.Fatalf("the test needs a high reserve, got %d", high)
	}

	// Two quiet samples are not enough.
	footprint.Observe(repeat(500, 10))
	footprint.Observe(repeat(500, 10))
	if got := footprint.ReserveMB(); got != high {
		t.Errorf("reserve after two quiet samples: got %d, want the unchanged %d", got, high)
	}

	// The third allows a step, and the step is limited rather than jumping to the target.
	footprint.Observe(repeat(500, 10))
	after := footprint.ReserveMB()
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

	footprint.Observe(repeat(10000, 10))

	if got := footprint.ReserveMB(); got != 2048 {
		t.Errorf("reserve: got %d, want the 2048MiB maximum", got)
	}
}

func TestTheReserveNeverFallsBelowTheMinimum(t *testing.T) {
	footprint := NewFootprint(footprintConfig(), 30000)

	for i := 0; i < 20; i++ {
		footprint.Observe([]int64{1, 1, 1})
	}

	if got := footprint.ReserveMB(); got != 512 {
		t.Errorf("reserve: got %d, want the 512MiB minimum", got)
	}
}

func TestAnEmptySampleChangesNothing(t *testing.T) {
	// A class with no running sandbox is between batches, not idle: dropping to the
	// minimum there would under-reserve the batch that follows.
	footprint := NewFootprint(footprintConfig(), 30000)
	footprint.Observe(repeat(4000, 10))
	before := footprint.ReserveMB()

	footprint.Observe(nil)

	if got := footprint.ReserveMB(); got != before {
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

	if cfg.MinReserveMB <= 0 || cfg.Percentile != 95 || cfg.ShrinkAfter <= 0 || cfg.ShrinkStep <= 0 {
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
