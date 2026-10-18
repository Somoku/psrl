// What one sandbox of a class costs, estimated from what its sandboxes actually use.
//
// Admission used to charge every sandbox its declared maximum, which is a protection
// bound rather than a cost: a step that admits 128 sandboxes of 30 GB each reserves
// 3.8 TB on a node that has 2.3 TB, so most of the batch waits for a reservation nobody
// is using. The estimate below is what replaces that maximum in the ledger -- the
// maximum stays where it belongs, as the kernel's hard bound for one sandbox.
//
// The policy is deliberately asymmetric. A reservation that is too small is the
// dangerous direction: the node admits work it cannot hold. So a reservation follows an
// observed peak up immediately, and comes back down only after several quiet samples and
// only part of the way. That is the "fast grow, slow shrink" the design asks for, and it
// is what keeps a reservation from following every tool call up and down.
package node

import (
	"math"
	"sort"
)

// FootprintConfig states how a class's per-sandbox reservation is estimated.
//
// The estimator does not know what it is estimating. It is applied to memory in MiB and
// to CPU in millicores, and the policy -- follow a peak up at once, come down slowly --
// is the same for both, because the dangerous direction is the same for both.
type FootprintConfig struct {
	// MinReserve is what a sandbox is charged before the class has any history. It
	// covers a shell, a tool call and the agent's own process; a step's first batch is
	// admitted against it rather than against the declared maximum. The unit is the
	// caller's: MiB for memory, millicores for CPU.
	MinReserve int64
	// Percentile is which observed per-sandbox usage the estimate tracks. The maximum
	// would make one outlier set the price for every sandbox; the median would ignore the
	// ones that matter. 95 is the default.
	Percentile float64
	// GrowthMargin is added on top of the percentile, and is also the allowance for a
	// group peaking together: sandboxes of one prompt run the same phases, so their peaks
	// correlate rather than adding independently.
	GrowthMargin float64
	// ShrinkAfter is how many consecutive samples may fall below the estimate before it
	// is lowered at all.
	ShrinkAfter int
	// ShrinkStep is the largest share of the estimate a single sample may remove.
	ShrinkStep float64
	// MaxGrowthFactor bounds how far one sample may raise the estimate, as a multiple of
	// the current one.
	//
	// Growth has to be immediate in direction -- under-reserving is the direction that
	// loses an episode -- but unbounded in magnitude it becomes a cliff. A class that was
	// first priced before its sandboxes had warmed up can measure sixty times higher on
	// the next sample, and a ledger that jumps that far in one tick is a ledger that
	// admitted the previous burst at a price that no longer exists: everything already
	// running was charged the old number, and every new request is refused until they
	// finish. Bounding the step closes the same gap over a few samples instead, which is
	// still fast against an episode and is absorbable by the rest of the system.
	MaxGrowthFactor float64
}

// FootprintConfigWithDefaults fills in the fields a deployment left unset.
//
// MinReserve has no default here: a memory footprint and a CPU footprint are charged in
// different units, so a shared floor would be a number in the wrong one. The caller
// states it, and a caller that does not gets the estimator's own floor of the same
// shape -- see ElasticConfigWithDefaults.
func FootprintConfigWithDefaults(cfg FootprintConfig) FootprintConfig {
	if cfg.MinReserve <= 0 {
		cfg.MinReserve = 1
	}
	if cfg.Percentile <= 0 || cfg.Percentile > 100 {
		cfg.Percentile = 95
	}
	if cfg.GrowthMargin <= 0 {
		// A quarter is the allowance for sandboxes of one class peaking together, which
		// they do: the sandboxes of one prompt run the same phases, so their peaks
		// correlate rather than adding independently. Zero would price a class at exactly
		// what its quietest sample showed.
		cfg.GrowthMargin = 0.25
	}
	if cfg.ShrinkAfter <= 0 {
		cfg.ShrinkAfter = 3
	}
	if cfg.ShrinkStep <= 0 || cfg.ShrinkStep > 1 {
		cfg.ShrinkStep = 0.25
	}
	if cfg.MaxGrowthFactor <= 0 {
		cfg.MaxGrowthFactor = DefaultMaxGrowthFactor
	}
	return cfg
}

// Footprint is one class's estimate of a sandbox's cost.
type Footprint struct {
	cfg FootprintConfig
	// estimate is the reservation one sandbox of this class is charged, in the unit the
	// caller samples in.
	estimate float64
	// ceiling is the largest reservation this class may charge: the largest declared
	// maximum its sandboxes ask for. A reservation above that would let the ledger charge
	// more than the kernel bound allows.
	ceiling int64
	// quiet counts consecutive samples below the estimate.
	quiet int
}

// DefaultMaxGrowthFactor is how far one sample may raise a class's estimate, as a
// multiple. Two means a class closes a sixty-fold gap in six samples -- thirty seconds at
// the default cadence -- which is fast against an episode and slow enough that the fleet's
// ledger and the node's agree while it happens.
const DefaultMaxGrowthFactor = 2.0

// NewFootprint returns an estimate for a class, seeded at the configured minimum.
func NewFootprint(cfg FootprintConfig, ceiling int64) *Footprint {
	cfg = FootprintConfigWithDefaults(cfg)
	footprint := &Footprint{cfg: cfg, ceiling: ceiling, estimate: float64(cfg.MinReserve)}
	footprint.clamp()
	return footprint
}

// Reserve is what one sandbox of this class is currently charged.
func (f *Footprint) Reserve() int64 {
	if f == nil {
		return 0
	}
	return int64(math.Round(f.estimate))
}

// Ceiling is the largest reservation this class may charge.
func (f *Footprint) Ceiling() int64 {
	if f == nil {
		return 0
	}
	return f.ceiling
}

// Observe updates the estimate from one sample of per-sandbox usage.
//
// The input is what the sandboxes in this class cannot give back: for memory that is
// anonymous pages and kernel memory rather than the whole usage, because the reclaimable
// part is exactly what the ledger does not have to hold -- the kernel can drop it when
// the node needs room. For CPU it is a rate in millicores, and the same asymmetry applies:
// a sandbox that ran flat out for one second is charged for the second, not for the peak
// a single thread can reach.
func (f *Footprint) Observe(samples []int64) {
	if f == nil || len(samples) == 0 {
		return
	}
	target := float64(percentile(samples, f.cfg.Percentile)) * (1 + f.cfg.GrowthMargin)
	if target < float64(f.cfg.MinReserve) {
		target = float64(f.cfg.MinReserve)
	}

	if target >= f.estimate {
		// Fast grow, and it resets the quiet count: one busy sample is enough to raise the
		// price, because under-reserving is the direction that loses sandboxes. The step is
		// bounded anyway, so a class that was priced before it had warmed up converges over
		// a few samples rather than in one jump the rest of the system cannot absorb.
		ceiling := f.estimate * f.cfg.MaxGrowthFactor
		if target > ceiling && ceiling > f.estimate {
			f.estimate = ceiling
		} else {
			f.estimate = target
		}
		f.quiet = 0
		f.clamp()
		return
	}

	f.quiet++
	if f.quiet < f.cfg.ShrinkAfter {
		return
	}
	// Slow shrink: part of the way, once per sample, so a batch that goes quiet together
	// gives capacity back over several ticks instead of all at once.
	step := f.estimate * f.cfg.ShrinkStep
	if next := f.estimate - step; next > target {
		f.estimate = next
	} else {
		f.estimate = target
	}
	f.clamp()
}

// clamp keeps the estimate between the minimum and the largest declared maximum.
func (f *Footprint) clamp() {
	if f.estimate < float64(f.cfg.MinReserve) {
		f.estimate = float64(f.cfg.MinReserve)
	}
	if f.ceiling > 0 && f.estimate > float64(f.ceiling) {
		f.estimate = float64(f.ceiling)
	}
}

// percentile returns the largest sample once the top (100-p) percent are set aside.
//
// Not the nearest-rank percentile, which for a batch of ten is the maximum: a class with
// one sandbox compiling and nine idle would then be priced as if all ten compile. Setting
// the top few aside is the trimming that makes the estimate describe the class rather than
// its busiest member, and a single sample still answers for itself.
func percentile(values []int64, p float64) int64 {
	sorted := append([]int64(nil), values...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	drop := int(math.Ceil((100 - p) / 100 * float64(len(sorted))))
	index := len(sorted) - 1 - drop
	if index < 0 {
		index = 0
	}
	return sorted[index]
}
