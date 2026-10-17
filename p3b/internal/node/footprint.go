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
type FootprintConfig struct {
	// MinReserveMB is what a sandbox is charged before the class has any history. It
	// covers a shell, a tool call and the agent's own process; a step's first batch is
	// admitted against it rather than against the declared maximum.
	MinReserveMB int64
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
}

// FootprintConfigWithDefaults fills in the fields a deployment left unset.
func FootprintConfigWithDefaults(cfg FootprintConfig) FootprintConfig {
	if cfg.MinReserveMB <= 0 {
		cfg.MinReserveMB = 512
	}
	if cfg.Percentile <= 0 || cfg.Percentile > 100 {
		cfg.Percentile = 95
	}
	if cfg.GrowthMargin < 0 {
		cfg.GrowthMargin = 0
	}
	if cfg.ShrinkAfter <= 0 {
		cfg.ShrinkAfter = 3
	}
	if cfg.ShrinkStep <= 0 || cfg.ShrinkStep > 1 {
		cfg.ShrinkStep = 0.25
	}
	return cfg
}

// Footprint is one class's estimate of a sandbox's cost.
type Footprint struct {
	cfg FootprintConfig
	// estimateMB is the reservation one sandbox of this class is charged.
	estimateMB float64
	// maxMB is the largest reservation this class may charge: the largest declared maximum
	// its sandboxes ask for. A reservation above that would let the ledger charge more than
	// the kernel bound allows.
	maxMB int64
	// quiet counts consecutive samples below the estimate.
	quiet int
}

// NewFootprint returns an estimate for a class, seeded at the configured minimum.
func NewFootprint(cfg FootprintConfig, maxMB int64) *Footprint {
	cfg = FootprintConfigWithDefaults(cfg)
	footprint := &Footprint{cfg: cfg, maxMB: maxMB, estimateMB: float64(cfg.MinReserveMB)}
	footprint.clamp()
	return footprint
}

// ReserveMB is what one sandbox of this class is currently charged.
func (f *Footprint) ReserveMB() int64 {
	if f == nil {
		return 0
	}
	return int64(math.Round(f.estimateMB))
}

// MaxMB is the largest reservation this class may charge.
func (f *Footprint) MaxMB() int64 {
	if f == nil {
		return 0
	}
	return f.maxMB
}

// Observe updates the estimate from one sample of per-sandbox resident usage.
//
// The input is what the sandboxes in this class cannot give back -- anonymous pages and
// kernel memory -- not their whole usage, because the reclaimable part is exactly what the
// ledger does not have to hold: the kernel can drop it when the node needs room.
func (f *Footprint) Observe(residentMB []int64) {
	if f == nil || len(residentMB) == 0 {
		return
	}
	target := float64(percentile(residentMB, f.cfg.Percentile)) * (1 + f.cfg.GrowthMargin)
	if target < float64(f.cfg.MinReserveMB) {
		target = float64(f.cfg.MinReserveMB)
	}

	if target >= f.estimateMB {
		// Fast grow, and it resets the quiet count: one busy sample is enough to raise the
		// price, because under-reserving is the direction that loses sandboxes.
		f.estimateMB = target
		f.quiet = 0
		f.clamp()
		return
	}

	f.quiet++
	if f.quiet < f.cfg.ShrinkAfter {
		return
	}
	// Slow shrink: part of the way, once per sample, so a batch that goes quiet together
	// gives memory back over several ticks instead of all at once.
	step := f.estimateMB * f.cfg.ShrinkStep
	if next := f.estimateMB - step; next > target {
		f.estimateMB = next
	} else {
		f.estimateMB = target
	}
	f.clamp()
}

// clamp keeps the estimate between the minimum and the largest declared maximum.
func (f *Footprint) clamp() {
	if f.estimateMB < float64(f.cfg.MinReserveMB) {
		f.estimateMB = float64(f.cfg.MinReserveMB)
	}
	if f.maxMB > 0 && f.estimateMB > float64(f.maxMB) {
		f.estimateMB = float64(f.maxMB)
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
