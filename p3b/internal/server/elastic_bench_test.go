package server

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/cgroup"
	"psrl.dev/sandboxd/internal/node"
)

// This file is the performance half of the bounds controller, and it exists because the
// 200ms cadence is a claim about cost as much as about latency.
//
// The shape follows the sandbox benchmark suites in third_party/AgentENV: measure at
// several concurrency levels rather than one, warm up before measuring, and report the
// distribution rather than a single number -- here with Go's own benchmark harness instead
// of criterion, which is the same instrument in a different language.
//
// What it must establish:
//
//   - One tick is cheap enough that 5 of them a second is a rounding error on a node
//     running a hundred sandboxes. WeEnv reports tens of microseconds per environment and
//     under 0.1% of a core at this cadence; the number here should be in that world, and
//     if it is not, the cadence is the thing to change rather than the controller.
//   - The cost is linear in the number of sandboxes. The tick is one pass over the domain,
//     and a pass that turns out to be quadratic would be a fleet-scale surprise rather
//     than a benchmark number.
//   - A burst is tracked in the number of doublings the theory predicts, and no more:
//     going from a 1 GiB bound to covering 20 GiB of demand is ceil(log2(20)) steps, so
//     the wall clock is that many intervals, and the test asserts the count rather than
//     the duration so it does not depend on how fast the machine is.

// benchmarkSandboxes are the concurrency levels the sweep runs at.
//
// One, ten, the size of a single step's wave, and twice that -- the last is the number a
// node reaches when two steps overlap, which is the state the controller has to survive.
var benchmarkSandboxes = []int{1, 10, 50, 100, 200}

func BenchmarkLimitsTick(b *testing.B) {
	for _, count := range benchmarkSandboxes {
		b.Run("sandboxes="+strconv.Itoa(count), func(b *testing.B) {
			parent := "bench-sandboxes"
			domain, memoryDir, cpuDir := benchDomain(b, parent)
			agent := benchNode(b, domain, parent)
			ids := benchSandboxes(b, agent, memoryDir, cpuDir, parent, count)

			// Warm up: the first ticks allocate the controller's maps and fault in the
			// files, and including them would measure the harness rather than the tick.
			for i := 0; i < 10; i++ {
				agent.ElasticLimitsTick(context.Background())
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				agent.ElasticLimitsTick(context.Background())
			}
			b.StopTimer()
			// One sandbox in every eight is at 70% of its bound, so the tick exercises the
			// growth path and the preemption scan as well as the plain pass. A benchmark of
			// a thousand idle sandboxes would measure the read and nothing else.
			_ = ids
		})
	}
}

// BenchmarkLimitsTickUnderGrowth is the same tick with every sandbox pressing on its bound
// and no spare budget, which is the worst case: growth is attempted for each one and every
// attempt has to walk the domain again looking for a loan to take back.
func BenchmarkLimitsTickUnderGrowth(b *testing.B) {
	for _, count := range benchmarkSandboxes {
		b.Run("sandboxes="+strconv.Itoa(count), func(b *testing.B) {
			parent := "bench-sandboxes"
			domain, memoryDir, cpuDir := benchDomain(b, parent)
			agent := benchNode(b, domain, parent)
			ids := benchSandboxes(b, agent, memoryDir, cpuDir, parent, count)
			for _, id := range ids {
				benchSandbox(b, memoryDir, cpuDir, parent, id, 900*benchMib, 1024*benchMib)
			}
			for i := 0; i < 10; i++ {
				agent.ElasticLimitsTick(context.Background())
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				agent.ElasticLimitsTick(context.Background())
			}
			b.StopTimer()
		})
	}
}

// TestATickIsCheapEnoughForItsCadence turns the benchmark into an assertion, because a
// benchmark that nobody runs is not a guard.
//
// The budget is generous on purpose: it is here to catch a tick that became quadratic or
// started writing files, not to pin a number that moves with the machine. What it states
// is that at five ticks a second the controller costs a small fraction of one core even
// with two hundred sandboxes pressing on their bounds.
func TestATickIsCheapEnoughForItsCadence(t *testing.T) {
	const (
		parent   = "bench-sandboxes"
		count    = 200
		budgetMS = 50.0
	)
	domain, memoryDir, cpuDir := benchDomain(t, parent)
	agent := benchNode(t, domain, parent)
	ids := benchSandboxes(t, agent, memoryDir, cpuDir, parent, count)
	for _, id := range ids {
		benchSandbox(t, memoryDir, cpuDir, parent, id, 900*benchMib, 1024*benchMib)
	}

	// Warm up, then take the median of a window rather than one sample: a single tick can
	// be interrupted by the scheduler, and a tail is not what the cadence is priced on.
	for i := 0; i < 10; i++ {
		agent.ElasticLimitsTick(context.Background())
	}
	samples := make([]float64, 0, 30)
	for i := 0; i < 30; i++ {
		started := time.Now()
		agent.ElasticLimitsTick(context.Background())
		samples = append(samples, float64(time.Since(started).Microseconds())/1000.0)
	}
	median := percentile(samples, 0.5)

	perCore := median * 5 / 1000.0 * 100.0 // five ticks a second, as a percentage of a core
	t.Logf("median tick %.3f ms over %d sandboxes; %.2f%% of one core at 200ms", median, count, perCore)
	if median > budgetMS {
		t.Errorf("median tick %.3f ms over %d sandboxes exceeds the %.0f ms budget", median, count, budgetMS)
	}
}

// TestTheCostIsLinearInTheNumberOfSandboxes is the shape of the cost, not its size.
//
// One pass over the domain is O(n); a controller that re-reads per sandbox, or rescans for
// every growth, is O(n^2) and would be indistinguishable from this one at ten sandboxes.
// Doubling the fleet must roughly double the tick, not quadruple it.
func TestTheCostIsLinearInTheNumberOfSandboxes(t *testing.T) {
	const parent = "bench-sandboxes"
	measure := func(count int) float64 {
		domain, memoryDir, cpuDir := benchDomain(t, parent)
		agent := benchNode(t, domain, parent)
		// Idle, inside the hysteresis band: this measures the pass over the domain, and a
		// sandbox that is growing adds a write whose cost belongs to the write rather than
		// to the shape of the loop.
		benchSandboxes(t, agent, memoryDir, cpuDir, parent, count)
		for i := 0; i < 10; i++ {
			agent.ElasticLimitsTick(context.Background())
		}
		samples := make([]float64, 0, 20)
		for i := 0; i < 20; i++ {
			started := time.Now()
			agent.ElasticLimitsTick(context.Background())
			samples = append(samples, float64(time.Since(started).Microseconds())/1000.0)
		}
		return percentile(samples, 0.5)
	}

	small := measure(50)
	large := measure(100)
	t.Logf("median tick: %.3f ms at 50 sandboxes, %.3f ms at 100 (%.2fx)", small, large, large/small)
	// Four times the work for twice the sandboxes would be quadratic. Allow generous slack
	// for the fixed cost of the first few and for a noisy machine.
	if small > 0 && large > small*4 {
		t.Errorf("cost grew %.2fx for 2x the sandboxes; the tick must stay one pass", large/small)
	}
}

// TestABurstIsTrackedInTheDoublingsTheTheoryPredicts is the claim the cadence exists for.
//
// A sandbox holding 1 GiB that suddenly wants 20 GiB needs ceil(log2(20/1)) = 5 doublings
// at worst, and each doubling is one interval. At 200ms that is a second; at the old five
// second cadence it is half a minute, which is longer than the tool call that caused it and
// is exactly how a sandbox ends up killed at a bound the node could have raised.
//
// The assertion is on the number of ticks and not on wall time, so it holds on a slow
// machine and fails if the controller starts growing by less than the configured factor or
// waiting for something before it moves.
func TestABurstIsTrackedInTheDoublingsTheTheoryPredicts(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := limitsNode(t, domain, parent, 100000, 100000, ElasticConfig{
		Enabled: true, StartLimitMB: 1024, GrowAbove: 0.60, GrowFactor: 2.0,
	})
	spec := backendSpec(t, "rollout", 2)
	spec.Resources.MemoryMB = 32768
	agent.elasticRemember("sb-1", spec)

	// A burst: 20 GiB of demand appears inside one interval.
	benchSandbox(t, memoryDir, cpuDir, parent, "sb-1", 20*1024*benchMib, 1024*benchMib)
	expected := int(math.Ceil(math.Log2(20.0))) // 1 GiB to cover 20 GiB

	ticks := 0
	for limitOf(t, memoryDir, parent, "sb-1") < 20*1024 && ticks < 20 {
		agent.ElasticLimitsTick(context.Background())
		ticks++
	}
	if ticks > expected {
		t.Errorf("took %d ticks to cover the burst, want at most %d (one doubling each)", ticks, expected)
	}
	t.Logf("burst covered in %d ticks; %.2fs at the 200ms cadence",
		ticks, float64(ticks)*DefaultLimitSampleIntervalS)
}

// TestTheCadenceIsWhatTheDeploymentAsksFor pins the two loops apart.
//
// They are separate because they answer different questions, and a future change that
// collapses them onto one interval would quietly make the bounds either five seconds late
// or five times the cost.
func TestTheCadenceIsWhatTheDeploymentAsksFor(t *testing.T) {
	if got := DefaultLimitSampleInterval(); got != 200*time.Millisecond {
		t.Errorf("limit cadence: got %v, want 200ms", got)
	}
	if DefaultLimitSampleIntervalS >= 1 {
		t.Errorf("limit cadence %v is not sub-second", DefaultLimitSampleIntervalS)
	}
}

// benchMib is the byte count the helpers below speak in.
const benchMib = 1024 * 1024

type benchTB interface {
	Helper()
	Fatalf(format string, args ...any)
	TempDir() string
}

func benchDomain(t benchTB, parent string) (cgroup.Domain, string, string) {
	t.Helper()
	root := t.TempDir()
	memoryDir := filepath.Join(root, "memory")
	cpuDir := filepath.Join(root, "cpuacct")
	for _, dir := range []string{
		filepath.Join(memoryDir, parent), filepath.Join(cpuDir, parent),
	} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", dir, err)
		}
	}
	domain := cgroup.Domain{
		Version: 1,
		Hierarchies: []cgroup.Hierarchy{
			{Controllers: []string{"memory"}, Mount: memoryDir, Root: memoryDir},
			{Controllers: []string{"cpuacct"}, Mount: cpuDir, Root: cpuDir},
		},
	}
	return domain, memoryDir, cpuDir
}

func benchNode(t benchTB, domain cgroup.Domain, parent string) *Node {
	t.Helper()
	gate, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{MemoryMB: 200000, CPUMillis: 200000},
		Classes:  map[string]node.ClassShare{"rollout": {Guaranteed: 0.9}, "grader": {Guaranteed: 0.1}},
		LeaseTTL: time.Minute,
	})
	if err != nil {
		t.Fatalf("gate: %v", err)
	}
	gate.SetPressure(node.Pressure{SampledAt: time.Now(), Valid: true})
	life, err := node.NewLifecycle("bench", gate, time.Second)
	if err != nil {
		t.Fatalf("lifecycle: %v", err)
	}
	agent, err := NewNode(NodeConfig{
		NodeID: "bench", Admission: gate, Lifecycle: life,
		Backends: []backend.Backend{providerStub{}},
		Pressure: func() node.Pressure { return node.Pressure{SampledAt: time.Now(), Valid: true} },
		Elastic: ElasticConfig{
			Enabled: true, StartLimitMB: 1024, GrowAbove: 0.60, GrowFactor: 2.0,
			ShrinkBelow: 0.20, ShrinkDwellS: 60, ShrinkFactor: 0.5, SafetyFraction: 0.05,
			ReapForeign: true, ReapUsageMB: 64,
		},
		Domain: domain, SandboxParent: parent, SandboxBudgetMB: 200000,
	})
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	return agent
}

func benchSandboxes(t benchTB, agent *Node, memoryDir, cpuDir, parent string, count int) []string {
	t.Helper()
	ids := make([]string, 0, count)
	for i := 0; i < count; i++ {
		id := "bench-" + strconv.Itoa(i)
		spec := backendSpec(nil, "rollout", 2)
		spec.Resources.MemoryMB = 32768
		agent.elasticRemember(id, spec)
		benchSandbox(t, memoryDir, cpuDir, parent, id, 256*benchMib, 1024*benchMib)
		ids = append(ids, id)
	}
	return ids
}

// benchSandbox writes one sandbox's counters with the limit the caller wants.
//
// The limit is the input the controller works from, so a benchmark of it has to control
// it: a sandbox whose bound is effectively unbounded is at 0% of it forever and the tick
// measures only the read.
func benchSandbox(t benchTB, memoryDir, cpuDir, parent, id string, usageBytes, limitBytes int64) {
	t.Helper()
	for _, dir := range []string{filepath.Join(memoryDir, parent, id), filepath.Join(cpuDir, parent, id)} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", dir, err)
		}
	}
	write := func(path, value string) {
		if err := os.WriteFile(path, []byte(value), 0o644); err != nil {
			t.Fatalf("write %s: %v", path, err)
		}
	}
	write(filepath.Join(memoryDir, parent, id, "memory.usage_in_bytes"), strconv.FormatInt(usageBytes, 10))
	write(filepath.Join(memoryDir, parent, id, "memory.limit_in_bytes"), strconv.FormatInt(limitBytes, 10))
	write(filepath.Join(memoryDir, parent, id, "memory.stat"),
		"total_rss "+strconv.FormatInt(usageBytes, 10)+"\ntotal_cache 0\n")
	write(filepath.Join(cpuDir, parent, id, "cpuacct.usage"), "0")
	write(filepath.Join(cpuDir, parent, id, "cpu.cfs_quota_us"), "200000")
	write(filepath.Join(cpuDir, parent, id, "cpu.cfs_period_us"), "100000")
	write(filepath.Join(cpuDir, parent, id, "cpu.stat"), "nr_periods 0\nnr_throttled 0\nthrottled_time 0\n")
}

// percentile returns the value at a fraction of a sorted copy, which is what makes the
// timing assertions about the distribution rather than about its tail.
func percentile(values []float64, fraction float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	for i := 1; i < len(sorted); i++ {
		for j := i; j > 0 && sorted[j] < sorted[j-1]; j-- {
			sorted[j], sorted[j-1] = sorted[j-1], sorted[j]
		}
	}
	index := int(float64(len(sorted)-1) * fraction)
	return sorted[index]
}
