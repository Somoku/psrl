package server

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/cgroup"
	"psrl.dev/sandboxd/internal/node"

	v1 "psrl.dev/sandboxd/api/v1"
)

// fakeDomain builds a cgroup domain over two temp directories -- one hierarchy that
// carries memory and one that carries cpuacct, which is how a v1 host mounts them -- so
// the controller's whole read path runs against files a test controls.
func fakeDomain(t *testing.T, parent string) (cgroup.Domain, string, string) {
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

// fakeSandbox writes one sandbox's cgroup files on both hierarchies.
func fakeSandbox(t *testing.T, memoryDir, cpuDir, parent, id string, usageBytes, cpuNanos int64, cpuCount float64) {
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
	write(filepath.Join(memoryDir, parent, id, "memory.stat"), "total_rss "+strconv.FormatInt(usageBytes, 10)+"\ntotal_cache 0\n")
	write(filepath.Join(memoryDir, parent, id, "memory.limit_in_bytes"), "9223372036854771712")
	write(filepath.Join(cpuDir, parent, id, "cpuacct.usage"), strconv.FormatInt(cpuNanos, 10))
	write(filepath.Join(cpuDir, parent, id, "cpu.cfs_quota_us"), strconv.FormatInt(int64(cpuCount*100000), 10))
	write(filepath.Join(cpuDir, parent, id, "cpu.cfs_period_us"), "100000")
	write(filepath.Join(cpuDir, parent, id, "cpu.stat"), "nr_periods 0\nnr_throttled 0\nthrottled_time 0\n")
}

// elasticNodeOver builds a node whose controller reads a domain a test controls.
func elasticNodeOver(t *testing.T, domain cgroup.Domain, parent string, envelopeCPUMillis int64) *Node {
	t.Helper()
	gate, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{MemoryMB: 4096, CPUMillis: envelopeCPUMillis},
		Classes:  map[string]node.ClassShare{"rollout": {Guaranteed: 1.0}},
		LeaseTTL: time.Minute,
	})
	if err != nil {
		t.Fatalf("gate: %v", err)
	}
	gate.SetPressure(node.Pressure{SampledAt: time.Now(), Valid: true})
	life, err := node.NewLifecycle("n1", gate, time.Second)
	if err != nil {
		t.Fatalf("lifecycle: %v", err)
	}
	agent, err := NewNode(NodeConfig{
		NodeID: "n1", Admission: gate, Lifecycle: life,
		Backends:      []backend.Backend{providerStub{}},
		Pressure:      func() node.Pressure { return node.Pressure{SampledAt: time.Now(), Valid: true} },
		Elastic:       ElasticConfig{Enabled: true},
		Domain:        domain,
		SandboxParent: parent,
	})
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	return agent
}

func TestTheCPUReserveFollowsAMeasuredRateNotTheDeclaredCores(t *testing.T) {
	// A sandbox declares two cores because one tool call may want them, and uses a
	// fraction of one between calls. The price has to follow the measurement: charging the
	// declaration caps the fleet at a number that has nothing to do with the load on it.
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := elasticNodeOver(t, domain, parent, 100000)
	id := admitSandbox(t, agent, backendSpec(t, "rollout", 2))
	fakeSandbox(t, memoryDir, cpuDir, parent, id, 512<<20, 0, 2)

	// The first sighting establishes the baseline and prices nothing: one reading of a
	// cumulative counter says nothing about load.
	report := agent.ElasticTick(context.Background())
	if len(report.ReservesCPUMillis) == 0 || report.ReservesCPUMillis["rollout"] != 250 {
		t.Fatalf("reserve before any measurement: got %v, want the 250 millicore minimum", report.ReservesCPUMillis)
	}

	// One second of CPU over a five second interval is 200 millicores.
	agent.elastic.lastSample = time.Now().Add(-5 * time.Second)
	fakeSandbox(t, memoryDir, cpuDir, parent, id, 512<<20, 1*time.Second.Nanoseconds(), 2)
	report = agent.ElasticTick(context.Background())

	if got := report.ObservedCPUMillis; got != 200 {
		t.Errorf("measured CPU: got %d millis, want 200 (1s of CPU over 5s)", got)
	}
	// Fast grow: the estimate is the measurement plus the growth margin.
	if got := agent.classCPUReservesMillis()["rollout"]; got != 250 {
		t.Errorf("CPU reserve: got %d millis, want 250 -- 200 x 1.25 rounded up to the minimum", got)
	}
	if got := report.ReservedCPUMillis; got != 250 {
		t.Errorf("ledger charge: got %d millis, want 250", got)
	}

	// A burst lifts it, and it converges on the new measurement within a few samples:
	// the direction must never lag, but a single sample may only double.
	report = settleTicks(t, agent, memoryDir, cpuDir, parent, id, 800, 0, 8)
	if got := agent.classCPUReservesMillis()["rollout"]; got != 1000 {
		t.Errorf("CPU reserve after a burst: got %d millis, want 1000 (800 x 1.25)", got)
	}

	// The declared two cores are what the node would have charged without measurement, so
	// the price is the whole difference this makes to a burst.
	charge := agent.elasticCharge(backendSpec(t, "rollout", 2))
	if charge.CPUMillis != 1000 {
		t.Errorf("charge: got %d millis, want the 1000 reserve rather than the 2000 declared", charge.CPUMillis)
	}
}

func TestAnIdleSandboxIsThrottledUnderCPUPressureAndReleasedWhenItPasses(t *testing.T) {
	// CPU is compressible, so a quota below what a sandbox is using costs it latency
	// rather than its life -- which is why the controller may move it at all. It is still
	// only moved on an idle sandbox, and only while the node is under pressure.
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	// An envelope of 4 cores: three and a half measured is 0.875, above the 0.85 shrink
	// threshold, and the same measurement against a 100 core envelope would be nothing.
	// Four declared cores in an envelope of four, so a sandbox measured at three and a half
	// is both under pressure -- 0.875 of the envelope -- and above the quota it would be
	// throttled to.
	agent := elasticNodeOver(t, domain, parent, 4000)
	id := admitSandbox(t, agent, backendSpec(t, "rollout", 4))
	fakeSandbox(t, memoryDir, cpuDir, parent, id, 512<<20, 0, 2)
	agent.ElasticTick(context.Background())

	agent.elastic.lastSample = time.Now().Add(-5 * time.Second)
	fakeSandbox(t, memoryDir, cpuDir, parent, id, 512<<20, 17500000000, 4) // 3.5 cores over 5s
	report := agent.ElasticTick(context.Background())

	if len(report.Throttled) != 1 || report.Throttled[0] != id {
		t.Fatalf("throttled: got %v, want the throttled sandbox", report.Throttled)
	}
	// The quota is what it measured plus a margin: 3500 + 250. A rate is a division by
	// the interval, which is a real clock and not exactly five seconds, so the target is
	// asserted as a band rather than a constant.
	assertQuotaNearMillis(t, cpuDir, parent, id, 3750, 20)
	if !agent.elastic.throttled[id] {
		t.Error("the sandbox is not recorded as throttled, so the next tick would throttle it again")
	}

	// Pressure passing restores what the sandbox declared, which is the whole point of a
	// compressible dimension: nothing was lost while it was throttled.
	agent.elastic.lastSample = time.Now().Add(-5 * time.Second)
	fakeSandbox(t, memoryDir, cpuDir, parent, id, 512<<20, 17500000000, 4)
	report = agent.ElasticTick(context.Background())
	if len(report.Released) != 1 {
		t.Fatalf("released: got %v, want the throttled sandbox", report.Released)
	}
	assertQuotaMillis(t, cpuDir, parent, id, 4000)
}

// admitSandbox puts a real record on the node's lifecycle, so the controller's audit does
// not drop a sandbox the test only wrote cgroup files for.
func admitSandbox(t *testing.T, agent *Node, spec backend.Spec) string {
	t.Helper()
	admitted, err := agent.Admit(context.Background(), &v1.AdmitRequest{Spec: specToProto(spec)})
	if err != nil || !admitted.GetAdmitted() {
		t.Fatalf("admit: %v (refusal %q)", err, admitted.GetRefusal())
	}
	created, err := agent.CreateOn(context.Background(), &v1.CreateOnRequest{
		LeaseId: admitted.GetLeaseId(), Backend: "stub", Spec: specToProto(spec),
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	return created.GetHandle().GetSandboxId()
}

func assertQuotaNearMillis(t *testing.T, cpuDir, parent, id string, want, tolerance int64) {
	t.Helper()
	got := quotaMillis(t, cpuDir, parent, id)
	if got < want-tolerance || got > want+tolerance {
		t.Errorf("quota: got %d millis, want %d +/- %d", got, want, tolerance)
	}
}

func assertQuotaMillis(t *testing.T, cpuDir, parent, id string, want int64) {
	t.Helper()
	got := quotaMillis(t, cpuDir, parent, id)
	if got != want {
		t.Errorf("quota: got %d millis, want %d", got, want)
	}
}

func quotaMillis(t *testing.T, cpuDir, parent, id string) int64 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(cpuDir, parent, id, "cpu.cfs_quota_us"))
	if err != nil {
		t.Fatalf("read the quota: %v", err)
	}
	return cgroup.QuotaToMillis(parseTestInt(t, string(raw)))
}

func parseTestInt(t *testing.T, raw string) int64 {
	t.Helper()
	var value int64
	for _, r := range raw {
		if r < '0' || r > '9' {
			continue
		}
		value = value*10 + int64(r-'0')
	}
	return value
}

// backendSpec builds a request of one class. The class is what the guarantee is keyed on,
// so it is the only field the wave-contention tests care about.
func backendSpec(t *testing.T, class string, cores float64) backend.Spec {
	if t != nil {
		t.Helper()
	}
	return backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "img"},
		Resources:     backend.Resources{MemoryMB: 30720, CPUCount: cores},
		ResourceClass: class,
	}
}

// setThrottled writes a throttling reading for one sandbox, as the kernel would after
// refusing it CPU.
// settleTicks runs enough ticks for the class price to converge on a measurement, at a
// sustained rate, and returns the last tick's report.
//
// One sample may only double the estimate, so a test that wants the converged price runs
// several. That is the point of the bound: the price still reaches the measurement, but
// over a few samples, so the fleet's ledger and the node's agree while it climbs. The
// counter has to keep advancing -- rewriting the same cumulative value would read as a
// sandbox that went idle.
func settleTicks(
	t *testing.T, agent *Node, memoryDir, cpuDir, parent, id string, rateMillis, throttleMillis int64, ticks int,
) ElasticReport {
	t.Helper()
	// millicores sustained over a five second interval, in nanoseconds.
	stepNS := rateMillis * 5 * int64(time.Second) / 1000
	throttleNS := throttleMillis * 5 * int64(time.Second) / 1000
	usage := agent.elastic.cpuUsageNanos[id]
	throttled := agent.elastic.cpuThrottledNanos[id]
	var report ElasticReport
	for i := 0; i < ticks; i++ {
		usage += stepNS
		throttled += throttleNS
		agent.elastic.lastSample = time.Now().Add(-5 * time.Second)
		fakeSandbox(t, memoryDir, cpuDir, parent, id, 512<<20, usage, 2)
		setThrottled(t, cpuDir, parent, id, throttled)
		report = agent.ElasticTick(context.Background())
	}
	return report
}

func setThrottled(t *testing.T, cpuDir, parent, id string, throttledNanos int64) {
	t.Helper()
	path := filepath.Join(cpuDir, parent, id, "cpu.stat")
	if err := os.WriteFile(path, []byte(fmt.Sprintf(
		"nr_periods 100\nnr_throttled 90\nthrottled_time %d\n", throttledNanos)), 0o644); err != nil {
		t.Fatalf("write cpu.stat: %v", err)
	}
}

func TestAThrottledSandboxIsPricedAtItsQuotaNotItsCensoredUsage(t *testing.T) {
	// Usage is censored from above by the quota: a sandbox that wanted more than it was
	// given only ever reaches the quota, so its measured rate stops describing demand.
	// Without the throttling signal a class can be priced at its idle rate while every
	// sandbox in it is being throttled -- which is what admitting past the machine's CPU
	// looks like from the ledger.
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := elasticNodeOver(t, domain, parent, 100000)
	id := admitSandbox(t, agent, backendSpec(t, "rollout", 2))
	fakeSandbox(t, memoryDir, cpuDir, parent, id, 512<<20, 0, 2)
	agent.ElasticTick(context.Background())

	// Five seconds in which the sandbox used 1 core and was throttled for the other four.
	// Measured usage says 1000 millis; the quota says 2000, and throttling says it wanted
	// at least that.
	report := settleTicks(t, agent, memoryDir, cpuDir, parent, id, 1000, 4000, 10)

	if got := report.ThrottledPct; got < 0.7 {
		t.Errorf("throttled share: got %.2f, want about 0.80", got)
	}
	// The price follows demand: 2000 millicores x the growth margin, reached over a few
	// samples rather than in one jump.
	if got := agent.classCPUReservesMillis()["rollout"]; got != 2500 {
		t.Errorf("CPU reserve: got %d millis, want 2500 (the 2000 quota x 1.25), not %d from censored usage",
			got, 1250)
	}
}

func TestAnUnthrottledSandboxIsStillPricedFromWhatItUsed(t *testing.T) {
	// The substitution has to be conditional. A sandbox that used one core and was never
	// throttled wanted one core, and charging it its quota would put the price back at the
	// declared maximum, which is the reservation this mechanism exists to remove.
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := elasticNodeOver(t, domain, parent, 100000)
	id := admitSandbox(t, agent, backendSpec(t, "rollout", 2))
	fakeSandbox(t, memoryDir, cpuDir, parent, id, 512<<20, 0, 2)
	agent.ElasticTick(context.Background())

	report := settleTicks(t, agent, memoryDir, cpuDir, parent, id, 1000, 0, 8)

	if got := report.ThrottledPct; got != 0 {
		t.Errorf("throttled share: got %.2f, want 0", got)
	}
	if got := agent.classCPUReservesMillis()["rollout"]; got != 1250 {
		t.Errorf("CPU reserve: got %d millis, want 1250 (1000 x 1.25)", got)
	}
}

func TestAThrottledSandboxIsNotShrunk(t *testing.T) {
	// A sandbox being throttled is trying to run, whether or not an RPC is in flight.
	// Shrinking it on the grounds that no command is outstanding is how a controller
	// squeezes a sandbox whose background process is mid-compile.
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := elasticNodeOver(t, domain, parent, 4000)
	id := admitSandbox(t, agent, backendSpec(t, "rollout", 4))
	fakeSandbox(t, memoryDir, cpuDir, parent, id, 512<<20, 0, 2)
	agent.ElasticTick(context.Background())

	// Throttled, so busy -- but the CPU fraction is computed from demand, which is now the
	// quota, so the node is over its shrink threshold and would otherwise act.
	agent.elastic.lastSample = time.Now().Add(-5 * time.Second)
	fakeSandbox(t, memoryDir, cpuDir, parent, id, 512<<20, 20*time.Second.Nanoseconds(), 4)
	setThrottled(t, cpuDir, parent, id, 4*time.Second.Nanoseconds())
	report := agent.ElasticTick(context.Background())

	if len(report.Throttled) != 0 {
		t.Errorf("a throttled sandbox must not have its quota cut: %v", report.Throttled)
	}
}
