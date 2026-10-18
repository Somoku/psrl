package server

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/cgroup"
	"psrl.dev/sandboxd/internal/node"
)

// limitsNode builds a node whose bounds controller reads a domain a test writes, with the
// envelope and the subtree budget stated separately: the two are different numbers in
// production and the controller's growth rule is the thing that has to respect that.
func limitsNode(
	t *testing.T, domain cgroup.Domain, parent string, envelopeMB, budgetMB int64, cfg ElasticConfig,
) *Node {
	t.Helper()
	gate, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{MemoryMB: envelopeMB, CPUMillis: 100000},
		Classes:  map[string]node.ClassShare{"rollout": {Guaranteed: 0.9}, "grader": {Guaranteed: 0.1}},
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
		Elastic:       cfg,
		Domain:        domain,
		SandboxParent: parent,
		// The subtree budget is what the kernel will enforce, which is the number growth
		// has to spend from.
		SandboxBudgetMB: budgetMB,
	})
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	return agent
}

// sandbox writes one sandbox's cgroup files on both hierarchies, with the limit the test
// wants rather than the effectively-unbounded one fakeSandbox writes.
//
// The limit is the input the whole controller works from, so a test of the bounds has to
// control it: a sandbox whose bound is 8 EiB is at 0% of it forever and no rule ever fires.
func sandbox(t *testing.T, memoryDir, cpuDir, parent, id string, usageBytes, limitBytes int64) {
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

func setKillCount(t *testing.T, memoryDir, parent, id string, kills int64) {
	t.Helper()
	path := filepath.Join(memoryDir, parent, id, "memory.oom_control")
	// The v1 file carries three fields and no oom_kill_disable, so the counter is written
	// alongside them the way the kernel writes it.
	body := "oom_kill_enable 0\nunder_oom 0\noom_kill " + strconv.FormatInt(kills, 10) + "\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func setFailCount(t *testing.T, memoryDir, parent, id string, fails int64) {
	t.Helper()
	path := filepath.Join(memoryDir, parent, id, "memory.failcnt")
	if err := os.WriteFile(path, []byte(strconv.FormatInt(fails, 10)), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func limitOf(t *testing.T, memoryDir, parent, id string) int64 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(memoryDir, parent, id, "memory.limit_in_bytes"))
	if err != nil {
		t.Fatalf("read limit: %v", err)
	}
	value, err := strconv.ParseInt(string(raw), 10, 64)
	if err != nil {
		t.Fatalf("parse limit %q: %v", raw, err)
	}
	return value / mib
}

// A sandbox is born under its guarantee, not under its declared maximum, and the maximum
// stays as the ceiling growth may reach.
func TestASandboxIsCreatedUnderItsGuaranteeNotItsMaximum(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, _, _ := fakeDomain(t, parent)
	agent := limitsNode(t, domain, parent, 100000, 100000, ElasticConfig{
		Enabled: true, StartLimitMB: 8192,
	})

	spec := backendSpec(t, "rollout", 2)
	spec.Resources.MemoryMB = 30720

	// The sequence CreateOn actually runs: lower a copy for the creation call, and
	// remember what the caller declared.
	//
	// Lowering the spec itself and then remembering it records the guarantee as the
	// ceiling. A sandbox whose ceiling is its guarantee has nowhere to grow, so every
	// growth rule silently does nothing and the sandbox is killed at its start bound --
	// the failure this whole mechanism exists to prevent, produced by the mechanism.
	created := agent.applyStartLimit(spec)
	if created.Resources.MemoryMB != 8192 {
		t.Errorf("creation limit: got %d MiB, want the 8192 MiB guarantee", created.Resources.MemoryMB)
	}
	agent.elasticRemember("sb-1", spec)

	if got := agent.elastic.startMBBySandbox["sb-1"]; got != 8192 {
		t.Errorf("remembered start: got %d, want 8192", got)
	}
	// The ceiling is still what was declared. Lowering the creation bound does not lower
	// what the sandbox may become -- only what it must be given before it has asked.
	if got := agent.elastic.maxMemoryMBBySandbox["sb-1"]; got != 30720 {
		t.Errorf("remembered ceiling: got %d, want the 30720 MiB declared", got)
	}
	if ceiling := agent.ceilingLocked("sb-1") / mib; ceiling != 30720 {
		t.Errorf("ceiling: got %d MiB, want 30720 -- growth would have nowhere to go", ceiling)
	}
}

// A sandbox declared smaller than the start is a smaller sandbox and gets one.
func TestAStartAboveWhatWasDeclaredIsClampedToTheDeclaration(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, _, _ := fakeDomain(t, parent)
	agent := limitsNode(t, domain, parent, 100000, 100000, ElasticConfig{
		Enabled: true, StartLimitMB: 8192,
	})
	spec := backendSpec(t, "rollout", 2)
	spec.Resources.MemoryMB = 2048

	if got := agent.applyStartLimit(spec).Resources.MemoryMB; got != 2048 {
		t.Errorf("creation limit: got %d, want the 2048 MiB declared", got)
	}
}

// The growth rule: usage climbing into its bound doubles it, and it stops at the ceiling.
func TestABoundFollowsDemandUpAndStopsAtTheCeiling(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	// Plenty of machine, so the rule under test is the growth rule and not the budget.
	agent := limitsNode(t, domain, parent, 100000, 100000, ElasticConfig{
		Enabled: true, StartLimitMB: 1024, GrowAbove: 0.60, GrowFactor: 2.0,
	})
	spec := backendSpec(t, "rollout", 2)
	spec.Resources.MemoryMB = 8192
	agent.elasticRemember("sb-1", spec)

	// Well inside the band: nothing moves.
	sandbox(t, memoryDir, cpuDir, parent, "sb-1", 300*mib, 1024*mib)
	agent.ElasticLimitsTick(context.Background())
	if got := limitOf(t, memoryDir, parent, "sb-1"); got != 1024 {
		t.Fatalf("limit inside the hysteresis band: got %d, want 1024", got)
	}

	// Past GrowAbove: doubles.
	sandbox(t, memoryDir, cpuDir, parent, "sb-1", 700*mib, 1024*mib)
	report := agent.ElasticLimitsTick(context.Background())
	if len(report.Grown) != 1 {
		t.Fatalf("grown: got %v, want this sandbox", report.Grown)
	}
	if got := limitOf(t, memoryDir, parent, "sb-1"); got != 2048 {
		t.Fatalf("limit after growth: got %d, want 2048", got)
	}

	// Again, and again, until the declared ceiling stops it.
	for want := int64(4096); want <= 8192; want *= 2 {
		sandbox(t, memoryDir, cpuDir, parent, "sb-1", int64(float64(want/2)*0.7)*mib, (want/2)*mib)
		agent.ElasticLimitsTick(context.Background())
		if got := limitOf(t, memoryDir, parent, "sb-1"); got != want {
			t.Fatalf("limit after growth: got %d, want %d", got, want)
		}
	}
	sandbox(t, memoryDir, cpuDir, parent, "sb-1", 6000*mib, 8192*mib)
	report = agent.ElasticLimitsTick(context.Background())
	if len(report.Grown) != 0 {
		t.Errorf("grown past the ceiling: %v", report.Grown)
	}
	if got := limitOf(t, memoryDir, parent, "sb-1"); got != 8192 {
		t.Errorf("limit: got %d, want the 8192 MiB ceiling", got)
	}
}

// An OOM is the emergency signal and gets a larger step than a threshold crossing.
func TestAnOOMGrowsTheBoundByMoreThanAThresholdCrossing(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := limitsNode(t, domain, parent, 100000, 100000, ElasticConfig{
		Enabled: true, StartLimitMB: 1024, GrowFactor: 2.0, EmergencyGrowFactor: 4.0,
	})
	spec := backendSpec(t, "rollout", 2)
	spec.Resources.MemoryMB = 32768
	agent.elasticRemember("sb-1", spec)

	sandbox(t, memoryDir, cpuDir, parent, "sb-1", 100*mib, 1024*mib)
	setKillCount(t, memoryDir, parent, "sb-1", 0)
	// The first sighting only records the counter: a cgroup that already says four kills
	// belongs to a sandbox that no longer exists.
	agent.ElasticLimitsTick(context.Background())
	if got := limitOf(t, memoryDir, parent, "sb-1"); got != 1024 {
		t.Fatalf("limit on first sighting: got %d, want 1024", got)
	}

	setKillCount(t, memoryDir, parent, "sb-1", 1)
	report := agent.ElasticLimitsTick(context.Background())
	if len(report.OOMed) != 1 {
		t.Fatalf("oomed: got %v, want this sandbox", report.OOMed)
	}
	if got := limitOf(t, memoryDir, parent, "sb-1"); got != 4096 {
		t.Errorf("limit after an OOM: got %d, want 4096 (4x, not 2x)", got)
	}
}

// A shrink has to survive the dwell before it happens, and the dwell is the whole point:
// an agent between tool calls looks idle and is about to allocate again.
func TestAShrinkWaitsForTheDwellAndThenHalves(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := limitsNode(t, domain, parent, 100000, 100000, ElasticConfig{
		Enabled: true, StartLimitMB: 1024, ShrinkBelow: 0.20, ShrinkDwellS: 60, ShrinkFactor: 0.5,
	})
	spec := backendSpec(t, "rollout", 2)
	spec.Resources.MemoryMB = 32768
	agent.elasticRemember("sb-1", spec)

	// Idle, but only just: the clock starts now.
	sandbox(t, memoryDir, cpuDir, parent, "sb-1", 100*mib, 8192*mib)
	agent.ElasticLimitsTick(context.Background())
	if got := limitOf(t, memoryDir, parent, "sb-1"); got != 8192 {
		t.Fatalf("limit before the dwell: got %d, want 8192", got)
	}
	if _, running := agent.elastic.lowSince["sb-1"]; !running {
		t.Fatal("the dwell clock is not running")
	}

	// Half the dwell: still nothing.
	agent.elastic.lowSince["sb-1"] = time.Now().Add(-30 * time.Second)
	agent.ElasticLimitsTick(context.Background())
	if got := limitOf(t, memoryDir, parent, "sb-1"); got != 8192 {
		t.Fatalf("limit halfway through the dwell: got %d, want 8192", got)
	}

	// The whole dwell: halved, and not fitted to the measurement.
	agent.elastic.lowSince["sb-1"] = time.Now().Add(-61 * time.Second)
	report := agent.ElasticLimitsTick(context.Background())
	if len(report.Shrunk) != 1 {
		t.Fatalf("shrunk: got %v, want this sandbox", report.Shrunk)
	}
	if got := limitOf(t, memoryDir, parent, "sb-1"); got != 4096 {
		t.Errorf("limit after the dwell: got %d, want 4096, not the 100 MiB it is using", got)
	}
}

// A shrink never goes below the guarantee, and never below what the sandbox is actually
// holding -- page cache included, because the bound charges it.
func TestAShrinkStopsAtTheGuaranteeAndAtWhatIsCharged(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := limitsNode(t, domain, parent, 100000, 100000, ElasticConfig{
		Enabled: true, StartLimitMB: 4096, ShrinkBelow: 0.20, ShrinkDwellS: 1, ShrinkFactor: 0.5,
		ReleaseMarginMB: 256, MinLimitMB: 256,
	})
	spec := backendSpec(t, "rollout", 2)
	spec.Resources.MemoryMB = 32768
	agent.elasticRemember("sb-1", spec)

	// Using 3 GiB, well under a fifth of 8 GiB, so a shrink is allowed. It must stop at
	// the 4 GiB guarantee even though halving would go to 4096 -- and even though the
	// measurement would let it go to 3 GiB.
	sandbox(t, memoryDir, cpuDir, parent, "sb-1", 3072*mib, 32768*mib)
	agent.ElasticLimitsTick(context.Background())
	got := limitOf(t, memoryDir, parent, "sb-1")
	if got < 4096 {
		t.Errorf("limit: got %d, want at least the 4096 MiB guarantee", got)
	}

	// With usage above the guarantee, the floor is what is charged plus the margin: a
	// bound below the pages is not a slowdown, it is a reclaim of the files the next
	// command is about to read.
	sandbox(t, memoryDir, cpuDir, parent, "sb-1", 6000*mib, 32768*mib)
	delete(agent.elastic.lowSince, "sb-1")
	agent.ElasticLimitsTick(context.Background())
	var total int64 = 6000 + 256
	for got := limitOf(t, memoryDir, parent, "sb-1"); got > 8192; {
		if got < total {
			t.Fatalf("limit: got %d, want at least %d (usage + margin)", got, total)
		}
		agent.elastic.lowSince["sb-1"] = time.Now().Add(-61 * time.Second)
		agent.ElasticLimitsTick(context.Background())
		next := limitOf(t, memoryDir, parent, "sb-1")
		if next >= got {
			break
		}
		got = next
	}
	if got < total {
		t.Errorf("final limit: got %d, want at least %d", got, total)
	}
}

// Growth spends what the machine can back, not what the ledger charged, and when there is
// not enough it takes back a loan instead of taking from a working set.
func TestGrowthTakesBackAnUnusedLoanRatherThanShrinkingAWorkingSet(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	// A 20 GiB subtree: two sandboxes at 8 GiB leave 4 GiB, and growth wants 8.
	agent := limitsNode(t, domain, parent, 100000, 20480, ElasticConfig{
		Enabled: true, StartLimitMB: 1024, GrowAbove: 0.60, GrowFactor: 2.0, SafetyFraction: 0.05,
	})

	lender := backendSpec(t, "rollout", 2)
	lender.Resources.MemoryMB = 32768
	agent.elasticRemember("lender", lender)
	borrower := backendSpec(t, "rollout", 2)
	borrower.Resources.MemoryMB = 32768
	agent.elasticRemember("borrower", borrower)

	// The lender borrowed up to 8 GiB and is using almost none of it.
	sandbox(t, memoryDir, cpuDir, parent, "lender", 512*mib, 8192*mib)
	// The borrower is at 8 GiB and needs more; it is using all of what it has.
	sandbox(t, memoryDir, cpuDir, parent, "borrower", 6000*mib, 8192*mib)

	report := agent.ElasticLimitsTick(context.Background())

	if got := limitOf(t, memoryDir, parent, "lender"); got != 1024 {
		t.Errorf("lender limit: got %d, want the 1024 MiB guarantee it was not using", got)
	}
	if got := limitOf(t, memoryDir, parent, "borrower"); got <= 8192 {
		t.Errorf("borrower limit: got %d, want it to have grown on what the lender gave back", got)
	}
	if len(report.Grown) != 1 {
		t.Errorf("grown: got %v, want the borrower", report.Grown)
	}
}

// Preemption is not a shrink: a sandbox spending its loan keeps it.
func TestASandboxSpendingItsLoanKeepsIt(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := limitsNode(t, domain, parent, 100000, 16384, ElasticConfig{
		Enabled: true, StartLimitMB: 2048, GrowAbove: 0.60, GrowFactor: 2.0, SafetyFraction: 0.05,
	})
	spec := backendSpec(t, "rollout", 2)
	spec.Resources.MemoryMB = 32768
	agent.elasticRemember("spender", spec)
	agent.elasticRemember("claimer", spec)

	// Both are above their guarantee. The spender is using more than its guarantee, so its
	// loan is not unused and must survive; the claimer wants to grow and cannot.
	sandbox(t, memoryDir, cpuDir, parent, "spender", 5000*mib, 8192*mib)
	sandbox(t, memoryDir, cpuDir, parent, "claimer", 6000*mib, 8192*mib)

	agent.ElasticLimitsTick(context.Background())

	if got := limitOf(t, memoryDir, parent, "spender"); got != 8192 {
		t.Errorf("spender limit: got %d, want 8192 -- it is using 5000 of a 2048 guarantee", got)
	}
}

// The outcome that matters: a sandbox killed again with nowhere left to grow is ended and
// said to have been ended for memory, rather than left to produce a starved trajectory.
func TestASandboxKilledAtItsCeilingIsEndedAndSaidSo(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := limitsNode(t, domain, parent, 100000, 100000, ElasticConfig{
		Enabled: true, StartLimitMB: 1024,
	})
	spec := backendSpec(t, "rollout", 2)
	spec.Resources.MemoryMB = 2048
	agent.elasticRemember("sb-1", spec)
	// The lifecycle has to hold it, or ending it is a no-op and the test proves nothing.
	agent.lifecycle.Reserve(node.Grant{LeaseID: "lease-1", Class: "rollout"}, "owner", "one_shot", time.Hour)
	if _, claimed := agent.lifecycle.Claim("lease-1"); !claimed {
		t.Fatal("the test needs the reserved grant")
	}
	if !agent.lifecycle.AdoptCreated("lease-1", backend.Handle{Backend: "stub", SandboxID: "sb-1"}, providerStub{}) {
		t.Fatal("the test needs the lifecycle to hold this sandbox")
	}

	// At the ceiling already.
	sandbox(t, memoryDir, cpuDir, parent, "sb-1", 2000*mib, 2048*mib)
	setKillCount(t, memoryDir, parent, "sb-1", 0)
	report := agent.ElasticLimitsTick(context.Background())
	if len(report.Ended) != 0 {
		t.Fatalf("ended on the first sighting: %v", report.Ended)
	}

	// The first kill with nowhere to grow is recorded, not acted on: one kill may be a
	// runaway child rather than the task itself.
	setKillCount(t, memoryDir, parent, "sb-1", 1)
	report = agent.ElasticLimitsTick(context.Background())
	if len(report.Ended) != 0 {
		t.Fatalf("ended on the first kill: %v", report.Ended)
	}
	if !agent.elastic.exhausted["sb-1"] {
		t.Fatal("the sandbox was not marked as having nowhere left to grow")
	}

	// The second one ends it.
	setKillCount(t, memoryDir, parent, "sb-1", 2)
	report = agent.ElasticLimitsTick(context.Background())
	if len(report.Ended) != 1 || report.Ended[0] != "sb-1" {
		t.Fatalf("ended: got %v, want this sandbox", report.Ended)
	}
	reason, known := agent.lifecycle.ExitReasonOf("sb-1")
	if !known || reason != node.ExitOOM {
		t.Errorf("exit reason: got %q (known=%v), want %q", reason, known, node.ExitOOM)
	}
}

// The bounds the machine has to back stay inside what it has, whatever the ledger charged.
//
// This is the whole reason a sandbox is created small: the ledger may sell the average, but
// the kernel's hard bounds are not a statistic, and a wave of them has to fit.
func TestTheSumOfTheBoundsStaysInsideTheSubtree(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	const budgetMB = 40960
	agent := limitsNode(t, domain, parent, 100000, budgetMB, ElasticConfig{
		Enabled: true, StartLimitMB: 1024, GrowAbove: 0.60, GrowFactor: 2.0, SafetyFraction: 0.05,
	})

	// Twenty sandboxes all demanding growth at once.
	ids := make([]string, 0, 20)
	for i := 0; i < 20; i++ {
		id := "sb-" + strconv.Itoa(i)
		spec := backendSpec(t, "rollout", 2)
		spec.Resources.MemoryMB = 32768
		agent.elasticRemember(id, spec)
		sandbox(t, memoryDir, cpuDir, parent, id, 900*mib, 1024*mib)
		ids = append(ids, id)
	}
	for i := 0; i < 6; i++ {
		for _, id := range ids {
			sandbox(t, memoryDir, cpuDir, parent, id, 900*mib, limitOf(t, memoryDir, parent, id)*mib)
		}
		agent.ElasticLimitsTick(context.Background())
	}

	report := agent.ElasticLimitsTick(context.Background())
	if report.LimitSumMB > budgetMB {
		t.Errorf("sum of bounds: got %d MiB, want no more than the %d MiB subtree", report.LimitSumMB, budgetMB)
	}
	if report.SpareMB < 0 {
		t.Errorf("spare: got %d MiB, want the safety share to have been kept", report.SpareMB)
	}
}

// A cgroup this node does not own contributes its memory to what the machine carries and
// nothing to what the node reports about its own sandboxes.
//
// The counters are cumulative and a cgroup outlives its container, so a domain-wide OOM
// total counts kills that belong to sandboxes that no longer exist. That number never goes
// down, points at nothing, and reads as an ongoing fault -- which is exactly how 71 kills
// sat in a log line for two hours without an owner.
func TestAForeignCgroupCountsItsMemoryAndNotItsFaults(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := limitsNode(t, domain, parent, 100000, 100000, ElasticConfig{
		Enabled: true, StartLimitMB: 1024,
	})

	spec := backendSpec(t, "rollout", 2)
	spec.Resources.MemoryMB = 32768
	agent.elasticRemember("mine", spec)
	// The lifecycle has to hold it, or the audit drops it as a sandbox that is not running
	// and it becomes foreign too.
	agent.lifecycle.Reserve(node.Grant{LeaseID: "lease-1", Class: "rollout"}, "owner", "one_shot", time.Hour)
	if _, claimed := agent.lifecycle.Claim("lease-1"); !claimed {
		t.Fatal("the test needs the reserved grant")
	}
	if !agent.lifecycle.AdoptCreated("lease-1", backend.Handle{Backend: "stub", SandboxID: "mine"}, providerStub{}) {
		t.Fatal("the test needs the lifecycle to hold this sandbox")
	}
	sandbox(t, memoryDir, cpuDir, parent, "mine", 512*mib, 1024*mib)
	setKillCount(t, memoryDir, parent, "mine", 3)

	// An orphan: in the domain, owned by nobody, carrying the kill count of a sandbox that
	// is long gone.
	sandbox(t, memoryDir, cpuDir, parent, "orphan", 2048*mib, 8192*mib)
	setKillCount(t, memoryDir, parent, "orphan", 68)

	agent.ElasticTick(context.Background())
	report := agent.ElasticTick(context.Background())

	if report.Foreign != 1 {
		t.Errorf("foreign count: got %d, want 1", report.Foreign)
	}
	if report.ForeignMB < 2048 {
		t.Errorf("foreign memory: got %d MiB, want at least the 2048 MiB it holds", report.ForeignMB)
	}
	// The kills are not this run's and must not be reported as though they were.
	if report.OOMKills != 3 {
		t.Errorf("oom kills: got %d, want 3 -- the orphan's 68 belong to nobody", report.OOMKills)
	}
	// But the memory is real wherever it sits, so it stays inside the observed total that
	// protects admission.
	if report.ObservedMB < 2048+512 {
		t.Errorf("observed: got %d MiB, want the orphan's memory to count against the node", report.ObservedMB)
	}
}

// The guards around reaping, which are the part that must never be wrong: a live sandbox's
// cgroup and an orphan holding real memory both survive.
//
// Whether the removal itself works is a question about the kernel, so it is left to the
// integration test, where there is a real cgroup filesystem to remove one from.
func TestReapingSparesLiveSandboxesAndOrphansHoldingMemory(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := limitsNode(t, domain, parent, 100000, 100000, ElasticConfig{
		Enabled: true, StartLimitMB: 1024, ReapForeign: true, ReapUsageMB: 64,
	})

	spec := backendSpec(t, "rollout", 2)
	spec.Resources.MemoryMB = 32768
	agent.elasticRemember("mine", spec)
	// Held by the lifecycle, or the audit drops it and it stops being this node's sandbox.
	agent.lifecycle.Reserve(node.Grant{LeaseID: "lease-1", Class: "rollout"}, "owner", "one_shot", time.Hour)
	if _, claimed := agent.lifecycle.Claim("lease-1"); !claimed {
		t.Fatal("the test needs the reserved grant")
	}
	if !agent.lifecycle.AdoptCreated("lease-1", backend.Handle{Backend: "stub", SandboxID: "mine"}, providerStub{}) {
		t.Fatal("the test needs the lifecycle to hold this sandbox")
	}
	sandbox(t, memoryDir, cpuDir, parent, "mine", 512*mib, 1024*mib)
	// Empty and owned by nothing: this one is reaped, and the reap is what proves the
	// guards were evaluated rather than the whole domain being skipped.
	sandbox(t, memoryDir, cpuDir, parent, "empty-orphan", 0, 1024*mib)
	// Nobody owns it, but it holds real memory. Removing the directory would not free
	// that memory -- it would only stop anyone from being able to say whose it is.
	sandbox(t, memoryDir, cpuDir, parent, "heavy-orphan", 4096*mib, 8192*mib)

	agent.ElasticTick(context.Background())
	report := agent.ElasticTick(context.Background())

	if _, err := os.Stat(filepath.Join(memoryDir, parent, "heavy-orphan")); err != nil {
		t.Errorf("an orphan holding 4 GiB was removed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(memoryDir, parent, "mine")); err != nil {
		t.Errorf("a live sandbox's cgroup was removed: %v", err)
	}
	if report.Foreign != 2 {
		t.Errorf("foreign count: got %d, want 2 -- the heavy orphan is still counted", report.Foreign)
	}
}

// Reaping is off unless a deployment turns it on: removing directories from a host's
// cgroup tree is not a thing a node should start doing because a flag defaulted to yes.
func TestOrphansAreLeftAloneUnlessReapingIsEnabled(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := limitsNode(t, domain, parent, 100000, 100000, ElasticConfig{
		Enabled: true, StartLimitMB: 1024,
	})
	sandbox(t, memoryDir, cpuDir, parent, "empty-orphan", 0, 1024*mib)

	agent.ElasticTick(context.Background())

	if _, err := os.Stat(filepath.Join(memoryDir, parent, "empty-orphan")); err != nil {
		t.Errorf("an orphan was removed with reaping off: %v", err)
	}
}

// A container between create and start is not an orphan.
//
// Docker creates a container's cgroups in every hierarchy at create time and writes the
// process into them at start. In between, the directory has no tasks and holds nothing and
// is indistinguishable from the residue of a destroyed container -- which is how a node
// removed one out from under a start and the start failed with ENODEV against a cgroup
// that no longer existed.
//
// Nothing is removed on the strength of one observation, so the window has to close before
// the node will act, and a runtime that claims the container is never contradicted.
func TestAContainerBeingCreatedIsNotReaped(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := limitsNode(t, domain, parent, 100000, 100000, ElasticConfig{
		Enabled: true, StartLimitMB: 1024, ReapForeign: true, ReapUsageMB: 64, ReapDwellS: 60,
	})
	sandbox(t, memoryDir, cpuDir, parent, "starting", 0, 1024*mib)

	now := time.Now()
	// The observable for "this is now a candidate" is that the node went and asked the
	// runtime what it owns. It only does that when something has waited out the dwell --
	// the removal itself is a question for the kernel and is covered by the integration
	// test, because a tmpfs directory full of files cannot be rmdir'd.
	agent.reapForeign(mustChildren(t, domain, parent), now)
	if !agent.elastic.ownedAt.IsZero() {
		t.Fatal("the node asked the runtime about a cgroup it had only just seen")
	}
	agent.reapForeign(mustChildren(t, domain, parent), now.Add(30*time.Second))
	if !agent.elastic.ownedAt.IsZero() {
		t.Fatal("the node asked the runtime before the dwell had passed")
	}
	agent.reapForeign(mustChildren(t, domain, parent), now.Add(61*time.Second))
	if agent.elastic.ownedAt.IsZero() {
		t.Fatal("the node never treated the orphan as a candidate")
	}
	// And the cgroup is still there: the guards decided, nothing was destroyed.
	if _, err := os.Stat(filepath.Join(memoryDir, parent, "starting")); err != nil {
		t.Errorf("the cgroup was removed: %v", err)
	}
}

// A runtime that says the container is its own is believed, whatever the cgroup looks like.
//
// A warm-pool entry has no lease and no tasks; a stopped container has no process. Both
// look exactly like garbage from the cgroup, and deleting either is destroying a sandbox.
func TestACgroupTheRuntimeOwnsIsNeverReaped(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := ownedNode(t, domain, parent, "theirs")
	sandbox(t, memoryDir, cpuDir, parent, "theirs", 0, 1024*mib)
	sandbox(t, memoryDir, cpuDir, parent, "nobody", 0, 1024*mib)

	now := time.Now()
	agent.reapForeign(mustChildren(t, domain, parent), now)
	reaped := agent.reapForeign(mustChildren(t, domain, parent), now.Add(120*time.Second))

	for _, id := range reaped {
		if id == "theirs" {
			t.Errorf("the runtime's own container was reaped")
		}
	}
	if _, err := os.Stat(filepath.Join(memoryDir, parent, "theirs")); err != nil {
		t.Errorf("the runtime's container lost its cgroup: %v", err)
	}
}

// Not knowing what the runtime owns is not permission to delete.
func TestARuntimeThatWillNotAnswerStopsAllReaping(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := brokenNode(t, domain, parent)
	sandbox(t, memoryDir, cpuDir, parent, "nobody", 0, 1024*mib)

	now := time.Now()
	agent.reapForeign(mustChildren(t, domain, parent), now)
	if got := agent.reapForeign(mustChildren(t, domain, parent), now.Add(120*time.Second)); len(got) != 0 {
		t.Errorf("reaped while the runtime would not answer: %v", got)
	}
	if _, err := os.Stat(filepath.Join(memoryDir, parent, "nobody")); err != nil {
		t.Errorf("removed a cgroup without knowing who owned it: %v", err)
	}
}

func mustChildren(t *testing.T, domain cgroup.Domain, parent string) []cgroup.Child {
	t.Helper()
	children, err := domain.Children(parent)
	if err != nil {
		t.Fatalf("read the domain: %v", err)
	}
	return children
}

// ownedBackend is a runtime that says which containers are its own, and brokenBackend one
// that will not answer. Both stand in for the property that decides reaping: only the
// runtime knows whether a cgroup belongs to something.
type ownedBackend struct {
	providerStub
	owned []string
	err   error
}

func (b ownedBackend) Reconcile(context.Context) ([]backend.Reconciled, error) {
	if b.err != nil {
		return nil, b.err
	}
	out := make([]backend.Reconciled, 0, len(b.owned))
	for _, id := range b.owned {
		out = append(out, backend.Reconciled{
			Handle: backend.Handle{Backend: "stub", SandboxID: id},
		})
	}
	return out, nil
}

func (b ownedBackend) FindByIdempotencyKey(context.Context, string) (backend.Reconciled, bool, error) {
	return backend.Reconciled{}, false, nil
}

// nodeWithBackend builds a reaping node over a runtime that reports the given containers.
func nodeWithBackend(t *testing.T, domain cgroup.Domain, parent string, be backend.Backend) *Node {
	t.Helper()
	gate, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{MemoryMB: 100000, CPUMillis: 100000},
		Classes:  map[string]node.ClassShare{"rollout": {Guaranteed: 0.9}, "grader": {Guaranteed: 0.1}},
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
		Backends: []backend.Backend{be},
		Pressure: func() node.Pressure { return node.Pressure{SampledAt: time.Now(), Valid: true} },
		Elastic: ElasticConfig{
			Enabled: true, StartLimitMB: 1024,
			ReapForeign: true, ReapUsageMB: 64, ReapDwellS: 60,
		},
		Domain: domain, SandboxParent: parent, SandboxBudgetMB: 100000,
	})
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	return agent
}

func ownedNode(t *testing.T, domain cgroup.Domain, parent string, owned ...string) *Node {
	t.Helper()
	return nodeWithBackend(t, domain, parent, ownedBackend{owned: owned})
}

func brokenNode(t *testing.T, domain cgroup.Domain, parent string) *Node {
	t.Helper()
	return nodeWithBackend(t, domain, parent, ownedBackend{err: errors.New("daemon unreachable")})
}
