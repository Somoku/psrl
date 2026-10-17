package pressure

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/node"
)

// -- the domain is the subtree, not its root ----------------------------------

// domainFixture builds a v1 root and one subtree under it, with different usage and
// limits, so a reader that reads the root is distinguishable from one that reads the
// subtree.
type domainFixture struct {
	root    string
	cpu     string
	memory  string
	cpuset  string
	subtree string
}

func newDomainFixture(t *testing.T) domainFixture {
	t.Helper()
	root := t.TempDir()
	fixture := domainFixture{
		root:    root,
		cpu:     filepath.Join(root, "cpu"),
		memory:  filepath.Join(root, "memory"),
		cpuset:  filepath.Join(root, "cpuset"),
		subtree: "psrl-sandboxes",
	}
	for _, dir := range []string{fixture.cpu, fixture.memory, fixture.cpuset} {
		if err := os.MkdirAll(filepath.Join(dir, fixture.subtree), 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
	}
	// The root: 800 of 1000 bytes used, one core, eight CPUs.
	writeFakeFile(t, fixture.memory, "memory.usage_in_bytes", "800\n")
	writeFakeFile(t, fixture.memory, "memory.limit_in_bytes", "1000\n")
	writeFakeFile(t, fixture.cpu, "cpuacct.usage", "0\n")
	writeFakeFile(t, fixture.cpu, "cpu.cfs_quota_us", "-1\n")
	writeFakeFile(t, fixture.cpu, "cpu.cfs_period_us", "100000\n")
	writeFakeFile(t, fixture.cpuset, "cpuset.cpus", "0-7\n")
	// The subtree: 200 of 2000, and the same CPU files.
	writeFakeFile(t, filepath.Join(fixture.memory, fixture.subtree), "memory.usage_in_bytes", "200\n")
	writeFakeFile(t, filepath.Join(fixture.memory, fixture.subtree), "memory.limit_in_bytes", "2000\n")
	writeFakeFile(t, filepath.Join(fixture.cpu, fixture.subtree), "cpuacct.usage", "0\n")
	writeFakeFile(t, filepath.Join(fixture.cpu, fixture.subtree), "cpu.cfs_quota_us", "-1\n")
	writeFakeFile(t, filepath.Join(fixture.cpu, fixture.subtree), "cpu.cfs_period_us", "100000\n")
	writeFakeFile(t, filepath.Join(fixture.cpuset, fixture.subtree), "cpuset.cpus", "0-3\n")
	return fixture
}

func TestADomainReaderReadsTheSubtreeNotItsRoot(t *testing.T) {
	fixture := newDomainFixture(t)

	src, err := v1Domain(fixture.cpu, fixture.memory, fixture.cpuset, fixture.subtree)
	if err != nil {
		t.Fatalf("v1Domain: %v", err)
	}
	reader, err := newReaderFrom(src)
	if err != nil {
		t.Fatalf("newReaderFrom: %v", err)
	}

	sample := reader.Read()

	// 200 of 2000 is 10%; the root would have reported 80%.
	if sample.MemUsedPct != 0.1 {
		t.Errorf("mem_used_pct: got %.3f, want 0.1 (the subtree, not the root)", sample.MemUsedPct)
	}
}

func TestADomainThatDoesNotExistIsAnError(t *testing.T) {
	fixture := newDomainFixture(t)

	_, err := v1Domain(fixture.cpu, fixture.memory, fixture.cpuset, "not-a-subtree")
	if err == nil {
		t.Fatal("a missing domain must be an error, not a fallback to the root")
	}
}

func TestADomainWithoutItsCPUCounterIsAnError(t *testing.T) {
	fixture := newDomainFixture(t)
	if err := os.Remove(filepath.Join(fixture.cpu, fixture.subtree, "cpuacct.usage")); err != nil {
		t.Fatal(err)
	}

	if _, err := v1Domain(fixture.cpu, fixture.memory, fixture.cpuset, fixture.subtree); err == nil {
		t.Fatal("a subtree with no CPU counter must be refused")
	}
}

func TestAV2DomainIsAlsoJoinedRatherThanFallenBack(t *testing.T) {
	root := t.TempDir()
	subtree := filepath.Join(root, "psrl-sandboxes")
	if err := os.MkdirAll(subtree, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFakeFile(t, subtree, "cpu.stat", "usage_usec 0\n")
	writeFakeFile(t, subtree, "memory.current", "100\n")
	writeFakeFile(t, subtree, "memory.max", "1000\n")

	src, err := v2Domain(root, "psrl-sandboxes")
	if err != nil {
		t.Fatalf("v2Domain: %v", err)
	}
	reader, err := newReaderFrom(src)
	if err != nil {
		t.Fatalf("newReaderFrom: %v", err)
	}

	if sample := reader.Read(); sample.MemUsedPct != 0.1 {
		t.Errorf("mem_used_pct: got %.3f, want 0.1", sample.MemUsedPct)
	}
}

// -- the CPU denominator ------------------------------------------------------

func TestTheQuotaIsTheCPUdenominator(t *testing.T) {
	fixture := newDomainFixture(t)
	// 180 cores of quota, on a machine with more.
	writeFakeFile(t, filepath.Join(fixture.cpu, fixture.subtree), "cpu.cfs_quota_us", "18000000\n")
	writeFakeFile(t, filepath.Join(fixture.cpu, fixture.subtree), "cpu.cfs_period_us", "100000\n")

	src, err := v1Domain(fixture.cpu, fixture.memory, fixture.cpuset, fixture.subtree)
	if err != nil {
		t.Fatalf("v1Domain: %v", err)
	}
	reader, err := newReaderFrom(src)
	if err != nil {
		t.Fatalf("newReaderFrom: %v", err)
	}

	if reader.cores != 180 {
		t.Errorf("cores: got %.0f, want the subtree's 180-core quota", reader.cores)
	}
}

func TestTheCPUSetIsTheDenominatorWithoutAQuota(t *testing.T) {
	fixture := newDomainFixture(t)

	src, err := v1Domain(fixture.cpu, fixture.memory, fixture.cpuset, fixture.subtree)
	if err != nil {
		t.Fatalf("v1Domain: %v", err)
	}
	reader, err := newReaderFrom(src)
	if err != nil {
		t.Fatalf("newReaderFrom: %v", err)
	}

	if reader.cores != 4 {
		t.Errorf("cores: got %.0f, want the subtree's four CPUs", reader.cores)
	}
}

func TestTheMachineIsTheDenominatorWithNeither(t *testing.T) {
	root := t.TempDir()
	dir := root
	writeFakeFile(t, dir, "cpuacct.usage", "0\n")
	writeFakeFile(t, dir, "memory.usage_in_bytes", "0\n")
	writeFakeFile(t, dir, "memory.limit_in_bytes", "1000\n")

	reader, err := newReaderFrom(&v1Source{cpuDir: dir, memoryDir: dir})
	if err != nil {
		t.Fatalf("newReaderFrom: %v", err)
	}

	if reader.cores != float64(hostCores()) {
		t.Errorf("cores: got %.0f, want the machine's %d", reader.cores, hostCores())
	}
}

func TestTheUnifiedQuotaIsTheCPUdenominator(t *testing.T) {
	dir := t.TempDir()
	writeFakeFile(t, dir, "cpu.stat", "usage_usec 0\n")
	writeFakeFile(t, dir, "cpu.max", "18000000 100000\n")
	writeFakeFile(t, dir, "memory.current", "0\n")
	writeFakeFile(t, dir, "memory.max", "1000\n")

	reader, err := newReaderFrom(&v2Source{dir: dir})
	if err != nil {
		t.Fatalf("newReaderFrom: %v", err)
	}

	if reader.cores != 180 {
		t.Errorf("cores: got %.0f, want 180", reader.cores)
	}
}

func TestCountCPUsHandlesRangesAndSingletons(t *testing.T) {
	for _, testCase := range []struct {
		list string
		want int
	}{
		{"0", 1},
		{"0-7", 8},
		{"0-3,8,10-11", 7},
		{"", 0},
	} {
		got, ok := countCPUs(testCase.list)
		if testCase.want == 0 {
			if ok {
				t.Errorf("%q: expected no count", testCase.list)
			}
			continue
		}
		if !ok || got != testCase.want {
			t.Errorf("%q: got %d (ok=%v), want %d", testCase.list, got, ok, testCase.want)
		}
	}
}

// -- validity -----------------------------------------------------------------

func TestAReadThatCannotSeeTheDomainIsInvalid(t *testing.T) {
	dir := t.TempDir()
	writeFakeFile(t, dir, "cpuacct.usage", "0\n")
	writeFakeFile(t, dir, "memory.usage_in_bytes", "10\n")
	writeFakeFile(t, dir, "memory.limit_in_bytes", "100\n")
	reader, err := newReaderFrom(&v1Source{cpuDir: dir, memoryDir: dir})
	if err != nil {
		t.Fatalf("newReaderFrom: %v", err)
	}
	if sample := reader.Read(); !sample.Valid {
		t.Fatal("a readable domain must produce a valid sample")
	}

	// The files vanish under the reader, which is what a cgroup removed by an
	// operator or an unmounted hierarchy looks like.
	if err := os.Remove(filepath.Join(dir, "memory.usage_in_bytes")); err != nil {
		t.Fatal(err)
	}

	sample := reader.Read()
	if sample.Valid {
		t.Error("an unreadable domain must not produce a valid sample")
	}
	if sample.MemUsedPct != 0 {
		t.Errorf("an unreadable domain must not report a fraction, got %.3f", sample.MemUsedPct)
	}
	if sample.SampledAt.IsZero() {
		t.Error("even an invalid sample carries when it was taken")
	}
}

// -- sampler ------------------------------------------------------------------

// newTestSampler builds a sampler over a fixture that reads cleanly.
func newTestSampler(t *testing.T, every time.Duration) (*Sampler, node.Pressure) {
	t.Helper()
	dir := t.TempDir()
	writeFakeFile(t, dir, "cpuacct.usage", "0\n")
	writeFakeFile(t, dir, "memory.usage_in_bytes", "250\n")
	writeFakeFile(t, dir, "memory.limit_in_bytes", "1000\n")
	reader, err := newReaderFrom(&v1Source{cpuDir: dir, memoryDir: dir})
	if err != nil {
		t.Fatalf("newReaderFrom: %v", err)
	}
	sampler, err := NewSampler(reader, every)
	if err != nil {
		t.Fatalf("NewSampler: %v", err)
	}
	return sampler, reader.Read()
}

func TestASamplerNeedsAReader(t *testing.T) {
	if _, err := NewSampler(nil, time.Second); err == nil {
		t.Fatal("a sampler without a reader must be refused")
	}
}

func TestAnOldSampleIsNotValid(t *testing.T) {
	sampler, _ := newTestSampler(t, time.Second)
	sampler.publish(node.Pressure{MemUsedPct: 0.25, SampledAt: time.Now().Add(-time.Minute), Valid: true})

	if sample := sampler.Sample(); sample.Valid {
		t.Error("a sample older than three intervals must not be reported as valid")
	}
}

func TestAFreshSampleIsValid(t *testing.T) {
	sampler, _ := newTestSampler(t, time.Second)
	sampler.publish(node.Pressure{MemUsedPct: 0.25, SampledAt: time.Now(), Valid: true})

	sample := sampler.Sample()
	if !sample.Valid || sample.MemUsedPct != 0.25 {
		t.Errorf("a fresh sample: got %+v", sample)
	}
}

func TestReadingTheSamplerDoesNotTouchTheDomain(t *testing.T) {
	// The point of the cache: the node reads pressure from admission and from every
	// report, and a read that measured would difference the CPU counter over the gap
	// between two RPCs. Removing the files proves no read happens.
	sampler, sample := newTestSampler(t, time.Hour)
	sampler.publish(sample)

	for i := 0; i < 100; i++ {
		if got := sampler.Sample(); !got.Valid || got.MemUsedPct != sample.MemUsedPct {
			t.Fatalf("sample %d changed: %+v", i, got)
		}
	}
}

func TestStartTakesTheFirstSampleBeforeReturning(t *testing.T) {
	sampler, _ := newTestSampler(t, time.Hour)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sampler.Start(ctx)

	// A node that starts serving immediately must have a memory level, not a zero
	// that reads as an empty domain.
	if sample := sampler.Sample(); !sample.Valid || sample.MemUsedPct != 0.25 {
		t.Errorf("the first sample: got %+v", sample)
	}
}
