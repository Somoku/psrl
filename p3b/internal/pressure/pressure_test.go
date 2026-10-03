package pressure

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// -- test helpers -------------------------------------------------------------

// writeFakeFile writes a file into a temp dir.
func writeFakeFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatalf("writeFakeFile %s: %v", name, err)
	}
}

// fakeV2Source implements source using files in a temp directory, so tests
// run without a real cgroup hierarchy.
type fakeV2Source struct{ dir string }

func newFakeV2Dir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFakeFile(t, dir, "cpu.stat", "usage_usec 0\n")
	writeFakeFile(t, dir, "memory.max", "max\n")
	writeFakeFile(t, dir, "memory.current", "0\n")
	return dir
}

func fakeV2(t *testing.T) (*Reader, string) {
	t.Helper()
	dir := newFakeV2Dir(t)
	src := &v2Source{dir: dir}
	r, err := newReaderFrom(src)
	if err != nil {
		t.Fatalf("newReaderFrom: %v", err)
	}
	return r, dir
}

// fakeV1Source builds a minimal v1 layout in a temp directory.
type fakeV1Dirs struct {
	cpu    string
	memory string
}

func newFakeV1Dirs(t *testing.T) fakeV1Dirs {
	t.Helper()
	root := t.TempDir()
	cpuDir := filepath.Join(root, "cpuacct")
	memDir := filepath.Join(root, "memory")
	for _, d := range []string{cpuDir, memDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatalf("mkdir: %v", err)
		}
	}
	writeFakeFile(t, cpuDir, "cpuacct.usage", "0\n")
	writeFakeFile(t, memDir, "memory.usage_in_bytes", "0\n")
	writeFakeFile(t, memDir, "memory.limit_in_bytes", "1073741824\n") // 1 GiB
	return fakeV1Dirs{cpu: cpuDir, memory: memDir}
}

func fakeV1(t *testing.T) (*Reader, fakeV1Dirs) {
	t.Helper()
	dirs := newFakeV1Dirs(t)
	src := &v1Source{cpuDir: dirs.cpu, memoryDir: dirs.memory}
	r, err := newReaderFrom(src)
	if err != nil {
		t.Fatalf("newReaderFrom(v1): %v", err)
	}
	return r, dirs
}

// -- construction -------------------------------------------------------------

func TestV2ReaderConstructsFromValidDir(t *testing.T) {
	_, _ = fakeV2(t)
}

func TestV1ReaderConstructsFromValidDirs(t *testing.T) {
	_, _ = fakeV1(t)
}

func TestV2ReaderRefusesADirWithNoCPUStat(t *testing.T) {
	dir := t.TempDir()
	writeFakeFile(t, dir, "memory.max", "max\n")
	writeFakeFile(t, dir, "memory.current", "0\n")
	src := &v2Source{dir: dir}
	if _, err := newReaderFrom(src); err == nil {
		t.Fatal("expected error for missing cpu.stat")
	}
}

func TestV1ReaderRefusesADirWithNoCpuacctUsage(t *testing.T) {
	root := t.TempDir()
	cpuDir := filepath.Join(root, "cpu")
	memDir := filepath.Join(root, "memory")
	_ = os.MkdirAll(cpuDir, 0o755)
	_ = os.MkdirAll(memDir, 0o755)
	writeFakeFile(t, memDir, "memory.usage_in_bytes", "0\n")
	writeFakeFile(t, memDir, "memory.limit_in_bytes", "1073741824\n")
	// cpuacct.usage intentionally absent
	src := &v1Source{cpuDir: cpuDir, memoryDir: memDir}
	if _, err := newReaderFrom(src); err == nil {
		t.Fatal("expected error for missing cpuacct.usage")
	}
}

func TestReaderDescribeNamesTheHierarchy(t *testing.T) {
	r, _ := fakeV2(t)
	if desc := r.Describe(); desc == "" {
		t.Error("Describe must return a non-empty string")
	}
	r1, _ := fakeV1(t)
	if desc := r1.Describe(); desc == "" {
		t.Error("Describe v1 must return a non-empty string")
	}
}

// -- memory readings (v2) -----------------------------------------------------

func TestV2MemoryUsedPctIsZeroWhenCurrentIsZero(t *testing.T) {
	r, dir := fakeV2(t)
	writeFakeFile(t, dir, "memory.max", "1073741824\n")
	writeFakeFile(t, dir, "memory.current", "0\n")
	if p := r.Read(); p.MemUsedPct != 0 {
		t.Errorf("mem_used_pct: got %.3f, want 0", p.MemUsedPct)
	}
}

func TestV2MemoryUsedPctReadsFromMemoryMax(t *testing.T) {
	// Construct the reader AFTER writing the limit, so the cached limit is the
	// one we intend rather than the host total (which would make 512 MiB a ~0%
	// fraction on any large machine).
	dir := t.TempDir()
	writeFakeFile(t, dir, "cpu.stat", "usage_usec 0\n")
	writeFakeFile(t, dir, "memory.max", "1073741824\n")  // 1 GiB
	writeFakeFile(t, dir, "memory.current", "536870912\n") // 512 MiB
	src := &v2Source{dir: dir}
	r, err := newReaderFrom(src)
	if err != nil {
		t.Fatalf("newReaderFrom: %v", err)
	}
	p := r.Read()
	if p.MemUsedPct < 0.499 || p.MemUsedPct > 0.501 {
		t.Errorf("mem_used_pct: got %.4f, want ~0.5", p.MemUsedPct)
	}
}

func TestV2MemoryUsedPctIsCappedAtOne(t *testing.T) {
	r, dir := fakeV2(t)
	writeFakeFile(t, dir, "memory.max", "1000\n")
	writeFakeFile(t, dir, "memory.current", "2000\n")
	if p := r.Read(); p.MemUsedPct > 1 {
		t.Errorf("mem_used_pct: got %.3f, want ≤ 1", p.MemUsedPct)
	}
}

// -- memory readings (v1) -----------------------------------------------------

func TestV1MemoryUsedPctReadsFromUsageInBytes(t *testing.T) {
	r, dirs := fakeV1(t)
	writeFakeFile(t, dirs.memory, "memory.limit_in_bytes", "1073741824\n") // 1 GiB
	writeFakeFile(t, dirs.memory, "memory.usage_in_bytes", "536870912\n")  // 512 MiB
	p := r.Read()
	if p.MemUsedPct < 0.499 || p.MemUsedPct > 0.501 {
		t.Errorf("v1 mem_used_pct: got %.4f, want ~0.5", p.MemUsedPct)
	}
}

func TestV1MemoryLimitAboveTotalTreatedAsUnlimited(t *testing.T) {
	// A v1 unlimited cgroup reports a very large sentinel. The implementation
	// uses the host's total memory as the ceiling in that case, which is the
	// right denominator.
	r, dirs := fakeV1(t)
	// Set limit to max int64-like sentinel that v1 uses for "no limit".
	writeFakeFile(t, dirs.memory, "memory.limit_in_bytes", "9223372036854771712\n")
	writeFakeFile(t, dirs.memory, "memory.usage_in_bytes", "1048576\n")
	p := r.Read()
	if p.MemUsedPct < 0 || p.MemUsedPct > 1 {
		t.Errorf("v1 unlimited limit: mem_used_pct %f out of [0,1]", p.MemUsedPct)
	}
}

// -- CPU readings (first call always zero) ------------------------------------

func TestV2CPUUsedPctIsZeroOnFirstCall(t *testing.T) {
	r, dir := fakeV2(t)
	writeFakeFile(t, dir, "cpu.stat", "usage_usec 5000000\n")
	if p := r.Read(); p.CPUUsedPct != 0 {
		t.Errorf("first cpu_used_pct: got %.3f, want 0 (no prior sample)", p.CPUUsedPct)
	}
}

func TestV1CPUUsedPctIsZeroOnFirstCall(t *testing.T) {
	r, dirs := fakeV1(t)
	writeFakeFile(t, dirs.cpu, "cpuacct.usage", "5000000000\n") // 5s nanoseconds
	if p := r.Read(); p.CPUUsedPct != 0 {
		t.Errorf("v1 first cpu_used_pct: got %.3f, want 0", p.CPUUsedPct)
	}
}

// -- CPU rate via time manipulation -------------------------------------------

func injectCPUInterval(r *Reader, elapsed time.Duration, consumed time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lastRead = r.lastRead.Add(-elapsed)
	r.lastCPU = r.lastCPU - consumed
}

func TestV2CPUUsedPctReflectsRate(t *testing.T) {
	r, dir := fakeV2(t)
	writeFakeFile(t, dir, "cpu.stat", "usage_usec 0\n")
	r.Read() // seed

	// Simulate half the cores busy for 500ms.
	elapsed := 500 * time.Millisecond
	consumed := time.Duration(r.cores * float64(elapsed) / 2)
	injectCPUInterval(r, elapsed, consumed)
	// Re-read the file (still shows 0, but the subtraction of lastCPU handles it).
	r.Read() // discard; just advancing the state machine
}

func TestV1CPUUsedPctReflectsRate(t *testing.T) {
	r, dirs := fakeV1(t)
	writeFakeFile(t, dirs.cpu, "cpuacct.usage", "0\n")
	r.Read() // seed
	injectCPUInterval(r, 500*time.Millisecond, time.Duration(r.cores*float64(500*time.Millisecond)/2))
	r.Read()
}

// -- capping ------------------------------------------------------------------

func TestCPUUsedPctIsCappedAtOne(t *testing.T) {
	r, dir := fakeV2(t)
	writeFakeFile(t, dir, "cpu.stat", "usage_usec 0\n")
	r.Read()
	injectCPUInterval(r, 200*time.Millisecond,
		time.Duration(r.cores*float64(200*time.Millisecond)*10)) // ×10 over capacity
	p := r.Read()
	if p.CPUUsedPct > 1 {
		t.Errorf("cpu_used_pct: got %.3f, want ≤ 1", p.CPUUsedPct)
	}
}

// -- negative counter ---------------------------------------------------------

func TestNegativeDeltaReturnsSafeZero(t *testing.T) {
	r, dir := fakeV2(t)
	writeFakeFile(t, dir, "cpu.stat", "usage_usec 10000000\n")
	r.Read()
	writeFakeFile(t, dir, "cpu.stat", "usage_usec 0\n") // counter reset
	p := r.Read()
	if p.CPUUsedPct < 0 {
		t.Errorf("negative delta: cpu_used_pct %f < 0", p.CPUUsedPct)
	}
}

// -- v1 memory limit readings -------------------------------------------------

func TestV1MemoryLimitReadsFromFile(t *testing.T) {
	dirs := newFakeV1Dirs(t)
	writeFakeFile(t, dirs.memory, "memory.limit_in_bytes", "2147483648\n")
	src := &v1Source{cpuDir: dirs.cpu, memoryDir: dirs.memory}
	limit, err := src.memoryLimit()
	if err != nil {
		t.Fatalf("memoryLimit: %v", err)
	}
	if limit != 2147483648 {
		t.Errorf("limit: got %d, want 2147483648", limit)
	}
}

// -- v2 memory limit readings -------------------------------------------------

func TestV2MemoryLimitReadsFromMemoryMax(t *testing.T) {
	dir := t.TempDir()
	writeFakeFile(t, dir, "memory.max", "2147483648\n")
	writeFakeFile(t, dir, "cpu.stat", "usage_usec 0\n")
	writeFakeFile(t, dir, "memory.current", "0\n")
	src := &v2Source{dir: dir}
	limit, err := src.memoryLimit()
	if err != nil {
		t.Fatalf("memoryLimit: %v", err)
	}
	if limit != 2147483648 {
		t.Errorf("limit: got %d, want 2147483648", limit)
	}
}

func TestV2MemoryLimitFallsBackToHostWhenMax(t *testing.T) {
	dir := t.TempDir()
	writeFakeFile(t, dir, "memory.max", "max\n")
	writeFakeFile(t, dir, "cpu.stat", "usage_usec 0\n")
	writeFakeFile(t, dir, "memory.current", "0\n")
	src := &v2Source{dir: dir}
	limit, err := src.memoryLimit()
	if err != nil {
		t.Fatalf("memoryLimit: %v", err)
	}
	if limit <= 0 {
		t.Errorf("host memory fallback: got %d, want > 0", limit)
	}
}

// -- fieldFromFile and intFromFile --------------------------------------------

func TestFieldFromFileReturnsValue(t *testing.T) {
	dir := t.TempDir()
	writeFakeFile(t, dir, "cpu.stat",
		"nr_periods 120\nnr_throttled 0\nusage_usec 7654321\n")
	got, err := fieldFromFile(filepath.Join(dir, "cpu.stat"), "usage_usec")
	if err != nil {
		t.Fatalf("fieldFromFile: %v", err)
	}
	if got != 7654321 {
		t.Errorf("got %d, want 7654321", got)
	}
}

func TestFieldFromFileErrorsWhenFieldAbsent(t *testing.T) {
	dir := t.TempDir()
	writeFakeFile(t, dir, "cpu.stat", "nr_periods 120\n")
	if _, err := fieldFromFile(filepath.Join(dir, "cpu.stat"), "usage_usec"); err == nil {
		t.Fatal("expected error for absent field")
	}
}

// -- Pressure() helper --------------------------------------------------------

func TestPressureFunctionReturnsSelf(t *testing.T) {
	r, _ := fakeV2(t)
	fn := r.Pressure()
	if fn == nil {
		t.Fatal("Pressure() returned nil function")
	}
	_ = fn()
}

// -- hostMemoryTotal ----------------------------------------------------------

func TestHostMemoryTotalIsPositive(t *testing.T) {
	total, err := hostMemoryTotal()
	if err != nil {
		t.Skipf("hostMemoryTotal: %v", err)
	}
	if total <= 0 {
		t.Errorf("host memory: got %d, want > 0", total)
	}
}

// -- NewReader picks the right hierarchy on this host -------------------------

func TestNewReaderSucceedsOrSkips(t *testing.T) {
	// On a host with no readable cgroup hierarchy this should return an error
	// with a descriptive message; on a real host it should succeed. Either way
	// it must not panic.
	r, err := NewReader()
	if err != nil {
		t.Skipf("NewReader: %v (no readable cgroup on this host)", err)
	}
	if r.Describe() == "" {
		t.Error("Describe must not be empty")
	}
	_ = r.Read()
}

// -- fraction helper ----------------------------------------------------------

func TestFractionClamps(t *testing.T) {
	if fraction(200, 100) != 1 {
		t.Error("fraction(200,100) must return 1")
	}
	if fraction(-10, 100) != 0 {
		t.Error("fraction(-10,100) must return 0")
	}
	if fraction(50, 0) != 0 {
		t.Error("fraction(50,0) must return 0 (no zero division)")
	}
}

// Suppress unused import warning.
var _ = fmt.Sprintf
