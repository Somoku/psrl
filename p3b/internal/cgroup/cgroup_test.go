package cgroup

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// -- fixtures ----------------------------------------------------------------

// fakeV1 builds a v1 layout where each controller is its own mount, rooted at the
// directory returned. Every mount point is the process's own cgroup, which is the
// namespaced case this deployment has.
type fakeV1 struct {
	memory string
	cpu    string
	cpuset string
}

func newFakeV1(t *testing.T) (Domain, fakeV1) {
	t.Helper()
	root := t.TempDir()
	dirs := fakeV1{
		memory: filepath.Join(root, "memory"),
		cpu:    filepath.Join(root, "cpu,cpuacct"),
		cpuset: filepath.Join(root, "cpuset"),
	}
	for _, dir := range []string{dirs.memory, dirs.cpu, dirs.cpuset} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatalf("mkdir %s: %v", dir, err)
		}
		writeFile(t, dir, "cgroup.procs", "")
	}
	writeFile(t, dirs.memory, "memory.limit_in_bytes", "2362232012800\n")
	writeFile(t, dirs.memory, "memory.usage_in_bytes", "268435456\n")
	writeFile(t, dirs.memory, "memory.memsw.limit_in_bytes", "2362232012800\n")
	writeFile(t, dirs.memory, "memory.use_hierarchy", "1\n")
	writeFile(t, dirs.memory, "memory.kmem.usage_in_bytes", "0\n")
	writeFile(t, dirs.memory, "memory.stat", "cache 0\ntotal_rss 104857600\ntotal_cache 0\n")
	writeFile(t, dirs.cpu, "cpu.cfs_period_us", "100000\n")
	writeFile(t, dirs.cpu, "cpu.cfs_quota_us", "100000\n")
	writeFile(t, dirs.cpuset, "cpuset.cpus", "0-7\n")
	writeFile(t, dirs.cpuset, "cpuset.mems", "0\n")
	writeFile(t, dirs.cpuset, "cpuset.effective_cpus", "0-7\n")
	writeFile(t, dirs.cpuset, "cpuset.effective_mems", "0\n")

	domain := Domain{Version: 1, Hierarchies: []Hierarchy{
		{Controllers: []string{"memory"}, Mount: dirs.memory, Root: dirs.memory},
		{Controllers: []string{"cpu", "cpuacct"}, Mount: dirs.cpu, Root: dirs.cpu},
		{Controllers: []string{"cpuset"}, Mount: dirs.cpuset, Root: dirs.cpuset},
	}}
	return domain, dirs
}

func writeFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}

func readFile(t *testing.T, dir, name string) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		t.Fatalf("read %s: %v", name, err)
	}
	return strings.TrimSpace(string(raw))
}

func request() Request {
	return Request{MemoryMB: 1000, Overcommit: 1.5, CPUMillis: 180000, Safety: 0.05}
}

// -- resolution --------------------------------------------------------------

func TestResolveReadsAV1HierarchyPerMount(t *testing.T) {
	dir := t.TempDir()
	mounts := filepath.Join(dir, "mounts")
	cgroup := filepath.Join(dir, "cgroup")
	memory := filepath.Join(dir, "memory")
	cpu := filepath.Join(dir, "cpu,cpuacct")
	for _, point := range []string{memory, cpu} {
		if err := os.MkdirAll(point, 0o755); err != nil {
			t.Fatal(err)
		}
		writeFile(t, point, "cgroup.procs", "")
	}
	writeFile(t, dir, "mounts", "cgroup "+memory+" cgroup rw,memory 0 0\n"+
		"cgroup "+cpu+" cgroup rw,cpu,cpuacct 0 0\n")
	writeFile(t, dir, "cgroup", "7:memory:/kubepods/pod/ctr\n2:cpu,cpuacct:/kubepods/pod/ctr\n")

	domain, err := resolveFrom(mounts, cgroup)
	if err != nil {
		t.Fatalf("resolveFrom: %v", err)
	}

	if domain.Version != 1 {
		t.Fatalf("version: got %d, want 1", domain.Version)
	}
	if len(domain.Hierarchies) != 2 {
		t.Fatalf("hierarchies: got %d, want 2", len(domain.Hierarchies))
	}
	// The path in /proc/self/cgroup does not exist under the mount, so the mount
	// point is the process's own cgroup.
	for _, hierarchy := range domain.Hierarchies {
		if hierarchy.Root != hierarchy.Mount {
			t.Errorf("%v: root %s should fall back to the mount %s", hierarchy.Controllers, hierarchy.Root, hierarchy.Mount)
		}
	}
}

func TestANamespacedHierarchyFallsBackToTheMountRoot(t *testing.T) {
	dir := t.TempDir()
	point := filepath.Join(dir, "memory")
	if err := os.MkdirAll(point, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, point, "cgroup.procs", "")
	writeFile(t, dir, "mounts", "cgroup "+point+" cgroup rw,memory 0 0\n")
	writeFile(t, dir, "cgroup", "7:memory:/kubepods/pod/ctr\n")

	domain, err := resolveFrom(filepath.Join(dir, "mounts"), filepath.Join(dir, "cgroup"))
	if err != nil {
		t.Fatalf("resolveFrom: %v", err)
	}

	if domain.Hierarchies[0].Root != point {
		t.Errorf("root: got %s, want the mount %s", domain.Hierarchies[0].Root, point)
	}
}

func TestAVisiblePathIsPreferredOverTheMountRoot(t *testing.T) {
	dir := t.TempDir()
	point := filepath.Join(dir, "memory")
	own := filepath.Join(point, "kubepods/pod/ctr")
	if err := os.MkdirAll(own, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, point, "cgroup.procs", "")
	writeFile(t, own, "cgroup.procs", "")
	writeFile(t, dir, "mounts", "cgroup "+point+" cgroup rw,memory 0 0\n")
	writeFile(t, dir, "cgroup", "7:memory:/kubepods/pod/ctr\n")

	domain, err := resolveFrom(filepath.Join(dir, "mounts"), filepath.Join(dir, "cgroup"))
	if err != nil {
		t.Fatalf("resolveFrom: %v", err)
	}

	if domain.Hierarchies[0].Root != own {
		t.Errorf("root: got %s, want %s", domain.Hierarchies[0].Root, own)
	}
}

func TestResolvePrefersTheUnifiedHierarchy(t *testing.T) {
	dir := t.TempDir()
	point := filepath.Join(dir, "unified")
	if err := os.MkdirAll(point, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, point, "cgroup.procs", "")
	writeFile(t, point, "cgroup.controllers", "cpu memory pids\n")
	writeFile(t, dir, "mounts",
		"cgroup "+filepath.Join(dir, "memory")+" cgroup rw,memory 0 0\n"+
			"cgroup2 "+point+" cgroup2 rw 0 0\n")
	writeFile(t, dir, "cgroup", "0::/\n7:memory:/somewhere\n")

	domain, err := resolveFrom(filepath.Join(dir, "mounts"), filepath.Join(dir, "cgroup"))
	if err != nil {
		t.Fatalf("resolveFrom: %v", err)
	}

	if domain.Version != 2 {
		t.Fatalf("version: got %d, want 2", domain.Version)
	}
	if len(domain.Hierarchies) != 1 {
		t.Fatalf("a unified host has one hierarchy, got %d", len(domain.Hierarchies))
	}
	if got := strings.Join(domain.Hierarchies[0].Controllers, ","); got != "cpu,memory,pids" {
		t.Errorf("controllers: got %q", got)
	}
}

func TestAHostWithNoCgroupMountIsAnError(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "mounts", "proc /proc proc rw 0 0\n")
	writeFile(t, dir, "cgroup", "0::/\n")

	if _, err := resolveFrom(filepath.Join(dir, "mounts"), filepath.Join(dir, "cgroup")); err == nil {
		t.Fatal("a host with no cgroup mount must not resolve to an empty domain")
	}
}

// -- budget ------------------------------------------------------------------

func TestTheSubtreeLimitIsTheEnvelopeTimesOvercommit(t *testing.T) {
	domain, _ := newFakeV1(t)

	budget, err := domain.Budget(request())
	if err != nil {
		t.Fatalf("Budget: %v", err)
	}

	want := int64(float64(1000*MiB) * 1.5)
	if budget.SubtreeLimitBytes != want {
		t.Errorf("limit: got %d, want %d", budget.SubtreeLimitBytes, want)
	}
	if budget.DeclaredMB != 1000 {
		t.Errorf("declared: got %dMB, want 1000MB", budget.DeclaredMB)
	}
	if budget.Capped() {
		t.Error("a budget the ancestor can hold must not be reported as capped")
	}
}

func TestTheAncestorCapsTheSubtreeAtWhatItCanHold(t *testing.T) {
	domain, dirs := newFakeV1(t)
	// 1GiB of ancestor memory, 100MiB unreclaimable, 5% safety: 923MiB of capacity.
	writeFile(t, dirs.memory, "memory.limit_in_bytes", "1073741824\n")
	writeFile(t, dirs.memory, "memory.stat", "total_rss 104857600\ntotal_cache 0\n")
	writeFile(t, dirs.memory, "memory.kmem.usage_in_bytes", "0\n")

	budget, err := domain.Budget(Request{MemoryMB: 700, Overcommit: 1.5, CPUMillis: 1000, Safety: 0.05})
	if err != nil {
		t.Fatalf("Budget: %v", err)
	}

	limit := int64(1073741824)
	capacity := limit - 104857600 - int64(float64(limit)*0.05)
	if budget.SubtreeLimitBytes != capacity {
		t.Errorf("limit: got %d, want the ancestor's capacity %d", budget.SubtreeLimitBytes, capacity)
	}
	if !budget.Capped() {
		t.Error("a budget reduced by its ancestor must say so")
	}
}

func TestAnEnvelopeTheAncestorCannotHoldIsRefused(t *testing.T) {
	domain, dirs := newFakeV1(t)
	writeFile(t, dirs.memory, "memory.limit_in_bytes", "1073741824\n")
	writeFile(t, dirs.memory, "memory.unreclaimable", "1")
	writeFile(t, dirs.memory, "memory.stat", "total_rss 104857600\n")
	writeFile(t, dirs.memory, "memory.kmem.usage_in_bytes", "0\n")

	_, err := domain.Budget(Request{MemoryMB: 1024, Overcommit: 1.5, CPUMillis: 1000, Safety: 0.05})
	if err == nil {
		t.Fatal("an envelope larger than the ancestor's capacity must be refused")
	}
	if !strings.Contains(err.Error(), "declares 1024MiB") {
		t.Errorf("error should name what was declared, got %v", err)
	}
}

func TestZeroOrNegativeOvercommitIsRefused(t *testing.T) {
	domain, _ := newFakeV1(t)

	if _, err := domain.Budget(Request{MemoryMB: 100, Overcommit: 0}); err == nil {
		t.Fatal("overcommit below one must be refused")
	}
	if _, err := domain.Budget(Request{MemoryMB: 0, Overcommit: 1}); err == nil {
		t.Fatal("a non-positive envelope must be refused")
	}
}

// -- setup -------------------------------------------------------------------

func TestSetupCreatesTheSubtreeAndWritesItsLimits(t *testing.T) {
	domain, dirs := newFakeV1(t)

	if _, err := domain.Setup(SandboxParent, HostLeaf, request()); err != nil {
		t.Fatalf("Setup: %v", err)
	}

	subtree := filepath.Join(dirs.memory, SandboxParent)
	limit := readFile(t, subtree, "memory.limit_in_bytes")
	if limit != "1572864000" {
		t.Errorf("memory limit: got %s, want 1572864000 (1000MiB x 1.5)", limit)
	}
	if got := readFile(t, filepath.Join(dirs.cpu, SandboxParent), "cpu.cfs_quota_us"); got != "18000000" {
		t.Errorf("cpu quota: got %s, want 18000000 (180 cores)", got)
	}
	if got := readFile(t, filepath.Join(dirs.cpuset, SandboxParent), "cpuset.cpus"); got != "0-7" {
		t.Errorf("cpuset: got %q, want the parent's set", got)
	}
}

// TestTheMemswLimitIsWrittenBeforeTheMemoryLimit proves the order the kernel
// requires: memory.limit_in_bytes cannot be raised above memory.memsw.limit_in_bytes,
// so a subtree written memory-first fails on a swap-accounted v1 host. Both files are
// made unwritable, so the error names whichever was attempted first.
func TestTheMemswLimitIsWrittenBeforeTheMemoryLimit(t *testing.T) {
	domain, dirs := newFakeV1(t)
	subtree := filepath.Join(dirs.memory, SandboxParent)
	for _, name := range []string{"memory.memsw.limit_in_bytes", "memory.limit_in_bytes"} {
		if err := os.MkdirAll(filepath.Join(subtree, name), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	_, err := domain.Setup(SandboxParent, HostLeaf, request())
	if err == nil {
		t.Fatal("a hierarchy that refuses both writes must fail setup")
	}
	if !strings.Contains(err.Error(), "memory.memsw.limit_in_bytes") {
		t.Errorf("memsw must be written first, got %v", err)
	}
}

func TestSetupTurnsOnHierarchicalAccounting(t *testing.T) {
	domain, dirs := newFakeV1(t)
	writeFile(t, dirs.memory, "memory.use_hierarchy", "0\n")

	if _, err := domain.Setup(SandboxParent, HostLeaf, request()); err != nil {
		t.Fatalf("Setup: %v", err)
	}

	if got := readFile(t, dirs.memory, "memory.use_hierarchy"); got != "1" {
		t.Errorf("use_hierarchy: got %q, want 1", got)
	}
}

func TestSetupMovesThisProcessIntoTheLeaf(t *testing.T) {
	domain, dirs := newFakeV1(t)

	if _, err := domain.Setup(SandboxParent, HostLeaf, request()); err != nil {
		t.Fatalf("Setup: %v", err)
	}

	for _, dir := range []string{dirs.memory, dirs.cpu, dirs.cpuset} {
		procs := readFile(t, filepath.Join(dir, HostLeaf), "cgroup.procs")
		if procs == "" {
			t.Errorf("%s: the service did not move into its leaf", dir)
		}
	}
}

func TestSetupIsIdempotent(t *testing.T) {
	domain, dirs := newFakeV1(t)

	first, err := domain.Setup(SandboxParent, HostLeaf, request())
	if err != nil {
		t.Fatalf("first Setup: %v", err)
	}
	second, err := domain.Setup(SandboxParent, HostLeaf, request())
	if err != nil {
		t.Fatalf("second Setup: %v", err)
	}

	if first.SubtreeLimitBytes != second.SubtreeLimitBytes {
		t.Errorf("limit changed between runs: %d then %d", first.SubtreeLimitBytes, second.SubtreeLimitBytes)
	}
	if got := readFile(t, filepath.Join(dirs.memory, SandboxParent), "memory.limit_in_bytes"); got != "1572864000" {
		t.Errorf("limit after the second run: got %s", got)
	}
}

func TestAFailedMoveIsAnError(t *testing.T) {
	domain, dirs := newFakeV1(t)
	// A directory where cgroup.procs must be makes the write fail regardless of the
	// privileges the test runs with.
	if err := os.MkdirAll(filepath.Join(dirs.memory, HostLeaf, "cgroup.procs"), 0o755); err != nil {
		t.Fatal(err)
	}

	if _, err := domain.Setup(SandboxParent, HostLeaf, request()); err == nil {
		t.Fatal("a hierarchy that refuses the move must fail setup")
	}
}

func TestSetupReadOnlyCreatesNothing(t *testing.T) {
	domain, dirs := newFakeV1(t)

	if _, err := domain.SetupReadOnly(SandboxParent, request()); err != nil {
		t.Fatalf("SetupReadOnly: %v", err)
	}

	if _, err := os.Stat(filepath.Join(dirs.memory, SandboxParent)); !os.IsNotExist(err) {
		t.Error("preflight must not create the subtree")
	}
	if got := readFile(t, dirs.memory, "cgroup.procs"); got != "" {
		t.Error("preflight must not move this process")
	}
}

func TestPreflightRefusesAnExistingSubtreeWithAnotherLimit(t *testing.T) {
	domain, dirs := newFakeV1(t)
	subtree := filepath.Join(dirs.memory, SandboxParent)
	if err := os.MkdirAll(subtree, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, subtree, "memory.limit_in_bytes", "1073741824\n")

	if _, err := domain.SetupReadOnly(SandboxParent, request()); err == nil {
		t.Fatal("a subtree whose limit differs from the resolved one must be refused")
	}
}
