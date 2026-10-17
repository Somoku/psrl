//go:build integration

package cgroup

import (
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

// testParent is a subtree of its own, never the deployment's: these tests create and
// remove it, and removing the subtree a running node reads would take that node's
// admission down with it.
const testParent = "psrl-test-sandboxes"

// These tests run on a deployment, because what they check depends on the topology:
// which hierarchies are mounted, whether the process's own path is visible under them,
// and whether Docker's driver places a container where the subtree says. A unit test can
// only assert the code's decisions, not the kernel's.

func TestSetupCreatesTheTreeAndMovesThisProcess(t *testing.T) {
	domain, err := Resolve()
	if err != nil {
		t.Skipf("no readable cgroup hierarchy on this host: %v", err)
	}
	t.Logf("domain: %s", domain)
	budget, err := domain.Setup(testParent, HostLeaf, Request{
		MemoryMB: 1024, Overcommit: 1, CPUMillis: 1000,
	})
	if err != nil {
		t.Fatalf("Setup: %v", err)
	}
	t.Cleanup(func() {
		for _, hierarchy := range domain.Hierarchies {
			_ = os.Remove(filepath.Join(hierarchy.Root, testParent))
			_ = os.Remove(filepath.Join(hierarchy.Root, HostLeaf))
		}
	})
	t.Logf("subtree limit %dMiB, ancestor %dMiB, unreclaimable %dMiB",
		budget.SubtreeLimitBytes/MiB, budget.AncestorLimitBytes/MiB, budget.UnreclaimableBytes/MiB)

	memory, ok := domain.memory()
	if !ok {
		t.Fatal("the domain has no memory hierarchy")
	}
	subtree := filepath.Join(memory.Root, testParent)
	if _, err := os.Stat(subtree); err != nil {
		t.Fatalf("the subtree was not created: %v", err)
	}
	limit, err := memoryLimit(subtree)
	if err != nil {
		t.Fatalf("read the subtree limit: %v", err)
	}
	if limit != budget.SubtreeLimitBytes {
		t.Errorf("subtree limit: got %d, want %d", limit, budget.SubtreeLimitBytes)
	}
	if got, err := os.ReadFile(filepath.Join(memory.Root, "memory.use_hierarchy")); err == nil {
		if strings.TrimSpace(string(got)) != "1" {
			t.Errorf("hierarchical accounting: got %q, want 1", strings.TrimSpace(string(got)))
		}
	}
	// The move is what keeps the parent free of processes, which cgroup v2 requires and
	// v1 tolerates.
	if !holdsProcess(filepath.Join(memory.Root, HostLeaf), strconv.Itoa(os.Getpid())) {
		t.Error("this process did not land in the host leaf")
	}
}

func TestSetupTwiceIsIdempotent(t *testing.T) {
	domain, err := Resolve()
	if err != nil {
		t.Skipf("no readable cgroup hierarchy on this host: %v", err)
	}
	request := Request{MemoryMB: 1024, Overcommit: 1, CPUMillis: 1000}
	first, err := domain.Setup(testParent, HostLeaf, request)
	if err != nil {
		t.Fatalf("first Setup: %v", err)
	}
	t.Cleanup(func() {
		for _, hierarchy := range domain.Hierarchies {
			_ = os.Remove(filepath.Join(hierarchy.Root, testParent))
			_ = os.Remove(filepath.Join(hierarchy.Root, HostLeaf))
		}
	})

	second, err := domain.Setup(testParent, HostLeaf, request)
	if err != nil {
		t.Fatalf("second Setup: %v", err)
	}
	if first.SubtreeLimitBytes != second.SubtreeLimitBytes {
		t.Errorf("limit: %d then %d", first.SubtreeLimitBytes, second.SubtreeLimitBytes)
	}
}

func TestAnExistingSubtreeWithAnotherLimitIsRefused(t *testing.T) {
	domain, err := Resolve()
	if err != nil {
		t.Skipf("no readable cgroup hierarchy on this host: %v", err)
	}
	if _, err := domain.Setup(testParent, HostLeaf, Request{MemoryMB: 1024, Overcommit: 1, CPUMillis: 1000}); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	t.Cleanup(func() {
		for _, hierarchy := range domain.Hierarchies {
			_ = os.Remove(filepath.Join(hierarchy.Root, testParent))
			_ = os.Remove(filepath.Join(hierarchy.Root, HostLeaf))
		}
	})

	// A running node must be restarted to change the limit, so a preflight that would
	// silently accept a different one would report a service that is not what runs.
	if _, err := domain.SetupReadOnly(testParent, Request{MemoryMB: 512, Overcommit: 1, CPUMillis: 1000}); err == nil {
		t.Fatal("a subtree whose limit differs from the resolved one must be refused")
	}
}

func TestTheDomainIsExclusiveOfContainersOutsideIt(t *testing.T) {
	// The whole point of the subtree: a container created outside it must not move the
	// reading, or the ceiling is measuring work it does not manage.
	domain, err := Resolve()
	if err != nil {
		t.Skipf("no readable cgroup hierarchy on this host: %v", err)
	}
	memory, ok := domain.memory()
	if !ok {
		t.Fatal("the domain has no memory hierarchy")
	}
	// A subtree large enough that 2GiB of test containers fit inside it.
	if _, err := domain.Setup(testParent, HostLeaf, Request{MemoryMB: 4096, Overcommit: 1, CPUMillis: 1000}); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	t.Cleanup(func() {
		for _, hierarchy := range domain.Hierarchies {
			_ = os.Remove(filepath.Join(hierarchy.Root, testParent))
			_ = os.Remove(filepath.Join(hierarchy.Root, HostLeaf))
		}
	})
	subtree := filepath.Join(memory.Root, testParent)
	before := subtreeBytes(t, subtree)

	runChargingContainer(t, "" /* the daemon's default parent */)
	time.Sleep(3 * time.Second)
	outside := subtreeBytes(t, subtree)

	if delta := outside - before; delta > 512*(1<<20) {
		t.Errorf("a container outside the subtree moved it by %dMiB, so the domain is not exclusive",
			delta/(1<<20))
	}

	runChargingContainer(t, testParent)
	time.Sleep(3 * time.Second)
	inside := subtreeBytes(t, subtree)

	if delta := inside - outside; delta < 512*(1<<20) {
		t.Errorf("a container inside the subtree charged at least 1GiB and the domain moved by %dMiB",
			delta/(1<<20))
	} else {
		t.Logf("a container inside the subtree moved it by %dMiB", delta/(1<<20))
	}
}

// runChargingContainer starts a container that holds 1GiB of tmpfs and removes it when
// the test ends.
func runChargingContainer(t *testing.T, parent string) {
	t.Helper()
	args := []string{"run", "-d", "--rm", "--shm-size", "1g"}
	if parent != "" {
		args = append(args, "--cgroup-parent", parent)
	}
	args = append(args, "alpine", "sh", "-c",
		"dd if=/dev/zero of=/dev/shm/charge bs=1M count=1024 2>/dev/null; sleep 60")
	out, err := exec.Command("docker", args...).Output()
	if err != nil {
		t.Fatalf("start the charging container: %v", err)
	}
	id := strings.TrimSpace(string(out))
	t.Cleanup(func() { _ = exec.Command("docker", "rm", "-f", id).Run() })
}

// subtreeBytes reads the subtree's memory usage.
func subtreeBytes(t *testing.T, dir string) int64 {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, "memory.usage_in_bytes"))
	if err != nil {
		t.Fatalf("read the subtree usage: %v", err)
	}
	value, err := strconv.ParseInt(strings.TrimSpace(string(raw)), 10, 64)
	if err != nil {
		t.Fatalf("parse the subtree usage: %v", err)
	}
	return value
}

// TestTheResolvedHierarchiesAreTheMountedOnes pins what Resolve reports against the
// kernel's own list, so a parser that silently drops a controller is visible.
func TestTheResolvedHierarchiesAreTheMountedOnes(t *testing.T) {
	domain, err := Resolve()
	if err != nil {
		t.Skipf("no readable cgroup hierarchy on this host: %v", err)
	}
	var resolved []string
	for _, hierarchy := range domain.Hierarchies {
		resolved = append(resolved, hierarchy.Controllers...)
	}
	for _, controller := range []string{"memory", "cpu", "cpuset"} {
		if !slices.Contains(resolved, controller) {
			t.Errorf("the %s controller is mounted but was not resolved (resolved: %v)", controller, resolved)
		}
	}
}
