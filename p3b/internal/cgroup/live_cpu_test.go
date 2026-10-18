//go:build integration

package cgroup

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// TestPerSandboxCPUMeasuresARateAndAQuotaCanBeMoved is the CPU half of what the elastic
// controller depends on, measured against the kernel.
//
// CPU is the compressible dimension, so unlike a memory shrink a lowered quota cannot
// kill the sandbox -- it can only make it slower. That is the whole reason the controller
// may move it on a running sandbox, and this is the evidence that both halves work: the
// cumulative counter advances while the sandbox burns a core, and a quota written here is
// one the kernel reports back.
func TestPerSandboxCPUMeasuresARateAndAQuotaCanBeMoved(t *testing.T) {
	domain, err := Resolve()
	if err != nil {
		t.Skipf("no readable cgroup hierarchy on this host: %v", err)
	}
	if _, ok := domain.accounting(); !ok {
		t.Skipf("this host has no CPU accounting hierarchy: %s", domain.String())
	}
	parent := testParent
	if _, err := domain.Setup(parent, HostLeaf, Request{MemoryMB: 8192, Overcommit: 1, CPUMillis: 4000}); err != nil {
		t.Fatalf("create the sandbox subtree: %v", err)
	}
	t.Cleanup(func() {
		for _, hierarchy := range domain.Hierarchies {
			_ = os.RemoveAll(filepath.Join(hierarchy.Root, parent))
			_ = os.Remove(filepath.Join(hierarchy.Root, HostLeaf))
		}
	})

	// One core for ten seconds, which is the shape a tool call has: a burst, then idle.
	container := runContainer(t, "--cgroup-parent", parent, "--name", "psrl-cpu-probe",
		"--cpus", "2", "alpine", "sh", "-c",
		"timeout 10 sh -c 'while :; do :; done'; sleep 300")
	t.Cleanup(func() { _ = exec.Command("docker", "rm", "-f", container).Run() })

	// Let the burn start, then take two readings far enough apart to divide.
	time.Sleep(3 * time.Second)
	first := childOf(t, domain, parent, container)
	started := time.Now()
	time.Sleep(4 * time.Second)
	second := childOf(t, domain, parent, container)

	elapsed := time.Since(started)
	if second.CPUUsageNanos <= first.CPUUsageNanos {
		t.Fatalf("the sandbox's CPU counter did not advance: %d then %d",
			first.CPUUsageNanos, second.CPUUsageNanos)
	}
	measured := float64(second.CPUUsageNanos-first.CPUUsageNanos) / float64(elapsed.Nanoseconds())
	t.Logf("burning: %.2f cores over %.1fs (quota %d millis)", measured, elapsed.Seconds(),
		QuotaToMillis(second.CPUQuotaMicros))
	if measured < 0.5 {
		t.Errorf("a sandbox burning one core measured %.2f cores, so the counter is not counting it", measured)
	}

	// Now the idle part: the same measurement over a quiet container must come down, which
	// is what lets a class's price fall between steps.
	time.Sleep(8 * time.Second)
	third := childOf(t, domain, parent, container)
	quietStart := time.Now()
	time.Sleep(4 * time.Second)
	fourth := childOf(t, domain, parent, container)
	idle := float64(fourth.CPUUsageNanos-third.CPUUsageNanos) / float64(time.Since(quietStart).Nanoseconds())
	t.Logf("idle: %.3f cores", idle)
	if idle > measured/2 {
		t.Errorf("an idle sandbox measured %.3f cores against %.2f while busy, so the rate is not tracking load",
			idle, measured)
	}

	// The quota is the knob the controller moves under pressure. Lowering it on a running
	// sandbox must be accepted by the kernel and reported back, and raising it again must
	// restore what the sandbox declared.
	if err := domain.SetChildCPUQuota(parent, container, 500); err != nil {
		t.Fatalf("lower the CPU quota: %v", err)
	}
	throttled := childOf(t, domain, parent, container)
	if got := QuotaToMillis(throttled.CPUQuotaMicros); got != 500 {
		t.Errorf("quota after throttling: got %d millis, want 500", got)
	}
	if out, err := exec.Command("docker", "exec", container, "sh", "-c", "echo alive").CombinedOutput(); err != nil {
		t.Fatalf("the sandbox did not survive the throttling: %v (%s)", err, out)
	}
	if err := domain.SetChildCPUQuota(parent, container, 2000); err != nil {
		t.Fatalf("restore the CPU quota: %v", err)
	}
	if got := QuotaToMillis(childOf(t, domain, parent, container).CPUQuotaMicros); got != 2000 {
		t.Errorf("quota after restore: got %d millis, want 2000", got)
	}
}

// TestChildrenWithoutAnAccountingHierarchyStillReportMemory keeps the CPU fields honest:
// a domain that cannot count CPU must not report a child as if it had measured zero load,
// because the caller then charges the request's declared cores, which is the behaviour it
// had before CPU was measured at all.
func TestChildrenWithoutAnAccountingHierarchyStillReportMemory(t *testing.T) {
	domain, err := Resolve()
	if err != nil {
		t.Skipf("no readable cgroup hierarchy on this host: %v", err)
	}
	stripped := Domain{Version: domain.Version}
	for _, hierarchy := range domain.Hierarchies {
		if hierarchy.Has("cpuacct") || (domain.Version == 2 && hierarchy.Has("cpu")) {
			continue
		}
		stripped.Hierarchies = append(stripped.Hierarchies, hierarchy)
	}
	if len(stripped.Hierarchies) == 0 {
		t.Skip("this host has no hierarchy without CPU accounting")
	}
	parent := testParent
	if _, err := stripped.Setup(parent, HostLeaf, Request{MemoryMB: 4096, Overcommit: 1}); err != nil {
		t.Fatalf("create the sandbox subtree: %v", err)
	}
	t.Cleanup(func() {
		for _, hierarchy := range stripped.Hierarchies {
			_ = os.RemoveAll(filepath.Join(hierarchy.Root, parent))
			_ = os.Remove(filepath.Join(hierarchy.Root, HostLeaf))
		}
	})
	container := runContainer(t, "--cgroup-parent", parent, "--name", "psrl-nocpuacct-probe",
		"alpine", "sleep", "300")
	t.Cleanup(func() { _ = exec.Command("docker", "rm", "-f", container).Run() })

	child := childOf(t, stripped, parent, container)
	if child.Usage <= 0 {
		t.Errorf("a child reported %d bytes of memory, so the memory read is not working", child.Usage)
	}
	if child.CPUUsageNanos != 0 {
		t.Errorf("a domain with no accounting hierarchy reported %d nanos of CPU", child.CPUUsageNanos)
	}
	if err := stripped.SetChildCPUQuota(parent, container, 500); err == nil {
		t.Error("a domain with no accounting hierarchy accepted a CPU quota write")
	} else if !strings.Contains(err.Error(), "no CPU accounting") {
		t.Errorf("unexpected error: %v", err)
	}
}

// TestAThrottledSandboxReportsIt is the signal the elastic controller depends on,
// measured against the kernel.
//
// Usage alone cannot say a sandbox wanted more CPU than it was given: cpuacct saturates
// at the quota, so a throttled sandbox and a busy-but-satisfied one read the same. The
// throttling counters are the only place the kernel says the sandbox asked and was
// refused. This burns one core inside a half-core quota and requires the counter to move.
func TestAThrottledSandboxReportsIt(t *testing.T) {
	domain, err := Resolve()
	if err != nil {
		t.Skipf("no readable cgroup hierarchy on this host: %v", err)
	}
	if _, ok := domain.accounting(); !ok {
		t.Skipf("this host has no CPU accounting hierarchy: %s", domain.String())
	}
	parent := testParent
	if _, err := domain.Setup(parent, HostLeaf, Request{MemoryMB: 8192, Overcommit: 1, CPUMillis: 4000}); err != nil {
		t.Fatalf("create the sandbox subtree: %v", err)
	}
	t.Cleanup(func() {
		for _, hierarchy := range domain.Hierarchies {
			_ = os.RemoveAll(filepath.Join(hierarchy.Root, parent))
			_ = os.Remove(filepath.Join(hierarchy.Root, HostLeaf))
		}
	})

	// Two threads burning inside a half-core quota: the demand is well above what the
	// quota allows, so the kernel throttles every period.
	container := runContainer(t, "--cgroup-parent", parent, "--name", "psrl-throttle-probe",
		"--cpus", "0.5", "alpine", "sh", "-c",
		"timeout 8 sh -c 'while :; do :; done & while :; do :; done'; sleep 300")
	t.Cleanup(func() { _ = exec.Command("docker", "rm", "-f", container).Run() })

	time.Sleep(4 * time.Second)
	child := childOf(t, domain, parent, container)
	t.Logf("throttled: %dns, quota: %d millis, usage: %dns",
		child.CPUThrottledNanos, QuotaToMillis(child.CPUQuotaMicros), child.CPUUsageNanos)
	if child.CPUThrottledNanos <= 0 {
		t.Error("a sandbox burning two threads in a half-core quota reported no throttling; " +
			"the controller would price this sandbox from usage that is censored at the quota")
	}
	if got := QuotaToMillis(child.CPUQuotaMicros); got != 500 {
		t.Errorf("quota: got %d millis, want 500 from --cpus 0.5", got)
	}
}
