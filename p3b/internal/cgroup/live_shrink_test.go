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

// TestLoweringALimitReclaimsCacheAndKeepsTheProcess is the mechanism the elastic
// controller depends on, measured against the kernel.
//
// cgroup v1 has no memory.high: memory.soft_limit_in_bytes is a hint consulted only when
// the whole host is under pressure, so the only way a v1 node can make a sandbox give
// memory back is to lower its hard limit. That works only if the sandbox's reclaimable
// pages are worth what the controller believes, so this charges 3 GiB of file cache and
// lowers the limit to what the container is resident plus a margin -- the floor the
// controller itself uses. The cache must go and the process must survive.
func TestLoweringALimitReclaimsCacheAndKeepsTheProcess(t *testing.T) {
	domain, err := Resolve()
	if err != nil {
		t.Skipf("no readable cgroup hierarchy on this host: %v", err)
	}
	parent := testParent
	if _, err := domain.Setup(parent, HostLeaf, Request{MemoryMB: 8192, Overcommit: 1, CPUMillis: 1000}); err != nil {
		t.Fatalf("create the sandbox subtree: %v", err)
	}
	t.Cleanup(func() {
		for _, hierarchy := range domain.Hierarchies {
			_ = os.RemoveAll(filepath.Join(hierarchy.Root, parent))
			_ = os.Remove(filepath.Join(hierarchy.Root, HostLeaf))
		}
	})

	// 3 GiB of file cache, then a process that stays alive and keeps its pages mapped.
	script := "dd if=/dev/zero of=/tmp/charge bs=1M count=3072 status=none; sleep 300"
	container := runContainer(t, "--cgroup-parent", parent, "--name", "psrl-elastic-probe",
		"--memory", "4g", "alpine", "sh", "-c", script)
	t.Cleanup(func() { _ = exec.Command("docker", "rm", "-f", container).Run() })

	time.Sleep(5 * time.Second)
	before := childOf(t, domain, parent, container)
	t.Logf("before: usage=%dMiB resident=%dMiB cache=%dMiB limit=%dMiB",
		before.Usage/(1<<20), before.Resident/(1<<20), before.Cache/(1<<20), before.Limit/(1<<20))
	if before.Cache < 1024*(1<<20) {
		t.Skipf("the container charged only %dMiB of cache, so this host cannot show the reclaim",
			before.Cache/(1<<20))
	}

	target := before.Resident + 256*(1<<20)
	if err := domain.SetChildLimit(parent, container, target); err != nil {
		t.Fatalf("lower the limit: %v", err)
	}
	// The kernel reclaims on the next allocation; give it a moment under the new limit.
	time.Sleep(5 * time.Second)

	after := childOf(t, domain, parent, container)
	t.Logf("after:  usage=%dMiB resident=%dMiB cache=%dMiB limit=%dMiB",
		after.Usage/(1<<20), after.Resident/(1<<20), after.Cache/(1<<20), after.Limit/(1<<20))
	if after.Usage >= before.Usage {
		t.Errorf("usage did not fall: %dMiB then %dMiB", before.Usage/(1<<20), after.Usage/(1<<20))
	}
	if reclaimed := before.Usage - after.Usage; reclaimed < 1024*(1<<20) {
		t.Errorf("only %dMiB was reclaimed from %dMiB of cache", reclaimed/(1<<20), before.Cache/(1<<20))
	}
	// The resident part is what must survive: a limit below it kills the process rather
	// than reclaiming.
	if after.Resident > target {
		t.Errorf("resident memory %dMiB exceeds the limit's intent %dMiB",
			after.Resident/(1<<20), target/(1<<20))
	}
	if out, err := exec.Command("docker", "exec", container, "sh", "-c", "echo alive").CombinedOutput(); err != nil {
		t.Fatalf("the sandbox did not survive the shrink: %v (%s)", err, out)
	}

	// And it must be reversible, which is what the controller does when the node has room.
	if err := domain.SetChildLimit(parent, container, 4*(1<<30)); err != nil {
		t.Fatalf("restore the limit: %v", err)
	}
	if restored := childOf(t, domain, parent, container); restored.Limit != 4*(1<<30) {
		t.Errorf("limit after restore: got %d, want %d", restored.Limit, int64(4*(1<<30)))
	}
}

// childOf reads one container's cgroup through the domain, which is how the controller
// sees it.
func childOf(t *testing.T, domain Domain, parent, container string) Child {
	t.Helper()
	children, err := domain.Children(parent)
	if err != nil {
		t.Fatalf("list the subtree: %v", err)
	}
	for _, child := range children {
		if child.ID == container {
			return child
		}
	}
	// Docker may name the cgroup after the container id in full; match a prefix either way.
	for _, child := range children {
		if strings.HasPrefix(container, child.ID) || strings.HasPrefix(child.ID, container) {
			return child
		}
	}
	t.Fatalf("%s is not among the subtree's %d children", container, len(children))
	return Child{}
}

// runContainer starts a detached container and returns its id.
func runContainer(t *testing.T, args ...string) string {
	t.Helper()
	out, err := exec.Command("docker", append([]string{"run", "-d", "--rm"}, args...)...).Output()
	if err != nil {
		t.Fatalf("docker run: %v", err)
	}
	return strings.TrimSpace(string(out))
}
