//go:build integration

package pressure

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/cgroup"
)

// allocatedBytes is how much the test makes a container charge. It is big enough
// that the answer cannot be confused with the churn of a shared host, and small
// enough to be unremarkable on one.
const allocatedBytes = 4 << 30

// TestAContainerThisNodeRunsIsInsideTheMeasuredDomain measures the question 5.2 of
// the sandbox optimization design asks, on the deployment rather than from the
// source.
//
// The design's own instruction is to check under the real Docker and cgroup
// configuration whether a container belongs to the subtree the sampler reads,
// because the answer depends on the deployment topology. This is that check: it
// charges a known amount of memory inside a container and requires the reading to
// move by it.
//
// A reader pointed at the service's own process cgroup, at a subtree the
// containers are not in, or at an ancestor that does not aggregate them fails
// here, and the failure reports both numbers so the direction of the error is
// visible. Requires a Docker daemon and a readable cgroup hierarchy, so it runs
// under the integration tag on a node.
func TestAContainerThisNodeRunsIsInsideTheMeasuredDomain(t *testing.T) {
	if _, err := exec.LookPath("docker"); err != nil {
		t.Skip("no docker CLI on this host")
	}
	reader, err := NewReader()
	if err != nil {
		t.Skipf("no readable cgroup hierarchy on this host: %v", err)
	}
	t.Logf("measuring %s", reader.Describe())

	before, limit, ok := reader.MeasuredBytes()
	if !ok {
		t.Skip("the memory usage file is not readable on this host")
	}
	t.Logf("domain usage %d bytes against a %d byte limit", before, limit)

	container := startAllocatingContainer(t)
	// The charge is not synchronous with the create: the container has to run its
	// allocation first.
	time.Sleep(10 * time.Second)

	after, _, ok := reader.MeasuredBytes()
	if !ok {
		t.Fatal("the measured domain stopped being readable during the test")
	}

	// The container's own accounting is the ground truth for what was allocated.
	// Without it, a container that failed to allocate would look exactly like a
	// reader that cannot see it.
	charged := containerMemoryBytes(t, container)
	if charged < allocatedBytes/2 {
		t.Fatalf("the container charged %d bytes of the %d it was asked for, so it never allocated",
			charged, allocatedBytes)
	}

	if delta := after - before; delta < allocatedBytes/2 {
		t.Fatalf("a container this node runs charged %d bytes and the measured domain moved by %d: %s does not contain it",
			charged, delta, reader.Describe())
	} else {
		t.Logf("a container charged %d bytes and the measured domain moved by %d", charged, delta)
	}
}

// startAllocatingContainer charges a known amount of memory and holds it for the
// length of the test.
//
// tmpfs pages are charged to the container that faulted them and are not
// reclaimable while the file exists, which makes the charge both real and stable
// for the duration of a measurement.
func startAllocatingContainer(t *testing.T) string {
	t.Helper()
	mib := allocatedBytes / (1 << 20)
	command := "dd if=/dev/zero of=/dev/shm/charge bs=1M count=" + strconv.Itoa(mib) + " 2>/dev/null; sleep 120"
	out, err := exec.Command("docker", "run", "-d", "--rm",
		"--shm-size", strconv.Itoa(mib)+"m", "alpine", "sh", "-c", command).Output()
	if err != nil {
		t.Fatalf("start the allocating container: %v", err)
	}
	id := strings.TrimSpace(string(out))
	t.Cleanup(func() {
		_ = exec.Command("docker", "rm", "-f", id).Run()
	})
	return id
}

// containerMemoryBytes asks the container how much memory it has been charged.
//
// The container's own cgroup mount is its own cgroup, so this needs no assumption
// about where a host keeps its container cgroups, which is exactly the assumption
// under test.
func containerMemoryBytes(t *testing.T, container string) int64 {
	t.Helper()
	out, err := exec.Command("docker", "exec", container, "sh", "-c",
		"cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || cat /sys/fs/cgroup/memory.current").Output()
	if err != nil {
		t.Fatalf("read the container's own memory cgroup: %v", err)
	}
	charged, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		t.Fatalf("parse the container's own memory cgroup reading %q: %v", strings.TrimSpace(string(out)), err)
	}
	return charged
}

// TestTheDomainReaderMeasuresTheSubtreeNotTheAncestor is the 5.2 claim measured rather
// than argued.
//
// The reader is pointed at the sandbox subtree, and two containers are charged: one
// inside it and one outside. The subtree's reading must follow the first and ignore the
// second, because a ceiling applied to the ancestor is a ceiling on work the node does
// not manage -- on these nodes that ancestor is the pod, and most of what it holds is a
// trainer's page cache.
func TestTheDomainReaderMeasuresTheSubtreeNotTheAncestor(t *testing.T) {
	if _, err := exec.LookPath("docker"); err != nil {
		t.Skip("no docker CLI on this host")
	}
	domain, err := cgroup.Resolve()
	if err != nil {
		t.Skipf("no readable cgroup hierarchy on this host: %v", err)
	}
	parent := cgroup.SandboxParent
	// Twice the test's charge: the container allocates 4GiB of tmpfs, and a subtree
	// limit equal to that would OOM-kill it instead of measuring it.
	if _, err := domain.Setup(parent, cgroup.HostLeaf, cgroup.Request{
		MemoryMB: 8192, Overcommit: 1, CPUMillis: 1000,
	}); err != nil {
		t.Fatalf("create the sandbox subtree: %v", err)
	}
	t.Cleanup(func() {
		for _, hierarchy := range domain.Hierarchies {
			_ = os.Remove(filepath.Join(hierarchy.Root, parent))
			_ = os.Remove(filepath.Join(hierarchy.Root, cgroup.HostLeaf))
		}
	})
	reader, err := NewDomainReader(parent)
	if err != nil {
		t.Fatalf("NewDomainReader(%q): %v", parent, err)
	}
	t.Logf("measuring %s", reader.Describe())

	before, limit, ok := reader.MeasuredBytes()
	if !ok {
		t.Fatal("the subtree's usage is not readable")
	}
	t.Logf("subtree usage %dMiB against a %dMiB limit", before/(1<<20), limit/(1<<20))

	outside := startAllocatingContainerIn(t, "")
	time.Sleep(10 * time.Second)
	afterOutside, _, ok := reader.MeasuredBytes()
	if !ok {
		t.Fatal("the subtree stopped being readable")
	}
	if delta := afterOutside - before; delta > 512*(1<<20) {
		t.Errorf("a container outside the subtree moved it by %dMiB, so the domain is not exclusive", delta/(1<<20))
	}

	inside := startAllocatingContainerIn(t, parent)
	time.Sleep(10 * time.Second)
	afterInside, _, ok := reader.MeasuredBytes()
	if !ok {
		t.Fatal("the subtree stopped being readable")
	}
	if charged := containerMemoryBytes(t, inside); charged < allocatedBytes/2 {
		t.Fatalf("the container charged %d bytes and never allocated", charged)
	}
	if delta := afterInside - afterOutside; delta < allocatedBytes/2 {
		t.Errorf("a container inside the subtree charged %d bytes and the domain moved by %d: %s does not contain it",
			allocatedBytes, delta, reader.Describe())
	} else {
		t.Logf("a container inside the subtree moved it by %dMiB", delta/(1<<20))
	}
	_ = outside
}

// startAllocatingContainerIn starts the allocating container under a cgroup parent, or
// under the daemon's default when parent is empty.
func startAllocatingContainerIn(t *testing.T, parent string) string {
	t.Helper()
	mib := allocatedBytes / (1 << 20)
	args := []string{"run", "-d", "--rm", "--shm-size", strconv.Itoa(mib) + "m"}
	if parent != "" {
		args = append(args, "--cgroup-parent", parent)
	}
	args = append(args, "alpine", "sh", "-c",
		"dd if=/dev/zero of=/dev/shm/charge bs=1M count="+strconv.Itoa(mib)+" 2>/dev/null; sleep 120")
	out, err := exec.Command("docker", args...).Output()
	if err != nil {
		t.Fatalf("start the allocating container: %v", err)
	}
	id := strings.TrimSpace(string(out))
	t.Cleanup(func() { _ = exec.Command("docker", "rm", "-f", id).Run() })
	return id
}
