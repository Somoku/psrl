//go:build integration

package server

import (
	"os"
	"path/filepath"
	"testing"

	"psrl.dev/sandboxd/internal/cgroup"
)

// TestAnOrphanedCgroupIsRemovedFromEveryHierarchy is the half of reaping the unit tests
// cannot reach: whether the kernel will actually let go.
//
// A tmpfs directory full of files cannot be rmdir'd, so the guards are testable there and
// the removal is not. Here the directory is a real cgroup: the control files are provided
// by the kernel rather than stored in it, so rmdir succeeds -- and rmdir is refused for a
// cgroup that still holds a task, which is the guard no caller can get wrong.
//
// It also asserts the removal happened on every hierarchy. A cgroup gone from memory but
// still present in the accounting one is invisible to half the controller and still
// charged by the other half, which is worse than leaving it alone.
func TestAnOrphanedCgroupIsRemovedFromEveryHierarchy(t *testing.T) {
	domain, err := cgroup.Resolve()
	if err != nil {
		t.Skipf("no readable cgroup hierarchy on this host: %v", err)
	}
	const parent = "psrl-reap-test"
	budget, err := domain.Setup(parent, "host", cgroup.Request{MemoryMB: 4096, Overcommit: 1, CPUMillis: 1000})
	if err != nil {
		t.Fatalf("create the sandbox subtree: %v", err)
	}
	t.Logf("subtree budget: %+v", budget)
	defer func() {
		_ = domain.RemoveChild("", parent)
	}()

	agent := benchNode(t, domain, parent)
	id := "orphan-1"
	// Created by hand in the memory hierarchy rather than through Setup, because Setup
	// moves the calling process into the leaf it creates and an orphan by definition has
	// nothing in it -- a cgroup with a task in it is one rmdir must refuse.
	memoryRoot := ""
	for _, hierarchy := range domain.Hierarchies {
		if hierarchy.Has("memory") {
			memoryRoot = hierarchy.Root
		}
	}
	if memoryRoot == "" {
		t.Skip("no memory hierarchy on this host")
	}
	if err := os.MkdirAll(filepath.Join(memoryRoot, parent, id), 0o755); err != nil {
		t.Fatalf("create the orphan: %v", err)
	}
	// A cgroup created this way has no usage, no tasks and no owner: exactly the shape a
	// destroyed container leaves behind.
	children, err := domain.Children(parent)
	if err != nil {
		t.Fatalf("read the domain: %v", err)
	}
	found := 0
	for _, child := range children {
		if child.ID == id {
			found++
		}
	}
	if found == 0 {
		t.Fatalf("the orphan is not in the domain: %v", children)
	}

	reaped := agent.reapForeign(children)
	if len(reaped) != 1 || reaped[0] != id {
		t.Fatalf("reaped: got %v, want the orphan", reaped)
	}
	for _, hierarchy := range domain.Hierarchies {
		if _, err := os.Stat(filepath.Join(hierarchy.Root, parent, id)); !os.IsNotExist(err) {
			t.Errorf("the orphan is still in the %v hierarchy", hierarchy.Controllers)
		}
	}

}
