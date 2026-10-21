package server

import (
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/placement"
)

func reconcileControl(t *testing.T) *Control {
	t.Helper()
	control, _ := newTestControl(t, &stubNodes{}, time.Now)
	control.remember("sb-1", sandboxRecord{
		handle: backend.Handle{Backend: "stub", SandboxID: "sb-1", NodeID: "node-a"},
		owner:  "owner-1",
	})
	return control
}

func inventory(node string, seq uint64, live ...string) placement.NodeView {
	return placement.NodeView{
		NodeID:            node,
		InventoryReported: true,
		ReportSeq:         seq,
		LiveSandboxIDs:    live,
	}
}

func TestReconcileKeepsAGrantUntilTwoEligibleReportsMissIt(t *testing.T) {
	control := reconcileControl(t)

	// The first report was generated before the grant was recorded, so it
	// cannot say the sandbox is absent.
	control.ReconcileGrants(inventory("node-a", 1))
	if len(control.sandbox) != 1 {
		t.Fatal("a report generated before the grant was recorded must not release it")
	}
	control.ReconcileGrants(inventory("node-a", 2))
	if len(control.sandbox) != 1 {
		t.Fatal("one missed report must not release a grant")
	}
	control.ReconcileGrants(inventory("node-a", 3))
	if len(control.sandbox) != 0 {
		t.Fatal("two eligible reports that both miss the sandbox must release the grant")
	}
}

func TestReconcileKeepsASandboxTheNodeStillLists(t *testing.T) {
	control := reconcileControl(t)
	for seq := uint64(2); seq <= 5; seq++ {
		control.ReconcileGrants(inventory("node-a", seq, "sb-1"))
	}
	if len(control.sandbox) != 1 {
		t.Fatal("a sandbox the node lists must stay charged")
	}
}

func TestReconcileSkipsAnOlderNodeAndATruncatedInventory(t *testing.T) {
	control := reconcileControl(t)
	old := placement.NodeView{NodeID: "node-a", ReportSeq: 1}
	control.ReconcileGrants(old)
	control.ReconcileGrants(old)
	truncated := inventory("node-a", 2)
	truncated.InventoryTruncated = true
	control.ReconcileGrants(truncated)
	control.ReconcileGrants(truncated)
	if len(control.sandbox) != 1 {
		t.Fatal("an absent or partial inventory is not evidence that a sandbox is gone")
	}
}

func TestReconcileResetsTheEpochWhenTheSequenceGoesBackwards(t *testing.T) {
	control := reconcileControl(t)
	// Apply a later report first, so the record is known to be on this node.
	control.ReconcileGrants(inventory("node-a", 5, "sb-1"))

	// The node restarted: its sequence starts over. The first report of the new
	// epoch cannot be compared with the old one and must only re-arm.
	control.ReconcileGrants(inventory("node-a", 1))
	if len(control.sandbox) != 1 {
		t.Fatal("a sequence regression must re-arm rather than release")
	}
	control.ReconcileGrants(inventory("node-a", 2))
	if len(control.sandbox) != 1 {
		t.Fatal("the new epoch's first two reports must not release")
	}
	control.ReconcileGrants(inventory("node-a", 3))
	if len(control.sandbox) != 1 {
		t.Fatal("one miss in the new epoch must not release")
	}
	control.ReconcileGrants(inventory("node-a", 4))
	if len(control.sandbox) != 0 {
		t.Fatal("two eligible misses in the new epoch must release")
	}
}

func TestReconcileIgnoresAnotherNodesInventory(t *testing.T) {
	control := reconcileControl(t)
	for seq := uint64(2); seq <= 4; seq++ {
		control.ReconcileGrants(inventory("node-b", seq))
	}
	if len(control.sandbox) != 1 {
		t.Fatal("a report from another node must not release this node's grant")
	}
}
