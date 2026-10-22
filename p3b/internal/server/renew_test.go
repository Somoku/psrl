package server

import (
	"context"
	"sync"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/placement"
)

// renewClient is a NodeClient whose renewal path can hold one node open, which
// is how a test reproduces the incident: one node unreachable while another
// still holds live sandboxes.
type renewClient struct {
	*stubNodes
	mu     sync.Mutex
	called []string
	block  map[string]<-chan struct{}
}

func (r *renewClient) RenewOwnerOn(ctx context.Context, nodeID, owner string) error {
	r.mu.Lock()
	r.called = append(r.called, nodeID)
	block := r.block[nodeID]
	r.mu.Unlock()
	if block != nil {
		select {
		case <-block:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

func (r *renewClient) calledNodes() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.called...)
}

func reportNode(t *testing.T, control *Control, id string) {
	t.Helper()
	control.monitor.Report(placement.NodeView{NodeID: id, SeenAt: time.Now()})
}

func TestRenewOwnerIsBoundedAndStillReachesTheHealthyNode(t *testing.T) {
	// One node accepts the call and never answers. The batch must return inside
	// its budget, and the node that can answer must still be renewed: a stalled
	// node is a delay for itself, never for the fleet.
	never := make(chan struct{})
	defer close(never)
	nodes := &renewClient{
		stubNodes: &stubNodes{},
		block:     map[string]<-chan struct{}{"node-b": never},
	}
	control, _ := newTestControl(t, nodes, time.Now)
	reportNode(t, control, "node-a")
	reportNode(t, control, "node-b")

	started := time.Now()
	control.RenewOwner(context.Background(), "owner-1")
	elapsed := time.Since(started)

	if elapsed > renewBatchBudget+time.Second {
		t.Fatalf("renewal took %s, want no more than the %s budget", elapsed, renewBatchBudget)
	}
	called := nodes.calledNodes()
	if len(called) != 2 {
		t.Fatalf("renewal reached %v, want both live nodes", called)
	}
	_, failures, targeted, last := control.RenewStats()
	if targeted != 2 {
		t.Fatalf("targeted %d nodes, want 2", targeted)
	}
	if failures != 1 {
		t.Fatalf("recorded %d failures, want 1 (the stalled node)", failures)
	}
	if last <= 0 {
		t.Fatal("the renewal duration must be recorded for diagnostics")
	}
}

func TestRenewOwnerTargetsOnlyTheNodesHoldingTheOwnersSandboxes(t *testing.T) {
	// A node that once reported but holds nothing for this owner must not be on
	// the heartbeat's path, even when it is unreachable.
	never := make(chan struct{})
	defer close(never)
	nodes := &renewClient{
		stubNodes: &stubNodes{},
		block:     map[string]<-chan struct{}{"node-b": never},
	}
	control, _ := newTestControl(t, nodes, time.Now)
	reportNode(t, control, "node-a")
	reportNode(t, control, "node-b")
	control.remember("sb-1", sandboxRecord{
		handle: backend.Handle{Backend: "stub", SandboxID: "sb-1", NodeID: "node-a"},
		owner:  "owner-1",
	})

	control.RenewOwner(context.Background(), "owner-1")

	if called := nodes.calledNodes(); len(called) != 1 || called[0] != "node-a" {
		t.Fatalf("renewal reached %v, want only node-a", called)
	}
}

func TestRenewOwnerReachesANodeTheControlHasNoRecordFor(t *testing.T) {
	// A control restart or a lost release can leave a live sandbox out of
	// c.sandbox. The node's own LiveOwners is then the only evidence the
	// control has, and without it that node's owner lease expires and the node
	// reclaims every sandbox it holds.
	nodes := &renewClient{stubNodes: &stubNodes{}}
	control, _ := newTestControl(t, nodes, time.Now)
	control.monitor.Report(placement.NodeView{
		NodeID: "node-a", SeenAt: time.Now(), LiveOwners: []string{"owner-1"},
	})
	control.monitor.Report(placement.NodeView{
		NodeID: "node-b", SeenAt: time.Now(), LiveOwners: []string{"owner-other"},
	})

	control.RenewOwner(context.Background(), "owner-1")

	if called := nodes.calledNodes(); len(called) != 1 || called[0] != "node-a" {
		t.Fatalf("renewal reached %v, want only the node reporting the owner", called)
	}
}

func TestRenewOwnerFallsBackToTheLiveFleetAfterAControlRestart(t *testing.T) {
	// A control plane that has no records yet still has to keep the caller's
	// sandboxes alive; the live fleet is the only evidence it has.
	nodes := &renewClient{stubNodes: &stubNodes{}}
	control, _ := newTestControl(t, nodes, time.Now)
	reportNode(t, control, "node-a")
	reportNode(t, control, "node-b")

	control.RenewOwner(context.Background(), "owner-with-no-records")

	if called := nodes.calledNodes(); len(called) != 2 {
		t.Fatalf("renewal reached %v, want the whole live fleet", called)
	}
}
