package placement

import (
	"errors"
	"strings"
	"testing"
	"time"
)

// TestARefusedNodeIsNotAskedAgain asserts the retry behaviour the control plane relies on.
//
// Placement ranks on reported headroom, which is a sample from the node's past. A refusal is
// the node's own ledger speaking about now, so it is the stronger of the two signals, and a
// retry that ignores it asks the same question and gets the same answer.
func TestARefusedNodeIsNotAskedAgain(t *testing.T) {
	node := NodeView{
		NodeID: "n1", SeenAt: time.Now(),
		Envelope:      Headroom{MemoryMB: 1000, CPUMillis: 20000},
		ClassHeadroom: map[string]Headroom{"rollout": {MemoryMB: 1000, CPUMillis: 20000}},
		Backends:      []BackendCapability{{Name: "docker"}},
	}
	service := mustService(t, &fakeMonitor{nodes: []NodeView{node}})

	want := Request{
		Backend: "docker", ResourceClass: "rollout",
		Footprint: Headroom{MemoryMB: 100, CPUMillis: 2000},
	}
	first, err := service.Choose(want)
	if err != nil {
		t.Fatalf("first choose: %v", err)
	}
	if first.NodeID != "n1" {
		t.Fatalf("first choose: got %q", first.NodeID)
	}

	// n1 refused, so the same request must not be sent back to it.
	want.ExcludeNodes = []string{"n1"}
	_, retryErr := service.Choose(want)
	if retryErr == nil {
		t.Fatal("a request was placed on a node it had just been refused by")
	}
	// And the refusal has to come back as exhaustion: the node is full, nothing is missing.
	// A capability error here would tell the caller the request can never be served, and the
	// caller would drop the trajectory instead of waiting for the node to free up.
	if !errors.Is(retryErr, ErrExhausted) {
		t.Errorf("a request refused by every node must report exhaustion, got %v", retryErr)
	}
	if got := retryErr.Error(); got != "" && !strings.Contains(got, "refused") {
		t.Errorf("the message must say the node refused it: %s", got)
	}
}

// TestAllNodesExcludedReportsExhaustionNotCapability is the regression for the production
// error `no sandbox node satisfies this request (live nodes: 0 ...)`. Live nodes is zero
// only because every live node was already excluded by this same request: the fleet was
// healthy and simply full. Reporting that as a capability gap turned a transient load event
// into a permanent-looking failure, and the caller dropped the work instead of retrying it.
