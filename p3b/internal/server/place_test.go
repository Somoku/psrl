package server

import (
	"context"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/monitor"
	"psrl.dev/sandboxd/internal/placement"
	"psrl.dev/sandboxd/internal/quota"
)

// failAdmitClient makes admission fail for named nodes, so a test can pin that
// one unreachable machine does not fail a create the other node can serve.
type failAdmitClient struct {
	*stubNodes
	fail   map[string]bool
	admits []string
}

func (c *failAdmitClient) Admit(_ context.Context, nodeID string, _ backend.Spec) (string, []int32, string, error) {
	c.admits = append(c.admits, nodeID)
	if c.fail[nodeID] {
		return "", nil, "", status.Error(codes.Unavailable, "node did not answer")
	}
	return "lease-" + nodeID, nil, "", nil
}

func (c *failAdmitClient) CreateOn(
	_ context.Context, nodeID, _, backendName string, _ backend.Spec, _ string,
) (backend.Created, error) {
	return backend.Created{Handle: backend.Handle{
		Backend: backendName, SandboxID: "sb-" + nodeID, NodeID: nodeID,
	}}, nil
}

func directControl(t *testing.T, nodes NodeClient, nodeIDs ...string) *Control {
	t.Helper()
	fleet := monitor.New(2 * time.Minute)
	place, err := placement.New(placement.Config{
		NodeTTL: 2 * time.Minute, ReservationTTL: time.Minute, SweepInterval: 15 * time.Second,
	}, fleet)
	if err != nil {
		t.Fatalf("placement: %v", err)
	}
	ledger, err := quota.New(quota.Config{
		Total:    quota.Amount{MemoryMB: 4096, CPUMillis: 8000, Sandboxes: 1 << 20},
		Classes:  map[string]quota.ClassShare{"rollout": {Guaranteed: 1.0}},
		LeaseTTL: time.Hour,
	})
	if err != nil {
		t.Fatalf("quota: %v", err)
	}
	registry, err := backend.NewRegistry([]backend.Backend{&fakeDirect{}}, "fake")
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	control, err := NewControl(ControlConfig{
		Registry: registry, Ledger: ledger, Placement: place, Monitor: fleet, Nodes: nodes,
	})
	if err != nil {
		t.Fatalf("control: %v", err)
	}
	for _, id := range nodeIDs {
		fleet.Report(placement.NodeView{
			NodeID:   id,
			SeenAt:   time.Now(),
			Backends: []placement.BackendCapability{{Name: "fake"}},
			Envelope: placement.Headroom{MemoryMB: 4096, CPUMillis: 8000},
			ClassHeadroom: map[string]placement.Headroom{
				"rollout": {MemoryMB: 4096, CPUMillis: 8000},
			},
		})
	}
	return control
}

func placeSpec() *v1.SandboxSpec {
	return specToProto(backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "alpine:latest"},
		Resources:     backend.Resources{MemoryMB: 64, CPUCount: 0.25},
		ResourceClass: "rollout",
		Metadata:      map[string]string{backend.OwnerMetadataKey: "worker"},
	})
}

func TestCreateSkipsANodeThatDoesNotAnswerAdmission(t *testing.T) {
	nodes := &failAdmitClient{
		stubNodes: &stubNodes{},
		fail:      map[string]bool{"node-a": true},
	}
	control := directControl(t, nodes, "node-a", "node-b")

	created, err := control.Create(context.Background(), &v1.CreateRequest{Spec: placeSpec()})
	if err != nil {
		t.Fatalf("a create must reach the node that can answer: %v", err)
	}
	if got := created.GetHandle().GetNodeId(); got != "node-b" {
		t.Fatalf("sandbox landed on %q, want node-b", got)
	}
	if len(nodes.admits) != 2 {
		t.Fatalf("admission was asked %d times, want two attempts (a then b)", len(nodes.admits))
	}
}

func TestCreateReportsATransportFaultWhenNoNodeAnswers(t *testing.T) {
	nodes := &failAdmitClient{
		stubNodes: &stubNodes{},
		fail:      map[string]bool{"node-a": true, "node-b": true},
	}
	control := directControl(t, nodes, "node-a", "node-b")

	_, err := control.Create(context.Background(), &v1.CreateRequest{Spec: placeSpec()})
	if status.Code(err) != codes.Unavailable {
		t.Fatalf("every node unreachable must be Unavailable, got %v", err)
	}
}
