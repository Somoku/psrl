package server

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/node"
)

// fakeDirect is a direct backend that records what it was asked to make, so a
// test can assert on creates rather than on a container.
type fakeDirect struct {
	created int
	release []string
	// byKey is what a create with an idempotency key has already made, which is
	// what a lost response leaves behind.
	byKey map[string]backend.Reconciled
}

func (f *fakeDirect) Name() string                 { return "fake" }
func (f *fakeDirect) Mode() backend.SchedulingMode { return backend.SchedulingDirect }
func (f *fakeDirect) Capabilities() backend.Capabilities {
	return backend.Capabilities{Features: []string{"freeze"}}
}

func (f *fakeDirect) Create(
	_ context.Context, nodeID string, spec backend.Spec, _ string,
) (backend.Created, error) {
	f.created++
	handle := backend.Handle{
		Backend: "fake", SandboxID: fmt.Sprintf("container-%d", f.created), NodeID: nodeID,
	}
	if spec.IdempotencyKey != "" {
		if f.byKey == nil {
			f.byKey = map[string]backend.Reconciled{}
		}
		f.byKey[spec.IdempotencyKey] = backend.Reconciled{
			Handle: handle, Owner: "worker", Running: true,
		}
	}
	return backend.Created{Handle: handle, Capabilities: f.Capabilities()}, nil
}

func (f *fakeDirect) Release(_ context.Context, handle backend.Handle) error {
	f.release = append(f.release, handle.SandboxID)
	return nil
}

func (f *fakeDirect) Status(context.Context, backend.Handle) (string, error) {
	return "running", nil
}

func (f *fakeDirect) Reconcile(context.Context) ([]backend.Reconciled, error) {
	return nil, nil
}

func (f *fakeDirect) FindByIdempotencyKey(
	_ context.Context, key string,
) (backend.Reconciled, bool, error) {
	found, ok := f.byKey[key]
	return found, ok, nil
}

func newNodeOverFake(t *testing.T, hosted backend.Backend) (*Node, *node.Admission, *node.Lifecycle, *v1.SandboxSpec) {
	t.Helper()
	gate, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{MemoryMB: 4096, CPUMillis: 100000},
		Classes:  map[string]node.ClassShare{"rollout": {Guaranteed: 1.0}},
		LeaseTTL: time.Minute,
	})
	if err != nil {
		t.Fatalf("admission: %v", err)
	}
	life, err := node.NewLifecycle("node-a", gate, time.Second)
	if err != nil {
		t.Fatalf("lifecycle: %v", err)
	}
	life.SetOwnerLiveness(gate.OwnerAlive)
	agent, err := NewNode(NodeConfig{
		NodeID: "node-a", Admission: gate, Lifecycle: life, Backends: []backend.Backend{hosted},
	})
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	spec := specToProto(backend.Spec{
		Source:         backend.Source{Kind: "image", Reference: "alpine:latest"},
		Resources:      backend.Resources{MemoryMB: 64, CPUCount: 0.25},
		ResourceClass:  "rollout",
		IdempotencyKey: "ticket-lost",
		Metadata:       map[string]string{backend.OwnerMetadataKey: "worker"},
	})
	return agent, gate, life, spec
}

// reserveOne is the admission half of a create, as the control plane does it.
func reserveOne(t *testing.T, agent *Node, spec *v1.SandboxSpec) string {
	t.Helper()
	response, err := agent.Admit(context.Background(), &v1.AdmitRequest{Spec: spec})
	if err != nil {
		t.Fatalf("admit: %v", err)
	}
	if !response.GetAdmitted() {
		t.Fatalf("admit refused: %s", response.GetRefusal())
	}
	return response.GetLeaseId()
}

func TestCreateOnAdoptsByIdempotencyKey(t *testing.T) {
	// A create whose response was lost leaves a container the caller never got a
	// handle for. A retry with the same key must adopt that container rather than
	// build a second one, which would hold twice the memory for one sandbox.
	hosted := &fakeDirect{}
	agent, gate, life, spec := newNodeOverFake(t, hosted)

	handled, err := agent.CreateOn(context.Background(), &v1.CreateOnRequest{
		LeaseId: reserveOne(t, agent, spec), Backend: "fake", Spec: spec,
	})
	if err != nil {
		t.Fatalf("first create: %v", err)
	}
	firstID := handled.GetHandle().GetSandboxId()

	// The retry: a fresh lease, the same key.
	retried, err := agent.CreateOn(context.Background(), &v1.CreateOnRequest{
		LeaseId: reserveOne(t, agent, spec), Backend: "fake", Spec: spec,
	})
	if err != nil {
		t.Fatalf("retry: %v", err)
	}

	if hosted.created != 1 {
		t.Fatalf("the retry made %d containers, want 1", hosted.created)
	}
	if got := retried.GetHandle().GetSandboxId(); got != firstID {
		t.Fatalf("the retry returned sandbox %q, want the first one %q", got, firstID)
	}
	// The container is one, so it must be charged once: the retry's reservation is
	// returned rather than held alongside the first attempt's.
	if granted := gate.Snapshot().Granted.MemoryMB; granted != 64 {
		t.Fatalf("the node accounts %dMB for one container, want 64MB", granted)
	}
	if resident := life.Snapshot().Resident; resident != 1 {
		t.Fatalf("resident %d, want 1", resident)
	}
}

func TestCreateOnChargesAContainerThatOutlivedARestart(t *testing.T) {
	// The same key can also name a container this node no longer has a record
	// for, which is what a restart in between leaves. Then the retry is the first
	// record of it, so the retry's reservation is what charges it.
	hosted := &fakeDirect{byKey: map[string]backend.Reconciled{
		"ticket-lost": {
			Handle: backend.Handle{Backend: "fake", SandboxID: "container-restored", NodeID: "node-a"},
			Owner:  "worker", Running: true,
		},
	}}
	agent, gate, life, spec := newNodeOverFake(t, hosted)

	handled, err := agent.CreateOn(context.Background(), &v1.CreateOnRequest{
		LeaseId: reserveOne(t, agent, spec), Backend: "fake", Spec: spec,
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	if hosted.created != 0 {
		t.Fatalf("the create made %d new containers for a container that exists", hosted.created)
	}
	if got := handled.GetHandle().GetSandboxId(); got != "container-restored" {
		t.Fatalf("adopted %q, want container-restored", got)
	}
	if granted := gate.Snapshot().Granted.MemoryMB; granted != 64 {
		t.Fatalf("the adopted container is charged %dMB, want 64MB", granted)
	}
	if resident := life.Snapshot().Resident; resident != 1 {
		t.Fatalf("resident %d, want 1", resident)
	}
}

// TestOwnerOfNamesTheCallerTheReclaimerKeyedOn pins the identity a node reclaims
// a sandbox by.
//
// Getting this wrong is not a cosmetic bug: ownership is what a silence budget
// measures, so an owner nobody renews reads as a dead caller and every sandbox
// it holds is reclaimed while it is still working. That happened once, because
// ownership was read from `workflow_id` — a phase-reservation key the caller
// usually leaves unset — while the caller stated its owner in metadata.
func TestOwnerOfNamesTheCallerTheReclaimerKeyedOn(t *testing.T) {
	tests := []struct {
		name string
		spec backend.Spec
		want string
	}{
		{
			name: "the owner the caller states",
			spec: backend.Spec{Metadata: map[string]string{backend.OwnerMetadataKey: "w0-host-1-abcd"}},
			want: "w0-host-1-abcd",
		},
		{
			name: "the stated owner wins over a workflow",
			spec: backend.Spec{
				WorkflowID: "phase-1",
				Metadata:   map[string]string{backend.OwnerMetadataKey: "w0-host-1-abcd"},
			},
			want: "w0-host-1-abcd",
		},
		{
			name: "a workflow stands in when no owner is stated",
			spec: backend.Spec{WorkflowID: "phase-1"},
			want: "phase-1",
		},
		{
			name: "neither stated",
			spec: backend.Spec{},
			want: "anonymous",
		},
		{
			name: "an empty owner is not a name",
			spec: backend.Spec{Metadata: map[string]string{backend.OwnerMetadataKey: ""}},
			want: "anonymous",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := ownerOf(test.spec); got != test.want {
				t.Errorf("ownerOf(%+v) = %q, want %q", test.spec, got, test.want)
			}
		})
	}
}
