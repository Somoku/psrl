package server_test

// Two-plane integration test: verifies the design claim that a caller on the
// control plane can create, exec, and release a sandbox that lives on a remote
// node, with the same code path as a single-process deployment.
//
// Design step 2.4 exit criterion: harness_agent_loop and the MiniSWE runner are
// unchanged while running against a remote node. The proxy to that exit criterion
// at the Go layer is this test: one goroutine runs a node-plane listener (the
// shape a standalone node process would have), another runs the control plane
// talking to it via FleetNodeClient, and the test verifies that create/exec/release
// round-trips over the JSON node-plane protocol without the control plane and
// the node knowing they share a machine.
//
// What this covers:
//   - RemoteNodeClient admit → leaseID crosses the wire correctly
//   - RemoteNodeClient createOn → backend.Created including Handle crosses
//   - RemoteNodeClient statusOn → string state crosses
//   - RemoteNodeClient execOn → exit code and stdout cross
//   - RemoteNodeClient releaseOn → sandbox is gone
//   - FleetNodeClient routes by nodeID
//   - NodeListener dispatches all five methods without error
//
// What this does not cover: a real container runtime. The backend is a fakeExecBackend
// that keeps sandboxes in memory and returns a canned exec output. The test is
// about the wire protocol and the routing, not about Docker.

import (
	"context"
	"net"
	"strings"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/server"
)

// fakeExecBackend is a minimal backend that keeps sandboxes in memory.
type fakeExecBackend struct {
	name     string
	created  map[string]bool
}

func newFakeExecBackend(name string) *fakeExecBackend {
	return &fakeExecBackend{name: name, created: map[string]bool{}}
}

func (f *fakeExecBackend) Name() string { return f.name }
func (f *fakeExecBackend) Mode() backend.SchedulingMode {
	return backend.SchedulingDirect
}
func (f *fakeExecBackend) Capabilities() backend.Capabilities {
	return backend.Capabilities{}
}
func (f *fakeExecBackend) Nodes(_ context.Context) ([]string, error) {
	return []string{"local"}, nil
}
func (f *fakeExecBackend) Create(_ context.Context, nodeID string, spec backend.Spec, _ string) (backend.Created, error) {
	id := "sb-" + spec.WorkflowID
	f.created[id] = true
	return backend.Created{
		Handle: backend.Handle{Backend: f.name, SandboxID: id, NodeID: nodeID},
	}, nil
}
func (f *fakeExecBackend) Release(_ context.Context, handle backend.Handle) error {
	delete(f.created, handle.SandboxID)
	return nil
}
func (f *fakeExecBackend) Status(_ context.Context, handle backend.Handle) (string, error) {
	if f.created[handle.SandboxID] {
		return "running", nil
	}
	return "terminated", nil
}
func (f *fakeExecBackend) Exec(_ context.Context, _ backend.Handle, command, _ string, _ map[string]string) (int, string, error) {
	return 0, "hello-from-" + f.name, nil
}

// buildNodeForTest builds a real Node backed by the fake backend, returning it
// and the admission/lifecycle it uses so the test can control them.
func buildNodeForTest(t *testing.T, nodeID string, b backend.Backend) *server.Node {
	t.Helper()
	gate, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{MemoryMB: 4096, CPUMillis: 4000},
		Classes:  map[string]node.ClassShare{"default": {Guaranteed: 0.5, Max: 1.0}},
		LeaseTTL: 30 * time.Second,
	})
	if err != nil {
		t.Fatalf("admission: %v", err)
	}
	life, err := node.NewLifecycle(nodeID, gate, node.Windows{
		PauseWindow:   5 * time.Second,
		ReapWindow:    10 * time.Second,
		Lifetime:      60 * time.Second,
		SweepInterval: 2 * time.Second,
	})
	if err != nil {
		t.Fatalf("lifecycle: %v", err)
	}
	n, err := server.NewNode(server.NodeConfig{
		NodeID: nodeID, Admission: gate, Lifecycle: life, Backends: []backend.Backend{b},
	})
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	return n
}

// startNodeListener starts a NodeListener on a random TCP port and returns the address.
func startNodeListener(t *testing.T, n *server.Node) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	nl := server.NewNodeListener(n, nil)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(func() {
		cancel()
		l.Close()
	})
	go func() { _ = nl.Serve(ctx, l) }()
	return l.Addr().String()
}

// TestTwoPlaneCreateExecRelease is the step 2.4 integration test.
//
// A control plane talks to a node plane over the node-plane JSON protocol.
// The test verifies that admit → create → exec → status → release all complete
// and that the exec output and sandbox state are correct at each step.
func TestTwoPlaneCreateExecRelease(t *testing.T) {
	b := newFakeExecBackend("fake")
	n := buildNodeForTest(t, "node-1", b)
	addr := startNodeListener(t, n)

	remote := server.NewRemoteNodeClient(addr, 10*time.Second)
	fleet := server.NewFleetNodeClient(map[string]*server.RemoteNodeClient{
		"node-1": remote,
	})

	ctx := context.Background()
	spec := backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "alpine:latest"},
		ResourceClass: "default",
		WorkflowID:    "wf-twoplane-test",
		Resources:     backend.Resources{MemoryMB: 256, CPUCount: 0.5},
	}

	// Step 1: admission
	leaseID, gpus, refusal, err := fleet.Admit(ctx, "node-1", spec)
	if err != nil {
		t.Fatalf("admit: %v", err)
	}
	if refusal != "" {
		t.Fatalf("unexpected refusal: %q", refusal)
	}
	if leaseID == "" {
		t.Fatal("admit returned empty lease_id")
	}
	_ = gpus

	// Step 2: create
	spec.AssignedGPUs = gpus
	created, err := fleet.CreateOn(ctx, "node-1", leaseID, "fake", spec, "")
	if err != nil {
		t.Fatalf("create_on: %v", err)
	}
	if created.Handle.SandboxID == "" {
		t.Fatal("create_on returned empty sandbox_id")
	}
	if created.Handle.NodeID != "node-1" {
		t.Errorf("node_id: got %q, want node-1", created.Handle.NodeID)
	}

	// Step 3: status should be running
	state, err := fleet.StatusOn(ctx, created.Handle)
	if err != nil {
		t.Fatalf("status_on: %v", err)
	}
	if state != "running" {
		t.Errorf("status before release: got %q, want running", state)
	}

	// Step 4: exec — the node proxies for a backend whose sandboxes have no
	// in-sandbox agent, which is exactly what the docker backend does.
	code, stdout, err := fleet.ExecOn(ctx, created.Handle, "echo hello", "", nil)
	if err != nil {
		t.Fatalf("exec_on: %v", err)
	}
	if code != 0 {
		t.Errorf("exec exit code: got %d, want 0", code)
	}
	if !strings.Contains(stdout, "hello-from-fake") {
		t.Errorf("exec stdout: got %q, want it to contain hello-from-fake", stdout)
	}

	// Step 5: release
	if err := fleet.ReleaseOn(ctx, created.Handle); err != nil {
		t.Fatalf("release_on: %v", err)
	}

	// Step 6: status should now be terminated
	state, err = fleet.StatusOn(ctx, created.Handle)
	if err != nil {
		t.Fatalf("status_on after release: %v", err)
	}
	if state != "terminated" {
		t.Errorf("status after release: got %q, want terminated", state)
	}
}

// TestTwoPlaneMultiNodeRouting verifies that FleetNodeClient routes to the
// correct node when there are two nodes configured.
func TestTwoPlaneMultiNodeRouting(t *testing.T) {
	b1 := newFakeExecBackend("fake1")
	b2 := newFakeExecBackend("fake2")
	n1 := buildNodeForTest(t, "node-a", b1)
	n2 := buildNodeForTest(t, "node-b", b2)
	addr1 := startNodeListener(t, n1)
	addr2 := startNodeListener(t, n2)

	fleet := server.NewFleetNodeClient(map[string]*server.RemoteNodeClient{
		"node-a": server.NewRemoteNodeClient(addr1, 5*time.Second),
		"node-b": server.NewRemoteNodeClient(addr2, 5*time.Second),
	})

	ctx := context.Background()
	spec := backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "alpine:latest"},
		ResourceClass: "default",
		WorkflowID:    "wf-routing-test",
		Resources:     backend.Resources{MemoryMB: 256},
	}

	// Place on node-b explicitly.
	leaseID, _, refusal, err := fleet.Admit(ctx, "node-b", spec)
	if err != nil || refusal != "" {
		t.Fatalf("admit node-b: err=%v refusal=%q", err, refusal)
	}
	created, err := fleet.CreateOn(ctx, "node-b", leaseID, "fake2", spec, "")
	if err != nil {
		t.Fatalf("create_on node-b: %v", err)
	}
	if created.Handle.NodeID != "node-b" {
		t.Errorf("handle node_id: got %q, want node-b", created.Handle.NodeID)
	}

	// Exec must reach node-b's backend, not node-a's.
	_, stdout, err := fleet.ExecOn(ctx, created.Handle, "echo hello", "", nil)
	if err != nil {
		t.Fatalf("exec_on: %v", err)
	}
	if !strings.Contains(stdout, "hello-from-fake2") {
		t.Errorf("exec reached wrong node: got stdout %q, want hello-from-fake2", stdout)
	}

	// node-a must not have seen the sandbox.
	if len(b1.created) != 0 {
		t.Errorf("node-a saw %d creates but should have seen none", len(b1.created))
	}
}

// TestTwoPlaneExecOnUnknownNodeReturnsError verifies that ExecOn named to a
// node not in the fleet returns an error rather than panicking or timing out.
func TestTwoPlaneExecOnUnknownNodeReturnsError(t *testing.T) {
	fleet := server.NewFleetNodeClient(map[string]*server.RemoteNodeClient{
		"node-a": server.NewRemoteNodeClient("127.0.0.1:1", 1*time.Second),
	})
	handle := backend.Handle{Backend: "fake", SandboxID: "sb-1", NodeID: "node-missing"}
	_, _, err := fleet.ExecOn(context.Background(), handle, "echo hi", "", nil)
	if err == nil {
		t.Fatal("expected error for sandbox on an unknown node, got nil")
	}
}
