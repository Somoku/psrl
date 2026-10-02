package opensandbox_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/backend/opensandbox"
)

// -- construction -------------------------------------------------------------

func TestNewPSRLModeRequiresNodes(t *testing.T) {
	_, err := opensandbox.New(opensandbox.Config{}, backend.SchedulingPSRL)
	if err == nil {
		t.Fatal("expected error for psrl mode with no nodes")
	}
}

func TestNewProviderModeRequiresGateway(t *testing.T) {
	_, err := opensandbox.New(opensandbox.Config{}, backend.SchedulingProvider)
	if err == nil {
		t.Fatal("expected error for provider mode with no gateway")
	}
}

func TestNewInvalidModeIsRefused(t *testing.T) {
	_, err := opensandbox.New(opensandbox.Config{Gateway: "http://x"}, backend.SchedulingMode("sideways"))
	if err == nil {
		t.Fatal("expected error for invalid scheduling mode")
	}
}

func TestNewNodeRequiresBothIDAndAddress(t *testing.T) {
	_, err := opensandbox.New(opensandbox.Config{
		Nodes: []opensandbox.NodeAddress{{NodeID: "n1", Address: ""}},
	}, backend.SchedulingPSRL)
	if err == nil {
		t.Fatal("expected error for node with empty address")
	}
}

func TestNewPSRLModeSucceeds(t *testing.T) {
	b, err := opensandbox.New(opensandbox.Config{
		Nodes: []opensandbox.NodeAddress{{NodeID: "n1", Address: "http://node1:8080"}},
	}, backend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if b.Mode() != backend.SchedulingPSRL {
		t.Fatalf("expected psrl mode, got %q", b.Mode())
	}
}

func TestNewProviderModeSucceeds(t *testing.T) {
	b, err := opensandbox.New(opensandbox.Config{
		Gateway: "https://gateway.example.com/v1",
	}, backend.SchedulingProvider)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if b.Mode() != backend.SchedulingProvider {
		t.Fatalf("expected provider mode, got %q", b.Mode())
	}
}

// -- capabilities differ by mode ----------------------------------------------

func TestCapabilitiesPSRLModeExcludesProviderOnlyFeatures(t *testing.T) {
	b := mustNew(t, backend.SchedulingPSRL, "http://n1:8080")
	caps := b.Capabilities()
	providerOnly := []string{"warm_pool", "image_block_delivery", "template_build", "resume_anywhere", "volume"}
	featureSet := make(map[string]bool, len(caps.Features))
	for _, f := range caps.Features {
		featureSet[f] = true
	}
	for _, f := range providerOnly {
		if featureSet[f] {
			t.Errorf("psrl mode declared provider-only feature %q", f)
		}
	}
}

func TestCapabilitiesPSRLModeIncludesDockerModeFeatures(t *testing.T) {
	b := mustNew(t, backend.SchedulingPSRL, "http://n1:8080")
	caps := b.Capabilities()
	required := []string{"freeze", "filesystem_snapshot", "restore", "egress_policy", "credential_injection"}
	featureSet := make(map[string]bool, len(caps.Features))
	for _, f := range caps.Features {
		featureSet[f] = true
	}
	for _, f := range required {
		if !featureSet[f] {
			t.Errorf("psrl mode missing docker-mode feature %q", f)
		}
	}
}

func TestCapabilitiesProviderModeIncludesFullSet(t *testing.T) {
	b := mustNew(t, backend.SchedulingProvider, "")
	caps := b.Capabilities()
	full := []string{"warm_pool", "image_block_delivery", "template_build", "resume_anywhere"}
	featureSet := make(map[string]bool, len(caps.Features))
	for _, f := range caps.Features {
		featureSet[f] = true
	}
	for _, f := range full {
		if !featureSet[f] {
			t.Errorf("provider mode missing full-cluster feature %q", f)
		}
	}
}

func TestCapabilitiesPSRLModeHasNoResumeAnywhere(t *testing.T) {
	// Snapshots in docker mode are node-local; cross-node resume is not possible.
	b := mustNew(t, backend.SchedulingPSRL, "http://n1:8080")
	for _, f := range b.Capabilities().Features {
		if f == "resume_anywhere" {
			t.Fatal("psrl mode must not declare resume_anywhere: docker snapshots are node-local")
		}
	}
}

// -- per-node routing ---------------------------------------------------------

func TestCreatePSRLModeRoutesToChosenNode(t *testing.T) {
	// The adapter makes two calls: POST /sandboxes (create) then
	// GET /sandboxes/{id}/endpoints/{port} (endpoint resolution). Capture
	// only the first path to verify that the create went to /sandboxes.
	var firstPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if firstPath == "" {
			firstPath = r.URL.Path
		}
		json.NewEncoder(w).Encode(map[string]any{"id": "sb-1", "status": map[string]string{"state": "Running"}})
	}))
	defer srv.Close()

	b, err := opensandbox.New(opensandbox.Config{
		Nodes: []opensandbox.NodeAddress{{NodeID: "n1", Address: srv.URL}},
	}, backend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new: %v", err)
	}

	spec := backend.Spec{
		Source:    backend.Source{Kind: "image", Reference: "alpine:latest"},
		Resources: backend.Resources{CPUCount: 1, MemoryMB: 256},
	}
	created, err := b.Create(context.Background(), "n1", spec, "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if created.Handle.NodeID != "n1" {
		t.Errorf("NodeID on handle: got %q, want n1", created.Handle.NodeID)
	}
	if firstPath != "/sandboxes" {
		t.Errorf("first request path: got %q, want /sandboxes", firstPath)
	}
}

func TestCreatePSRLModeUnknownNodeIsRefused(t *testing.T) {
	b := mustNew(t, backend.SchedulingPSRL, "http://n1:8080")
	spec := backend.Spec{Source: backend.Source{Kind: "image", Reference: "alpine:latest"}}
	_, err := b.Create(context.Background(), "unknown-node", spec, "")
	if err == nil {
		t.Fatal("expected error for unknown node id")
	}
}

// -- operation guards ---------------------------------------------------------

func TestSnapshotRefusesFullState(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()
	b := mustNewWithServer(t, backend.SchedulingPSRL, srv.URL)
	handle := backend.Handle{Backend: "opensandbox", SandboxID: "sb-1", NodeID: "n1"}
	_, err := b.Snapshot(context.Background(), handle, "full_state")
	if err == nil {
		t.Fatal("expected error when requesting full_state snapshot")
	}
}

func TestSnapshotRefusesFullStateInProviderMode(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer srv.Close()
	b, err := opensandbox.New(opensandbox.Config{Gateway: srv.URL}, backend.SchedulingProvider)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	handle := backend.Handle{Backend: "opensandbox", SandboxID: "sb-1"}
	_, err = b.Snapshot(context.Background(), handle, "full_state")
	if err == nil {
		t.Fatal("expected error when requesting full_state snapshot in provider mode")
	}
}

func TestPauseRefusesHibernate(t *testing.T) {
	b := mustNew(t, backend.SchedulingPSRL, "http://n1:8080")
	handle := backend.Handle{Backend: "opensandbox", SandboxID: "sb-1", NodeID: "n1"}
	err := b.Pause(context.Background(), handle, "hibernate")
	if err == nil {
		t.Fatal("expected error when requesting hibernate pause")
	}
}

func TestPauseAcceptsFreeze(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	b := mustNewWithServer(t, backend.SchedulingPSRL, srv.URL)
	handle := backend.Handle{Backend: "opensandbox", SandboxID: "sb-1", NodeID: "n1"}
	if err := b.Pause(context.Background(), handle, "freeze"); err != nil {
		t.Fatalf("pause with freeze: %v", err)
	}
}

func TestPauseAcceptsEmptyMode(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	b := mustNewWithServer(t, backend.SchedulingPSRL, srv.URL)
	handle := backend.Handle{Backend: "opensandbox", SandboxID: "sb-1", NodeID: "n1"}
	if err := b.Pause(context.Background(), handle, ""); err != nil {
		t.Fatalf("pause with empty mode: %v", err)
	}
}

// -- nodes listing ------------------------------------------------------------

func TestNodesPSRLModeReturnsConfiguredIDs(t *testing.T) {
	b, err := opensandbox.New(opensandbox.Config{
		Nodes: []opensandbox.NodeAddress{
			{NodeID: "n2", Address: "http://n2:8080"},
			{NodeID: "n1", Address: "http://n1:8080"},
		},
	}, backend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	ids, err := b.Nodes(context.Background())
	if err != nil {
		t.Fatalf("nodes: %v", err)
	}
	if len(ids) != 2 {
		t.Fatalf("expected 2 node ids, got %d", len(ids))
	}
	// Nodes() returns sorted IDs.
	if ids[0] != "n1" || ids[1] != "n2" {
		t.Errorf("unexpected ordering: %v", ids)
	}
}

func TestNodesProviderModeReturnsEmpty(t *testing.T) {
	b := mustNew(t, backend.SchedulingProvider, "")
	ids, err := b.Nodes(context.Background())
	if err != nil {
		t.Fatalf("nodes: %v", err)
	}
	if len(ids) != 0 {
		t.Errorf("provider mode should return no node ids, got %v", ids)
	}
}

// -- preflight ----------------------------------------------------------------

func TestPreflightPSRLModeProbesEachNode(t *testing.T) {
	probed := make(map[string]int)
	srv1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		probed["n1"]++
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer srv1.Close()
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		probed["n2"]++
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer srv2.Close()

	b, err := opensandbox.New(opensandbox.Config{
		Nodes: []opensandbox.NodeAddress{
			{NodeID: "n1", Address: srv1.URL},
			{NodeID: "n2", Address: srv2.URL},
		},
	}, backend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := b.Preflight(context.Background()); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	if probed["n1"] == 0 || probed["n2"] == 0 {
		t.Errorf("preflight did not probe both nodes: %v", probed)
	}
}

func TestPreflightPSRLModeFailsIfANodeIsDown(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}))
	defer srv.Close()

	b, err := opensandbox.New(opensandbox.Config{
		Nodes: []opensandbox.NodeAddress{
			{NodeID: "up",   Address: srv.URL},
			{NodeID: "down", Address: "http://127.0.0.1:1"}, // nothing listening
		},
	}, backend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := b.Preflight(context.Background()); err == nil {
		t.Fatal("expected preflight to fail when a node is unreachable")
	}
}

// -- resource limits ----------------------------------------------------------

func TestCreateSendsResourceLimitsAsKubernetesQuantities(t *testing.T) {
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		json.NewEncoder(w).Encode(map[string]any{"id": "sb-1"})
	}))
	defer srv.Close()
	b := mustNewWithServer(t, backend.SchedulingPSRL, srv.URL)
	spec := backend.Spec{
		Source:    backend.Source{Kind: "image", Reference: "alpine:latest"},
		Resources: backend.Resources{CPUCount: 2, MemoryMB: 512},
	}
	if _, err := b.Create(context.Background(), "n1", spec, ""); err != nil {
		t.Fatalf("create: %v", err)
	}
	limits, _ := body["resourceLimits"].(map[string]any)
	if limits["cpu"] != "2000m" {
		t.Errorf("cpu: got %q, want 2000m", limits["cpu"])
	}
	if limits["memory"] != "512Mi" {
		t.Errorf("memory: got %q, want 512Mi", limits["memory"])
	}
}

// -- helpers ------------------------------------------------------------------

func mustNew(t *testing.T, mode backend.SchedulingMode, nodeAddr string) *opensandbox.Backend {
	t.Helper()
	var cfg opensandbox.Config
	if mode == backend.SchedulingPSRL {
		cfg.Nodes = []opensandbox.NodeAddress{{NodeID: "n1", Address: nodeAddr}}
	} else {
		cfg.Gateway = "https://gateway.example.com/v1"
	}
	b, err := opensandbox.New(cfg, mode)
	if err != nil {
		t.Fatalf("opensandbox.New: %v", err)
	}
	return b
}

func mustNewWithServer(t *testing.T, mode backend.SchedulingMode, serverURL string) *opensandbox.Backend {
	t.Helper()
	var cfg opensandbox.Config
	if mode == backend.SchedulingPSRL {
		cfg.Nodes = []opensandbox.NodeAddress{{NodeID: "n1", Address: serverURL}}
	} else {
		cfg.Gateway = serverURL
	}
	b, err := opensandbox.New(cfg, mode)
	if err != nil {
		t.Fatalf("opensandbox.New: %v", err)
	}
	return b
}
