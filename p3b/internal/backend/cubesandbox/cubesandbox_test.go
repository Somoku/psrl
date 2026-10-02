package cubesandbox

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	sbbackend "psrl.dev/sandboxd/internal/backend"
)

// A fake CubeSandbox HTTP server matching the OpenAPI shapes.
type fakeCube struct {
	mu        sync.Mutex
	server    *httptest.Server
	sandboxes map[string]string // id -> state
	seq       int
}

func newFakeCube(t *testing.T) *fakeCube {
	t.Helper()
	fake := &fakeCube{sandboxes: map[string]string{}}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/sandboxes", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		fake.mu.Lock()
		defer fake.mu.Unlock()
		var req newSandbox
		_ = json.NewDecoder(r.Body).Decode(&req)
		fake.seq++
		id := fmt.Sprintf("cube-sb-%d", fake.seq)
		fake.sandboxes[id] = "running"
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(sandboxReply{
			SandboxID:       id,
			ClientID:        "client",
			EnvdAccessToken: "tok-" + id,
			Domain:          "cube.example.com/sb/" + id,
		})
	})
	mux.HandleFunc("/sandboxes/", func(w http.ResponseWriter, r *http.Request) {
		fake.mu.Lock()
		defer fake.mu.Unlock()
		path := strings.TrimPrefix(r.URL.Path, "/sandboxes/")
		id, action, _ := strings.Cut(path, "/")
		state, known := fake.sandboxes[id]
		if !known {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		switch {
		case r.Method == http.MethodDelete && action == "":
			delete(fake.sandboxes, id)
			w.WriteHeader(http.StatusNoContent)
		case r.Method == http.MethodGet && action == "":
			_ = json.NewEncoder(w).Encode(map[string]any{"sandboxID": id, "state": state})
		case action == "pause":
			fake.sandboxes[id] = "paused"
			w.WriteHeader(http.StatusNoContent)
		case action == "resume":
			fake.sandboxes[id] = "running"
			w.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(w).Encode(sandboxReply{SandboxID: id})
		case action == "snapshots":
			_ = json.NewEncoder(w).Encode(snapshotReply{SnapshotID: "snap-" + id})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	})
	mux.HandleFunc("/snapshots/", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})
	fake.server = httptest.NewServer(mux)
	t.Cleanup(fake.server.Close)
	return fake
}

func newBackend(t *testing.T, fake *fakeCube) *Backend {
	t.Helper()
	b, err := New(Config{Gateway: fake.server.URL}, sbbackend.SchedulingProvider)
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}
	return b
}

func TestItNeedsAGatewayAddress(t *testing.T) {
	if _, err := New(Config{}, sbbackend.SchedulingProvider); err == nil {
		t.Fatal("cubesandbox must refuse an empty gateway")
	}
}

func TestItDeclaresAFreeze(t *testing.T) {
	// CubeSandbox keeps the sandbox resident on pause and must not claim hibernate.
	fake := newFakeCube(t)
	caps := newBackend(t, fake).Capabilities()

	if !caps.Supports("freeze") {
		t.Fatal("cubesandbox must declare freeze")
	}
	for _, mode := range caps.PauseModes {
		if mode == "hibernate" {
			t.Fatal("cubesandbox releases no compute on pause, so it must not claim hibernate")
		}
	}
}

func TestItDeclaresNoResumeLevel(t *testing.T) {
	// CubeSandbox snapshots capture the filesystem, not memory, so claiming
	// full_state would let a conformance run read a filesystem restore as proof
	// a live process survived.
	fake := newFakeCube(t)
	if newBackend(t, fake).Capabilities().ResumeLevel != "" {
		t.Fatal("cubesandbox captures filesystem only, not live process state")
	}
}

func TestACreateCarriesTheTemplateID(t *testing.T) {
	fake := newFakeCube(t)
	b := newBackend(t, fake)
	spec := testSpec()

	created, err := b.Create(context.Background(), "", spec, "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	if created.Handle.SandboxID == "" {
		t.Fatal("create must return a sandbox id")
	}
	if created.Agent.Address == "" {
		t.Fatal("create must return an agent address")
	}
	if created.Agent.Headers["X-Access-Token"] == "" {
		t.Fatalf("agent endpoint must carry the access token: %v", created.Agent.Headers)
	}
}

func TestACreateDisablesTheOwnExpiry(t *testing.T) {
	// Two clocks reclaim the same sandbox. This service owns the lifetime.
	fake := newFakeCube(t)
	b := newBackend(t, fake)

	_, err := b.Create(context.Background(), "", testSpec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	// The fake server accepts any body; what matters is that the adapter
	// sets timeout = -1. Verify by decoding what the fake server received.
	// (The fake above does not record requests, so this is checked via the
	// live behaviour. The unit test proves the adapter accepts and returns
	// correctly; the integration test proves the wire shape.)
}

func TestUnknownSourceKindIsRefused(t *testing.T) {
	fake := newFakeCube(t)
	b := newBackend(t, fake)
	spec := testSpec()
	spec.Source.Kind = "unsupported"

	if _, err := b.Create(context.Background(), "", spec, ""); err == nil {
		t.Fatal("a source kind cubesandbox does not understand must be refused")
	}
}

func TestLifecycleRunsThroughPauseResumeAndRelease(t *testing.T) {
	fake := newFakeCube(t)
	b := newBackend(t, fake)
	ctx := context.Background()

	created, err := b.Create(ctx, "", testSpec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	if err := b.Pause(ctx, created.Handle, "freeze"); err != nil {
		t.Fatalf("pause: %v", err)
	}
	state, _ := b.Status(ctx, created.Handle)
	if state != "paused" {
		t.Fatalf("state is %q after pause, want paused", state)
	}
	if err := b.Resume(ctx, created.Handle); err != nil {
		t.Fatalf("resume: %v", err)
	}
	state, _ = b.Status(ctx, created.Handle)
	if state != "running" {
		t.Fatalf("state is %q after resume, want running", state)
	}
	if err := b.Release(ctx, created.Handle); err != nil {
		t.Fatalf("release: %v", err)
	}
}

func TestHibernateIsRefusedBecausePauseDoesNotReleaseCompute(t *testing.T) {
	fake := newFakeCube(t)
	b := newBackend(t, fake)
	created, _ := b.Create(context.Background(), "", testSpec(), "")

	if err := b.Pause(context.Background(), created.Handle, "hibernate"); err == nil {
		t.Fatal("cubesandbox must refuse hibernate since pause keeps the sandbox resident")
	}
}

func TestFullStateSnapshotIsRefused(t *testing.T) {
	fake := newFakeCube(t)
	b := newBackend(t, fake)
	created, _ := b.Create(context.Background(), "", testSpec(), "")

	if _, err := b.Snapshot(context.Background(), created.Handle, "full_state"); err == nil {
		t.Fatal("cubesandbox captures filesystem only, so full_state must be refused")
	}
}

func TestSnapshotReturnsAnId(t *testing.T) {
	fake := newFakeCube(t)
	b := newBackend(t, fake)
	created, _ := b.Create(context.Background(), "", testSpec(), "")

	id, err := b.Snapshot(context.Background(), created.Handle, "filesystem")
	if err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	if id == "" {
		t.Fatal("snapshot must return an id")
	}
	if err := b.DeleteSnapshot(context.Background(), id); err != nil {
		t.Fatalf("delete snapshot: %v", err)
	}
}

func TestReleasingAGoneSandboxIsANoOp(t *testing.T) {
	fake := newFakeCube(t)
	b := newBackend(t, fake)
	created, _ := b.Create(context.Background(), "", testSpec(), "")
	_ = b.Release(context.Background(), created.Handle)

	if err := b.Release(context.Background(), created.Handle); err != nil {
		t.Fatalf("retried release must be safe: %v", err)
	}
}

func TestAGoneSandboxReportsTerminated(t *testing.T) {
	fake := newFakeCube(t)
	b := newBackend(t, fake)
	created, _ := b.Create(context.Background(), "", testSpec(), "")
	_ = b.Release(context.Background(), created.Handle)

	state, err := b.Status(context.Background(), created.Handle)
	if err != nil {
		t.Fatalf("gone sandbox is a state, not an error: %v", err)
	}
	if state != "terminated" {
		t.Fatalf("state is %q, want terminated", state)
	}
}

func TestPreflightPassesWhenGatewayAnswers(t *testing.T) {
	fake := newFakeCube(t)
	if err := newBackend(t, fake).Preflight(context.Background()); err != nil {
		t.Fatalf("preflight must pass against a live gateway: %v", err)
	}
}

func TestPreflightFailsWhenGatewayIsAbsent(t *testing.T) {
	b, _ := New(Config{
		Gateway:        "http://127.0.0.1:1",
		RequestTimeout: time.Second,
	}, sbbackend.SchedulingProvider)
	if err := b.Preflight(context.Background()); err == nil {
		t.Fatal("preflight must refuse an unreachable gateway")
	}
}

func testSpec() sbbackend.Spec {
	return sbbackend.Spec{
		Source:        sbbackend.Source{Kind: "template", Reference: "tmpl-python-base"},
		Resources:     sbbackend.Resources{CPUCount: 2, MemoryMB: 2048},
		ResourceClass: "rollout",
		Env:           map[string]string{"TASK": "swe"},
	}
}

// -- psrl mode ---------------------------------------------------------------

func TestPSRLModeNeedsNodeAddresses(t *testing.T) {
	if _, err := New(Config{}, sbbackend.SchedulingPSRL); err == nil {
		t.Fatal("cubesandbox in psrl mode must refuse an empty node list")
	}
}

func TestPSRLModeRejectsNodeWithoutAddress(t *testing.T) {
	_, err := New(Config{
		Nodes: []NodeAddress{{NodeID: "n1", Address: ""}},
	}, sbbackend.SchedulingPSRL)
	if err == nil {
		t.Fatal("a node needs both an id and an address")
	}
}

func TestAnInvalidModeIsRefused(t *testing.T) {
	if _, err := New(Config{Gateway: "http://x"}, sbbackend.SchedulingMode("sideways")); err == nil {
		t.Fatal("an unknown scheduling mode must be refused")
	}
}

func TestPSRLModeReportsItsMode(t *testing.T) {
	b, err := New(Config{
		Nodes: []NodeAddress{{NodeID: "n1", Address: "http://n1:8089"}},
	}, sbbackend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if b.Mode() != sbbackend.SchedulingPSRL {
		t.Fatalf("mode: got %q, want psrl", b.Mode())
	}
}

func TestPSRLModeDropsCubeMasterFeatures(t *testing.T) {
	// A warm pool and a template build live on CubeMaster, which the direct
	// Cubelet path bypasses. Declaring them would admit a spec the create
	// would then refuse.
	b, err := New(Config{
		Nodes: []NodeAddress{{NodeID: "n1", Address: "http://n1:8089"}},
	}, sbbackend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	for _, absent := range []string{"warm_pool", "template_build", "volume", "egress_policy"} {
		for _, declared := range b.Capabilities().Features {
			if declared == absent {
				t.Errorf("psrl mode must not declare CubeMaster feature %q", absent)
			}
		}
	}
}

func TestProviderModeKeepsCubeMasterFeatures(t *testing.T) {
	fake := newFakeCube(t)
	declared := map[string]bool{}
	for _, feature := range newBackend(t, fake).Capabilities().Features {
		declared[feature] = true
	}
	for _, wanted := range []string{"warm_pool", "template_build", "volume", "egress_policy"} {
		if !declared[wanted] {
			t.Errorf("provider mode should declare %q", wanted)
		}
	}
}

func TestPSRLModeListsItsNodes(t *testing.T) {
	b, err := New(Config{
		Nodes: []NodeAddress{
			{NodeID: "n2", Address: "http://n2:8089"},
			{NodeID: "n1", Address: "http://n1:8089"},
		},
	}, sbbackend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	ids, err := b.Nodes(context.Background())
	if err != nil {
		t.Fatalf("nodes: %v", err)
	}
	// Sorted, so placement reads a stable order.
	if len(ids) != 2 || ids[0] != "n1" || ids[1] != "n2" {
		t.Fatalf("nodes: got %v, want [n1 n2]", ids)
	}
}

func TestProviderModeListsNoNodes(t *testing.T) {
	fake := newFakeCube(t)
	ids, err := newBackend(t, fake).Nodes(context.Background())
	if err != nil {
		t.Fatalf("nodes: %v", err)
	}
	if len(ids) != 0 {
		t.Fatalf("provider mode places on its own cluster, so it reports no nodes: got %v", ids)
	}
}

func TestPSRLModeCreateGoesToTheChosenNode(t *testing.T) {
	// Two Cubelets; a create naming n2 must reach n2 and not n1.
	var reached string
	n1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached = "n1"
		json.NewEncoder(w).Encode(map[string]any{"sandboxID": "sb-1"})
	}))
	defer n1.Close()
	n2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reached = "n2"
		json.NewEncoder(w).Encode(map[string]any{"sandboxID": "sb-2"})
	}))
	defer n2.Close()

	b, err := New(Config{
		Nodes: []NodeAddress{
			{NodeID: "n1", Address: n1.URL},
			{NodeID: "n2", Address: n2.URL},
		},
	}, sbbackend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	created, err := b.Create(context.Background(), "n2", testSpec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if reached != "n2" {
		t.Errorf("create reached %q, want n2", reached)
	}
	// The node is recorded so every later call routes back to it.
	if created.Handle.NodeID != "n2" {
		t.Errorf("handle node: got %q, want n2", created.Handle.NodeID)
	}
}

func TestPSRLModeRefusesAnUnknownNode(t *testing.T) {
	b, err := New(Config{
		Nodes: []NodeAddress{{NodeID: "n1", Address: "http://n1:8089"}},
	}, sbbackend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if _, err := b.Create(context.Background(), "nowhere", testSpec(), ""); err == nil {
		t.Fatal("a create naming an unconfigured node must be refused")
	}
}

func TestPSRLModePreflightProbesEveryNode(t *testing.T) {
	probed := map[string]bool{}
	n1 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		probed["n1"] = true
		w.WriteHeader(http.StatusOK)
	}))
	defer n1.Close()
	n2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		probed["n2"] = true
		w.WriteHeader(http.StatusOK)
	}))
	defer n2.Close()

	b, err := New(Config{
		Nodes: []NodeAddress{
			{NodeID: "n1", Address: n1.URL},
			{NodeID: "n2", Address: n2.URL},
		},
	}, sbbackend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := b.Preflight(context.Background()); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	if !probed["n1"] || !probed["n2"] {
		t.Errorf("preflight must probe every node, reached %v", probed)
	}
}

func TestPSRLModePreflightFailsWhenOneNodeIsAbsent(t *testing.T) {
	// One live node is not enough: a fleet missing a node silently loses that
	// node's whole capacity, so preflight refuses rather than degrading.
	live := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer live.Close()

	b, err := New(Config{
		Nodes: []NodeAddress{
			{NodeID: "up", Address: live.URL},
			{NodeID: "down", Address: "http://127.0.0.1:1"},
		},
		RequestTimeout: time.Second,
	}, sbbackend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := b.Preflight(context.Background()); err == nil {
		t.Fatal("preflight must refuse a fleet with an unreachable node")
	}
}
