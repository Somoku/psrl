package opensandbox

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

// A fake OpenSandbox HTTP server matching the SDK's documented wire shapes.
type fakeOpen struct {
	mu        sync.Mutex
	server    *httptest.Server
	sandboxes map[string]string // id -> state
	seq       int
}

func newFakeOpen(t *testing.T) *fakeOpen {
	t.Helper()
	fake := &fakeOpen{sandboxes: map[string]string{}}
	mux := http.NewServeMux()
	mux.HandleFunc("/sandboxes", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			// List: minimal response for preflight check.
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"items":      []any{},
				"pagination": map[string]any{"page": 1},
			})
			return
		}
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		fake.mu.Lock()
		defer fake.mu.Unlock()
		fake.seq++
		id := fmt.Sprintf("osb-%d", fake.seq)
		fake.sandboxes[id] = "Running"
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(sandboxInfo{
			ID:     id,
			Status: struct{ State string `json:"state"` }{State: "Running"},
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
			_ = json.NewEncoder(w).Encode(sandboxInfo{
				ID:     id,
				Status: struct{ State string `json:"state"` }{State: state},
			})
		case action == "pause":
			fake.sandboxes[id] = "Paused"
			w.WriteHeader(http.StatusNoContent)
		case action == "resume":
			fake.sandboxes[id] = "Running"
			w.WriteHeader(http.StatusOK)
		case action == "snapshots":
			_ = json.NewEncoder(w).Encode(snapshotInfo{ID: "snap-" + id})
		case strings.HasPrefix(action, "endpoints/"):
			_ = json.NewEncoder(w).Encode(map[string]any{
				"endpoint": "https://osb.example.com/" + id,
				"headers":  map[string]string{"Authorization": "Bearer tok-" + id},
			})
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

func newOpenBackend(t *testing.T, fake *fakeOpen) *Backend {
	t.Helper()
	b, err := New(Config{Gateway: fake.server.URL})
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}
	return b
}

func openSpec() sbbackend.Spec {
	return sbbackend.Spec{
		Source:        sbbackend.Source{Kind: "image", Reference: "python:3.11-slim"},
		Resources:     sbbackend.Resources{CPUCount: 2, MemoryMB: 2048},
		ResourceClass: "rollout",
		Env:           map[string]string{"TASK": "swe"},
	}
}

func TestItNeedsAGateway(t *testing.T) {
	if _, err := New(Config{}); err == nil {
		t.Fatal("opensandbox must refuse an empty gateway")
	}
}

func TestItDeclaresAFreeze(t *testing.T) {
	fake := newFakeOpen(t)
	caps := newOpenBackend(t, fake).Capabilities()

	if !caps.Supports("freeze") {
		t.Fatal("opensandbox must declare freeze")
	}
	for _, mode := range caps.PauseModes {
		if mode == "hibernate" {
			t.Fatal("opensandbox keeps the sandbox resident on pause, so it must not claim hibernate")
		}
	}
}

func TestItDeclaresImageBlockDelivery(t *testing.T) {
	// OpenSandbox's block-level image delivery is a real differentiator from
	// plain layer pulling; callers that need it must be able to require it.
	fake := newFakeOpen(t)
	caps := newOpenBackend(t, fake).Capabilities()

	if !caps.Supports("image_block_delivery") {
		t.Fatal("opensandbox must declare image_block_delivery")
	}
}

func TestItDeclaresNoResumeLevel(t *testing.T) {
	// Filesystem snapshots do not carry live process state.
	fake := newFakeOpen(t)
	if newOpenBackend(t, fake).Capabilities().ResumeLevel != "" {
		t.Fatal("opensandbox captures filesystem only, not live process state")
	}
}

func TestResourcesAreConvertedToKubernetesQuantities(t *testing.T) {
	// 2 CPUs → "2000m", 2048 MB → "2048Mi"; 0 GPU → absent.
	limits := resourceLimits(sbbackend.Spec{
		Resources: sbbackend.Resources{CPUCount: 2, MemoryMB: 2048},
	})
	if limits["cpu"] != "2000m" {
		t.Fatalf("cpu limit is %q, want \"2000m\"", limits["cpu"])
	}
	if limits["memory"] != "2048Mi" {
		t.Fatalf("memory limit is %q, want \"2048Mi\"", limits["memory"])
	}
	if _, has := limits["gpu"]; has {
		t.Fatal("a spec with no GPU must not send a gpu limit")
	}
}

func TestACreateReturnsASandboxAndAnAgentEndpoint(t *testing.T) {
	fake := newFakeOpen(t)
	b := newOpenBackend(t, fake)

	created, err := b.Create(context.Background(), "", openSpec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if created.Handle.SandboxID == "" {
		t.Fatal("create must return a sandbox id")
	}
	if created.Agent.Address == "" {
		t.Fatal("create must return an agent endpoint")
	}
}

func TestLifecycleRunsThroughPauseResumeAndRelease(t *testing.T) {
	fake := newFakeOpen(t)
	b := newOpenBackend(t, fake)
	ctx := context.Background()

	created, err := b.Create(ctx, "", openSpec(), "")
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

func TestHibernateIsRefused(t *testing.T) {
	fake := newFakeOpen(t)
	b := newOpenBackend(t, fake)
	created, _ := b.Create(context.Background(), "", openSpec(), "")

	if err := b.Pause(context.Background(), created.Handle, "hibernate"); err == nil {
		t.Fatal("opensandbox must refuse hibernate")
	}
}

func TestFullStateSnapshotIsRefused(t *testing.T) {
	fake := newFakeOpen(t)
	b := newOpenBackend(t, fake)
	created, _ := b.Create(context.Background(), "", openSpec(), "")

	if _, err := b.Snapshot(context.Background(), created.Handle, "full_state"); err == nil {
		t.Fatal("opensandbox captures filesystem only, so full_state must be refused")
	}
}

func TestSnapshotReturnsAnId(t *testing.T) {
	fake := newFakeOpen(t)
	b := newOpenBackend(t, fake)
	created, _ := b.Create(context.Background(), "", openSpec(), "")

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

func TestReleasingAGoneSandboxIsIdempotent(t *testing.T) {
	fake := newFakeOpen(t)
	b := newOpenBackend(t, fake)
	created, _ := b.Create(context.Background(), "", openSpec(), "")
	_ = b.Release(context.Background(), created.Handle)

	if err := b.Release(context.Background(), created.Handle); err != nil {
		t.Fatalf("retried release must be safe: %v", err)
	}
}

func TestAGoneSandboxReportsTerminated(t *testing.T) {
	fake := newFakeOpen(t)
	b := newOpenBackend(t, fake)
	created, _ := b.Create(context.Background(), "", openSpec(), "")
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
	fake := newFakeOpen(t)
	if err := newOpenBackend(t, fake).Preflight(context.Background()); err != nil {
		t.Fatalf("preflight must pass against a live gateway: %v", err)
	}
}

func TestPreflightFailsWhenGatewayIsAbsent(t *testing.T) {
	b, _ := New(Config{
		Gateway:        "http://127.0.0.1:1",
		RequestTimeout: time.Second,
	})
	if err := b.Preflight(context.Background()); err == nil {
		t.Fatal("preflight must refuse an unreachable gateway")
	}
}
