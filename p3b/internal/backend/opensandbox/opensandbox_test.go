package opensandbox_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/backend/opensandbox"
)

// What is worth asserting here is the boundary this adapter draws: that a mode is
// refused unless it has what it needs to run, that each mode declares only what it
// can serve, and that a spec the runtime cannot honour is refused rather than
// quietly downgraded.
//
// Direct mode's container composition and agent staging need a live Docker daemon,
// so they belong to the integration suite. What this file covers of direct mode is
// its refusals, which are what a misconfiguration hits first.

// -- construction: each mode needs what it needs ------------------------------

func TestProviderModeNeedsAGateway(t *testing.T) {
	if _, err := opensandbox.New(opensandbox.Config{}, backend.SchedulingProvider); err == nil {
		t.Fatal("provider mode must refuse an empty gateway: there is nothing to call")
	}
}

func TestDirectModeNeedsAnAgentImage(t *testing.T) {
	// Without the agent image there is nothing to stage into a sandbox, so the
	// sandbox would start with no way to run a command.
	_, err := opensandbox.New(opensandbox.Config{
		Socket: "unix:///var/run/docker.sock",
		NodeID: "node-1",
	}, backend.SchedulingDirect)
	if err == nil {
		t.Fatal("direct mode must refuse a missing execd_image")
	}
	if !strings.Contains(err.Error(), "execd_image") {
		t.Errorf("the error should name execd_image, got: %v", err)
	}
}

func TestDirectModeNeedsASocket(t *testing.T) {
	_, err := opensandbox.New(opensandbox.Config{
		ExecdImage: "opensandbox/execd:latest",
		NodeID:     "node-1",
	}, backend.SchedulingDirect)
	if err == nil {
		t.Fatal("direct mode must refuse a missing socket: it drives the runtime itself")
	}
	if !strings.Contains(err.Error(), "socket") {
		t.Errorf("the error should name socket, got: %v", err)
	}
}

func TestDirectModeNeedsANodeID(t *testing.T) {
	// The node is recorded on every handle, and a restore has to route back to the
	// node holding the snapshot.
	_, err := opensandbox.New(opensandbox.Config{
		ExecdImage: "opensandbox/execd:latest",
		Socket:     "unix:///var/run/docker.sock",
	}, backend.SchedulingDirect)
	if err == nil {
		t.Fatal("direct mode must refuse a missing node id")
	}
	if !strings.Contains(err.Error(), "node id") {
		t.Errorf("the error should name the node id, got: %v", err)
	}
}

func TestAnUnknownModeIsRefused(t *testing.T) {
	_, err := opensandbox.New(opensandbox.Config{Gateway: "http://gateway"},
		backend.SchedulingMode("sideways"))
	if err == nil {
		t.Fatal("a mode that is neither direct nor provider must be refused")
	}
}

func TestDirectModeRefusesAnUnreachableDaemon(t *testing.T) {
	// The daemon is dialled in New rather than on the first create, so a
	// deployment pointing at nothing fails at startup instead of inside an episode.
	_, err := opensandbox.New(opensandbox.Config{
		ExecdImage: "opensandbox/execd:latest",
		Socket:     "tcp://127.0.0.1:1",
		NodeID:     "node-1",
		StageDir:   t.TempDir(),
	}, backend.SchedulingDirect)
	if err == nil {
		t.Fatal("direct mode must refuse a daemon that is not answering")
	}
}

// -- provider mode -------------------------------------------------------------

func TestProviderModeReportsItsMode(t *testing.T) {
	if mode := providerBackend(t, "https://gateway.example.com/v1").Mode(); mode != backend.SchedulingProvider {
		t.Fatalf("mode: got %q, want provider", mode)
	}
}

func TestProviderModePlacesOnItsOwnCluster(t *testing.T) {
	// Provider mode reports no nodes, which is how the service knows not to charge
	// this machine's envelope for a sandbox running elsewhere.
	ids, err := providerBackend(t, "https://gateway.example.com/v1").Nodes(context.Background())
	if err != nil {
		t.Fatalf("nodes: %v", err)
	}
	if len(ids) != 0 {
		t.Fatalf("provider mode must report no nodes, got %v", ids)
	}
}

func TestProviderModeDeclaresWhatOnlyAClusterHas(t *testing.T) {
	declared := featureSet(providerBackend(t, "https://gateway.example.com/v1"))
	for _, wanted := range []string{
		"warm_pool", "template_build", "image_block_delivery", "resume_anywhere", "volume",
	} {
		if !declared[wanted] {
			t.Errorf("provider mode should declare the cluster feature %q", wanted)
		}
	}
}

func TestProviderModeCreateReachesTheGateway(t *testing.T) {
	var firstPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if firstPath == "" {
			firstPath = r.URL.Path
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "sb-1", "status": map[string]string{"state": "Running"},
		})
	}))
	defer srv.Close()

	created, err := providerBackend(t, srv.URL).Create(context.Background(), "", testSpec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if created.Handle.SandboxID != "sb-1" {
		t.Errorf("sandbox id: got %q, want sb-1", created.Handle.SandboxID)
	}
	if firstPath != "/sandboxes" {
		t.Errorf("create went to %q, want /sandboxes", firstPath)
	}
}

func TestProviderModeSendsKubernetesQuantities(t *testing.T) {
	// The cluster API rejects raw integers, so cpu must be millicores and memory a
	// binary-suffix string.
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			_ = json.NewDecoder(r.Body).Decode(&body)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"id": "sb-1"})
	}))
	defer srv.Close()

	if _, err := providerBackend(t, srv.URL).Create(context.Background(), "", testSpec(), ""); err != nil {
		t.Fatalf("create: %v", err)
	}
	limits, _ := body["resourceLimits"].(map[string]any)
	if limits["cpu"] != "2000m" {
		t.Errorf("cpu: got %v, want 2000m", limits["cpu"])
	}
	if limits["memory"] != "512Mi" {
		t.Errorf("memory: got %v, want 512Mi", limits["memory"])
	}
}

func TestProviderModePreflightRefusesASilentGateway(t *testing.T) {
	if err := providerBackend(t, "http://127.0.0.1:1").Preflight(context.Background()); err == nil {
		t.Fatal("preflight must refuse a gateway that is not answering")
	}
}

func TestProviderModePreflightProvesAuthNotJustReachability(t *testing.T) {
	// A listening port says nothing about whether the key is accepted, so the probe
	// is a real list rather than a health check.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
	}))
	defer srv.Close()
	if err := providerBackend(t, srv.URL).Preflight(context.Background()); err == nil {
		t.Fatal("preflight must refuse a gateway that rejects the key")
	}
}

// -- refusals that hold in both modes ------------------------------------------

func TestAFullStateSnapshotIsRefused(t *testing.T) {
	// Neither mode captures memory. Serving full_state as a filesystem snapshot
	// would let a conformance run read a workspace restore as proof that a live
	// process survived a move.
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer srv.Close()
	handle := backend.Handle{Backend: "opensandbox", SandboxID: "sb-1"}
	if _, err := providerBackend(t, srv.URL).Snapshot(context.Background(), handle, "full_state"); err == nil {
		t.Fatal("a full_state snapshot must be refused")
	}
}

func TestAHibernateIsRefused(t *testing.T) {
	// The sandbox stays resident on pause in both modes, so a caller asking for its
	// compute back is told no rather than served a freeze.
	handle := backend.Handle{Backend: "opensandbox", SandboxID: "sb-1"}
	b := providerBackend(t, "https://gateway.example.com/v1")
	if err := b.Pause(context.Background(), handle, "hibernate"); err == nil {
		t.Fatal("a hibernate must be refused: pause keeps the sandbox resident")
	}
}

func TestAFreezeIsAccepted(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	handle := backend.Handle{Backend: "opensandbox", SandboxID: "sb-1"}
	if err := providerBackend(t, srv.URL).Pause(context.Background(), handle, "freeze"); err != nil {
		t.Fatalf("a freeze is the mode this backend serves: %v", err)
	}
}

func TestAnEmptyPauseModeTakesTheDefault(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	handle := backend.Handle{Backend: "opensandbox", SandboxID: "sb-1"}
	if err := providerBackend(t, srv.URL).Pause(context.Background(), handle, ""); err != nil {
		t.Fatalf("an unstated pause mode takes the backend's own: %v", err)
	}
}

func TestAnUnknownSourceKindIsRefused(t *testing.T) {
	spec := testSpec()
	spec.Source.Kind = "snapshot-stream"
	b := providerBackend(t, "https://gateway.example.com/v1")
	if _, err := b.Create(context.Background(), "", spec, ""); err == nil {
		t.Fatal("a source kind this backend cannot run must be refused")
	}
}

func TestASourceWithNoReferenceIsRefused(t *testing.T) {
	spec := testSpec()
	spec.Source.Reference = ""
	b := providerBackend(t, "https://gateway.example.com/v1")
	if _, err := b.Create(context.Background(), "", spec, ""); err == nil {
		t.Fatal("a source with no reference must be refused")
	}
}

// -- reporting -----------------------------------------------------------------

func TestProviderModeStagesNoAgent(t *testing.T) {
	// There is no staged agent to report in provider mode: the cluster provides its
	// own, and reporting a digest here would invent one.
	if digest := providerBackend(t, "https://gateway.example.com/v1").StagedAgentDigest(); digest != "" {
		t.Errorf("provider mode stages no agent, got digest %q", digest)
	}
}

func TestTheRegistryKeyIsStable(t *testing.T) {
	// The name is how a spec pins this backend, so it is part of the contract.
	if name := providerBackend(t, "https://gateway.example.com/v1").Name(); name != "opensandbox" {
		t.Errorf("name: got %q, want opensandbox", name)
	}
}

// -- helpers -------------------------------------------------------------------

func providerBackend(t *testing.T, gateway string) *opensandbox.Backend {
	t.Helper()
	b, err := opensandbox.New(opensandbox.Config{Gateway: gateway}, backend.SchedulingProvider)
	if err != nil {
		t.Fatalf("opensandbox.New: %v", err)
	}
	return b
}

func featureSet(b *opensandbox.Backend) map[string]bool {
	declared := map[string]bool{}
	for _, feature := range b.Capabilities().Features {
		declared[feature] = true
	}
	return declared
}

func testSpec() backend.Spec {
	return backend.Spec{
		Source:    backend.Source{Kind: "image", Reference: "alpine:latest"},
		Resources: backend.Resources{CPUCount: 2, MemoryMB: 512},
	}
}
