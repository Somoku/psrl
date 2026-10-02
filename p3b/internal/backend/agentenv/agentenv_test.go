package agentenv

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

	"psrl.dev/sandboxd/internal/backend"
)

// A server that answers AgentENV's documented shapes.
//
// A real deployment needs kernel 6.8+ for ublk and /dev/kvm, which this node
// does not have, so the contract under test is the wire shape taken from the
// project's own `src/api/openapi.yml`. What this cannot prove is that AgentENV
// behaves as documented; what it does prove is that the adapter asks for the
// right things and reads the answers correctly, which is the part this
// repository owns.
type fakeAgentENV struct {
	mu        sync.Mutex
	server    *httptest.Server
	sandboxes map[string]string // id -> state
	creates   []coldSandbox
	bindings  []map[string]any
	seq       int
	failNext  int
}

func newFake(t *testing.T) *fakeAgentENV {
	t.Helper()
	fake := &fakeAgentENV{sandboxes: map[string]string{}}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("/sandboxes-cold", func(w http.ResponseWriter, r *http.Request) {
		fake.mu.Lock()
		defer fake.mu.Unlock()
		if fake.failNext > 0 {
			fake.failNext--
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		var body coldSandbox
		_ = json.NewDecoder(r.Body).Decode(&body)
		fake.creates = append(fake.creates, body)
		fake.seq++
		id := fmt.Sprintf("sbx-%d", fake.seq)
		fake.sandboxes[id] = "running"
		w.Header().Set("x-agentenv-sandbox-id", id)
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"sandboxID":       id,
			"clientID":        "client",
			"envdAccessToken": map[string]string{"token": "tok-" + id},
		})
	})
	mux.HandleFunc("/v1/assignments", func(w http.ResponseWriter, r *http.Request) {
		fake.mu.Lock()
		defer fake.mu.Unlock()
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		fake.bindings = append(fake.bindings, body)
		w.WriteHeader(http.StatusOK)
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
			_ = json.NewEncoder(w).Encode(map[string]string{"state": state})
		case action == "pause":
			fake.sandboxes[id] = "paused"
			w.WriteHeader(http.StatusOK)
		case action == "resume":
			fake.sandboxes[id] = "running"
			w.WriteHeader(http.StatusOK)
		case action == "snapshots":
			_ = json.NewEncoder(w).Encode(map[string]string{"snapshotID": "snap-" + id})
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

func (f *fakeAgentENV) lastCreate() coldSandbox {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.creates) == 0 {
		return coldSandbox{}
	}
	return f.creates[len(f.creates)-1]
}

func (f *fakeAgentENV) bindingCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.bindings)
}

func (f *fakeAgentENV) liveCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.sandboxes)
}

func psrlBackend(t *testing.T, fake *fakeAgentENV, scheduler string) *Backend {
	t.Helper()
	b, err := New(Config{
		Nodes:     []NodeAddress{{NodeID: "node-a", Address: fake.server.URL}},
		Scheduler: scheduler,
	}, backend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}
	return b
}

func spec() backend.Spec {
	return backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "registry.example.com/task:v1"},
		Resources:     backend.Resources{CPUCount: 2, MemoryMB: 2048, DiskMB: 8192},
		ResourceClass: "rollout",
		Env:           map[string]string{"TASK": "swe"},
	}
}

func TestPsrlModeNeedsItsNodeAddresses(t *testing.T) {
	// This service chooses the node, so it has to know what the nodes are.
	if _, err := New(Config{Gateway: "http://gw:8080"}, backend.SchedulingPSRL); err == nil {
		t.Fatal("psrl mode without node addresses must be refused")
	}
}

func TestProviderModeNeedsAGateway(t *testing.T) {
	if _, err := New(Config{Nodes: []NodeAddress{{NodeID: "a", Address: "http://a:8000"}}},
		backend.SchedulingProvider); err == nil {
		t.Fatal("provider mode without a gateway must be refused")
	}
}

func TestANodeWithoutAnIdIsRefused(t *testing.T) {
	_, err := New(Config{Nodes: []NodeAddress{{Address: "http://a:8000"}}}, backend.SchedulingPSRL)
	if err == nil {
		t.Fatal("a node this service places against must be identifiable")
	}
}

func TestItDeclaresAFullStateResume(t *testing.T) {
	// This is the reason to run AgentENV: a container backend cannot carry a live
	// process across a node, and a caller must be able to require that.
	fake := newFake(t)
	capabilities := psrlBackend(t, fake, "").Capabilities()

	if capabilities.ResumeLevel != "full_state" {
		t.Fatalf("resume level is %q, want full_state", capabilities.ResumeLevel)
	}
	if !capabilities.Supports("resume_anywhere") || !capabilities.Supports("native_fork") {
		t.Fatalf("capabilities are %v", capabilities.Features)
	}
}

func TestItDeclaresAHibernationRatherThanAFreeze(t *testing.T) {
	// Its pause writes memory out and releases compute. Reporting a freeze would
	// tell a caller the sandbox stayed resident.
	fake := newFake(t)
	capabilities := psrlBackend(t, fake, "").Capabilities()

	if capabilities.Supports("freeze") {
		t.Fatal("agentenv releases compute on pause, so it must not claim a freeze")
	}
	if !capabilities.Supports("hibernate") {
		t.Fatal("agentenv must declare the hibernation it does offer")
	}
}

func TestACreateCarriesTheSpecResources(t *testing.T) {
	fake := newFake(t)
	b := psrlBackend(t, fake, "")

	if _, err := b.Create(context.Background(), "node-a", spec(), ""); err != nil {
		t.Fatalf("create: %v", err)
	}

	created := fake.lastCreate()
	if created.Image != "registry.example.com/task:v1" {
		t.Fatalf("image is %q", created.Image)
	}
	if created.CPUCount != 2 || created.MemoryMB != 2048 || created.DiskSizeMB != 8192 {
		t.Fatalf("resources are cpu=%d mem=%d disk=%d", created.CPUCount, created.MemoryMB, created.DiskSizeMB)
	}
	if created.EnvVars["TASK"] != "swe" {
		t.Fatalf("env is %v", created.EnvVars)
	}
}

func TestACreateDisablesTheBackendOwnExpiry(t *testing.T) {
	// This service owns the lifetime through its reclamation sweep. A second
	// clock would reclaim a sandbox the service still believes it holds.
	fake := newFake(t)
	b := psrlBackend(t, fake, "")

	if _, err := b.Create(context.Background(), "node-a", spec(), ""); err != nil {
		t.Fatalf("create: %v", err)
	}

	if fake.lastCreate().AutoPause {
		t.Fatal("agentenv's own auto-pause must be off, or two clocks reclaim one sandbox")
	}
}

func TestTheCreatedSandboxCarriesItsOwnAgentEndpoint(t *testing.T) {
	// This is what keeps command traffic off the control plane: the SDK gets an
	// address and an access token with the handle.
	fake := newFake(t)
	b := psrlBackend(t, fake, "")

	created, err := b.Create(context.Background(), "node-a", spec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	if created.Agent.Address == "" {
		t.Fatal("a provider sandbox must report where its agent is")
	}
	if created.Agent.Headers["X-Access-Token"] == "" {
		t.Fatalf("the agent endpoint must carry the token it requires: %v", created.Agent.Headers)
	}
	if !strings.Contains(created.Agent.Address, created.Handle.SandboxID) {
		t.Fatalf("the agent address %q does not name the sandbox", created.Agent.Address)
	}
}

func TestPlacingServiceRegistersTheBindingSoProviderRoutingKeepsWorking(t *testing.T) {
	fake := newFake(t)
	b := psrlBackend(t, fake, fake.server.URL)

	if _, err := b.Create(context.Background(), "node-a", spec(), ""); err != nil {
		t.Fatalf("create: %v", err)
	}

	if fake.bindingCount() != 1 {
		t.Fatalf("recorded %d bindings, want 1", fake.bindingCount())
	}
}

func TestADeploymentWithNoSchedulerRegistersNothing(t *testing.T) {
	fake := newFake(t)
	b := psrlBackend(t, fake, "")

	if _, err := b.Create(context.Background(), "node-a", spec(), ""); err != nil {
		t.Fatalf("create: %v", err)
	}

	if fake.bindingCount() != 0 {
		t.Fatal("a deployment running no scheduler must not be told about bindings")
	}
}

func TestAFailedRegistrationDoesNotFailTheCreate(t *testing.T) {
	// The sandbox exists and is reachable through this service either way; only
	// AgentENV's own lookup is degraded.
	fake := newFake(t)
	b, err := New(Config{
		Nodes:     []NodeAddress{{NodeID: "node-a", Address: fake.server.URL}},
		Scheduler: "http://127.0.0.1:1",
	}, backend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}

	if _, err := b.Create(context.Background(), "node-a", spec(), ""); err != nil {
		t.Fatalf("a dead scheduler must not fail a create: %v", err)
	}
}

func TestBackendOptionsReachTheProvider(t *testing.T) {
	// A provider's own tuning stays reachable without making the portable spec
	// unportable.
	fake := newFake(t)
	b := psrlBackend(t, fake, "")
	tuned := spec()
	tuned.BackendOptions = map[string]map[string]string{"agentenv": {"secure": "true"}}

	if _, err := b.Create(context.Background(), "node-a", tuned, ""); err != nil {
		t.Fatalf("create: %v", err)
	}

	if !fake.lastCreate().Secure {
		t.Fatal("the backend's own option did not reach it")
	}
}

func TestAnotherBackendOptionsAreIgnoredHere(t *testing.T) {
	// Namespaced, so a key meant for a different backend is not applied here.
	fake := newFake(t)
	b := psrlBackend(t, fake, "")
	tuned := spec()
	tuned.BackendOptions = map[string]map[string]string{"cubesandbox": {"secure": "true"}}

	if _, err := b.Create(context.Background(), "node-a", tuned, ""); err != nil {
		t.Fatalf("create: %v", err)
	}

	if fake.lastCreate().Secure {
		t.Fatal("another backend's option was applied here")
	}
}

func TestACreateOnAnUnknownNodeIsRefused(t *testing.T) {
	fake := newFake(t)
	b := psrlBackend(t, fake, "")

	if _, err := b.Create(context.Background(), "node-z", spec(), ""); err == nil {
		t.Fatal("placing onto a node this backend does not know must be refused")
	}
}

func TestLifecycleRunsThroughPauseResumeAndRelease(t *testing.T) {
	fake := newFake(t)
	b := psrlBackend(t, fake, "")
	ctx := context.Background()
	created, err := b.Create(ctx, "node-a", spec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	if err := b.Pause(ctx, created.Handle, "hibernate"); err != nil {
		t.Fatalf("pause: %v", err)
	}
	if state, _ := b.Status(ctx, created.Handle); state != "paused" {
		t.Fatalf("state is %q, want paused", state)
	}
	if err := b.Resume(ctx, created.Handle); err != nil {
		t.Fatalf("resume: %v", err)
	}
	if state, _ := b.Status(ctx, created.Handle); state != "running" {
		t.Fatalf("state is %q, want running", state)
	}
	if err := b.Release(ctx, created.Handle); err != nil {
		t.Fatalf("release: %v", err)
	}
	if fake.liveCount() != 0 {
		t.Fatalf("%d sandboxes survived the release", fake.liveCount())
	}
}

func TestAFreezeIsRefusedBecauseThePauseReleasesCompute(t *testing.T) {
	fake := newFake(t)
	b := psrlBackend(t, fake, "")
	created, _ := b.Create(context.Background(), "node-a", spec(), "")

	if err := b.Pause(context.Background(), created.Handle, "freeze"); err == nil {
		t.Fatal("a freeze must be refused rather than served as a hibernation")
	}
}

func TestAFilesystemSnapshotIsRefusedBecauseTheCaptureIncludesMemory(t *testing.T) {
	// Reporting it as a filesystem capture would understate what was taken, and
	// a caller comparing resume levels would then make the wrong choice.
	fake := newFake(t)
	b := psrlBackend(t, fake, "")
	created, _ := b.Create(context.Background(), "node-a", spec(), "")

	if _, err := b.Snapshot(context.Background(), created.Handle, "filesystem"); err == nil {
		t.Fatal("agentenv captures memory too, so a filesystem-only claim must be refused")
	}
}

func TestASnapshotReturnsARestorableId(t *testing.T) {
	fake := newFake(t)
	b := psrlBackend(t, fake, "")
	created, _ := b.Create(context.Background(), "node-a", spec(), "")

	id, err := b.Snapshot(context.Background(), created.Handle, "full_state")
	if err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	if id == "" {
		t.Fatal("a snapshot must return an id a restore can name")
	}
	if err := b.DeleteSnapshot(context.Background(), id); err != nil {
		t.Fatalf("delete snapshot: %v", err)
	}
}

func TestReleasingAGoneSandboxIsANoOp(t *testing.T) {
	fake := newFake(t)
	b := psrlBackend(t, fake, "")
	created, _ := b.Create(context.Background(), "node-a", spec(), "")
	if err := b.Release(context.Background(), created.Handle); err != nil {
		t.Fatalf("first release: %v", err)
	}

	if err := b.Release(context.Background(), created.Handle); err != nil {
		t.Fatalf("a retried release must be safe: %v", err)
	}
}

func TestAGoneSandboxReportsTerminatedRatherThanAnError(t *testing.T) {
	fake := newFake(t)
	b := psrlBackend(t, fake, "")
	created, _ := b.Create(context.Background(), "node-a", spec(), "")
	_ = b.Release(context.Background(), created.Handle)

	state, err := b.Status(context.Background(), created.Handle)

	if err != nil {
		t.Fatalf("a gone sandbox is a state, not an error: %v", err)
	}
	if state != "terminated" {
		t.Fatalf("state is %q, want terminated", state)
	}
}

func TestPreflightRefusesADeploymentThatIsNotAnswering(t *testing.T) {
	b, err := New(Config{
		Nodes:          []NodeAddress{{NodeID: "node-a", Address: "http://127.0.0.1:1"}},
		RequestTimeout: time.Second,
	}, backend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}

	if err := b.Preflight(context.Background()); err == nil {
		t.Fatal("preflight must refuse a node that cannot be reached")
	}
}

func TestPreflightAcceptsALiveDeployment(t *testing.T) {
	fake := newFake(t)

	if err := psrlBackend(t, fake, "").Preflight(context.Background()); err != nil {
		t.Fatalf("preflight: %v", err)
	}
}

func TestProviderModeSendsEverythingToTheGateway(t *testing.T) {
	// In provider mode AgentENV places, so this adapter must not name a node.
	fake := newFake(t)
	b, err := New(Config{Gateway: fake.server.URL}, backend.SchedulingProvider)
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}

	created, err := b.Create(context.Background(), "", spec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := b.Release(context.Background(), created.Handle); err != nil {
		t.Fatalf("release: %v", err)
	}
}

func TestConcurrentCreatesAndReleasesAllSucceed(t *testing.T) {
	fake := newFake(t)
	b := psrlBackend(t, fake, fake.server.URL)
	ctx := context.Background()

	var wg sync.WaitGroup
	errs := make(chan error, 32)
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			created, err := b.Create(ctx, "node-a", spec(), "")
			if err != nil {
				errs <- err
				return
			}
			if err := b.Release(ctx, created.Handle); err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent lifecycle: %v", err)
	}

	if fake.liveCount() != 0 {
		t.Fatalf("%d sandboxes survived the burst", fake.liveCount())
	}
}

func TestTheAdapterSatisfiesEveryContractItClaims(t *testing.T) {
	fake := newFake(t)
	b := psrlBackend(t, fake, "")

	var (
		_ backend.Backend       = b
		_ backend.NodeScheduled = b
		_ backend.Stateful      = b
		_ backend.Preflighter   = b
	)
}
