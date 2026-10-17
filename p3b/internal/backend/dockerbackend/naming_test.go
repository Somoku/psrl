package dockerbackend

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"psrl.dev/sandboxd/internal/backend"
)

// probeEngine serves the endpoints a create and a preflight touch, and records what a
// create was asked for. It can refuse creates with a named status, which is how a name
// conflict and an ordinary refusal are told apart without a daemon.
type probeEngine struct {
	mu            sync.Mutex
	names         []string
	bodies        []map[string]any
	conflictsLeft int
	createStatus  int
	cgroupDriver  string
	infoCalls     int
}

func (e *probeEngine) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch {
	case strings.HasSuffix(r.URL.Path, "/version"):
		writeBody(w, `{"ApiVersion":"1.45","MinAPIVersion":"1.24","Version":"28.0"}`)
	case strings.HasSuffix(r.URL.Path, "/info"):
		e.mu.Lock()
		e.infoCalls++
		driver := e.cgroupDriver
		e.mu.Unlock()
		writeBody(w, `{"CgroupDriver":"`+driver+`","Runtimes":{"runc":{"path":"runc"}}}`)
	case strings.Contains(r.URL.Path, "/images/"):
		writeBody(w, `{}`)
	case strings.HasSuffix(r.URL.Path, "/containers/create"):
		e.mu.Lock()
		e.names = append(e.names, r.URL.Query().Get("name"))
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		e.bodies = append(e.bodies, body)
		conflict := e.conflictsLeft > 0
		if conflict {
			e.conflictsLeft--
		}
		status := e.createStatus
		e.mu.Unlock()
		if conflict {
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(`{"message":"Conflict. The container name is already in use"}`))
			return
		}
		if status >= 400 {
			w.WriteHeader(status)
			_, _ = w.Write([]byte(`{"message":"the daemon refused"}`))
			return
		}
		writeBody(w, `{"Id":"c-1"}`)
	case strings.HasSuffix(r.URL.Path, "/start"):
		w.WriteHeader(http.StatusOK)
	case strings.Contains(r.URL.Path, "/archive"):
		w.WriteHeader(http.StatusOK)
	case strings.HasSuffix(r.URL.Path, "/exec"):
		writeBody(w, `{"Id":"exec-1"}`)
	case strings.Contains(r.URL.Path, "/exec/exec-1/start"):
		w.WriteHeader(http.StatusOK)
	case strings.Contains(r.URL.Path, "/exec/exec-1/json"):
		writeBody(w, `{"ExitCode":0,"Running":false}`)
	default:
		http.NotFound(w, r)
	}
}

// socket starts the engine on a Unix socket and returns its path.
func (e *probeEngine) socket(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "docker.sock")
	listener, err := net.Listen("unix", path)
	if err != nil {
		t.Fatalf("listen on %s: %v", path, err)
	}
	server := &http.Server{Handler: e}
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(func() { _ = server.Close() })
	return path
}

func (e *probeEngine) createdNames() []string {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]string(nil), e.names...)
}

func (e *probeEngine) createdBodies() []map[string]any {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]map[string]any(nil), e.bodies...)
}

func probeBackend(t *testing.T, engine *probeEngine, parent string) *Backend {
	t.Helper()
	b, err := New(Config{
		Socket:       engine.socket(t),
		APIVersion:   "v1.40",
		CgroupParent: parent,
	}, backend.SchedulingDirect)
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}
	return b
}

func imageSpec() backend.Spec {
	return backend.Spec{
		Source:    backend.Source{Kind: "image", Reference: "img"},
		Resources: backend.Resources{MemoryMB: 256, CPUCount: 1},
	}
}

func TestASandboxIsCreatedUnderAChosenName(t *testing.T) {
	// The name is how pool inventory is told from a claimed sandbox, so a daemon-chosen
	// name would leave the two indistinguishable and the pool's restart behaviour
	// undefined.
	engine := &probeEngine{}
	b := probeBackend(t, engine, "")

	if _, err := b.Create(context.Background(), "node-a", imageSpec(), ""); err != nil {
		t.Fatalf("Create: %v", err)
	}

	names := engine.createdNames()
	if len(names) != 1 {
		t.Fatalf("creates: got %d, want 1", len(names))
	}
	if !strings.HasPrefix(names[0], sandboxNamePrefix) {
		t.Errorf("name %q does not start with %q", names[0], sandboxNamePrefix)
	}
}

func TestACreateRetriesANameConflict(t *testing.T) {
	engine := &probeEngine{conflictsLeft: 1}
	b := probeBackend(t, engine, "")

	if _, err := b.Create(context.Background(), "node-a", imageSpec(), ""); err != nil {
		t.Fatalf("a name collision must be retried, not surfaced: %v", err)
	}

	names := engine.createdNames()
	if len(names) != 2 {
		t.Fatalf("creates: got %d, want 2 (one conflict, one retry)", len(names))
	}
	if names[0] == names[1] {
		t.Errorf("the retry reused the conflicting name %q", names[0])
	}
}

func TestANameConflictThatKeepsHappeningIsAnError(t *testing.T) {
	engine := &probeEngine{conflictsLeft: 99}
	b := probeBackend(t, engine, "")

	if _, err := b.Create(context.Background(), "node-a", imageSpec(), ""); err == nil {
		t.Fatal("a name that always conflicts must fail the create")
	}
	if got := len(engine.createdNames()); got != 3 {
		t.Errorf("attempts: got %d, want 3", got)
	}
}

func TestARefusalThatIsNotAConflictIsNotRetried(t *testing.T) {
	// Only a name collision is worth another attempt. A daemon that refused a create
	// for any other reason will refuse it again, and retrying hides the reason.
	engine := &probeEngine{createStatus: http.StatusBadRequest}
	b := probeBackend(t, engine, "")

	if _, err := b.Create(context.Background(), "node-a", imageSpec(), ""); err == nil {
		t.Fatal("a refused create must fail")
	}
	if got := len(engine.createdNames()); got != 1 {
		t.Errorf("attempts: got %d, want 1", got)
	}
}

func TestASandboxIsCreatedInTheConfiguredSubtree(t *testing.T) {
	engine := &probeEngine{}
	b := probeBackend(t, engine, "psrl-sandboxes")

	if _, err := b.Create(context.Background(), "node-a", imageSpec(), ""); err != nil {
		t.Fatalf("Create: %v", err)
	}

	bodies := engine.createdBodies()
	if len(bodies) != 1 {
		t.Fatalf("creates: got %d, want 1", len(bodies))
	}
	hostConfig, ok := bodies[0]["HostConfig"].(map[string]any)
	if !ok {
		t.Fatalf("HostConfig is %T", bodies[0]["HostConfig"])
	}
	if got := hostConfig["CgroupParent"]; got != "psrl-sandboxes" {
		t.Errorf("CgroupParent: got %v, want the configured subtree", got)
	}
}

func TestNoSubtreeIsWrittenWhenNoneIsConfigured(t *testing.T) {
	// A deployment that has not adopted the subtree must keep the daemon's default
	// placement rather than an empty parent that means "the root".
	engine := &probeEngine{}
	b := probeBackend(t, engine, "")

	if _, err := b.Create(context.Background(), "node-a", imageSpec(), ""); err != nil {
		t.Fatalf("Create: %v", err)
	}

	bodies := engine.createdBodies()
	hostConfig := bodies[0]["HostConfig"].(map[string]any)
	if _, present := hostConfig["CgroupParent"]; present {
		t.Error("CgroupParent must be absent when the backend has no subtree")
	}
}

func TestPreflightRefusesACgroupDriverThatWouldIgnoreTheSubtree(t *testing.T) {
	// The systemd driver reads CgroupParent as a slice name, so the container would be
	// placed somewhere the sampler is not reading.
	engine := &probeEngine{cgroupDriver: "systemd"}
	b := probeBackend(t, engine, "psrl-sandboxes")

	err := b.Preflight(context.Background())
	if err == nil {
		t.Fatal("a daemon whose driver would ignore the subtree must be refused")
	}
	if !strings.Contains(err.Error(), "cgroup driver") {
		t.Errorf("the refusal must name the driver, got %v", err)
	}
}

func TestPreflightAcceptsTheCgroupfsDriver(t *testing.T) {
	engine := &probeEngine{cgroupDriver: "cgroupfs"}
	b := probeBackend(t, engine, "psrl-sandboxes")

	if err := b.Preflight(context.Background()); err != nil {
		t.Fatalf("Preflight: %v", err)
	}
}

func TestPreflightDoesNotAskAboutTheDriverWithoutASubtree(t *testing.T) {
	engine := &probeEngine{}
	b := probeBackend(t, engine, "")

	if err := b.Preflight(context.Background()); err != nil {
		t.Fatalf("Preflight: %v", err)
	}
	if engine.infoCalls != 0 {
		t.Errorf("info calls: got %d, want 0 (no subtree to place containers in)", engine.infoCalls)
	}
}
