package dockerbackend

import (
	"context"
	"net"
	"net/http"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// fakeEngine serves the slice of the Engine API these tests drive, so a backend can
// be exercised without a daemon. A delay stands in for a daemon that is too busy to
// answer promptly, which is the shape the real faults had.
type fakeEngine struct {
	execStartDelay time.Duration
	createDelay    time.Duration

	// Optional canned failure responses, so a test can state the Engine's own
	// wording for a stopped container or a missing path. Zero means the happy
	// path the other tests use.
	execCreateStatus int
	execCreateBody   string
	execStartStatus  int
	execStartBody    string
	// execStartOutput is a 200 response's hijacked stream, where runc reports a
	// container it could not enter. execInspectBody is the exec's final state.
	execStartOutput []byte
	execInspectBody string
	archiveStatus   int
	archiveBody     string
	inspectBody     string

	mu          sync.Mutex
	inFlight    int
	maxInFlight int
}

func (f *fakeEngine) socket(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "docker.sock")
	listener, err := net.Listen("unix", path)
	if err != nil {
		t.Fatalf("listen on %s: %v", path, err)
	}
	server := &http.Server{Handler: f}
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(func() { _ = server.Close() })
	return path
}

// peakInFlight is how many create+start pairs this engine saw at once, which is the
// quantity the create gate bounds.
func (f *fakeEngine) peakInFlight() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.maxInFlight
}

func (f *fakeEngine) enterCreate() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.inFlight++
	if f.inFlight > f.maxInFlight {
		f.maxInFlight = f.inFlight
	}
}

func (f *fakeEngine) leaveCreate() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.inFlight--
}

func (f *fakeEngine) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch {
	case strings.HasSuffix(r.URL.Path, "/exec"):
		if f.execCreateStatus != 0 {
			w.WriteHeader(f.execCreateStatus)
			writeBody(w, f.execCreateBody)
			return
		}
		writeBody(w, `{"Id":"exec-1"}`)
	case strings.Contains(r.URL.Path, "/exec/exec-1/start"):
		time.Sleep(f.execStartDelay)
		if f.execStartStatus != 0 {
			w.WriteHeader(f.execStartStatus)
			writeBody(w, f.execStartBody)
			return
		}
		w.WriteHeader(http.StatusOK)
		if len(f.execStartOutput) > 0 {
			_, _ = w.Write(f.execStartOutput)
		}
	case strings.Contains(r.URL.Path, "/exec/exec-1/json"):
		if f.execInspectBody != "" {
			writeBody(w, f.execInspectBody)
			return
		}
		writeBody(w, `{"ExitCode":0,"Running":false}`)
	case strings.Contains(r.URL.Path, "/archive"):
		if f.archiveStatus != 0 {
			w.WriteHeader(f.archiveStatus)
			writeBody(w, f.archiveBody)
			return
		}
		w.WriteHeader(http.StatusOK)
	case strings.Contains(r.URL.Path, "/containers/") && strings.HasSuffix(r.URL.Path, "/json"):
		if f.inspectBody != "" {
			writeBody(w, f.inspectBody)
			return
		}
		writeBody(w, `{"State":{"Status":"running","Running":true}}`)
	case strings.Contains(r.URL.Path, "/images/"):
		// The image is present, so no pull is attempted.
		writeBody(w, `{}`)
	case strings.HasSuffix(r.URL.Path, "/containers/create"):
		f.enterCreate()
		time.Sleep(f.createDelay)
		writeBody(w, `{"Id":"c-1"}`)
	case strings.HasSuffix(r.URL.Path, "/start"):
		time.Sleep(f.createDelay)
		f.leaveCreate()
		w.WriteHeader(http.StatusOK)
	default:
		http.NotFound(w, r)
	}
}

func writeBody(w http.ResponseWriter, body string) {
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write([]byte(body))
}

func newFakeBackend(t *testing.T, engine *fakeEngine, requestTimeout time.Duration, maxCreate int) *Backend {
	t.Helper()
	b, err := New(Config{
		Socket:               engine.socket(t),
		APIVersion:           "v1.40",
		RequestTimeout:       requestTimeout,
		MaxCreateConcurrency: maxCreate,
	}, backend.SchedulingDirect)
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}
	return b
}

func TestAWriteIsNotBoundedByTheRequestTimeout(t *testing.T) {
	// A write runs `mkdir -p` first, and that is an exec. How long a command takes is
	// the command's business, and charging it to the request timeout is what turned a
	// daemon too busy to start `mkdir -p /tmp` into "docker exec start: context
	// deadline exceeded" and cost the episode its grader.
	engine := &fakeEngine{execStartDelay: 300 * time.Millisecond}
	b := newFakeBackend(t, engine, 50*time.Millisecond, 0)

	handle := backend.Handle{Backend: "docker", SandboxID: "c-1", NodeID: "node-a"}
	if err := b.WriteFile(context.Background(), handle, "/tmp/psrl-model.patch", []byte("diff --git a/a b/a\n")); err != nil {
		t.Fatalf("a slow mkdir must not fail the write: %v", err)
	}
}

func TestTheCreateGateBoundsConcurrentCreates(t *testing.T) {
	const gate = 2
	engine := &fakeEngine{createDelay: 30 * time.Millisecond}
	b := newFakeBackend(t, engine, time.Second, gate)

	var group sync.WaitGroup
	for range 8 {
		group.Add(1)
		go func() {
			defer group.Done()
			spec := backend.Spec{Source: backend.Source{Kind: "image", Reference: "img"}}
			_, _ = b.Create(context.Background(), "node-a", spec, "")
		}()
	}
	group.Wait()

	if peak := engine.peakInFlight(); peak > gate {
		t.Fatalf("the create gate admitted %d creates at once, want at most %d", peak, gate)
	}
}

func TestANegativeCreateConcurrencyIsRefused(t *testing.T) {
	// A negative value is a configuration mistake, and reading it as unbounded hides
	// the one setting that keeps a burst off the daemon's create queue.
	_, err := New(Config{Socket: "/tmp/unused.sock", MaxCreateConcurrency: -1}, backend.SchedulingDirect)

	if err == nil {
		t.Fatal("a negative create concurrency must be refused, not read as unbounded")
	}
}
