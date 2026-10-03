//go:build integration

package server

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

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/backend/agentenv"
	"psrl.dev/sandboxd/internal/backend/dockerbackend"
	"psrl.dev/sandboxd/internal/monitor"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/placement"
	"psrl.dev/sandboxd/internal/quota"
	"psrl.dev/sandboxd/internal/timing"
)

// Two backends in one fleet, which is the case the whole service exists for: a
// caller states what it needs, the service routes to a backend that can provide
// it, and one quota ledger bounds both.
//
// Docker is the real daemon. AgentEnv is a server speaking its documented
// shapes, because a microVM deployment needs kernel 6.8+ and /dev/kvm that this
// node does not have -- so what is proven here is the routing and the shared
// accounting, not AgentEnv's own runtime.

func fakeMicroVM(t *testing.T) *httptest.Server {
	t.Helper()
	var mu sync.Mutex
	live := map[string]bool{}
	seq := 0
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("/sandboxes-cold", func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		seq++
		id := fmt.Sprintf("vm-%d", seq)
		live[id] = true
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"sandboxID":       id,
			"envdAccessToken": map[string]string{"token": "tok"},
		})
	})
	mux.HandleFunc("/sandboxes/", func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		id, _, _ := strings.Cut(strings.TrimPrefix(r.URL.Path, "/sandboxes/"), "/")
		if !live[id] {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if r.Method == http.MethodDelete {
			delete(live, id)
			w.WriteHeader(http.StatusNoContent)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"state": "running"})
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return server
}

type mixedStack struct {
	control  *Control
	ledger   *quota.Ledger
	microVM  *httptest.Server
	nodeName string
}

func newMixedStack(t *testing.T, fleetMemoryMB int64) *mixedStack {
	t.Helper()
	docker, nodeAgent := dockerNodeFor(t, fleetMemoryMB)
	vm := fakeMicroVM(t)

	// Provider mode: AgentEnv's own control plane places, so this service passes
	// no node and the gateway resolves it. That is the shape a deployment uses
	// when the provider's scheduler is worth keeping.
	micro, err := agentenv.New(agentenv.Config{Gateway: vm.URL}, backend.SchedulingProvider)
	if err != nil {
		t.Fatalf("agentenv backend: %v", err)
	}

	spans, err := timing.New(2*time.Second, 8*time.Second, 5*time.Second, nil)
	if err != nil {
		t.Fatalf("timing: %v", err)
	}
	mon := monitor.New(spans.NodeTTL)
	mon.Report(nodeAgent.View())

	ledger, err := quota.New(quota.Config{
		Total:    quota.Amount{MemoryMB: fleetMemoryMB, CPUMillis: 100000, Sandboxes: 1000},
		Classes:  map[string]quota.ClassShare{"rollout": {Guaranteed: 0.7}, "grader": {Guaranteed: 0.2}},
		LeaseTTL: spans.CapacityLeaseTTL(),
	})
	if err != nil {
		t.Fatalf("ledger: %v", err)
	}
	place, err := placement.New(placement.Config{
		NodeTTL: spans.NodeTTL, ReservationTTL: spans.ReservationTTL(), SweepInterval: spans.SweepInterval(),
	}, mon)
	if err != nil {
		t.Fatalf("placement: %v", err)
	}
	// Docker is the default, so a spec that states no requirement lands there and
	// only one that needs what Docker lacks is routed to the microVM.
	registry, err := backend.NewRegistry([]backend.Backend{docker, micro}, "docker")
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	control, err := NewControl(ControlConfig{
		Registry: registry, Ledger: ledger, Placement: place, Monitor: mon,
		Nodes: NewLocalNodeClient(nodeAgent), AcquireTimeout: time.Minute,
	})
	if err != nil {
		t.Fatalf("control: %v", err)
	}
	return &mixedStack{control: control, ledger: ledger, microVM: vm, nodeName: "node-a"}
}

func dockerNodeFor(t *testing.T, memoryMB int64) (*dockerbackend.Backend, *Node) {
	t.Helper()
	socket := dockerSocket(t)
	docker, err := dockerbackend.New(dockerbackend.Config{
		Socket: socket, APIVersion: "v1.40", NodeID: "node-a",
		OwnerID:        fmt.Sprintf("sandboxd-mixed-%d", time.Now().UnixNano()),
		RequestTimeout: 30 * time.Second, PullTimeout: 3 * time.Minute,
	}, backend.SchedulingDirect)
	if err != nil {
		t.Fatalf("docker backend: %v", err)
	}
	if err := docker.Preflight(context.Background()); err != nil {
		t.Skipf("daemon not usable: %v", err)
	}
	spans, _ := timing.New(2*time.Second, 8*time.Second, 5*time.Second, nil)
	gate, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{MemoryMB: memoryMB, CPUMillis: 100000},
		Classes:  map[string]node.ClassShare{"rollout": {Guaranteed: 0.7}, "grader": {Guaranteed: 0.2}},
		LeaseTTL: spans.CapacityLeaseTTL(),
	})
	if err != nil {
		t.Fatalf("admission: %v", err)
	}
	life, err := node.NewLifecycle("node-a", gate, node.Windows{
		PauseWindow: spans.PauseWindow(), ReapWindow: spans.ReapWindow(),
		Lifetime: spans.Lifetime(), SweepInterval: spans.SweepInterval(),
	})
	if err != nil {
		t.Fatalf("lifecycle: %v", err)
	}
	agent, err := NewNode(NodeConfig{
		NodeID: "node-a", Admission: gate, Lifecycle: life, Backends: []backend.Backend{docker},
	})
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	return docker, agent
}

func TestASpecWithNoRequirementsLandsOnTheDefaultBackend(t *testing.T) {
	s := newMixedStack(t, 4096)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer s.control.Release(ctx, created.GetHandle())

	if created.GetHandle().GetBackend() != "docker" {
		t.Fatalf("landed on %q, want the default backend", created.GetHandle().GetBackend())
	}
}

func TestASpecNeedingAFullStateResumeIsRoutedToTheBackendThatHasOne(t *testing.T) {
	// This is the routing the service exists for: the caller names a guarantee,
	// not a backend, and gets the one backend that can keep it.
	s := newMixedStack(t, 4096)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	spec := createSpec(64)
	spec.RequiredFeatures = []v1.Feature{v1.Feature_RESUME_ANYWHERE}
	spec.RequiredResumeLevel = v1.ResumeLevel_FULL_STATE

	created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: spec})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer s.control.Release(ctx, created.GetHandle())

	if created.GetHandle().GetBackend() != "agentenv" {
		t.Fatalf("landed on %q: only the microVM backend resumes a live process elsewhere",
			created.GetHandle().GetBackend())
	}
}

func TestAPinOverridesCapabilityRouting(t *testing.T) {
	// An ablation has to be able to hold the backend fixed while everything else
	// stays the same.
	s := newMixedStack(t, 4096)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	spec := createSpec(64)
	spec.Backend = "agentenv"

	created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: spec})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer s.control.Release(ctx, created.GetHandle())

	if created.GetHandle().GetBackend() != "agentenv" {
		t.Fatalf("a pin was not honoured: landed on %q", created.GetHandle().GetBackend())
	}
}

func TestOneQuotaBoundsBothBackends(t *testing.T) {
	// The cross-backend ledger is the thing no single backend can provide: a
	// rollout on the microVM and one on Docker draw from the same share.
	s := newMixedStack(t, 4096)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	before := s.ledger.Headroom("rollout").MemoryMB

	onDocker, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)})
	if err != nil {
		t.Fatalf("docker create: %v", err)
	}
	defer s.control.Release(ctx, onDocker.GetHandle())

	pinned := createSpec(128)
	pinned.Backend = "agentenv"
	onMicroVM, err := s.control.Create(ctx, &v1.CreateRequest{Spec: pinned})
	if err != nil {
		t.Fatalf("microvm create: %v", err)
	}
	defer s.control.Release(ctx, onMicroVM.GetHandle())

	after := s.ledger.Headroom("rollout").MemoryMB
	if after != before-192 {
		t.Fatalf("headroom went %d -> %d, want both backends charged (-192MB)", before, after)
	}
}

func TestReleasingOnOneBackendReturnsQuotaTheOtherCanUse(t *testing.T) {
	// A 128MB fleet: the microVM takes it all, and only its release lets Docker in.
	s := newMixedStack(t, 128)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	pinned := createSpec(128)
	pinned.Backend = "agentenv"
	onMicroVM, err := s.control.Create(ctx, &v1.CreateRequest{Spec: pinned})
	if err != nil {
		t.Fatalf("microvm create: %v", err)
	}

	if _, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)}); err == nil {
		t.Fatal("the fleet is spent on the microVM, so Docker must be refused")
	}

	if _, err := s.control.Release(ctx, onMicroVM.GetHandle()); err != nil {
		t.Fatalf("release: %v", err)
	}
	onDocker, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)})
	if err != nil {
		t.Fatalf("after the release Docker must be admitted: %v", err)
	}
	s.control.Release(ctx, onDocker.GetHandle())
}

func TestASpecNoBackendCanServeNamesWhatWasMissing(t *testing.T) {
	s := newMixedStack(t, 4096)
	spec := createSpec(64)
	// Neither backend binds a host path and resumes a live process elsewhere.
	spec.RequiredFeatures = []v1.Feature{v1.Feature_HOST_MOUNT, v1.Feature_NATIVE_FORK}

	_, err := s.control.Create(context.Background(), &v1.CreateRequest{Spec: spec})

	if err == nil {
		t.Fatal("a spec no backend can serve must be refused")
	}
	if !strings.Contains(err.Error(), "docker") || !strings.Contains(err.Error(), "agentenv") {
		t.Fatalf("the refusal must say why each backend was rejected, got %v", err)
	}
}

func TestAMixedBurstKeepsBothBackendsInsideOneFleet(t *testing.T) {
	// 512MB of 64MB sandboxes is eight, whichever backend serves them.
	s := newMixedStack(t, 512)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	var mu sync.Mutex
	var handles []*v1.SandboxHandle
	var wg sync.WaitGroup
	for i := 0; i < 24; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			spec := createSpec(64)
			if i%2 == 0 {
				spec.Backend = "agentenv"
			}
			created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: spec})
			if err != nil {
				return
			}
			mu.Lock()
			handles = append(handles, created.GetHandle())
			mu.Unlock()
		}(i)
	}
	wg.Wait()
	defer func() {
		for _, handle := range handles {
			s.control.Release(context.Background(), handle)
		}
	}()

	if len(handles) != 8 {
		t.Fatalf("admitted %d across two backends onto a 512MB fleet, want 8", len(handles))
	}
	backends := map[string]int{}
	for _, handle := range handles {
		backends[handle.GetBackend()]++
	}
	if len(backends) < 2 {
		t.Fatalf("the burst used only %v; a mixed fleet must serve from both", backends)
	}
}
