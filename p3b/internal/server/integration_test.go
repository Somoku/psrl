//go:build integration

package server

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/backend/dockerbackend"
	"psrl.dev/sandboxd/internal/monitor"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/placement"
	"psrl.dev/sandboxd/internal/quota"
	"psrl.dev/sandboxd/internal/timing"
)

// The whole service, wired as it is deployed: a control plane over a monitor and
// a placement engine, one node agent with its own admission, and a real Docker
// daemon underneath. A test against fakes at every layer would only prove the
// layers agree with each other.

type stack struct {
	control *Control
	node    *Node
	monitor *monitor.Monitor
	ledger  *quota.Ledger
	place   *placement.Service
	backend *dockerbackend.Backend
}

// dockerSocket returns the daemon to test against, skipping where none is.
func dockerSocket(t *testing.T) string {
	t.Helper()
	socket := os.Getenv("DOCKER_SOCKET")
	if socket == "" {
		socket = "/var/run/docker.sock"
	}
	if _, err := os.Stat(socket); err != nil {
		t.Skipf("no docker socket at %s", socket)
	}
	return socket
}

func newStack(t *testing.T, fleetMemoryMB int64) *stack {
	t.Helper()
	socket := dockerSocket(t)

	// Short spans, so a reclamation test does not wait an episode.
	spans, err := timing.New(2*time.Second, 8*time.Second, 5*time.Second, nil)
	if err != nil {
		t.Fatalf("timing: %v", err)
	}

	docker, err := dockerbackend.New(dockerbackend.Config{
		Socket:         socket,
		APIVersion:     "v1.40",
		NodeID:         "node-a",
		OwnerID:        fmt.Sprintf("sandboxd-int-%d", time.Now().UnixNano()),
		RequestTimeout: 30 * time.Second,
		PullTimeout:    3 * time.Minute,
	}, backend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("docker backend: %v", err)
	}
	if err := docker.Preflight(context.Background()); err != nil {
		t.Skipf("daemon not usable: %v", err)
	}

	gate, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{MemoryMB: fleetMemoryMB, CPUMillis: 100000},
		Classes:  map[string]node.ClassShare{"rollout": {Guaranteed: 0.7}, "grader": {Guaranteed: 0.2}},
		LeaseTTL: spans.CapacityLeaseTTL(),
	})
	if err != nil {
		t.Fatalf("admission: %v", err)
	}
	life, err := node.NewLifecycle("node-a", gate, node.Windows{
		PauseWindow:   spans.PauseWindow(),
		ReapWindow:    spans.ReapWindow(),
		Lifetime:      spans.Lifetime(),
		SweepInterval: spans.SweepInterval(),
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

	mon := monitor.New(spans.NodeTTL)
	mon.Report(agent.View())

	ledger, err := quota.New(quota.Config{
		Total:    quota.Amount{MemoryMB: fleetMemoryMB, CPUMillis: 100000, Sandboxes: 1000},
		Classes:  map[string]quota.ClassShare{"rollout": {Guaranteed: 0.7}, "grader": {Guaranteed: 0.2}},
		LeaseTTL: spans.CapacityLeaseTTL(),
	})
	if err != nil {
		t.Fatalf("ledger: %v", err)
	}
	place, err := placement.New(placement.Config{
		NodeTTL:        spans.NodeTTL,
		ReservationTTL: spans.ReservationTTL(),
		SweepInterval:  spans.SweepInterval(),
	}, mon)
	if err != nil {
		t.Fatalf("placement: %v", err)
	}
	registry, err := backend.NewRegistry([]backend.Backend{docker}, "docker")
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	control, err := NewControl(ControlConfig{
		Registry: registry, Ledger: ledger, Placement: place, Monitor: mon,
		Nodes: NewLocalNodeClient(agent), AcquireTimeout: time.Minute,
	})
	if err != nil {
		t.Fatalf("control: %v", err)
	}
	return &stack{control: control, node: agent, monitor: mon, ledger: ledger, place: place, backend: docker}
}

func (s *stack) refresh() { s.monitor.Report(s.node.View()) }

func createSpec(memoryMB int64) *v1.SandboxSpec {
	memory := memoryMB
	cpu := 0.25
	return &v1.SandboxSpec{
		Source:        &v1.Source{Kind: v1.Source_IMAGE, Reference: "alpine:latest"},
		Resources:     &v1.Resources{MemoryMb: &memory, CpuCount: &cpu},
		ResourceClass: "rollout",
	}
}

func TestOneSandboxGoesEndToEndThroughEveryLayer(t *testing.T) {
	s := newStack(t, 4096)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer s.control.Release(ctx, created.GetHandle())

	if created.GetHandle().GetNodeId() != "node-a" {
		t.Fatalf("sandbox landed on %q, want node-a", created.GetHandle().GetNodeId())
	}
	status, err := s.control.Status(ctx, created.GetHandle())
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if status.GetStatus() != v1.SandboxStatus_RUNNING {
		t.Fatalf("status is %v, want RUNNING", status.GetStatus())
	}

	// The command path goes to the node, never through the control plane.
	code, output, err := s.node.Exec(ctx,
		backend.Handle{SandboxID: created.GetHandle().GetSandboxId()}, "echo wired", "", nil)
	if err != nil || code != 0 {
		t.Fatalf("exec: code=%d err=%v", code, err)
	}
	if !strings.Contains(output, "wired") {
		t.Fatalf("output %q", output)
	}
}

func TestReleasingReturnsBothTheFleetQuotaAndTheNodeEnvelope(t *testing.T) {
	s := newStack(t, 4096)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	before := s.ledger.Headroom("rollout").MemoryMB

	created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	during := s.ledger.Headroom("rollout").MemoryMB
	if during != before-64 {
		t.Fatalf("quota went %d -> %d, want -64MB charged", before, during)
	}

	if _, err := s.control.Release(ctx, created.GetHandle()); err != nil {
		t.Fatalf("release: %v", err)
	}

	if after := s.ledger.Headroom("rollout").MemoryMB; after != before {
		t.Fatalf("quota went %d -> %d, want the charge returned", before, after)
	}
	if left := s.node.lifecycle.Snapshot().Resident; left != 0 {
		t.Fatalf("%d sandboxes still held by the node", left)
	}
}

func TestAFleetWithoutRoomRefusesRatherThanOversubscribing(t *testing.T) {
	// A 128MB fleet takes two 64MB sandboxes and must refuse the third, at the
	// quota rather than at the node: the fleet ledger is the outer bound.
	s := newStack(t, 128)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	var handles []*v1.SandboxHandle
	defer func() {
		for _, handle := range handles {
			s.control.Release(context.Background(), handle)
		}
	}()
	for i := 0; i < 2; i++ {
		created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)})
		if err != nil {
			t.Fatalf("create %d: %v", i, err)
		}
		handles = append(handles, created.GetHandle())
	}

	if _, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)}); err == nil {
		t.Fatal("a third sandbox must be refused on a full fleet")
	}
}

func TestASpecNoBackendCanServeIsRefusedAtAdmission(t *testing.T) {
	// Refused rather than degraded: a workspace restore must never be handed to a
	// caller that asked for a live process to survive.
	s := newStack(t, 4096)
	spec := createSpec(64)
	spec.RequiredFeatures = []v1.Feature{v1.Feature_NATIVE_FORK}

	_, err := s.control.Create(context.Background(), &v1.CreateRequest{Spec: spec})

	if err == nil {
		t.Fatal("docker has no native fork, so this spec must be refused")
	}
	if !strings.Contains(err.Error(), "native_fork") {
		t.Fatalf("the refusal must name what was missing, got %v", err)
	}
}

func TestANodeUnderPressureSendsTheRequestElsewhere(t *testing.T) {
	// The node keeps the last word: placement compares a headroom reported
	// seconds ago, and the machine knows its pressure now.
	s := newStack(t, 4096)
	s.node.pressure = func() node.Pressure { return node.Pressure{MemUsedPct: 0.99} }
	s.node.admission = mustPressureGate(t)

	_, err := s.control.Create(context.Background(), &v1.CreateRequest{Spec: createSpec(64)})

	if err == nil {
		t.Fatal("a node under pressure must refuse, and no other node exists")
	}
}

func mustPressureGate(t *testing.T) *node.Admission {
	t.Helper()
	gate, err := node.NewAdmission(node.Config{
		Envelope:        node.Resources{MemoryMB: 4096, CPUMillis: 100000},
		Classes:         map[string]node.ClassShare{"rollout": {Guaranteed: 1.0}},
		LocalMemCeiling: 0.9,
		LeaseTTL:        time.Minute,
	})
	if err != nil {
		t.Fatalf("admission: %v", err)
	}
	return gate
}

func TestAnIdleSandboxIsPausedAndThenReclaimedByTheNode(t *testing.T) {
	// Reclamation is the node's, so this runs without the control plane doing
	// anything: a caller that walked away must not leave the sandbox resident.
	s := newStack(t, 4096)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	defer s.control.Release(context.Background(), created.GetHandle())

	// Past the 4s pause window derived from a 2s episode deadline.
	time.Sleep(5 * time.Second)
	paused, err := s.node.Sweep(ctx, &v1.Empty{})
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if len(paused.GetPaused()) != 1 {
		t.Fatalf("the idle sandbox was not paused: %+v", paused)
	}

	// Past the 12s reap window.
	time.Sleep(9 * time.Second)
	reaped, err := s.node.Sweep(ctx, &v1.Empty{})
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if len(reaped.GetReapedIdle()) != 1 {
		t.Fatalf("the idle sandbox was not reclaimed: %+v", reaped)
	}
}

func TestABurstOfCreatesStaysInsideTheFleet(t *testing.T) {
	// Creates arrive in bursts, so the invariant that matters is that concurrency
	// never admits past the fleet, whatever the interleaving.
	s := newStack(t, 512)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	var mu sync.Mutex
	var handles []*v1.SandboxHandle
	var wg sync.WaitGroup
	for i := 0; i < 24; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			created, err := s.control.Create(ctx, &v1.CreateRequest{Spec: createSpec(64)})
			if err != nil {
				return
			}
			mu.Lock()
			handles = append(handles, created.GetHandle())
			mu.Unlock()
		}()
	}
	wg.Wait()
	defer func() {
		for _, handle := range handles {
			s.control.Release(context.Background(), handle)
		}
	}()

	// 512MB fleet, 64MB each: exactly eight fit.
	if len(handles) != 8 {
		t.Fatalf("admitted %d sandboxes onto a 512MB fleet of 64MB each, want 8", len(handles))
	}
	if granted := s.ledger.Report()["rollout"].Granted.MemoryMB; granted != 512 {
		t.Fatalf("the ledger accounts %dMB, want 512MB", granted)
	}
}

func TestTheFleetAndQuotaReportsAreServable(t *testing.T) {
	// The planes report and never log, and the trainer's metric hook is the only
	// reader, so these have to answer even on an empty fleet.
	s := newStack(t, 4096)
	ctx := context.Background()

	fleet, err := s.control.Fleet(ctx, &v1.Empty{})
	if err != nil {
		t.Fatalf("fleet: %v", err)
	}
	if len(fleet.GetNodes()) != 1 {
		t.Fatalf("fleet reports %d nodes, want 1", len(fleet.GetNodes()))
	}

	quotaReport, err := s.control.Quota(ctx, &v1.Empty{})
	if err != nil {
		t.Fatalf("quota: %v", err)
	}
	if _, present := quotaReport.GetClasses()["rollout"]; !present {
		t.Fatalf("quota report is missing the rollout class: %v", quotaReport.GetClasses())
	}
}

func TestAGroupIsAdmittedMemberByMember(t *testing.T) {
	// A group is a completion unit rather than a scheduling unit, so no member
	// holds capacity while a sibling queues.
	s := newStack(t, 4096)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	group, err := s.control.CreateGroup(ctx, &v1.CreateGroupRequest{
		Specs: []*v1.SandboxSpec{createSpec(64), createSpec(64), createSpec(64)},
	})
	if err != nil {
		t.Fatalf("create group: %v", err)
	}
	defer func() {
		for _, member := range group.GetMembers() {
			s.control.Release(context.Background(), member.GetHandle())
		}
	}()

	if len(group.GetMembers()) != 3 {
		t.Fatalf("group has %d members, want 3", len(group.GetMembers()))
	}
}

func TestAFailedGroupLeavesNothingCharged(t *testing.T) {
	// 128MB takes two members; the third fails and the first two must be returned.
	s := newStack(t, 128)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	before := s.ledger.Headroom("rollout").MemoryMB

	_, err := s.control.CreateGroup(ctx, &v1.CreateGroupRequest{
		Specs: []*v1.SandboxSpec{createSpec(64), createSpec(64), createSpec(64)},
	})

	if err == nil {
		t.Fatal("a group larger than the fleet must fail")
	}
	if after := s.ledger.Headroom("rollout").MemoryMB; after != before {
		t.Fatalf("quota went %d -> %d; a failed group must leave nothing charged", before, after)
	}
}
