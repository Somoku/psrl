package placement

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"
)

type fakeMonitor struct {
	mu      sync.Mutex
	nodes   []NodeView
	version uint64
}

func (m *fakeMonitor) Version() uint64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.version
}

func (m *fakeMonitor) Fleet() []NodeView {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]NodeView, len(m.nodes))
	copy(out, m.nodes)
	return out
}

func (m *fakeMonitor) set(nodes []NodeView) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.nodes = nodes
	m.version++
}

func node(id string, memMB int64, opts ...func(*NodeView)) NodeView {
	view := NodeView{
		NodeID:          id,
		SeenAt:          time.Now(),
		Backends:        []BackendCapability{{Name: "docker", Features: map[string]struct{}{}}},
		ClassHeadroom:   map[string]Headroom{"rollout": {MemoryMB: memMB, CPUMillis: memMB}},
		Envelope:        Headroom{MemoryMB: 1000, CPUMillis: 1000},
		ImageDigests:    map[string]struct{}{},
		ImageReferences: map[string]struct{}{},
		Labels:          map[string]struct{}{},
	}
	for _, opt := range opts {
		opt(&view)
	}
	return view
}

func withFeatures(names ...string) func(*NodeView) {
	return func(v *NodeView) {
		for _, name := range names {
			v.Backends[0].Features[name] = struct{}{}
		}
	}
}

// hosting adds a second runtime to a node, which is the shape of a machine
// running a container daemon and a microVM side by side.
func hosting(name, resumeLevel string, features ...string) func(*NodeView) {
	return func(v *NodeView) {
		hosted := BackendCapability{
			Name: name, ResumeLevel: resumeLevel, Features: map[string]struct{}{},
		}
		for _, feature := range features {
			hosted.Features[feature] = struct{}{}
		}
		v.Backends = append(v.Backends, hosted)
	}
}

func withImage(ref string) func(*NodeView) {
	return func(v *NodeView) { v.ImageReferences[ref] = struct{}{} }
}

func draining(v *NodeView) { v.Draining = true }

func config() Config {
	return Config{NodeTTL: 120 * time.Second, ReservationTTL: 60 * time.Second, SweepInterval: 10 * time.Second}
}

func mustService(t *testing.T, monitor Monitor) *Service {
	t.Helper()
	service, err := New(config(), monitor)
	if err != nil {
		t.Fatalf("new placement: %v", err)
	}
	return service
}

func request(memMB int64) Request {
	return Request{Backend: "docker", ResourceClass: "rollout", Footprint: Headroom{MemoryMB: memMB}}
}

func TestAReservationTTLAboveTheNodeTTLIsRefused(t *testing.T) {
	cfg := config()
	cfg.ReservationTTL = cfg.NodeTTL

	if _, err := New(cfg, &fakeMonitor{}); err == nil {
		t.Fatal("a drained node would leave reservations pointing at nothing")
	}
}

func TestASweepSlowerThanTheReservationTTLIsRefused(t *testing.T) {
	cfg := config()
	cfg.SweepInterval = cfg.ReservationTTL

	if _, err := New(cfg, &fakeMonitor{}); err == nil {
		t.Fatal("a sweep must be faster than the TTL it enforces")
	}
}

func TestAFleetWithNoCapableNodeIsADeploymentFault(t *testing.T) {
	monitor := &fakeMonitor{nodes: []NodeView{node("a", 500)}}
	service := mustService(t, monitor)

	req := request(10)
	req.RequiredFeatures = []string{"native_fork"}
	_, err := service.Choose(req)

	if !errors.Is(err, ErrNoCandidate) {
		t.Fatalf("want ErrNoCandidate, got %v", err)
	}
}

func TestAFullFleetIsSeparateFromAnIncapableOne(t *testing.T) {
	// The caller's recourse differs: one clears as sandboxes are released and the
	// other never will, so they must not arrive as the same error.
	monitor := &fakeMonitor{nodes: []NodeView{node("a", 5)}}
	service := mustService(t, monitor)

	_, err := service.Choose(request(100))

	if !errors.Is(err, ErrExhausted) {
		t.Fatalf("want ErrExhausted, got %v", err)
	}
}

func TestADrainingNodeIsNeverChosen(t *testing.T) {
	// It still holds the memory of what it could not destroy.
	monitor := &fakeMonitor{nodes: []NodeView{node("a", 500, draining)}}
	service := mustService(t, monitor)

	_, err := service.Choose(request(10))

	if !errors.Is(err, ErrNoCandidate) {
		t.Fatalf("a draining node must not be a candidate, got %v", err)
	}
}

func TestASilentNodeIsDrainedAtItsTTL(t *testing.T) {
	stale := node("a", 500)
	stale.SeenAt = time.Now().Add(-5 * time.Minute)
	service := mustService(t, &fakeMonitor{nodes: []NodeView{stale}})

	_, err := service.Choose(request(10))

	if !errors.Is(err, ErrNoCandidate) {
		t.Fatalf("a node past its TTL must not be chosen, got %v", err)
	}
}

func TestTheLeastLoadedNodeWins(t *testing.T) {
	monitor := &fakeMonitor{nodes: []NodeView{node("busy", 100), node("idle", 900)}}
	service := mustService(t, monitor)

	decision, err := service.Choose(request(10))

	if err != nil {
		t.Fatalf("choose: %v", err)
	}
	if decision.NodeID != "idle" {
		t.Fatalf("chose %q, want the emptier node", decision.NodeID)
	}
}

func TestLocalityDecidesBetweenSimilarlyLoadedNodes(t *testing.T) {
	// Locality is an optimisation, so it breaks a tie rather than overriding load.
	monitor := &fakeMonitor{nodes: []NodeView{node("plain", 500), node("cached", 500, withImage("task:v1"))}}
	service := mustService(t, monitor)

	req := request(10)
	req.ImageReferences = []string{"task:v1"}
	decision, err := service.Choose(req)

	if err != nil {
		t.Fatalf("choose: %v", err)
	}
	if decision.NodeID != "cached" {
		t.Fatalf("chose %q, want the node already holding the image", decision.NodeID)
	}
}

func TestLoadBeatsLocality(t *testing.T) {
	// Ranking locality first draws a whole batch onto whichever node holds the
	// image, and the rest then queue invisibly at a node that cannot admit them.
	monitor := &fakeMonitor{nodes: []NodeView{node("full", 20, withImage("task:v1")), node("empty", 900)}}
	service := mustService(t, monitor)

	req := request(10)
	req.ImageReferences = []string{"task:v1"}
	decision, err := service.Choose(req)

	if err != nil {
		t.Fatalf("choose: %v", err)
	}
	if decision.NodeID != "empty" {
		t.Fatalf("chose %q: a genuinely fuller node must lose despite holding the image", decision.NodeID)
	}
}

func TestABurstDoesNotHerdOntoOneNode(t *testing.T) {
	// A reservation is charged here before the node can report it, so without the
	// in-flight overlay every request in a burst reads the same view and picks the
	// same node. This is the property AgentEnv's scheduler does not have.
	monitor := &fakeMonitor{nodes: []NodeView{node("a", 100), node("b", 100), node("c", 100)}}
	service := mustService(t, monitor)

	counts := map[string]int{}
	for i := 0; i < 30; i++ {
		decision, err := service.Choose(request(10))
		if err != nil {
			t.Fatalf("choose %d: %v", i, err)
		}
		counts[decision.NodeID]++
	}

	for id, n := range counts {
		if n != 10 {
			t.Fatalf("node %q took %d of 30; a burst must spread: %v", id, n, counts)
		}
	}
}

func TestAFreshViewClearsTheInFlightOverlay(t *testing.T) {
	// Once the node reports, its own accounting includes the grant, so keeping the
	// promise charged as well would double-count it.
	monitor := &fakeMonitor{nodes: []NodeView{node("a", 100)}}
	service := mustService(t, monitor)
	if _, err := service.Choose(request(60)); err != nil {
		t.Fatalf("first choose: %v", err)
	}

	// The node now reports the grant itself, with a newer timestamp.
	fresh := node("a", 40)
	fresh.SeenAt = time.Now().Add(time.Second)
	monitor.set([]NodeView{fresh})

	if _, err := service.Choose(request(30)); err != nil {
		t.Fatalf("a reported grant must not be charged twice: %v", err)
	}
}

func TestReleasingAReservationReturnsItsRoom(t *testing.T) {
	monitor := &fakeMonitor{nodes: []NodeView{node("a", 100)}}
	service := mustService(t, monitor)
	decision, err := service.Choose(request(90))
	if err != nil {
		t.Fatalf("choose: %v", err)
	}

	service.Release(decision.ReservationID)

	if _, err := service.Choose(request(90)); err != nil {
		t.Fatalf("released room must be reusable: %v", err)
	}
}

func TestAnUnrenewedReservationIsSwept(t *testing.T) {
	monitor := &fakeMonitor{nodes: []NodeView{node("a", 100)}}
	service := mustService(t, monitor)
	clock := time.Now()
	service.SetClock(func() time.Time { return clock })
	if _, err := service.Choose(request(90)); err != nil {
		t.Fatalf("choose: %v", err)
	}

	clock = clock.Add(2 * time.Minute)
	swept := service.Sweep()

	if len(swept) != 1 {
		t.Fatalf("swept %d reservations, want 1", len(swept))
	}
}

func TestARenewedReservationSurvivesTheSweep(t *testing.T) {
	monitor := &fakeMonitor{nodes: []NodeView{node("a", 100)}}
	service := mustService(t, monitor)
	clock := time.Now()
	service.SetClock(func() time.Time { return clock })
	decision, err := service.Choose(request(90))
	if err != nil {
		t.Fatalf("choose: %v", err)
	}

	for i := 0; i < 5; i++ {
		clock = clock.Add(20 * time.Second)
		service.Renew(decision.ReservationID)
	}

	if swept := service.Sweep(); len(swept) != 0 {
		t.Fatalf("a renewed reservation must survive, swept %d", len(swept))
	}
}

func TestANodeWithNoDeclaredEnvelopeIsNeverExcludedForCapacity(t *testing.T) {
	// It schedules its own capacity, so this service has no basis to refuse it.
	unbounded := node("provider", 0)
	unbounded.ClassHeadroom = map[string]Headroom{}
	service := mustService(t, &fakeMonitor{nodes: []NodeView{unbounded}})

	if _, err := service.Choose(request(100000)); err != nil {
		t.Fatalf("a provider node must not be filtered on capacity: %v", err)
	}
}

func TestPlacementIsReproducibleForEquallyGoodNodes(t *testing.T) {
	// Two equally good nodes must not alternate, or a run's placement cannot be
	// reproduced from its inputs.
	for i := 0; i < 5; i++ {
		monitor := &fakeMonitor{nodes: []NodeView{node("b", 500), node("a", 500)}}
		service := mustService(t, monitor)
		decision, err := service.Choose(request(10))
		if err != nil {
			t.Fatalf("choose: %v", err)
		}
		if decision.NodeID != "a" {
			t.Fatalf("run %d chose %q, want the stable first id", i, decision.NodeID)
		}
	}
}

func TestAResumeRequirementIsRefusedRatherThanDowngraded(t *testing.T) {
	// A workspace restore must not be readable as proof a live process survived.
	weak := node("a", 500, withFeatures("resume_anywhere"))
	weak.Backends[0].ResumeLevel = "filesystem"
	service := mustService(t, &fakeMonitor{nodes: []NodeView{weak}})

	req := request(10)
	req.RequiredFeatures = []string{"resume_anywhere"}
	req.RequiredResume = "full_state"
	_, err := service.Choose(req)

	if !errors.Is(err, ErrNoCandidate) {
		t.Fatalf("a weaker resume must be refused, got %v", err)
	}
}

func TestARequestIsMatchedToTheRuntimeThatHasTheCapability(t *testing.T) {
	// A node hosting both docker and agentenv satisfies a full-state resume
	// through the agentenv runtime only. The decision backend must name agentenv,
	// not docker, so the node creates on the right runtime.
	mixed := node("a", 500, hosting("agentenv", "full_state", "resume_anywhere", "native_fork"))
	service := mustService(t, &fakeMonitor{nodes: []NodeView{mixed}})

	req := Request{ResourceClass: "rollout", Footprint: Headroom{MemoryMB: 10}}
	req.RequiredFeatures = []string{"resume_anywhere"}
	req.RequiredResume = "full_state"
	decision, err := service.Choose(req)
	if err != nil {
		t.Fatalf("choose: %v", err)
	}
	if decision.Backend != "agentenv" {
		t.Fatalf("landed on %q, want the runtime that actually has full_state resume", decision.Backend)
	}
}

func TestARequestWithNoRequirementsPicksTheFirstRuntime(t *testing.T) {
	// A spec that states no requirements gets the default runtime, which is the
	// first one declared on the node.
	service := mustService(t, &fakeMonitor{nodes: []NodeView{node("a", 500)}})

	decision, err := service.Choose(request(10))
	if err != nil {
		t.Fatalf("choose: %v", err)
	}
	if decision.Backend != "docker" {
		t.Fatalf("got %q, want the first declared runtime", decision.Backend)
	}
}

func TestAPinNameASpecificRuntimeOnAMixedNode(t *testing.T) {
	// A caller can pin the backend name to hold it fixed across an ablation
	// while everything else in the spec stays the same.
	mixed := node("a", 500, hosting("agentenv", "full_state", "resume_anywhere"))
	service := mustService(t, &fakeMonitor{nodes: []NodeView{mixed}})

	req := request(10)
	req.Backend = "agentenv"
	decision, err := service.Choose(req)
	if err != nil {
		t.Fatalf("choose: %v", err)
	}
	if decision.Backend != "agentenv" {
		t.Fatalf("pin was not honoured: got %q", decision.Backend)
	}
}

func TestANodeWithNoRuntimeThatMatchesIsTreatedAsIncapable(t *testing.T) {
	// A node that only hosts docker cannot satisfy a request that needs a
	// feature only agentenv has. The error must be ErrNoCandidate, not
	// ErrExhausted, because adding more capacity would not fix it.
	dockerOnly := node("a", 500)
	service := mustService(t, &fakeMonitor{nodes: []NodeView{dockerOnly}})

	req := request(10)
	req.RequiredFeatures = []string{"resume_anywhere"}
	req.RequiredResume = "full_state"
	_, err := service.Choose(req)
	if !errors.Is(err, ErrNoCandidate) {
		t.Fatalf("a docker-only node must refuse a full-state resume with ErrNoCandidate, got %v", err)
	}
}

func TestConcurrentChoosesNeverOverfillANode(t *testing.T) {
	monitor := &fakeMonitor{nodes: []NodeView{node("a", 100)}}
	service := mustService(t, monitor)

	var wg sync.WaitGroup
	var mu sync.Mutex
	granted := 0
	for i := 0; i < 200; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := service.Choose(request(10)); err == nil {
				mu.Lock()
				granted++
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	if granted != 10 {
		t.Fatalf("placed %d x 10MB onto a 100MB node, want exactly 10", granted)
	}
}

func BenchmarkChoose(b *testing.B) {
	nodes := make([]NodeView, 160)
	for i := range nodes {
		nodes[i] = node(fmt.Sprintf("node-%03d", i), 1<<40)
	}
	service, err := New(config(), &fakeMonitor{nodes: nodes})
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decision, err := service.Choose(request(1))
		if err != nil {
			b.Fatal(err)
		}
		service.Release(decision.ReservationID)
	}
}

func BenchmarkChooseFleetSizes(b *testing.B) {
	for _, size := range []int{16, 160, 1600} {
		b.Run(fmt.Sprintf("nodes=%d", size), func(b *testing.B) {
			nodes := make([]NodeView, size)
			for i := range nodes {
				nodes[i] = node(fmt.Sprintf("node-%04d", i), 1<<40)
			}
			service, err := New(config(), &fakeMonitor{nodes: nodes})
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				decision, err := service.Choose(request(1))
				if err != nil {
					b.Fatal(err)
				}
				service.Release(decision.ReservationID)
			}
		})
	}
}

// -- dedicated placement: forbidden labels ------------------------------------

// withLabel marks a node as belonging to a class, which is how a trainer node is
// told apart from an env node.
func withLabel(names ...string) func(*NodeView) {
	return func(v *NodeView) {
		for _, name := range names {
			v.Labels[name] = struct{}{}
		}
	}
}

func TestAForbiddenLabelKeepsASandboxOffThatNode(t *testing.T) {
	// The failure this prevents: a GPU sandbox landing on a trainer node, where it
	// competes with the trainer for the memory the trainer already reserved
	// outside this service's accounting.
	service := mustService(t, &fakeMonitor{nodes: []NodeView{
		node("trainer-1", 1000, withLabel("trainer")),
	}})
	req := request(10)
	req.ForbiddenLabels = []string{"trainer"}

	if _, err := service.Choose(req); err == nil {
		t.Fatal("a node carrying a forbidden label must not be chosen")
	}
}

func TestAForbiddenLabelStillAllowsAnUnlabelledNode(t *testing.T) {
	service := mustService(t, &fakeMonitor{nodes: []NodeView{
		node("trainer-1", 1000, withLabel("trainer")),
		node("env-1", 1000),
	}})
	req := request(10)
	req.ForbiddenLabels = []string{"trainer"}

	decision, err := service.Choose(req)
	if err != nil {
		t.Fatalf("choose: %v", err)
	}
	if decision.NodeID != "env-1" {
		t.Errorf("node: got %q, want env-1 (the one without the forbidden label)", decision.NodeID)
	}
}

func TestSeveralForbiddenLabelsAreAllExcluded(t *testing.T) {
	service := mustService(t, &fakeMonitor{nodes: []NodeView{
		node("trainer-1", 1000, withLabel("trainer")),
		node("infra-1", 1000, withLabel("infra")),
		node("env-1", 1000, withLabel("env")),
	}})
	req := request(10)
	req.ForbiddenLabels = []string{"trainer", "infra"}

	decision, err := service.Choose(req)
	if err != nil {
		t.Fatalf("choose: %v", err)
	}
	if decision.NodeID != "env-1" {
		t.Errorf("node: got %q, want env-1", decision.NodeID)
	}
}

func TestNoForbiddenLabelsLeavesEveryNodeEligible(t *testing.T) {
	// The common case must not be changed by the mechanism existing: a request
	// that names no forbidden label is placed exactly as before.
	service := mustService(t, &fakeMonitor{nodes: []NodeView{
		node("trainer-1", 1000, withLabel("trainer")),
	}})
	if _, err := service.Choose(request(10)); err != nil {
		t.Fatalf("a request naming no forbidden label must still place: %v", err)
	}
}

func TestARequiredAndAForbiddenLabelApplyTogether(t *testing.T) {
	// They are separate constraints, so both are enforced on one request: the
	// node must carry the required label and must not carry the forbidden one.
	service := mustService(t, &fakeMonitor{nodes: []NodeView{
		node("gpu-trainer", 1000, withLabel("gpu", "trainer")),
		node("gpu-env", 1000, withLabel("gpu")),
	}})
	req := request(10)
	req.RequiredLabel = "gpu"
	req.ForbiddenLabels = []string{"trainer"}

	decision, err := service.Choose(req)
	if err != nil {
		t.Fatalf("choose: %v", err)
	}
	if decision.NodeID != "gpu-env" {
		t.Errorf("node: got %q, want gpu-env (has gpu, lacks trainer)", decision.NodeID)
	}
}

func TestAForbiddenLabelIsRefusedEvenWhenItIsTheOnlyNodeWithRoom(t *testing.T) {
	// Capacity does not override the constraint. A dedicated request that cannot
	// be honoured has to fail rather than land somewhere it must not.
	service := mustService(t, &fakeMonitor{nodes: []NodeView{
		node("trainer-big", 1<<20, withLabel("trainer")),
		node("env-full", 0),
	}})
	req := request(1000)
	req.ForbiddenLabels = []string{"trainer"}

	if _, err := service.Choose(req); err == nil {
		t.Fatal("a forbidden label must hold even when the forbidden node is the only one with room")
	}
}
