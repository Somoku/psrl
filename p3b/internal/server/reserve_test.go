package server

import (
	"context"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/monitor"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/placement"
	"psrl.dev/sandboxd/internal/quota"
)

// -- what a node publishes ------------------------------------------------------

// elasticNode builds a node with elastic pricing on, over a stub backend.
func elasticNode(t *testing.T, reserveMB int64) *Node {
	t.Helper()
	gate, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{MemoryMB: 4096, CPUMillis: 100000},
		Classes:  map[string]node.ClassShare{"rollout": {Guaranteed: 1.0}},
		LeaseTTL: time.Minute,
	})
	if err != nil {
		t.Fatalf("gate: %v", err)
	}
	gate.SetPressure(node.Pressure{SampledAt: time.Now(), Valid: true})
	life, err := node.NewLifecycle("n1", gate, time.Second)
	if err != nil {
		t.Fatalf("lifecycle: %v", err)
	}
	agent, err := NewNode(NodeConfig{
		NodeID: "n1", Admission: gate, Lifecycle: life,
		Backends: []backend.Backend{providerStub{}},
		Pressure: func() node.Pressure { return node.Pressure{SampledAt: time.Now(), Valid: true} },
		Elastic: ElasticConfig{
			Enabled:      true,
			Footprint:    node.FootprintConfig{MinReserve: reserveMB, GrowthMargin: 0.25},
			CPUFootprint: node.FootprintConfig{MinReserve: 250, GrowthMargin: 0.25},
		},
	})
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	return agent
}

func TestANodePublishesWhatOneOfItsSandboxesCosts(t *testing.T) {
	// Admission above the node -- the fleet quota and placement -- charges a request before
	// this node sees it, so a node that prices a sandbox but keeps the number to itself
	// leaves every caller charging the declared maximum.
	agent := elasticNode(t, 512)

	view := agent.View()

	if got := view.ClassReserveMB["rollout"]; got != 512 {
		t.Errorf("published reserve: got %dMB, want the 512MiB minimum", got)
	}
}

func TestThePublishedReserveFollowsMeasurement(t *testing.T) {
	agent := elasticNode(t, 512)

	// Several samples: one sample may only double, so a large measurement is converged on
	// over a few ticks rather than in a jump the rest of the system cannot absorb.
	for i := 0; i < 8; i++ {
		agent.memoryFootprintFor("rollout").Observe([]int64{1000})
	}

	if got := agent.View().ClassReserveMB["rollout"]; got != 1250 {
		t.Errorf("published reserve: got %dMB, want 1250 (1000 x 1.25)", got)
	}
}

func TestANodePublishesWhatOneOfItsSandboxesCostsInCPU(t *testing.T) {
	// CPU is the dimension a burst is bounded by: every sandbox declares the cores one
	// tool call may want, and a fleet that charges the declaration admits a number that
	// has nothing to do with the load on it.
	agent := elasticNode(t, 512)

	view := agent.View()

	if got := view.ClassReserveCPUMillis["rollout"]; got != 250 {
		t.Errorf("published CPU reserve: got %d millis, want the 250 millicore minimum", got)
	}
	for i := 0; i < 8; i++ {
		agent.cpuFootprintFor("rollout").Observe([]int64{800})
	}
	if got := agent.View().ClassReserveCPUMillis["rollout"]; got != 1000 {
		t.Errorf("published CPU reserve after measurement: got %d millis, want 1000 (800 x 1.25)", got)
	}
}

func TestANodeWithoutElasticPricingPublishesNothing(t *testing.T) {
	// Nothing published is what keeps a deployment that has not opted in on the old
	// behaviour: the caller charges the request's maximum.
	agent := elasticNode(t, 512)
	agent.elastic = nil

	if got := agent.View().ClassReserveMB; got != nil {
		t.Errorf("published reserve: got %v, want none", got)
	}
}

// -- what the fleet charges -----------------------------------------------------

func TestTheFleetChargesThePublishedReserveNotTheMaximum(t *testing.T) {
	control, ledger := newTestControl(t, &stubNodes{}, time.Now)
	spec := backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "img"},
		Resources:     backend.Resources{MemoryMB: 512, CPUCount: 1},
		ResourceClass: "rollout",
	}

	if _, err := control.Create(context.Background(), &v1.CreateRequest{Spec: specToProto(spec)}); err != nil {
		t.Fatalf("first create: %v", err)
	}
	if got := ledger.Report()["rollout"].Granted.MemoryMB; got != 512 {
		t.Fatalf("charge without a published reserve: got %dMB, want the 512MB maximum", got)
	}

	control.monitor.Report(placement.NodeView{
		NodeID: "n1", SeenAt: time.Now(),
		ClassReserveMB: map[string]int64{"rollout": 128},
	})

	if _, err := control.Create(context.Background(), &v1.CreateRequest{Spec: specToProto(spec)}); err != nil {
		t.Fatalf("second create: %v", err)
	}
	if got := ledger.Report()["rollout"].Granted.MemoryMB; got != 512+128 {
		t.Errorf("charge with a published reserve: got %dMB, want 512 + 128", got)
	}
}

func TestTheChargedSpecCarriesTheReserveForMemoryAndCPUOnly(t *testing.T) {
	// Disk and devices are stated requirements rather than measured costs: disk is
	// consumed rather than held, and a GPU is a whole object a sandbox either has or does
	// not. So a published reserve reprices memory and CPU and leaves the rest.
	control, _ := newTestControl(t, &stubNodes{}, time.Now)
	control.monitor.Report(placement.NodeView{
		NodeID: "n1", SeenAt: time.Now(),
		ClassReserveMB:        map[string]int64{"rollout": 128},
		ClassReserveCPUMillis: map[string]int64{"rollout": 250},
	})
	spec := backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "img"},
		Resources:     backend.Resources{MemoryMB: 512, CPUCount: 2, DiskMB: 4096},
		ResourceClass: "rollout",
	}

	charged := control.charged(spec)

	if charged.Resources.MemoryMB != 128 {
		t.Errorf("memory: got %d, want the 128MB reserve", charged.Resources.MemoryMB)
	}
	// 250 millicores is a quarter of a core, and the request declared two.
	if charged.Resources.CPUCount*1000 != 250 {
		t.Errorf("CPU: got %.3f cores, want the 250 millicore reserve", charged.Resources.CPUCount)
	}
	if charged.Resources.DiskMB != 4096 {
		t.Errorf("disk must be untouched: %+v", charged.Resources)
	}
	if spec.Resources.MemoryMB != 512 || spec.Resources.CPUCount != 2 {
		t.Error("the caller's spec must not be mutated")
	}
}

func TestTheFleetChargesThePublishedCPUReserveNotTheDeclaredCores(t *testing.T) {
	// The whole point of the CPU price: a fleet selling 360 cores admits 180 sandboxes at
	// two declared cores each, and the sandboxes together were measured using a fraction
	// of one.
	control, ledger := newTestControl(t, &stubNodes{}, time.Now)
	spec := backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "img"},
		Resources:     backend.Resources{MemoryMB: 512, CPUCount: 2},
		ResourceClass: "rollout",
	}
	if _, err := control.Create(context.Background(), &v1.CreateRequest{Spec: specToProto(spec)}); err != nil {
		t.Fatalf("first create: %v", err)
	}
	if got := ledger.Report()["rollout"].Granted.CPUMillis; got != 2000 {
		t.Fatalf("charge without a published reserve: got %d millis, want the 2000 declared", got)
	}

	control.monitor.Report(placement.NodeView{
		NodeID: "n1", SeenAt: time.Now(),
		ClassReserveCPUMillis: map[string]int64{"rollout": 250},
	})

	if _, err := control.Create(context.Background(), &v1.CreateRequest{Spec: specToProto(spec)}); err != nil {
		t.Fatalf("second create: %v", err)
	}
	if got := ledger.Report()["rollout"].Granted.CPUMillis; got != 2250 {
		t.Errorf("charge with a published reserve: got %d millis, want 2000 + 250", got)
	}
}

func TestALargerPublishedReserveThanTheRequestIsNotCharged(t *testing.T) {
	// A reserve above what the caller asked for would charge more than the request, which
	// is the one direction the repricing must never move in.
	control, _ := newTestControl(t, &stubNodes{}, time.Now)
	control.monitor.Report(placement.NodeView{
		NodeID: "n1", SeenAt: time.Now(),
		ClassReserveMB:        map[string]int64{"rollout": 4096},
		ClassReserveCPUMillis: map[string]int64{"rollout": 8000},
	})
	spec := backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "img"},
		Resources:     backend.Resources{MemoryMB: 512, CPUCount: 2},
		ResourceClass: "rollout",
	}

	charged := control.charged(spec)

	if charged.Resources.MemoryMB != 512 || charged.Resources.CPUCount != 2 {
		t.Errorf("a reserve above the request must be ignored: %+v", charged.Resources)
	}
}

func TestTheLargestPublishedReserveWins(t *testing.T) {
	// Nodes state what a sandbox costs them. Charging the largest is the direction that
	// cannot over-admit a fleet whose nodes priced themselves differently.
	mon := monitor.New(time.Minute)
	mon.Report(placement.NodeView{NodeID: "n1", SeenAt: time.Now(), ClassReserveMB: map[string]int64{"rollout": 128}})
	mon.Report(placement.NodeView{NodeID: "n2", SeenAt: time.Now(), ClassReserveMB: map[string]int64{"rollout": 256}})

	reserve, known := mon.ClassReserveMB("rollout")

	if !known || reserve != 256 {
		t.Errorf("reserve: got %d (known=%v), want 256", reserve, known)
	}
	if _, known := mon.ClassReserveMB("grader"); known {
		t.Error("a class no node priced must report nothing")
	}
}

func TestTheReserveSurvivesTheNodePlaneWire(t *testing.T) {
	// The control reads the reserve off the node-plane report, so a field the wire drops is
	// a price the fleet never charges.
	original := placement.NodeView{
		NodeID: "n1", SeenAt: time.Now(),
		ClassReserveMB:        map[string]int64{"rollout": 512, "grader": 256},
		ClassReserveCPUMillis: map[string]int64{"rollout": 250, "grader": 500},
		ClassHeadroom:         map[string]placement.Headroom{"rollout": {MemoryMB: 4096}},
	}

	round := nodeViewFromWire(nodeViewToWire(original))

	if round.ClassReserveMB["rollout"] != 512 || round.ClassReserveMB["grader"] != 256 {
		t.Errorf("memory reserve after a round trip: got %v", round.ClassReserveMB)
	}
	if round.ClassReserveCPUMillis["rollout"] != 250 || round.ClassReserveCPUMillis["grader"] != 500 {
		t.Errorf("CPU reserve after a round trip: got %v", round.ClassReserveCPUMillis)
	}
}

// TestTheFleetRepricesWhatItAlreadyGranted is the failure from the production log.
//
// The nodes reprice their reservations as they measure. The fleet charged each grant what
// its class cost at admission and forgot it. When the class's price moved from the floor
// to tens of gigabytes, the fleet was still granting against the old number, every one of
// those grants was refused by the node that owned the pages, and the caller saw capacity
// exhaustion until the under-charged sandboxes happened to finish -- which is what turned
// into trajectories kept but ungraded and whole groups retried.
func TestTheFleetRepricesWhatItAlreadyGranted(t *testing.T) {
	control, ledger := newTestControl(t, &stubNodes{}, time.Now)
	spec := backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "img"},
		Resources:     backend.Resources{MemoryMB: 512, CPUCount: 2},
		ResourceClass: "rollout",
	}
	if _, err := control.Create(context.Background(), &v1.CreateRequest{Spec: specToProto(spec)}); err != nil {
		t.Fatalf("create: %v", err)
	}
	// Admitted before the class had any history: charged the caller's own size.
	if got := ledger.Report()["rollout"].Granted.MemoryMB; got != 512 {
		t.Fatalf("charge at admission: got %dMB, want 512MB", got)
	}

	// A node measures the class properly and publishes a higher price.
	control.monitor.Report(placement.NodeView{
		NodeID: "n1", SeenAt: time.Now(),
		ClassReserveMB:        map[string]int64{"rollout": 400},
		ClassReserveCPUMillis: map[string]int64{"rollout": 1500},
	})
	control.RepriceGrants()

	if got := ledger.Report()["rollout"].Granted.MemoryMB; got != 400 {
		t.Errorf("charge after the price moved: got %dMB, want 400MB", got)
	}
	if got := ledger.Report()["rollout"].Granted.CPUMillis; got != 1500 {
		t.Errorf("CPU charge after the price moved: got %d millis, want 1500", got)
	}

	// Repricing is idempotent: a second pass with no new price changes nothing.
	control.RepriceGrants()
	if got := ledger.Report()["rollout"].Granted.MemoryMB; got != 400 {
		t.Errorf("charge after a no-op reprice: got %dMB, want 400MB", got)
	}
}

func TestRepricingNeverChargesMoreThanTheCallerDeclared(t *testing.T) {
	// A price above the request is not a reason to charge more than the request: the
	// caller's declared size is a ceiling, and the published price only ever lowers it.
	control, ledger := newTestControl(t, &stubNodes{}, time.Now)
	spec := backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "img"},
		Resources:     backend.Resources{MemoryMB: 512, CPUCount: 2},
		ResourceClass: "rollout",
	}
	if _, err := control.Create(context.Background(), &v1.CreateRequest{Spec: specToProto(spec)}); err != nil {
		t.Fatalf("create: %v", err)
	}
	control.monitor.Report(placement.NodeView{
		NodeID: "n1", SeenAt: time.Now(),
		ClassReserveMB: map[string]int64{"rollout": 40000},
	})
	control.RepriceGrants()

	if got := ledger.Report()["rollout"].Granted.MemoryMB; got != 512 {
		t.Errorf("charge: got %dMB, want the 512MB the caller declared", got)
	}
}

// refusingNodes refuses every admission, with the reason a full node gives.
type refusingNodes struct {
	stubNodes
	admits int
}

func (r *refusingNodes) Admit(context.Context, string, backend.Spec) (string, []int32, string, error) {
	r.admits++
	return "", nil, "class_full", nil
}

// directStub is a backend this service schedules itself, so placement and node admission
// are both in the path. The provider-mode stub used elsewhere hands the whole decision to
// its own scheduler, which would leave nothing to assert against.
type directStub struct{ providerStub }

func (directStub) Mode() backend.SchedulingMode { return backend.SchedulingDirect }

// TestAFleetThatRefusedEverythingReportsExhaustion is the end-to-end form of the production
// failure. Two healthy nodes, both momentarily full: the caller must be told to wait, not
// that nothing can serve the request -- and the loop must not spend an attempt after the
// last node has refused, because that attempt is the one that loses the distinction.
func TestAFleetThatRefusedEverythingReportsExhaustion(t *testing.T) {
	nodes := &refusingNodes{}
	control := controlWithNodes(t, nodes)
	control.monitor.Report(twoNodeView("n1"))
	control.monitor.Report(twoNodeView("n2"))

	spec := backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "img"},
		Resources:     backend.Resources{MemoryMB: 512, CPUCount: 2},
		ResourceClass: "rollout",
	}
	_, err := control.Create(context.Background(), &v1.CreateRequest{Spec: specToProto(spec)})
	if err == nil {
		t.Fatal("expected the create to fail when every node refuses")
	}
	if got := status.Code(err); got != codes.ResourceExhausted {
		t.Errorf("error class: got %v, want ResourceExhausted so the caller knows to wait, not give up", got)
	}
	if nodes.admits != 2 {
		t.Errorf("admit attempts: got %d, want 2 -- one per live node, no attempt after the last refusal", nodes.admits)
	}
	if got := err.Error(); !strings.Contains(got, "refused") {
		t.Errorf("the message must say the nodes refused it: %s", got)
	}
}

// controlWithNodes builds a control plane whose backend this service schedules itself.
func controlWithNodes(t *testing.T, nodes NodeClient) *Control {
	t.Helper()
	fleet := monitor.New(2 * time.Minute)
	place, err := placement.New(placement.Config{
		NodeTTL: 2 * time.Minute, ReservationTTL: time.Minute, SweepInterval: 15 * time.Second,
	}, fleet)
	if err != nil {
		t.Fatalf("placement.New: %v", err)
	}
	ledger, err := quota.New(quota.Config{
		Total:    quota.Amount{MemoryMB: 4096, CPUMillis: 8000, Sandboxes: 1 << 20},
		Classes:  map[string]quota.ClassShare{"rollout": {Guaranteed: 0.5}, "grader": {Guaranteed: 0.5}},
		LeaseTTL: time.Hour,
	})
	if err != nil {
		t.Fatalf("quota.New: %v", err)
	}
	registry, err := backend.NewRegistry([]backend.Backend{directStub{}}, "")
	if err != nil {
		t.Fatalf("NewRegistry: %v", err)
	}
	control, err := NewControl(ControlConfig{Registry: registry, Ledger: ledger, Placement: place, Monitor: fleet, Nodes: nodes})
	if err != nil {
		t.Fatalf("NewControl: %v", err)
	}
	return control
}

func twoNodeView(id string) placement.NodeView {
	return placement.NodeView{
		NodeID: id, SeenAt: time.Now(),
		Envelope:      placement.Headroom{MemoryMB: 1000, CPUMillis: 20000},
		ClassHeadroom: map[string]placement.Headroom{"rollout": {MemoryMB: 1000, CPUMillis: 20000}},
		// The stub backend this test registers is named "stub"; a node advertising docker
		// would be filtered out before anything worth asserting happens.
		Backends: []placement.BackendCapability{{Name: "stub"}},
	}
}
