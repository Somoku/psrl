package server

import (
	"context"
	"testing"
	"time"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/monitor"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/placement"
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
			Enabled:   true,
			Footprint: node.FootprintConfig{MinReserveMB: reserveMB, GrowthMargin: 0.25},
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

	if got := view.ClassReserve["rollout"]; got != 512 {
		t.Errorf("published reserve: got %dMB, want the 512MiB minimum", got)
	}
}

func TestThePublishedReserveFollowsMeasurement(t *testing.T) {
	agent := elasticNode(t, 512)

	agent.footprintFor("rollout").Observe([]int64{1000})

	if got := agent.View().ClassReserve["rollout"]; got != 1250 {
		t.Errorf("published reserve: got %dMB, want 1250 (1000 x 1.25)", got)
	}
}

func TestANodeWithoutElasticPricingPublishesNothing(t *testing.T) {
	// Nothing published is what keeps a deployment that has not opted in on the old
	// behaviour: the caller charges the request's maximum.
	agent := elasticNode(t, 512)
	agent.elastic = nil

	if got := agent.View().ClassReserve; got != nil {
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
		ClassReserve: map[string]int64{"rollout": 128},
	})

	if _, err := control.Create(context.Background(), &v1.CreateRequest{Spec: specToProto(spec)}); err != nil {
		t.Fatalf("second create: %v", err)
	}
	if got := ledger.Report()["rollout"].Granted.MemoryMB; got != 512+128 {
		t.Errorf("charge with a published reserve: got %dMB, want 512 + 128", got)
	}
}

func TestTheChargedSpecCarriesTheReserveOnlyForMemory(t *testing.T) {
	// CPU, disk and devices are stated requirements rather than measured costs, so only
	// memory is repriced.
	control, _ := newTestControl(t, &stubNodes{}, time.Now)
	control.monitor.Report(placement.NodeView{
		NodeID: "n1", SeenAt: time.Now(),
		ClassReserve: map[string]int64{"rollout": 128},
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
	if charged.Resources.CPUCount != 2 || charged.Resources.DiskMB != 4096 {
		t.Errorf("the other dimensions must be untouched: %+v", charged.Resources)
	}
	if spec.Resources.MemoryMB != 512 {
		t.Error("the caller's spec must not be mutated")
	}
}

func TestTheLargestPublishedReserveWins(t *testing.T) {
	// Nodes state what a sandbox costs them. Charging the largest is the direction that
	// cannot over-admit a fleet whose nodes priced themselves differently.
	mon := monitor.New(time.Minute)
	mon.Report(placement.NodeView{NodeID: "n1", SeenAt: time.Now(), ClassReserve: map[string]int64{"rollout": 128}})
	mon.Report(placement.NodeView{NodeID: "n2", SeenAt: time.Now(), ClassReserve: map[string]int64{"rollout": 256}})

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
		ClassReserve:  map[string]int64{"rollout": 512, "grader": 256},
		ClassHeadroom: map[string]placement.Headroom{"rollout": {MemoryMB: 4096}},
	}

	round := nodeViewFromWire(nodeViewToWire(original))

	if round.ClassReserve["rollout"] != 512 || round.ClassReserve["grader"] != 256 {
		t.Errorf("reserve after a round trip: got %v", round.ClassReserve)
	}
}
