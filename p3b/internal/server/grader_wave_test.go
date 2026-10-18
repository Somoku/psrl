package server

import (
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/node"

	v1 "psrl.dev/sandboxd/api/v1"
)

// twoWaveNode builds a node whose envelope is split evenly between rollout and grading,
// which is how the deployment is configured.
func twoWaveNode(t *testing.T, envelopeMB int64) *Node {
	t.Helper()
	gate, err := node.NewAdmission(node.Config{
		// Twenty cores, so ten two-core sandboxes fit and the memory envelope is the only
		// dimension under test.
		Envelope: node.Resources{MemoryMB: envelopeMB, CPUMillis: 20000},
		Classes: map[string]node.ClassShare{
			"rollout": {Guaranteed: 0.45},
			"grader":  {Guaranteed: 0.45},
		},
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
	})
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	return agent
}

func specOf(class string, memoryMB int64) backend.Spec {
	spec := backendSpec(nil, class, 2)
	spec.Resources.MemoryMB = memoryMB
	spec.Resources.CPUCount = 2
	return spec
}

// TestARefusedRequestHoldsItsClassGuarantee is the mechanism that keeps the grading wave
// admittable.
//
// A guarantee is only real if it is held back from the other classes while its own class is
// waiting, and the node is the only place that can see a class waiting. Without it the first
// wave takes the whole envelope and the second wave -- grading -- never gets in, which at
// this deployment surfaced as trajectories kept but ungraded, i.e. reward signal lost to an
// infrastructure fault rather than to the policy.
func TestARefusedRequestHoldsItsClassGuarantee(t *testing.T) {
	agent := twoWaveNode(t, 1000)

	// The first wave: rollout takes the whole machine, the way a step starts.
	var leases []string
	for i := 0; i < 10; i++ {
		admitted, err := agent.Admit(nil, &v1.AdmitRequest{Spec: specToProto(specOf("rollout", 100))})
		if err != nil {
			t.Fatalf("rollout admit: %v", err)
		}
		if !admitted.GetAdmitted() {
			t.Fatalf("rollout %d was refused with %q: the envelope should hold ten of 100MB",
				i, admitted.GetRefusal())
		}
		leases = append(leases, admitted.GetLeaseId())
	}

	// The second wave arrives and is refused. That refusal is the node learning that
	// grading is waiting, and it is the only place the node can learn it. Several graders
	// queue at once, which is what a finished batch looks like.
	for i := 0; i < 3; i++ {
		refused, err := agent.Admit(nil, &v1.AdmitRequest{Spec: specToProto(specOf("grader", 100))})
		if err != nil {
			t.Fatalf("grader admit: %v", err)
		}
		if refused.GetAdmitted() {
			t.Fatalf("grader %d was admitted while rollout holds the envelope", i)
		}
	}

	// Episodes finish and half the rollout sandboxes go back.
	for _, lease := range leases[:5] {
		agent.admission.Return(node.Grant{
			LeaseID:    lease,
			Class:      "rollout",
			Resources:  node.Resources{MemoryMB: 100, CPUMillis: 2000},
			GPUIndices: nil,
		})
	}

	// Grading gets in on the capacity that came back.
	grader, err := agent.Admit(nil, &v1.AdmitRequest{Spec: specToProto(specOf("grader", 100))})
	if err != nil {
		t.Fatalf("grader retry: %v", err)
	}
	if !grader.GetAdmitted() {
		t.Errorf("grader was refused with %q after 500MB came back", grader.GetRefusal())
	}

	// And rollout may not take the rest: grading's share is held for the graders that
	// are still waiting. 400MB does not fit in what is left of a 1000MB envelope once
	// 600MB is used and 450MB is reserved.
	big, err := agent.Admit(nil, &v1.AdmitRequest{Spec: specToProto(specOf("rollout", 400))})
	if err != nil {
		t.Fatalf("big rollout admit: %v", err)
	}
	if big.GetAdmitted() {
		t.Error("rollout took 400MB while grading had a waiting claim on its share")
	}
}

func TestWithoutRefusedDemandTheFirstWaveTakesEverything(t *testing.T) {
	// The counterfactual, so the assertion above is about the mechanism and not about the
	// envelope: with no record of grading waiting, rollout is free to take the machine.
	agent := twoWaveNode(t, 1000)
	gate := agent.admission
	// Not a single refusal, so nothing is held back.
	if got := gate.Headroom("rollout").MemoryMB; got != 1000 {
		t.Fatalf("headroom with no waiting class: got %d, want the whole envelope", got)
	}
	admitted, err := agent.Admit(nil, &v1.AdmitRequest{Spec: specToProto(specOf("rollout", 1000))})
	if err != nil {
		t.Fatalf("rollout admit: %v", err)
	}
	if !admitted.GetAdmitted() {
		t.Errorf("rollout was refused with %q although nothing else was waiting", admitted.GetRefusal())
	}
}

func TestADemandThatWentQuietGivesItsShareBack(t *testing.T) {
	// The claim is bounded: a caller that has gone away is not waiting, and holding its
	// share idle starves every other class for nothing.
	agent := twoWaveNode(t, 1000)
	agent.admission.SetDemandTTL(time.Second)

	if _, err := agent.Admit(nil, &v1.AdmitRequest{Spec: specToProto(specOf("grader", 2000))}); err != nil {
		t.Fatalf("grader admit: %v", err)
	}
	before := agent.admission.Headroom("rollout")

	advanceDemand(t, agent, 2*time.Second)
	after := agent.admission.Headroom("rollout")

	if after.MemoryMB <= before.MemoryMB {
		t.Errorf("headroom must grow once the demand lapses: %d then %d", before.MemoryMB, after.MemoryMB)
	}
}

// advanceDemand moves the node's clock forward by walking the queued stamps, since the gate
// takes its time from a function a test cannot reach from here.
func advanceDemand(t *testing.T, agent *Node, by time.Duration) {
	t.Helper()
	clock := time.Now()
	agent.admission.SetClock(func() time.Time { return clock.Add(by) })
	agent.admission.ExpireDemand()
	agent.admission.SetClock(func() time.Time { return clock })
}

// TestThrottledSandboxesDoNotLookIdle guards the signal the price depends on: without it a
// node can believe it has headroom while every sandbox it admitted is being throttled.
func TestThrottledSandboxesDoNotLookIdle(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := elasticNodeOver(t, domain, parent, 100000)
	id := admitSandbox(t, agent, backendSpec(t, "rollout", 2))
	fakeSandbox(t, memoryDir, cpuDir, parent, id, 512<<20, 0, 2)
	agent.ElasticTick(nil)

	agent.elastic.lastSample = time.Now().Add(-5 * time.Second)
	fakeSandbox(t, memoryDir, cpuDir, parent, id, 512<<20, 5*time.Second.Nanoseconds(), 2)
	setThrottled(t, cpuDir, parent, id, 5*time.Second.Nanoseconds())
	report := agent.ElasticTick(nil)

	if report.ThrottledPct < 0.9 {
		t.Errorf("throttled share: got %.2f, want about 1.0", report.ThrottledPct)
	}
}
