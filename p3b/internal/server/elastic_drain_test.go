package server

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	v1 "psrl.dev/sandboxd/api/v1"
)

// TestAWaveThatDrainsGivesTheWholeNodeBack is the production failure, in miniature.
//
// A step admits a wave, the controller reprices the class as the wave works, and the wave
// then finishes. The charge the node is left holding has to be zero, because what it is
// holding is nothing.
//
// It was not. Repricing used to be pushed into the ledger's own charge for the sandboxes
// already running: the class was raised by the price delta times however many sandboxes
// were running at the time, and lowered by the delta times however many were still there
// when it fell. A wave drains from the top, so the two never match, and a class with no
// sandboxes left is never repriced at all. The residue compounded across a run until the
// nodes were charging 271 and 302 cores against a 180 core envelope, at which point they
// refused every request -- grading included -- while measuring 42 MiB in use. Capacity
// that a node has charged to nothing cannot be given back by waiting, and a caller's
// retry budget only postpones the refusal it cannot clear.
func TestAWaveThatDrainsGivesTheWholeNodeBack(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	const wave = 6
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	// Twenty cores: a wave of six two-core sandboxes fits with room to spare, so
	// anything that refuses here is the ledger and not the envelope.
	agent := elasticNodeOver(t, domain, parent, 20000)

	handles := make([]*v1.SandboxHandle, 0, wave)
	ids := make([]string, 0, wave)
	for i := 0; i < wave; i++ {
		spec := backendSpec(t, "rollout", 2)
		admitted, err := agent.Admit(context.Background(), &v1.AdmitRequest{Spec: specToProto(spec)})
		if err != nil || !admitted.GetAdmitted() {
			t.Fatalf("admit %d of %d: %v (refusal %q)", i, wave, err, admitted.GetRefusal())
		}
		created, err := agent.CreateOn(context.Background(), &v1.CreateOnRequest{
			LeaseId: admitted.GetLeaseId(), Backend: "stub", Spec: specToProto(spec),
		})
		if err != nil {
			t.Fatalf("create %d: %v", i, err)
		}
		handles = append(handles, created.GetHandle())
		ids = append(ids, created.GetHandle().GetSandboxId())
	}

	// The wave works and gets expensive. Each tick raises the class's price while every
	// sandbox is still running, which is the direction the old ledger could not undo.
	usage := make([]int64, wave)
	for tick := 0; tick < 6; tick++ {
		for i := range ids {
			usage[i] += int64(400+tick*200) * 5 * int64(time.Second) / 1000 // 400..1400 millicores
			fakeSandbox(t, memoryDir, cpuDir, parent, ids[i], int64(512+128*tick)<<20, usage[i], 2)
		}
		agent.elastic.lastSample = time.Now().Add(-5 * time.Second)
		agent.ElasticTick(context.Background())
	}
	if len(agent.admission.Held()) == 0 {
		t.Fatal("the node should hold the wave's leases")
	}
	if got := agent.admission.Snapshot().Granted.CPUMillis; got == 0 {
		t.Fatal("the wave should be charged something")
	}

	// The wave finishes, one sandbox at a time, the way episodes end. Destroying the
	// container destroys its cgroup, so the domain has to go quiet too -- a test that
	// left the files behind would be measuring a node whose sandboxes never left.
	for i, handle := range handles {
		if _, err := agent.ReleaseOn(context.Background(), handle); err != nil {
			t.Fatalf("release %d: %v", i, err)
		}
		for _, dir := range []string{memoryDir, cpuDir} {
			if err := os.RemoveAll(filepath.Join(dir, parent, ids[i])); err != nil {
				t.Fatalf("remove cgroup %d: %v", i, err)
			}
		}
	}
	if held := agent.admission.Held(); len(held) != 0 {
		t.Fatalf("leases still open after the wave drained: %d", len(held))
	}

	// Now the domain is empty and the node has nothing. A tick has to agree.
	agent.elastic.lastSample = time.Now().Add(-5 * time.Second)
	report := agent.ElasticTick(context.Background())

	if report.ReservedMB != 0 {
		t.Errorf("memory still charged with nothing running: %d MiB", report.ReservedMB)
	}
	if report.ReservedCPUMillis != 0 {
		t.Errorf("CPU still charged with nothing running: %d millis", report.ReservedCPUMillis)
	}

	// And the node has to be able to admit again, which is what the caller actually
	// notices: a grader that arrives after the rollout wave must find room.
	spec := backendSpec(t, "grader", 2)
	admitted, err := agent.Admit(context.Background(), &v1.AdmitRequest{Spec: specToProto(spec)})
	if err != nil {
		t.Fatalf("grader admit: %v", err)
	}
	if !admitted.GetAdmitted() {
		t.Errorf("grader was refused with %q after the wave drained", admitted.GetRefusal())
	}
}

// TestAnEmptyNodeIsAccountedFor is the alarm the incident did not have.
//
// The node's charge cannot exceed its envelope, so it is not a capacity signal -- it is
// the ledger saying it has lost track of what it promised. Reporting it as a refusal
// would make a broken node indistinguishable from a full one, which is exactly the
// confusion that cost a run: the numbers were in the log for ten minutes and read as
// load. The audit has to say which it is.
func TestAnEmptyNodeIsAccountedFor(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, _, _ := fakeDomain(t, parent)
	agent := elasticNodeOver(t, domain, parent, 20000)

	granted, envelope, accounted := agent.LedgerAudit()
	if !accounted {
		t.Fatalf("an empty node is over its envelope: %+v over %+v", granted, envelope)
	}

	// Fill it. Everything admitted is charged, and nothing is charged that was not.
	for {
		spec := backendSpec(t, "rollout", 20)
		admitted, err := agent.Admit(context.Background(), &v1.AdmitRequest{Spec: specToProto(spec)})
		if err != nil {
			t.Fatalf("admit: %v", err)
		}
		if !admitted.GetAdmitted() {
			break
		}
	}
	granted, envelope, accounted = agent.LedgerAudit()
	if !accounted {
		t.Errorf("a node filled through admission is over its envelope: %+v over %+v", granted, envelope)
	}
}

// TestRepricingMovesThePriceNotTheCharge pins where the measurement is allowed to act.
//
// A price is a forecast about the next sandbox; a charge is a promise already made. The
// controller may move the first as often as it likes, because nothing has been promised
// yet. It must not move the second, because the return only knows the amount it was
// handed and the difference has nowhere to go.
func TestRepricingMovesThePriceNotTheCharge(t *testing.T) {
	const parent = "psrl-test-sandboxes"
	domain, memoryDir, cpuDir := fakeDomain(t, parent)
	agent := elasticNodeOver(t, domain, parent, 100000)
	id := admitSandbox(t, agent, backendSpec(t, "rollout", 2))
	fakeSandbox(t, memoryDir, cpuDir, parent, id, 512<<20, 0, 2)
	agent.ElasticTick(context.Background())

	charged := agent.admission.Snapshot().Granted.CPUMillis

	// Drive the class's price up and back down with the sandbox still running.
	settleTicks(t, agent, memoryDir, cpuDir, parent, id, 1200, 0, 8)
	high := agent.ElasticTick(context.Background()).ReservesCPUMillis["rollout"]
	settleTicks(t, agent, memoryDir, cpuDir, parent, id, 300, 0, 8)
	low := agent.ElasticTick(context.Background()).ReservesCPUMillis["rollout"]

	if high == low {
		t.Fatalf("the price did not move: %d then %d", high, low)
	}
	if got := agent.admission.Snapshot().Granted.CPUMillis; got != charged {
		t.Errorf("charge followed the price: got %d, want the %d this sandbox was admitted at", got, charged)
	}
}
