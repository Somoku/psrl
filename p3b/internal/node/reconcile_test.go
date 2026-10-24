package node

import (
	"context"
	"fmt"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// liveSandbox is one container a backend reports it already holds.
func liveSandbox(id, owner, class string, memoryMB, cpuMillis int64, devices ...int32) backend.Reconciled {
	return backend.Reconciled{
		Handle:     backend.Handle{Backend: "fake", SandboxID: id, NodeID: "node-a"},
		Owner:      owner,
		Class:      class,
		ExecMode:   "persistent",
		Resources:  backend.Resources{MemoryMB: memoryMB, CPUCount: float64(cpuMillis) / 1000},
		GPUIndices: devices,
		Lifetime:   testLifetime,
		CreatedAt:  time.Now(),
		Running:    true,
	}
}

func TestReconcileChargesMemoryCPUAndDevices(t *testing.T) {
	// The charge must match what the create charged, or the node's own
	// accounting disagrees with what the machine is holding.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	b := &fakeBackend{found: []backend.Reconciled{
		liveSandbox("sb-live", "worker", "rollout", 200, 2000, 1, 3),
	}}

	report, err := life.Reconcile(context.Background(), []backend.Backend{b})
	if err != nil {
		t.Fatalf("reconcile: %v", err)
	}

	if report.Adopted != 1 || report.Reclaimed != 0 || report.Failures != 0 {
		t.Fatalf("reconcile report %+v, want one adoption", report)
	}
	snapshot := gate.Snapshot()
	if snapshot.Granted.MemoryMB != 200 || snapshot.Granted.CPUMillis != 2000 {
		t.Fatalf("granted %+v, want the container's own 200MB/2000m", snapshot.Granted)
	}
	if snapshot.FreeGPUs != 2 {
		t.Fatalf("free devices %d, want 2 of 4 (1 and 3 taken)", snapshot.FreeGPUs)
	}
	if resident := life.Snapshot().Resident; resident != 1 {
		t.Fatalf("resident %d, want 1", resident)
	}
}

func TestReconcileReclaimsADeadContainerAndAdoptsALiveOne(t *testing.T) {
	// An exited container holds no memory, so charging it would waste the
	// envelope, but leaving it behind is litter. A running one is the opposite:
	// it must be charged and owned.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	dead := liveSandbox("sb-dead", "worker", "rollout", 100, 1000)
	dead.Running = false
	b := &fakeBackend{found: []backend.Reconciled{
		liveSandbox("sb-live", "worker", "rollout", 200, 1000),
		dead,
	}}

	report, err := life.Reconcile(context.Background(), []backend.Backend{b})
	if err != nil {
		t.Fatalf("reconcile: %v", err)
	}

	if report.Adopted != 1 || report.Reclaimed != 1 {
		t.Fatalf("report %+v, want one adoption and one reclaim", report)
	}
	if released := b.releasedIDs(); len(released) != 1 || released[0] != "sb-dead" {
		t.Fatalf("released %v, want only the exited container", released)
	}
	if snapshot := gate.Snapshot(); snapshot.Granted.MemoryMB != 200 {
		t.Fatalf("granted %dMB, want only the live container's 200MB", snapshot.Granted.MemoryMB)
	}
}

func TestAContainerFoundAtStartupIsReclaimedWhenItsOwnerGoesSilent(t *testing.T) {
	// Adoption puts the container under the sweep, so the node can reclaim work
	// whose caller died before the restart, without waiting for its lifetime.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	b := &fakeBackend{found: []backend.Reconciled{
		liveSandbox("sb-orphan", "gone", "rollout", 300, 1000),
	}}
	if _, err := life.Reconcile(context.Background(), []backend.Backend{b}); err != nil {
		t.Fatalf("reconcile: %v", err)
	}

	life.SetOwnerLiveness(func(string) bool { return false })
	report := life.Sweep(context.Background())
	life.WaitCleanup()

	if len(report.ReclaimedOrphan) != 1 {
		t.Fatalf("report %+v, want the adopted sandbox reclaimed as an orphan", report)
	}
	if snapshot := gate.Snapshot(); snapshot.Granted.MemoryMB != 0 {
		t.Fatalf("granted %dMB, want the reclaim to return all of it", snapshot.Granted.MemoryMB)
	}
}

func TestReconcileRefusesADeviceTwoRecordsClaim(t *testing.T) {
	// Two containers on one device is a fault the node must not paper over: it
	// would let two sandboxes each believe they own the device.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	b := &fakeBackend{found: []backend.Reconciled{
		liveSandbox("sb-one", "worker", "rollout", 100, 1000, 2),
		liveSandbox("sb-two", "worker", "rollout", 100, 1000, 2),
	}}

	if _, err := life.Reconcile(context.Background(), []backend.Backend{b}); err == nil {
		t.Fatal("a doubled device claim must be refused rather than charged twice")
	}
}

func TestReconcileFailsTheStartupWhenTheRuntimeCannotBeEnumerated(t *testing.T) {
	// A node that cannot list its containers does not know what it holds, and
	// starting to admit anyway is how one restart becomes a host OOM.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	b := &fakeBackend{reconcileErr: fmt.Errorf("daemon unreachable")}

	if _, err := life.Reconcile(context.Background(), []backend.Backend{b}); err == nil {
		t.Fatal("an unreachable daemon must fail the startup")
	}
}

func TestReconcileLeavesABackendWithoutADaemonAlone(t *testing.T) {
	// A provider backend's objects live on an external platform, so there is
	// nothing on this node to enumerate and no error to raise.
	gate := mustGate(t, envelope())
	life, _ := NewLifecycle("node-a", gate, time.Second)
	report, err := life.Reconcile(context.Background(), []backend.Backend{providerish{}})
	if err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	if report != (ReconcileReport{}) {
		t.Fatalf("report %+v, want nothing found and nothing done", report)
	}
}

// providerish is a backend that cannot enumerate a local runtime, standing in
// for one whose objects live on an external platform.
type providerish struct{}

func (providerish) Name() string                       { return "providerish" }
func (providerish) Mode() backend.SchedulingMode       { return backend.SchedulingProvider }
func (providerish) Capabilities() backend.Capabilities { return backend.Capabilities{} }
func (providerish) Create(context.Context, string, backend.Spec, string) (backend.Created, error) {
	return backend.Created{}, nil
}
func (providerish) Release(context.Context, backend.Handle) error { return nil }
func (providerish) Status(context.Context, backend.Handle) (string, error) {
	return "running", nil
}
