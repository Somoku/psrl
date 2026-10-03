package cubesandbox

import (
	"context"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"

	sbbackend "psrl.dev/sandboxd/internal/backend"
	cubebox "psrl.dev/sandboxd/internal/backend/cubesandbox/cubeletpb/services/cubebox/v1"
	errorcode "psrl.dev/sandboxd/internal/backend/cubesandbox/cubeletpb/services/errorcode/v1"
)

// Direct mode drives a Cubelet over gRPC, so these tests run real gRPC servers.
// A stub that answered HTTP would pass while the production path failed on the
// wire, which is the failure that made this suite worth rewriting.

// -- construction ---------------------------------------------------------------

func TestDirectModeNeedsNodeAddresses(t *testing.T) {
	if _, err := New(Config{}, sbbackend.SchedulingDirect); err == nil {
		t.Fatal("direct mode must refuse an empty node list: there is no Cubelet to call")
	}
}

func TestDirectModeRejectsANodeWithoutAnAddress(t *testing.T) {
	_, err := New(Config{
		Nodes: []NodeAddress{{NodeID: "n1", Address: ""}},
	}, sbbackend.SchedulingDirect)
	if err == nil {
		t.Fatal("a node needs both an id and an address")
	}
}

func TestDirectModeReportsItsMode(t *testing.T) {
	b := directBackend(t, map[string]*fakeCubelet{"n1": newFakeCubelet(t)})
	if b.Mode() != sbbackend.SchedulingDirect {
		t.Fatalf("mode: got %q, want direct", b.Mode())
	}
}

// -- capabilities ---------------------------------------------------------------

func TestDirectModeDropsCubeMasterFeatures(t *testing.T) {
	// A warm pool and a template build live on CubeMaster, which the direct path
	// bypasses. Freeze is dropped too: CubeboxMgr has no pause RPC, so declaring it
	// would let the reclaimer pause-on-idle and believe compute was released.
	b := directBackend(t, map[string]*fakeCubelet{"n1": newFakeCubelet(t)})
	declared := map[string]bool{}
	for _, feature := range b.Capabilities().Features {
		declared[feature] = true
	}
	for _, absent := range []string{"warm_pool", "template_build", "volume", "egress_policy", "freeze"} {
		if declared[absent] {
			t.Errorf("direct mode must not declare %q", absent)
		}
	}
}

func TestDirectModeDeclaresNoPauseMode(t *testing.T) {
	b := directBackend(t, map[string]*fakeCubelet{"n1": newFakeCubelet(t)})
	if modes := b.Capabilities().PauseModes; len(modes) != 0 {
		t.Fatalf("a Cubelet cannot pause, so no pause mode may be declared, got %v", modes)
	}
}

func TestDirectModeStillDeclaresWhatACubeletServes(t *testing.T) {
	b := directBackend(t, map[string]*fakeCubelet{"n1": newFakeCubelet(t)})
	declared := map[string]bool{}
	for _, feature := range b.Capabilities().Features {
		declared[feature] = true
	}
	for _, wanted := range []string{"filesystem_snapshot", "restore", "image_on_demand"} {
		if !declared[wanted] {
			t.Errorf("direct mode should declare %q: a Cubelet serves it", wanted)
		}
	}
}

// -- placement ------------------------------------------------------------------

func TestDirectModeListsItsNodesInAStableOrder(t *testing.T) {
	b := directBackend(t, map[string]*fakeCubelet{
		"n2": newFakeCubelet(t),
		"n1": newFakeCubelet(t),
	})
	ids, err := b.Nodes(context.Background())
	if err != nil {
		t.Fatalf("nodes: %v", err)
	}
	// Sorted, so placement does not read Go's map iteration order.
	if len(ids) != 2 || ids[0] != "n1" || ids[1] != "n2" {
		t.Fatalf("nodes: got %v, want [n1 n2]", ids)
	}
}

func TestDirectModeCreateReachesTheChosenCubelet(t *testing.T) {
	n1, n2 := newFakeCubelet(t), newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"n1": n1, "n2": n2})

	created, err := b.Create(context.Background(), "n2", testSpec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if n1.creates() != 0 {
		t.Error("a create naming n2 must not reach n1")
	}
	if n2.creates() != 1 {
		t.Errorf("n2 should have seen one create, saw %d", n2.creates())
	}
	// The node is recorded so release, status, and snapshot route back to it.
	if created.Handle.NodeID != "n2" {
		t.Errorf("handle node: got %q, want n2", created.Handle.NodeID)
	}
}

func TestDirectModeRefusesAnUnconfiguredNode(t *testing.T) {
	b := directBackend(t, map[string]*fakeCubelet{"n1": newFakeCubelet(t)})
	if _, err := b.Create(context.Background(), "nowhere", testSpec(), ""); err == nil {
		t.Fatal("a create naming an unconfigured Cubelet must be refused")
	}
}

// -- the create request ---------------------------------------------------------

func TestACreateCarriesTheImageAndResources(t *testing.T) {
	// Cubelet states resources as Kubernetes quantity strings rather than
	// integers, so the conversion is part of the contract.
	node := newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"n1": node})
	if _, err := b.Create(context.Background(), "n1", testSpec(), ""); err != nil {
		t.Fatalf("create: %v", err)
	}
	request := node.lastCreate()
	if len(request.GetContainers()) != 1 {
		t.Fatalf("expected one container, got %d", len(request.GetContainers()))
	}
	container := request.GetContainers()[0]
	if got := container.GetImage().GetImage(); got != "tmpl-python-base" {
		t.Errorf("image: got %q, want tmpl-python-base", got)
	}
	if got := container.GetResources().GetCpu(); got != "2000m" {
		t.Errorf("cpu: got %q, want 2000m", got)
	}
	if got := container.GetResources().GetMem(); got != "2048Mi" {
		t.Errorf("mem: got %q, want 2048Mi", got)
	}
}

func TestACreateSetsTheLimitToTheRequest(t *testing.T) {
	// One stated number means one number: a burstable gap nobody asked for would
	// let a sandbox exceed what admission charged it.
	node := newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"n1": node})
	if _, err := b.Create(context.Background(), "n1", testSpec(), ""); err != nil {
		t.Fatalf("create: %v", err)
	}
	resources := node.lastCreate().GetContainers()[0].GetResources()
	if resources.GetCpu() != resources.GetCpuLimit() {
		t.Errorf("cpu request %q and limit %q should match", resources.GetCpu(), resources.GetCpuLimit())
	}
	if resources.GetMem() != resources.GetMemLimit() {
		t.Errorf("mem request %q and limit %q should match", resources.GetMem(), resources.GetMemLimit())
	}
}

func TestACreateCarriesTheOwnerForSweeps(t *testing.T) {
	// A reclaim must be able to tell this service's sandboxes from another's.
	node := newFakeCubelet(t)
	b, err := New(Config{
		Nodes:   []NodeAddress{{NodeID: "n1", Address: node.address}},
		OwnerID: "sandboxd-node-1",
	}, sbbackend.SchedulingDirect)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	t.Cleanup(func() { b.direct.Close() })
	if _, err := b.Create(context.Background(), "n1", testSpec(), ""); err != nil {
		t.Fatalf("create: %v", err)
	}
	if got := node.lastCreate().GetAnnotations()[annotationOwner]; got != "sandboxd-node-1" {
		t.Errorf("owner annotation: got %q, want sandboxd-node-1", got)
	}
}

func TestACreateWithNoResourcesSendsNone(t *testing.T) {
	// A zero quantity is rejected by Cubelet, so an unstated resource sends
	// nothing rather than "0m".
	node := newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"n1": node})
	spec := testSpec()
	spec.Resources = sbbackend.Resources{}
	if _, err := b.Create(context.Background(), "n1", spec, ""); err != nil {
		t.Fatalf("create: %v", err)
	}
	if resources := node.lastCreate().GetContainers()[0].GetResources(); resources != nil {
		t.Errorf("a spec stating no resources must send none, got %v", resources)
	}
}

// -- failures the Cubelet reports in its reply ----------------------------------

func TestACubeletRefusalIsAnError(t *testing.T) {
	// The RPC succeeds and the refusal is inside the reply, so a backend that only
	// checked the transport error would read a refusal as a live sandbox.
	node := newFakeCubelet(t)
	node.failCreateWith(errorcode.ErrorCode_CreateContainerFailed, "no capacity")
	b := directBackend(t, map[string]*fakeCubelet{"n1": node})
	if _, err := b.Create(context.Background(), "n1", testSpec(), ""); err == nil {
		t.Fatal("a refusal carried in the reply must be an error")
	}
}

func TestACreateWithNoSandboxIDIsAnError(t *testing.T) {
	node := newFakeCubelet(t)
	node.returnEmptyID()
	b := directBackend(t, map[string]*fakeCubelet{"n1": node})
	if _, err := b.Create(context.Background(), "n1", testSpec(), ""); err == nil {
		t.Fatal("a create that returns no id must be an error: there is nothing to release")
	}
}

// -- status ---------------------------------------------------------------------

func TestAnAbsentSandboxReadsAsTerminated(t *testing.T) {
	// Cubelet's List carries no status field: presence is the signal, and absence
	// from its own node's list is the one unambiguous way to know it is gone.
	node := newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"n1": node})
	handle := sbbackend.Handle{Backend: "cubesandbox", SandboxID: "sb-gone", NodeID: "n1"}
	state, err := b.Status(context.Background(), handle)
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if state != "terminated" {
		t.Errorf("state: got %q, want terminated", state)
	}
}

func TestALiveSandboxReadsAsRunning(t *testing.T) {
	node := newFakeCubelet(t)
	node.addSandbox("sb-1", 0)
	b := directBackend(t, map[string]*fakeCubelet{"n1": node})
	handle := sbbackend.Handle{Backend: "cubesandbox", SandboxID: "sb-1", NodeID: "n1"}
	state, err := b.Status(context.Background(), handle)
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if state != "running" {
		t.Errorf("state: got %q, want running", state)
	}
}

func TestAPausedSandboxIsReadFromPausedAt(t *testing.T) {
	// paused_at is the only state marker the record carries.
	node := newFakeCubelet(t)
	node.addSandbox("sb-1", time.Now().Unix())
	b := directBackend(t, map[string]*fakeCubelet{"n1": node})
	handle := sbbackend.Handle{Backend: "cubesandbox", SandboxID: "sb-1", NodeID: "n1"}
	state, err := b.Status(context.Background(), handle)
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if state != "paused" {
		t.Errorf("state: got %q, want paused", state)
	}
}

// -- operations a Cubelet cannot serve ------------------------------------------

func TestDirectModeRefusesAPause(t *testing.T) {
	// CubeboxMgr has no pause RPC. Returning nil would let the reclaimer believe
	// it had released the sandbox's compute when nothing happened.
	b := directBackend(t, map[string]*fakeCubelet{"n1": newFakeCubelet(t)})
	handle := sbbackend.Handle{Backend: "cubesandbox", SandboxID: "sb-1", NodeID: "n1"}
	err := b.Pause(context.Background(), handle, "freeze")
	if err == nil {
		t.Fatal("a directly driven Cubelet cannot pause, and must say so")
	}
	if !strings.Contains(err.Error(), "CubeboxMgr") {
		t.Errorf("the error should name what is missing, got: %v", err)
	}
}

func TestDirectModeRefusesAResume(t *testing.T) {
	b := directBackend(t, map[string]*fakeCubelet{"n1": newFakeCubelet(t)})
	handle := sbbackend.Handle{Backend: "cubesandbox", SandboxID: "sb-1", NodeID: "n1"}
	if err := b.Resume(context.Background(), handle); err == nil {
		t.Fatal("nothing was paused, so a resume must be refused rather than reported done")
	}
}

func TestDirectModeRefusesAFullStateSnapshot(t *testing.T) {
	b := directBackend(t, map[string]*fakeCubelet{"n1": newFakeCubelet(t)})
	handle := sbbackend.Handle{Backend: "cubesandbox", SandboxID: "sb-1", NodeID: "n1"}
	if _, err := b.Snapshot(context.Background(), handle, "full_state"); err == nil {
		t.Fatal("a commit captures the filesystem, so full_state must be refused")
	}
}

// -- release and snapshot -------------------------------------------------------

func TestAReleaseReachesTheHoldingNode(t *testing.T) {
	n1, n2 := newFakeCubelet(t), newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"n1": n1, "n2": n2})
	handle := sbbackend.Handle{Backend: "cubesandbox", SandboxID: "sb-1", NodeID: "n2"}
	if err := b.Release(context.Background(), handle); err != nil {
		t.Fatalf("release: %v", err)
	}
	if n1.destroys() != 0 {
		t.Error("a release on n2 must not reach n1")
	}
	if n2.destroys() != 1 {
		t.Errorf("n2 should have seen one destroy, saw %d", n2.destroys())
	}
}

func TestASnapshotCommitsOnTheHoldingNode(t *testing.T) {
	node := newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"n1": node})
	handle := sbbackend.Handle{Backend: "cubesandbox", SandboxID: "sb-1", NodeID: "n1"}
	snapshotID, err := b.Snapshot(context.Background(), handle, "filesystem")
	if err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	if snapshotID == "" {
		t.Fatal("a snapshot must return an id a restore can name")
	}
	if node.commits() != 1 {
		t.Errorf("expected one commit, saw %d", node.commits())
	}
}

// -- preflight ------------------------------------------------------------------

func TestDirectModePreflightProbesEveryCubelet(t *testing.T) {
	n1, n2 := newFakeCubelet(t), newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"n1": n1, "n2": n2})
	if err := b.Preflight(context.Background()); err != nil {
		t.Fatalf("preflight: %v", err)
	}
	if n1.lists() == 0 || n2.lists() == 0 {
		t.Errorf("preflight must reach every Cubelet, saw n1=%d n2=%d", n1.lists(), n2.lists())
	}
}

func TestDirectModePreflightFailsWhenOneCubeletIsAbsent(t *testing.T) {
	// One live node is not enough: a fleet missing a node silently loses that
	// node's whole capacity, so preflight refuses rather than degrading.
	live := newFakeCubelet(t)
	b, err := New(Config{
		Nodes: []NodeAddress{
			{NodeID: "up", Address: live.address},
			{NodeID: "down", Address: "127.0.0.1:1"},
		},
		RequestTimeout: 2 * time.Second,
	}, sbbackend.SchedulingDirect)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	t.Cleanup(func() { b.direct.Close() })
	if err := b.Preflight(context.Background()); err == nil {
		t.Fatal("preflight must refuse a fleet with an unreachable Cubelet")
	}
}

// -- address handling -----------------------------------------------------------

func TestASchemeIsStrippedFromACubeletAddress(t *testing.T) {
	// gRPC dials host:port. An address carrying a scheme is a configuration people
	// write by habit, and it must not become an unroutable target.
	for _, address := range []string{
		"grpc://10.0.0.1:8089", "http://10.0.0.1:8089", "tcp://10.0.0.1:8089", "10.0.0.1:8089",
	} {
		if got := grpcTarget(address); got != "10.0.0.1:8089" {
			t.Errorf("grpcTarget(%q): got %q, want 10.0.0.1:8089", address, got)
		}
	}
}

// -- a fake Cubelet -------------------------------------------------------------

// fakeCubelet is a real gRPC server implementing CubeboxMgr, so the adapter is
// exercised over the wire it uses in production.
type fakeCubelet struct {
	cubebox.UnimplementedCubeboxMgrServer

	address string

	mu           sync.Mutex
	createCount  int
	destroyCount int
	listCount    int
	commitCount  int
	last         *cubebox.RunCubeSandboxRequest
	sandboxes    map[string]int64 // id -> pausedAt
	failCode     errorcode.ErrorCode
	failMessage  string
	emptyID      bool

	// block, inFlight and peak let a concurrency test observe how many creates
	// the backend allowed to reach the node at once. Without holding creates open
	// the high-water mark is always one and the bound is untested.
	block    chan struct{}
	inFlight int
	peak     int
}

// blockCreates holds every Create open until releaseCreates, so a test can read
// the number in flight at the high-water mark.
func (f *fakeCubelet) blockCreates() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.block = make(chan struct{})
}

func (f *fakeCubelet) releaseCreates() {
	f.mu.Lock()
	gate := f.block
	f.block = nil
	f.mu.Unlock()
	if gate != nil {
		close(gate)
	}
}

func (f *fakeCubelet) inFlightPeak() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.peak
}

func newFakeCubelet(t *testing.T) *fakeCubelet {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	fake := &fakeCubelet{address: listener.Addr().String(), sandboxes: map[string]int64{}}
	server := grpc.NewServer()
	cubebox.RegisterCubeboxMgrServer(server, fake)
	go func() { _ = server.Serve(listener) }()
	t.Cleanup(server.Stop)
	return fake
}

func (f *fakeCubelet) failCreateWith(code errorcode.ErrorCode, message string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.failCode, f.failMessage = code, message
}

func (f *fakeCubelet) returnEmptyID() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.emptyID = true
}

func (f *fakeCubelet) addSandbox(id string, pausedAt int64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.sandboxes[id] = pausedAt
}

func (f *fakeCubelet) creates() int  { f.mu.Lock(); defer f.mu.Unlock(); return f.createCount }
func (f *fakeCubelet) destroys() int { f.mu.Lock(); defer f.mu.Unlock(); return f.destroyCount }
func (f *fakeCubelet) lists() int    { f.mu.Lock(); defer f.mu.Unlock(); return f.listCount }
func (f *fakeCubelet) commits() int  { f.mu.Lock(); defer f.mu.Unlock(); return f.commitCount }

func (f *fakeCubelet) lastCreate() *cubebox.RunCubeSandboxRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.last
}

func (f *fakeCubelet) Create(
	_ context.Context, in *cubebox.RunCubeSandboxRequest,
) (*cubebox.RunCubeSandboxResponse, error) {
	f.mu.Lock()
	f.createCount++
	f.last = in
	f.inFlight++
	if f.inFlight > f.peak {
		f.peak = f.inFlight
	}
	gate := f.block
	f.mu.Unlock()
	if gate != nil {
		<-gate
	}
	f.mu.Lock()
	f.inFlight--
	defer f.mu.Unlock()
	// The zero value is OK, so an unset failCode means "do not fail".
	if f.failCode != errorcode.ErrorCode_OK {
		return &cubebox.RunCubeSandboxResponse{
			Ret: &errorcode.Ret{RetCode: f.failCode, RetMsg: f.failMessage},
		}, nil
	}
	if f.emptyID {
		return &cubebox.RunCubeSandboxResponse{}, nil
	}
	return &cubebox.RunCubeSandboxResponse{SandboxID: "sb-created"}, nil
}

func (f *fakeCubelet) Destroy(
	_ context.Context, _ *cubebox.DestroyCubeSandboxRequest,
) (*cubebox.DestroyCubeSandboxResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.destroyCount++
	return &cubebox.DestroyCubeSandboxResponse{}, nil
}

func (f *fakeCubelet) List(
	_ context.Context, in *cubebox.ListCubeSandboxRequest,
) (*cubebox.ListCubeSandboxResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.listCount++
	reply := &cubebox.ListCubeSandboxResponse{}
	if in.Id == nil {
		return reply, nil
	}
	pausedAt, known := f.sandboxes[in.GetId()]
	if !known {
		return reply, nil
	}
	reply.Items = []*cubebox.CubeSandbox{{Id: in.GetId(), PausedAt: pausedAt}}
	return reply, nil
}

func (f *fakeCubelet) CommitSandbox(
	_ context.Context, _ *cubebox.CommitSandboxRequest,
) (*cubebox.CommitSandboxResponse, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.commitCount++
	return &cubebox.CommitSandboxResponse{}, nil
}

func (f *fakeCubelet) CleanupTemplate(
	_ context.Context, _ *cubebox.CleanupTemplateRequest,
) (*cubebox.CleanupTemplateResponse, error) {
	return &cubebox.CleanupTemplateResponse{}, nil
}

// -- helpers -------------------------------------------------------------------

func directBackend(t *testing.T, nodes map[string]*fakeCubelet) *Backend {
	t.Helper()
	addresses := make([]NodeAddress, 0, len(nodes))
	for id, node := range nodes {
		addresses = append(addresses, NodeAddress{NodeID: id, Address: node.address})
	}
	b, err := New(Config{Nodes: addresses, RequestTimeout: 5 * time.Second}, sbbackend.SchedulingDirect)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { b.direct.Close() })
	return b
}
