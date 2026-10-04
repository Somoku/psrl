package cubesandbox

// Concurrency behaviour for both scheduling modes.
//
// The unit suites cover what each mode asks for and what it refuses. What they
// do not cover is what happens when many creates arrive at once, and that is
// where the two modes differ most: direct mode holds one gRPC connection per
// Cubelet and multiplexes over it, while provider mode opens HTTP requests to
// one gateway. Both have a concurrency bound, and a bound that does not hold is
// the failure that turns latency into timeouts under a rollout step's burst.
//
// These run against the in-package fake Cubelet (a real gRPC server over the
// generated protobufs) and an httptest gateway, so the transport is real and
// only the runtime underneath is not.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"

	sbbackend "psrl.dev/sandboxd/internal/backend"
	cubebox "psrl.dev/sandboxd/internal/backend/cubesandbox/cubeletpb/services/cubebox/v1"
)

func stressSpec(workflowID string) sbbackend.Spec {
	return sbbackend.Spec{
		Source:        sbbackend.Source{Kind: "template", Reference: "tpl-stress"},
		Resources:     sbbackend.Resources{MemoryMB: 256, CPUCount: 1},
		ResourceClass: "rollout",
		WorkflowID:    workflowID,
	}
}

// -- direct mode --------------------------------------------------------------

func TestDirectModeServesConcurrentCreatesOverOneConnection(t *testing.T) {
	// gRPC multiplexes concurrent calls over one HTTP/2 connection, which is why
	// this backend keeps a connection per Cubelet rather than a pool. If that
	// assumption were wrong the calls would serialise, so what is asserted is that
	// every create completes and the Cubelet saw all of them.
	node := newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"cube-1": node})

	const total = 64
	var (
		wg      sync.WaitGroup
		ok      atomic.Int64
		failed  atomic.Int64
		firstEr atomic.Value
	)
	for i := 0; i < total; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, err := b.Create(context.Background(), "cube-1", stressSpec(fmt.Sprintf("wf-%d", i)), "")
			if err != nil {
				failed.Add(1)
				firstEr.CompareAndSwap(nil, err.Error())
				return
			}
			ok.Add(1)
		}(i)
	}
	wg.Wait()

	if failed.Load() != 0 {
		t.Fatalf("%d of %d concurrent creates failed; first error: %v",
			failed.Load(), total, firstEr.Load())
	}
	if ok.Load() != total {
		t.Errorf("completed %d creates, want %d", ok.Load(), total)
	}
	if got := node.creates(); got != total {
		t.Errorf("the Cubelet saw %d creates, want %d: a create that never reached the node "+
			"would be a sandbox the caller believes it has", got, total)
	}
}

func TestDirectModeHonoursItsCreateBound(t *testing.T) {
	// A Cubelet boots a microVM per create, so an unbounded burst turns latency
	// into timeouts rather than throughput. The semaphore is what prevents that,
	// and a bound that does not hold is invisible until a real node is overwhelmed.
	//
	// The fake blocks inside Create until released, so the number in flight at the
	// high-water mark is exactly what the bound allowed.
	node := newFakeCubelet(t)
	node.blockCreates()

	addresses := []NodeAddress{{NodeID: "cube-1", Address: node.address}}
	b, err := New(Config{
		Nodes: addresses, RequestTimeout: 10 * time.Second, MaxCreateConcurrency: 4,
	}, sbbackend.SchedulingDirect)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { b.direct.Close() })

	var wg sync.WaitGroup
	for i := 0; i < 24; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, _ = b.Create(context.Background(), "cube-1", stressSpec(fmt.Sprintf("wf-%d", i)), "")
		}(i)
	}

	// Let the in-flight set settle, then read the high-water mark before releasing.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && node.inFlightPeak() < 4 {
		time.Sleep(10 * time.Millisecond)
	}
	peak := node.inFlightPeak()
	node.releaseCreates()
	wg.Wait()

	if peak > 4 {
		t.Errorf("%d creates were in flight at once against a bound of 4: the semaphore is "+
			"not holding, so a burst reaches the node unthrottled", peak)
	}
	if peak == 0 {
		t.Error("no create was observed in flight, so this measured nothing")
	}
}

func TestDirectModeSpreadsAcrossTheCubeletsItIsGiven(t *testing.T) {
	// Placement chooses the node and passes its id; the backend must call that
	// Cubelet and no other. A backend that ignored the id would send every sandbox
	// to one node while p3b's accounting believed the fleet was balanced.
	first, second := newFakeCubelet(t), newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"cube-1": first, "cube-2": second})

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			node := "cube-1"
			if i%2 == 1 {
				node = "cube-2"
			}
			if _, err := b.Create(context.Background(), node, stressSpec(fmt.Sprintf("wf-%d", i)), ""); err != nil {
				t.Errorf("create on %s: %v", node, err)
			}
		}(i)
	}
	wg.Wait()

	if first.creates() != 10 || second.creates() != 10 {
		t.Errorf("creates landed %d / %d, want 10 / 10: the backend is not honouring the "+
			"node placement chose", first.creates(), second.creates())
	}
}

func TestDirectModeReleasesConcurrentlyWithoutLosingAny(t *testing.T) {
	// A release that is dropped under load leaks a microVM, which is the most
	// expensive thing this backend can leak.
	node := newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"cube-1": node})

	const total = 32
	handles := make([]sbbackend.Handle, 0, total)
	for i := 0; i < total; i++ {
		created, err := b.Create(context.Background(), "cube-1", stressSpec(fmt.Sprintf("wf-%d", i)), "")
		if err != nil {
			t.Fatalf("create %d: %v", i, err)
		}
		handles = append(handles, created.Handle)
	}

	var wg sync.WaitGroup
	for _, handle := range handles {
		wg.Add(1)
		go func(h sbbackend.Handle) {
			defer wg.Done()
			if err := b.Release(context.Background(), h); err != nil {
				t.Errorf("release %s: %v", h.SandboxID, err)
			}
		}(handle)
	}
	wg.Wait()

	if got := node.destroys(); got != total {
		t.Errorf("the Cubelet saw %d destroys for %d sandboxes: the difference is leaked", got, total)
	}
}

// -- provider mode ------------------------------------------------------------

func TestProviderModeServesConcurrentCreates(t *testing.T) {
	// One gateway, many concurrent requests. What this pins is that the adapter's
	// shared http.Client is safe to use this way and that no reply is mixed up
	// between callers -- each create must come back with its own sandbox id.
	var served atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			id := served.Add(1)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"sandboxID":       fmt.Sprintf("cube-%d", id),
				"domain":          fmt.Sprintf("cube-%d.cube.app", id),
				"envdAccessToken": fmt.Sprintf("tok-%d", id),
			})
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer srv.Close()

	b, err := New(Config{Gateway: srv.URL, RequestTimeout: 10 * time.Second}, sbbackend.SchedulingProvider)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	const total = 64
	var (
		mu   sync.Mutex
		seen = map[string]bool{}
		wg   sync.WaitGroup
	)
	for i := 0; i < total; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			created, err := b.Create(context.Background(), "", stressSpec(fmt.Sprintf("wf-%d", i)), "")
			if err != nil {
				t.Errorf("create %d: %v", i, err)
				return
			}
			mu.Lock()
			defer mu.Unlock()
			if seen[created.Handle.SandboxID] {
				t.Errorf("sandbox id %s came back for two callers: replies are being crossed",
					created.Handle.SandboxID)
			}
			seen[created.Handle.SandboxID] = true
			// The agent endpoint is what makes commands bypass this service, so an
			// empty one under load would silently cost every command a detour.
			if created.Agent.Address == "" {
				t.Errorf("create %d returned no agent address", i)
			}
		}(i)
	}
	wg.Wait()

	if len(seen) != total {
		t.Errorf("%d distinct sandboxes for %d creates", len(seen), total)
	}
	if served.Load() != total {
		t.Errorf("the gateway served %d creates, want %d", served.Load(), total)
	}
}

func TestProviderModePassesTheNodeChoiceToTheGateway(t *testing.T) {
	// Provider mode means CubeMaster places. The adapter must not invent a node id,
	// because a handle naming a node this service chose would make a later routed
	// call go somewhere CubeMaster never put the sandbox.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"sandboxID": "cube-1"})
	}))
	defer srv.Close()

	b, err := New(Config{Gateway: srv.URL, RequestTimeout: 5 * time.Second}, sbbackend.SchedulingProvider)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	created, err := b.Create(context.Background(), "", stressSpec("wf-1"), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if created.Handle.NodeID != "" {
		t.Errorf("provider mode returned node id %q; CubeMaster owns placement, so this "+
			"service must not name a node", created.Handle.NodeID)
	}
}

// -- the capability difference, asserted as a pair ----------------------------

func TestTheTwoModesDeclareDifferentCapabilitiesOnPurpose(t *testing.T) {
	// The divergence is the point: provider mode wraps CubeMaster and can promise
	// its features, direct mode bypasses it and cannot. Declaring a feature a mode
	// cannot serve gets a spec admitted and then failed at create, which costs a
	// sample; this asserts the two lists differ in exactly the expected direction.
	node := newFakeCubelet(t)
	direct := directBackend(t, map[string]*fakeCubelet{"cube-1": node})

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	provider, err := New(Config{Gateway: srv.URL}, sbbackend.SchedulingProvider)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	has := func(list []string, want string) bool {
		for _, item := range list {
			if item == want {
				return true
			}
		}
		return false
	}
	directFeatures := direct.Capabilities().Features
	providerFeatures := provider.Capabilities().Features

	// CubeMaster-level features: provider only. volume is absent from this list
	// because a Cubelet mounts storage natively, so direct mode serves it too --
	// the divergence is only what genuinely has no CubeboxMgr equivalent.
	for _, cubeMasterOnly := range []string{"warm_pool", "template_build", "egress_policy"} {
		if !has(providerFeatures, cubeMasterOnly) {
			t.Errorf("provider mode should declare %q: CubeMaster provides it", cubeMasterOnly)
		}
		if has(directFeatures, cubeMasterOnly) {
			t.Errorf("direct mode declares %q, but it bypasses CubeMaster entirely, so a spec "+
				"requiring it would be admitted and then fail at create", cubeMasterOnly)
		}
	}
	// A freeze is a CubeMaster operation; CubeboxMgr has no pause RPC. Declaring it
	// in direct mode would let the reclaimer pause-on-idle and believe it had
	// released compute it had not.
	if has(directFeatures, "freeze") {
		t.Error("direct mode must not declare freeze: a Cubelet has no pause RPC")
	}
	if len(direct.Capabilities().PauseModes) != 0 {
		t.Errorf("direct mode declares pause modes %v, want none", direct.Capabilities().PauseModes)
	}
	// Neither captures memory, so neither may claim a resume level.
	if direct.Capabilities().ResumeLevel != "" || provider.Capabilities().ResumeLevel != "" {
		t.Error("a CubeSandbox snapshot captures the filesystem and not memory, so neither " +
			"mode may declare a resume level")
	}
	// What a Cubelet does serve must still be declared, including volume.
	for _, cubeletServes := range []string{"filesystem_snapshot", "restore", "image_on_demand", "volume"} {
		if !has(directFeatures, cubeletServes) {
			t.Errorf("direct mode should declare %q: a Cubelet serves it", cubeletServes)
		}
	}
}

// -- benchmarks ----------------------------------------------------------------

func BenchmarkDirectModeCreate(b *testing.B) {
	node := newFakeCubeletB(b)
	addresses := []NodeAddress{{NodeID: "cube-1", Address: node.address}}
	backendUnderTest, err := New(Config{
		Nodes: addresses, RequestTimeout: 10 * time.Second,
	}, sbbackend.SchedulingDirect)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	defer backendUnderTest.direct.Close()

	b.ResetTimer()
	b.RunParallel(func(p *testing.PB) {
		for p.Next() {
			if _, err := backendUnderTest.Create(
				context.Background(), "cube-1", stressSpec("wf-bench"), ""); err != nil {
				b.Fatalf("create: %v", err)
			}
		}
	})
}

func BenchmarkProviderModeCreate(b *testing.B) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"sandboxID": "cube-1", "domain": "x.cube.app"})
	}))
	defer srv.Close()
	backendUnderTest, err := New(Config{
		Gateway: srv.URL, RequestTimeout: 10 * time.Second,
	}, sbbackend.SchedulingProvider)
	if err != nil {
		b.Fatalf("New: %v", err)
	}

	b.ResetTimer()
	b.RunParallel(func(p *testing.PB) {
		for p.Next() {
			if _, err := backendUnderTest.Create(
				context.Background(), "", stressSpec("wf-bench"), ""); err != nil {
				b.Fatalf("create: %v", err)
			}
		}
	})
}

// newFakeCubeletB is newFakeCubelet for a benchmark, which has no *testing.T.
func newFakeCubeletB(b *testing.B) *fakeCubelet {
	b.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		b.Fatalf("listen: %v", err)
	}
	fake := &fakeCubelet{address: listener.Addr().String(), sandboxes: map[string]int64{}}
	server := grpc.NewServer()
	cubebox.RegisterCubeboxMgrServer(server, fake)
	go func() { _ = server.Serve(listener) }()
	b.Cleanup(server.Stop)
	return fake
}

// -- direct-mode feature parity ------------------------------------------------

func TestDirectModeServesVolumesNatively(t *testing.T) {
	// The gap that was closable. A Cubelet mounts storage itself, so a volume
	// request must reach it as a real Volume plus a matching VolumeMount -- both
	// halves, keyed by name. A volume with no mount is storage the sandbox cannot
	// see; a mount with no volume fails the create.
	node := newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"cube-1": node})

	spec := stressSpec("wf-vol")
	spec.BackendOptions = map[string]map[string]string{"cubesandbox": {
		"volume.workspace": "/workspace",
		"volume.dataset":   "/data:ro:driver=cubecow",
		"volume.scratch":   "/scratch:size=4Gi",
	}}
	if _, err := b.Create(context.Background(), "cube-1", spec, ""); err != nil {
		t.Fatalf("create with volumes: %v", err)
	}

	request := node.lastCreate()
	if len(request.GetVolumes()) != 3 {
		t.Fatalf("the create carried %d volumes, want 3", len(request.GetVolumes()))
	}
	// Sorted by name, so one spec always produces one request body.
	names := make([]string, 0, 3)
	for _, volume := range request.GetVolumes() {
		names = append(names, volume.GetName())
	}
	if names[0] != "dataset" || names[1] != "scratch" || names[2] != "workspace" {
		t.Errorf("volumes are not in a stable order: %v", names)
	}

	// A driver means a plugin volume -- the shape CubeMaster's named volumes
	// resolve to, so the same cubecow plumbing is reached either way.
	var dataset, scratch *cubebox.Volume
	for _, volume := range request.GetVolumes() {
		switch volume.GetName() {
		case "dataset":
			dataset = volume
		case "scratch":
			scratch = volume
		}
	}
	if dataset.GetVolumeSource().GetPluginVolume().GetDriver() != "cubecow" {
		t.Errorf("a volume naming a driver must become a plugin volume, got %v",
			dataset.GetVolumeSource())
	}
	// No driver means node-local scratch, and a size applies to it.
	if scratch.GetVolumeSource().GetEmptyDir().GetSizeLimit() != "4Gi" {
		t.Errorf("an emptyDir volume lost its size limit: %v", scratch.GetVolumeSource())
	}

	// Every volume must have its mount, with read-only carried through.
	mounts := request.GetContainers()[0].GetVolumeMounts()
	if len(mounts) != 3 {
		t.Fatalf("the container carried %d mounts for 3 volumes", len(mounts))
	}
	byName := map[string]*cubebox.VolumeMounts{}
	for _, mount := range mounts {
		byName[mount.GetName()] = mount
	}
	if byName["workspace"].GetContainerPath() != "/workspace" {
		t.Errorf("workspace mounted at %q", byName["workspace"].GetContainerPath())
	}
	if !byName["dataset"].GetReadonly() {
		t.Error("a volume marked ro must mount read-only, or a task can edit data it was given to read")
	}
	if byName["workspace"].GetReadonly() {
		t.Error("a volume with no ro attribute must mount writable")
	}
}

func TestAMalformedVolumeFailsTheCreateRatherThanBeingDropped(t *testing.T) {
	// A typo would otherwise start a sandbox whose storage is silently missing, and
	// the task inside it would fail for a reason that looks like its own bug.
	node := newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"cube-1": node})

	for name, value := range map[string]string{
		"relative target":   "workspace",
		"empty target":      "",
		"unknown attribute": "/workspace:shared",
	} {
		spec := stressSpec("wf-bad")
		spec.BackendOptions = map[string]map[string]string{"cubesandbox": {"volume.v": value}}
		if _, err := b.Create(context.Background(), "cube-1", spec, ""); err == nil {
			t.Errorf("%s (%q) was accepted; it must fail the create", name, value)
		}
	}
	// None of them should have reached the node.
	if node.creates() != 0 {
		t.Errorf("a malformed volume spec reached the Cubelet %d times", node.creates())
	}
}

func TestAVolumeOptionDoesNotLeakIntoAnnotations(t *testing.T) {
	// volume.* is consumed into real volumes. Passing it through as an annotation
	// as well would send Cubelet a key it ignores and make the request body lie
	// about what was asked for.
	node := newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"cube-1": node})

	spec := stressSpec("wf-vol-ann")
	spec.BackendOptions = map[string]map[string]string{"cubesandbox": {
		"volume.workspace": "/workspace",
		"custom.key":       "kept",
	}}
	if _, err := b.Create(context.Background(), "cube-1", spec, ""); err != nil {
		t.Fatalf("create: %v", err)
	}
	annotations := node.lastCreate().GetAnnotations()
	if _, leaked := annotations["volume.workspace"]; leaked {
		t.Error("a volume option leaked into the annotations")
	}
	if annotations["custom.key"] != "kept" {
		t.Error("an unrecognised option should still pass through as an annotation")
	}
}

func TestProviderModeRefusesADetachedRun(t *testing.T) {
	// In provider mode the sandbox's envd agent is the command path and p3b is not
	// in it. Accepting the call and doing nothing would be worse than refusing.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"sandboxID": "cube-1"})
	}))
	defer srv.Close()
	b, err := New(Config{Gateway: srv.URL}, sbbackend.SchedulingProvider)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	err = b.RunDetached(context.Background(),
		sbbackend.Handle{Backend: "cubesandbox", SandboxID: "cube-1"}, "echo hi", "", nil)
	if err == nil {
		t.Fatal("provider mode must refuse a detached run and point at the agent endpoint")
	}
	if !strings.Contains(err.Error(), "agent") {
		t.Errorf("the refusal should name the agent endpoint, got: %v", err)
	}
}

func TestProviderModeRefusesAnInPlaceRestore(t *testing.T) {
	// CubeMaster restores by creating a sandbox from a snapshot id, which is the
	// create path. Two ways to do one thing is how the modes drift.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	b, err := New(Config{Gateway: srv.URL}, sbbackend.SchedulingProvider)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := b.Restore(context.Background(),
		sbbackend.Handle{SandboxID: "cube-1"}, "snap-1"); err == nil {
		t.Fatal("provider mode must refuse an in-place restore")
	}
}

func TestARestoreNeedsASnapshotID(t *testing.T) {
	// An empty id would roll the sandbox back to nothing, which the node would
	// either refuse confusingly or honour destructively.
	b := directBackend(t, map[string]*fakeCubelet{"cube-1": newFakeCubelet(t)})
	if err := b.Restore(context.Background(),
		sbbackend.Handle{SandboxID: "sb-1", NodeID: "cube-1"}, ""); err == nil {
		t.Fatal("a restore with no snapshot id must be refused")
	}
}

// -- isolating where the direct/provider gap actually comes from ---------------
//
// The earlier comparison of BenchmarkDirectModeCreate against
// BenchmarkProviderModeCreate is not a protocol measurement, and reading it as
// one is a mistake worth preventing. The two benchmarks differ in three ways at
// once: the wire protocol, the size and shape of the request body, and what the
// stub server does on receipt. Attributing the whole gap to gRPC framing would
// be an unsupported conclusion.
//
// These benchmarks separate the variables so the real contributor is visible.

// BenchmarkMarshalDirectRequest measures serialising a direct-mode create body
// and nothing else: no connection, no server, no framing.
func BenchmarkMarshalDirectRequest(b *testing.B) {
	runtime := &directRuntime{ownerID: "owner-1"}
	request, err := runtime.createRequest(stressSpec("wf-marshal"))
	if err != nil {
		b.Fatalf("createRequest: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := proto.Marshal(request); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkMarshalProviderRequest is the same for the provider body.
//
// Run as a pair with the above, this is what says whether protobuf-vs-JSON is
// the gap. It is not: the provider body is six scalars and the direct body is a
// nested message with three maps and two slices, so this measures payload shape
// far more than it measures encoding.
func BenchmarkMarshalProviderRequest(b *testing.B) {
	body := newSandbox{
		TemplateID: "tpl-stress",
		Timeout:    -1,
		AutoPause:  false,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := json.Marshal(body); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkBuildDirectRequest measures assembling the body, with no encoding at
// all. This is the part this service owns and can therefore optimise.
func BenchmarkBuildDirectRequest(b *testing.B) {
	runtime := &directRuntime{ownerID: "owner-1"}
	spec := stressSpec("wf-build")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := runtime.createRequest(spec); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDirectCreateEqualisedPayload is BenchmarkDirectModeCreate with the
// body reduced toward the provider body's shape -- no annotations beyond the
// owner, no labels, no resources. What remains of the gap after this is the
// transport, which is the number the protocol comparison actually wanted.
func BenchmarkDirectCreateEqualisedPayload(b *testing.B) {
	node := newFakeCubeletB(b)
	addresses := []NodeAddress{{NodeID: "cube-1", Address: node.address}}
	backendUnderTest, err := New(Config{
		Nodes: addresses, RequestTimeout: 10 * time.Second,
	}, sbbackend.SchedulingDirect)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	defer backendUnderTest.direct.Close()

	// Source and nothing else: the smallest body this RPC accepts.
	spec := sbbackend.Spec{Source: sbbackend.Source{Kind: "template", Reference: "tpl"}}
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(p *testing.PB) {
		for p.Next() {
			if _, err := backendUnderTest.Create(context.Background(), "cube-1", spec, ""); err != nil {
				b.Fatalf("create: %v", err)
			}
		}
	})
}

// BenchmarkRawGRPCRoundTrip and BenchmarkRawHTTPRoundTrip strip the adapter out
// entirely: one trivial call against each stub server, same machine, no body
// worth measuring. What is left is the transport plus the stub itself.
//
// This is the control the earlier protocol comparison lacked. If these two differ
// by about what the full create benchmarks differ by, then the adapter is not
// where the gap lives and there is nothing in this package to optimise.
func BenchmarkRawGRPCRoundTrip(b *testing.B) {
	node := newFakeCubeletB(b)
	conn, err := grpc.NewClient(node.address, cubeletDialOptions()...)
	if err != nil {
		b.Fatalf("dial: %v", err)
	}
	defer conn.Close()
	client := cubebox.NewCubeboxMgrClient(conn)

	b.ResetTimer()
	b.RunParallel(func(p *testing.PB) {
		for p.Next() {
			if _, err := client.List(context.Background(), &cubebox.ListCubeSandboxRequest{}); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkRawHTTPRoundTrip(b *testing.B) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	client := &http.Client{Transport: gatewayTransport()}

	b.ResetTimer()
	b.RunParallel(func(p *testing.PB) {
		for p.Next() {
			resp, err := client.Get(srv.URL)
			if err != nil {
				b.Fatal(err)
			}
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}
	})
}

// BenchmarkRawGRPCRoundTripSharedContext isolates one thing the adapter does
// control: whether each call allocates a fresh context with a timeout.
//
// Every lifecycle method here wraps context.WithTimeout, which allocates a
// cancelCtx, registers a timer with the runtime, and on return deregisters it.
// At gRPC's ~21us round trip that is noise; this measures whether it is.
func BenchmarkRawGRPCRoundTripSharedContext(b *testing.B) {
	node := newFakeCubeletB(b)
	conn, err := grpc.NewClient(node.address, cubeletDialOptions()...)
	if err != nil {
		b.Fatalf("dial: %v", err)
	}
	defer conn.Close()
	client := cubebox.NewCubeboxMgrClient(conn)

	b.ResetTimer()
	b.RunParallel(func(p *testing.PB) {
		for p.Next() {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			_, err := client.List(ctx, &cubebox.ListCubeSandboxRequest{})
			cancel()
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkRawGRPCNoKeepaliveNoWindow is the dial options A/B.
//
// cubeletDialOptions raises the stream windows and enables idle keepalive. The
// windows help a large message; the keepalive is a liveness mechanism that
// cannot help latency. This measures whether either costs anything on the hot
// path, so the tuning is justified by measurement rather than plausibility.
func BenchmarkRawGRPCNoKeepaliveNoWindow(b *testing.B) {
	node := newFakeCubeletB(b)
	conn, err := grpc.NewClient(node.address,
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		b.Fatalf("dial: %v", err)
	}
	defer conn.Close()
	client := cubebox.NewCubeboxMgrClient(conn)

	b.ResetTimer()
	b.RunParallel(func(p *testing.PB) {
		for p.Next() {
			if _, err := client.List(context.Background(), &cubebox.ListCubeSandboxRequest{}); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// -- is gRPC's cost structural, or is it one connection? -----------------------
//
// The earlier conclusion -- "the gap is the transport, so there is nothing to
// optimise" -- tested the dial options but not the thing most likely to matter:
// how many connections carry the load.
//
// gRPC-Go multiplexes every RPC for one target over a single HTTP/2 connection,
// and that connection has one loopyWriter goroutine serialising frame writes.
// Go's http.Transport does the opposite: up to MaxIdleConnsPerHost separate TCP
// connections, each with its own write path. So the comparison may have been
// measuring one serialised writer against a hundred parallel ones rather than
// anything intrinsic to the protocols.
//
// If that is the cause, fanning the same calls over several connections should
// recover most of the gap, and the fix is a connection pool rather than nothing.

// BenchmarkRawGRPCRoundTripPooled issues the same calls over a pool of
// connections, round-robined, so no single loopyWriter carries the whole burst.
func BenchmarkRawGRPCRoundTripPooled(b *testing.B) {
	const poolSize = 8
	node := newFakeCubeletB(b)
	clients := make([]cubebox.CubeboxMgrClient, 0, poolSize)
	for i := 0; i < poolSize; i++ {
		conn, err := grpc.NewClient(node.address, cubeletDialOptions()...)
		if err != nil {
			b.Fatalf("dial %d: %v", i, err)
		}
		defer conn.Close()
		clients = append(clients, cubebox.NewCubeboxMgrClient(conn))
	}

	var turn atomic.Uint64
	b.ResetTimer()
	b.RunParallel(func(p *testing.PB) {
		for p.Next() {
			client := clients[turn.Add(1)%poolSize]
			if _, err := client.List(context.Background(), &cubebox.ListCubeSandboxRequest{}); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkRawHTTPRoundTripSingleConn is the mirror control: HTTP held to one
// connection, which is what gRPC does by default. If HTTP's advantage was
// parallel sockets rather than a lighter protocol, constraining it this way
// should erase most of that advantage.
func BenchmarkRawHTTPRoundTripSingleConn(b *testing.B) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()
	client := &http.Client{Transport: &http.Transport{
		MaxIdleConns:        1,
		MaxIdleConnsPerHost: 1,
		MaxConnsPerHost:     1,
		IdleConnTimeout:     90 * time.Second,
	}}

	b.ResetTimer()
	b.RunParallel(func(p *testing.PB) {
		for p.Next() {
			resp, err := client.Get(srv.URL)
			if err != nil {
				b.Fatal(err)
			}
			_, _ = io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
		}
	})
}

func TestEveryConnectionToACubeletIsUsed(t *testing.T) {
	// The pool exists to parallelise the frame writer, which only works if calls
	// actually spread across the connections. A round-robin that always returned
	// index 0 would compile, pass every other test, and quietly restore the
	// single-writer bottleneck the pool was added to remove.
	node := newFakeCubelet(t)
	b := directBackend(t, map[string]*fakeCubelet{"cube-1": node})

	held, err := b.direct.node("cube-1")
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	if len(held.clients) != connsPerCubelet {
		t.Fatalf("the Cubelet holds %d connections, want %d", len(held.clients), connsPerCubelet)
	}

	// Distinct stubs across one full cycle.
	seen := map[cubebox.CubeboxMgrClient]bool{}
	for i := 0; i < connsPerCubelet; i++ {
		seen[held.client()] = true
	}
	if len(seen) != connsPerCubelet {
		t.Errorf("one round-robin cycle used %d distinct connections, want %d: calls are "+
			"collapsing onto a subset of the pool", len(seen), connsPerCubelet)
	}
}

func TestClosingTheRuntimeReleasesEveryConnection(t *testing.T) {
	// Eight connections per Cubelet means a leak is eight sockets per node, not
	// one, so Close has to walk the whole pool.
	node := newFakeCubelet(t)
	addresses := []NodeAddress{{NodeID: "cube-1", Address: node.address}}
	backendUnderTest, err := New(Config{
		Nodes: addresses, RequestTimeout: 5 * time.Second,
	}, sbbackend.SchedulingDirect)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	held, err := backendUnderTest.direct.node("cube-1")
	if err != nil {
		t.Fatalf("node: %v", err)
	}
	conns := held.conns
	backendUnderTest.direct.Close()

	for i, conn := range conns {
		// Shutdown is the state a closed ClientConn reports.
		if state := conn.GetState().String(); state != "SHUTDOWN" {
			t.Errorf("connection %d is %q after Close, want SHUTDOWN", i, state)
		}
	}
}
