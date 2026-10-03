package dockerbackend

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// -- a fake runtime the pool can drive ----------------------------------------

// fakeRuntime stands in for the daemon. It records what the pool asked for, so a
// test asserts on the pool's decisions rather than on a container.
type fakeRuntime struct {
	mu        sync.Mutex
	created   []string
	destroyed []string
	nextID    int
	failNext  bool
	createErr error
}

func (f *fakeRuntime) create(_ context.Context, _ backend.Spec) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failNext {
		f.failNext = false
		if f.createErr != nil {
			return "", f.createErr
		}
		return "", fmt.Errorf("daemon refused")
	}
	f.nextID++
	id := fmt.Sprintf("warm-%d", f.nextID)
	f.created = append(f.created, id)
	return id, nil
}

func (f *fakeRuntime) destroy(_ context.Context, containerID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.destroyed = append(f.destroyed, containerID)
	return nil
}

func (f *fakeRuntime) createdCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.created)
}

func (f *fakeRuntime) destroyedCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.destroyed)
}

func poolConfig() WarmPoolConfig {
	return WarmPoolConfig{
		Image:          "alpine:latest",
		Size:           3,
		MemoryMB:       512,
		CPUCount:       1,
		EntryTTL:       10 * time.Minute,
		RefillInterval: time.Minute,
	}
}

func newTestPool(t *testing.T, cfg WarmPoolConfig) (*warmPool, *fakeRuntime) {
	t.Helper()
	runtime := &fakeRuntime{}
	pool, err := newWarmPool(cfg, runtime.create, runtime.destroy)
	if err != nil {
		t.Fatalf("newWarmPool: %v", err)
	}
	return pool, runtime
}

func matchingSpec() backend.Spec {
	return backend.Spec{
		Source:    backend.Source{Kind: "image", Reference: "alpine:latest"},
		Resources: backend.Resources{MemoryMB: 256, CPUCount: 0.5},
	}
}

// -- configuration -------------------------------------------------------------

func TestAZeroSizeDisablesThePool(t *testing.T) {
	cfg := poolConfig()
	cfg.Size = 0
	pool, _ := newTestPool(t, cfg)
	if pool != nil {
		t.Fatal("a pool of size zero must be nil rather than an empty pool")
	}
	// A nil pool must still be safe to call: Create goes through claim on every
	// request, and a nil check at each call site would be a second place to forget.
	if _, ok := pool.claim(context.Background(), matchingSpec()); ok {
		t.Error("a disabled pool must not report a claim")
	}
	pool.refill(context.Background())
	pool.sweep(context.Background())
	pool.Stop(context.Background())
	if report := pool.Snapshot(); report.Size != 0 {
		t.Errorf("a disabled pool reports size %d, want 0", report.Size)
	}
}

func TestAPoolWithNoImageIsRefused(t *testing.T) {
	cfg := poolConfig()
	cfg.Image = ""
	if _, err := newWarmPool(cfg, nil, nil); err == nil {
		t.Fatal("a sized pool with no image must be refused: there is nothing to build")
	}
}

func TestARefillSlowerThanTheTTLIsRefused(t *testing.T) {
	// An entry that expires before the pool can replace it means the pool is
	// never warm, which is a configuration error rather than a slow pool.
	cfg := poolConfig()
	cfg.RefillInterval = cfg.EntryTTL
	if _, err := newWarmPool(cfg, nil, nil); err == nil {
		t.Fatal("a refill interval at or above the entry TTL must be refused")
	}
}

func TestAZeroTTLIsRefused(t *testing.T) {
	cfg := poolConfig()
	cfg.EntryTTL = 0
	if _, err := newWarmPool(cfg, nil, nil); err == nil {
		t.Fatal("an entry TTL of zero must be refused")
	}
}

// -- refill --------------------------------------------------------------------

func TestRefillFillsToSize(t *testing.T) {
	pool, runtime := newTestPool(t, poolConfig())
	pool.refill(context.Background())
	if got := runtime.createdCount(); got != 3 {
		t.Errorf("created %d entries, want 3", got)
	}
	if report := pool.Snapshot(); report.Ready != 3 {
		t.Errorf("ready %d, want 3", report.Ready)
	}
}

func TestRefillIsIdempotentOnceFull(t *testing.T) {
	pool, runtime := newTestPool(t, poolConfig())
	pool.refill(context.Background())
	pool.refill(context.Background())
	if got := runtime.createdCount(); got != 3 {
		t.Errorf("a second refill on a full pool created more: %d, want 3", got)
	}
}

func TestOneFailedCreateStopsThatPass(t *testing.T) {
	// A daemon that refused one create will refuse the next, so a pass that kept
	// trying would spend the whole refill interval failing.
	pool, runtime := newTestPool(t, poolConfig())
	runtime.failNext = true
	pool.refill(context.Background())
	if got := runtime.createdCount(); got != 0 {
		t.Errorf("created %d after the first create failed, want 0", got)
	}
	if report := pool.Snapshot(); report.Failures != 1 {
		t.Errorf("failures %d, want 1", report.Failures)
	}
}

func TestAFailedRefillRecoversOnTheNextPass(t *testing.T) {
	pool, runtime := newTestPool(t, poolConfig())
	runtime.failNext = true
	pool.refill(context.Background())
	pool.refill(context.Background())
	if got := runtime.createdCount(); got != 3 {
		t.Errorf("created %d on the recovery pass, want 3", got)
	}
}

// -- claim ---------------------------------------------------------------------

func TestAClaimTakesAReadyEntry(t *testing.T) {
	pool, _ := newTestPool(t, poolConfig())
	pool.refill(context.Background())

	containerID, ok := pool.claim(context.Background(), matchingSpec())
	if !ok {
		t.Fatal("a matching spec must claim a ready entry")
	}
	if containerID == "" {
		t.Error("a claim must return a container id")
	}
	if report := pool.Snapshot(); report.Ready != 2 {
		t.Errorf("ready after one claim: %d, want 2", report.Ready)
	}
}

func TestAnEmptyPoolReportsNoClaim(t *testing.T) {
	pool, _ := newTestPool(t, poolConfig())
	// Never refilled, so nothing is ready. The caller falls through to a cold
	// create, which is correct rather than an error.
	if _, ok := pool.claim(context.Background(), matchingSpec()); ok {
		t.Fatal("an empty pool must report no claim")
	}
}

func TestClaimsDrainThePoolExactlyOnce(t *testing.T) {
	// The failure this prevents: two concurrent creates handed the same container.
	pool, _ := newTestPool(t, poolConfig())
	pool.refill(context.Background())

	seen := map[string]bool{}
	for i := 0; i < 3; i++ {
		containerID, ok := pool.claim(context.Background(), matchingSpec())
		if !ok {
			t.Fatalf("claim %d failed on a pool of three", i)
		}
		if seen[containerID] {
			t.Fatalf("container %s was claimed twice", containerID)
		}
		seen[containerID] = true
	}
	if _, ok := pool.claim(context.Background(), matchingSpec()); ok {
		t.Error("a fourth claim on a pool of three must fail")
	}
}

func TestConcurrentClaimsNeverRepeatAContainer(t *testing.T) {
	cfg := poolConfig()
	cfg.Size = 32
	pool, _ := newTestPool(t, cfg)
	pool.refill(context.Background())

	var (
		mu     sync.Mutex
		claims []string
		wg     sync.WaitGroup
	)
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if containerID, ok := pool.claim(context.Background(), matchingSpec()); ok {
				mu.Lock()
				claims = append(claims, containerID)
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

	if len(claims) != 32 {
		t.Errorf("64 concurrent claims on a pool of 32 returned %d, want 32", len(claims))
	}
	seen := map[string]bool{}
	for _, containerID := range claims {
		if seen[containerID] {
			t.Fatalf("container %s was handed to two callers", containerID)
		}
		seen[containerID] = true
	}
}

// -- matching ------------------------------------------------------------------

func TestADifferentImageDoesNotMatch(t *testing.T) {
	pool, _ := newTestPool(t, poolConfig())
	pool.refill(context.Background())
	spec := matchingSpec()
	spec.Source.Reference = "ubuntu:22.04"
	if _, ok := pool.claim(context.Background(), spec); ok {
		t.Fatal("a spec naming another image must not be served from the pool")
	}
}

func TestALargerFootprintDoesNotMatch(t *testing.T) {
	// A container's cgroup limits are written at create. Handing a sandbox that
	// asked for 1 GB an entry built with 512 MB would run it under a limit its
	// admission never charged.
	pool, _ := newTestPool(t, poolConfig())
	pool.refill(context.Background())
	spec := matchingSpec()
	spec.Resources.MemoryMB = 1024
	if _, ok := pool.claim(context.Background(), spec); ok {
		t.Fatal("a spec asking for more memory than entries were built with must not match")
	}
}

func TestASmallerFootprintMatches(t *testing.T) {
	pool, _ := newTestPool(t, poolConfig())
	pool.refill(context.Background())
	spec := matchingSpec()
	spec.Resources.MemoryMB = 128
	if _, ok := pool.claim(context.Background(), spec); !ok {
		t.Fatal("a spec asking for less than entries were built with must be served")
	}
}

func TestAGPURequestDoesNotMatch(t *testing.T) {
	// Entries are built with no device requests, so a GPU sandbox handed one
	// would see no device while admission had charged it for one.
	pool, _ := newTestPool(t, poolConfig())
	pool.refill(context.Background())
	spec := matchingSpec()
	spec.Resources.GPUCount = 1
	if _, ok := pool.claim(context.Background(), spec); ok {
		t.Fatal("a GPU request must not be served from the pool")
	}
}

func TestAnAssignedDeviceDoesNotMatch(t *testing.T) {
	pool, _ := newTestPool(t, poolConfig())
	pool.refill(context.Background())
	spec := matchingSpec()
	spec.AssignedGPUs = []int32{0}
	if _, ok := pool.claim(context.Background(), spec); ok {
		t.Fatal("a spec with a granted device must not be served from the pool")
	}
}

func TestATemplateSourceDoesNotMatch(t *testing.T) {
	pool, _ := newTestPool(t, poolConfig())
	pool.refill(context.Background())
	spec := matchingSpec()
	spec.Source.Kind = "template"
	if _, ok := pool.claim(context.Background(), spec); ok {
		t.Fatal("a template source must not be served from an image pool")
	}
}

// -- expiry --------------------------------------------------------------------

func TestAnExpiredEntryIsNotClaimed(t *testing.T) {
	// A container that has idled past its TTL may have had its image layers
	// collected underneath it, so it is destroyed rather than handed out.
	pool, runtime := newTestPool(t, poolConfig())
	now := time.Now()
	pool.now = func() time.Time { return now }
	pool.refill(context.Background())

	// Move past the TTL.
	pool.now = func() time.Time { return now.Add(11 * time.Minute) }
	if _, ok := pool.claim(context.Background(), matchingSpec()); ok {
		t.Fatal("an expired entry must not be claimed")
	}
	if got := runtime.destroyedCount(); got != 3 {
		t.Errorf("destroyed %d expired entries, want 3", got)
	}
	if report := pool.Snapshot(); report.Expired != 3 {
		t.Errorf("expired count %d, want 3", report.Expired)
	}
}

func TestASweepDestroysExpiredEntries(t *testing.T) {
	pool, runtime := newTestPool(t, poolConfig())
	now := time.Now()
	pool.now = func() time.Time { return now }
	pool.refill(context.Background())

	pool.now = func() time.Time { return now.Add(11 * time.Minute) }
	if swept := pool.sweep(context.Background()); swept != 3 {
		t.Errorf("sweep reported %d, want 3", swept)
	}
	if got := runtime.destroyedCount(); got != 3 {
		t.Errorf("destroyed %d, want 3", got)
	}
	if report := pool.Snapshot(); report.Ready != 0 {
		t.Errorf("ready after sweep: %d, want 0", report.Ready)
	}
}

func TestASweepKeepsFreshEntries(t *testing.T) {
	pool, runtime := newTestPool(t, poolConfig())
	pool.refill(context.Background())
	if swept := pool.sweep(context.Background()); swept != 0 {
		t.Errorf("a sweep over fresh entries reported %d, want 0", swept)
	}
	if got := runtime.destroyedCount(); got != 0 {
		t.Errorf("destroyed %d fresh entries, want 0", got)
	}
}

// -- shutdown ------------------------------------------------------------------

func TestStopDrainsThePool(t *testing.T) {
	// An entry left behind is a container with this service's owner label and no
	// owner, which the next start would reclaim as an orphan.
	pool, runtime := newTestPool(t, poolConfig())
	pool.refill(context.Background())
	pool.Stop(context.Background())
	if got := runtime.destroyedCount(); got != 3 {
		t.Errorf("destroyed %d on shutdown, want 3", got)
	}
	if report := pool.Snapshot(); report.Ready != 0 {
		t.Errorf("ready after stop: %d, want 0", report.Ready)
	}
}

func TestStartFillsThePool(t *testing.T) {
	pool, runtime := newTestPool(t, poolConfig())
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pool.Start(ctx)
	defer pool.Stop(context.Background())

	// Start fills once immediately so the first create can already be warm.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if runtime.createdCount() == 3 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if got := runtime.createdCount(); got != 3 {
		t.Errorf("Start created %d entries, want 3", got)
	}
}

// -- reporting -----------------------------------------------------------------

func TestTheHitRatioReportsWhatThePoolServed(t *testing.T) {
	// A pool whose hit ratio is low is a pool whose budget is wrong, and that is
	// only visible as a number.
	pool, _ := newTestPool(t, poolConfig())
	pool.refill(context.Background())

	// Three hits, then two misses.
	for i := 0; i < 3; i++ {
		pool.claim(context.Background(), matchingSpec())
	}
	pool.claim(context.Background(), matchingSpec())
	pool.claim(context.Background(), matchingSpec())

	report := pool.Snapshot()
	if report.Claims != 5 {
		t.Errorf("claims %d, want 5", report.Claims)
	}
	if report.Hits != 3 {
		t.Errorf("hits %d, want 3", report.Hits)
	}
	if ratio := report.HitRatio(); ratio < 0.59 || ratio > 0.61 {
		t.Errorf("hit ratio %.3f, want ~0.6", ratio)
	}
}

func TestAPoolWithNoClaimsHasNoHitRatio(t *testing.T) {
	// Zero claims is not zero hit rate: dividing would report a pool that is
	// failing when nothing has asked it for anything.
	pool, _ := newTestPool(t, poolConfig())
	if ratio := pool.Snapshot().HitRatio(); ratio != 0 {
		t.Errorf("hit ratio with no claims: %.3f, want 0", ratio)
	}
}

// -- a non-matching claim does not consume an entry ----------------------------

func TestANonMatchingClaimLeavesThePoolIntact(t *testing.T) {
	pool, _ := newTestPool(t, poolConfig())
	pool.refill(context.Background())

	spec := matchingSpec()
	spec.Source.Reference = "ubuntu:22.04"
	pool.claim(context.Background(), spec)

	if report := pool.Snapshot(); report.Ready != 3 {
		t.Errorf("ready after a non-matching claim: %d, want 3", report.Ready)
	}
	// A spec the pool cannot serve is not a claim against it: counting it would
	// make the hit ratio describe the workload rather than the pool.
	if report := pool.Snapshot(); report.Claims != 0 {
		t.Errorf("claims after a non-matching spec: %d, want 0", report.Claims)
	}
}
