package dockerbackend

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// -- fixtures -----------------------------------------------------------------

// fakeRuntime stands in for the daemon. It records what the pool asked for, so a test
// asserts on the pool's decisions rather than on a container.
type fakeRuntime struct {
	mu        sync.Mutex
	created   []backend.Spec
	destroyed []string
	nextID    int
	failNext  bool
}

func (f *fakeRuntime) create(_ context.Context, spec backend.Spec) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failNext {
		f.failNext = false
		return "", fmt.Errorf("daemon refused")
	}
	f.nextID++
	id := fmt.Sprintf("warm-%d", f.nextID)
	f.created = append(f.created, spec)
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

// fakeBudget stands in for the node's admission gate.
type fakeBudget struct {
	mu        sync.Mutex
	reserved  int64
	allow     int64
	pressure  float64
	valid     bool
	acquired  int
	released  int
	refusedAt int
}

func newFakeBudget(allowMB int64) *fakeBudget {
	return &fakeBudget{allow: allowMB, valid: true}
}

func (b *fakeBudget) AcquirePool(want backend.Resources) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.acquired++
	if b.reserved+want.MemoryMB > b.allow {
		b.refusedAt = b.acquired
		return false
	}
	b.reserved += want.MemoryMB
	return true
}

func (b *fakeBudget) ReleasePool(want backend.Resources) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.released++
	b.reserved -= want.MemoryMB
	if b.reserved < 0 {
		b.reserved = 0
	}
}

func (b *fakeBudget) MemoryPressure() (float64, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.pressure, b.valid
}

func (b *fakeBudget) reservedMB() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.reserved
}

// template is the spec every entry is built from in these tests.
func template() backend.Spec {
	return backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: "alpine:latest"},
		Resources:     backend.Resources{MemoryMB: 512, CPUCount: 1},
		ResourceClass: "rollout",
		Env:           map[string]string{"A": "1"},
		Workdir:       "/work",
		ExecMode:      "one_shot",
		Lifetime:      2 * time.Hour,
	}
}

func poolConfig() WarmPoolConfig {
	return WarmPoolConfig{
		Template:       template(),
		Size:           3,
		EntryTTL:       10 * time.Minute,
		RefillInterval: time.Minute,
	}
}

func newTestPool(t *testing.T, cfg WarmPoolConfig, budget backend.PoolBudget) (*warmPool, *fakeRuntime) {
	t.Helper()
	runtime := &fakeRuntime{}
	pool, err := newWarmPool(cfg, runtime.create, runtime.destroy, budget)
	if err != nil {
		t.Fatalf("newWarmPool: %v", err)
	}
	return pool, runtime
}

// matchingSpec is the template as a caller would state it: the same container, with the
// per-caller identity the template deliberately does not carry.
func matchingSpec() backend.Spec {
	spec := template()
	spec.Metadata = map[string]string{"owner_id": "caller-1"}
	spec.IdempotencyKey = "task-1:rollout"
	return spec
}

// -- configuration -------------------------------------------------------------

func TestAZeroSizeDisablesThePool(t *testing.T) {
	cfg := poolConfig()
	cfg.Size = 0
	pool, _ := newTestPool(t, cfg, nil)
	if pool != nil {
		t.Fatal("a pool of size zero must be nil rather than an empty pool")
	}
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
	cfg.Template.Source.Reference = ""
	if _, err := newWarmPool(cfg, nil, nil, nil); err == nil {
		t.Fatal("a sized pool with no image must be refused: there is nothing to build")
	}
}

func TestAPoolTemplateMustNameAClass(t *testing.T) {
	// The class is a label on the entry. Without one, a claimed entry is unattributable
	// after a restart, which is the state this whole design removes.
	cfg := poolConfig()
	cfg.Template.ResourceClass = ""
	if _, err := newWarmPool(cfg, nil, nil, nil); err == nil {
		t.Fatal("a template without a class must be refused")
	}
}

func TestAPoolTemplateMustNotCarryPerCallerIdentity(t *testing.T) {
	// Metadata and an idempotency key name one sandbox, and a shared entry belongs to
	// none: baking either into the template would label every entry with one episode's
	// identity, and labels cannot be changed after create.
	withMetadata := poolConfig()
	withMetadata.Template.Metadata = map[string]string{"owner_id": "caller-1"}
	if _, err := newWarmPool(withMetadata, nil, nil, nil); err == nil {
		t.Fatal("a template with metadata must be refused")
	}

	withKey := poolConfig()
	withKey.Template.IdempotencyKey = "task-1:rollout"
	if _, err := newWarmPool(withKey, nil, nil, nil); err == nil {
		t.Fatal("a template with an idempotency key must be refused")
	}
}

func TestAPoolTemplateMustNotRequestDevices(t *testing.T) {
	cfg := poolConfig()
	cfg.Template.Resources.GPUCount = 1
	if _, err := newWarmPool(cfg, nil, nil, nil); err == nil {
		t.Fatal("a template that asks for a device must be refused")
	}
}

func TestARefillSlowerThanTheTTLIsRefused(t *testing.T) {
	cfg := poolConfig()
	cfg.RefillInterval = cfg.EntryTTL
	if _, err := newWarmPool(cfg, nil, nil, nil); err == nil {
		t.Fatal("a refill interval at or above the entry TTL must be refused")
	}
}

func TestAZeroTTLIsRefused(t *testing.T) {
	cfg := poolConfig()
	cfg.EntryTTL = 0
	if _, err := newWarmPool(cfg, nil, nil, nil); err == nil {
		t.Fatal("an entry TTL of zero must be refused")
	}
}

func TestATTLThatOutlivesTheLifetimeIsRefused(t *testing.T) {
	// An entry's lifetime runs from when it was built, so a long TTL hands the caller a
	// sandbox that is reclaimed sooner than its own budget expects.
	cfg := poolConfig()
	cfg.EntryTTL = cfg.Template.Lifetime / 2
	cfg.RefillInterval = time.Minute
	if _, err := newWarmPool(cfg, nil, nil, nil); err == nil {
		t.Fatal("an entry TTL far inside the lifetime must be refused")
	}
}

func TestReclaimBelowPauseIsRefused(t *testing.T) {
	cfg := poolConfig()
	cfg.PauseAbove = 0.8
	cfg.ReclaimAbove = 0.5
	if _, err := newWarmPool(cfg, nil, nil, nil); err == nil {
		t.Fatal("reclaiming below the pause threshold must be refused")
	}
}

// -- matching ------------------------------------------------------------------

func TestOnlyAnExactSpecMatches(t *testing.T) {
	// The entry's limits and labels were written at create, so a caller that differs in
	// any of them would be served a container that is not what admission charged for.
	pool, _ := newTestPool(t, poolConfig(), nil)

	cases := []struct {
		name   string
		mutate func(*backend.Spec)
	}{
		{"image", func(s *backend.Spec) { s.Source.Reference = "other" }},
		{"memory", func(s *backend.Spec) { s.Resources.MemoryMB = 1024 }},
		{"cpu", func(s *backend.Spec) { s.Resources.CPUCount = 2 }},
		{"disk", func(s *backend.Spec) { s.Resources.DiskMB = 100 }},
		{"workdir", func(s *backend.Spec) { s.Workdir = "/elsewhere" }},
		{"class", func(s *backend.Spec) { s.ResourceClass = "grader" }},
		{"exec mode", func(s *backend.Spec) { s.ExecMode = "persistent" }},
		{"env value", func(s *backend.Spec) { s.Env = map[string]string{"A": "2"} }},
		{"env key", func(s *backend.Spec) { s.Env = map[string]string{"B": "1"} }},
		{"extra env", func(s *backend.Spec) { s.Env = map[string]string{"A": "1", "B": "2"} }},
		{"devices", func(s *backend.Spec) { s.AssignedGPUs = []int32{0} }},
	}
	for _, testCase := range cases {
		spec := matchingSpec()
		testCase.mutate(&spec)
		if pool.matches(spec) {
			t.Errorf("a spec differing in %s must not match", testCase.name)
		}
	}

	if !pool.matches(matchingSpec()) {
		t.Error("the template as a caller states it must match")
	}
}

func TestAShorterLifetimeIsServed(t *testing.T) {
	// Lifetime is a backstop, not a charge: an entry built to live two hours is a valid
	// sandbox for a request that only needs one.
	pool, _ := newTestPool(t, poolConfig(), nil)

	spec := matchingSpec()
	spec.Lifetime = time.Hour

	if !pool.matches(spec) {
		t.Error("an entry that outlives the request must serve it")
	}
}

func TestALongerLifetimeIsNotServed(t *testing.T) {
	// The reverse is not a match: the container's label says the template's lifetime, so
	// a request that needs longer would be reclaimed before its own budget expects.
	pool, _ := newTestPool(t, poolConfig(), nil)

	spec := matchingSpec()
	spec.Lifetime = 3 * time.Hour

	if pool.matches(spec) {
		t.Error("an entry that cannot outlive the request must not serve it")
	}
}

func TestAnUnstatedLifetimeOnlyMatchesAnUnstatedTemplate(t *testing.T) {
	// Zero means "the deployment's default". An entry that states a lifetime would
	// replace that default with its own, in whichever direction it happens to fall.
	pool, _ := newTestPool(t, poolConfig(), nil)
	unstated := matchingSpec()
	unstated.Lifetime = 0
	if pool.matches(unstated) {
		t.Error("a request that states no lifetime must not take an entry that does")
	}

	cfg := poolConfig()
	cfg.Template.Lifetime = 0
	unstatedPool, _ := newTestPool(t, cfg, nil)
	if !unstatedPool.matches(unstated) {
		t.Error("a template that states no lifetime must serve a request that states none")
	}
	stated := matchingSpec()
	if unstatedPool.matches(stated) {
		t.Error("a template that states no lifetime must not serve a request that does")
	}
}

func TestPerCallerIdentityDoesNotAffectMatching(t *testing.T) {
	// Metadata and the idempotency key are exactly the fields a shared entry cannot
	// carry, so they must not be part of the match.
	pool, _ := newTestPool(t, poolConfig(), nil)

	spec := matchingSpec()
	spec.Metadata = map[string]string{"owner_id": "someone-else", "task": "9"}
	spec.IdempotencyKey = "another-key"

	if !pool.matches(spec) {
		t.Error("per-caller identity must not stop an otherwise matching claim")
	}
}

// -- refill, expiry and drain ---------------------------------------------------

func TestRefillBuildsToSize(t *testing.T) {
	pool, runtime := newTestPool(t, poolConfig(), nil)

	pool.refill(context.Background())

	if got := runtime.createdCount(); got != 3 {
		t.Fatalf("created: got %d, want 3", got)
	}
	if got := pool.Snapshot().Ready; got != 3 {
		t.Errorf("ready: got %d, want 3", got)
	}
}

func TestRefillStopsAtTheBudget(t *testing.T) {
	// The pool is an optimization and the envelope is not: an entry that does not fit is
	// not built.
	budget := newFakeBudget(1024) // two entries of 512
	pool, runtime := newTestPool(t, poolConfig(), budget)

	pool.refill(context.Background())

	if got := runtime.createdCount(); got != 2 {
		t.Errorf("created: got %d, want 2 (the budget)", got)
	}
	if got := budget.reservedMB(); got != 1024 {
		t.Errorf("reserved: got %d, want 1024", got)
	}
}

func TestAReleaseThatOutlivesTheBudgetIsRefused(t *testing.T) {
	// A double release would let the node promise the same memory twice.
	budget := newFakeBudget(1024)
	pool, _ := newTestPool(t, poolConfig(), budget)
	pool.refill(context.Background())

	pool.Stop(context.Background())

	if got := budget.reservedMB(); got != 0 {
		t.Errorf("reserved after drain: got %d, want 0", got)
	}
}

func TestRefillStopsWhenMeasurementIsUnavailable(t *testing.T) {
	// An unusable reading is not an idle node: building entries into a machine nobody can
	// see is the one direction this must never fail in.
	budget := newFakeBudget(1 << 20)
	budget.valid = false
	pool, runtime := newTestPool(t, poolConfig(), budget)

	pool.refill(context.Background())

	if got := runtime.createdCount(); got != 0 {
		t.Errorf("created: got %d, want 0 while measurement is unavailable", got)
	}
}

func TestRefillStopsAtThePauseThreshold(t *testing.T) {
	budget := newFakeBudget(1 << 20)
	budget.pressure = 0.85
	pool, runtime := newTestPool(t, poolConfig(), budget)
	pool.cfg.PauseAbove = 0.8
	pool.cfg.ReclaimAbove = 0.95

	pool.refill(context.Background())

	if got := runtime.createdCount(); got != 0 {
		t.Errorf("created: got %d, want 0 above the pause threshold", got)
	}
}

func TestRefillReclaimsIdleEntriesAtTheReclaimThreshold(t *testing.T) {
	budget := newFakeBudget(1 << 20)
	pool, runtime := newTestPool(t, poolConfig(), budget)
	pool.cfg.PauseAbove = 0.5
	pool.cfg.ReclaimAbove = 0.8
	pool.refill(context.Background())
	if runtime.createdCount() != 3 {
		t.Fatalf("the test needs a filled pool, got %d", runtime.createdCount())
	}

	budget.pressure = 0.9
	pool.refill(context.Background())

	if got := runtime.destroyedCount(); got != 3 {
		t.Errorf("destroyed: got %d, want 3 (idle inventory)", got)
	}
	if got := budget.reservedMB(); got != 0 {
		t.Errorf("reserved after reclaim: got %d, want 0", got)
	}
}

func TestAnExpiredEntryIsDestroyedAndItsBudgetReturned(t *testing.T) {
	budget := newFakeBudget(1 << 20)
	pool, runtime := newTestPool(t, poolConfig(), budget)
	pool.refill(context.Background())
	pool.now = func() time.Time { return time.Now().Add(time.Hour) }

	if got := pool.sweep(context.Background()); got != 3 {
		t.Errorf("swept: got %d, want 3", got)
	}
	if got := runtime.destroyedCount(); got != 3 {
		t.Errorf("destroyed: got %d, want 3", got)
	}
	if got := budget.reservedMB(); got != 0 {
		t.Errorf("reserved: got %d, want 0", got)
	}
}

func TestThePoolDrainsOnStop(t *testing.T) {
	pool, runtime := newTestPool(t, poolConfig(), nil)
	pool.refill(context.Background())

	pool.Stop(context.Background())

	if got := runtime.destroyedCount(); got != 3 {
		t.Errorf("destroyed on stop: got %d, want 3", got)
	}
	if got := pool.Snapshot().Ready; got != 0 {
		t.Errorf("ready after stop: got %d, want 0", got)
	}
}

func TestAClaimTakesOneEntryAndTransfersItsBudget(t *testing.T) {
	budget := newFakeBudget(1 << 20)
	pool, _ := newTestPool(t, poolConfig(), budget)
	pool.refill(context.Background())

	entry, ok := pool.claim(context.Background(), matchingSpec())
	if !ok {
		t.Fatal("a matching spec must claim an entry")
	}
	if entry.containerID == "" {
		t.Fatal("the claim must name a container")
	}
	pool.adopt(entry)

	if got := budget.reservedMB(); got != 1024 {
		t.Errorf("reserved after one claim: got %d, want 1024 (two entries left)", got)
	}
	if got := pool.Snapshot().Hits; got != 1 {
		t.Errorf("hits: got %d, want 1", got)
	}
}

func TestAnUnclaimReturnsTheEntryWithoutReleasingItsBudget(t *testing.T) {
	// A daemon that did not answer an inspect is not evidence the entry is bad, so it goes
	// back to the pool still reserved.
	budget := newFakeBudget(1 << 20)
	pool, _ := newTestPool(t, poolConfig(), budget)
	pool.refill(context.Background())
	entry, _ := pool.claim(context.Background(), matchingSpec())

	pool.unclaim(entry)

	if got := pool.Snapshot().Ready; got != 3 {
		t.Errorf("ready: got %d, want 3", got)
	}
	if got := budget.reservedMB(); got != 1536 {
		t.Errorf("reserved: got %d, want 1536 (the entry still exists)", got)
	}
	if got := budget.released; got != 0 {
		t.Errorf("releases: got %d, want 0", got)
	}
}

// -- verification and adoption ---------------------------------------------------

func TestAClaimVerifiesTheEntryAndRenamesIt(t *testing.T) {
	engine := &probeEngine{}
	b := probeBackend(t, engine, "psrl-sandboxes")
	tmpl := template()
	b.pool, _ = newWarmPool(WarmPoolConfig{
		Template: tmpl, Size: 1, EntryTTL: 10 * time.Minute, RefillInterval: time.Minute,
	}, b.createPoolEntry, b.remove, nil)
	entry := warmEntry{containerID: "c-1", createdAt: time.Now()}
	engine.inspectBody = `{
		"Name": "/psrl-warm-abc",
		"State": {"Running": true},
		"Config": {
			"Image": "alpine:latest",
			"Env": ["A=1", "PATH=/usr/bin"],
			"WorkingDir": "/work",
			"Labels": {"psrl.resource_class": "rollout", "psrl.owner": ""}
		},
		"HostConfig": {"Memory": 536870912, "CpuQuota": 100000, "CpuPeriod": 100000, "CgroupParent": "psrl-sandboxes"}
	}`

	created, adopted := b.adoptWarm(context.Background(), entry, tmpl, "node-a")

	if !adopted {
		t.Fatal("a verified entry must be adopted")
	}
	if !created.WarmStart {
		t.Error("the reply must say the sandbox was warm")
	}
	renames := func() []string {
		engine.mu.Lock()
		defer engine.mu.Unlock()
		return append([]string(nil), engine.renameCalls...)
	}()
	if len(renames) != 1 || !strings.HasPrefix(renames[0], sandboxNamePrefix) {
		t.Errorf("renames: got %v, want one psrl-sbx-* rename", renames)
	}
}

func TestAMismatchedEntryIsDestroyedRatherThanAdopted(t *testing.T) {
	// The entry's limits were written at create and cannot be corrected, so a mismatch is
	// destroyed and the caller gets a cold create instead.
	engine := &probeEngine{}
	b := probeBackend(t, engine, "psrl-sandboxes")
	tmpl := template()
	b.pool, _ = newWarmPool(WarmPoolConfig{
		Template: tmpl, Size: 1, EntryTTL: 10 * time.Minute, RefillInterval: time.Minute,
	}, b.createPoolEntry, b.remove, nil)
	entry := warmEntry{containerID: "c-1", createdAt: time.Now()}
	// Half the memory the spec asks for, which is the under-charge a `<=` match would let
	// through.
	engine.inspectBody = `{
		"Name": "/psrl-warm-abc",
		"State": {"Running": true},
		"Config": {"Image": "alpine:latest", "Env": ["A=1"], "WorkingDir": "/work",
			"Labels": {"psrl.resource_class": "rollout", "psrl.owner": ""}},
		"HostConfig": {"Memory": 268435456, "CpuQuota": 100000, "CpuPeriod": 100000, "CgroupParent": "psrl-sandboxes"}
	}`

	_, adopted := b.adoptWarm(context.Background(), entry, tmpl, "node-a")

	if adopted {
		t.Fatal("an entry whose limits differ must not be adopted")
	}
	if got := b.pool.Snapshot().Mismatches; got != 1 {
		t.Errorf("mismatches: got %d, want 1", got)
	}
	engine.mu.Lock()
	defer engine.mu.Unlock()
	if len(engine.renameCalls) != 0 {
		t.Error("a mismatched entry must not be renamed into a sandbox")
	}
}

func TestAnUnreadableEntryIsReturnedToThePool(t *testing.T) {
	// A daemon that did not answer is not evidence the entry is bad.
	engine := &probeEngine{}
	b := probeBackend(t, engine, "")
	tmpl := template()
	b.pool, _ = newWarmPool(WarmPoolConfig{
		Template: tmpl, Size: 1, EntryTTL: 10 * time.Minute, RefillInterval: time.Minute,
	}, b.createPoolEntry, b.remove, nil)
	b.pool.ready = nil
	engine.inspectBody = `not-json`

	_, adopted := b.adoptWarm(context.Background(), warmEntry{containerID: "c-1", createdAt: time.Now()}, tmpl, "node-a")

	if adopted {
		t.Fatal("an unreadable entry must not be adopted")
	}
	if got := b.pool.Snapshot().Ready; got != 1 {
		t.Errorf("ready: got %d, want 1 (put back for the next claim)", got)
	}
}

func TestAFailedRenameDestroysTheEntry(t *testing.T) {
	// The handle must never be returned for a container still named as inventory: a
	// restart would then destroy a live sandbox.
	engine := &probeEngine{renameFails: true}
	b := probeBackend(t, engine, "")
	tmpl := template()
	b.pool, _ = newWarmPool(WarmPoolConfig{
		Template: tmpl, Size: 1, EntryTTL: 10 * time.Minute, RefillInterval: time.Minute,
	}, b.createPoolEntry, b.remove, nil)
	engine.inspectBody = `{
		"Name": "/psrl-warm-abc",
		"State": {"Running": true},
		"Config": {"Image": "alpine:latest", "Env": ["A=1"], "WorkingDir": "/work",
			"Labels": {"psrl.resource_class": "rollout", "psrl.owner": ""}},
		"HostConfig": {"Memory": 536870912, "CpuQuota": 100000, "CpuPeriod": 100000}
	}`

	_, adopted := b.adoptWarm(context.Background(), warmEntry{containerID: "c-1", createdAt: time.Now()}, tmpl, "node-a")

	if adopted {
		t.Fatal("a container that could not be renamed must not be handed over")
	}
	if got := b.pool.Snapshot().Ready; got != 0 {
		t.Errorf("ready: got %d, want 0 (the entry is gone)", got)
	}
}

// -- reconcile classification ----------------------------------------------------

func TestAPoolEntryNameMarksInventory(t *testing.T) {
	for name, want := range map[string]bool{
		"/psrl-warm-abc": true,
		"psrl-warm-abc":  true,
		"/psrl-sbx-abc":  false,
		"/other":         false,
	} {
		if got := isPoolInventoryName(name); got != want {
			t.Errorf("%q: got %v, want %v", name, got, want)
		}
	}
}
