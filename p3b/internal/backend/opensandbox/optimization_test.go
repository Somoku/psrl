package opensandbox

// These cover the three things the optimisations changed, and each case is
// written against the failure it prevents rather than against the mechanism.
//
// The port pool is tested for the property that motivated sharding -- that two
// creates do not queue behind each other -- and for the invariants a partition
// could plausibly break: no port handed to two callers, no port left unreachable,
// no create refused while the range still has room.
//
// The readiness probe is tested for the thing a fixed-interval poll got wrong: a
// sandbox that is already serving must not be charged a sleep.
//
// The warm pool is tested for the resource that makes it different from the
// container backend's pool -- the port reservation that transfers with an entry.

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// -- port pool: sharding ------------------------------------------------------

func TestAPortIsNeverHandedToTwoCallers(t *testing.T) {
	// The invariant a partition could break: a port that lives in two shards, or a
	// fall-through that hands out a port another shard already reserved.
	pool := newPortPool(61000, 61999, 8)

	var (
		mu    sync.Mutex
		seen  = map[int]bool{}
		turn  = uint64(0)
		taken []int
		wg    sync.WaitGroup
	)
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			mu.Lock()
			turn++
			hint := turn
			mu.Unlock()
			port, err := pool.take(hint)
			if err != nil {
				return
			}
			mu.Lock()
			defer mu.Unlock()
			if seen[port] {
				t.Errorf("port %d was handed to two callers", port)
			}
			seen[port] = true
			taken = append(taken, port)
		}()
	}
	wg.Wait()
	if len(taken) != 64 {
		t.Errorf("64 concurrent takes returned %d ports, want 64", len(taken))
	}
	if got := pool.reserved(); got != 64 {
		t.Errorf("the pool reports %d reserved, want 64", got)
	}
}

func TestEveryPortInTheRangeIsReachable(t *testing.T) {
	// A width that does not divide evenly must not leave a tail of the range
	// unreachable: the last shard absorbs the remainder.
	pool := newPortPool(61000, 61099, 7) // 100 ports, 7 shards -> 14 each + 2 spare

	lowest, highest := 1<<31, 0
	for i := 0; i < 100; i++ {
		port, err := pool.take(uint64(i))
		if err != nil {
			t.Fatalf("take %d of 100 failed while the range should still have room: %v", i, err)
		}
		if port < lowest {
			lowest = port
		}
		if port > highest {
			highest = port
		}
	}
	if lowest != 61000 || highest != 61099 {
		t.Errorf("the pool handed out %d-%d, want the full 61000-61099: part of the range "+
			"is unreachable", lowest, highest)
	}
	if _, err := pool.take(0); err == nil {
		t.Error("a 101st take on a 100-port range must be refused")
	}
}

func TestAnExhaustedShardFallsThroughRatherThanRefusing(t *testing.T) {
	// A shard is an optimisation, not a capacity limit. A create refused while
	// thousands of ports sat free in the next shard would be a scheduling defect
	// introduced by a performance change.
	pool := newPortPool(61000, 61127, 2) // two shards of 64

	// Drain shard 0 by always hinting it.
	for i := 0; i < 64; i++ {
		if _, err := pool.take(0); err != nil {
			t.Fatalf("draining shard 0 failed at %d: %v", i, err)
		}
	}
	// The 65th request still prefers shard 0, which is empty. It must come back
	// with a port from shard 1 rather than an error.
	port, err := pool.take(0)
	if err != nil {
		t.Fatalf("a request whose preferred shard is exhausted was refused while the other "+
			"shard was empty: %v", err)
	}
	if port < 61064 {
		t.Errorf("fall-through returned %d, which is inside the exhausted shard", port)
	}
}

func TestShardsAreClampedToKeepThemUseful(t *testing.T) {
	// Sharding a small range produces shards that exhaust immediately and send
	// every create through the fall-through path, which is slower than the single
	// lock it replaced.
	pool := newPortPool(61000, 61099, 64) // 100 ports asked for 64 shards
	if pool.shardCount() > 100/minPortsPerShard {
		t.Errorf("a 100-port range was cut into %d shards; a shard below %d ports costs "+
			"more than it saves", pool.shardCount(), minPortsPerShard)
	}
	if pool.shardCount() < 1 {
		t.Fatal("a pool must always have at least one shard")
	}
}

func TestAReleasedPortIsReusedWithoutAScan(t *testing.T) {
	// The free list is what keeps the critical section constant-time once every
	// port in a shard has been used at least once.
	pool := newPortPool(61000, 61063, 1)
	first, err := pool.take(0)
	if err != nil {
		t.Fatalf("take: %v", err)
	}
	pool.release(first)
	// The released port is the most recent entry on the free list, so it comes back
	// immediately rather than after the cursor wraps the whole shard.
	again, err := pool.take(0)
	if err != nil {
		t.Fatalf("take after release: %v", err)
	}
	if again != first {
		t.Errorf("a released port came back as %d rather than %d, so the free list is not "+
			"being used", again, first)
	}
}

func TestReleasingAPortOutsideTheRangeIsIgnored(t *testing.T) {
	// A handle from an earlier process can carry a port this pool does not own, and
	// a release that panicked or corrupted a shard would take the service down for
	// a stale container.
	pool := newPortPool(61000, 61063, 2)
	pool.release(10)    // below
	pool.release(70000) // above
	if got := pool.reserved(); got != 0 {
		t.Errorf("reserved is %d after releasing ports outside the range, want 0", got)
	}
}

// -- readiness probe ----------------------------------------------------------

func TestTheProbeIntervalStartsBelowTheExpectedBindTime(t *testing.T) {
	// This is the fixed-interval poll's defect stated as a bound: a 25 ms tick was
	// charged to every create, including one whose agent bound in 10 ms. The first
	// delay has to sit under the bind time or the saving does not exist.
	if agentProbeFirstDelay >= 10*time.Millisecond {
		t.Errorf("the first probe delay is %s, which is at or above the agent's typical bind "+
			"time; a warm create would be charged a sleep it did not need", agentProbeFirstDelay)
	}
	if agentProbeMaxDelay <= agentProbeFirstDelay {
		t.Error("the backoff ceiling must be above the first delay, or there is no backoff")
	}
	if agentProbeMaxDelay > agentReadyTimeout {
		t.Error("the backoff ceiling must stay inside the readiness timeout")
	}
}

func TestAnAgentAddressWithNoPortIsRefused(t *testing.T) {
	// The dialler needs an authority. A malformed address has to fail as a create
	// error rather than as a dial to a nonsense target that waits out the timeout.
	if _, err := hostPortOf("not-a-url"); err == nil {
		t.Error("an address with no host must be refused")
	}
	if host, err := hostPortOf("http://127.0.0.1:61000"); err != nil || host != "127.0.0.1:61000" {
		t.Errorf("hostPortOf gave (%q, %v), want (127.0.0.1:61000, nil)", host, err)
	}
}

func TestPortOfAddress(t *testing.T) {
	// The warm pool has to know which port an entry holds so a discard can return
	// it. The only place that is recorded is the agent address.
	port, err := portOfAddress("http://127.0.0.1:61042")
	if err != nil {
		t.Fatalf("portOfAddress: %v", err)
	}
	if port != 61042 {
		t.Errorf("port: got %d, want 61042", port)
	}
	if _, err := portOfAddress("http://127.0.0.1"); err == nil {
		t.Error("an address with no port must be refused, or a discard would leak it")
	}
}

// -- warm pool ---------------------------------------------------------------

type fakeBuilder struct {
	mu        sync.Mutex
	built     []warmEntry
	destroyed []warmEntry
	nextPort  int
	failNext  bool
}

func (f *fakeBuilder) build(_ context.Context, _ backend.Spec) (warmEntry, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failNext {
		f.failNext = false
		return warmEntry{}, context.DeadlineExceeded
	}
	if f.nextPort == 0 {
		f.nextPort = 61000
	}
	f.nextPort++
	entry := warmEntry{
		containerID: "warm-" + time.Now().Format("150405.000000000"),
		hostPort:    f.nextPort,
		address:     "http://127.0.0.1:" + itoa(f.nextPort),
		createdAt:   time.Now(),
	}
	f.built = append(f.built, entry)
	return entry, nil
}

func (f *fakeBuilder) destroy(_ context.Context, entry warmEntry) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.destroyed = append(f.destroyed, entry)
}

func (f *fakeBuilder) counts() (int, int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.built), len(f.destroyed)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	digits := []byte{}
	for n > 0 {
		digits = append([]byte{byte('0' + n%10)}, digits...)
		n /= 10
	}
	return string(digits)
}

func warmConfig() WarmPoolConfig {
	return WarmPoolConfig{
		Image: "alpine:latest", Size: 3, MemoryMB: 512, CPUCount: 1,
		EntryTTL: 10 * time.Minute, RefillInterval: time.Minute,
	}
}

func newTestWarmPool(t *testing.T, cfg WarmPoolConfig, portRange int) (*warmPool, *fakeBuilder) {
	t.Helper()
	builder := &fakeBuilder{}
	pool, err := newWarmPool(cfg, portRange, builder.build, builder.destroy)
	if err != nil {
		t.Fatalf("newWarmPool: %v", err)
	}
	return pool, builder
}

func TestAPoolLargerThanAQuarterOfThePortRangeIsRefused(t *testing.T) {
	// Every entry holds a published host port for as long as it is held, so an
	// oversized pool starves cold creates of ports. Refusing at construction beats
	// discovering it as create failures under load.
	cfg := warmConfig()
	cfg.Size = 300
	if _, err := newWarmPool(cfg, 1000, nil, nil); err == nil {
		t.Fatal("a 300-entry pool on a 1000-port range must be refused")
	}
	cfg.Size = 250
	if _, err := newWarmPool(cfg, 1000, nil, nil); err != nil {
		t.Fatalf("a 250-entry pool on a 1000-port range is exactly the limit: %v", err)
	}
}

func TestAZeroSizePoolIsDisabledAndNilSafe(t *testing.T) {
	cfg := warmConfig()
	cfg.Size = 0
	pool, _ := newTestWarmPool(t, cfg, 4000)
	if pool != nil {
		t.Fatal("a pool of size zero must be nil rather than an empty pool")
	}
	// Every method is called on the create path, so a nil check at each call site
	// would be a second place to forget.
	if _, ok := pool.claim(context.Background(), warmSpec()); ok {
		t.Error("a disabled pool must not report a claim")
	}
	pool.refill(context.Background())
	pool.sweep(context.Background())
	pool.Stop(context.Background())
	if pool.Snapshot().Size != 0 {
		t.Error("a disabled pool must report nothing")
	}
}

func warmSpec() backend.Spec {
	return backend.Spec{
		Source:    backend.Source{Kind: "image", Reference: "alpine:latest"},
		Resources: backend.Resources{MemoryMB: 256, CPUCount: 0.5},
	}
}

func TestAClaimTransfersTheEntrysPort(t *testing.T) {
	// This is the resource that makes this pool different from the container
	// backend's. If a claim did not transfer the reservation, the port would be
	// released while a live sandbox was still published on it, and the next create
	// to take it would fail to bind for a reason that looks like a daemon fault.
	pool, builder := newTestWarmPool(t, warmConfig(), 4000)
	pool.refill(context.Background())

	entry, claimed := pool.claim(context.Background(), warmSpec())
	if !claimed {
		t.Fatal("a matching spec must claim a ready entry")
	}
	if entry.hostPort == 0 {
		t.Error("a claimed entry carries no port, so a release cannot return it")
	}
	if entry.address == "" {
		t.Error("a claimed entry carries no agent address, so the caller cannot reach it")
	}
	// A claim is not a destroy: the port stays reserved because the sandbox is live.
	_, destroyed := builder.counts()
	if destroyed != 0 {
		t.Errorf("claiming destroyed %d entries; a claim transfers ownership rather than "+
			"releasing it", destroyed)
	}
}

func TestAnExpiredEntryIsDestroyedRatherThanClaimed(t *testing.T) {
	pool, builder := newTestWarmPool(t, warmConfig(), 4000)
	now := time.Now()
	pool.now = func() time.Time { return now }
	pool.refill(context.Background())

	pool.now = func() time.Time { return now.Add(11 * time.Minute) }
	if _, claimed := pool.claim(context.Background(), warmSpec()); claimed {
		t.Fatal("an entry past its TTL must not be handed to a caller")
	}
	if _, destroyed := builder.counts(); destroyed != 3 {
		t.Errorf("destroyed %d expired entries, want 3 — an entry that is neither claimed "+
			"nor destroyed leaks both a container and a port", destroyed)
	}
}

func TestASpecWithItsOwnEnvironmentIsNotServedFromThePool(t *testing.T) {
	// Environment is written into the container at create and cannot be changed in
	// place. Handing back an entry without it would surface inside the episode as a
	// command behaving oddly, which is far worse than a cold create.
	pool, _ := newTestWarmPool(t, warmConfig(), 4000)
	pool.refill(context.Background())

	spec := warmSpec()
	spec.Env = map[string]string{"TASK_TOKEN": "abc"}
	if _, claimed := pool.claim(context.Background(), spec); claimed {
		t.Fatal("a spec carrying environment must not be served from a pool built without it")
	}
}

func TestASpecWithItsOwnWorkdirIsNotServedFromThePool(t *testing.T) {
	pool, _ := newTestWarmPool(t, warmConfig(), 4000)
	pool.refill(context.Background())

	spec := warmSpec()
	spec.Workdir = "/testbed"
	if _, claimed := pool.claim(context.Background(), spec); claimed {
		t.Fatal("a spec naming a workdir must not be served from a pool built at /")
	}
}

func TestAGPUSpecIsNotServedFromThePool(t *testing.T) {
	pool, _ := newTestWarmPool(t, warmConfig(), 4000)
	pool.refill(context.Background())

	spec := warmSpec()
	spec.Resources.GPUCount = 1
	if _, claimed := pool.claim(context.Background(), spec); claimed {
		t.Fatal("a GPU request must not be served from a pool built with no devices")
	}
}

func TestALargerFootprintIsNotServedFromThePool(t *testing.T) {
	pool, _ := newTestWarmPool(t, warmConfig(), 4000)
	pool.refill(context.Background())

	spec := warmSpec()
	spec.Resources.MemoryMB = 1024
	if _, claimed := pool.claim(context.Background(), spec); claimed {
		t.Fatal("a spec asking for more memory than entries were built with must not match: " +
			"the cgroup limit is already written")
	}
}

func TestConcurrentClaimsNeverRepeatAnEntry(t *testing.T) {
	cfg := warmConfig()
	cfg.Size = 24
	pool, _ := newTestWarmPool(t, cfg, 4000)
	pool.refill(context.Background())

	var (
		mu     sync.Mutex
		ids    = map[string]bool{}
		claims int
		wg     sync.WaitGroup
	)
	for i := 0; i < 48; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			entry, ok := pool.claim(context.Background(), warmSpec())
			if !ok {
				return
			}
			mu.Lock()
			defer mu.Unlock()
			if ids[entry.containerID] {
				t.Errorf("entry %s was handed to two callers", entry.containerID)
			}
			ids[entry.containerID] = true
			claims++
		}()
	}
	wg.Wait()
	if claims != 24 {
		t.Errorf("48 concurrent claims on a pool of 24 returned %d, want 24", claims)
	}
}

func TestStopDrainsEveryEntry(t *testing.T) {
	// An entry left behind is a container with this service's owner label and no
	// owner, and a port still reserved in a range the next process rebuilds from
	// scratch.
	pool, builder := newTestWarmPool(t, warmConfig(), 4000)
	pool.refill(context.Background())
	pool.Stop(context.Background())
	if _, destroyed := builder.counts(); destroyed != 3 {
		t.Errorf("destroyed %d on shutdown, want 3", destroyed)
	}
	if pool.Snapshot().Ready != 0 {
		t.Error("the pool still reports ready entries after Stop")
	}
}

func TestOneFailedBuildStopsThatPass(t *testing.T) {
	// A daemon that refused one build will refuse the next, so a pass that kept
	// trying would spend the whole refill interval failing.
	pool, builder := newTestWarmPool(t, warmConfig(), 4000)
	builder.failNext = true
	pool.refill(context.Background())
	if built, _ := builder.counts(); built != 0 {
		t.Errorf("built %d after the first build failed, want 0", built)
	}
	if pool.Snapshot().Failures != 1 {
		t.Errorf("failures %d, want 1", pool.Snapshot().Failures)
	}
	// The next pass recovers.
	pool.refill(context.Background())
	if built, _ := builder.counts(); built != 3 {
		t.Errorf("built %d on the recovery pass, want 3", built)
	}
}

func TestTheHitRatioReportsWhatThePoolServed(t *testing.T) {
	pool, _ := newTestWarmPool(t, warmConfig(), 4000)
	pool.refill(context.Background())
	for i := 0; i < 3; i++ {
		pool.claim(context.Background(), warmSpec())
	}
	pool.claim(context.Background(), warmSpec())
	pool.claim(context.Background(), warmSpec())

	report := pool.Snapshot()
	if report.Claims != 5 || report.Hits != 3 {
		t.Errorf("claims=%d hits=%d, want 5 and 3", report.Claims, report.Hits)
	}
	if ratio := report.HitRatio(); ratio < 0.59 || ratio > 0.61 {
		t.Errorf("hit ratio %.3f, want ~0.6", ratio)
	}
}

func TestANonMatchingSpecIsNotCountedAsAClaim(t *testing.T) {
	// Counting it would make the hit ratio describe the workload rather than the
	// pool, and the ratio is the number an operator sizes the budget from.
	pool, _ := newTestWarmPool(t, warmConfig(), 4000)
	pool.refill(context.Background())

	spec := warmSpec()
	spec.Source.Reference = "ubuntu:22.04"
	pool.claim(context.Background(), spec)

	report := pool.Snapshot()
	if report.Claims != 0 {
		t.Errorf("claims after a spec the pool cannot serve: %d, want 0", report.Claims)
	}
	if report.Ready != 3 {
		t.Errorf("ready after a non-matching claim: %d, want 3", report.Ready)
	}
}

// -- the measurement that motivated sharding ----------------------------------

// BenchmarkPortTakeUnsharded and BenchmarkPortTakeSharded measure the thing the
// change was for. Run with:
//
//	go test ./internal/backend/opensandbox/ -run XXX -bench PortTake -cpu 8
//
// The unsharded case is every goroutine on one mutex, which is what the code did
// before. The sharded case is the same work spread over independent locks. What
// the pair demonstrates is not that one is faster on average -- the critical
// section is identical -- but that the unsharded one serialises, so its cost per
// operation grows with the number of contending goroutines while the sharded one
// stays flat.
func BenchmarkPortTakeUnsharded(b *testing.B) {
	pool := newPortPool(40000, 65000, 1)
	var turn atomic.Uint64
	b.RunParallel(func(p *testing.PB) {
		for p.Next() {
			port, err := pool.take(turn.Add(1))
			if err != nil {
				// Exhaustion is not what is being measured; recycle and continue.
				b.StopTimer()
				pool = newPortPool(40000, 65000, 1)
				b.StartTimer()
				continue
			}
			pool.release(port)
		}
	})
}

func BenchmarkPortTakeSharded(b *testing.B) {
	pool := newPortPool(40000, 65000, 16)
	var turn atomic.Uint64
	b.RunParallel(func(p *testing.PB) {
		for p.Next() {
			port, err := pool.take(turn.Add(1))
			if err != nil {
				b.StopTimer()
				pool = newPortPool(40000, 65000, 16)
				b.StartTimer()
				continue
			}
			pool.release(port)
		}
	})
}
