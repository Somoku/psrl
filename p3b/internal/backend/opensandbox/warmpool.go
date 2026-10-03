// warmpool.go: pre-built, agent-ready sandboxes for direct mode.
//
// # What this removes
//
// A cold create in this mode pays for four things in sequence: the container is
// composed and created, the daemon builds its namespaces and publishes a port,
// the container starts, and then the create waits for execd inside it to bind and
// answer. The last step is the one that distinguishes this backend from the plain
// container backend, and it is unavoidable work -- the sandbox genuinely is not
// usable until its agent serves.
//
// Unavoidable is not the same as unavoidable *while someone is waiting*. A warm
// pool does all four steps ahead of demand and holds the result, so a create that
// finds a match pays none of them. What is left is a label write and a map
// update, which is why a claim is two orders of magnitude cheaper than a build
// rather than merely faster.
//
// This matters more here than for the container backend, because here the pool
// removes two costs rather than one: the container build *and* the agent
// readiness wait.
//
// # What a pool entry actually owns
//
// Three resources, and all three transfer together on a claim:
//
//   - the container, which is what the handle names
//   - the host port it is published on, which is reserved in the shared port pool
//   - the agent endpoint, already proven to answer
//
// Port ownership is the part that has to be exactly right. A pool entry holds a
// real reservation in the same port pool that cold creates draw from, taken when
// the entry was built. If a claim forgot to transfer it, the port would be
// released while a live sandbox was still published on it, and the next create to
// take that port would fail to bind for a reason that looks like a daemon fault.
// If a discard forgot to release it, the range would leak one port per expired
// entry. So the pool's budget is also a claim on the port range, and sizing the
// pool is sizing both.
//
// # Why an entry is never adopted until it is claimed
//
// The reclamation sweep exists to destroy sandboxes nobody is using, and a pool
// entry is by definition not in use. It is kept out of the lifecycle entirely
// until a claim, and the pool's own TTL is what retires a stale one. Those are
// two different decisions with two different reasons, and conflating them would
// have the reaper delete the pool it was meant to accelerate.
package opensandbox

import (
	"context"
	"fmt"
	"sync"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// WarmPoolConfig is what an operator states about the pool.
type WarmPoolConfig struct {
	// Image is what entries are built from. One image rather than a set: a pool
	// split across images divides a fixed budget by the number of images and stops
	// being a pool for any of them. A spec naming another image is a cold create,
	// which is the correct outcome rather than a failure.
	Image string
	// Size is how many ready sandboxes are held. Zero disables the pool.
	//
	// It is also a reservation against the port range, because every entry holds a
	// published port. A pool sized near the range would starve cold creates of
	// ports, so Validate refuses that.
	Size int
	// MemoryMB and CPUCount are the footprint entries are built with. A claim
	// matches on these, so a spec asking for more is a cold create.
	MemoryMB int64
	CPUCount float64
	// EntryTTL is how long an unclaimed entry is kept. A sandbox that has idled
	// for hours may have had its image layers collected underneath it, and its
	// agent has been holding a connection open for no reason.
	EntryTTL time.Duration
	// RefillInterval is how often the pool tops itself back up.
	RefillInterval time.Duration
}

// Validate refuses a configuration the pool cannot honour.
//
// portRange is the size of the range the pool draws from, so the check can refuse
// a pool that would starve cold creates rather than discovering it under load.
func (c WarmPoolConfig) Validate(portRange int) error {
	if c.Size <= 0 {
		return nil
	}
	if c.Image == "" {
		return fmt.Errorf("a warm pool of %d entries needs an image to build them from", c.Size)
	}
	if c.EntryTTL <= 0 {
		return fmt.Errorf("warm pool entry TTL must be greater than zero")
	}
	if c.RefillInterval <= 0 {
		return fmt.Errorf("warm pool refill interval must be greater than zero")
	}
	if c.RefillInterval >= c.EntryTTL {
		return fmt.Errorf(
			"warm pool refill interval (%s) must be shorter than the entry TTL (%s), "+
				"or an entry expires before it can be replaced", c.RefillInterval, c.EntryTTL)
	}
	// Every entry holds a published host port for as long as it is held, so the
	// pool is a standing claim on the range. Leaving most of the range for cold
	// creates keeps the pool an accelerator rather than a competitor.
	if limit := portRange / 4; c.Size > limit {
		return fmt.Errorf(
			"a warm pool of %d entries claims too much of a %d-port range: each entry holds a "+
				"published port for as long as it is held, so keep the pool at or below %d "+
				"(a quarter of the range) and leave the rest for cold creates",
			c.Size, portRange, limit)
	}
	return nil
}

func (c WarmPoolConfig) enabled() bool { return c.Size > 0 && c.Image != "" }

// warmEntry is one pre-built, agent-ready sandbox.
type warmEntry struct {
	containerID string
	hostPort    int
	address     string
	createdAt   time.Time
}

// warmPool holds ready sandboxes for one direct runtime.
type warmPool struct {
	cfg WarmPoolConfig
	// build and destroy are the runtime's own operations, injected so the pool is
	// testable without a daemon and so it cannot grow a second transport.
	build   func(ctx context.Context, spec backend.Spec) (warmEntry, error)
	destroy func(ctx context.Context, entry warmEntry)
	now     func() time.Time

	mu    sync.Mutex
	ready []warmEntry
	// filling bounds concurrent refills so a claim burst cannot start Size builds
	// at once and reproduce the thundering herd the pool exists to avoid.
	filling bool

	claims   int64
	hits     int64
	expired  int64
	created  int64
	failures int64

	stop    chan struct{}
	stopped sync.WaitGroup
}

func newWarmPool(
	cfg WarmPoolConfig,
	portRange int,
	build func(context.Context, backend.Spec) (warmEntry, error),
	destroy func(context.Context, warmEntry),
) (*warmPool, error) {
	if err := cfg.Validate(portRange); err != nil {
		return nil, err
	}
	if !cfg.enabled() {
		return nil, nil
	}
	return &warmPool{cfg: cfg, build: build, destroy: destroy, now: time.Now}, nil
}

// matches reports whether a pool entry can serve this spec.
//
// The image must be the same and the footprint no larger than what entries were
// built with. A spec asking for less is served: running under a higher limit than
// requested is what a cold create would also do once the node admitted it. A spec
// asking for more is not, because the container's cgroup limits are already
// written and cannot be raised in place.
func (p *warmPool) matches(spec backend.Spec) bool {
	if spec.Source.Kind != "" && spec.Source.Kind != "image" {
		return false
	}
	if spec.Source.Reference != p.cfg.Image {
		return false
	}
	if spec.Resources.MemoryMB > p.cfg.MemoryMB {
		return false
	}
	if spec.Resources.CPUCount > p.cfg.CPUCount {
		return false
	}
	// A spec that needs devices is never served from the pool: entries are built
	// with no device requests, and a GPU sandbox handed one would see no device
	// while admission had charged it for one.
	if spec.Resources.GPUCount > 0 || len(spec.AssignedGPUs) > 0 {
		return false
	}
	// Environment and working directory are written into the container at create
	// and cannot be changed in place, so an entry built without them cannot serve a
	// spec that asks for them. This is the check that keeps the pool from silently
	// handing back a sandbox missing the variables the task was configured with --
	// a failure that would surface inside the episode as a command behaving oddly
	// rather than as a scheduling refusal.
	//
	// A workdir is compared rather than ignored for the same reason: a harness that
	// expects to start in /testbed and lands in / reports a confusing failure.
	if len(spec.Env) > 0 || spec.Workdir != "" && spec.Workdir != "/" {
		return false
	}
	return true
}

// claim takes one ready entry for this spec, or reports that there was none.
//
// An expired entry is discarded here as well as in the sweep: a claim arriving
// between two sweeps must not be handed a sandbox the sweep would have destroyed.
func (p *warmPool) claim(ctx context.Context, spec backend.Spec) (warmEntry, bool) {
	if p == nil || !p.matches(spec) {
		return warmEntry{}, false
	}
	deadline := p.now().Add(-p.cfg.EntryTTL)

	p.mu.Lock()
	p.claims++
	var (
		taken   warmEntry
		found   bool
		expired []warmEntry
	)
	for len(p.ready) > 0 {
		entry := p.ready[len(p.ready)-1]
		p.ready = p.ready[:len(p.ready)-1]
		if entry.createdAt.Before(deadline) {
			expired = append(expired, entry)
			p.expired++
			continue
		}
		taken, found = entry, true
		p.hits++
		break
	}
	p.mu.Unlock()

	// Destroy outside the lock: a daemon that is slow to remove must not hold up
	// the next claim.
	for _, entry := range expired {
		p.destroy(ctx, entry)
	}
	return taken, found
}

// refill tops the pool back up to its size, one pass.
//
// Called on its own cadence rather than after each claim, because a claim is on
// the create path and a create that waited for its own replacement would pay
// exactly the cost the pool exists to remove.
func (p *warmPool) refill(ctx context.Context) {
	if p == nil {
		return
	}
	p.mu.Lock()
	if p.filling {
		p.mu.Unlock()
		return
	}
	needed := p.cfg.Size - len(p.ready)
	if needed <= 0 {
		p.mu.Unlock()
		return
	}
	p.filling = true
	p.mu.Unlock()
	defer func() {
		p.mu.Lock()
		p.filling = false
		p.mu.Unlock()
	}()

	spec := backend.Spec{
		Source:    backend.Source{Kind: "image", Reference: p.cfg.Image},
		Resources: backend.Resources{MemoryMB: p.cfg.MemoryMB, CPUCount: p.cfg.CPUCount},
		// An entry belongs to no class until it is claimed. The label is for a sweep
		// to recognise it, and it is replaced when the entry is adopted.
		ResourceClass: warmPoolClass,
	}
	for i := 0; i < needed; i++ {
		entry, err := p.build(ctx, spec)
		if err != nil {
			p.mu.Lock()
			p.failures++
			p.mu.Unlock()
			// One failure stops this pass rather than retrying in a loop: a daemon
			// that refused one build will refuse the next, and the next pass is the
			// right place to try again.
			return
		}
		p.mu.Lock()
		p.ready = append(p.ready, entry)
		p.created++
		p.mu.Unlock()
	}
}

// sweep destroys entries that have outlived their TTL.
func (p *warmPool) sweep(ctx context.Context) int {
	if p == nil {
		return 0
	}
	deadline := p.now().Add(-p.cfg.EntryTTL)
	p.mu.Lock()
	kept := p.ready[:0]
	var stale []warmEntry
	for _, entry := range p.ready {
		if entry.createdAt.Before(deadline) {
			stale = append(stale, entry)
			p.expired++
			continue
		}
		kept = append(kept, entry)
	}
	p.ready = kept
	p.mu.Unlock()

	for _, entry := range stale {
		p.destroy(ctx, entry)
	}
	return len(stale)
}

// Start runs the refill and sweep loops until Stop.
func (p *warmPool) Start(ctx context.Context) {
	if p == nil {
		return
	}
	p.mu.Lock()
	if p.stop != nil {
		p.mu.Unlock()
		return
	}
	stop := make(chan struct{})
	p.stop = stop
	p.mu.Unlock()

	p.stopped.Add(1)
	go func() {
		defer p.stopped.Done()
		// Fill once immediately, so the first create can already be warm.
		p.refill(ctx)
		ticker := time.NewTicker(p.cfg.RefillInterval)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				p.sweep(ctx)
				p.refill(ctx)
			}
		}
	}()
}

// Stop ends the loops and destroys every entry still held.
//
// Draining matters twice over here: an entry left behind is a container with this
// service's owner label and no owner, and it is also a port still reserved in a
// range the next process will rebuild from scratch.
func (p *warmPool) Stop(ctx context.Context) {
	if p == nil {
		return
	}
	p.mu.Lock()
	stop := p.stop
	p.stop = nil
	p.mu.Unlock()
	if stop != nil {
		close(stop)
		p.stopped.Wait()
	}

	p.mu.Lock()
	draining := p.ready
	p.ready = nil
	p.mu.Unlock()
	for _, entry := range draining {
		p.destroy(ctx, entry)
	}
}

// WarmPoolReport is the pool's own accounting, for the metric hook.
type WarmPoolReport struct {
	Ready    int
	Size     int
	Claims   int64
	Hits     int64
	Expired  int64
	Created  int64
	Failures int64
}

// HitRatio is the fraction of claims the pool served. It is the number that says
// whether the budget is right: a ratio near zero is a pool that is too small or
// built for the wrong spec, and both are invisible without it.
func (r WarmPoolReport) HitRatio() float64 {
	if r.Claims == 0 {
		return 0
	}
	return float64(r.Hits) / float64(r.Claims)
}

// Snapshot reports what the pool holds. The planes report and never log.
func (p *warmPool) Snapshot() WarmPoolReport {
	if p == nil {
		return WarmPoolReport{}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return WarmPoolReport{
		Ready: len(p.ready), Size: p.cfg.Size,
		Claims: p.claims, Hits: p.hits, Expired: p.expired,
		Created: p.created, Failures: p.failures,
	}
}

// warmPoolClass is the resource class an entry carries before it is claimed. It
// is a label value rather than a real class, and a sweep uses it to tell an
// unclaimed entry from a live sandbox.
const warmPoolClass = "psrl-warm-pool"

// buildWarmEntry creates one unclaimed, agent-ready sandbox.
//
// It goes through the same path as a cold create -- same body builder, same
// readiness wait -- so an entry is indistinguishable from what a cold create
// would have produced. The only difference is that nothing is waiting for it.
func (r *directRuntime) buildWarmEntry(ctx context.Context, spec backend.Spec) (warmEntry, error) {
	created, err := r.createCold(ctx, spec)
	if err != nil {
		return warmEntry{}, err
	}
	port := 0
	if parsed, perr := portOfAddress(created.Agent.Address); perr == nil {
		port = parsed
	}
	return warmEntry{
		containerID: created.Handle.SandboxID,
		hostPort:    port,
		address:     created.Agent.Address,
		createdAt:   r.nowOrWall(),
	}, nil
}

// destroyWarmEntry removes an unclaimed entry and returns its port.
//
// The port goes back through the quarantine rather than straight to the free
// list, because the daemon's userland proxy holds the host socket for a moment
// after the container is gone -- the same reason a released sandbox quarantines.
func (r *directRuntime) destroyWarmEntry(ctx context.Context, entry warmEntry) {
	removeCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	_ = r.docker.removeContainer(removeCtx, entry.containerID, true)
	if entry.hostPort != 0 {
		r.ports.releaseAfter(entry.hostPort, portQuarantine)
	}
}

// nowOrWall is the runtime's clock, so a test can drive entry ages.
func (r *directRuntime) nowOrWall() time.Time { return time.Now() }

// portOfAddress extracts the host port from an agent address.
func portOfAddress(address string) (int, error) {
	hostPort, err := hostPortOf(address)
	if err != nil {
		return 0, err
	}
	_, portText, found := lastColon(hostPort)
	if !found {
		return 0, fmt.Errorf("the agent address %q carries no port", address)
	}
	port := 0
	for _, digit := range portText {
		if digit < '0' || digit > '9' {
			return 0, fmt.Errorf("the agent address %q has a non-numeric port", address)
		}
		port = port*10 + int(digit-'0')
	}
	return port, nil
}

// lastColon splits on the final colon, which is what separates host from port in
// both a bare authority and a bracketed IPv6 one.
func lastColon(s string) (string, string, bool) {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == ':' {
			return s[:i], s[i+1:], true
		}
	}
	return s, "", false
}
