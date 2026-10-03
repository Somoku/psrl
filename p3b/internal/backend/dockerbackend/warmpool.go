// warmpool.go: a pre-created container pool for the docker backend.
//
// A cold create pays for three things in sequence: the image has to be present,
// the daemon has to build the container's namespaces and cgroups, and the
// container has to start. Under concurrency the middle step is the expensive
// one -- veth pairs, netns, and iptables rules are serialized on kernel locks --
// which is why create latency scales so badly with concurrency while exec
// latency stays flat.
//
// A warm pool pays that cost before anyone is waiting. Containers are created
// ahead of demand and held idle; a create then claims one and renames it into
// the caller's sandbox, which is a label write rather than a container build.
//
// Three properties make the pool safe rather than merely fast.
//
// It has its own budget, separate from admission. A pool entry occupies memory
// the node has not promised to any class, so sizing it against the admission
// envelope would let the pool starve the work it exists to accelerate. The
// budget is stated in entries and is the operator's statement of how much
// headroom the node keeps idle.
//
// A pool entry is never reaped as idle. The reclaimer's whole purpose is to
// destroy sandboxes nobody is using, and a pool entry is by definition not in
// use; it is excluded by never being adopted into the lifecycle until it is
// claimed. An entry that has sat past its own TTL is destroyed by the pool's
// own sweep instead, which is a different decision with a different reason.
//
// A claim is only valid when the entry matches what the caller asked for.
// Resources and image are part of the match, because a sandbox handed a
// container built for a different footprint would run under limits its
// admission never charged.
package dockerbackend

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// WarmPoolConfig is what an operator states about the pool.
//
// Size is in entries rather than bytes, because what the pool removes from the
// create path is per-container work rather than per-byte work: the cost is the
// same whether the container is 256 MB or 2 GB.
type WarmPoolConfig struct {
	// Image is what pool entries are built from. One image rather than a set:
	// a pool across images divides a fixed budget by the number of images and
	// stops being a pool for any of them. A spec naming a different image is a
	// cold create, which is the correct outcome rather than a failure.
	Image string
	// Size is how many entries are held ready. Zero disables the pool.
	Size int
	// MemoryMB and CPUCount are the footprint entries are built with. A claim
	// matches on these, so a spec asking for more is a cold create.
	MemoryMB int64
	CPUCount float64
	// EntryTTL is how long an unclaimed entry is kept. An entry older than this
	// is destroyed by the pool's sweep rather than handed to a caller: a
	// container that has idled for an hour may have had its image layers
	// garbage-collected underneath it.
	EntryTTL time.Duration
	// RefillInterval is how often the pool tops itself back up to Size.
	RefillInterval time.Duration
}

// Validate refuses a configuration the pool cannot honour.
func (c WarmPoolConfig) Validate() error {
	if c.Size <= 0 {
		// Disabled, which needs nothing else to be set.
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
			"warm pool refill interval (%s) must be shorter than the entry TTL (%s), or an entry expires before it can be replaced",
			c.RefillInterval, c.EntryTTL)
	}
	return nil
}

// enabled reports whether this configuration asks for a pool at all.
func (c WarmPoolConfig) enabled() bool { return c.Size > 0 && c.Image != "" }

// warmEntry is one pre-created container waiting to be claimed.
type warmEntry struct {
	containerID string
	createdAt   time.Time
}

// warmPool holds pre-created containers for one backend.
type warmPool struct {
	cfg WarmPoolConfig
	// create and destroy are the backend's own operations, injected so the pool
	// is testable without a daemon and so it cannot grow a second transport.
	create  func(ctx context.Context, spec backend.Spec) (string, error)
	destroy func(ctx context.Context, containerID string) error
	now     func() time.Time

	mu      sync.Mutex
	ready   []warmEntry
	// filling bounds concurrent refills, so a burst of claims cannot start Size
	// creates at once and reproduce the thundering herd the pool exists to avoid.
	filling bool

	// Counters for the metric hook. A pool whose hit ratio is low is a pool
	// whose budget is wrong, and that is only visible as a number.
	claims    int64
	hits      int64
	expired   int64
	created   int64
	failures  int64

	stop    chan struct{}
	stopped sync.WaitGroup
}

// newWarmPool returns a pool, or nil when the configuration disables it.
func newWarmPool(
	cfg WarmPoolConfig,
	create func(context.Context, backend.Spec) (string, error),
	destroy func(context.Context, string) error,
) (*warmPool, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if !cfg.enabled() {
		return nil, nil
	}
	return &warmPool{
		cfg: cfg, create: create, destroy: destroy, now: time.Now,
	}, nil
}

// matches reports whether a pool entry can serve this spec.
//
// The image must be the same and the footprint must be no larger than what
// entries were built with. A spec asking for less is served, because running
// under a higher limit than requested is what a cold create would also do once
// the node admitted it; a spec asking for more is not, because the container's
// cgroup limits are already written.
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
	// with no device requests, and a GPU sandbox handed one would see no GPU
	// while admission had charged it for the device.
	if spec.Resources.GPUCount > 0 || len(spec.AssignedGPUs) > 0 {
		return false
	}
	return true
}

// claim takes one ready entry for this spec, or reports that there was none.
//
// An expired entry is discarded rather than returned, and the discard happens
// here as well as in the sweep: a claim arriving between two sweeps must not be
// handed a container that the sweep would have destroyed.
func (p *warmPool) claim(ctx context.Context, spec backend.Spec) (string, bool) {
	if p == nil || !p.matches(spec) {
		return "", false
	}
	deadline := p.now().Add(-p.cfg.EntryTTL)

	p.mu.Lock()
	p.claims++
	var (
		taken   string
		expired []string
	)
	for len(p.ready) > 0 {
		entry := p.ready[len(p.ready)-1]
		p.ready = p.ready[:len(p.ready)-1]
		if entry.createdAt.Before(deadline) {
			expired = append(expired, entry.containerID)
			p.expired++
			continue
		}
		taken = entry.containerID
		p.hits++
		break
	}
	p.mu.Unlock()

	// Destroy outside the lock: a daemon that is slow to remove must not hold up
	// the next claim.
	for _, containerID := range expired {
		_ = p.destroy(ctx, containerID)
	}
	if taken == "" {
		return "", false
	}
	return taken, true
}

// refill tops the pool back up to its size, one pass.
//
// It is called on its own cadence rather than after each claim, because a claim
// is on the create path and a create that waited for its own replacement would
// pay exactly the cost the pool exists to remove.
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
		// A pool entry belongs to no class until it is claimed. The label is for a
		// sweep to recognise it, and it is replaced when the entry is adopted.
		ResourceClass: warmPoolClass,
	}
	for i := 0; i < needed; i++ {
		containerID, err := p.create(ctx, spec)
		if err != nil {
			p.mu.Lock()
			p.failures++
			p.mu.Unlock()
			// One failure stops this pass rather than retrying in a loop: a daemon
			// that refused one create will refuse the next, and the next pass is
			// the right place to try again.
			return
		}
		p.mu.Lock()
		p.ready = append(p.ready, warmEntry{containerID: containerID, createdAt: p.now()})
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
	var stale []string
	for _, entry := range p.ready {
		if entry.createdAt.Before(deadline) {
			stale = append(stale, entry.containerID)
			p.expired++
			continue
		}
		kept = append(kept, entry)
	}
	p.ready = kept
	p.mu.Unlock()

	for _, containerID := range stale {
		_ = p.destroy(ctx, containerID)
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
// Draining on shutdown matters: an entry left behind is a container with this
// service's owner label and no owner, which the next start would reclaim as an
// orphan. Destroying them here makes shutdown leave nothing to clean up.
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
		_ = p.destroy(ctx, entry.containerID)
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

// warmPoolClass is the resource class a pool entry carries before it is
// claimed. It is a label value rather than a real class, and a sweep uses it to
// tell an unclaimed entry from a live sandbox.
const warmPoolClass = "psrl-warm-pool"

// adoptWarm relabels a claimed entry as the caller's sandbox.
//
// A container's labels are immutable after create, so the claim is recorded in
// this service's own map rather than on the container. What the container keeps
// is the owner label, which is what a reclaim sweep matches on, so an entry
// claimed by a process that then died is still recognised as this service's.
func (b *Backend) adoptWarm(containerID string, spec backend.Spec, nodeID string) backend.Created {
	b.mu.Lock()
	b.known[containerID] = containerID
	b.mu.Unlock()
	return backend.Created{
		Handle:       backend.Handle{Backend: b.Name(), SandboxID: containerID, NodeID: nodeID},
		Capabilities: b.Capabilities(),
		Agent:        backend.AgentEndpoint{Address: ""},
		// The caller is told this was warm, because a benchmark that cannot tell a
		// warm claim from a cold create cannot measure the pool.
		WarmStart: true,
	}
}

// createPoolEntry builds one unclaimed container for the pool.
//
// It goes through the same body builder as a real create, so an entry is
// identical to what a cold create would have produced. The difference is only
// that nothing is waiting for it.
func (b *Backend) createPoolEntry(ctx context.Context, spec backend.Spec) (string, error) {
	if err := b.ensureImage(ctx, spec.Source.Reference); err != nil {
		return "", err
	}
	if b.createSem != nil {
		select {
		case b.createSem <- struct{}{}:
			defer func() { <-b.createSem }()
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
	createCtx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	var created struct {
		ID string `json:"Id"`
	}
	if err := b.call(
		createCtx, http.MethodPost, "/containers/create", b.createBody(spec, b.cfg.NodeID), &created,
	); err != nil {
		return "", fmt.Errorf("warm pool create: %w", err)
	}
	if err := b.call(createCtx, http.MethodPost, "/containers/"+created.ID+"/start", nil, nil); err != nil {
		_ = b.remove(ctx, created.ID)
		return "", fmt.Errorf("warm pool start: %w", err)
	}
	return created.ID, nil
}
