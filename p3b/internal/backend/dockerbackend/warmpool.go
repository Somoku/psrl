// warmpool.go: a pre-created container pool for the docker backend.
//
// A cold create pays for three things in sequence: the image has to be present, the
// daemon has to build the container's namespaces and cgroups, and the container has to
// start. Under concurrency the middle step is the expensive one -- veth pairs, netns,
// and iptables rules are serialized on kernel locks -- which is why create latency
// scales so badly with concurrency while exec latency stays flat.
//
// A warm pool pays that cost before anyone is waiting. Containers are created ahead of
// demand and held idle; a create then claims one and renames it into the caller's
// sandbox.
//
// An entry is built from a complete spec, not from an image and a size. Everything the
// container will be -- image, footprint, environment, workdir, class, exec mode,
// lifetime -- is written at create and cannot be changed afterwards, so an entry is
// only safe to hand out when it already matches the caller exactly. A claim therefore
// verifies the container against the caller's spec before it is handed over, and
// destroys it rather than adopting a mismatch.
//
// Two properties keep the pool from being a leak. It has its own budget, separate from
// the classes: an entry occupies memory the node has not promised to any episode, so
// sizing it against a class would let the pool starve the work it exists to accelerate.
// And it is never built or kept while the node measures itself as full.
package dockerbackend

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// WarmPoolConfig is what an operator states about the pool.
//
// Size is in entries rather than bytes, because what the pool removes from the create
// path is per-container work rather than per-byte work: the cost is the same whether
// the container is 256 MB or 2 GB.
type WarmPoolConfig struct {
	// Template is the complete spec entries are built from. It carries everything
	// that ends up in the container, because a label written at create is immutable:
	// an entry that does not already match its caller cannot be made to.
	//
	// It must not carry per-caller identity (metadata, an idempotency key, devices).
	// Those name one sandbox, and a shared entry belongs to none yet.
	Template backend.Spec
	// Size is how many entries are held ready. Zero disables the pool.
	Size int
	// EntryTTL is how long an unclaimed entry is kept. An entry older than this is
	// destroyed rather than handed to a caller: a container that has idled for an hour
	// may have had its image layers garbage-collected underneath it.
	EntryTTL time.Duration
	// RefillInterval is how often the pool tops itself back up to Size.
	RefillInterval time.Duration
	// PauseAbove stops refill when the measured memory fraction reaches it. Zero
	// takes 0.9, matching a node that has not tuned it.
	PauseAbove float64
	// ReclaimAbove destroys idle entries when the measured memory fraction reaches it.
	// Zero takes the pause threshold, so reaching it both stops the refill and gives
	// back what the pool was holding.
	ReclaimAbove float64
}

const (
	defaultPoolPauseAbove   = 0.9
	defaultPoolReclaimAbove = 0.9
)

// Validate refuses a configuration the pool cannot honour.
func (c WarmPoolConfig) Validate() error {
	if c.Size <= 0 {
		// Disabled, which needs nothing else to be set.
		return nil
	}
	template := c.Template
	if template.Source.Kind != "image" || strings.TrimSpace(template.Source.Reference) == "" {
		return fmt.Errorf("a warm pool of %d entries must name an image to build them from", c.Size)
	}
	if template.ResourceClass == "" {
		return fmt.Errorf(
			"a warm pool template must name a resource class: the class is a label on the entry, " +
				"and an entry without one cannot be charged or attributed after a restart")
	}
	if len(template.Metadata) > 0 {
		return fmt.Errorf(
			"a warm pool template must not carry metadata: it is a label on the container, " +
				"labels are immutable, and a shared entry would wear one caller's identity")
	}
	if template.IdempotencyKey != "" {
		return fmt.Errorf("a warm pool template must not carry an idempotency key: it names one sandbox")
	}
	if template.Resources.GPUCount > 0 || len(template.AssignedGPUs) > 0 {
		return fmt.Errorf("a warm pool template must not request devices: entries are built with none")
	}
	if c.EntryTTL <= 0 {
		return fmt.Errorf("warm pool entry TTL must be greater than zero")
	}
	if c.RefillInterval <= 0 {
		return fmt.Errorf("warm pool refill interval must be greater than zero")
	}
	if c.RefillInterval >= c.EntryTTL {
		return fmt.Errorf(
			"warm pool refill interval (%s) must be shorter than the entry TTL (%s), or an entry expires "+
				"before it can be replaced", c.RefillInterval, c.EntryTTL)
	}
	if template.Lifetime > 0 && c.EntryTTL*10 > template.Lifetime {
		return fmt.Errorf(
			"warm pool entry TTL (%s) must stay well inside the template's lifetime (%s): an entry's "+
				"lifetime runs from when it was built, so a long TTL hands the caller a sandbox that is "+
				"reclaimed early", c.EntryTTL, template.Lifetime)
	}
	pause, reclaim := c.thresholds()
	if pause <= 0 || pause > 1 || reclaim <= 0 || reclaim > 1 {
		return fmt.Errorf("warm pool thresholds must be in (0, 1], got pause=%.3f reclaim=%.3f", pause, reclaim)
	}
	if reclaim < pause {
		return fmt.Errorf(
			"warm pool reclaim threshold (%.3f) is below its pause threshold (%.3f); reclaiming below the "+
				"point that stops refill would give back entries the pool is still trying to keep", reclaim, pause)
	}
	return nil
}

// thresholds returns the resolved pressure thresholds.
func (c WarmPoolConfig) thresholds() (pause, reclaim float64) {
	pause = c.PauseAbove
	if pause <= 0 {
		pause = defaultPoolPauseAbove
	}
	reclaim = c.ReclaimAbove
	if reclaim <= 0 {
		reclaim = pause
	}
	return pause, reclaim
}

// enabled reports whether this configuration asks for a pool at all.
func (c WarmPoolConfig) enabled() bool { return c.Size > 0 && c.Template.Source.Reference != "" }

// warmEntry is one pre-created container waiting to be claimed.
type warmEntry struct {
	containerID string
	createdAt   time.Time
	// reserved reports whether this entry's footprint is still held against the
	// node's budget. Exactly one path clears it -- claim, expiry, or drain -- so a
	// footprint cannot be returned twice.
	reserved bool
}

// warmPool holds pre-created containers for one backend.
type warmPool struct {
	cfg WarmPoolConfig
	// create and destroy are the backend's own operations, injected so the pool is
	// testable without a daemon and so it cannot grow a second transport.
	create  func(ctx context.Context, spec backend.Spec) (string, error)
	destroy func(ctx context.Context, containerID string) error
	// budget is nil when the node has no gate to ask, which is the case in a unit
	// test and in a deployment that runs without admission.
	budget backend.PoolBudget
	now    func() time.Time

	mu    sync.Mutex
	ready []warmEntry
	// filling bounds concurrent refills, so a burst of claims cannot start Size
	// creates at once and reproduce the thundering herd the pool exists to avoid.
	filling bool

	// Counters for the metric hook. A pool whose hit ratio is low is a pool whose
	// budget is wrong, and that is only visible as a number.
	claims     int64
	hits       int64
	expired    int64
	created    int64
	failures   int64
	mismatches int64

	stop    chan struct{}
	stopped sync.WaitGroup
}

// newWarmPool returns a pool, or nil when the configuration disables it.
func newWarmPool(
	cfg WarmPoolConfig,
	create func(context.Context, backend.Spec) (string, error),
	destroy func(context.Context, string) error,
	budget backend.PoolBudget,
) (*warmPool, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if !cfg.enabled() {
		return nil, nil
	}
	return &warmPool{
		cfg: cfg, create: create, destroy: destroy, budget: budget, now: time.Now,
	}, nil
}

// matches reports whether a pool entry can serve this spec.
//
// It compares every field that ends up in the container, and compares them exactly
// rather than allowing a smaller request to be served by a larger entry. The
// admission charge is the caller's spec and the entry's cgroup limits were written at
// pool create, so a larger entry would let the sandbox use more than the node charged
// for, and a smaller one would throttle it below what was charged. Both are silent.
func (p *warmPool) matches(spec backend.Spec) bool {
	template := p.cfg.Template
	if spec.Source.Kind != "" && spec.Source.Kind != "image" {
		return false
	}
	if spec.Source.Reference != template.Source.Reference {
		return false
	}
	if spec.Resources.MemoryMB != template.Resources.MemoryMB ||
		spec.Resources.CPUCount != template.Resources.CPUCount ||
		spec.Resources.DiskMB != template.Resources.DiskMB {
		return false
	}
	if spec.Resources.GPUCount > 0 || len(spec.AssignedGPUs) > 0 {
		return false
	}
	if spec.Workdir != template.Workdir {
		return false
	}
	if spec.ResourceClass != template.ResourceClass {
		return false
	}
	if spec.ExecMode != template.ExecMode {
		return false
	}
	if spec.Lifetime != template.Lifetime {
		return false
	}
	if len(spec.Env) != len(template.Env) {
		return false
	}
	for key, want := range template.Env {
		if spec.Env[key] != want {
			return false
		}
	}
	return true
}

// claim takes one ready entry for this spec, or reports that there was none.
//
// An expired entry is discarded rather than returned, and the discard happens here as
// well as in the sweep: a claim arriving between two sweeps must not be handed a
// container that the sweep would have destroyed.
func (p *warmPool) claim(ctx context.Context, spec backend.Spec) (warmEntry, bool) {
	if p == nil || !p.matches(spec) {
		return warmEntry{}, false
	}
	deadline := p.now().Add(-p.cfg.EntryTTL)

	p.mu.Lock()
	p.claims++
	var (
		taken   warmEntry
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
		taken = entry
		break
	}
	p.mu.Unlock()

	// Destroy outside the lock: a daemon that is slow to remove must not hold up the
	// next claim.
	for _, entry := range expired {
		p.discard(ctx, entry)
	}
	if taken.containerID == "" {
		return warmEntry{}, false
	}
	return taken, true
}

// unclaim returns an entry to the pool after a claim that could not be completed.
//
// It is for a transient failure -- a daemon that did not answer an inspect -- where the
// entry itself is still good. The entry keeps its original creation time so the TTL
// still bounds how long it may sit there.
func (p *warmPool) unclaim(entry warmEntry) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.ready = append(p.ready, entry)
}

// adopt records that an entry became a caller's sandbox, returning its footprint to
// the classes it was held from.
func (p *warmPool) adopt(entry warmEntry) {
	p.mu.Lock()
	p.hits++
	release := entry.reserved
	entry.reserved = false
	p.mu.Unlock()
	if release && p.budget != nil {
		p.budget.ReleasePool(p.cfg.Template.Resources)
	}
}

// discard destroys an entry and returns its footprint, counting it as expired.
func (p *warmPool) discard(ctx context.Context, entry warmEntry) {
	p.mu.Lock()
	release := entry.reserved
	entry.reserved = false
	p.mu.Unlock()
	if release && p.budget != nil {
		p.budget.ReleasePool(p.cfg.Template.Resources)
	}
	if err := p.destroy(ctx, entry.containerID); err != nil {
		// A container that cannot be destroyed is still memory the node holds, but the
		// pool has given up its claim on it: the reclaimer owns it now, by owner label.
		p.mu.Lock()
		p.failures++
		p.mu.Unlock()
	}
}

// refill tops the pool back up to its size, one pass.
//
// It is called on its own cadence rather than after each claim, because a claim is on
// the create path and a create that waited for its own replacement would pay exactly
// the cost the pool exists to remove.
func (p *warmPool) refill(ctx context.Context) {
	if p == nil {
		return
	}
	// Measurement first: prefetching into a node that reports itself full is an
	// optimization that causes the failures it exists to avoid, and an unusable
	// reading is not a reason to guess in the permissive direction.
	if p.budget != nil {
		measured, ok := p.budget.MemoryPressure()
		pause, reclaim := p.cfg.thresholds()
		if !ok || measured >= pause {
			if ok && measured >= reclaim {
				p.reclaimIdle(ctx)
			}
			return
		}
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

	spec := p.cfg.Template
	for i := 0; i < needed; i++ {
		if p.budget != nil && !p.budget.AcquirePool(spec.Resources) {
			// The entry would not fit inside the declared envelope. Stopping is the
			// right answer: the pool is an optimization and the envelope is not.
			return
		}
		containerID, err := p.create(ctx, spec)
		if err != nil {
			if p.budget != nil {
				p.budget.ReleasePool(spec.Resources)
			}
			p.mu.Lock()
			p.failures++
			p.mu.Unlock()
			// One failure stops this pass rather than retrying in a loop: a daemon
			// that refused one create will refuse the next, and the next pass is
			// the right place to try again.
			return
		}
		p.mu.Lock()
		p.ready = append(p.ready, warmEntry{containerID: containerID, createdAt: p.now(), reserved: p.budget != nil})
		p.created++
		p.mu.Unlock()
	}
}

// reclaimIdle destroys every unclaimed entry, for a node under memory pressure.
//
// Only inventory is touched. A claimed entry is a caller's sandbox, which this pool no
// longer owns and must never destroy.
func (p *warmPool) reclaimIdle(ctx context.Context) {
	p.mu.Lock()
	idle := p.ready
	p.ready = nil
	p.mu.Unlock()
	for _, entry := range idle {
		p.discard(ctx, entry)
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
		p.discard(ctx, entry)
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
		p.discard(ctx, entry)
	}
}

// WarmPoolReport is the pool's own accounting, for the metric hook.
type WarmPoolReport struct {
	Ready      int
	Size       int
	Claims     int64
	Hits       int64
	Expired    int64
	Created    int64
	Failures   int64
	Mismatches int64
}

// HitRatio is the fraction of claims the pool served. It is the number that says
// whether the budget is right: a ratio near zero is a pool that is too small or
// built for the wrong spec, and both are invisible without it.
//
// A mismatch counts as a claim that was not served, which is what it is: the entry
// existed but could not be handed over.
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
		Created: p.created, Failures: p.failures, Mismatches: p.mismatches,
	}
}

// isPoolInventoryName reports whether a container name marks unclaimed inventory.
func isPoolInventoryName(name string) bool {
	return strings.HasPrefix(strings.TrimPrefix(name, "/"), warmNamePrefix)
}

// SetPoolBudget installs the node's budget on this backend's pool.
//
// It is separate from construction because the backend is built before the admission
// gate it borrows from, and a pool that started refilling without a budget would
// prefetch memory the node never agreed to hold. Call it before Start.
func (b *Backend) SetPoolBudget(budget backend.PoolBudget) {
	if b.pool == nil {
		return
	}
	b.pool.mu.Lock()
	defer b.pool.mu.Unlock()
	b.pool.budget = budget
}

// createPoolEntry builds one unclaimed container for the pool.
//
// It goes through the same body builder and the same create gate as a real create, so
// an entry is identical to what a cold create would have produced and cannot stampede
// the daemon. The only differences are its name -- which marks it as inventory -- and
// that nothing is waiting for it.
func (b *Backend) createPoolEntry(ctx context.Context, spec backend.Spec) (string, error) {
	if err := b.ensureImage(ctx, spec.Source.Reference); err != nil {
		return "", err
	}
	release, err := b.acquireCreateSlot(ctx)
	if err != nil {
		return "", err
	}
	defer release()

	created, err := b.createNamed(ctx, warmNamePrefix, b.createBody(spec, b.cfg.NodeID))
	if err != nil {
		return "", fmt.Errorf("warm pool create: %w", err)
	}
	if err := b.call(ctx, http.MethodPost, "/containers/"+created.ID+"/start", nil, nil); err != nil {
		_ = b.remove(ctx, created.ID)
		return "", fmt.Errorf("warm pool start: %w", err)
	}
	return created.ID, nil
}

// inspection is the slice of a container's inspect reply a claim verifies.
type inspection struct {
	Name  string `json:"Name"`
	State struct {
		Running bool `json:"Running"`
	} `json:"State"`
	Config struct {
		Image      string            `json:"Image"`
		Env        []string          `json:"Env"`
		WorkingDir string            `json:"WorkingDir"`
		Labels     map[string]string `json:"Labels"`
	} `json:"Config"`
	HostConfig struct {
		Memory       int64  `json:"Memory"`
		CpuQuota     int64  `json:"CpuQuota"`
		CpuPeriod    int64  `json:"CpuPeriod"`
		CgroupParent string `json:"CgroupParent"`
	} `json:"HostConfig"`
}

// inspectForClaim reads a claimed entry back before it is handed over.
func (b *Backend) inspectForClaim(ctx context.Context, containerID string) (inspection, error) {
	var reply inspection
	if err := b.call(ctx, http.MethodGet, "/containers/"+containerID+"/json", nil, &reply); err != nil {
		return inspection{}, err
	}
	return reply, nil
}

// verifyClaim checks that a claimed entry is what the caller asked for.
//
// The entry was built from the pool's template, so this can only fail when the
// template and the caller disagree -- which exact matching should have prevented, and
// which is exactly why it is checked: a mismatch that slipped through would hand a
// caller a container with someone else's limits, and the only symptom would be an
// episode that behaved unlike its spec.
func (b *Backend) verifyClaim(reply inspection, spec backend.Spec) error {
	if !reply.State.Running {
		return fmt.Errorf("the entry is not running")
	}
	if got, want := reply.HostConfig.Memory, spec.Resources.MemoryMB*1024*1024; got != want {
		return fmt.Errorf("memory limit is %d, want %d", got, want)
	}
	if spec.Resources.CPUCount > 0 {
		wantQuota := int64(spec.Resources.CPUCount * 100000)
		if reply.HostConfig.CpuQuota != wantQuota {
			return fmt.Errorf("CPU quota is %d, want %d", reply.HostConfig.CpuQuota, wantQuota)
		}
	}
	if b.cfg.CgroupParent != "" && reply.HostConfig.CgroupParent != b.cfg.CgroupParent {
		return fmt.Errorf("cgroup parent is %q, want %q", reply.HostConfig.CgroupParent, b.cfg.CgroupParent)
	}
	if reply.Config.Image != spec.Source.Reference {
		return fmt.Errorf("image is %q, want %q", reply.Config.Image, spec.Source.Reference)
	}
	if reply.Config.WorkingDir != spec.Workdir {
		return fmt.Errorf("workdir is %q, want %q", reply.Config.WorkingDir, spec.Workdir)
	}
	// Env is a subset check rather than equality: the image's own ENV is merged into
	// the container's, so an entry legitimately carries more than the spec states.
	environment := map[string]string{}
	for _, entry := range reply.Config.Env {
		key, value, found := strings.Cut(entry, "=")
		if found {
			environment[key] = value
		}
	}
	for key, want := range spec.Env {
		if got, present := environment[key]; !present || got != want {
			return fmt.Errorf("environment %s is %q (present=%v), want %q", key, got, present, want)
		}
	}
	if got := reply.Config.Labels[labelClass]; got != spec.ResourceClass {
		return fmt.Errorf("class label is %q, want %q", got, spec.ResourceClass)
	}
	if got := reply.Config.Labels[labelOwner]; got != b.cfg.OwnerID {
		return fmt.Errorf("owner label is %q, want %q", got, b.cfg.OwnerID)
	}
	if !isPoolInventoryName(reply.Name) {
		return fmt.Errorf("name %q does not mark pool inventory", reply.Name)
	}
	return nil
}

// adoptWarm verifies a claimed entry, renames it into the caller's sandbox, and returns
// it. A false result means the caller should cold-create instead.
//
// The order is load-bearing. Verification comes before the rename so a container that
// does not match is destroyed while it is still inventory; and the rename comes before
// the handle is returned so a container that is handed to a caller is never still named
// as inventory. Returning first and renaming after would let a restart destroy a live
// sandbox, which is the failure the old label marker could not avoid.
func (b *Backend) adoptWarm(
	ctx context.Context, entry warmEntry, spec backend.Spec, nodeID string,
) (backend.Created, bool) {
	reply, err := b.inspectForClaim(ctx, entry.containerID)
	if err != nil {
		// A daemon that did not answer is not evidence the entry is bad, and destroying
		// a good container on a blip would turn a transient fault into a lost entry.
		b.pool.unclaim(entry)
		return backend.Created{}, false
	}
	if err := b.verifyClaim(reply, spec); err != nil {
		b.pool.noteMismatch()
		b.pool.discard(ctx, entry)
		return backend.Created{}, false
	}
	name := newContainerName(sandboxNamePrefix)
	if err := b.call(ctx, http.MethodPost,
		"/containers/"+entry.containerID+"/rename?name="+url.QueryEscape(name), nil, nil); err != nil {
		b.pool.discard(ctx, entry)
		return backend.Created{}, false
	}

	b.pool.adopt(entry)
	b.mu.Lock()
	b.known[entry.containerID] = entry.containerID
	b.mu.Unlock()
	return backend.Created{
		Handle: backend.Handle{Backend: b.Name(), SandboxID: entry.containerID, NodeID: nodeID},
		// A container has no in-sandbox agent, so the node agent proxies commands for
		// it. Every provider backend reports a real address here instead.
		Agent:        backend.AgentEndpoint{Address: ""},
		Capabilities: b.Capabilities(),
		// The caller is told this was warm, because a benchmark that cannot tell a
		// warm claim from a cold create cannot measure the pool.
		WarmStart: true,
	}, true
}

// noteMismatch counts a claim that was destroyed rather than handed over.
func (p *warmPool) noteMismatch() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.mismatches++
}

// PoolInventoryNames reports whether a container name marks unclaimed pool inventory.
func PoolInventoryNames(name string) bool { return isPoolInventoryName(name) }
