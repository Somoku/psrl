package server

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/cgroup"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/placement"
)

// Node is one machine's agent: admission, lifecycle, reclamation, and the
// report the fleet view is built from.
//
// It is the final authority on admission. A placement decision arrives here as
// a proposal, and this is where it meets the node's live pressure.
//
// It hosts every runtime installed on this machine rather than one, because the
// runtimes share the machine: a container daemon and a microVM node draw on the
// same memory, so one admission ledger has to cover both or the node admits
// twice what it has.
type Node struct {
	v1.UnimplementedSandboxNodeServer

	nodeID    string
	admission *node.Admission
	lifecycle *node.Lifecycle
	backends  map[string]backend.Backend
	// order is the declaration order, so a report lists runtimes the same way
	// twice and a default is the first declared.
	order []string

	// pressure reads what the machine is actually doing. Without it the node
	// would only know its own accounting, which is a record of reservations
	// rather than of load.
	pressure func() node.Pressure

	// elastic prices a sandbox from what its class actually costs and asks idle sandboxes
	// for their reclaimable memory. nil when the deployment has not enabled it, which
	// leaves admission on declared maxima.
	elastic *elastic
	// elasticDomain and elasticParent are where the controller reads per-sandbox usage and
	// lowers a limit. Without a domain it cannot do either, so it does nothing.
	elasticDomain cgroup.Domain
	elasticParent string
	// elasticBudget is the subtree's memory limit, the number every fraction is taken of.
	elasticBudget int64
}

// NodeConfig is what a node agent needs to run.
type NodeConfig struct {
	NodeID    string
	Admission *node.Admission
	Lifecycle *node.Lifecycle
	// Backends are every runtime this machine hosts, in declaration order.
	Backends []backend.Backend
	Pressure func() node.Pressure
	// Elastic turns on usage-based admission and the pressure-driven release. Disabled is
	// the previous behaviour: charge each sandbox its declared maximum.
	Elastic ElasticConfig
	// Domain and SandboxParent are the cgroup subtree the sandboxes live in. Without them
	// the controller has nothing to measure and stays inert.
	Domain        cgroup.Domain
	SandboxParent string
	// SandboxBudgetMB is that subtree's memory limit.
	SandboxBudgetMB int64
}

// NewNode returns the node-level server.
func NewNode(cfg NodeConfig) (*Node, error) {
	if cfg.NodeID == "" || cfg.Admission == nil || cfg.Lifecycle == nil || len(cfg.Backends) == 0 {
		return nil, fmt.Errorf("a node agent needs an id, admission, a lifecycle, and at least one backend")
	}
	backends := make(map[string]backend.Backend, len(cfg.Backends))
	order := make([]string, 0, len(cfg.Backends))
	for _, hosted := range cfg.Backends {
		if _, duplicate := backends[hosted.Name()]; duplicate {
			return nil, fmt.Errorf("node %q hosts backend %q twice; a runtime is keyed by its name",
				cfg.NodeID, hosted.Name())
		}
		backends[hosted.Name()] = hosted
		order = append(order, hosted.Name())
	}
	if cfg.Pressure == nil {
		cfg.Pressure = func() node.Pressure { return node.Pressure{} }
	}
	agent := &Node{
		nodeID:        cfg.NodeID,
		admission:     cfg.Admission,
		lifecycle:     cfg.Lifecycle,
		backends:      backends,
		order:         order,
		pressure:      cfg.Pressure,
		elasticDomain: cfg.Domain,
		elasticParent: cfg.SandboxParent,
		elasticBudget: cfg.SandboxBudgetMB,
	}
	if cfg.Elastic.Enabled {
		agent.elastic = &elastic{
			cfg:                   ElasticConfigWithDefaults(cfg.Elastic),
			memoryFootprints:      map[string]*node.Footprint{},
			cpuFootprints:         map[string]*node.Footprint{},
			maxMemoryMBBySandbox:  map[string]int64{},
			maxCPUMillisBySandbox: map[string]int64{},
			startMBBySandbox:      map[string]int64{},
			lowSince:              map[string]time.Time{},
			failCountBySandbox:    map[string]int64{},
			oomCountBySandbox:     map[string]int64{},
			exhausted:             map[string]bool{},
			classBySandbox:        map[string]string{},
			throttled:             map[string]bool{},
			cpuUsageNanos:         map[string]int64{},
			cpuRateMillis:         map[string]int64{},
			cpuThrottledNanos:     map[string]int64{},
			cpuThrottle:           map[string]float64{},
			previousPrice:         map[string]node.Resources{},
			classCount:            map[string]int{},
		}
		// Seed one estimate per declared class, so the node's first report already
		// publishes a price. Without it the batch admitted before the first report -- which
		// is every batch after a restart -- is charged the request's maximum, the
		// reservation this exists to remove.
		for class := range cfg.Admission.Snapshot().ClassHeadroom {
			agent.memoryFootprintFor(class)
			agent.cpuFootprintFor(class)
		}
	}
	return agent, nil
}

// Admit accepts or refuses one request against this node's live state.
//
// A refusal is returned as a reason rather than an error, because it is not a
// failure: the control plane reads it and asks placement for a different node.
func (n *Node) Admit(ctx context.Context, req *v1.AdmitRequest) (*v1.AdmitResponse, error) {
	spec := specFromProto(req.GetSpec())
	n.admission.SetPressure(n.pressure())
	owner := ownerOf(spec)
	// Charge what a sandbox of this class actually costs rather than its declared maximum.
	// The maximum stays as the kernel's hard bound for this sandbox; the ledger's job is to
	// know what the node has promised, and nothing has promised the maximum.
	//
	// CPU is charged the same way, and it is the dimension that decides how large a burst
	// the fleet admits: every sandbox declares the cores one tool call may want, and nearly
	// all of them are idle between calls.
	// Charge the class's measured price plus the growth it is committed to, because the
	// question admission answers is not "does this fit now" but "will it still fit when
	// it has grown, and will the ones before it".
	charge := n.elasticCharge(spec)
	if growth := n.admission.GrowthPerSandbox(spec.ResourceClass); growth.MemoryMB > 0 || growth.CPUMillis > 0 {
		if charge.MemoryMB > 0 {
			charge.MemoryMB += growth.MemoryMB
		}
		if charge.CPUMillis > 0 {
			charge.CPUMillis += growth.CPUMillis
		}
	}
	grant, refusal, ok := n.admission.Reserve(spec.ResourceClass, owner, charge)
	if !ok {
		// The refusal is what makes this class's guarantee real. Without it the node has
		// no record of a class waiting, so nothing is held back from the classes that are
		// running, and the first one to arrive can take the whole envelope -- which is how
		// grading sandboxes ended up un-admittable behind a full rollout wave.
		n.admission.Enqueue(spec.ResourceClass)
		n.admission.ExpireDemand()
		return &v1.AdmitResponse{Admitted: false, Refusal: string(refusal)}, nil
	}
	// One waiter of this class is served, so its claim on the guarantee is released. The
	// demand that is still waiting keeps the share held.
	n.admission.Dequeue(spec.ResourceClass)
	// The record is created from the grant, so a grant that never becomes a
	// container is still swept rather than held until the next restart.
	n.lifecycle.Reserve(grant, owner, spec.ExecMode, lifetimeOf(spec))
	return &v1.AdmitResponse{Admitted: true, LeaseId: grant.LeaseID, GpuIndices: grant.GPUIndices}, nil
}

// CreateOn provisions a sandbox against an admission this node already granted.
//
// The lease must exist: creating without one would put a sandbox on the node
// that its own accounting never charged. The runtime is named by the caller
// rather than chosen here, because routing already matched the spec's
// requirements against one and a second choice could disagree with it.
func (n *Node) CreateOn(ctx context.Context, req *v1.CreateOnRequest) (*v1.CreateResponse, error) {
	leaseID := req.GetLeaseId()
	grant, claimed := n.lifecycle.Claim(leaseID)
	if !claimed {
		return nil, status.Errorf(codes.FailedPrecondition,
			"lease %q was not granted by this node or has already been used, "+
				"so a sandbox created against it would not be charged", leaseID)
	}
	// The record is held in flight for as long as this runs, so the sweep's clock cannot
	// reap a create that is still pulling its image.
	release := n.lifecycle.BeginCreate(leaseID)
	defer release()
	hosted, hosts := n.hosted(req.GetBackend())
	if !hosts {
		// No sandbox exists, so the reserved budget must not stay held.
		n.lifecycle.FailCreate(leaseID)
		return nil, status.Errorf(codes.FailedPrecondition,
			"node %q does not host backend %q (hosts: %v)", n.nodeID, req.GetBackend(), n.order)
	}
	spec := specFromProto(req.GetSpec())
	spec.AssignedGPUs = grant.GPUIndices
	spec.Lifetime = lifetimeOf(spec)
	// The container carries the owner as a label, so a restarted node recovers it
	// from the daemon alone.
	owner := ownerOf(spec)
	if spec.Metadata == nil {
		spec.Metadata = map[string]string{}
	}
	spec.Metadata[backend.OwnerMetadataKey] = owner

	// The container is born under its guarantee, not under its declared maximum. Doing it
	// here rather than by lowering the limit afterwards means there is no window in which
	// the sandbox is bounded by a number the node has not agreed to back, and it is the
	// only point at which the *creation* limit and the *ceiling* can be different numbers
	// without a second write racing the sandbox's own first allocations.
	spec = n.applyStartLimit(spec)

	if handled, found, err := n.adoptByIdempotencyKey(ctx, hosted, leaseID, spec.IdempotencyKey); found {
		return handled, err
	}

	created, err := hosted.Create(ctx, n.nodeID, spec, req.GetCallbackTarget())
	if err != nil {
		// The sandbox does not exist, so its admission must not stay charged.
		n.lifecycle.FailCreate(leaseID)
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	// The kernel limit is the declared maximum even though the ledger charges the class's
	// estimate, so the controller needs both numbers -- for CPU too, because a throttled
	// quota is restored to what the sandbox declared.
	n.elasticRemember(created.Handle.SandboxID, spec)
	if !n.lifecycle.AdoptCreated(leaseID, created.Handle, hosted) {
		// The record was reclaimed while the create was in flight, so this
		// container has no owner to charge it to. Destroy it on a budget of its
		// own rather than leak it or release capacity it still holds.
		return nil, destroyReclaimed(hosted, created.Handle)
	}
	return createdToProto(created), nil
}

// adoptByIdempotencyKey resolves a create whose key names a container that
// already exists. The second result is whether this retry is resolved by the
// answer, which is true both when a container was found and when asking failed:
// either way the caller must not go on to create one.
//
// A create whose response was lost leaves a container the caller never got a
// handle for, and its key is on the container. So the recovery is to ask the
// runtime rather than to build a second container, which would double the
// memory a single logical sandbox holds.
//
// The reservation is the thing that has to be unwound. When the node still owns
// the container, the record from the first attempt is the authority and this
// retry's grant is returned, so one container stays charged once. When it does
// not, the container outlived a restart and this retry is its first record.
func (n *Node) adoptByIdempotencyKey(
	ctx context.Context, hosted backend.Backend, leaseID, key string,
) (*v1.CreateResponse, bool, error) {
	if key == "" {
		return nil, false, nil
	}
	finder, canFind := hosted.(backend.Reconciler)
	if !canFind {
		// A runtime that cannot look an object up by its key cannot be asked, and
		// a missed retry is a smaller fault than a create that never happens.
		return nil, false, nil
	}
	found, exists, err := finder.FindByIdempotencyKey(ctx, key)
	if err != nil {
		n.lifecycle.FailCreate(leaseID)
		return nil, true, status.Error(codes.Unavailable, err.Error())
	}
	if !exists {
		return nil, false, nil
	}
	if owned, hostedBy, held := n.lifecycle.Lookup(found.Handle.SandboxID); held {
		n.lifecycle.FailCreate(leaseID)
		return createdToProto(backend.Created{
			Handle: owned, Capabilities: hostedBy.Capabilities(),
		}), true, nil
	}
	if !n.lifecycle.AdoptCreated(leaseID, found.Handle, hosted) {
		return nil, true, destroyReclaimed(hosted, found.Handle)
	}
	return createdToProto(backend.Created{
		Handle: found.Handle, Capabilities: hosted.Capabilities(),
	}), true, nil
}

// destroyReclaimed destroys a container whose record was reclaimed while the
// create was in flight, so a create that lost its race leaks nothing and
// returns no capacity the container still holds.
func destroyReclaimed(hosted backend.Backend, handle backend.Handle) error {
	cleanupCtx, cancel := context.WithTimeout(context.Background(), createCleanupTimeout)
	defer cancel()
	if err := hosted.Release(cleanupCtx, handle); err != nil {
		return status.Errorf(codes.Internal,
			"sandbox %s was reclaimed while its create was in flight and could not be destroyed: %v",
			handle.SandboxID, err)
	}
	return status.Errorf(codes.Aborted,
		"sandbox %s was reclaimed while its create was in flight", handle.SandboxID)
}

// createCleanupTimeout bounds destroying a sandbox whose create lost its record.
const createCleanupTimeout = 30 * time.Second

// lifetimeOf is how long a sandbox may exist, defaulting a spec that states none
// so a caller cannot create an unbounded one.
func lifetimeOf(spec backend.Spec) time.Duration {
	if spec.Lifetime > 0 {
		return spec.Lifetime
	}
	return node.DefaultLifetime
}

// hosted returns one runtime by name. An empty name takes the first declared,
// which is what a single-runtime node reports.
func (n *Node) hosted(name string) (backend.Backend, bool) {
	if name == "" {
		return n.backends[n.order[0]], true
	}
	hosted, hosts := n.backends[name]
	return hosted, hosts
}

// ReleaseOn destroys one sandbox this node holds.
func (n *Node) ReleaseOn(ctx context.Context, handle *v1.SandboxHandle) (*v1.Empty, error) {
	if err := n.lifecycle.Release(ctx, handle.GetSandboxId(), node.ExitReleased); err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	return &v1.Empty{}, nil
}

// LedgerAudit reports whether the node's charge can still be accounted for by its
// envelope.
//
// Granted above Envelope is not a full node. Every admission is checked against the
// headroom that is left, so the charge can only pass the envelope if something changed
// it without going through an admission -- which means the ledger is charging capacity
// to leases it no longer holds, and every request from then on is refused against
// capacity nobody is using. It is reported rather than corrected: guessing at which
// lease is wrong would trade a visible fault for an invisible one.
func (n *Node) LedgerAudit() (granted node.Resources, envelope node.Resources, accounted bool) {
	report := n.admission.Snapshot()
	return report.Granted, report.Envelope, report.Granted.FitsIn(report.Envelope)
}

// Report is this node's view of itself, which the monitor collects.
//
// Headroom is reported per class because that is the granularity admission
// works at: a scheduler given the envelope remainder sends work this node will
// refuse, and the mismatch shows up only as sandboxes that never start.
func (n *Node) Report(context.Context, *v1.Empty) (*v1.NodeView, error) {
	return n.view(), nil
}

func (n *Node) view() *v1.NodeView {
	// One sample per report, used for both admission and the wire fields. Calling it
	// twice made the reported CPU figure a rate over the microseconds between two
	// RPC-internal calls, which is noise, and it disagreed with what admission stored.
	pressure := n.pressure()
	n.admission.SetPressure(pressure)
	report := n.admission.Snapshot()
	life := n.lifecycle.Snapshot()
	out := &v1.NodeView{
		NodeId:        n.nodeID,
		ClassHeadroom: map[string]*v1.Headroom{},
		Envelope:      headroomToProto(report.Envelope),
		LiveSandboxes: int32(life.Resident),
		CpuUsedPct:    pressure.CPUUsedPct,
		MemUsedPct:    pressure.MemUsedPct,
		GpuFree:       int32(report.FreeGPUs),
		Draining:      report.Draining,
	}
	for class, headroom := range n.admission.ClassHeadroom() {
		out.ClassHeadroom[class] = headroomToProto(headroom)
	}
	// Per runtime, because a node hosting a container daemon and a microVM
	// satisfies a full-state resume through one of them and not the other.
	for _, name := range n.order {
		capabilities := n.backends[name].Capabilities()
		hosted := &v1.BackendView{
			Name:        name,
			ResumeLevel: resumeLevelValue(capabilities.ResumeLevel),
		}
		for _, feature := range capabilities.Features {
			hosted.Features = append(hosted.Features, featureValue(feature))
			if feature == "host_mount" {
				hosted.HostMounts = true
			}
		}
		out.Backends = append(out.Backends, hosted)
	}
	return out
}

// View returns this node's state as placement reads it, for an in-process
// deployment where the monitor and the node share a process.
func (n *Node) View() placement.NodeView {
	proto := n.view()
	view := placement.NodeView{
		NodeID:                proto.GetNodeId(),
		SeenAt:                time.Now(),
		ClassHeadroom:         map[string]placement.Headroom{},
		LiveSandboxes:         int(proto.GetLiveSandboxes()),
		CPUUsedPct:            proto.GetCpuUsedPct(),
		MemUsedPct:            proto.GetMemUsedPct(),
		GPUFree:               proto.GetGpuFree(),
		Draining:              proto.GetDraining(),
		ImageDigests:          map[string]struct{}{},
		ImageReferences:       map[string]struct{}{},
		Labels:                map[string]struct{}{},
		ClassReserveMB:        n.classMemoryReservesMB(),
		ClassReserveCPUMillis: n.classCPUReservesMillis(),
	}
	if envelope := proto.GetEnvelope(); envelope != nil {
		view.Envelope = placement.Headroom{
			MemoryMB: envelope.GetMemoryMb(), CPUMillis: envelope.GetCpuMillis(),
			GPUCount: envelope.GetGpuCount(), DiskMB: envelope.GetDiskMb(),
		}
	}
	for class, headroom := range proto.GetClassHeadroom() {
		view.ClassHeadroom[class] = placement.Headroom{
			MemoryMB: headroom.GetMemoryMb(), CPUMillis: headroom.GetCpuMillis(),
			GPUCount: headroom.GetGpuCount(), DiskMB: headroom.GetDiskMb(),
		}
	}
	for _, name := range n.order {
		capabilities := n.backends[name].Capabilities()
		hosted := placement.BackendCapability{
			Name:        name,
			Features:    make(map[string]struct{}, len(capabilities.Features)),
			ResumeLevel: capabilities.ResumeLevel,
		}
		for _, feature := range capabilities.Features {
			hosted.Features[feature] = struct{}{}
			if feature == "host_mount" {
				hosted.HostMounts = true
			}
		}
		view.Backends = append(view.Backends, hosted)
	}
	return view
}

// Sweep runs one reclamation pass on demand, for an operator or a test. The
// node also sweeps on its own cadence.
func (n *Node) Sweep(ctx context.Context, _ *v1.Empty) (*v1.SweepReport, error) {
	report := n.lifecycle.Sweep(ctx)
	return &v1.SweepReport{
		ReapedLifetime:  report.ReapedLifetime,
		ReclaimedOrphan: report.ReclaimedOrphan,
		SkippedBusy:     int32(report.SkippedBusy),
		Failures:        int32(report.Failures),
	}, nil
}

// sessionGoneError is a command against a sandbox this node no longer holds.
//
// It carries the lifecycle's attribution when one is still known, and maps to
// gRPC NotFound so an SDK classifies it as a lost session rather than a
// transport fault. `reason` is empty when the node never held the sandbox.
type sessionGoneError struct {
	sandboxID string
	nodeID    string
	reason    string
}

func (e *sessionGoneError) Error() string {
	if e.reason != "" {
		return fmt.Sprintf("sandbox %q is not held by node %q (exit_reason=%s)",
			e.sandboxID, e.nodeID, e.reason)
	}
	return fmt.Sprintf("sandbox %q is not held by node %q", e.sandboxID, e.nodeID)
}

// ExitReason exposes the attribution to the wire layer without that layer
// having to type-switch on this package's concrete error.
func (e *sessionGoneError) ExitReason() string { return e.reason }

func (e *sessionGoneError) GRPCStatus() *status.Status {
	return status.New(codes.NotFound, e.Error())
}

// Exec runs one command in a sandbox this node holds.
//
// It routes to the exec strategy the sandbox was created with. A persistent
// shell keeps shell state (working directory, environment, activated venv)
// alive between calls, which is what a multi-turn harness needs. One-shot
// starts a fresh process per call, which is cheaper and correct for a grader.
// The strategy is chosen at sandbox creation time and stored on the lifecycle
// resident so this path cannot diverge from what the spec asked for.
func (n *Node) Exec(ctx context.Context, handle backend.Handle, command, workdir string, env map[string]string) (int, string, error) {
	_, held, owned := n.lifecycle.Lookup(handle.SandboxID)
	if !owned {
		// The sandbox is gone. Report why it left when the lifecycle still
		// remembers, so the caller can tell a reaped-idle sandbox from one that
		// was never here -- and gets a lost-session rather than a transport
		// fault.
		reason, _ := n.lifecycle.ExitReasonOf(handle.SandboxID)
		return 0, "", &sessionGoneError{
			sandboxID: handle.SandboxID,
			nodeID:    n.nodeID,
			reason:    string(reason),
		}
	}
	// A backend whose sandboxes run their own in-sandbox agent is never reached
	// here — the SDK contacts the agent directly.
	type execModer interface {
		ExecWithMode(context.Context, backend.Handle, string, string, string, map[string]string) (int, string, error)
	}
	// Mark the sandbox busy for the whole command, so the reclamation sweep
	// cannot read a long command as idle and pause a running test suite.
	done := n.lifecycle.Begin(handle.SandboxID)
	defer done()
	if moded, ok := held.(execModer); ok {
		execMode := n.lifecycle.ExecMode(handle.SandboxID)
		return moded.ExecWithMode(ctx, handle, execMode, command, workdir, env)
	}
	runner, canRun := held.(interface {
		Exec(context.Context, backend.Handle, string, string, map[string]string) (int, string, error)
	})
	if !canRun {
		return 0, "", fmt.Errorf("backend %q runs commands through its own in-sandbox agent", held.Name())
	}
	return runner.Exec(ctx, handle, command, workdir, env)
}

// Touch records activity on a sandbox this node holds.
//
// The SDK calls it for a sandbox whose commands go straight to an in-container
// agent, which never reaches Exec; without it the reclamation sweep would judge
// that sandbox idle from creation and pause it mid-episode.
func (n *Node) Touch(handle backend.Handle) {
	n.lifecycle.Touch(handle.SandboxID)
}

// RenewOwner records that an owner is still using this node's sandboxes.
//
// The reclamation sweep reads owner liveness from the admission gate, so a
// heartbeat is what keeps a live caller's sandboxes from being reclaimed as
// orphans. An owner this node never admitted is ignored: a renewal for another
// node's caller must not resurrect anything here.
func (n *Node) RenewOwner(owner string) {
	n.admission.Renew(owner)
}

// StatusOn reports a sandbox's state from this node.
func (n *Node) StatusOn(ctx context.Context, handle backend.Handle) (string, error) {
	stored, held, owned := n.lifecycle.Lookup(handle.SandboxID)
	if !owned {
		return "terminated", nil
	}
	return held.Status(ctx, stored)
}

// ownerOf names the caller whose liveness decides when a sandbox is an orphan.
//
// The caller states it in the spec's metadata, because that is the identity the
// reclaimer is keyed on and nothing else on the spec carries it: `workflow_id`
// reserves a scheduling phase, which a caller may share across sandboxes or not
// set at all, and reading ownership out of it left every sandbox owned by
// "anonymous", an owner that never renews, so a silence budget would reclaim
// live work. `workflow_id` stays a fallback for a caller that has no separate
// owner and means its workflow to be one.
func ownerOf(spec backend.Spec) string {
	if owner := spec.Metadata[backend.OwnerMetadataKey]; owner != "" {
		return owner
	}
	if spec.WorkflowID != "" {
		return spec.WorkflowID
	}
	return "anonymous"
}
