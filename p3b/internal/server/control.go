package server

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/monitor"
	"psrl.dev/sandboxd/internal/placement"
	"psrl.dev/sandboxd/internal/quota"
)

// NodeClient reaches one node agent. It is an interface so the control plane can
// be driven against an in-process node in a test and over gRPC in a cluster,
// with the same code path either way.
type NodeClient interface {
	Admit(ctx context.Context, nodeID string, spec backend.Spec) (leaseID string, gpus []int32, refusal string, err error)
	// CreateOn names the runtime to provision on, because one node can host
	// several and routing already matched the spec against one of them.
	CreateOn(
		ctx context.Context, nodeID, leaseID, backendName string, spec backend.Spec, callback string,
	) (backend.Created, error)
	ReleaseOn(ctx context.Context, handle backend.Handle) error
	StatusOn(ctx context.Context, handle backend.Handle) (string, error)

	// ExecOn, ReadBytesOn, and WriteBytesOn route command and file traffic to
	// the node that holds the sandbox. In a single-process deployment they call
	// the node directly; in a fleet they reach the right machine over the
	// node-plane protocol so the control plane never proxies data traffic.
	ExecOn(ctx context.Context, handle backend.Handle, command, cwd string, env map[string]string) (int, string, error)
	ReadBytesOn(ctx context.Context, handle backend.Handle, path string) (string, error)
	WriteBytesOn(ctx context.Context, handle backend.Handle, path, data string) error
}

// Control is the cluster-level service.
//
// It routes, admits against the cross-backend quota, places, and then asks the
// chosen node to accept. The node keeps the last word, so a refusal here is a
// reason to ask placement again rather than a failed episode.
type Control struct {
	v1.UnimplementedSandboxControlServer

	registry  *backend.Registry
	ledger    *quota.Ledger
	placement *placement.Service
	monitor   *monitor.Monitor
	nodes     NodeClient

	// placementAttempts bounds how many nodes one create will try before giving
	// up. A node refusing is normal under load; an unbounded retry would hide a
	// fleet that is genuinely full behind a slow create.
	placementAttempts int
	acquireTimeout    time.Duration
	// admitPollInterval is how often a queued request re-tries its class. Short
	// against a sandbox lifetime, so a freed share is picked up promptly, and long
	// enough that a queue of waiters is not a spin on the ledger's lock.
	admitPollInterval time.Duration

	mu       sync.Mutex
	sandbox  map[string]sandboxRecord
	ownerSeq uint64
}

type sandboxRecord struct {
	handle        backend.Handle
	backend       backend.Backend
	grantID       string
	reservationID string
	class         string
}

// ControlConfig is what the control plane needs to run.
type ControlConfig struct {
	Registry          *backend.Registry
	Ledger            *quota.Ledger
	Placement         *placement.Service
	Monitor           *monitor.Monitor
	Nodes             NodeClient
	PlacementAttempts int
	AcquireTimeout    time.Duration
	// AdmitPollInterval overrides how often a request queued on a full class
	// re-tries. Zero takes the default.
	AdmitPollInterval time.Duration
}

// NewControl returns the cluster-level server.
func NewControl(cfg ControlConfig) (*Control, error) {
	if cfg.Registry == nil || cfg.Ledger == nil || cfg.Placement == nil || cfg.Monitor == nil {
		return nil, fmt.Errorf("the control plane needs a registry, a ledger, placement, and a monitor")
	}
	if cfg.PlacementAttempts <= 0 {
		cfg.PlacementAttempts = 3
	}
	if cfg.AcquireTimeout <= 0 {
		cfg.AcquireTimeout = 30 * time.Minute
	}
	if cfg.AdmitPollInterval <= 0 {
		cfg.AdmitPollInterval = 250 * time.Millisecond
	}
	return &Control{
		registry:          cfg.Registry,
		ledger:            cfg.Ledger,
		placement:         cfg.Placement,
		monitor:           cfg.Monitor,
		nodes:             cfg.Nodes,
		placementAttempts: cfg.PlacementAttempts,
		acquireTimeout:    cfg.AcquireTimeout,
		admitPollInterval: cfg.AdmitPollInterval,
		sandbox:           map[string]sandboxRecord{},
	}, nil
}

// Create provisions one sandbox: quota, then routing, then placement, then the
// node's own admission.
//
// Every failure path returns what it took. A create that gave up still holding
// a grant or a reservation would starve the fleet one slot at a time.
func (c *Control) Create(ctx context.Context, req *v1.CreateRequest) (*v1.CreateResponse, error) {
	spec := specFromProto(req.GetSpec())
	selected, err := c.registry.Select(spec)
	if err != nil {
		// No backend can serve this spec. Waiting will not add a capability.
		return nil, status.Error(codes.FailedPrecondition, err.Error())
	}

	owner := c.ownerFor(req.GetCallbackTarget())
	grantID := fmt.Sprintf("grant-%d", time.Now().UnixNano())
	amount := quotaAmount(spec)
	if err := c.admit(ctx, grantID, spec.ResourceClass, owner, amount); err != nil {
		return nil, err
	}

	created, err := c.place(ctx, selected, spec, req.GetCallbackTarget(), owner)
	if err != nil {
		c.ledger.Release(grantID)
		return nil, err
	}
	c.remember(created.handle.SandboxID, sandboxRecord{
		handle: created.handle, backend: selected,
		grantID: grantID, reservationID: created.reservationID, class: spec.ResourceClass,
	})
	return createdToProto(created.created), nil
}

// admit charges one request against its class, waiting for room when the class
// is momentarily full.
//
// A class at its share is a queue, not a verdict: the share clears as sandboxes
// are released, and the request is for capacity that will exist rather than for a
// capability that never will. Returning ResourceExhausted immediately made the
// caller's retry the queue, and a caller that treats the refusal as terminal --
// a grader step, which cannot be retried without redoing the rollout -- loses the
// episode to a wait it was never offered. The wait is bounded by AcquireTimeout,
// which is derived from the episode deadline for exactly this purpose: a queue
// longer than an episode is a fault rather than contention.
//
// The demand is enqueued for the whole wait, which is what reserves the class's
// guarantee against borrowers while it waits. It is withdrawn on every exit path,
// so a caller that gave up does not leave the class reserving room for a request
// that no longer exists.
func (c *Control) admit(ctx context.Context, grantID, class, owner string, amount quota.Amount) error {
	admitted, err := c.ledger.Acquire(grantID, class, owner, amount)
	if err != nil {
		return status.Error(codes.Internal, err.Error())
	}
	if admitted {
		return nil
	}

	c.ledger.Enqueue(class, amount)
	defer c.ledger.Cancel(class, amount)

	// Polled rather than signalled: a release happens on another caller's
	// goroutine and the ledger publishes no readiness channel, so the choice is
	// between a poll here and a condition variable threaded through every release
	// path. The interval is short against a sandbox lifetime and the waiters are
	// few, so the poll costs a lock acquisition per class per interval.
	ticker := time.NewTicker(c.admitPollInterval)
	defer ticker.Stop()
	deadline := time.Now().Add(c.acquireTimeout)

	for {
		select {
		case <-ctx.Done():
			// The caller's own deadline fired. Its error is the honest one: nothing
			// about the fleet is known to be wrong.
			return status.FromContextError(ctx.Err()).Err()
		case <-ticker.C:
			admitted, err := c.ledger.Acquire(grantID, class, owner, amount)
			if err != nil {
				return status.Error(codes.Internal, err.Error())
			}
			if admitted {
				return nil
			}
			if time.Now().After(deadline) {
				return status.Errorf(codes.ResourceExhausted,
					"the %q class did not come free within %s; this clears as sandboxes are released",
					class, c.acquireTimeout)
			}
		}
	}
}

type placed struct {
	created       backend.Created
	handle        backend.Handle
	reservationID string
}

// place chooses a node and asks it to accept, trying another node when one
// refuses.
//
// A node refusal is information rather than an error: it knows its own pressure
// and placement only knows what the node last reported, so the right response is
// to ask placement again with that node now charged.
func (c *Control) place(ctx context.Context, selected backend.Backend, spec backend.Spec, callback, owner string) (placed, error) {
	if selected.Mode() == backend.SchedulingProvider {
		// The backend's own control plane chooses the node, so this service does
		// quota only and passes an empty node id.
		created, err := selected.Create(ctx, "", spec, callback)
		if err != nil {
			return placed{}, status.Error(codes.Unavailable, err.Error())
		}
		return placed{created: created, handle: created.Handle}, nil
	}

	var lastRefusal string
	for attempt := 0; attempt < c.placementAttempts; attempt++ {
		decision, err := c.placement.Choose(placementRequest(spec, selected.Name(), owner))
		if err != nil {
			if errors.Is(err, placement.ErrExhausted) {
				return placed{}, status.Error(codes.ResourceExhausted, err.Error())
			}
			return placed{}, status.Error(codes.FailedPrecondition, err.Error())
		}
		leaseID, gpus, refusal, err := c.nodes.Admit(ctx, decision.NodeID, spec)
		if err != nil {
			c.placement.Cancel(decision.ReservationID)
			return placed{}, status.Errorf(codes.Unavailable, "node %s did not answer: %v", decision.NodeID, err)
		}
		if refusal != "" {
			// Try elsewhere. The reservation is withdrawn so the node is not held for
			// a sandbox it just declined.
			c.placement.Cancel(decision.ReservationID)
			lastRefusal = refusal
			continue
		}
		withDevices := spec
		withDevices.AssignedGPUs = gpus
		// The runtime placement matched on, not the one routing picked: on a node
		// hosting several they agree, and where they could not, placement's is the
		// one whose capabilities were checked against this node.
		created, err := c.nodes.CreateOn(ctx, decision.NodeID, leaseID, decision.Backend, withDevices, callback)
		if err != nil {
			c.placement.Cancel(decision.ReservationID)
			return placed{}, status.Errorf(codes.Unavailable, "node %s could not create: %v", decision.NodeID, err)
		}
		return placed{created: created, handle: created.Handle, reservationID: decision.ReservationID}, nil
	}
	return placed{}, status.Errorf(codes.ResourceExhausted,
		"every node placement tried refused this request (last reason: %q); this clears as sandboxes are released",
		lastRefusal)
}

// CreateGroup provisions one group.
//
// Members are admitted independently, because a group is a completion unit
// rather than a scheduling unit: the algorithm needs its trajectories to finish
// together, not to start together. No member holds capacity while a sibling
// queues, which is what keeps two concurrent groups from starving each other.
func (c *Control) CreateGroup(ctx context.Context, req *v1.CreateGroupRequest) (*v1.CreateGroupResponse, error) {
	specs := req.GetSpecs()
	if len(specs) == 0 {
		return nil, status.Error(codes.InvalidArgument, "a sandbox group needs at least one member spec")
	}
	out := &v1.CreateGroupResponse{}
	created := make([]*v1.SandboxHandle, 0, len(specs))
	for _, spec := range specs {
		member, err := c.Create(ctx, &v1.CreateRequest{Spec: spec, CallbackTarget: req.GetCallbackTarget()})
		if err != nil {
			// A failed group leaves nothing charged.
			for _, handle := range created {
				_, _ = c.Release(ctx, handle)
			}
			return nil, err
		}
		created = append(created, member.GetHandle())
		out.Members = append(out.Members, member)
	}
	return out, nil
}

// Release destroys one sandbox and returns everything it held.
func (c *Control) Release(ctx context.Context, handle *v1.SandboxHandle) (*v1.Empty, error) {
	record, known := c.forget(handle.GetSandboxId())
	if !known {
		// Already gone is the outcome the caller wanted.
		return &v1.Empty{}, nil
	}
	var err error
	if record.backend.Mode() == backend.SchedulingProvider {
		err = record.backend.Release(ctx, record.handle)
	} else {
		err = c.nodes.ReleaseOn(ctx, record.handle)
	}
	if err != nil {
		// The sandbox may still exist, so put it back: a caller must be able to
		// retry, and the quota must stay charged until destruction is confirmed.
		c.remember(handle.GetSandboxId(), record)
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	if record.reservationID != "" {
		c.placement.Release(record.reservationID)
	}
	c.ledger.Release(record.grantID)
	return &v1.Empty{}, nil
}

// Status reports one sandbox's portable state.
func (c *Control) Status(ctx context.Context, handle *v1.SandboxHandle) (*v1.StatusResponse, error) {
	record, known := c.lookup(handle.GetSandboxId())
	if !known {
		return &v1.StatusResponse{Status: v1.SandboxStatus_TERMINATED}, nil
	}
	var (
		state string
		err   error
	)
	if record.backend.Mode() == backend.SchedulingProvider {
		state, err = record.backend.Status(ctx, record.handle)
	} else {
		state, err = c.nodes.StatusOn(ctx, record.handle)
	}
	if err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	return &v1.StatusResponse{Status: statusValue(state)}, nil
}

// Pause releases a sandbox's compute while keeping its state.
func (c *Control) Pause(ctx context.Context, req *v1.PauseRequest) (*v1.Empty, error) {
	record, stateful, err := c.stateful(req.GetHandle())
	if err != nil {
		return nil, err
	}
	if err := stateful.Pause(ctx, record.handle, pauseModeName(req.GetMode())); err != nil {
		return nil, status.Error(codes.FailedPrecondition, err.Error())
	}
	return &v1.Empty{}, nil
}

// Resume restarts a paused sandbox.
func (c *Control) Resume(ctx context.Context, handle *v1.SandboxHandle) (*v1.Empty, error) {
	record, stateful, err := c.stateful(handle)
	if err != nil {
		return nil, err
	}
	if err := stateful.Resume(ctx, record.handle); err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	return &v1.Empty{}, nil
}

// Snapshot captures a sandbox's state.
func (c *Control) Snapshot(ctx context.Context, req *v1.SnapshotRequest) (*v1.SnapshotRef, error) {
	record, stateful, err := c.stateful(req.GetHandle())
	if err != nil {
		return nil, err
	}
	kind := snapshotKindName(req.GetKind())
	id, err := stateful.Snapshot(ctx, record.handle, kind)
	if err != nil {
		return nil, status.Error(codes.FailedPrecondition, err.Error())
	}
	return &v1.SnapshotRef{Backend: record.handle.Backend, SnapshotId: id, Kind: req.GetKind()}, nil
}

// DeleteSnapshot removes a captured state.
func (c *Control) DeleteSnapshot(ctx context.Context, ref *v1.SnapshotRef) (*v1.Empty, error) {
	selected, err := c.registry.Get(ref.GetBackend())
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	stateful, ok := selected.(backend.Stateful)
	if !ok {
		return nil, status.Errorf(codes.FailedPrecondition, "backend %q keeps no snapshots", selected.Name())
	}
	if err := stateful.DeleteSnapshot(ctx, ref.GetSnapshotId()); err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	return &v1.Empty{}, nil
}

func (c *Control) stateful(handle *v1.SandboxHandle) (sandboxRecord, backend.Stateful, error) {
	record, known := c.lookup(handle.GetSandboxId())
	if !known {
		return sandboxRecord{}, nil, status.Errorf(codes.NotFound, "sandbox %q is not owned by this service", handle.GetSandboxId())
	}
	stateful, ok := record.backend.(backend.Stateful)
	if !ok {
		return sandboxRecord{}, nil, status.Errorf(codes.FailedPrecondition,
			"backend %q has no state operations, so this cannot be served rather than being served weakly",
			record.backend.Name())
	}
	return record, stateful, nil
}

// Fleet reports what the monitor sees, for the caller's metric hook.
func (c *Control) Fleet(context.Context, *v1.Empty) (*v1.FleetReport, error) {
	report := c.placement.Snapshot()
	out := &v1.FleetReport{
		DrainedNodes:      int32(report.DrainedNodes),
		ReservationsOpen:  int32(report.ReservationsOpen),
		CapacityExhausted: report.CapacityExhausted,
		NoCandidate:       report.NoCandidate,
		LocalityHitRatio:  report.LocalityHitRatio,
	}
	for _, view := range c.monitor.Fleet() {
		reported := &v1.NodeView{
			NodeId:        view.NodeID,
			LiveSandboxes: int32(view.LiveSandboxes),
			CpuUsedPct:    view.CPUUsedPct,
			MemUsedPct:    view.MemUsedPct,
			GpuFree:       view.GPUFree,
			Draining:      view.Draining,
		}
		for _, hosted := range view.Backends {
			reported.Backends = append(reported.Backends, &v1.BackendView{Name: hosted.Name})
		}
		out.Nodes = append(out.Nodes, reported)
	}
	return out, nil
}

// Quota reports the cross-backend ledger, for the caller's metric hook.
func (c *Control) Quota(context.Context, *v1.Empty) (*v1.QuotaReport, error) {
	out := &v1.QuotaReport{Classes: map[string]*v1.ClassQuota{}}
	for name, report := range c.ledger.Report() {
		out.Classes[name] = &v1.ClassQuota{
			GuaranteedShare: report.Guaranteed,
			MaxShare:        report.Max,
			Granted:         quotaHeadroomToProto(report.Granted),
			Headroom:        quotaHeadroomToProto(report.Headroom),
			Queued:          int32(report.Queued),
		}
	}
	return out, nil
}

func (c *Control) ownerFor(callback string) string {
	if callback != "" {
		return callback
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ownerSeq++
	return fmt.Sprintf("anonymous-%d", c.ownerSeq)
}

func (c *Control) remember(id string, record sandboxRecord) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sandbox[id] = record
}

func (c *Control) lookup(id string) (sandboxRecord, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	record, known := c.sandbox[id]
	return record, known
}

func (c *Control) forget(id string) (sandboxRecord, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	record, known := c.sandbox[id]
	delete(c.sandbox, id)
	return record, known
}

// RenewOwner keeps one caller's grants and reservations alive.
func (c *Control) RenewOwner(owner string) {
	c.ledger.Renew(owner)
}
