package server

import (
	"context"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/placement"
)

// Node is one machine's agent: admission, lifecycle, reclamation, and the
// report the fleet view is built from.
//
// It is the final authority on admission. A placement decision arrives here as
// a proposal, and this is where it meets the node's live pressure.
type Node struct {
	v1.UnimplementedSandboxNodeServer

	nodeID    string
	admission *node.Admission
	lifecycle *node.Lifecycle
	backend   backend.Backend

	// pressure reads what the machine is actually doing. Without it the node
	// would only know its own accounting, which is a record of reservations
	// rather than of load.
	pressure func() node.Pressure

	mu     sync.Mutex
	leases map[string]node.Grant
}

// NodeConfig is what a node agent needs to run.
type NodeConfig struct {
	NodeID    string
	Admission *node.Admission
	Lifecycle *node.Lifecycle
	Backend   backend.Backend
	Pressure  func() node.Pressure
}

// NewNode returns the node-level server.
func NewNode(cfg NodeConfig) (*Node, error) {
	if cfg.NodeID == "" || cfg.Admission == nil || cfg.Lifecycle == nil || cfg.Backend == nil {
		return nil, fmt.Errorf("a node agent needs an id, admission, a lifecycle, and a backend")
	}
	if cfg.Pressure == nil {
		cfg.Pressure = func() node.Pressure { return node.Pressure{} }
	}
	return &Node{
		nodeID:    cfg.NodeID,
		admission: cfg.Admission,
		lifecycle: cfg.Lifecycle,
		backend:   cfg.Backend,
		pressure:  cfg.Pressure,
		leases:    map[string]node.Grant{},
	}, nil
}

// Admit accepts or refuses one request against this node's live state.
//
// A refusal is returned as a reason rather than an error, because it is not a
// failure: the control plane reads it and asks placement for a different node.
func (n *Node) Admit(ctx context.Context, req *v1.AdmitRequest) (*v1.AdmitResponse, error) {
	spec := specFromProto(req.GetSpec())
	n.admission.SetPressure(n.pressure())
	grant, refusal, ok := n.admission.Admit(spec.ResourceClass, ownerOf(spec), nodeResources(spec))
	if !ok {
		return &v1.AdmitResponse{Admitted: false, Refusal: string(refusal)}, nil
	}
	n.mu.Lock()
	n.leases[grant.LeaseID] = grant
	n.mu.Unlock()
	return &v1.AdmitResponse{Admitted: true, LeaseId: grant.LeaseID, GpuIndices: grant.GPUIndices}, nil
}

// CreateOn provisions a sandbox against an admission this node already granted.
//
// The lease must exist: creating without one would put a sandbox on the node
// that its own accounting never charged.
func (n *Node) CreateOn(ctx context.Context, req *v1.CreateOnRequest) (*v1.CreateResponse, error) {
	n.mu.Lock()
	grant, granted := n.leases[req.GetLeaseId()]
	n.mu.Unlock()
	if !granted {
		return nil, status.Errorf(codes.FailedPrecondition,
			"lease %q was not granted by this node, so a sandbox created against it would not be charged",
			req.GetLeaseId())
	}
	spec := specFromProto(req.GetSpec())
	spec.AssignedGPUs = grant.GPUIndices
	created, err := n.backend.Create(ctx, n.nodeID, spec, req.GetCallbackTarget())
	if err != nil {
		// The sandbox does not exist, so its admission must not stay charged.
		n.admission.Release(grant.LeaseID)
		n.forgetLease(grant.LeaseID)
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	n.lifecycle.Adopt(created.Handle, n.backend, grant.LeaseID, ownerOf(spec))
	n.forgetLease(grant.LeaseID)
	return createdToProto(created), nil
}

// ReleaseOn destroys one sandbox this node holds.
func (n *Node) ReleaseOn(ctx context.Context, handle *v1.SandboxHandle) (*v1.Empty, error) {
	if err := n.lifecycle.Release(ctx, handle.GetSandboxId(), node.ExitReleased); err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	return &v1.Empty{}, nil
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
	n.admission.SetPressure(n.pressure())
	report := n.admission.Snapshot()
	life := n.lifecycle.Snapshot()
	pressure := n.pressure()
	out := &v1.NodeView{
		NodeId:        n.nodeID,
		Backend:       n.backend.Name(),
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
	capabilities := n.backend.Capabilities()
	out.ResumeLevel = resumeLevelValue(capabilities.ResumeLevel)
	for _, feature := range capabilities.Features {
		out.Features = append(out.Features, featureValue(feature))
		if feature == "host_mount" {
			out.HostMounts = true
		}
	}
	return out
}

// View returns this node's state as placement reads it, for an in-process
// deployment where the monitor and the node share a process.
func (n *Node) View() placement.NodeView {
	proto := n.view()
	view := placement.NodeView{
		NodeID:          proto.GetNodeId(),
		Backend:         proto.GetBackend(),
		SeenAt:          time.Now(),
		ClassHeadroom:   map[string]placement.Headroom{},
		LiveSandboxes:   int(proto.GetLiveSandboxes()),
		CPUUsedPct:      proto.GetCpuUsedPct(),
		MemUsedPct:      proto.GetMemUsedPct(),
		GPUFree:         proto.GetGpuFree(),
		Draining:        proto.GetDraining(),
		HostMounts:      proto.GetHostMounts(),
		ImageDigests:    map[string]struct{}{},
		ImageReferences: map[string]struct{}{},
		Labels:          map[string]struct{}{},
		Features:        map[string]struct{}{},
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
	capabilities := n.backend.Capabilities()
	for _, feature := range capabilities.Features {
		view.Features[feature] = struct{}{}
	}
	view.ResumeLevel = capabilities.ResumeLevel
	return view
}

// Sweep runs one reclamation pass on demand, for an operator or a test. The
// node also sweeps on its own cadence.
func (n *Node) Sweep(ctx context.Context, _ *v1.Empty) (*v1.SweepReport, error) {
	report := n.lifecycle.Sweep(ctx)
	n.admission.Expire()
	return &v1.SweepReport{
		Paused:          report.Paused,
		ReapedIdle:      report.ReapedIdle,
		ReapedLifetime:  report.ReapedLifetime,
		ReclaimedOrphan: report.ReclaimedOrphan,
		SkippedBusy:     int32(report.SkippedBusy),
		Failures:        int32(report.Failures),
	}, nil
}

// Exec runs one command in a sandbox this node holds.
//
// It lives on the node rather than the control plane because command traffic is
// the hot path: an episode issues dozens of commands, and routing them through
// the cluster would add a hop and a serialization each without making a
// decision. A backend whose sandboxes run their own agent is reached directly by
// the SDK and never arrives here at all.
func (n *Node) Exec(ctx context.Context, handle backend.Handle, command, workdir string, env map[string]string) (int, string, error) {
	_, held, owned := n.lifecycle.Lookup(handle.SandboxID)
	if !owned {
		return 0, "", fmt.Errorf("sandbox %q is not held by node %q", handle.SandboxID, n.nodeID)
	}
	runner, canRun := held.(interface {
		Exec(context.Context, backend.Handle, string, string, map[string]string) (int, string, error)
	})
	if !canRun {
		return 0, "", fmt.Errorf("backend %q runs commands through its own in-sandbox agent", held.Name())
	}
	// Mark the sandbox busy for the whole command, so the reclamation sweep
	// cannot read a long command as idle and pause a running test suite.
	done := n.lifecycle.Begin(handle.SandboxID)
	defer done()
	return runner.Exec(ctx, handle, command, workdir, env)
}

// StatusOn reports a sandbox's state from this node.
func (n *Node) StatusOn(ctx context.Context, handle backend.Handle) (string, error) {
	stored, held, owned := n.lifecycle.Lookup(handle.SandboxID)
	if !owned {
		return "terminated", nil
	}
	return held.Status(ctx, stored)
}

func (n *Node) forgetLease(leaseID string) {
	n.mu.Lock()
	defer n.mu.Unlock()
	delete(n.leases, leaseID)
}

func ownerOf(spec backend.Spec) string {
	if spec.WorkflowID != "" {
		return spec.WorkflowID
	}
	return "anonymous"
}
