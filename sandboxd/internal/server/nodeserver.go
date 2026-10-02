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

	mu     sync.Mutex
	leases map[string]node.Grant
}

// NodeConfig is what a node agent needs to run.
type NodeConfig struct {
	NodeID    string
	Admission *node.Admission
	Lifecycle *node.Lifecycle
	// Backends are every runtime this machine hosts, in declaration order.
	Backends []backend.Backend
	Pressure func() node.Pressure
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
	return &Node{
		nodeID:    cfg.NodeID,
		admission: cfg.Admission,
		lifecycle: cfg.Lifecycle,
		backends:  backends,
		order:     order,
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
// that its own accounting never charged. The runtime is named by the caller
// rather than chosen here, because routing already matched the spec's
// requirements against one and a second choice could disagree with it.
func (n *Node) CreateOn(ctx context.Context, req *v1.CreateOnRequest) (*v1.CreateResponse, error) {
	n.mu.Lock()
	grant, granted := n.leases[req.GetLeaseId()]
	n.mu.Unlock()
	if !granted {
		return nil, status.Errorf(codes.FailedPrecondition,
			"lease %q was not granted by this node, so a sandbox created against it would not be charged",
			req.GetLeaseId())
	}
	hosted, hosts := n.hosted(req.GetBackend())
	if !hosts {
		// The admission is released rather than held: no sandbox exists, and a
		// misrouted request must not cost the node a slot.
		n.admission.Release(grant.LeaseID)
		n.forgetLease(grant.LeaseID)
		return nil, status.Errorf(codes.FailedPrecondition,
			"node %q does not host backend %q (hosts: %v)", n.nodeID, req.GetBackend(), n.order)
	}
	spec := specFromProto(req.GetSpec())
	spec.AssignedGPUs = grant.GPUIndices
	created, err := hosted.Create(ctx, n.nodeID, spec, req.GetCallbackTarget())
	if err != nil {
		// The sandbox does not exist, so its admission must not stay charged.
		n.admission.Release(grant.LeaseID)
		n.forgetLease(grant.LeaseID)
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	n.lifecycle.Adopt(created.Handle, hosted, grant.LeaseID, ownerOf(spec))
	n.forgetLease(grant.LeaseID)
	return createdToProto(created), nil
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
		NodeID:          proto.GetNodeId(),
		SeenAt:          time.Now(),
		ClassHeadroom:   map[string]placement.Headroom{},
		LiveSandboxes:   int(proto.GetLiveSandboxes()),
		CPUUsedPct:      proto.GetCpuUsedPct(),
		MemUsedPct:      proto.GetMemUsedPct(),
		GPUFree:         proto.GetGpuFree(),
		Draining:        proto.GetDraining(),
		ImageDigests:    map[string]struct{}{},
		ImageReferences: map[string]struct{}{},
		Labels:          map[string]struct{}{},
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
