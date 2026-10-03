package server

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"sort"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/monitor"
)

// Ensure FleetNodeClient satisfies the NodeClient interface at compile time.
var _ NodeClient = (*FleetNodeClient)(nil)

// RemoteNodeClient reaches one node agent over the node-plane protocol.
//
// Each call opens a new connection. That is intentional: the node-plane is a
// low-frequency control channel (admit + create_on per episode, release on
// completion), so a persistent connection would buy nothing and would complicate
// node restarts. Exec traffic is the one exception — it is high-frequency but
// also short, so one connection per command is acceptable given that exec only
// goes to the node the sandbox lives on.
type RemoteNodeClient struct {
	address string
	timeout time.Duration
}

// NewRemoteNodeClient returns a client for a node at the given address.
//
// Addresses follow the same "unix://..." / "host:port" convention as the SDK's
// own listen address.
func NewRemoteNodeClient(address string, timeout time.Duration) *RemoteNodeClient {
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	return &RemoteNodeClient{address: address, timeout: timeout}
}

// call dials the node, sends one request, reads one reply, and closes.
func (r *RemoteNodeClient) call(ctx context.Context, method string, payload any) (json.RawMessage, error) {
	conn, err := dialNode(r.address, r.timeout)
	if err != nil {
		return nil, status.Errorf(codes.Unavailable, "node at %s: %v", r.address, err)
	}
	defer conn.Close()

	raw, err := json.Marshal(request{Method: method, Payload: mustMarshal(payload)})
	if err != nil {
		return nil, err
	}
	var header [4]byte
	binary.BigEndian.PutUint32(header[:], uint32(len(raw)))
	if _, err := conn.Write(append(header[:], raw...)); err != nil {
		return nil, status.Errorf(codes.Unavailable, "node at %s: write: %v", r.address, err)
	}

	if _, err := io.ReadFull(conn, header[:]); err != nil {
		return nil, status.Errorf(codes.Unavailable, "node at %s: read header: %v", r.address, err)
	}
	size := binary.BigEndian.Uint32(header[:])
	if size == 0 || size > 64<<20 {
		return nil, status.Errorf(codes.Unavailable, "node at %s: reply frame out of range (%d bytes)", r.address, size)
	}
	body := make([]byte, size)
	if _, err := io.ReadFull(conn, body); err != nil {
		return nil, status.Errorf(codes.Unavailable, "node at %s: read body: %v", r.address, err)
	}

	var rep reply
	if err := json.Unmarshal(body, &rep); err != nil {
		return nil, status.Errorf(codes.Internal, "node at %s: malformed reply: %v", r.address, err)
	}
	if rep.Error != nil {
		return nil, wireErrorToStatus(rep.Error)
	}
	result, err := json.Marshal(rep.Result)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "node at %s: cannot re-marshal result: %v", r.address, err)
	}
	return result, nil
}

func (r *RemoteNodeClient) admit(ctx context.Context, spec backend.Spec) (string, []int32, string, error) {
	raw, err := r.call(ctx, "admit", map[string]any{"spec": specToWire(spec)})
	if err != nil {
		return "", nil, "", err
	}
	var out struct {
		Admitted   bool    `json:"admitted"`
		LeaseID    string  `json:"lease_id"`
		GPUIndices []int32 `json:"gpu_indices"`
		Refusal    string  `json:"refusal"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", nil, "", fmt.Errorf("admit reply: %w", err)
	}
	if !out.Admitted {
		return "", nil, out.Refusal, nil
	}
	return out.LeaseID, out.GPUIndices, "", nil
}

func (r *RemoteNodeClient) createOn(ctx context.Context, leaseID, backendName string, spec backend.Spec, callback string) (backend.Created, error) {
	raw, err := r.call(ctx, "create_on", map[string]any{
		"lease_id": leaseID, "backend": backendName,
		"spec": specToWire(spec), "callback_target": callback,
	})
	if err != nil {
		return backend.Created{}, err
	}
	return createdFromWire(raw)
}

func (r *RemoteNodeClient) releaseOn(ctx context.Context, handle backend.Handle) error {
	_, err := r.call(ctx, "release_on", handleToWire(handle))
	return err
}

func (r *RemoteNodeClient) statusOn(ctx context.Context, handle backend.Handle) (string, error) {
	raw, err := r.call(ctx, "status_on", handleToWire(handle))
	if err != nil {
		return "", err
	}
	var out struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", fmt.Errorf("status_on reply: %w", err)
	}
	return out.Status, nil
}

func (r *RemoteNodeClient) execOn(ctx context.Context, handle backend.Handle, command, cwd string, env map[string]string) (int, string, error) {
	raw, err := r.call(ctx, "exec", map[string]any{
		"backend": handle.Backend, "sandbox_id": handle.SandboxID, "node_id": handle.NodeID,
		"command": command, "cwd": cwd, "env": env,
	})
	if err != nil {
		return 0, "", err
	}
	var out struct {
		ExitCode int    `json:"exit_code"`
		Stdout   string `json:"stdout"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return 0, "", fmt.Errorf("exec reply: %w", err)
	}
	return out.ExitCode, out.Stdout, nil
}

func (r *RemoteNodeClient) readBytesOn(ctx context.Context, handle backend.Handle, path string) (string, error) {
	raw, err := r.call(ctx, "read_bytes", map[string]any{
		"backend": handle.Backend, "sandbox_id": handle.SandboxID, "node_id": handle.NodeID,
		"path": path,
	})
	if err != nil {
		return "", err
	}
	var out struct {
		Data string `json:"data"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", fmt.Errorf("read_bytes reply: %w", err)
	}
	return out.Data, nil
}

func (r *RemoteNodeClient) writeBytesOn(ctx context.Context, handle backend.Handle, path, data string) error {
	_, err := r.call(ctx, "write_bytes", map[string]any{
		"backend": handle.Backend, "sandbox_id": handle.SandboxID, "node_id": handle.NodeID,
		"path": path, "data": data,
	})
	return err
}

func (r *RemoteNodeClient) report(ctx context.Context) (nodeViewWire, error) {
	raw, err := r.call(ctx, "report", map[string]any{})
	if err != nil {
		return nodeViewWire{}, err
	}
	var out nodeViewWire
	if err := json.Unmarshal(raw, &out); err != nil {
		return nodeViewWire{}, fmt.Errorf("report reply: %w", err)
	}
	return out, nil
}

// FleetNodeClient routes node-plane calls to many remote nodes by nodeID.
//
// It also runs a background poller that calls each node's `report` endpoint on
// a configurable interval and feeds the received views to a monitor, so the
// control plane's placement service sees a fleet view that ages at the same
// rate as the single-process republish loop.
type FleetNodeClient struct {
	mu    sync.RWMutex
	nodes map[string]*RemoteNodeClient
	order []string // sorted nodeIDs for stable Nodes() output
}

// NewFleetNodeClient returns a client for all the nodes in the fleet.
func NewFleetNodeClient(nodes map[string]*RemoteNodeClient) *FleetNodeClient {
	order := make([]string, 0, len(nodes))
	for id := range nodes {
		order = append(order, id)
	}
	sort.Strings(order)
	return &FleetNodeClient{nodes: nodes, order: order}
}

func (f *FleetNodeClient) node(nodeID string) (*RemoteNodeClient, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	n, ok := f.nodes[nodeID]
	if !ok {
		return nil, status.Errorf(codes.NotFound,
			"node %q is not configured in this control plane's fleet", nodeID)
	}
	return n, nil
}

// Admit asks the named node to accept one sandbox request.
func (f *FleetNodeClient) Admit(ctx context.Context, nodeID string, spec backend.Spec) (string, []int32, string, error) {
	n, err := f.node(nodeID)
	if err != nil {
		return "", nil, "", err
	}
	return n.admit(ctx, spec)
}

// CreateOn provisions a sandbox on the named node against an admission it already granted.
func (f *FleetNodeClient) CreateOn(ctx context.Context, nodeID, leaseID, backendName string, spec backend.Spec, callback string) (backend.Created, error) {
	n, err := f.node(nodeID)
	if err != nil {
		return backend.Created{}, err
	}
	return n.createOn(ctx, leaseID, backendName, spec, callback)
}

// ReleaseOn destroys one sandbox on the node that holds it.
func (f *FleetNodeClient) ReleaseOn(ctx context.Context, handle backend.Handle) error {
	n, err := f.node(handle.NodeID)
	if err != nil {
		return err
	}
	return n.releaseOn(ctx, handle)
}

// StatusOn reports one sandbox's state from the node that holds it.
func (f *FleetNodeClient) StatusOn(ctx context.Context, handle backend.Handle) (string, error) {
	n, err := f.node(handle.NodeID)
	if err != nil {
		return "", err
	}
	return n.statusOn(ctx, handle)
}

// ExecOn runs one command in a sandbox on the node that holds it.
func (f *FleetNodeClient) ExecOn(ctx context.Context, handle backend.Handle, command, cwd string, env map[string]string) (int, string, error) {
	n, err := f.node(handle.NodeID)
	if err != nil {
		return 0, "", err
	}
	return n.execOn(ctx, handle, command, cwd, env)
}

// ReadBytesOn reads a file from a sandbox on the node that holds it.
func (f *FleetNodeClient) ReadBytesOn(ctx context.Context, handle backend.Handle, path string) (string, error) {
	n, err := f.node(handle.NodeID)
	if err != nil {
		return "", err
	}
	return n.readBytesOn(ctx, handle, path)
}

// WriteBytesOn writes a file into a sandbox on the node that holds it.
func (f *FleetNodeClient) WriteBytesOn(ctx context.Context, handle backend.Handle, path, data string) error {
	n, err := f.node(handle.NodeID)
	if err != nil {
		return err
	}
	return n.writeBytesOn(ctx, handle, path, data)
}

// PollFleet runs a polling loop that pulls each node's view into the fleet
// monitor. It returns when ctx is cancelled.
//
// interval governs how often each node is polled. A node that does not answer
// within its own timeout is skipped for that round and its view ages out
// naturally through the monitor's TTL.
func (f *FleetNodeClient) PollFleet(ctx context.Context, fleet *monitor.Monitor, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			f.mu.RLock()
			nodes := make(map[string]*RemoteNodeClient, len(f.nodes))
			for id, n := range f.nodes {
				nodes[id] = n
			}
			f.mu.RUnlock()

			for _, n := range nodes {
				go func(node *RemoteNodeClient) {
					pollCtx, cancel := context.WithTimeout(ctx, node.timeout)
					defer cancel()
					wire, err := node.report(pollCtx)
					if err != nil {
						// The node is unreachable. Its view will age out through the
						// monitor's TTL; we do not evict it here because a transient
						// network blip should not cause a mass placement failure.
						return
					}
					fleet.Report(nodeViewFromWire(wire))
				}(n)
			}
		}
	}
}

// -- helpers -------------------------------------------------------------------

func dialNode(address string, timeout time.Duration) (net.Conn, error) {
	if path, found := cutPrefix(address, "unix://"); found {
		return net.DialTimeout("unix", path, timeout)
	}
	return net.DialTimeout("tcp", address, timeout)
}

func cutPrefix(s, prefix string) (string, bool) {
	if len(s) >= len(prefix) && s[:len(prefix)] == prefix {
		return s[len(prefix):], true
	}
	return "", false
}

func wireErrorToStatus(e *wireError) error {
	var code codes.Code
	switch e.Code {
	case "capability":
		code = codes.FailedPrecondition
	case "capacity_timeout":
		code = codes.ResourceExhausted
	case "session_lost":
		code = codes.NotFound
	case "transport":
		code = codes.Unavailable
	case "command_timeout":
		code = codes.DeadlineExceeded
	default:
		code = codes.Internal
	}
	return status.Error(code, e.Message)
}

func mustMarshal(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("remotenode: unexpected marshal failure: %v", err))
	}
	return b
}

func handleToWire(h backend.Handle) map[string]any {
	return map[string]any{"backend": h.Backend, "sandbox_id": h.SandboxID, "node_id": h.NodeID}
}

// specToWire converts a backend.Spec to the wire shape jsonSpec uses.
func specToWire(spec backend.Spec) map[string]any {
	s := map[string]any{
		"source":                map[string]any{"kind": spec.Source.Kind, "reference": spec.Source.Reference},
		"resource_class":        spec.ResourceClass,
		"workflow_id":           spec.WorkflowID,
		"idempotency_key":       spec.IdempotencyKey,
		"env":                   spec.Env,
		"metadata":              spec.Metadata,
		"workdir":               spec.Workdir,
		"exec_mode":             spec.ExecMode,
		"required_resume_level": spec.RequiredResume,
		"backend":               spec.Backend,
		"required_features":     spec.RequiredFeatures,
		"required_node_label":   spec.RequiredNodeLabel,
		"forbidden_node_labels": spec.ForbiddenNodeLabels,
	}
	resources := map[string]any{}
	if spec.Resources.CPUCount > 0 {
		resources["cpu_count"] = spec.Resources.CPUCount
	}
	if spec.Resources.MemoryMB > 0 {
		resources["memory_mb"] = spec.Resources.MemoryMB
	}
	if spec.Resources.DiskMB > 0 {
		resources["disk_mb"] = spec.Resources.DiskMB
	}
	if spec.Resources.GPUCount > 0 {
		resources["gpu_count"] = spec.Resources.GPUCount
	}
	s["resources"] = resources
	if len(spec.BackendOptions) > 0 {
		s["backend_options"] = spec.BackendOptions
	}
	return s
}

// createdFromWire parses a create reply's JSON into a backend.Created.
func createdFromWire(raw json.RawMessage) (backend.Created, error) {
	var out struct {
		Handle struct {
			Backend   string `json:"backend"`
			SandboxID string `json:"sandbox_id"`
			NodeID    string `json:"node_id"`
		} `json:"handle"`
		Capabilities struct {
			Features    []string `json:"features"`
			ResumeLevel string   `json:"resume_level"`
			PauseModes  []string `json:"pause_modes"`
		} `json:"capabilities"`
		Agent struct {
			Address           string            `json:"address"`
			Headers           map[string]string `json:"headers"`
			CallbackHostAlias string            `json:"callback_host_alias"`
			CallbackPort      int32             `json:"callback_port"`
		} `json:"agent"`
		WarmStart bool `json:"warm_start"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return backend.Created{}, fmt.Errorf("create_on reply: %w", err)
	}
	return backend.Created{
		Handle: backend.Handle{
			Backend:   out.Handle.Backend,
			SandboxID: out.Handle.SandboxID,
			NodeID:    out.Handle.NodeID,
		},
		Capabilities: backend.Capabilities{
			Features:    out.Capabilities.Features,
			ResumeLevel: out.Capabilities.ResumeLevel,
			PauseModes:  out.Capabilities.PauseModes,
		},
		Agent: backend.AgentEndpoint{
			Address:           out.Agent.Address,
			Headers:           out.Agent.Headers,
			CallbackHostAlias: out.Agent.CallbackHostAlias,
			CallbackPort:      out.Agent.CallbackPort,
		},
		WarmStart: out.WarmStart,
	}, nil
}

