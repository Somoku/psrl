package server

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/placement"
)

// NodeListener serves the node-plane protocol: a control process in a fleet
// reaches nodes over this rather than in-process.
//
// The wire format matches JSONListener: a four-byte big-endian length followed
// by a JSON frame. The method namespace is disjoint from the SDK methods on
// purpose — a node listening on one port serves only its node-plane callers,
// not the SDK.
type NodeListener struct {
	node *Node
	log  *slog.Logger

	mu       sync.Mutex
	listener net.Listener
	closed   bool
	conns    sync.WaitGroup
}

// NewNodeListener returns a node-plane server. It exposes the node's admission,
// lifecycle, and exec surfaces to a remote control plane.
func NewNodeListener(node *Node, log *slog.Logger) *NodeListener {
	if log == nil {
		log = slog.Default()
	}
	return &NodeListener{node: node, log: log}
}

// Serve accepts connections until the context is cancelled.
func (n *NodeListener) Serve(ctx context.Context, listener net.Listener) error {
	n.mu.Lock()
	n.listener = listener
	n.mu.Unlock()

	go func() {
		<-ctx.Done()
		_ = n.Close()
	}()

	for {
		conn, err := listener.Accept()
		if err != nil {
			n.mu.Lock()
			closed := n.closed
			n.mu.Unlock()
			if closed || errors.Is(err, net.ErrClosed) {
				n.conns.Wait()
				return nil
			}
			return err
		}
		n.conns.Add(1)
		go func() {
			defer n.conns.Done()
			n.handle(ctx, conn)
		}()
	}
}

// Close stops accepting and waits for in-flight connections to drain.
func (n *NodeListener) Close() error {
	n.mu.Lock()
	if n.closed {
		n.mu.Unlock()
		return nil
	}
	n.closed = true
	listener := n.listener
	n.mu.Unlock()
	if listener != nil {
		return listener.Close()
	}
	return nil
}

func (n *NodeListener) handle(ctx context.Context, conn net.Conn) {
	defer conn.Close()
	for {
		var header [4]byte
		if _, err := io.ReadFull(conn, header[:]); err != nil {
			return
		}
		size := binary.BigEndian.Uint32(header[:])
		if size == 0 || size > 64<<20 {
			n.log.Warn("node-plane frame out of range", "bytes", size)
			return
		}
		body := make([]byte, size)
		if _, err := io.ReadFull(conn, body); err != nil {
			return
		}
		var req request
		if err := json.Unmarshal(body, &req); err != nil {
			n.writeReply(conn, reply{Error: &wireError{Code: "invalid", Message: err.Error()}})
			continue
		}
		result, err := n.dispatch(ctx, req)
		if err != nil {
			n.writeReply(conn, reply{Error: classify(err)})
			continue
		}
		n.writeReply(conn, reply{Result: result})
	}
}

func (n *NodeListener) writeReply(conn net.Conn, out reply) {
	encoded, err := json.Marshal(out)
	if err != nil {
		n.log.Error("node-plane reply could not be encoded", "error", err)
		return
	}
	var header [4]byte
	binary.BigEndian.PutUint32(header[:], uint32(len(encoded)))
	if _, err := conn.Write(append(header[:], encoded...)); err != nil {
		n.log.Debug("node-plane reply could not be written", "error", err)
	}
}

func (n *NodeListener) dispatch(ctx context.Context, req request) (any, error) {
	switch req.Method {
	case "admit":
		return n.admit(ctx, req.Payload)
	case "create_on":
		return n.createOn(ctx, req.Payload)
	case "release_on":
		return n.releaseOn(ctx, req.Payload)
	case "status_on":
		return n.statusOn(ctx, req.Payload)
	case "exec":
		return n.exec(ctx, req.Payload)
	case "read_bytes":
		return n.readBytes(ctx, req.Payload)
	case "write_bytes":
		return n.writeBytes(ctx, req.Payload)
	case "report":
		return n.report(ctx)
	default:
		return nil, status.Errorf(codes.Unimplemented, "the node-plane has no method %q", req.Method)
	}
}

func (n *NodeListener) admit(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload struct {
		Spec jsonSpec `json:"spec"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	resp, err := n.node.Admit(ctx, &v1.AdmitRequest{Spec: payload.Spec.toProto()})
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"admitted":    resp.GetAdmitted(),
		"lease_id":    resp.GetLeaseId(),
		"gpu_indices": resp.GetGpuIndices(),
		"refusal":     resp.GetRefusal(),
	}, nil
}

func (n *NodeListener) createOn(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload struct {
		LeaseID        string   `json:"lease_id"`
		Backend        string   `json:"backend"`
		Spec           jsonSpec `json:"spec"`
		CallbackTarget string   `json:"callback_target"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	resp, err := n.node.CreateOn(ctx, &v1.CreateOnRequest{
		LeaseId:        payload.LeaseID,
		Backend:        payload.Backend,
		Spec:           payload.Spec.toProto(),
		CallbackTarget: payload.CallbackTarget,
	})
	if err != nil {
		return nil, err
	}
	return createdJSON(resp), nil
}

func (n *NodeListener) releaseOn(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload handlePayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	if _, err := n.node.ReleaseOn(ctx, payload.proto()); err != nil {
		return nil, err
	}
	return map[string]any{}, nil
}

func (n *NodeListener) statusOn(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload handlePayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	state, err := n.node.StatusOn(ctx, payload.handle())
	if err != nil {
		return nil, err
	}
	return map[string]any{"status": state}, nil
}

func (n *NodeListener) exec(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload struct {
		handlePayload
		Command string            `json:"command"`
		Cwd     string            `json:"cwd"`
		Env     map[string]string `json:"env"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	code, output, err := n.node.Exec(ctx, payload.handle(), payload.Command, payload.Cwd, payload.Env)
	if err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	return map[string]any{"exit_code": code, "stdout": output, "stderr": "", "truncated": false}, nil
}

// readBytes fetches one file out of a sandbox.
//
// The archive path is preferred where the backend offers it. The shell fallback
// puts the whole file on a command line, and Linux caps a single argument at
// 128 KiB (MAX_ARG_STRLEN), so a large file fails there with "argument list too
// long" however much memory the node has.
func (n *NodeListener) readBytes(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload struct {
		handlePayload
		Path string `json:"path"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	handle := payload.handle()
	if hosted, hosts := n.node.hosted(handle.Backend); hosts {
		if files, ok := hosted.(backend.FileHandler); ok {
			data, err := files.ReadFile(ctx, handle, payload.Path)
			if err != nil {
				return nil, status.Errorf(codes.NotFound, "reading %q: %v", payload.Path, err)
			}
			return map[string]any{"data": base64.StdEncoding.EncodeToString(data)}, nil
		}
	}
	code, output, err := n.node.Exec(ctx, handle, fmt.Sprintf("base64 %q", payload.Path), "", nil)
	if err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	if code != 0 {
		return nil, status.Errorf(codes.NotFound, "reading %q exited %d", payload.Path, code)
	}
	return map[string]any{"data": compactBase64(output)}, nil
}

// writeBytes puts one file into a sandbox.
//
// Same reason as readBytes for preferring the archive path: the shell fallback
// cannot carry a file past the kernel's single-argument limit, and a model patch
// routinely exceeds it.
func (n *NodeListener) writeBytes(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload struct {
		handlePayload
		Path string `json:"path"`
		Data string `json:"data"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	handle := payload.handle()
	if hosted, hosts := n.node.hosted(handle.Backend); hosts {
		if files, ok := hosted.(backend.FileHandler); ok {
			decoded, err := base64.StdEncoding.DecodeString(payload.Data)
			if err != nil {
				return nil, status.Errorf(codes.InvalidArgument, "writing %q: %v", payload.Path, err)
			}
			if err := files.WriteFile(ctx, handle, payload.Path, decoded); err != nil {
				return nil, status.Errorf(codes.FailedPrecondition, "writing %q: %v", payload.Path, err)
			}
			return map[string]any{}, nil
		}
	}
	command := fmt.Sprintf("mkdir -p \"$(dirname %q)\" && printf %%s %q | base64 -d > %q",
		payload.Path, payload.Data, payload.Path)
	code, output, err := n.node.Exec(ctx, handle, command, "", nil)
	if err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	if code != 0 {
		return nil, status.Errorf(codes.FailedPrecondition, "writing %q exited %d: %s", payload.Path, code, output)
	}
	return map[string]any{}, nil
}

// report returns the node's view in a form the control can feed to the monitor.
//
// SeenAt is omitted here and filled in by the caller on receipt — the node's
// clock and the control's may differ, and what the control wants is the time of
// receipt, not the time of assembly.
func (n *NodeListener) report(_ context.Context) (any, error) {
	view := n.node.View()
	return nodeViewToWire(view), nil
}

// nodeViewWire is the over-the-wire shape of a NodeView. It uses slices for the
// set fields because map[string]struct{} does not serialize as JSON.
type nodeViewWire struct {
	NodeID          string              `json:"node_id"`
	Backends        []backendCapWire    `json:"backends"`
	Envelope        headroomWire        `json:"envelope"`
	ClassHeadroom   map[string]headroomWire `json:"class_headroom"`
	LiveSandboxes   int                 `json:"live_sandboxes"`
	CPUUsedPct      float64             `json:"cpu_used_pct"`
	MemUsedPct      float64             `json:"mem_used_pct"`
	GPUFree         int32               `json:"gpu_free"`
	Draining        bool                `json:"draining"`
	ImageDigests    []string            `json:"image_digests"`
	ImageReferences []string            `json:"image_references"`
	Labels          []string            `json:"labels"`
}

type backendCapWire struct {
	Name        string            `json:"name"`
	Features    []string          `json:"features"`
	ResumeLevel string            `json:"resume_level"`
	HostMounts  bool              `json:"host_mounts"`
}

type headroomWire struct {
	MemoryMB  int64   `json:"memory_mb"`
	CPUMillis int64   `json:"cpu_millis"`
	GPUCount  int32   `json:"gpu_count"`
	DiskMB    int64   `json:"disk_mb"`
}

func nodeViewToWire(view placement.NodeView) nodeViewWire {
	w := nodeViewWire{
		NodeID:        view.NodeID,
		LiveSandboxes: view.LiveSandboxes,
		CPUUsedPct:    view.CPUUsedPct,
		MemUsedPct:    view.MemUsedPct,
		GPUFree:       view.GPUFree,
		Draining:      view.Draining,
		Envelope: headroomWire{
			MemoryMB: view.Envelope.MemoryMB, CPUMillis: view.Envelope.CPUMillis,
			GPUCount: view.Envelope.GPUCount, DiskMB: view.Envelope.DiskMB,
		},
		ClassHeadroom: make(map[string]headroomWire, len(view.ClassHeadroom)),
	}
	for class, h := range view.ClassHeadroom {
		w.ClassHeadroom[class] = headroomWire{
			MemoryMB: h.MemoryMB, CPUMillis: h.CPUMillis, GPUCount: h.GPUCount, DiskMB: h.DiskMB,
		}
	}
	for _, b := range view.Backends {
		bc := backendCapWire{Name: b.Name, ResumeLevel: b.ResumeLevel, HostMounts: b.HostMounts}
		for f := range b.Features {
			bc.Features = append(bc.Features, f)
		}
		w.Backends = append(w.Backends, bc)
	}
	for d := range view.ImageDigests {
		w.ImageDigests = append(w.ImageDigests, d)
	}
	for r := range view.ImageReferences {
		w.ImageReferences = append(w.ImageReferences, r)
	}
	for l := range view.Labels {
		w.Labels = append(w.Labels, l)
	}
	return w
}

func nodeViewFromWire(w nodeViewWire) placement.NodeView {
	view := placement.NodeView{
		NodeID:          w.NodeID,
		SeenAt:          time.Now(),
		LiveSandboxes:   w.LiveSandboxes,
		CPUUsedPct:      w.CPUUsedPct,
		MemUsedPct:      w.MemUsedPct,
		GPUFree:         w.GPUFree,
		Draining:        w.Draining,
		ClassHeadroom:   make(map[string]placement.Headroom, len(w.ClassHeadroom)),
		ImageDigests:    make(map[string]struct{}, len(w.ImageDigests)),
		ImageReferences: make(map[string]struct{}, len(w.ImageReferences)),
		Labels:          make(map[string]struct{}, len(w.Labels)),
		Envelope: placement.Headroom{
			MemoryMB: w.Envelope.MemoryMB, CPUMillis: w.Envelope.CPUMillis,
			GPUCount: w.Envelope.GPUCount, DiskMB: w.Envelope.DiskMB,
		},
	}
	for class, h := range w.ClassHeadroom {
		view.ClassHeadroom[class] = placement.Headroom{
			MemoryMB: h.MemoryMB, CPUMillis: h.CPUMillis, GPUCount: h.GPUCount, DiskMB: h.DiskMB,
		}
	}
	for _, b := range w.Backends {
		bc := placement.BackendCapability{
			Name:        b.Name,
			ResumeLevel: b.ResumeLevel,
			HostMounts:  b.HostMounts,
			Features:    make(map[string]struct{}, len(b.Features)),
		}
		for _, f := range b.Features {
			bc.Features[f] = struct{}{}
		}
		view.Backends = append(view.Backends, bc)
	}
	for _, d := range w.ImageDigests {
		view.ImageDigests[d] = struct{}{}
	}
	for _, r := range w.ImageReferences {
		view.ImageReferences[r] = struct{}{}
	}
	for _, l := range w.Labels {
		view.Labels[l] = struct{}{}
	}
	return view
}
