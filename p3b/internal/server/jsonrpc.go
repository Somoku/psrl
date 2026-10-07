package server

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"sync"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/placement"
)

// JSONListener serves the SDK's wire format: a four-byte length then a JSON
// frame.
//
// JSON rather than gRPC for the Python side, deliberately. The SDK's job is to
// serialize and forward, the control path is about two calls per episode, and a
// generated stub would make the SDK's install depend on a protobuf runtime for
// no measurable gain. The message shapes are still the ones the proto defines,
// so the Go-to-Go path and a future generated SDK speak the same contract.
type JSONListener struct {
	control *Control
	node    *Node
	log     *slog.Logger

	// dataPlane resolves a node id to the address the SDK can reach that node's
	// own plane on, so a sandbox with no in-sandbox agent still gets a data-plane
	// address and its commands stay off the control plane. Nil in a combined
	// deployment, where the SDK already shares a socket with the node.
	dataPlane func(nodeID string) string

	mu       sync.Mutex
	listener net.Listener
	closed   bool
	conns    sync.WaitGroup
}

// NewJSONListener returns the SDK-facing server. The node may be nil on a
// control-only process; command calls then report that they have nowhere to go
// rather than failing obscurely.
func NewJSONListener(control *Control, node *Node, log *slog.Logger) *JSONListener {
	if log == nil {
		log = slog.Default()
	}
	return &JSONListener{control: control, node: node, log: log}
}

// WithDataPlane supplies the node-id-to-address lookup a fleet deployment needs,
// so a create reply can tell the SDK which node to send commands to.
func (j *JSONListener) WithDataPlane(resolve func(nodeID string) string) *JSONListener {
	j.dataPlane = resolve
	return j
}

// Serve accepts connections until the listener is closed.
func (j *JSONListener) Serve(ctx context.Context, listener net.Listener) error {
	j.mu.Lock()
	j.listener = listener
	j.mu.Unlock()

	go func() {
		<-ctx.Done()
		_ = j.Close()
	}()

	for {
		conn, err := listener.Accept()
		if err != nil {
			j.mu.Lock()
			closed := j.closed
			j.mu.Unlock()
			if closed || errors.Is(err, net.ErrClosed) {
				j.conns.Wait()
				return nil
			}
			return err
		}
		j.conns.Add(1)
		go func() {
			defer j.conns.Done()
			j.handle(ctx, conn)
		}()
	}
}

// Close stops accepting and waits for live connections to drain.
func (j *JSONListener) Close() error {
	j.mu.Lock()
	if j.closed {
		j.mu.Unlock()
		return nil
	}
	j.closed = true
	listener := j.listener
	j.mu.Unlock()
	if listener != nil {
		return listener.Close()
	}
	return nil
}

type request struct {
	Method  string          `json:"method"`
	Payload json.RawMessage `json:"payload"`
}

type reply struct {
	Result any        `json:"result,omitempty"`
	Error  *wireError `json:"error,omitempty"`
}

type wireError struct {
	// Code is the service's classification, not a guess from the message: the
	// caller's response differs per type, and an infrastructure fault read as a
	// task result would corrupt a reward.
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (j *JSONListener) handle(ctx context.Context, conn net.Conn) {
	defer conn.Close()
	for {
		var header [4]byte
		if _, err := io.ReadFull(conn, header[:]); err != nil {
			// A client that closed its connection is the normal end of a session.
			return
		}
		size := binary.BigEndian.Uint32(header[:])
		if size == 0 || size > 64<<20 {
			j.log.Warn("sandbox SDK frame out of range", "bytes", size)
			return
		}
		body := make([]byte, size)
		if _, err := io.ReadFull(conn, body); err != nil {
			return
		}
		var req request
		if err := json.Unmarshal(body, &req); err != nil {
			j.write(conn, reply{Error: &wireError{Code: "invalid", Message: err.Error()}})
			continue
		}
		result, err := j.dispatch(ctx, req)
		if err != nil {
			j.write(conn, reply{Error: classify(err)})
			continue
		}
		j.write(conn, reply{Result: result})
	}
}

func (j *JSONListener) write(conn net.Conn, out reply) {
	encoded, err := json.Marshal(out)
	if err != nil {
		j.log.Error("sandbox SDK reply could not be encoded", "error", err)
		return
	}
	var header [4]byte
	binary.BigEndian.PutUint32(header[:], uint32(len(encoded)))
	if _, err := conn.Write(append(header[:], encoded...)); err != nil {
		j.log.Debug("sandbox SDK reply could not be written", "error", err)
	}
}

// classify maps a failure onto the attribution the caller acts on.
func classify(err error) *wireError {
	code := "internal"
	switch status.Code(err) {
	case codes.FailedPrecondition:
		code = "capability"
	case codes.ResourceExhausted:
		code = "capacity_timeout"
	case codes.NotFound:
		code = "session_lost"
	case codes.Unavailable:
		code = "transport"
	case codes.DeadlineExceeded:
		code = "command_timeout"
	}
	return &wireError{Code: code, Message: err.Error()}
}

func (j *JSONListener) dispatch(ctx context.Context, req request) (any, error) {
	switch req.Method {
	case "create":
		return j.create(ctx, req.Payload)
	case "create_group":
		return j.createGroup(ctx, req.Payload)
	case "release":
		return j.release(ctx, req.Payload)
	case "status":
		return j.status(ctx, req.Payload)
	case "pause":
		return j.pause(ctx, req.Payload)
	case "resume":
		return j.resume(ctx, req.Payload)
	case "snapshot":
		return j.snapshot(ctx, req.Payload)
	case "delete_snapshot":
		return j.deleteSnapshot(ctx, req.Payload)
	case "exec":
		return j.exec(ctx, req.Payload)
	case "read_bytes":
		return j.readBytes(ctx, req.Payload)
	case "write_bytes":
		return j.writeBytes(ctx, req.Payload)
	case "fleet":
		return j.fleet(ctx)
	case "quota":
		return j.quota(ctx)
	default:
		return nil, status.Errorf(codes.Unimplemented, "the sandbox service has no method %q", req.Method)
	}
}

type specEnvelope struct {
	Spec           jsonSpec `json:"spec"`
	CallbackTarget string   `json:"callback_target"`
}

type jsonSpec struct {
	Source struct {
		Kind      string `json:"kind"`
		Reference string `json:"reference"`
	} `json:"source"`
	Resources struct {
		CPUCount *float64 `json:"cpu_count"`
		MemoryMB *int64   `json:"memory_mb"`
		DiskMB   *int64   `json:"disk_mb"`
		GPUCount *int32   `json:"gpu_count"`
	} `json:"resources"`
	ResourceClass       string                       `json:"resource_class"`
	WorkflowID          string                       `json:"workflow_id"`
	IdempotencyKey      string                       `json:"idempotency_key"`
	Env                 map[string]string            `json:"env"`
	Metadata            map[string]string            `json:"metadata"`
	Workdir             string                       `json:"workdir"`
	ExecMode            string                       `json:"exec_mode"`
	RequiredFeatures    []string                     `json:"required_features"`
	RequiredResumeLevel string                       `json:"required_resume_level"`
	Backend             string                       `json:"backend"`
	RequiredNodeLabel   string                       `json:"required_node_label"`
	ForbiddenNodeLabels []string                     `json:"forbidden_node_labels"`
	BackendOptions      map[string]map[string]string `json:"backend_options"`
}

func (s jsonSpec) toProto() *v1.SandboxSpec {
	out := &v1.SandboxSpec{
		Source:              &v1.Source{Kind: sourceKindValue(s.Source.Kind), Reference: s.Source.Reference},
		ResourceClass:       s.ResourceClass,
		WorkflowId:          s.WorkflowID,
		IdempotencyKey:      s.IdempotencyKey,
		Env:                 s.Env,
		Metadata:            s.Metadata,
		Workdir:             s.Workdir,
		ExecMode:            execModeValue(s.ExecMode),
		RequiredResumeLevel: resumeLevelValue(s.RequiredResumeLevel),
		Backend:             s.Backend,
		Resources: &v1.Resources{
			CpuCount: s.Resources.CPUCount,
			MemoryMb: s.Resources.MemoryMB,
			DiskMb:   s.Resources.DiskMB,
			GpuCount: s.Resources.GPUCount,
		},
	}
	for _, feature := range s.RequiredFeatures {
		out.RequiredFeatures = append(out.RequiredFeatures, featureValue(feature))
	}
	if len(s.BackendOptions) > 0 {
		out.BackendOptions = map[string]*v1.BackendOptions{}
		for name, values := range s.BackendOptions {
			out.BackendOptions[name] = &v1.BackendOptions{Values: values}
		}
	}
	return out
}

func (j *JSONListener) create(ctx context.Context, raw json.RawMessage) (any, error) {
	var env specEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, err
	}
	created, err := j.control.Create(ctx, &v1.CreateRequest{Spec: env.Spec.toProto(), CallbackTarget: env.CallbackTarget})
	if err != nil {
		return nil, err
	}
	return j.createdJSON(created), nil
}

func (j *JSONListener) createGroup(ctx context.Context, raw json.RawMessage) (any, error) {
	var env struct {
		Specs          []jsonSpec `json:"specs"`
		CallbackTarget string     `json:"callback_target"`
	}
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, err
	}
	req := &v1.CreateGroupRequest{CallbackTarget: env.CallbackTarget}
	for _, spec := range env.Specs {
		req.Specs = append(req.Specs, spec.toProto())
	}
	group, err := j.control.CreateGroup(ctx, req)
	if err != nil {
		return nil, err
	}
	members := make([]any, 0, len(group.GetMembers()))
	for _, member := range group.GetMembers() {
		members = append(members, j.createdJSON(member))
	}
	return map[string]any{"members": members}, nil
}

// createdJSON renders a create reply, resolving the data-plane address this
// deployment's topology implies.
func (j *JSONListener) createdJSON(created *v1.CreateResponse) map[string]any {
	out := createdJSON(created)
	// A sandbox with no agent of its own is driven by the node that owns it, so
	// the node's address is what the SDK needs to keep its commands off the
	// control plane. Resolved here rather than carried up from the backend,
	// because the address belongs to this deployment's topology and the backend
	// only knows that it has no agent to report.
	if j.dataPlane != nil && created.GetAgent().GetAddress() == "" {
		if agent, ok := out["agent"].(map[string]any); ok {
			agent["data_plane"] = j.dataPlane(created.GetHandle().GetNodeId())
		}
	}
	return out
}

// createdJSON renders a create reply with no topology applied. The node plane
// uses this for its own replies, where the caller is the control plane rather
// than the SDK and no data-plane address is wanted.
func createdJSON(created *v1.CreateResponse) map[string]any {
	features := make([]string, 0, len(created.GetCapabilities().GetFeatures()))
	for _, feature := range created.GetCapabilities().GetFeatures() {
		features = append(features, featureName(feature))
	}
	modes := make([]string, 0, len(created.GetCapabilities().GetPauseModes()))
	for _, mode := range created.GetCapabilities().GetPauseModes() {
		modes = append(modes, pauseModeName(mode))
	}
	return map[string]any{
		"handle": map[string]any{
			"backend":    created.GetHandle().GetBackend(),
			"sandbox_id": created.GetHandle().GetSandboxId(),
			"node_id":    created.GetHandle().GetNodeId(),
		},
		"capabilities": map[string]any{
			"features":     features,
			"resume_level": resumeLevelName(created.GetCapabilities().GetResumeLevel()),
			"pause_modes":  modes,
		},
		"agent": map[string]any{
			"address":             created.GetAgent().GetAddress(),
			"data_plane":          "",
			"headers":             created.GetAgent().GetHeaders(),
			"callback_host_alias": created.GetAgent().GetCallbackHostAlias(),
			"callback_port":       created.GetAgent().GetCallbackPort(),
		},
		"warm_start": created.GetWarmStart(),
	}
}

type handlePayload struct {
	Backend   string `json:"backend"`
	SandboxID string `json:"sandbox_id"`
	NodeID    string `json:"node_id"`
}

func (h handlePayload) proto() *v1.SandboxHandle {
	return &v1.SandboxHandle{Backend: h.Backend, SandboxId: h.SandboxID, NodeId: h.NodeID}
}

func (h handlePayload) handle() backend.Handle {
	return backend.Handle{Backend: h.Backend, SandboxID: h.SandboxID, NodeID: h.NodeID}
}

func (j *JSONListener) release(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload handlePayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	if _, err := j.control.Release(ctx, payload.proto()); err != nil {
		return nil, err
	}
	return map[string]any{}, nil
}

func (j *JSONListener) status(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload handlePayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	reply, err := j.control.Status(ctx, payload.proto())
	if err != nil {
		return nil, err
	}
	return map[string]any{"status": statusName(reply.GetStatus())}, nil
}

func (j *JSONListener) pause(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload struct {
		handlePayload
		Mode string `json:"mode"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	if _, err := j.control.Pause(ctx, &v1.PauseRequest{
		Handle: payload.proto(), Mode: pauseModeValue(payload.Mode),
	}); err != nil {
		return nil, err
	}
	return map[string]any{}, nil
}

func (j *JSONListener) resume(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload handlePayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	if _, err := j.control.Resume(ctx, payload.proto()); err != nil {
		return nil, err
	}
	return map[string]any{}, nil
}

func (j *JSONListener) snapshot(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload struct {
		handlePayload
		Kind string `json:"kind"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	ref, err := j.control.Snapshot(ctx, &v1.SnapshotRequest{
		Handle: payload.proto(), Kind: snapshotKindValue(payload.Kind),
	})
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"backend":      ref.GetBackend(),
		"snapshot_id":  ref.GetSnapshotId(),
		"kind":         snapshotKindName(ref.GetKind()),
		"resume_level": resumeLevelName(ref.GetResumeLevel()),
	}, nil
}

func (j *JSONListener) deleteSnapshot(ctx context.Context, raw json.RawMessage) (any, error) {
	var payload struct {
		Backend    string `json:"backend"`
		SnapshotID string `json:"snapshot_id"`
		Kind       string `json:"kind"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	if _, err := j.control.DeleteSnapshot(ctx, &v1.SnapshotRef{
		Backend: payload.Backend, SnapshotId: payload.SnapshotID, Kind: snapshotKindValue(payload.Kind),
	}); err != nil {
		return nil, err
	}
	return map[string]any{}, nil
}

// exec runs one command. In a fleet deployment the control plane forwards to
// the node that holds the sandbox via NodeClient.ExecOn, so a control-only
// process can still serve exec without being co-located with the sandbox.
// In a single-process deployment ExecOn calls the local node directly.
func (j *JSONListener) exec(ctx context.Context, raw json.RawMessage) (any, error) {
	if j.control.nodes == nil {
		return nil, status.Error(codes.Unimplemented,
			"no node client is configured; this process cannot route commands")
	}
	var payload struct {
		handlePayload
		Command string            `json:"command"`
		Cwd     string            `json:"cwd"`
		Env     map[string]string `json:"env"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	code, output, err := j.control.nodes.ExecOn(ctx, payload.handle(), payload.Command, payload.Cwd, payload.Env)
	if err != nil {
		return nil, status.Error(codes.Unavailable, err.Error())
	}
	return map[string]any{"exit_code": code, "stdout": output, "stderr": "", "truncated": false}, nil
}

func (j *JSONListener) readBytes(ctx context.Context, raw json.RawMessage) (any, error) {
	if j.control.nodes == nil {
		return nil, status.Error(codes.Unimplemented, "no node client is configured; this process cannot route file reads")
	}
	var payload struct {
		handlePayload
		Path string `json:"path"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	data, err := j.control.nodes.ReadBytesOn(ctx, payload.handle(), payload.Path)
	if err != nil {
		return nil, err
	}
	return map[string]any{"data": data}, nil
}

func (j *JSONListener) writeBytes(ctx context.Context, raw json.RawMessage) (any, error) {
	if j.control.nodes == nil {
		return nil, status.Error(codes.Unimplemented, "no node client is configured; this process cannot route file writes")
	}
	var payload struct {
		handlePayload
		Path string `json:"path"`
		Data string `json:"data"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	if err := j.control.nodes.WriteBytesOn(ctx, payload.handle(), payload.Path, payload.Data); err != nil {
		return nil, err
	}
	return map[string]any{}, nil
}

// compactBase64 strips the line breaks `base64` emits, which a decoder on the
// other side would otherwise have to tolerate.
func compactBase64(raw string) string {
	out := make([]byte, 0, len(raw))
	for i := 0; i < len(raw); i++ {
		if raw[i] != '\n' && raw[i] != '\r' {
			out = append(out, raw[i])
		}
	}
	// Validate here rather than letting the caller fail on a decode: a command
	// that printed something other than base64 is a node fault, not a bad file.
	if _, err := base64.StdEncoding.DecodeString(string(out)); err != nil {
		return ""
	}
	return string(out)
}

func (j *JSONListener) fleet(ctx context.Context) (any, error) {
	report, err := j.control.Fleet(ctx, &v1.Empty{})
	if err != nil {
		return nil, err
	}
	nodes := make([]any, 0, len(report.GetNodes()))
	for _, node := range report.GetNodes() {
		hosted := make([]string, 0, len(node.GetBackends()))
		for _, b := range node.GetBackends() {
			hosted = append(hosted, b.GetName())
		}
		nodes = append(nodes, map[string]any{
			"node_id":        node.GetNodeId(),
			"backends":       hosted,
			"live_sandboxes": node.GetLiveSandboxes(),
			"cpu_used_pct":   node.GetCpuUsedPct(),
			"mem_used_pct":   node.GetMemUsedPct(),
			"gpu_free":       node.GetGpuFree(),
			"draining":       node.GetDraining(),
		})
	}
	return map[string]any{
		"nodes":              nodes,
		"drained_nodes":      report.GetDrainedNodes(),
		"reservations_open":  report.GetReservationsOpen(),
		"capacity_exhausted": report.GetCapacityExhausted(),
		"no_candidate":       report.GetNoCandidate(),
		"locality_hit_ratio": report.GetLocalityHitRatio(),
	}, nil
}

func (j *JSONListener) quota(ctx context.Context) (any, error) {
	report, err := j.control.Quota(ctx, &v1.Empty{})
	if err != nil {
		return nil, err
	}
	classes := map[string]any{}
	for name, quota := range report.GetClasses() {
		classes[name] = map[string]any{
			"guaranteed_share":   quota.GetGuaranteedShare(),
			"max_share":          quota.GetMaxShare(),
			"granted_memory_mb":  quota.GetGranted().GetMemoryMb(),
			"headroom_memory_mb": quota.GetHeadroom().GetMemoryMb(),
			"queued":             quota.GetQueued(),
		}
	}
	return map[string]any{"classes": classes}, nil
}

// Enum values the SDK sends as names, mapped back explicitly so an unknown one
// is "unspecified" rather than a silently wrong value.

func sourceKindValue(name string) v1.Source_Kind {
	switch name {
	case "image":
		return v1.Source_IMAGE
	case "template":
		return v1.Source_TEMPLATE
	default:
		return v1.Source_KIND_UNSPECIFIED
	}
}

func execModeValue(name string) v1.ExecMode {
	switch name {
	case "persistent":
		return v1.ExecMode_PERSISTENT
	case "one_shot":
		return v1.ExecMode_ONE_SHOT
	default:
		return v1.ExecMode_EXEC_MODE_UNSPECIFIED
	}
}

func snapshotKindValue(name string) v1.SnapshotKind {
	switch name {
	case "filesystem":
		return v1.SnapshotKind_SNAPSHOT_FILESYSTEM
	case "full_state":
		return v1.SnapshotKind_SNAPSHOT_FULL_STATE
	default:
		return v1.SnapshotKind_SNAPSHOT_KIND_UNSPECIFIED
	}
}

func statusName(s v1.SandboxStatus) string {
	switch s {
	case v1.SandboxStatus_RUNNING:
		return "running"
	case v1.SandboxStatus_PAUSED:
		return "paused"
	case v1.SandboxStatus_EXITED:
		return "exited"
	case v1.SandboxStatus_TERMINATED:
		return "terminated"
	default:
		return "unknown"
	}
}

// RefreshFleet republishes this process's node into the monitor, for a
// single-process deployment where the two share memory.
func (j *JSONListener) RefreshFleet(report func(placement.NodeView)) {
	if j.node == nil {
		return
	}
	report(j.node.View())
}
