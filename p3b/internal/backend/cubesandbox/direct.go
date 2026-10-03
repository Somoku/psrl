// Direct mode: this service drives each Cubelet over gRPC and places sandboxes
// itself, with CubeMaster neither deployed nor in the path.
//
// The seam this uses is one CubeSandbox draws itself. A Cubelet serves a complete
// node-level lifecycle -- CubeboxMgr, with Create, Destroy, Exec, CommitSandbox,
// RollbackSandbox and more -- and CubeMaster's only contribution above it is
// choosing which Cubelet to call: its own helpers take the node endpoint as a
// plain argument, and its scheduler exists to produce that argument. Replacing
// that choice with this service's placement leaves everything underneath intact.
//
// What this costs is the request body. CubeMaster assembles RunCubeSandboxRequest
// across a large amount of code, and the parts of it that matter have to be
// assembled here instead: the container spec, the volumes, the annotations that
// select a runtime handler and an instance type. That is the real work in this
// file, not the RPC.
//
// What this does not reimplement is anything below the Cubelet: the microVM boot,
// the overlaybd layer system, and the local snapshot store are CubeSandbox's.
//
// Two CubeMaster-level features are therefore absent here, and the adapter says so
// rather than declaring them: a warm pool is a CubeMaster pool, and a template
// build is a CubeMaster API. A spec requiring either is refused at admission
// instead of failing at create.
package cubesandbox

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"psrl.dev/sandboxd/internal/backend"
	cubebox "psrl.dev/sandboxd/internal/backend/cubesandbox/cubeletpb/services/cubebox/v1"
	errorcode "psrl.dev/sandboxd/internal/backend/cubesandbox/cubeletpb/services/errorcode/v1"
	images "psrl.dev/sandboxd/internal/backend/cubesandbox/cubeletpb/services/images/v1"
)

// Annotation keys CubeMaster uses to steer a create. They are passed through
// rather than invented: a Cubelet reads the same keys whoever called it.
const (
	// annotationInstanceType selects the lifecycle shape Cubelet applies.
	annotationInstanceType = "cube.master.instance.type"
	// annotationOwner records which service created the sandbox, so a sweep can
	// tell its own sandboxes from another's.
	annotationOwner = "psrl.sandboxd/owner"
	// annotationSandboxClass carries the resource class for correlation.
	annotationSandboxClass = "psrl.sandboxd/resource-class"
)

// mainContainer is the name given to the sandbox's single container. Cubelet
// addresses containers by name within a sandbox, and Exec needs one.
const mainContainer = "sandbox"

// cubeletConn is one Cubelet and the connection to it.
type cubeletConn struct {
	nodeID string
	target string
	conn   *grpc.ClientConn
	client cubebox.CubeboxMgrClient
}

// directRuntime holds a connection per Cubelet and places across them.
//
// Connections are built once and kept: gRPC multiplexes concurrent calls over one
// HTTP/2 connection, so a burst of creates needs no connection pool and pays no
// handshake per sandbox.
type directRuntime struct {
	mu      sync.RWMutex
	nodes   map[string]*cubeletConn
	ownerID string

	// createSem bounds concurrent creates against one fleet. A Cubelet boots a
	// microVM per create, and an unbounded burst turns latency into timeouts.
	createSem chan struct{}

	requestTimeout time.Duration
	createTimeout  time.Duration
}

// newDirectRuntime dials every configured Cubelet.
//
// Dialling here rather than on the first create means a node that is not
// listening fails the deployment at startup, not inside an episode. gRPC dials
// lazily by default, so the connection is forced to be established.
func newDirectRuntime(
	ctx context.Context, nodes []NodeAddress, ownerID string,
	maxCreateConcurrency int, requestTimeout, createTimeout time.Duration,
) (*directRuntime, error) {
	runtime := &directRuntime{
		nodes:          make(map[string]*cubeletConn, len(nodes)),
		ownerID:        ownerID,
		requestTimeout: requestTimeout,
		createTimeout:  createTimeout,
	}
	if maxCreateConcurrency > 0 {
		runtime.createSem = make(chan struct{}, maxCreateConcurrency)
	}
	for _, node := range nodes {
		target := grpcTarget(node.Address)
		conn, err := grpc.NewClient(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			runtime.Close()
			return nil, fmt.Errorf("cubelet %s at %s cannot be dialled: %w", node.NodeID, target, err)
		}
		runtime.nodes[node.NodeID] = &cubeletConn{
			nodeID: node.NodeID,
			target: target,
			conn:   conn,
			client: cubebox.NewCubeboxMgrClient(conn),
		}
	}
	return runtime, nil
}

// Close releases every connection.
func (r *directRuntime) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, node := range r.nodes {
		if node.conn != nil {
			_ = node.conn.Close()
		}
	}
}

// nodeIDs are the Cubelets placement may choose between.
func (r *directRuntime) nodeIDs() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ids := make([]string, 0, len(r.nodes))
	for id := range r.nodes {
		ids = append(ids, id)
	}
	return ids
}

// node resolves one Cubelet, or says it is not configured.
func (r *directRuntime) node(nodeID string) (*cubeletConn, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if nodeID == "" {
		// An operation not tied to a sandbox. Any Cubelet can answer it.
		for _, node := range r.nodes {
			return node, nil
		}
		return nil, fmt.Errorf("cubesandbox has no configured cubelets")
	}
	node, known := r.nodes[nodeID]
	if !known {
		return nil, fmt.Errorf("cubelet %q is not configured", nodeID)
	}
	return node, nil
}

// create boots one microVM sandbox on the chosen Cubelet.
func (r *directRuntime) create(ctx context.Context, nodeID string, spec backend.Spec) (backend.Created, error) {
	node, err := r.node(nodeID)
	if err != nil {
		return backend.Created{}, err
	}

	if r.createSem != nil {
		select {
		case r.createSem <- struct{}{}:
			defer func() { <-r.createSem }()
		case <-ctx.Done():
			return backend.Created{}, ctx.Err()
		}
	}

	ctx, cancel := context.WithTimeout(ctx, r.createTimeout)
	defer cancel()

	reply, err := node.client.Create(ctx, r.createRequest(spec))
	if err != nil {
		return backend.Created{}, fmt.Errorf("cubelet %s create: %w", node.nodeID, err)
	}
	if err := retError(reply.GetRet()); err != nil {
		return backend.Created{}, fmt.Errorf("cubelet %s refused the create: %w", node.nodeID, err)
	}
	if reply.GetSandboxID() == "" {
		return backend.Created{}, fmt.Errorf("cubelet %s created a sandbox but returned no id", node.nodeID)
	}

	return backend.Created{
		Handle: backend.Handle{
			Backend:   "cubesandbox",
			SandboxID: reply.GetSandboxID(),
			NodeID:    node.nodeID,
		},
		// A Cubelet sandbox runs envd, but its address is published by CubeMaster
		// rather than returned here, so there is no in-sandbox endpoint to report.
		// An empty address makes the owning node proxy command traffic, which keeps
		// it node-local and off the control plane either way.
		Agent: backend.AgentEndpoint{},
	}, nil
}

// createRequest assembles what CubeMaster would have assembled.
//
// The portable spec carries less than RunCubeSandboxRequest can express, and the
// fields it does not reach are left at their zero value rather than guessed:
// Cubelet treats an empty runtime handler as "the default", which is the right
// behaviour for a spec that asked for no particular one.
func (r *directRuntime) createRequest(spec backend.Spec) *cubebox.RunCubeSandboxRequest {
	annotations := map[string]string{
		annotationOwner: r.ownerID,
	}
	if spec.ResourceClass != "" {
		annotations[annotationSandboxClass] = spec.ResourceClass
	}
	// Backend options are namespaced to this backend and are the only way a caller
	// reaches a Cubelet knob the portable spec has no field for.
	instanceType := ""
	if options, named := spec.Options("cubesandbox"); named {
		for key, value := range options {
			switch key {
			case "instance_type":
				instanceType = value
			case "runtime_handler", "network_type", "namespace", "backend":
				// Handled below, where the request field exists.
			default:
				annotations[key] = value
			}
		}
	}
	if instanceType != "" {
		annotations[annotationInstanceType] = instanceType
	}

	container := &cubebox.ContainerConfig{
		Name:  mainContainer,
		Image: &images.ImageSpec{Image: spec.Source.Reference},
		Envs:  keyValues(withDevicePartition(spec)),
	}
	if spec.Workdir != "" {
		container.WorkingDir = spec.Workdir
	}
	if resources := containerResources(spec); resources != nil {
		container.Resources = resources
	}

	request := &cubebox.RunCubeSandboxRequest{
		RequestID:   newRequestID(),
		Containers:  []*cubebox.ContainerConfig{container},
		Annotations: annotations,
		Labels:      labelsFrom(spec),
	}
	if options, named := spec.Options("cubesandbox"); named {
		request.RuntimeHandler = options["runtime_handler"]
		request.NetworkType = options["network_type"]
		request.Namespace = options["namespace"]
		request.Backend = options["backend"]
		request.InstanceType = instanceType
	}
	return request
}

// release destroys one sandbox.
func (r *directRuntime) release(ctx context.Context, handle backend.Handle) error {
	node, err := r.node(handle.NodeID)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, r.requestTimeout)
	defer cancel()
	reply, err := node.client.Destroy(ctx, &cubebox.DestroyCubeSandboxRequest{
		RequestID: newRequestID(),
		SandboxID: handle.SandboxID,
	})
	if err != nil {
		return fmt.Errorf("cubelet %s release %s: %w", node.nodeID, handle.SandboxID, err)
	}
	if err := retError(reply.GetRet()); err != nil && !isAlreadyGone(err) {
		return fmt.Errorf("cubelet %s release %s: %w", node.nodeID, handle.SandboxID, err)
	}
	// Already gone is the outcome the caller wanted.
	return nil
}

// status reports one sandbox's portable state.
//
// Cubelet answers List rather than a per-sandbox Get, so the id is the filter.
// The reply carries no status field and no result code: a sandbox that exists is
// in the items, and one that is gone is absent. Paused is distinguished by
// paused_at being set, which is the only state marker the record carries.
func (r *directRuntime) status(ctx context.Context, handle backend.Handle) (string, error) {
	node, err := r.node(handle.NodeID)
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(ctx, r.requestTimeout)
	defer cancel()
	sandboxID := handle.SandboxID
	reply, err := node.client.List(ctx, &cubebox.ListCubeSandboxRequest{Id: &sandboxID})
	if err != nil {
		return "", fmt.Errorf("cubelet %s status %s: %w", node.nodeID, handle.SandboxID, err)
	}
	items := reply.GetItems()
	if len(items) == 0 {
		// Absent from its own node's list is the one unambiguous signal Cubelet
		// gives that a sandbox is gone.
		return "terminated", nil
	}
	if items[0].GetPausedAt() > 0 {
		return "paused", nil
	}
	return "running", nil
}

// snapshot commits the sandbox into a node-local template snapshot.
//
// The capture is the filesystem, so "full_state" is refused rather than served:
// a caller that expected a live process to survive a restore would otherwise read
// a workspace restore as proof that it did. The snapshot stays on this node unless
// the deployment configures a shared CoW backend, which is why the node travels
// on the handle.
func (r *directRuntime) snapshot(ctx context.Context, handle backend.Handle, kind string) (string, error) {
	if kind != "" && kind != "filesystem" {
		return "", fmt.Errorf(
			"a cubesandbox commit captures the filesystem, so it takes a filesystem snapshot rather than %q", kind)
	}
	node, err := r.node(handle.NodeID)
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(ctx, r.createTimeout)
	defer cancel()
	templateID := "psrl-" + handle.SandboxID + "-" + strconv.FormatInt(time.Now().UnixNano(), 36)
	reply, err := node.client.CommitSandbox(ctx, &cubebox.CommitSandboxRequest{
		RequestID:  newRequestID(),
		SandboxID:  handle.SandboxID,
		TemplateID: templateID,
	})
	if err != nil {
		return "", fmt.Errorf("cubelet %s snapshot %s: %w", node.nodeID, handle.SandboxID, err)
	}
	if err := retError(reply.GetRet()); err != nil {
		return "", fmt.Errorf("cubelet %s snapshot %s: %w", node.nodeID, handle.SandboxID, err)
	}
	return templateID, nil
}

// deleteSnapshot removes a committed template from the node that holds it.
func (r *directRuntime) deleteSnapshot(ctx context.Context, snapshotID string) error {
	// A snapshot is node-local, and the id does not name its node, so every
	// Cubelet is asked. One of them holds it; the others report nothing to clean,
	// which is not a failure.
	r.mu.RLock()
	nodes := make([]*cubeletConn, 0, len(r.nodes))
	for _, node := range r.nodes {
		nodes = append(nodes, node)
	}
	r.mu.RUnlock()

	ctx, cancel := context.WithTimeout(ctx, r.requestTimeout)
	defer cancel()
	var lastErr error
	for _, node := range nodes {
		reply, err := node.client.CleanupTemplate(ctx, &cubebox.CleanupTemplateRequest{
			RequestID:  newRequestID(),
			TemplateID: snapshotID,
		})
		if err != nil {
			lastErr = err
			continue
		}
		if err := retError(reply.GetRet()); err == nil || isAlreadyGone(err) {
			return nil
		}
	}
	if lastErr != nil {
		return fmt.Errorf("deleting snapshot %s: %w", snapshotID, lastErr)
	}
	return nil
}

// preflight confirms every Cubelet answers.
//
// List with no filter is the cheapest call that proves the service is serving
// rather than merely that a port is open.
func (r *directRuntime) preflight(ctx context.Context) error {
	r.mu.RLock()
	nodes := make([]*cubeletConn, 0, len(r.nodes))
	for _, node := range r.nodes {
		nodes = append(nodes, node)
	}
	r.mu.RUnlock()

	for _, node := range nodes {
		callCtx, cancel := context.WithTimeout(ctx, r.requestTimeout)
		_, err := node.client.List(callCtx, &cubebox.ListCubeSandboxRequest{})
		cancel()
		if err != nil {
			return fmt.Errorf("cubelet %s at %s is not answering: %w", node.nodeID, node.target, err)
		}
	}
	return nil
}

// -- translation ---------------------------------------------------------------

// retError turns a Cubelet result code into an error, or nil on success.
func retError(ret *errorcode.Ret) error {
	if ret == nil {
		// No result block is success: Cubelet omits it when there is nothing to
		// report.
		return nil
	}
	// Two codes mean success: Success (200) is what a Cubelet sets explicitly, and
	// OK (0) is the zero value a reply that set no code carries. Treating only one
	// as success would read half the successful replies as failures.
	switch ret.GetRetCode() {
	case errorcode.ErrorCode_Success, errorcode.ErrorCode_OK:
		return nil
	}
	return &cubeletError{code: ret.GetRetCode(), message: ret.GetRetMsg()}
}

type cubeletError struct {
	code    errorcode.ErrorCode
	message string
}

func (e *cubeletError) Error() string {
	return fmt.Sprintf("cubelet returned %s: %s", e.code.String(), e.message)
}

// Code exposes the result code so a caller can tell a sandbox that is gone from
// one that failed.
func (e *cubeletError) Code() errorcode.ErrorCode { return e.code }

// isAlreadyGone reports whether an error means the sandbox or template does not
// exist, which for a delete is the outcome the caller wanted.
func isAlreadyGone(err error) bool {
	var typed *cubeletError
	for err != nil {
		if candidate, ok := err.(*cubeletError); ok {
			typed = candidate
			break
		}
		unwrapper, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = unwrapper.Unwrap()
	}
	if typed == nil {
		return false
	}
	// Cubelet names its absence codes with NotFound / NotExist, so matching the
	// name keeps this working as codes are added rather than pinning each number.
	name := typed.code.String()
	return strings.Contains(name, "NotFound") || strings.Contains(name, "NotExist")
}

// containerResources maps the portable numeric resources onto Cubelet's.
//
// Cubelet states resources as quantity strings rather than integers, in the
// Kubernetes form: cpu in millicores and memory with a binary suffix. Both the
// request and the limit are set to the same value, because the portable spec
// states one number and a burstable gap nobody asked for would let a sandbox
// exceed what admission charged it.
//
// A zero field states no requirement and sends nothing, which is not the same as
// requesting zero: a zero quantity is rejected.
func containerResources(spec backend.Spec) *cubebox.Resource {
	if spec.Resources.CPUCount <= 0 && spec.Resources.MemoryMB <= 0 {
		return nil
	}
	resources := &cubebox.Resource{}
	if spec.Resources.CPUCount > 0 {
		cpu := fmt.Sprintf("%dm", int64(spec.Resources.CPUCount*1000))
		resources.Cpu = cpu
		resources.CpuLimit = cpu
	}
	if spec.Resources.MemoryMB > 0 {
		mem := fmt.Sprintf("%dMi", spec.Resources.MemoryMB)
		resources.Mem = mem
		resources.MemLimit = mem
	}
	return resources
}

// withDevicePartition returns the spec's environment with the granted GPU
// indices stated as CUDA_VISIBLE_DEVICES.
//
// Admission partitions the node's devices and records the grant on the spec, but
// a partition nothing enforces is a comment: every CUDA process in the sandbox
// would enumerate the whole machine and contend with a sibling that was granted
// a different device. A spec that was granted none is returned unchanged rather
// than being given an empty value, which would hide the node's devices from a
// sandbox that never asked about them.
func withDevicePartition(spec backend.Spec) map[string]string {
	if len(spec.AssignedGPUs) == 0 {
		return spec.Env
	}
	env := make(map[string]string, len(spec.Env)+1)
	for key, value := range spec.Env {
		env[key] = value
	}
	indices := make([]string, 0, len(spec.AssignedGPUs))
	for _, index := range spec.AssignedGPUs {
		indices = append(indices, strconv.Itoa(int(index)))
	}
	env["CUDA_VISIBLE_DEVICES"] = strings.Join(indices, ",")
	return env
}

func keyValues(env map[string]string) []*cubebox.KeyValue {
	if len(env) == 0 {
		return nil
	}
	pairs := make([]*cubebox.KeyValue, 0, len(env))
	for key, value := range env {
		pairs = append(pairs, &cubebox.KeyValue{Key: key, Value: value})
	}
	return pairs
}

func labelsFrom(spec backend.Spec) map[string]string {
	if len(spec.Metadata) == 0 {
		return nil
	}
	labels := make(map[string]string, len(spec.Metadata))
	for key, value := range spec.Metadata {
		labels[key] = value
	}
	return labels
}

// grpcTarget strips a scheme a URL-shaped address may carry: gRPC dials host:port.
func grpcTarget(address string) string {
	for _, scheme := range []string{"grpc://", "http://", "https://", "tcp://"} {
		if trimmed, found := strings.CutPrefix(address, scheme); found {
			return strings.TrimRight(trimmed, "/")
		}
	}
	return strings.TrimRight(address, "/")
}

func newRequestID() string {
	return "psrl-" + strconv.FormatInt(time.Now().UnixNano(), 36)
}
