// Direct mode: this service drives the Docker runtime itself and stages
// OpenSandbox's agent into each sandbox.
//
// What this buys over reaching the same runtime through OpenSandbox's lifecycle
// server is entirely in the create path. That server is a Python process: it
// holds one event loop, it starts a thread per create, and its port allocator
// serializes every create on one process-wide lock. A rollout step that opens a
// hundred sandboxes at once meets all three. Driving the daemon from here makes
// the same work concurrent, because the only limit is the semaphore this file
// keeps and the daemon's own throughput.
//
// What it does not change is the data plane. The agent is the same Go binary
// either way, it listens on the same port, and the SDK reaches it directly in
// both shapes -- so commands, files, PTYs, and code contexts are identical. The
// lifecycle server was never in the command path, and removing it does not make
// the command path shorter.
//
// Three things that would otherwise be the lifecycle server's job are done here:
// staging the agent (once, in execd.go, rather than per container), allocating
// the port the agent is reached on, and composing the container so bootstrap.sh
// starts the agent beside the workload.
package opensandbox

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// agentPort is where execd listens inside every sandbox.
const agentPort = 44772

// Labels this service writes on each container. They are what makes a restart
// able to find its own sandboxes, and what keeps a reclaim from touching a
// container this service did not create.
const (
	sandboxIDLabel = "opensandbox.psrl/sandbox-id"
	nodeIDLabel    = "opensandbox.psrl/node-id"
	agentPortLabel = "opensandbox.psrl/agent-port"
)

// directRuntime drives one Docker daemon and composes agent-bearing sandboxes.
type directRuntime struct {
	docker  *dockerClient
	agent   stagedAgent
	nodeID  string
	ownerID string

	// networkMode is passed through to the daemon. "host" removes the per-sandbox
	// veth, netns, and iptables work from the create path, which on a cgroup v1
	// kernel is the difference between a create that scales and one that does not.
	// It also means the agent is reached on the host's own port, so no mapping is
	// allocated.
	networkMode string

	// runtime is an OCI runtime name (gVisor, Kata). Empty uses the daemon's.
	runtime string

	// createSem bounds concurrent create+start pairs. The daemon serializes parts
	// of container setup in the kernel, so an unbounded burst converts latency
	// into timeouts rather than throughput.
	createSem chan struct{}

	ports *portPool
}

// directConfig is what a direct runtime needs to be built.
type directConfig struct {
	Socket               string
	APIVersion           string
	NodeID               string
	OwnerID              string
	Runtime              string
	NetworkMode          string
	MaxCreateConcurrency int
	ExecdImage           string
	StageDir             string
	PortMin              int
	PortMax              int
}

// newDirectRuntime dials the daemon and stages the agent.
//
// Staging happens here rather than on the first create, so an unusable agent
// image fails the deployment at startup instead of inside an episode.
func newDirectRuntime(ctx context.Context, cfg directConfig) (*directRuntime, error) {
	client, err := newDockerClient(cfg.Socket, cfg.APIVersion)
	if err != nil {
		return nil, err
	}
	if err := client.ping(ctx); err != nil {
		return nil, fmt.Errorf("the Docker daemon at %s is not answering: %w", cfg.Socket, err)
	}
	agent, err := stageAgent(ctx, client, cfg.ExecdImage, cfg.StageDir)
	if err != nil {
		return nil, err
	}
	runtime := &directRuntime{
		docker:      client,
		agent:       agent,
		nodeID:      cfg.NodeID,
		ownerID:     cfg.OwnerID,
		networkMode: cfg.NetworkMode,
		runtime:     cfg.Runtime,
	}
	if cfg.MaxCreateConcurrency > 0 {
		runtime.createSem = make(chan struct{}, cfg.MaxCreateConcurrency)
	}
	// Only a mapped deployment needs a pool. Under host networking the agent
	// listens on the host port directly and there is nothing to allocate.
	if !runtime.hostNetworked() {
		runtime.ports = newPortPool(cfg.PortMin, cfg.PortMax)
	}
	return runtime, nil
}

func (r *directRuntime) hostNetworked() bool { return r.networkMode == "host" }

// features are the capabilities the staged agent actually serves.
//
// Read from what was staged rather than declared as a constant: an agent image
// without bubblewrap serves every route except isolated sessions, and claiming
// that route anyway would admit a spec the sandbox then could not satisfy.
func (r *directRuntime) features() []string {
	features := []string{
		// The agent's own surface.
		"persistent_session",
		"background_command",
		"code_interpreter",
		"pty",
		// The runtime's.
		"freeze",
		"filesystem_snapshot",
		"restore",
		"image_on_demand",
		"credential_injection",
		"host_mount",
	}
	if r.agent.HasBwrap {
		features = append(features, "isolated_session")
	}
	if r.runtime != "" {
		features = append(features, "isolation_runtime")
	}
	sort.Strings(features)
	return features
}

// create composes and starts one sandbox, and returns where its agent answers.
func (r *directRuntime) create(ctx context.Context, spec backend.Spec) (backend.Created, error) {
	if spec.Source.Kind != "" && spec.Source.Kind != "image" {
		return backend.Created{}, fmt.Errorf(
			"a directly driven sandbox starts from an image, not source kind %q", spec.Source.Kind)
	}
	if spec.Source.Reference == "" {
		return backend.Created{}, fmt.Errorf("a directly driven sandbox needs an image reference")
	}

	sandboxID := newSandboxID()

	// A mapped deployment needs a host port before the container is composed,
	// because the mapping is part of the create body.
	hostPort := agentPort
	if r.ports != nil {
		reserved, err := r.ports.take()
		if err != nil {
			return backend.Created{}, err
		}
		hostPort = reserved
		defer func() {
			// Released on every failure path. A port leaked per failed create
			// would exhaust the range over a long run.
			if hostPort != 0 {
				r.ports.release(hostPort)
			}
		}()
	}

	body := r.createBody(sandboxID, spec, hostPort)

	if r.createSem != nil {
		select {
		case r.createSem <- struct{}{}:
			defer func() { <-r.createSem }()
		case <-ctx.Done():
			return backend.Created{}, ctx.Err()
		}
	}

	containerID, err := r.docker.createContainer(ctx, body, "")
	if err != nil {
		return backend.Created{}, fmt.Errorf("creating the sandbox container: %w", err)
	}
	if err := r.docker.startContainer(ctx, containerID); err != nil {
		// The container exists but will not run. Removing it here keeps a failed
		// create from leaving a stopped container behind for the sweep to find.
		removeCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_ = r.docker.removeContainer(removeCtx, containerID, true)
		return backend.Created{}, fmt.Errorf("starting the sandbox container: %w", err)
	}

	address := fmt.Sprintf("http://127.0.0.1:%d", hostPort)
	// Ownership of the port passes to the live sandbox; the deferred release must
	// not take it back.
	hostPort = 0

	return backend.Created{
		Handle: backend.Handle{
			Backend:   "opensandbox",
			SandboxID: containerID,
			NodeID:    r.nodeID,
		},
		Agent: backend.AgentEndpoint{Address: address},
	}, nil
}

// createBody composes the container: the workload image, the agent mounted
// beside it, and bootstrap.sh as the entrypoint.
//
// bootstrap.sh is the contract. It starts the agent, then execs the workload's
// own command, so a sandbox image needs no awareness of OpenSandbox. What it
// reads from the environment is set here: which binary to run, whether the agent
// is the container's init, and what the workload command was.
func (r *directRuntime) createBody(sandboxID string, spec backend.Spec, hostPort int) map[string]any {
	env := []string{
		// Where bootstrap.sh finds the agent.
		"EXECD=" + SandboxAgentDir + "/" + stagedExecd,
		// The agent becomes PID 1 and reaps orphans. Without this a long-lived
		// sandbox accumulates zombies, because a bare container's PID 1 is the
		// workload and it does not reap.
		"EXECD_INIT=true",
	}
	for key, value := range spec.Env {
		env = append(env, key+"="+value)
	}
	// Sorted so one spec always produces one body: an idempotent retry must not
	// differ from its first attempt by map order alone.
	sort.Strings(env)

	// The workload's own command, handed to bootstrap.sh to exec once the agent
	// is up. Empty leaves the image's own entrypoint, which bootstrap.sh resolves.
	if spec.Workdir != "" {
		env = append(env, "EXECD_WORKDIR="+spec.Workdir)
	}

	labels := map[string]string{
		sandboxIDLabel: sandboxID,
		nodeIDLabel:    r.nodeID,
		agentPortLabel: strconv.Itoa(hostPort),
		ownerLabel:     r.ownerID,
	}
	for key, value := range spec.Metadata {
		labels[key] = value
	}

	hostConfig := map[string]any{
		// Read-only: a sandbox must not be able to replace the agent it is
		// observed through.
		"Binds":         []string{r.agent.Dir + ":" + SandboxAgentDir + ":ro"},
		"RestartPolicy": map[string]any{"Name": "no"},
	}
	if r.networkMode != "" {
		hostConfig["NetworkMode"] = r.networkMode
	}
	if r.runtime != "" {
		hostConfig["Runtime"] = r.runtime
	}
	if spec.Resources.MemoryMB > 0 {
		hostConfig["Memory"] = spec.Resources.MemoryMB * 1024 * 1024
	}
	if spec.Resources.CPUCount > 0 {
		hostConfig["NanoCpus"] = int64(spec.Resources.CPUCount * 1e9)
	}
	if len(spec.AssignedGPUs) > 0 {
		hostConfig["DeviceRequests"] = []map[string]any{{
			"Driver":       "nvidia",
			"DeviceIDs":    gpuIDs(spec.AssignedGPUs),
			"Capabilities": [][]string{{"gpu"}},
		}}
	}

	body := map[string]any{
		"Image":      spec.Source.Reference,
		"Entrypoint": []string{SandboxAgentDir + "/" + stagedBootstrap},
		"Env":        env,
		"Labels":     labels,
		"HostConfig": hostConfig,
	}
	if spec.Workdir != "" {
		body["WorkingDir"] = spec.Workdir
	}
	// Under host networking the agent is already on a host port, so a mapping
	// would be rejected as well as pointless.
	if !r.hostNetworked() {
		port := strconv.Itoa(agentPort) + "/tcp"
		body["ExposedPorts"] = map[string]any{port: map[string]any{}}
		hostConfig["PortBindings"] = map[string]any{
			port: []map[string]string{{"HostIp": "127.0.0.1", "HostPort": strconv.Itoa(hostPort)}},
		}
	}
	return body
}

// release destroys one sandbox and returns its port to the pool.
func (r *directRuntime) release(ctx context.Context, handle backend.Handle) error {
	if r.ports != nil {
		if port := r.portOf(ctx, handle.SandboxID); port > 0 {
			defer r.ports.release(port)
		}
	}
	if err := r.docker.removeContainer(ctx, handle.SandboxID, true); err != nil {
		return fmt.Errorf("releasing sandbox %s: %w", handle.SandboxID, err)
	}
	return nil
}

// portOf reads the port this service recorded on the container.
//
// Read back rather than remembered in a map, because the container is the source
// of truth: a restarted service must release the ports of sandboxes it did not
// create in this process.
func (r *directRuntime) portOf(ctx context.Context, containerID string) int {
	var inspected struct {
		Config struct {
			Labels map[string]string `json:"Labels"`
		} `json:"Config"`
	}
	if err := r.docker.call(ctx, "GET", "/containers/"+containerID+"/json", nil, &inspected); err != nil {
		return 0
	}
	port, err := strconv.Atoi(inspected.Config.Labels[agentPortLabel])
	if err != nil {
		return 0
	}
	return port
}

// status maps the daemon's container state onto the portable one.
func (r *directRuntime) status(ctx context.Context, handle backend.Handle) (string, error) {
	var inspected struct {
		State struct {
			Status string `json:"Status"`
			Paused bool   `json:"Paused"`
		} `json:"State"`
	}
	if err := r.docker.call(ctx, "GET", "/containers/"+handle.SandboxID+"/json", nil, &inspected); err != nil {
		if isDockerNotFound(err) {
			return "terminated", nil
		}
		return "", err
	}
	if inspected.State.Paused {
		return "paused", nil
	}
	switch inspected.State.Status {
	case "running":
		return "running", nil
	case "paused":
		return "paused", nil
	case "exited", "dead", "removing":
		return "terminated", nil
	case "created", "restarting":
		return "unknown", nil
	default:
		return "unknown", nil
	}
}

// pause freezes the sandbox in place.
//
// A container freeze keeps memory resident, so this is not a hibernation and
// does not return the sandbox's compute. Requesting one is refused rather than
// served as a freeze: a caller that expected its memory to be written out and
// its compute released would otherwise be told it was.
func (r *directRuntime) pause(ctx context.Context, handle backend.Handle, mode string) error {
	if mode != "" && mode != "freeze" {
		return fmt.Errorf(
			"a directly driven sandbox stays resident on pause, so it freezes rather than %q", mode)
	}
	return r.docker.call(ctx, "POST", "/containers/"+handle.SandboxID+"/pause", nil, nil)
}

func (r *directRuntime) resume(ctx context.Context, handle backend.Handle) error {
	return r.docker.call(ctx, "POST", "/containers/"+handle.SandboxID+"/unpause", nil, nil)
}

// snapshot commits the sandbox's filesystem and returns the image reference.
//
// The commit captures the writable layer and not memory, so "full_state" is
// refused. The result is a local image: it restores on this node, and a restore
// elsewhere needs the image published, which this runtime does not do on its own.
func (r *directRuntime) snapshot(ctx context.Context, handle backend.Handle, kind string) (string, error) {
	if kind != "" && kind != "filesystem" {
		return "", fmt.Errorf(
			"a container commit captures the filesystem, so it takes a filesystem snapshot rather than %q", kind)
	}
	tag := "sb-" + handle.SandboxID[:min(12, len(handle.SandboxID))] + "-" + strconv.FormatInt(time.Now().UnixNano(), 36)
	path := fmt.Sprintf("/commit?container=%s&repo=%s&tag=%s", handle.SandboxID, snapshotRepository, tag)
	var committed struct {
		ID string `json:"Id"`
	}
	if err := r.docker.call(ctx, "POST", path, nil, &committed); err != nil {
		return "", fmt.Errorf("committing sandbox %s: %w", handle.SandboxID, err)
	}
	return snapshotRepository + ":" + tag, nil
}

const snapshotRepository = "psrl-opensandbox-snapshot"

// deleteSnapshot removes a committed image.
func (r *directRuntime) deleteSnapshot(ctx context.Context, snapshotID string) error {
	err := r.docker.call(ctx, "DELETE", "/images/"+snapshotID+"?force=1", nil, nil)
	if err != nil && isDockerNotFound(err) {
		return nil
	}
	return err
}

// -- port allocation ----------------------------------------------------------

// portPool hands out host ports for the agent.
//
// A pool rather than probing for a free port: probing means a bind, a close, and
// a race between the two, and serializing that behind a lock is exactly the
// bottleneck this mode exists to avoid. Reserving from a known range under one
// short critical section is O(1) and never touches the network stack.
type portPool struct {
	mu    sync.Mutex
	next  int
	min   int
	max   int
	taken map[int]bool
}

func newPortPool(minPort, maxPort int) *portPool {
	if minPort <= 0 {
		minPort = 45000
	}
	if maxPort <= minPort {
		maxPort = minPort + 10000
	}
	return &portPool{next: minPort, min: minPort, max: maxPort, taken: map[int]bool{}}
}

// take reserves the next free port, wrapping once before giving up.
func (p *portPool) take() (int, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	span := p.max - p.min + 1
	for tried := 0; tried < span; tried++ {
		port := p.next
		p.next++
		if p.next > p.max {
			p.next = p.min
		}
		if !p.taken[port] {
			p.taken[port] = true
			return port, nil
		}
	}
	return 0, fmt.Errorf(
		"every port in %d-%d is reserved; this clears as sandboxes are released", p.min, p.max)
}

func (p *portPool) release(port int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	delete(p.taken, port)
}

// -- helpers ------------------------------------------------------------------

func gpuIDs(indices []int32) []string {
	ids := make([]string, 0, len(indices))
	for _, index := range indices {
		ids = append(ids, strconv.Itoa(int(index)))
	}
	return ids
}

// newSandboxID is this service's own correlation id, recorded as a label.
//
// The container id remains the handle, because it is what the daemon answers to.
// This one makes a container traceable to the create that asked for it even
// after the handle is gone.
func newSandboxID() string {
	return "sb-" + strconv.FormatInt(time.Now().UnixNano(), 36)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// agentAddressFrom rebuilds an agent endpoint from a container's labels, for a
// handle that was created by an earlier process.
func (r *directRuntime) agentAddressFrom(ctx context.Context, containerID string) string {
	port := r.portOf(ctx, containerID)
	if port == 0 {
		return ""
	}
	return fmt.Sprintf("http://127.0.0.1:%d", port)
}

// trimmed is used for error text where a full container id is noise.
func trimmed(id string) string {
	if len(id) > 12 {
		return id[:12]
	}
	return strings.TrimSpace(id)
}
