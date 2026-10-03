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
//
// One constraint shapes the rest. The agent listens on a fixed port and nothing
// in the bootstrap path can change it, so each sandbox needs its own network
// namespace and a published port -- host networking would have every sandbox
// after the first fail to bind. That is why this backend keeps a port pool and
// refuses network_mode "host", even though the container backend prefers it.
package opensandbox

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// agentPort is where execd listens inside every sandbox.
const agentPort = 44772

// keepaliveCommand is what a sandbox's PID 1 runs so the container stays up.
//
// A sandbox exists to serve commands over its agent, not to run one workload and
// exit, so its init has to block. `sleep infinity` is not portable to a BusyBox
// image, and `tail -f /dev/null` is the form that works everywhere.
const keepaliveCommand = "exec tail -f /dev/null"

// agentReadyTimeout bounds the wait for execd to answer after its container
// starts.
//
// A started container is not a usable sandbox: the agent is a process inside it
// and it binds its port a moment later. Returning before it answers would hand
// back a sandbox whose first command fails for a reason that looks like a
// workload fault, so the wait happens here and a startup failure stays in the
// create where it belongs.
const agentReadyTimeout = 30 * time.Second

// Readiness probe pacing.
//
// The first delay sits below the agent's typical bind time rather than above it,
// so a warm create is not charged for a sleep it did not need. Backing off from
// there bounds the cost of a cold create: an image that has to be pulled can take
// seconds, and a 2 ms spin across that would be thousands of pointless syscalls.
const (
	agentProbeFirstDelay = 2 * time.Millisecond
	agentProbeMaxDelay   = 50 * time.Millisecond
	// agentDialTimeout bounds one connect attempt. The target is loopback, so a
	// connect that has not completed in this long is refused rather than slow.
	agentDialTimeout = 250 * time.Millisecond
)

// portQuarantine is how long a released port waits before it can be handed out
// again.
//
// The daemon's userland proxy keeps the host socket for a moment after the
// container it served is removed, so an immediately reused port fails the next
// create with "address already in use". That reads as pool exhaustion and is not:
// it is a reuse race, and a short quarantine removes it.
const portQuarantine = 15 * time.Second

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

	// networkMode is passed through to the daemon. It cannot be "host" here --
	// New refuses that, because the agent's port is fixed and every sandbox would
	// contend for it -- so this is "bridge" or a user-defined network.
	networkMode string

	// runtime is an OCI runtime name (gVisor, Kata). Empty uses the daemon's.
	runtime string

	// createSem bounds concurrent create+start pairs. The daemon serializes parts
	// of container setup in the kernel, so an unbounded burst converts latency
	// into timeouts rather than throughput.
	createSem chan struct{}

	ports *portPool

	// portTurn spreads creates across the pool's shards. A counter rather than a
	// hash of the sandbox id: round-robin is a perfect distribution, where a hash
	// is only an approximately uniform one and can collide two concurrent creates
	// onto one shard for no reason.
	portTurn atomic.Uint64

	// probe is a plain HTTP client for reaching the agent. It cannot be the Docker
	// client: that one dials a unix socket for every request, so it would send the
	// agent's probe to the daemon.
	probe *http.Client

	// pool holds pre-built, agent-ready sandboxes. Nil when disabled, and every
	// method on it is nil-safe so the create path needs no branch.
	pool *warmPool
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
	// WarmPool pre-builds agent-ready sandboxes. A zero Size disables it.
	WarmPool WarmPoolConfig
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
	// Host networking cannot work here, and the failure it produces is subtle
	// enough to be worth refusing outright.
	//
	// The agent's listening port is a compile-time default overridden only by a
	// -port flag, and bootstrap.sh execs the agent with a fixed argument list, so
	// nothing in this path can give two sandboxes different ports. Sharing the
	// host's network namespace therefore means every sandbox after the first finds
	// 44772 taken: the container starts, the agent dies, and the create fails at
	// the readiness wait with no indication that the cause was the network mode.
	//
	// A per-sandbox namespace with a published port is the shape that works, and it
	// is what the port pool exists for.
	if cfg.NetworkMode == "host" {
		return nil, fmt.Errorf(
			"opensandbox in direct mode cannot use network_mode \"host\": the agent's port is " +
				"fixed at 44772 and every sandbox would contend for it on the host. " +
				"Use \"bridge\" (or leave network_mode unset) so each sandbox gets its own " +
				"namespace and a published port")
	}
	runtime := &directRuntime{
		docker:      client,
		agent:       agent,
		nodeID:      cfg.NodeID,
		ownerID:     cfg.OwnerID,
		networkMode: cfg.NetworkMode,
		runtime:     cfg.Runtime,
		probe: &http.Client{Transport: &http.Transport{
			MaxIdleConns: 256, MaxIdleConnsPerHost: 256, IdleConnTimeout: 90 * time.Second,
		}},
	}
	if cfg.MaxCreateConcurrency > 0 {
		runtime.createSem = make(chan struct{}, cfg.MaxCreateConcurrency)
	}
	// Every sandbox is published on its own host port, so the pool is always built.
	//
	// It is sharded for the create concurrency, because that is how many creates
	// can be inside the allocator at once and therefore how many independent lanes
	// remove queueing rather than merely shortening it. An unbounded create
	// concurrency still gets a bounded number of shards: the point is to cover the
	// creates actually in flight, and the daemon cannot run unboundedly many.
	shards := cfg.MaxCreateConcurrency
	if shards <= 0 {
		shards = defaultPortShards
	}
	runtime.ports = newPortPool(cfg.PortMin, cfg.PortMax, shards)

	// The pool draws its ports from the same range cold creates do, so its budget
	// is validated against that range rather than in isolation.
	pool, err := newWarmPool(cfg.WarmPool, runtime.ports.span(),
		runtime.buildWarmEntry, runtime.destroyWarmEntry)
	if err != nil {
		return nil, err
	}
	runtime.pool = pool
	return runtime, nil
}

// defaultPortShards is used when create concurrency is unbounded. It is a
// compromise: enough lanes that a realistic burst stops colliding, few enough
// that each shard keeps a useful number of ports.
const defaultPortShards = 16

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

// create returns a usable sandbox, from the warm pool when one matches.
//
// The pool is tried first because a claim removes the whole cost of this
// function: the container build, the port publish, the start, and the agent
// readiness wait all happened before anyone was waiting. A miss falls through to
// the cold path, which is the correct outcome rather than a failure.
func (r *directRuntime) create(ctx context.Context, spec backend.Spec) (backend.Created, error) {
	if entry, claimed := r.pool.claim(ctx, spec); claimed {
		return backend.Created{
			Handle: backend.Handle{
				Backend:   "opensandbox",
				SandboxID: entry.containerID,
				NodeID:    r.nodeID,
			},
			Agent: backend.AgentEndpoint{Address: entry.address},
			// Reported so a benchmark can tell a claim from a build. Without it the
			// pool's effect is invisible in the only number that matters.
			WarmStart: true,
		}, nil
	}
	return r.createCold(ctx, spec)
}

// createCold composes and starts one sandbox, and returns where its agent answers.
func (r *directRuntime) createCold(ctx context.Context, spec backend.Spec) (backend.Created, error) {
	if spec.Source.Kind != "" && spec.Source.Kind != "image" {
		return backend.Created{}, fmt.Errorf(
			"a directly driven sandbox starts from an image, not source kind %q", spec.Source.Kind)
	}
	if spec.Source.Reference == "" {
		return backend.Created{}, fmt.Errorf("a directly driven sandbox needs an image reference")
	}

	sandboxID := newSandboxID()

	// The host port is reserved before the container is composed, because the
	// mapping is part of the create body.
	hostPort, err := r.ports.take(r.portTurn.Add(1))
	if err != nil {
		return backend.Created{}, err
	}
	defer func() {
		// Released on every failure path. A port leaked per failed create would
		// exhaust the range over a long run.
		if hostPort != 0 {
			r.ports.release(hostPort)
		}
	}()

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

	// A started container is not yet a usable sandbox: the agent binds its port a
	// moment after PID 1 runs. Waiting here keeps a startup failure inside the
	// create, where it reads as one, instead of surfacing as a failed first command.
	if err := r.awaitAgent(ctx, address); err != nil {
		removeCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_ = r.docker.removeContainer(removeCtx, containerID, true)
		return backend.Created{}, err
	}

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

// awaitAgent waits until the agent is serving, and returns as soon as it is.
//
// # Why this is not a fixed-interval poll
//
// A fixed interval pays its own period on every create. The agent binds in
// roughly ten milliseconds on a warm node, so a 25 ms tick spent most of its time
// sleeping past a sandbox that was already ready, and that sleep was added to
// every create in the fleet. Worse, each tick was a full HTTP request: a dial, a
// request line, headers, a response, and a body read, to answer a question that
// is really just "is the port bound yet".
//
// Two changes follow from that. The probe is a TCP connect, which is the actual
// event -- the kernel completes the handshake the moment the agent calls listen,
// and nothing cheaper can tell us sooner. And the interval starts far below the
// expected bind time and backs off, so a warm create returns on its first or
// second attempt while a cold one does not turn into a spin.
//
// The TCP connect is necessary but not sufficient: a bound port proves a listener
// exists, not that it serves. So one HTTP probe confirms, once, after the connect
// succeeds. That keeps the guarantee the old loop made -- the create does not
// return until the agent answers -- while paying for it once instead of per tick.
func (r *directRuntime) awaitAgent(ctx context.Context, address string) error {
	ctx, cancel := context.WithTimeout(ctx, agentReadyTimeout)
	defer cancel()

	target, err := hostPortOf(address)
	if err != nil {
		return err
	}

	delay := agentProbeFirstDelay
	for {
		if r.portIsBound(ctx, target) {
			// Bound. Confirm it serves, then the sandbox is usable.
			if r.agentAnswers(ctx, address) {
				return nil
			}
			// Listening but not yet answering is a narrow window between listen and
			// the handler being ready. Keep the short delay rather than backing off:
			// the remaining wait is sub-millisecond in practice.
			delay = agentProbeFirstDelay
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf(
				"the sandbox agent at %s did not answer within %s: the container started but "+
					"execd never bound its port", address, agentReadyTimeout)
		case <-time.After(delay):
		}
		if delay < agentProbeMaxDelay {
			delay *= 2
			if delay > agentProbeMaxDelay {
				delay = agentProbeMaxDelay
			}
		}
	}
}

// portIsBound reports whether anything is listening, by completing a handshake.
//
// A refused connection is the normal answer before the agent binds, so it is not
// an error here. The dial timeout is short because the target is loopback: a
// connect that has not completed in this long is not slow, it is refused.
func (r *directRuntime) portIsBound(ctx context.Context, hostPort string) bool {
	dialCtx, cancel := context.WithTimeout(ctx, agentDialTimeout)
	defer cancel()
	conn, err := (&net.Dialer{}).DialContext(dialCtx, "tcp", hostPort)
	if err != nil {
		return false
	}
	_ = conn.Close()
	return true
}

// hostPortOf reduces an agent address to the authority a dialler wants.
func hostPortOf(address string) (string, error) {
	parsed, err := url.Parse(address)
	if err != nil || parsed.Host == "" {
		return "", fmt.Errorf("the agent address %q is not a URL this backend can dial", address)
	}
	return parsed.Host, nil
}

func (r *directRuntime) agentAnswers(ctx context.Context, address string) bool {
	probeCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	request, err := http.NewRequestWithContext(probeCtx, http.MethodGet, address+"/ping", nil)
	if err != nil {
		return false
	}
	response, err := r.probe.Do(request)
	if err != nil {
		return false
	}
	defer response.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 256))
	return true
}

// createBody composes the container: the workload image, the agent mounted
// beside it, and bootstrap.sh as the entrypoint.
//
// bootstrap.sh is the contract. It starts the agent, then execs the workload's
// own command, so a sandbox image needs no awareness of OpenSandbox. What it
// reads from the environment is set here: which binary to run, whether the agent
// is the container's init, and what the workload command is.
//
// That last part is not optional. bootstrap.sh resolves "no command" to a bare
// non-interactive shell, which reads EOF from a closed stdin and exits at once --
// and because the agent runs as that shell's init, the whole container exits with
// it before a single command can arrive. A sandbox is a thing that waits, so the
// command it waits with is stated explicitly.
func (r *directRuntime) createBody(sandboxID string, spec backend.Spec, hostPort int) map[string]any {
	env := []string{
		// Where bootstrap.sh finds the agent.
		"EXECD=" + SandboxAgentDir + "/" + stagedExecd,
		// The agent becomes PID 1 and reaps orphans. Without this a long-lived
		// sandbox accumulates zombies, because a bare container's PID 1 is the
		// workload and it does not reap.
		"EXECD_INIT=true",
		// The keepalive. A sandbox serves commands over its agent rather than
		// running one workload to completion, so its PID 1 has to outlive the
		// create: this blocks forever and costs nothing.
		"BOOTSTRAP_CMD=" + keepaliveCommand,
	}
	for key, value := range spec.Env {
		env = append(env, key+"="+value)
	}
	// Sorted so one spec always produces one body: an idempotent retry must not
	// differ from its first attempt by map order alone.
	sort.Strings(env)

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
		// Restrict the sandbox to exactly its granted devices. The device request
		// above tells the runtime which to attach; this tells every CUDA process
		// inside which it may use. Without it a sandbox admitted for one device
		// enumerates all of them and can contend with a sibling's workload.
		env = append(env, "CUDA_VISIBLE_DEVICES="+strings.Join(gpuIDs(spec.AssignedGPUs), ","))
		sort.Strings(env)
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
	// The agent is reached through a published port, one per sandbox, because its
	// in-container port is the same for all of them.
	port := strconv.Itoa(agentPort) + "/tcp"
	body["ExposedPorts"] = map[string]any{port: map[string]any{}}
	hostConfig["PortBindings"] = map[string]any{
		port: []map[string]string{{"HostIp": "127.0.0.1", "HostPort": strconv.Itoa(hostPort)}},
	}
	return body
}

// release destroys one sandbox and returns its port to the pool.
func (r *directRuntime) release(ctx context.Context, handle backend.Handle) error {
	port := r.portOf(ctx, handle.SandboxID)
	if err := r.docker.removeContainer(ctx, handle.SandboxID, true); err != nil {
		return fmt.Errorf("releasing sandbox %s: %w", handle.SandboxID, err)
	}
	if port > 0 {
		// Returned after a delay rather than at once. The daemon's proxy holds the
		// listening socket briefly after the container is gone, so a port handed
		// straight to the next create is rejected with "address already in use" --
		// a failure that looks like exhaustion but is a reuse race. Quarantining
		// costs nothing: the pool is far larger than the live sandbox count.
		r.ports.releaseAfter(port, portQuarantine)
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
//
// # Why it is sharded
//
// One mutex over the whole range makes every concurrent create contend on the
// same lock. The critical section is short, so throughput survives, but the
// tail does not: under a burst of N creates the Nth waiter is queued behind
// N-1 predecessors, and that queueing is what shows up as a p95 several times
// the p50 while the median is unchanged. The work per create did not grow; the
// waiting did.
//
// Sharding removes the contention rather than shortening it. The range is cut
// into independent shards, each with its own lock, cursor, and reservation set,
// and a create is routed to one shard. Two creates collide only when they land
// on the same shard, so with S shards the expected contention falls by a factor
// of S. Nothing is shared between shards, so there is no coordination to pay
// for: this is a partition, not a finer-grained lock over shared state.
//
// A shard that is exhausted does not fail the create. It falls through to its
// neighbours in order, because a shard is an optimisation for the common case
// and must not become an artificial capacity limit -- a create refused while
// thousands of ports sat free in the next shard would be a scheduling defect
// introduced by a performance change.
type portPool struct {
	shards []*portShard
	min    int
	max    int
	// width is the ports per shard, kept so shardOf is arithmetic rather than a
	// scan. The last shard is wider by the range's remainder.
	width int
}

// portShard owns a disjoint slice of the range.
type portShard struct {
	mu    sync.Mutex
	next  int
	min   int
	max   int
	taken map[int]bool
	// free holds ports whose quarantine has elapsed, newest last. Taking from
	// here is a slice pop rather than a scan over `taken`, which is what keeps
	// the critical section constant-time once a node reaches steady state and
	// every port in the shard has been used at least once.
	free []int
}

// Default pool bounds, chosen to sit above the kernel's ephemeral range.
//
// Linux allocates outbound source ports from net.ipv4.ip_local_port_range,
// commonly 32768-60999. A pool inside that window collides with transient
// outbound connections: the port is free when it is handed out and taken by the
// time the daemon binds it, which surfaces as "address already in use" on a
// create that did nothing wrong. Staying above the range removes the class of
// failure rather than retrying through it.
const (
	defaultPortMin = 61000
	defaultPortMax = 65000

	// minPortsPerShard keeps a shard large enough to be worth having.
	//
	// Sharding a small range produces shards that exhaust immediately and send
	// every create through the fall-through path, which is slower than the single
	// lock it replaced. Below this many ports per shard the pool stays unsharded.
	minPortsPerShard = 64
)

// newPortPool builds a pool over the range, sharded for the stated concurrency.
//
// shards is normally the create concurrency: that is the number of creates that
// can be in the allocator at once, so it is the number of independent lanes that
// removes queueing entirely. It is clamped by the range size, because shards
// smaller than minPortsPerShard cost more than they save.
func newPortPool(minPort, maxPort, shards int) *portPool {
	if minPort <= 0 {
		minPort = defaultPortMin
	}
	if maxPort <= minPort {
		maxPort = defaultPortMax
	}
	span := maxPort - minPort + 1
	if shards <= 0 {
		shards = 1
	}
	if limit := span / minPortsPerShard; shards > limit {
		shards = limit
	}
	if shards < 1 {
		shards = 1
	}

	width := span / shards
	pool := &portPool{min: minPort, max: maxPort, width: width, shards: make([]*portShard, 0, shards)}
	// Contiguous, disjoint slices. The last shard absorbs the remainder so no
	// port in the configured range is left unreachable.
	for i := 0; i < shards; i++ {
		low := minPort + i*width
		high := low + width - 1
		if i == shards-1 {
			high = maxPort
		}
		pool.shards = append(pool.shards, &portShard{
			next: low, min: low, max: high, taken: map[int]bool{},
		})
	}
	return pool
}

// shardCount reports how many independent lanes the pool was built with.
func (p *portPool) shardCount() int { return len(p.shards) }

// span is how many ports the pool covers, which is what sizes a warm pool's
// standing claim against it.
func (p *portPool) span() int { return p.max - p.min + 1 }

// take reserves a port, preferring the caller's own shard.
//
// hint selects the shard. Callers pass something stable and well distributed for
// the request (a counter, or a hash of the sandbox id); the value only has to
// spread, not to be meaningful.
func (p *portPool) take(hint uint64) (int, error) {
	count := uint64(len(p.shards))
	start := int(hint % count)
	// The preferred shard first, then every other one. A create is refused only
	// when the whole configured range is reserved, which is the same condition the
	// unsharded pool refused on.
	for offset := 0; offset < len(p.shards); offset++ {
		if port, ok := p.shards[(start+offset)%len(p.shards)].take(); ok {
			return port, nil
		}
	}
	return 0, fmt.Errorf(
		"every port in %d-%d is reserved; this clears as sandboxes are released", p.min, p.max)
}

// take reserves a port from this shard, or reports that it has none.
func (s *portShard) take() (int, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	// A port that has already served and outlived its quarantine is the cheapest
	// one to hand out: no scan, no cursor arithmetic.
	if n := len(s.free); n > 0 {
		port := s.free[n-1]
		s.free = s.free[:n-1]
		s.taken[port] = true
		return port, true
	}
	span := s.max - s.min + 1
	for tried := 0; tried < span; tried++ {
		port := s.next
		s.next++
		if s.next > s.max {
			s.next = s.min
		}
		if !s.taken[port] {
			s.taken[port] = true
			return port, true
		}
	}
	return 0, false
}

// shardOf returns the shard that owns a port.
//
// Computed rather than searched. The shards are contiguous and equal-width by
// construction, so the owner follows from the offset by division -- a scan here
// would make every release O(shards) and spend more than the sharding saved,
// which is exactly what a first cut of this measured.
func (p *portPool) shardOf(port int) *portShard {
	if port < p.min || port > p.max {
		return nil
	}
	index := (port - p.min) / p.width
	// The last shard absorbs the range's remainder, so an offset past the final
	// boundary still belongs to it.
	if index >= len(p.shards) {
		index = len(p.shards) - 1
	}
	return p.shards[index]
}

// release returns a port for immediate reuse.
//
// Used on a failed create, where the daemon never bound the port, so there is no
// socket lingering and no quarantine to serve.
func (p *portPool) release(port int) {
	shard := p.shardOf(port)
	if shard == nil {
		return
	}
	shard.mu.Lock()
	defer shard.mu.Unlock()
	delete(shard.taken, port)
	shard.free = append(shard.free, port)
}

// releaseAfter returns a port to the pool once the delay has passed.
//
// The port stays reserved in the meantime, so no other create can take it while
// the kernel still holds its socket.
func (p *portPool) releaseAfter(port int, delay time.Duration) {
	time.AfterFunc(delay, func() { p.release(port) })
}

// reserved reports how many ports are held across every shard, for a test and
// for the metric hook.
func (p *portPool) reserved() int {
	total := 0
	for _, shard := range p.shards {
		shard.mu.Lock()
		total += len(shard.taken)
		shard.mu.Unlock()
	}
	return total
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
