// Package opensandbox runs sandboxes on OpenSandbox's runtime.
//
// OpenSandbox is two separable things, and this adapter treats them as such.
//
// Its data plane is a Go agent, execd, that runs inside every sandbox and serves
// commands, files, PTYs, code contexts, and metrics on port 44772. That agent is
// the reason to use OpenSandbox at all: it is more capable than driving a bare
// container, and the work in it is already done.
//
// Its control plane is a Python lifecycle server that creates containers, tracks
// them, allocates ports, and copies the agent in. Every one of those jobs is one
// this service already does, and does concurrently.
//
// # Scheduling modes
//
// In direct mode this service drives the container runtime itself and stages the
// agent into each sandbox. The Python lifecycle server is not deployed and is not
// in any path. This removes the create-path serialization that server imposes --
// one event loop, a thread per create, and a process-wide lock in its port
// allocator -- which is what a rollout step opening a hundred sandboxes at once
// actually runs into. The agent is unchanged, so the data plane is identical.
//
// In provider mode the adapter sends every request to a Gateway and lets
// OpenSandbox's own scheduler decide the node. This is the right shape for a
// Kubernetes-backed cluster, where the controller owns placement and provides
// warm pools, template builds, block-level image delivery, and an artifact store
// that makes a snapshot restorable on any node. There this service does
// cross-backend quota only.
//
// The two modes declare different capabilities, because they can serve different
// things. Direct mode adds what the agent serves (persistent sessions, background
// commands, code contexts, PTYs) and omits what only a cluster has (warm pools,
// template builds, block delivery, cross-node resume). Provider mode is the
// reverse. Neither claims more than it can do: a spec requiring an absent feature
// is refused at admission rather than failing inside an episode.
//
// One trap worth stating. A snapshot in direct mode is a container commit, which
// lands as a local image with no registry push, so it restores on the node that
// took it and nowhere else. resume_anywhere is therefore not declared, and the
// node is recorded on the handle so a restore routes back to it.
package opensandbox

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

const (
	// execdPort is where the agent listens inside a sandbox. In provider mode the
	// adapter resolves the endpoint for this port at create time so the SDK can
	// reach the agent without a second round trip.
	execdPort = 44772
)

// Config is one OpenSandbox deployment.
//
// The fields divide by mode. A field a mode does not use is absent from its
// configuration rather than ignored, so a deployment that sets a gateway on a
// directly driven backend sees it do nothing and can tell.
type Config struct {
	// Gateway is the cluster entry point, used in provider mode.
	// Typically https://host/v1.
	Gateway string
	// APIKey is the OPEN-SANDBOX-API-KEY header value, used in provider mode.
	// Empty skips the header.
	APIKey string

	// -- direct mode ----------------------------------------------------------

	// ExecdImage carries the agent binary, bootstrap.sh, and bubblewrap. It is
	// staged once at startup and then bind-mounted into every sandbox, so the
	// per-sandbox cost is a mount rather than a copy.
	ExecdImage string
	// Socket is the Docker daemon this service drives, as unix:// or tcp://.
	Socket     string
	APIVersion string
	// StageDir is where the agent artifacts are extracted on this host. Every
	// sandbox mounts it read-only.
	StageDir string
	// NodeID is this machine's identity, recorded on each handle.
	NodeID string
	// OwnerID labels containers this service created, so a sweep never touches
	// one it did not.
	OwnerID string
	// Runtime is an OCI runtime name (gVisor, Kata). Empty uses the daemon's own.
	Runtime string
	// NetworkMode is passed to the daemon. "host" removes the per-sandbox veth,
	// netns, and iptables work from the create path, and lets the agent be reached
	// on a host port without a mapping.
	NetworkMode string
	// MaxCreateConcurrency bounds concurrent container creates. The daemon
	// serializes parts of container setup in the kernel, so an unbounded burst
	// turns latency into timeouts rather than throughput. Zero leaves it unbounded.
	MaxCreateConcurrency int
	// PortMin and PortMax bound the host ports the agent is published on, when a
	// mapping is needed at all. Unused under host networking.
	PortMin int
	PortMax int

	// -- both modes -----------------------------------------------------------

	RequestTimeout time.Duration
	// CreateTimeout is separate because a cold start may need to pull layers or
	// restore a snapshot. The project's latency targets allow minutes; capping
	// at a coordination deadline reads a slow registry as a node fault.
	CreateTimeout time.Duration
}

// Backend is OpenSandbox as a sandbox backend.
type Backend struct {
	cfg  Config
	mode backend.SchedulingMode
	http *http.Client

	// direct is set in direct mode and nil in provider mode. Its presence is what
	// every lifecycle method dispatches on.
	direct *directRuntime
}

// New returns an OpenSandbox backend in the given scheduling mode.
//
// Direct mode dials the Docker daemon and stages the agent here rather than on
// the first create, so an unreachable daemon or an unusable agent image fails the
// deployment at startup instead of inside an episode.
func New(cfg Config, mode backend.SchedulingMode) (*Backend, error) {
	if !mode.Valid() {
		return nil, fmt.Errorf("opensandbox scheduling mode %q is not direct or provider", mode)
	}
	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = 60 * time.Second
	}
	if cfg.CreateTimeout <= 0 {
		cfg.CreateTimeout = 5 * time.Minute
	}

	if mode == backend.SchedulingProvider {
		if cfg.Gateway == "" {
			return nil, fmt.Errorf("opensandbox in provider mode needs a gateway address")
		}
		cfg.Gateway = normalize(cfg.Gateway)
		return &Backend{
			cfg:  cfg,
			mode: mode,
			http: &http.Client{Transport: &http.Transport{
				MaxIdleConns: 128, MaxIdleConnsPerHost: 32, IdleConnTimeout: 90 * time.Second,
			}},
		}, nil
	}

	// Direct mode: this service is the control plane, so it needs a runtime to
	// drive and an agent image to stage.
	if cfg.ExecdImage == "" {
		return nil, fmt.Errorf(
			"opensandbox in direct mode needs execd_image: the agent is staged from it into every sandbox")
	}
	if cfg.Socket == "" {
		return nil, fmt.Errorf(
			"opensandbox in direct mode needs socket: this service drives the container runtime itself")
	}
	if cfg.NodeID == "" {
		return nil, fmt.Errorf("opensandbox in direct mode needs a node id to record on each sandbox")
	}
	if cfg.StageDir == "" {
		cfg.StageDir = "/var/lib/sandboxd/opensandbox-agent"
	}
	if cfg.OwnerID == "" {
		cfg.OwnerID = "sandboxd-" + cfg.NodeID
	}

	runtime, err := newDirectRuntime(context.Background(), directConfig{
		Socket:               cfg.Socket,
		APIVersion:           cfg.APIVersion,
		NodeID:               cfg.NodeID,
		OwnerID:              cfg.OwnerID,
		Runtime:              cfg.Runtime,
		NetworkMode:          cfg.NetworkMode,
		MaxCreateConcurrency: cfg.MaxCreateConcurrency,
		ExecdImage:           cfg.ExecdImage,
		StageDir:             cfg.StageDir,
		PortMin:              cfg.PortMin,
		PortMax:              cfg.PortMax,
	})
	if err != nil {
		return nil, err
	}
	return &Backend{cfg: cfg, mode: mode, direct: runtime}, nil
}

// Name is the registry key for this backend.
func (b *Backend) Name() string { return "opensandbox" }

// Mode says who places sandboxes for this deployment.
func (b *Backend) Mode() backend.SchedulingMode { return b.mode }

// Capabilities declares what this deployment can actually serve.
//
// The two modes differ because the things underneath them differ, and the
// difference is not cosmetic. Direct mode runs the agent on a container runtime
// this service drives, so it can promise everything the agent serves and nothing
// that needs a cluster. Provider mode runs on OpenSandbox's own cluster, so it is
// the reverse: warm pools, template builds, block-level image delivery, and an
// artifact store that makes a snapshot restorable anywhere.
//
// Declaring a feature a deployment cannot serve is worse than declaring too few.
// A spec that requires an absent feature is refused at admission with a message
// naming what was missing; one that is admitted and then refused by the runtime
// fails inside an episode and costs the sample.
func (b *Backend) Capabilities() backend.Capabilities {
	if b.mode == backend.SchedulingDirect {
		// Read from the runtime rather than written as a constant: whether the
		// staged agent carries bubblewrap, and whether an isolation runtime is
		// configured, are properties of this deployment.
		return backend.Capabilities{
			Features: b.direct.features(),
			// A container commit captures the writable layer and not memory, so a
			// restore recreates the workspace and every process starts fresh. The
			// image is local, so the restore is too: resume_anywhere is absent from
			// the feature list for that reason, and the node travels on the handle.
			ResumeLevel: "filesystem",
			PauseModes:  []string{"freeze"},
		}
	}
	return backend.Capabilities{
		Features: []string{
			"freeze",
			"filesystem_snapshot",
			"restore",
			"resume_anywhere",
			"warm_pool",
			"image_on_demand",
			"image_block_delivery",
			"template_build",
			"volume",
			"egress_policy",
			"credential_injection",
			"isolation_runtime",
		},
		// Filesystem-level: the cluster's snapshot captures the root filesystem and
		// not memory. It restores on any node, which is what resume_anywhere says,
		// but the restored processes are new.
		ResumeLevel: "filesystem",
		PauseModes:  []string{"freeze"},
	}
}

// Nodes reports the nodes this service may place against.
//
// Direct mode drives one daemon, so this backend instance is one node -- the same
// shape the container backend has, and for the same reason: a Docker daemon is per
// machine. Provider mode returns nothing, because placement is the provider's.
func (b *Backend) Nodes(context.Context) ([]string, error) {
	if b.mode == backend.SchedulingProvider {
		return nil, nil
	}
	return []string{b.cfg.NodeID}, nil
}

// Headroom is reported by this node's own admission rather than read from the
// runtime.
//
// Admission here tracks every grant exactly and in real time, while a runtime's
// metrics are a snapshot on its own cadence. Reading both would be two
// disagreeing accounts of one envelope, and during a burst the stale one would win.
func (b *Backend) Headroom(context.Context, string) (map[string]backend.Resources, error) {
	return nil, nil
}

// RegisterBinding is a no-op.
//
// OpenSandbox exposes no assignment RPC for an outside placer to register with,
// and none is needed: in direct mode this service holds the node on the handle,
// and in provider mode the provider already knows where it put the sandbox.
func (b *Backend) RegisterBinding(context.Context, string, string) error {
	return nil
}

// Create provisions one sandbox.
//
// In direct mode the container is composed and started here, with the agent
// mounted beside the workload. In provider mode the request goes to the gateway
// and OpenSandbox's scheduler places it. Both return the resolved agent endpoint,
// so the SDK speaks to the sandbox directly from the first command either way.
func (b *Backend) Create(
	ctx context.Context, nodeID string, spec backend.Spec, _ string,
) (backend.Created, error) {
	if b.mode == backend.SchedulingDirect {
		created, err := b.direct.create(ctx, spec)
		if err != nil {
			return backend.Created{}, err
		}
		created.Capabilities = b.Capabilities()
		return created, nil
	}

	if spec.Source.Kind != "" &&
		spec.Source.Kind != "image" && spec.Source.Kind != "template" {
		return backend.Created{},
			fmt.Errorf("opensandbox runs from an image or a template, not source kind %q", spec.Source.Kind)
	}
	if spec.Source.Reference == "" {
		return backend.Created{},
			fmt.Errorf("opensandbox needs a source reference (image URI or template id)")
	}

	req := createRequest{
		ResourceLimits: resourceLimits(spec),
		Env:            spec.Env,
		Metadata:       spec.Metadata,
	}
	if spec.Source.Kind == "template" {
		req.SnapshotID = spec.Source.Reference
	} else {
		req.Image = &imageSpec{URI: spec.Source.Reference}
		// Required by the server for an image-based create, and it has to block:
		// see providerEntrypoint.
		req.Entrypoint = providerEntrypoint
	}
	applyOptions(&req, spec)

	createCtx, cancel := context.WithTimeout(ctx, b.cfg.CreateTimeout)
	defer cancel()
	raw, err := b.call(createCtx, http.MethodPost, b.cfg.Gateway+"/sandboxes", req)
	if err != nil {
		return backend.Created{}, fmt.Errorf("opensandbox create: %w", err)
	}
	var info sandboxInfo
	if err := json.Unmarshal(raw, &info); err != nil {
		return backend.Created{}, fmt.Errorf("opensandbox create reply: %w", err)
	}
	if info.ID == "" {
		return backend.Created{}, fmt.Errorf("opensandbox create returned no sandbox id")
	}

	// Resolved here so the SDK reaches the agent without a second round trip.
	agentAddress, agentHeaders := b.fetchEndpoint(ctx, b.cfg.Gateway, info.ID)

	return backend.Created{
		Handle:       backend.Handle{Backend: b.Name(), SandboxID: info.ID, NodeID: nodeID},
		Capabilities: b.Capabilities(),
		Agent: backend.AgentEndpoint{
			Address: agentAddress,
			Headers: agentHeaders,
		},
	}, nil
}

// Release destroys one sandbox.
func (b *Backend) Release(ctx context.Context, handle backend.Handle) error {
	if b.mode == backend.SchedulingDirect {
		return b.direct.release(ctx, handle)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err := b.call(ctx, http.MethodDelete, b.cfg.Gateway+"/sandboxes/"+handle.SandboxID, nil)
	if err != nil && !isNotFound(err) {
		return fmt.Errorf("opensandbox release %s: %w", handle.SandboxID, err)
	}
	return nil
}

// Status reports a sandbox's portable state.
func (b *Backend) Status(ctx context.Context, handle backend.Handle) (string, error) {
	if b.mode == backend.SchedulingDirect {
		return b.direct.status(ctx, handle)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	raw, err := b.call(ctx, http.MethodGet, b.cfg.Gateway+"/sandboxes/"+handle.SandboxID, nil)
	if err != nil {
		if isNotFound(err) {
			return "terminated", nil
		}
		return "", err
	}
	var info sandboxInfo
	if err := json.Unmarshal(raw, &info); err != nil {
		return "unknown", nil
	}
	// The server's own state set is Pending, Running, Pausing, Paused, Resuming,
	// Stopping, Terminated, Failed. Every one of them is mapped, because the
	// portable vocabulary has no "starting" and a state that fell through to
	// "unknown" would make the lifecycle treat a sandbox it is still being charged
	// for as one it cannot account for.
	switch info.Status.State {
	case "Running":
		return "running", nil
	case "Pending", "Resuming":
		// Live and charged, just not serving commands yet. Reported as running
		// because the portable status set has no transitional value, and the
		// alternatives are both worse: "unknown" reads as a lost sandbox, and
		// "terminated" would have the lifecycle release something that is starting.
		// Create already waits for readiness, so a caller does not normally observe
		// this state at all.
		return "running", nil
	case "Paused", "Pausing":
		return "paused", nil
	case "Terminated", "Stopping", "Failed":
		return "terminated", nil
	default:
		return "unknown", nil
	}
}

// Pause freezes the sandbox in place.
//
// The sandbox stays resident either way: a container freeze keeps its memory, and
// OpenSandbox's own pause does too. A hibernate would return the sandbox's
// compute, which neither does, so asking for one is an error rather than a silent
// downgrade -- a caller that expected its memory written out would otherwise be
// told it was.
func (b *Backend) Pause(ctx context.Context, handle backend.Handle, mode string) error {
	if b.mode == backend.SchedulingDirect {
		return b.direct.pause(ctx, handle, mode)
	}
	if mode != "" && mode != "freeze" {
		return fmt.Errorf(
			"opensandbox keeps the sandbox resident on pause, so it freezes rather than %q", mode)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err := b.call(ctx, http.MethodPost, b.cfg.Gateway+"/sandboxes/"+handle.SandboxID+"/pause", nil)
	return err
}

// Resume unfreezes a paused sandbox.
func (b *Backend) Resume(ctx context.Context, handle backend.Handle) error {
	if b.mode == backend.SchedulingDirect {
		return b.direct.resume(ctx, handle)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err := b.call(ctx, http.MethodPost, b.cfg.Gateway+"/sandboxes/"+handle.SandboxID+"/resume", nil)
	return err
}

// Snapshot captures the sandbox's filesystem.
//
// Neither mode captures memory, so "full_state" is refused in both. Serving it as
// a filesystem snapshot would let a conformance run read a workspace restore as
// proof that a live process survived a move.
func (b *Backend) Snapshot(ctx context.Context, handle backend.Handle, kind string) (string, error) {
	if b.mode == backend.SchedulingDirect {
		return b.direct.snapshot(ctx, handle, kind)
	}
	if kind != "" && kind != "filesystem" {
		return "", fmt.Errorf(
			"opensandbox captures the filesystem, so it takes a filesystem snapshot rather than %q", kind)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.CreateTimeout)
	defer cancel()
	raw, err := b.call(ctx, http.MethodPost,
		b.cfg.Gateway+"/sandboxes/"+handle.SandboxID+"/snapshots", map[string]any{})
	if err != nil {
		return "", fmt.Errorf("opensandbox snapshot: %w", err)
	}
	var info snapshotInfo
	if err := json.Unmarshal(raw, &info); err != nil {
		return "", err
	}
	if info.ID == "" {
		return "", fmt.Errorf("opensandbox snapshot returned no id")
	}
	return info.ID, nil
}

// DeleteSnapshot removes a previously captured snapshot.
func (b *Backend) DeleteSnapshot(ctx context.Context, snapshotID string) error {
	if b.mode == backend.SchedulingDirect {
		return b.direct.deleteSnapshot(ctx, snapshotID)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err := b.call(ctx, http.MethodDelete, b.cfg.Gateway+"/snapshots/"+snapshotID, nil)
	if err != nil && !isNotFound(err) {
		return err
	}
	return nil
}

// Preflight refuses a deployment this adapter cannot drive, at startup rather
// than inside the first episode.
//
// Direct mode has already done the expensive part in New: the daemon was dialled
// and the agent staged, and a failure there prevented construction. What is left
// is to confirm the daemon is still answering.
func (b *Backend) Preflight(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	if b.mode == backend.SchedulingDirect {
		if err := b.direct.docker.ping(ctx); err != nil {
			return fmt.Errorf("the Docker daemon at %s is not answering: %w", b.cfg.Socket, err)
		}
		return nil
	}
	// A minimal list proves auth and routing, not just that a port is open.
	if _, err := b.call(ctx, http.MethodGet, b.cfg.Gateway+"/sandboxes?pageSize=1", nil); err != nil {
		return fmt.Errorf("the opensandbox gateway at %s is not answering: %w", b.cfg.Gateway, err)
	}
	return nil
}

// StagedAgentDigest reports which agent image this backend staged.
//
// Exposed for operational reporting: a fleet running two agent versions is a
// configuration drift an operator wants to see named rather than inferred.
func (b *Backend) StagedAgentDigest() string {
	if b.mode != backend.SchedulingDirect {
		return ""
	}
	return b.direct.agent.Digest
}

// -- provider-mode plumbing ---------------------------------------------------

// imageSpec is the OpenSandbox image spec for a create request.
type imageSpec struct {
	URI string `json:"uri"`
}

type createRequest struct {
	Image *imageSpec `json:"image,omitempty"`
	// Entrypoint is what the sandbox's PID 1 runs. The server requires it
	// whenever an image is given ("Entrypoint is required when image is
	// provided"), and rejects the create with a 422 otherwise.
	//
	// It must block rather than exit. A sandbox exists to serve commands through
	// its agent, not to run one workload and finish, so an entrypoint that
	// returned would take the sandbox down with it the moment it was created.
	Entrypoint     []string          `json:"entrypoint,omitempty"`
	SnapshotID     string            `json:"snapshotId,omitempty"`
	ResourceLimits map[string]string `json:"resourceLimits"`
	Env            map[string]string `json:"env,omitempty"`
	Metadata       map[string]string `json:"metadata,omitempty"`
}

// providerEntrypoint is the blocking PID 1 for a provider-mode sandbox.
//
// `tail -f /dev/null` rather than `sleep infinity`: the latter is not present in
// a BusyBox image, and an entrypoint that is missing from the image fails the
// create for a reason that reads as a server fault.
var providerEntrypoint = []string{"tail", "-f", "/dev/null"}

type sandboxInfo struct {
	ID     string `json:"id"`
	Status struct {
		State string `json:"state"`
	} `json:"status"`
}

type snapshotInfo struct {
	ID string `json:"id"`
}

// fetchEndpoint retrieves the public agent endpoint for a sandbox.
//
// A failure is not fatal: the SDK falls back to the control path for commands,
// which is slower but functional. The cluster caches the resolution, so repeat
// fetches are cheap.
func (b *Backend) fetchEndpoint(ctx context.Context, target, sandboxID string) (string, map[string]string) {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	path := fmt.Sprintf("%s/sandboxes/%s/endpoints/%d", target, sandboxID, execdPort)
	raw, err := b.call(ctx, http.MethodGet, path, nil)
	if err != nil {
		return "", nil
	}
	var ep struct {
		Endpoint string            `json:"endpoint"`
		Headers  map[string]string `json:"headers"`
	}
	if err := json.Unmarshal(raw, &ep); err != nil || ep.Endpoint == "" {
		return "", nil
	}
	return ep.Endpoint, ep.Headers
}

// resourceLimits converts the portable numeric spec into OpenSandbox's
// Kubernetes-style quantity strings.
//
// The cluster API does not accept raw integers: CPU must be a millicore string
// ("500m", "2000m") and memory a binary-suffix string ("256Mi", "2Gi"). A spec
// with no stated CPU sends no entry rather than "0m", which the server rejects.
func resourceLimits(spec backend.Spec) map[string]string {
	limits := make(map[string]string, 2)
	if spec.Resources.CPUCount > 0 {
		millis := int64(spec.Resources.CPUCount * 1000)
		limits["cpu"] = fmt.Sprintf("%dm", millis)
	}
	if spec.Resources.MemoryMB > 0 {
		limits["memory"] = fmt.Sprintf("%dMi", spec.Resources.MemoryMB)
	}
	if spec.Resources.GPUCount > 0 {
		limits["gpu"] = fmt.Sprintf("%d", spec.Resources.GPUCount)
	}
	return limits
}

// applyOptions folds this backend's own tuning into the request.
func applyOptions(req *createRequest, spec backend.Spec) {
	options, named := spec.Options("opensandbox")
	if !named {
		return
	}
	for key, value := range options {
		switch key {
		case "cpu":
			req.ResourceLimits["cpu"] = value
		case "memory":
			req.ResourceLimits["memory"] = value
		}
	}
}

func (b *Backend) call(ctx context.Context, method, url string, body any) ([]byte, error) {
	var payload *bytes.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		payload = bytes.NewReader(encoded)
	} else {
		payload = bytes.NewReader(nil)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, payload)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if b.cfg.APIKey != "" {
		req.Header.Set("OPEN-SANDBOX-API-KEY", b.cfg.APIKey)
	}
	resp, err := b.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw := new(bytes.Buffer)
	if _, err := raw.ReadFrom(resp.Body); err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, &apiError{status: resp.StatusCode, body: strings.TrimSpace(raw.String())}
	}
	return raw.Bytes(), nil
}

type apiError struct {
	status int
	body   string
}

func (e *apiError) Error() string {
	return fmt.Sprintf("opensandbox API returned %d: %s", e.status, e.body)
}

// Status returns the HTTP status, so a caller can tell backpressure from a fault:
// a cluster answers 429 when it is busy, which clears, and 5xx when it is broken,
// which does not.
func (e *apiError) Status() int { return e.status }

func isNotFound(err error) bool {
	var typed *apiError
	for err != nil {
		if candidate, ok := err.(*apiError); ok {
			typed = candidate
			break
		}
		unwrapper, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = unwrapper.Unwrap()
	}
	return typed != nil && typed.status == http.StatusNotFound
}

func normalize(address string) string {
	if address == "" {
		return ""
	}
	if !strings.Contains(address, "://") {
		address = "https://" + address
	}
	return strings.TrimRight(address, "/")
}
