// Package opensandbox runs sandboxes on OpenSandbox nodes.
//
// OpenSandbox speaks a versioned REST API at /v1: POST /sandboxes creates from
// an image or a snapshot, DELETE /sandboxes/{id} destroys, GET /sandboxes/{id}
// reports state, and /pause+/resume are explicit endpoints. Snapshots land at
// /sandboxes/{id}/snapshots; the endpoint for in-sandbox commands is fetched at
// /sandboxes/{id}/endpoints/{port}.
//
// Resources are stated as Kubernetes-style quantity strings ("500m", "256Mi"),
// not raw integers, so this adapter converts from the portable spec's numeric
// form.
//
// # Scheduling modes
//
// This adapter supports both scheduling modes defined by the backend contract.
//
// In provider mode the adapter sends every request to the Gateway URL and lets
// OpenSandbox's own scheduler decide the node. This is correct for a k8s-backed
// deployment where the controller owns placement (FastSandbox top-K,
// BatchSandbox pool assignment) and p3b does cross-backend quota only.
//
// In psrl mode the adapter talks to each OpenSandbox node's docker-mode server
// directly, and p3b's Placement and Admission own the node decision. This is
// the right shape when:
//   - The deployment runs one docker-mode opensandbox-server per node.
//   - OpenSandbox's own scheduler has nothing to add (no k8s, no pool manager).
//   - A scheduler ablation must hold everything but the control plane fixed.
//
// The two shapes diverge in Capabilities: provider mode can declare warm pools,
// template builds, and image block delivery because those live on the k8s
// cluster. Docker mode exposes none of them (the server returns 400 or 501),
// so psrl mode declares only what is verified to work.
//
// One specific trap: snapshots in docker mode are node-local. The server calls
// container.commit() with no registry push, so a snapshot taken in psrl mode
// cannot be restored on a different node. RESUME_ANYWHERE is therefore not
// declared in psrl mode, and cross-node restore is not supported.
package opensandbox

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

const (
	// execdPort is the standard port for the execd service inside a sandbox.
	// Commands arrive here; the adapter fetches the public endpoint at create
	// time so the SDK can reach it without a second round trip.
	execdPort = 44772
)

// Config is one OpenSandbox deployment.
type Config struct {
	// Gateway is the cluster entry point, used in provider mode.
	// Typically https://host/v1.
	Gateway string
	// Nodes are the per-node docker-mode server addresses, used in psrl mode.
	// Each one serves the full sandbox API on its own port, so placement here
	// means choosing which to call. The key is a node ID; the value is the
	// base URL of that node's opensandbox-server.
	Nodes []NodeAddress
	// APIKey is the OPEN-SANDBOX-API-KEY header value. Empty skips the header.
	APIKey         string
	RequestTimeout time.Duration
	// CreateTimeout is separate because a cold start may need to pull layers or
	// restore a snapshot. The project's latency targets allow minutes; capping
	// at a coordination deadline reads a slow registry as a node fault.
	CreateTimeout time.Duration
}

// NodeAddress is one docker-mode opensandbox-server node.
type NodeAddress struct {
	NodeID  string
	Address string
}

// Backend is OpenSandbox as a sandbox backend.
type Backend struct {
	cfg  Config
	mode backend.SchedulingMode
	http *http.Client

	mu    sync.RWMutex
	nodes map[string]string // nodeID -> base URL
}

// New returns an OpenSandbox backend in the given scheduling mode.
//
// Provider mode requires Gateway. Psrl mode requires at least one node address.
// Configuring both and selecting psrl is valid: the gateway is unused but not
// an error, so a mixed-fleet deploy.json can share one config stanza.
func New(cfg Config, mode backend.SchedulingMode) (*Backend, error) {
	if !mode.Valid() {
		return nil, fmt.Errorf("opensandbox scheduling mode %q is not psrl or provider", mode)
	}
	if mode == backend.SchedulingProvider && cfg.Gateway == "" {
		return nil, fmt.Errorf("opensandbox in provider mode needs a gateway address")
	}
	if mode == backend.SchedulingPSRL && len(cfg.Nodes) == 0 {
		return nil, fmt.Errorf(
			"opensandbox in psrl mode needs its node addresses, because this service chooses the node itself")
	}
	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = 60 * time.Second
	}
	if cfg.CreateTimeout <= 0 {
		cfg.CreateTimeout = 5 * time.Minute
	}
	nodes := make(map[string]string, len(cfg.Nodes))
	for _, node := range cfg.Nodes {
		if node.NodeID == "" || node.Address == "" {
			return nil, fmt.Errorf("an opensandbox node needs both an id and an address")
		}
		nodes[node.NodeID] = normalize(node.Address)
	}
	cfg.Gateway = normalize(cfg.Gateway)
	return &Backend{
		cfg:   cfg,
		mode:  mode,
		nodes: nodes,
		http: &http.Client{Transport: &http.Transport{
			MaxIdleConns: 128, MaxIdleConnsPerHost: 32, IdleConnTimeout: 90 * time.Second,
		}},
	}, nil
}

// Name is the registry key for this backend.
func (b *Backend) Name() string { return "opensandbox" }

// Mode says who places sandboxes for this deployment.
func (b *Backend) Mode() backend.SchedulingMode { return b.mode }

// Capabilities declares what this deployment of OpenSandbox actually provides.
//
// The two modes have genuinely different capability sets because the underlying
// server behaviour differs. Provider mode wraps a k8s-backed cluster that has
// warm pools, image block delivery, template builds, and cross-node snapshots.
// Psrl mode wraps a docker-mode server that has none of those things: the
// relevant endpoints return 400 or 501. Declaring them anyway would let a
// caller's spec require a feature the server will refuse, and that refusal
// would arrive as a provider error at create time rather than a clean
// SandboxCapabilityError at admission.
//
// What docker mode does retain: egress sidecar, credential vault/proxy, and
// gVisor/Kata isolation runtimes. Those are confirmed present.
//
// Snapshot in docker mode captures the filesystem via container.commit() with
// no registry push, so a snapshot is node-local and RESUME_ANYWHERE is not
// declared. FILESYSTEM_SNAPSHOT and RESTORE are declared because they work —
// the constraint is that a restore must go to the same node, which the service
// enforces by recording the NodeID on the handle.
func (b *Backend) Capabilities() backend.Capabilities {
	if b.mode == backend.SchedulingProvider {
		// Provider mode: the k8s cluster provides the full feature set.
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
			// Filesystem-level snapshot only: the sandbox's root FS is captured,
			// but running processes and memory are not. A restore recreates the
			// workspace; every process starts fresh.
			ResumeLevel: "filesystem",
			PauseModes:  []string{"freeze"},
		}
	}
	// Psrl mode: docker-mode server, per-node deployment. The features above
	// that require k8s or FastSandbox are absent.
	return backend.Capabilities{
		Features: []string{
			"freeze",
			"filesystem_snapshot",
			"restore",
			"egress_policy",
			"credential_injection",
			"isolation_runtime",
		},
		// Snapshots are node-local in docker mode, so resume is also local-only.
		// Declaring filesystem here (not resume_anywhere) is correct: the
		// service will route a restore to the node that holds the snapshot.
		ResumeLevel: "filesystem",
		PauseModes:  []string{"freeze"},
	}
}

// Nodes returns the runtime addresses this service may place against.
//
// Only meaningful in psrl mode; returns an empty list in provider mode because
// placement is the provider's responsibility.
func (b *Backend) Nodes(context.Context) ([]string, error) {
	if b.mode == backend.SchedulingProvider {
		return nil, nil
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	ids := make([]string, 0, len(b.nodes))
	for id := range b.nodes {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids, nil
}

// Headroom is reported by this service's own node admission rather than read
// from OpenSandbox's metrics.
//
// OpenSandbox's docker server reports host metrics at GET /metrics, which
// describe the host rather than the sandboxes and refresh on the server's own
// cadence. Admission here tracks every grant exactly and in real time. Reading
// both would be two disagreeing accounts of one envelope, and the stale one
// would win during a burst.
func (b *Backend) Headroom(context.Context, string) (map[string]backend.Resources, error) {
	return nil, nil
}

// imageSpec is the OpenSandbox image spec for a create request.
type imageSpec struct {
	URI string `json:"uri"`
}

type createRequest struct {
	Image          *imageSpec        `json:"image,omitempty"`
	SnapshotID     string            `json:"snapshotId,omitempty"`
	ResourceLimits map[string]string `json:"resourceLimits"`
	Env            map[string]string `json:"env,omitempty"`
	Metadata       map[string]string `json:"metadata,omitempty"`
}

type sandboxInfo struct {
	ID     string `json:"id"`
	Status struct {
		State string `json:"state"`
	} `json:"status"`
}

// Create provisions one sandbox.
//
// In psrl mode it goes straight to the chosen node's docker server; in provider
// mode to the gateway, which asks OpenSandbox's own scheduler. The request body
// is the same in both modes, so the two shapes cannot drift in what they ask.
//
// An "image" source becomes an ImageSpec. Resources are converted from the
// portable numeric form (CPUCount, MemoryMB) to Kubernetes quantity strings,
// which is the only resource form OpenSandbox accepts.
func (b *Backend) Create(
	ctx context.Context, nodeID string, spec backend.Spec, _ string,
) (backend.Created, error) {
	if spec.Source.Kind != "" &&
		spec.Source.Kind != "image" && spec.Source.Kind != "template" {
		return backend.Created{},
			fmt.Errorf("opensandbox runs from an image or a template, not source kind %q", spec.Source.Kind)
	}
	if spec.Source.Reference == "" {
		return backend.Created{},
			fmt.Errorf("opensandbox needs a source reference (image URI or template id)")
	}

	target, err := b.target(nodeID)
	if err != nil {
		return backend.Created{}, err
	}

	req := createRequest{
		ResourceLimits: resourceLimits(spec),
		Env:            spec.Env,
		Metadata:       spec.Metadata,
	}
	req.Image = &imageSpec{URI: spec.Source.Reference}
	applyOptions(&req, spec)

	createCtx, cancel := context.WithTimeout(ctx, b.cfg.CreateTimeout)
	defer cancel()
	raw, err := b.call(createCtx, http.MethodPost, target+"/sandboxes", req)
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

	// Fetch the command endpoint so the SDK can reach execd without a second
	// call. The endpoint is per-port; execd always listens on execdPort.
	agentAddress, agentHeaders := b.fetchEndpoint(ctx, target, info.ID)

	return backend.Created{
		Handle:       backend.Handle{Backend: b.Name(), SandboxID: info.ID, NodeID: nodeID},
		Capabilities: b.Capabilities(),
		Agent: backend.AgentEndpoint{
			Address: agentAddress,
			Headers: agentHeaders,
		},
	}, nil
}

// fetchEndpoint retrieves the public execd endpoint for a sandbox.
//
// A failure is not fatal: the SDK falls back to the control path for commands,
// which is slower but functional. The endpoint cache on the server side makes
// subsequent fetches cheap.
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
// OpenSandbox does not accept raw integers: CPU must be a millicore string
// ("500m", "2000m") and memory a binary-suffix string ("256Mi", "2Gi").
// A spec with no stated CPU sends no entry rather than "0m", which the server
// would reject.
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

// Release destroys one sandbox.
func (b *Backend) Release(ctx context.Context, handle backend.Handle) error {
	target, err := b.target(handle.NodeID)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err = b.call(ctx, http.MethodDelete, target+"/sandboxes/"+handle.SandboxID, nil)
	if err != nil && !isNotFound(err) {
		return fmt.Errorf("opensandbox release %s: %w", handle.SandboxID, err)
	}
	return nil
}

// Status reports a sandbox's portable state.
func (b *Backend) Status(ctx context.Context, handle backend.Handle) (string, error) {
	target, err := b.target(handle.NodeID)
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	raw, err := b.call(ctx, http.MethodGet, target+"/sandboxes/"+handle.SandboxID, nil)
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
	switch info.Status.State {
	case "Running":
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
// OpenSandbox's pause keeps the sandbox resident (freeze, not hibernate). A
// sandbox that is frozen still holds its memory and compute on the node; a
// caller that needed compute to be released should use a backend that supports
// hibernate instead. Requesting hibernate is an error rather than a silent
// downgrade, because a caller expecting compute back would be misled.
func (b *Backend) Pause(ctx context.Context, handle backend.Handle, mode string) error {
	if mode != "" && mode != "freeze" {
		return fmt.Errorf(
			"opensandbox keeps the sandbox resident on pause, so it freezes rather than %q", mode)
	}
	target, err := b.target(handle.NodeID)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err = b.call(ctx, http.MethodPost, target+"/sandboxes/"+handle.SandboxID+"/pause", nil)
	return err
}

// Resume unfreezes a paused sandbox.
func (b *Backend) Resume(ctx context.Context, handle backend.Handle) error {
	target, err := b.target(handle.NodeID)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err = b.call(ctx, http.MethodPost, target+"/sandboxes/"+handle.SandboxID+"/resume", nil)
	return err
}

type snapshotInfo struct {
	ID string `json:"id"`
}

// Snapshot captures the sandbox's filesystem.
//
// OpenSandbox snapshots capture the filesystem only; "full_state" is refused
// rather than served as a filesystem snapshot, because a caller that expected
// its process to survive a resume would read a workspace restore as proof. In
// psrl mode the snapshot is node-local (docker mode uses container.commit()
// with no push), so the NodeID is preserved on the handle to route a restore
// back to the same node.
func (b *Backend) Snapshot(ctx context.Context, handle backend.Handle, kind string) (string, error) {
	if kind != "" && kind != "filesystem" {
		return "", fmt.Errorf(
			"opensandbox captures the filesystem, so it takes a filesystem snapshot rather than %q", kind)
	}
	target, err := b.target(handle.NodeID)
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.CreateTimeout)
	defer cancel()
	raw, err := b.call(ctx, http.MethodPost,
		target+"/sandboxes/"+handle.SandboxID+"/snapshots", map[string]any{})
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
	// Snapshot deletion is fleet-scoped in provider mode (the cluster owns the
	// artifact store) and node-scoped in psrl mode (docker commit, no registry).
	// In both cases the path is the same; we use anyTarget() which in provider
	// mode returns the gateway and in psrl mode returns any configured node.
	target, err := b.anyTarget()
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err = b.call(ctx, http.MethodDelete, target+"/snapshots/"+snapshotID, nil)
	if err != nil && !isNotFound(err) {
		return err
	}
	return nil
}

// Preflight confirms the deployment is reachable.
//
// In provider mode it probes the gateway. In psrl mode it probes every
// configured node, because each one is a distinct server whose absence would
// silently halve (or worse) the fleet.
func (b *Backend) Preflight(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	targets := map[string]string{}
	if b.mode == backend.SchedulingProvider {
		targets["gateway"] = b.cfg.Gateway
	} else {
		b.mu.RLock()
		for id, address := range b.nodes {
			targets[id] = address
		}
		b.mu.RUnlock()
	}
	for name, address := range targets {
		// A minimal list proves auth and routing, not just TCP connectivity.
		if _, err := b.call(ctx, http.MethodGet, address+"/health", nil); err != nil {
			return fmt.Errorf("opensandbox node %s at %s is not answering: %w", name, address, err)
		}
	}
	return nil
}

// target returns the URL a single-sandbox call goes to.
//
// In provider mode every call goes to the gateway, which resolves the sandbox
// itself. In psrl mode a call goes to the node that holds the sandbox, because
// this service chose that node and recorded it on the handle.
func (b *Backend) target(nodeID string) (string, error) {
	if b.mode == backend.SchedulingProvider {
		return b.cfg.Gateway, nil
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if nodeID == "" {
		// No node ID: the caller did not supply one, which happens for operations
		// (like deleteSnapshot) that are not tied to a specific sandbox. Return
		// any configured node as a best-effort target.
		for _, address := range b.nodes {
			return address, nil
		}
		return "", fmt.Errorf("opensandbox has no configured nodes")
	}
	address, known := b.nodes[nodeID]
	if !known {
		return "", fmt.Errorf("opensandbox node %q is not configured", nodeID)
	}
	return address, nil
}

// anyTarget returns a target URL for operations that are not tied to a node.
func (b *Backend) anyTarget() (string, error) {
	return b.target("")
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
