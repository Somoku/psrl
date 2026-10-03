// Package cubesandbox runs sandboxes on CubeSandbox nodes.
//
// CubeSandbox speaks an E2B-compatible HTTP API: a template is the source (not
// an image tag), and /sandboxes/{id}/pause freezes the sandbox in place while
// /sandboxes/{id}/snapshots saves a named checkpoint.
//
// # Scheduling modes
//
// This adapter supports both scheduling modes defined by the backend contract.
//
// In provider mode the adapter sends every request to the Gateway URL and lets
// CubeSandbox's CubeMaster scheduler decide the node. This is the correct
// shape when the full CubeMaster feature set (overlaybd, warm pools, snapshot
// store, image pipeline) is needed and CubeMaster's placement decisions are
// acceptable.
//
// In direct mode the adapter talks to each Cubelet over gRPC, bypassing CubeMaster
// entirely, and p3b's Placement and Admission own the node decision. A Cubelet
// serves the whole node-level lifecycle through CubeboxMgr, and CubeMaster's only
// contribution above it is choosing which Cubelet to call. That path lives in
// direct.go.
//
// The two modes speak different protocols, which is why they are not one code
// path: CubeMaster serves an E2B-compatible HTTP API, and a Cubelet serves gRPC.
//
// Capability divergence: provider mode can claim warm_pool, template_build,
// volume, and egress_policy because those are CubeMaster-level features. Psrl
// mode bypasses CubeMaster, so those are absent. Neither mode declares a resume
// level: a CubeSandbox snapshot captures the filesystem and not memory.
//
// What this adapter does not reimplement: the microVM boot, the overlaybd
// layer system, and the snapshot store are CubeSandbox's, and they are the
// reason to run it at all.
package cubesandbox

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// Config is one CubeSandbox deployment.
type Config struct {
	// Gateway is the cluster entry point, used in provider mode.
	// E.g. https://cube.example.com
	Gateway string
	// Nodes are the per-node Cubelet addresses, used in direct mode. Each one
	// serves the full sandbox API on its own port, so placement here means
	// choosing which to call.
	Nodes []NodeAddress
	// APIKey is the X-Api-Key header value, used in provider mode. Empty skips
	// the header.
	APIKey string
	// OwnerID labels sandboxes this service created, so a sweep never touches one
	// it did not.
	OwnerID string
	// MaxCreateConcurrency bounds concurrent creates in direct mode. A Cubelet
	// boots a microVM per create, so an unbounded burst turns latency into
	// timeouts rather than throughput. Zero leaves it unbounded.
	MaxCreateConcurrency int
	RequestTimeout       time.Duration
	// CreateTimeout is separate because a cold start must boot a microVM and
	// may need to pull layers. Capping that at a coordination deadline would
	// read a slow template as a cluster fault.
	CreateTimeout time.Duration
}

// NodeAddress is one Cubelet node.
type NodeAddress struct {
	NodeID  string
	Address string
}

// Backend is CubeSandbox as a sandbox backend.
type Backend struct {
	cfg  Config
	mode backend.SchedulingMode
	http *http.Client

	// direct is set in direct mode and nil in provider mode. Its presence is what
	// every lifecycle method dispatches on.
	direct *directRuntime
}

// New returns a CubeSandbox backend in the given scheduling mode.
//
// Provider mode requires Gateway. Direct mode requires at least one node address.
func New(cfg Config, mode backend.SchedulingMode) (*Backend, error) {
	if !mode.Valid() {
		return nil, fmt.Errorf("cubesandbox scheduling mode %q is not direct or provider", mode)
	}
	if mode == backend.SchedulingProvider && cfg.Gateway == "" {
		return nil, fmt.Errorf("cubesandbox in provider mode needs a gateway address")
	}
	if mode == backend.SchedulingDirect && len(cfg.Nodes) == 0 {
		return nil, fmt.Errorf(
			"cubesandbox in direct mode needs its node addresses, because this service chooses the node itself")
	}
	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = 60 * time.Second
	}
	if cfg.CreateTimeout <= 0 {
		cfg.CreateTimeout = 5 * time.Minute
	}
	for _, node := range cfg.Nodes {
		if node.NodeID == "" || node.Address == "" {
			return nil, fmt.Errorf("a cubesandbox node needs both an id and an address")
		}
	}

	if mode == backend.SchedulingDirect {
		runtime, err := newDirectRuntime(
			context.Background(), cfg.Nodes, cfg.OwnerID,
			cfg.MaxCreateConcurrency, cfg.RequestTimeout, cfg.CreateTimeout,
		)
		if err != nil {
			return nil, err
		}
		return &Backend{cfg: cfg, mode: mode, direct: runtime}, nil
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

// Name is the registry key for this backend.
func (b *Backend) Name() string { return "cubesandbox" }

// Mode says who places sandboxes for this deployment.
func (b *Backend) Mode() backend.SchedulingMode { return b.mode }

// Capabilities declares what this deployment of CubeSandbox actually provides.
//
// Provider mode wraps CubeMaster, which has warm pools, template builds, volume
// attachment, and egress policy. Direct mode bypasses CubeMaster and talks
// directly to each Cubelet, so those CubeMaster-level features are absent. What
// a Cubelet provides directly: microVM create/exec/snapshot/pause/resume/delete.
//
// Neither mode declares a resume level, because a CubeSandbox snapshot captures
// the filesystem and not memory. Claiming full_state would let a conformance run
// read a filesystem restore as proof a live process survived a move.
func (b *Backend) Capabilities() backend.Capabilities {
	if b.mode == backend.SchedulingProvider {
		return backend.Capabilities{
			Features: []string{
				"freeze",
				"filesystem_snapshot",
				"restore",
				"warm_pool",
				"image_on_demand",
				"template_build",
				"volume",
				"egress_policy",
			},
			// No resume level: a snapshot captures the filesystem but not memory, so
			// a restore on another host starts from a checkpoint rather than a live
			// process. Claiming full_state would let a conformance run read a
			// filesystem restore as proof a live process survived the move.
			ResumeLevel: "",
			PauseModes:  []string{"freeze"},
		}
	}
	// Direct mode: a Cubelet, so every CubeMaster-level feature is absent. A warm
	// pool is a CubeMaster pool and a template build is a CubeMaster API; neither
	// is reachable from a Cubelet. Declaring either would let a spec requiring it
	// be admitted and then fail at create.
	//
	// Freeze is absent for the same reason and it is worth naming: CubeboxMgr has
	// no pause RPC. A pause is a CubeMaster operation above the node, so a
	// deployment needing one uses provider mode. Declaring freeze here would let
	// the reclaimer pause-on-idle and believe it had released compute it had not.
	return backend.Capabilities{
		Features: []string{
			"filesystem_snapshot",
			"restore",
			"image_on_demand",
		},
		// Same reasoning as provider mode, and additionally the snapshot is
		// node-local here unless the deployment configures a shared CoW backend.
		ResumeLevel: "",
		PauseModes:  nil,
	}
}

// Nodes returns the Cubelet addresses this service may place against.
//
// Only meaningful in direct mode; returns nil in provider mode because
// CubeMaster owns placement.
func (b *Backend) Nodes(context.Context) ([]string, error) {
	if b.mode == backend.SchedulingProvider {
		return nil, nil
	}
	ids := b.direct.nodeIDs()
	// Sorted so placement reads a stable order rather than Go's map iteration.
	sort.Strings(ids)
	return ids, nil
}

// Headroom is reported by this service's own node admission.
func (b *Backend) Headroom(context.Context, string) (map[string]backend.Resources, error) {
	return nil, nil
}

// RegisterBinding is a no-op for CubeSandbox.
//
// CubeMaster has no external RecordAssignment RPC comparable to AgentENV's,
// so there is nothing to register. In direct mode this service chose the node
// and holds the NodeID on the handle; no registration is needed for routing.
func (b *Backend) RegisterBinding(_ context.Context, _, _ string) error {
	return nil
}

type newSandbox struct {
	TemplateID string            `json:"templateID"`
	Timeout    int               `json:"timeout"`
	AutoPause  bool              `json:"autoPause"`
	EnvVars    map[string]string `json:"envVars,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
	Secure     *bool             `json:"secure,omitempty"`
}

type sandboxReply struct {
	SandboxID       string `json:"sandboxID"`
	ClientID        string `json:"clientID"`
	EnvdAccessToken string `json:"envdAccessToken"`
	Domain          string `json:"domain"`
}

// Create provisions one microVM sandbox.
//
// In direct mode it goes straight to the chosen Cubelet; in provider mode to the
// CubeMaster gateway. The request body is the same in both modes so the two
// shapes cannot drift in what they ask for.
//
// The source must be a template: CubeSandbox requires a templateID rather than
// a raw image reference. A "template" source kind carries the template ID
// directly; an "image" source maps to the same field.
func (b *Backend) Create(
	ctx context.Context, nodeID string, spec backend.Spec, _ string,
) (backend.Created, error) {
	if spec.Source.Kind != "" && spec.Source.Kind != "template" && spec.Source.Kind != "image" {
		return backend.Created{},
			fmt.Errorf("cubesandbox runs from a template or an image reference, not source kind %q", spec.Source.Kind)
	}
	if spec.Source.Reference == "" {
		return backend.Created{}, fmt.Errorf("cubesandbox needs a source reference (template id or image tag)")
	}

	if b.mode == backend.SchedulingDirect {
		created, err := b.direct.create(ctx, nodeID, spec)
		if err != nil {
			return backend.Created{}, err
		}
		created.Capabilities = b.Capabilities()
		return created, nil
	}

	body := newSandbox{
		TemplateID: spec.Source.Reference,
		// Disable the provider's own expiry: this service owns the lifetime
		// through its reclamation sweep. -1 means no timeout.
		Timeout:   -1,
		AutoPause: false,
		EnvVars:   spec.Env,
		Metadata:  spec.Metadata,
	}
	applyOptions(&body, spec)

	createCtx, cancel := context.WithTimeout(ctx, b.cfg.CreateTimeout)
	defer cancel()
	raw, err := b.call(createCtx, http.MethodPost, b.cfg.Gateway+"/sandboxes", body)
	if err != nil {
		return backend.Created{}, fmt.Errorf("cubesandbox create: %w", err)
	}
	var reply sandboxReply
	if err := json.Unmarshal(raw, &reply); err != nil {
		return backend.Created{}, fmt.Errorf("cubesandbox create reply: %w", err)
	}
	if reply.SandboxID == "" {
		return backend.Created{}, fmt.Errorf("cubesandbox create returned no sandbox id")
	}

	// Commands go to the sandbox's own envd agent, not through this service.
	agentAddress := ""
	if reply.Domain != "" {
		agentAddress = "https://" + reply.Domain
	}
	agent := backend.AgentEndpoint{Address: agentAddress}
	if reply.EnvdAccessToken != "" {
		agent.Headers = map[string]string{"X-Access-Token": reply.EnvdAccessToken}
	}
	return backend.Created{
		Handle:       backend.Handle{Backend: b.Name(), SandboxID: reply.SandboxID, NodeID: nodeID},
		Capabilities: b.Capabilities(),
		Agent:        agent,
	}, nil
}

// applyOptions folds this backend's own tuning into the request.
func applyOptions(body *newSandbox, spec backend.Spec) {
	options, named := spec.Options("cubesandbox")
	if !named {
		return
	}
	for key, value := range options {
		switch key {
		case "secure":
			secure := value == "true"
			body.Secure = &secure
		}
	}
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
		return fmt.Errorf("cubesandbox release %s: %w", handle.SandboxID, err)
	}
	return nil
}

type sandboxDetail struct {
	State string `json:"state"`
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
	var detail sandboxDetail
	if err := json.Unmarshal(raw, &detail); err != nil {
		return "unknown", nil
	}
	switch strings.ToLower(detail.State) {
	case "running":
		return "running", nil
	case "paused", "pausing":
		return "paused", nil
	default:
		return "unknown", nil
	}
}

// Pause freezes the sandbox in place.
//
// CubeSandbox's pause is an in-place freeze: the sandbox stays resident and
// compute is not released. Hibernate is not offered, so requesting one is an
// error rather than being served as a freeze: a caller that expected compute
// back would otherwise be misled.
func (b *Backend) Pause(ctx context.Context, handle backend.Handle, mode string) error {
	if mode != "" && mode != "freeze" {
		return fmt.Errorf(
			"cubesandbox keeps the sandbox resident on pause, so it freezes rather than %q", mode)
	}
	if b.mode == backend.SchedulingDirect {
		// CubeboxMgr exposes no pause RPC: a freeze is a CubeMaster operation
		// above the Cubelet. Saying so is better than silently doing nothing and
		// letting the reclaimer believe it released the sandbox's compute.
		return fmt.Errorf(
			"a directly driven cubesandbox cannot be paused: CubeboxMgr has no pause RPC, " +
				"so a freeze needs CubeMaster (provider mode)")
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err := b.call(ctx, http.MethodPost, b.cfg.Gateway+"/sandboxes/"+handle.SandboxID+"/pause", nil)
	return err
}

// Resume unfreezes a paused sandbox.
func (b *Backend) Resume(ctx context.Context, handle backend.Handle) error {
	if b.mode == backend.SchedulingDirect {
		// Symmetric with Pause: nothing was frozen, so there is nothing to thaw.
		return fmt.Errorf(
			"a directly driven cubesandbox is never paused, so there is nothing to resume")
	}
	// -1 keeps the sandbox's current expiry; this service owns the lifetime.
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err := b.call(ctx, http.MethodPost, b.cfg.Gateway+"/sandboxes/"+handle.SandboxID+"/resume",
		map[string]any{"timeout": -1})
	return err
}

type snapshotReply struct {
	SnapshotID string `json:"snapshotID"`
}

// Snapshot captures the sandbox's filesystem.
//
// CubeSandbox snapshots capture the writable layer; "full_state" is refused
// rather than served as a filesystem snapshot, because a caller that expected
// a live process to survive a resume would be misled. In direct mode the
// snapshot is node-local unless a shared store is configured externally.
func (b *Backend) Snapshot(ctx context.Context, handle backend.Handle, kind string) (string, error) {
	if b.mode == backend.SchedulingDirect {
		return b.direct.snapshot(ctx, handle, kind)
	}
	if kind != "" && kind != "filesystem" {
		return "", fmt.Errorf(
			"cubesandbox captures the filesystem, so it takes a filesystem snapshot rather than %q", kind)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.CreateTimeout)
	defer cancel()
	raw, err := b.call(ctx, http.MethodPost,
		b.cfg.Gateway+"/sandboxes/"+handle.SandboxID+"/snapshots", map[string]any{})
	if err != nil {
		return "", fmt.Errorf("cubesandbox snapshot: %w", err)
	}
	var reply snapshotReply
	if err := json.Unmarshal(raw, &reply); err != nil {
		return "", err
	}
	if reply.SnapshotID == "" {
		return "", fmt.Errorf("cubesandbox snapshot returned no id")
	}
	return reply.SnapshotID, nil
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

// Preflight confirms the deployment is reachable.
//
// In provider mode it probes the gateway. In direct mode it probes every
// configured Cubelet, because each one is a distinct server.
func (b *Backend) Preflight(ctx context.Context) error {
	if b.mode == backend.SchedulingDirect {
		return b.direct.preflight(ctx)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	if _, err := b.call(ctx, http.MethodGet, b.cfg.Gateway+"/health", nil); err != nil {
		return fmt.Errorf("the cubesandbox gateway at %s is not answering: %w", b.cfg.Gateway, err)
	}
	return nil
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
		req.Header.Set("X-Api-Key", b.cfg.APIKey)
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
	return fmt.Sprintf("cubesandbox API returned %d: %s", e.status, e.body)
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
