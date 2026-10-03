// Package agentenv runs sandboxes as Firecracker microVMs on AgentENV nodes.
//
// AgentENV ships a gateway and a scheduler above its node runtimes, and this
// adapter can use either shape. In provider mode it talks to the gateway and
// lets AgentENV place. In direct mode it talks to each node runtime directly,
// because AgentENV's own placement is round-robin over a boolean filter that
// discards the resource hint it is given, and it holds no reservation between
// choosing a node and the node charging the work -- so a burst of concurrent
// creates all read the same stale snapshot and land on one node.
//
// Direct mode is not a workaround. A node runtime serves the whole API on its
// own port, and the scheduler exposes RecordAssignment precisely so an outside
// placer can register where a sandbox landed and leave AgentENV's routing
// intact.
//
// What this adapter does not do is reimplement anything underneath: the
// microVM, the lazy image layers, the memory sharing, and the snapshot store
// are AgentENV's, and they are the reason to run it at all.
package agentenv

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

// Config is one AgentENV deployment.
type Config struct {
	// Gateway is the cluster entry point, used in provider mode.
	Gateway string
	// Nodes are the runtime addresses, used in direct mode. Each one serves the
	// whole sandbox API, so placement here is choosing which to call.
	Nodes []NodeAddress
	// Scheduler is where a binding is registered after this service places a
	// sandbox itself, so AgentENV's own routing and lookups keep working. Empty
	// skips registration, which is correct for a deployment running no scheduler.
	Scheduler string

	APIKey         string
	RequestTimeout time.Duration
	// CreateTimeout is separate because a cold start pulls and converts OCI
	// layers, which the project documents as tens of seconds. Holding it to a
	// coordination deadline would report a slow registry as a node fault.
	CreateTimeout time.Duration
}

// NodeAddress is one runtime node.
type NodeAddress struct {
	NodeID  string
	Address string
}

// Backend is AgentENV as a sandbox backend.
type Backend struct {
	cfg  Config
	mode backend.SchedulingMode
	http *http.Client

	mu    sync.RWMutex
	nodes map[string]string
}

// New returns an AgentENV backend in the given scheduling mode.
func New(cfg Config, mode backend.SchedulingMode) (*Backend, error) {
	if !mode.Valid() {
		return nil, fmt.Errorf("agentenv scheduling mode %q is not direct or provider", mode)
	}
	if mode == backend.SchedulingProvider && cfg.Gateway == "" {
		return nil, fmt.Errorf("agentenv in provider mode needs a gateway address")
	}
	if mode == backend.SchedulingDirect && len(cfg.Nodes) == 0 {
		return nil, fmt.Errorf(
			"agentenv in direct mode needs its node addresses, because this service chooses the node itself")
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
			return nil, fmt.Errorf("an agentenv node needs both an id and an address")
		}
		nodes[node.NodeID] = normalize(node.Address)
	}
	cfg.Gateway = normalize(cfg.Gateway)
	cfg.Scheduler = normalize(cfg.Scheduler)
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
func (b *Backend) Name() string { return "agentenv" }

// Mode says who places sandboxes for this deployment.
func (b *Backend) Mode() backend.SchedulingMode { return b.mode }

// Capabilities declares what AgentENV offers.
//
// A microVM pause writes memory out and releases compute, so it is a
// hibernation rather than a freeze, and the resume restores that memory: this
// is the backend that can carry a live process across a node, which is what
// makes it the one to pick for preemption-safe rollouts.
func (b *Backend) Capabilities() backend.Capabilities {
	return backend.Capabilities{
		Features: []string{
			"hibernate",
			"full_state_snapshot",
			"restore",
			"resume_anywhere",
			"native_fork",
			"image_on_demand",
			"template_build",
			"warm_pool",
			"isolation_runtime",
		},
		ResumeLevel: "full_state",
		PauseModes:  []string{"hibernate"},
	}
}

// Nodes returns the runtime addresses this service may place against.
func (b *Backend) Nodes(context.Context) ([]string, error) {
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
// from AgentENV.
//
// AgentENV's node metrics are a snapshot it refreshes on its own cadence, and
// admission here already tracks every grant exactly. Reading both would be two
// disagreeing accounts of one envelope, and the stale one would win whenever a
// burst outran the refresh.
func (b *Backend) Headroom(context.Context, string) (map[string]backend.Resources, error) {
	return nil, nil
}

// RegisterBinding tells AgentENV's scheduler where a sandbox landed.
//
// This is what keeps its own routing working while this service does the
// placing: a later lookup for the sandbox resolves to the right node instead of
// failing. A deployment running no scheduler configures none and skips it.
func (b *Backend) RegisterBinding(ctx context.Context, nodeID, sandboxID string) error {
	if b.cfg.Scheduler == "" || b.mode != backend.SchedulingDirect {
		return nil
	}
	b.mu.RLock()
	address := b.nodes[nodeID]
	b.mu.RUnlock()
	payload := map[string]any{
		"sandbox_id": sandboxID,
		"node":       map[string]any{"node_id": nodeID, "endpoint": address},
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	// A failed registration does not fail the create: the sandbox exists and is
	// reachable through this service either way, and AgentENV's own lookup is
	// the only thing degraded.
	_, _ = b.call(ctx, http.MethodPost, b.cfg.Scheduler+"/v1/assignments", payload)
	return nil
}

type coldSandbox struct {
	Image               string            `json:"image"`
	Timeout             int32             `json:"timeout,omitempty"`
	AutoPause           bool              `json:"autoPause"`
	Secure              bool              `json:"secure,omitempty"`
	AllowInternetAccess *bool             `json:"allowInternetAccess,omitempty"`
	Metadata            map[string]string `json:"metadata,omitempty"`
	EnvVars             map[string]string `json:"envVars,omitempty"`
	CPUCount            int32             `json:"cpuCount,omitempty"`
	MemoryMB            int64             `json:"memoryMB,omitempty"`
	DiskSizeMB          int64             `json:"diskSizeMB,omitempty"`
}

type sandboxReply struct {
	SandboxID  string `json:"sandboxID"`
	ClientID   string `json:"clientID"`
	TemplateID string `json:"templateID"`
	EnvdAccess struct {
		Token string `json:"token"`
	} `json:"envdAccessToken"`
}

// Create provisions one microVM.
//
// In direct mode it goes straight to the chosen node; in provider mode to the
// gateway, which asks AgentENV's scheduler. The request body is the same, so
// the two modes cannot drift in what they ask for.
func (b *Backend) Create(ctx context.Context, nodeID string, spec backend.Spec, callback string) (backend.Created, error) {
	if spec.Source.Kind != "" && spec.Source.Kind != "image" && spec.Source.Kind != "template" {
		return backend.Created{}, fmt.Errorf("agentenv runs an image or a template, not source kind %q", spec.Source.Kind)
	}
	target, err := b.target(nodeID)
	if err != nil {
		return backend.Created{}, err
	}

	body := coldSandbox{
		Image: spec.Source.Reference,
		// Its own expiry is left off: this service owns the lifetime through its
		// reclamation sweep, and a second clock would reclaim a sandbox the
		// service still believes it holds.
		AutoPause: false,
		Metadata:  spec.Metadata,
		EnvVars:   spec.Env,
	}
	if spec.Resources.CPUCount > 0 {
		body.CPUCount = int32(spec.Resources.CPUCount + 0.5)
	}
	if spec.Resources.MemoryMB > 0 {
		body.MemoryMB = spec.Resources.MemoryMB
	}
	if spec.Resources.DiskMB > 0 {
		body.DiskSizeMB = spec.Resources.DiskMB
	}
	applyOptions(&body, spec)

	createCtx, cancel := context.WithTimeout(ctx, b.cfg.CreateTimeout)
	defer cancel()
	raw, err := b.call(createCtx, http.MethodPost, target+"/sandboxes-cold", body)
	if err != nil {
		return backend.Created{}, fmt.Errorf("agentenv create: %w", err)
	}
	var reply sandboxReply
	if err := json.Unmarshal(raw, &reply); err != nil {
		return backend.Created{}, fmt.Errorf("agentenv create reply: %w", err)
	}
	if reply.SandboxID == "" {
		return backend.Created{}, fmt.Errorf("agentenv create returned no sandbox id")
	}
	if err := b.RegisterBinding(ctx, nodeID, reply.SandboxID); err != nil {
		return backend.Created{}, err
	}

	// The sandbox runs envd, so commands go straight to it and never through
	// this service. That is the whole reason a provider backend costs less per
	// command than a bare container.
	agent := backend.AgentEndpoint{Address: target + "/sandboxes/" + reply.SandboxID}
	if reply.EnvdAccess.Token != "" {
		agent.Headers = map[string]string{"X-Access-Token": reply.EnvdAccess.Token}
	}
	return backend.Created{
		Handle:       backend.Handle{Backend: b.Name(), SandboxID: reply.SandboxID, NodeID: nodeID},
		Capabilities: b.Capabilities(),
		Agent:        agent,
	}, nil
}

// applyOptions folds this backend's own tuning into the request.
//
// Namespaced on the spec, so a key naming another backend is not silently
// applied here, and an unknown key under this backend's name is an error rather
// than a no-op: a typo in a tuning knob is invisible until a run is slow.
func applyOptions(body *coldSandbox, spec backend.Spec) {
	options, named := spec.Options("agentenv")
	if !named {
		return
	}
	for key, value := range options {
		switch key {
		case "secure":
			body.Secure = value == "true"
		case "allow_internet_access":
			allow := value == "true"
			body.AllowInternetAccess = &allow
		}
	}
}

// Release destroys one microVM.
func (b *Backend) Release(ctx context.Context, handle backend.Handle) error {
	target, err := b.target(handle.NodeID)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err = b.call(ctx, http.MethodDelete, target+"/sandboxes/"+handle.SandboxID, nil)
	if err != nil && !isNotFound(err) {
		return fmt.Errorf("agentenv release %s: %w", handle.SandboxID, err)
	}
	// Already gone is the outcome the caller wanted.
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
	var reply struct {
		State string `json:"state"`
	}
	if err := json.Unmarshal(raw, &reply); err != nil {
		return "unknown", nil
	}
	switch strings.ToLower(reply.State) {
	case "running":
		return "running", nil
	case "paused":
		return "paused", nil
	case "stopped", "terminated":
		return "terminated", nil
	default:
		return "unknown", nil
	}
}

// Pause writes the microVM's memory out and releases its compute.
//
// A hibernation rather than a freeze, and the difference is reported rather
// than smoothed over: a caller that needed the sandbox to stay resident would
// otherwise be told it did.
func (b *Backend) Pause(ctx context.Context, handle backend.Handle, mode string) error {
	if mode != "" && mode != "hibernate" {
		return fmt.Errorf("agentenv writes memory out and releases compute, so it hibernates rather than %q", mode)
	}
	return b.post(ctx, handle, "/pause")
}

// Resume restores a hibernated microVM, on this node or another.
func (b *Backend) Resume(ctx context.Context, handle backend.Handle) error {
	return b.post(ctx, handle, "/resume")
}

// Snapshot captures memory and filesystem together.
//
// This is the capture that makes a resume elsewhere preserve a live process,
// which is the one thing the container backend cannot do.
func (b *Backend) Snapshot(ctx context.Context, handle backend.Handle, kind string) (string, error) {
	if kind != "" && kind != "full_state" {
		return "", fmt.Errorf(
			"agentenv captures memory with the filesystem, so it takes a full_state snapshot rather than %q", kind)
	}
	target, err := b.target(handle.NodeID)
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.CreateTimeout)
	defer cancel()
	raw, err := b.call(ctx, http.MethodPost, target+"/sandboxes/"+handle.SandboxID+"/snapshots", map[string]any{})
	if err != nil {
		return "", fmt.Errorf("agentenv snapshot: %w", err)
	}
	var reply struct {
		SnapshotID string `json:"snapshotID"`
		ID         string `json:"id"`
	}
	if err := json.Unmarshal(raw, &reply); err != nil {
		return "", err
	}
	if reply.SnapshotID != "" {
		return reply.SnapshotID, nil
	}
	if reply.ID == "" {
		return "", fmt.Errorf("agentenv snapshot returned no id")
	}
	return reply.ID, nil
}

// DeleteSnapshot removes a capture.
func (b *Backend) DeleteSnapshot(ctx context.Context, snapshotID string) error {
	target, err := b.target("")
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

// Preflight refuses a deployment this adapter cannot drive, at startup rather
// than inside the first episode.
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
		if _, err := b.call(ctx, http.MethodGet, address+"/health", nil); err != nil {
			return fmt.Errorf("agentenv %s at %s is not answering: %w", name, address, err)
		}
	}
	return nil
}

func (b *Backend) post(ctx context.Context, handle backend.Handle, path string) error {
	target, err := b.target(handle.NodeID)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.CreateTimeout)
	defer cancel()
	_, err = b.call(ctx, http.MethodPost, target+"/sandboxes/"+handle.SandboxID+path, map[string]any{})
	return err
}

// target returns the address a call goes to.
//
// In provider mode every call goes to the gateway, which resolves the sandbox
// itself. In direct mode a call goes to the node holding it, because this service
// chose that node and knows where the sandbox is.
func (b *Backend) target(nodeID string) (string, error) {
	if b.mode == backend.SchedulingProvider {
		return b.cfg.Gateway, nil
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if nodeID == "" {
		for _, address := range b.nodes {
			return address, nil
		}
		return "", fmt.Errorf("agentenv has no configured nodes")
	}
	address, known := b.nodes[nodeID]
	if !known {
		return "", fmt.Errorf("agentenv node %q is not configured", nodeID)
	}
	return address, nil
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
		req.Header.Set("X-API-Key", b.cfg.APIKey)
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
	return fmt.Sprintf("agentenv API returned %d: %s", e.status, e.body)
}

// Status returns the HTTP status, so a caller can tell backpressure from a
// fault: AgentENV answers 429 when it is busy, which clears, and 5xx when it is
// broken, which does not.
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
		address = "http://" + address
	}
	return strings.TrimRight(address, "/")
}
