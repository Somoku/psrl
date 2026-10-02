// Package cubesandbox runs sandboxes on a CubeSandbox cluster.
//
// CubeSandbox speaks an E2B-compatible HTTP API: a template is the source (not
// an image tag), and /sandboxes/{id}/pause freezes the sandbox in place while
// /sandboxes/{id}/snapshots saves a named checkpoint. Because the provider
// places sandboxes itself this adapter only runs in provider mode; the psrl
// scheduling path does not apply.
//
// What this adapter does not do is reimplement anything underneath: the
// microVM boot, the overlaybd layer system, warm pools, and the snapshot store
// are CubeSandbox's, and they are the reason to run it at all.
package cubesandbox

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

// Config is one CubeSandbox gateway.
type Config struct {
	// Gateway is the cluster entry point (e.g. https://cube.example.com).
	Gateway string
	// APIKey is the X-Api-Key header value. Empty skips the header, which is
	// correct for a cluster with no authentication layer.
	APIKey         string
	RequestTimeout time.Duration
	// CreateTimeout is separate because a cold start must boot a microVM and may
	// need to pull layers. The project's own latency targets allow tens of
	// seconds; capping that at a coordination deadline would read a slow template
	// as a cluster fault.
	CreateTimeout time.Duration
}

// Backend is CubeSandbox as a sandbox backend.
type Backend struct {
	cfg  Config
	http *http.Client
}

// New returns a CubeSandbox backend. CubeSandbox has its own scheduler, so
// provider mode is the only shape this backend runs in.
func New(cfg Config) (*Backend, error) {
	if cfg.Gateway == "" {
		return nil, fmt.Errorf("cubesandbox needs a gateway address")
	}
	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = 60 * time.Second
	}
	if cfg.CreateTimeout <= 0 {
		cfg.CreateTimeout = 5 * time.Minute
	}
	cfg.Gateway = normalize(cfg.Gateway)
	return &Backend{
		cfg: cfg,
		http: &http.Client{Transport: &http.Transport{
			MaxIdleConns: 128, MaxIdleConnsPerHost: 32, IdleConnTimeout: 90 * time.Second,
		}},
	}, nil
}

// Name is the registry key for this backend.
func (b *Backend) Name() string { return "cubesandbox" }

// Mode is always provider: CubeSandbox places sandboxes on its own cluster and
// this service does cross-backend quota only.
func (b *Backend) Mode() backend.SchedulingMode { return backend.SchedulingProvider }

// Capabilities declares what CubeSandbox offers.
//
// CubeSandbox's pause is in-place (freeze, not hibernate): the sandbox stays
// resident and compute is not released. It also supports filesystem snapshots
// for checkpoint/restore across nodes, template build, warm pools, and its own
// image delivery pipeline.
func (b *Backend) Capabilities() backend.Capabilities {
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
		// No resume level: a snapshot captures the filesystem but not memory, so a
		// restore on another host starts from a checkpoint rather than a live
		// process. Claiming full_state would let a conformance run read a filesystem
		// restore as proof a live process survived the move.
		ResumeLevel: "",
		PauseModes:  []string{"freeze"},
	}
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
// The source must be a template: CubeSandbox requires a templateID rather than
// an image reference, because templates encode the microVM parameters (size,
// layers, config) that a plain image tag does not. A "template" source kind
// carries the template ID directly; an "image" source maps to the same field,
// which CubeSandbox resolves to a built template.
func (b *Backend) Create(
	ctx context.Context, _ string, spec backend.Spec, _ string,
) (backend.Created, error) {
	if spec.Source.Kind != "" && spec.Source.Kind != "template" && spec.Source.Kind != "image" {
		return backend.Created{},
			fmt.Errorf("cubesandbox runs from a template or an image reference, not source kind %q", spec.Source.Kind)
	}
	if spec.Source.Reference == "" {
		return backend.Created{}, fmt.Errorf("cubesandbox needs a source reference (template id or image tag)")
	}
	body := newSandbox{
		TemplateID: spec.Source.Reference,
		// Disable the provider's own expiry: this service owns the lifetime through
		// its reclamation sweep. -1 means no timeout.
		Timeout:   -1,
		AutoPause: false,
		EnvVars:   spec.Env,
		Metadata:  spec.Metadata,
	}
	applyOptions(&body, spec)

	createCtx, cancel := context.WithTimeout(ctx, b.cfg.CreateTimeout)
	defer cancel()
	raw, err := b.call(createCtx, http.MethodPost, "/sandboxes", body)
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
	// The domain is the stable DNS name the cluster assigns; access requires the
	// envdAccessToken.
	agentAddress := ""
	if reply.Domain != "" {
		agentAddress = "https://" + reply.Domain
	}
	agent := backend.AgentEndpoint{Address: agentAddress}
	if reply.EnvdAccessToken != "" {
		agent.Headers = map[string]string{"X-Access-Token": reply.EnvdAccessToken}
	}
	return backend.Created{
		Handle:       backend.Handle{Backend: b.Name(), SandboxID: reply.SandboxID},
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
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err := b.call(ctx, http.MethodDelete, "/sandboxes/"+handle.SandboxID, nil)
	if err != nil && !isNotFound(err) {
		return fmt.Errorf("cubesandbox release %s: %w", handle.SandboxID, err)
	}
	// 404 means the sandbox was already destroyed, which is the outcome the
	// caller wanted.
	return nil
}

type sandboxDetail struct {
	State string `json:"state"`
}

// Status reports a sandbox's portable state.
func (b *Backend) Status(ctx context.Context, handle backend.Handle) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	raw, err := b.call(ctx, http.MethodGet, "/sandboxes/"+handle.SandboxID, nil)
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
// compute is not released. A hibernate (write-and-release) is not offered, so
// requesting one is an error rather than being served as a freeze: a caller
// that expected its compute to be returned would otherwise be misled.
func (b *Backend) Pause(ctx context.Context, handle backend.Handle, mode string) error {
	if mode != "" && mode != "freeze" {
		return fmt.Errorf(
			"cubesandbox keeps the sandbox resident on pause, so it freezes rather than %q", mode)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err := b.call(ctx, http.MethodPost, "/sandboxes/"+handle.SandboxID+"/pause", nil)
	return err
}

// Resume unfreezes a paused sandbox.
func (b *Backend) Resume(ctx context.Context, handle backend.Handle) error {
	// -1 keeps the sandbox's current expiry; this service owns the lifetime, so
	// resetting the TTL here would add a second clock.
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err := b.call(ctx, http.MethodPost, "/sandboxes/"+handle.SandboxID+"/resume",
		map[string]any{"timeout": -1})
	return err
}

type snapshotReply struct {
	SnapshotID string `json:"snapshotID"`
}

// Snapshot captures the sandbox's filesystem.
//
// CubeSandbox snapshots capture the filesystem, so the only kind this backend
// offers is "filesystem". A full_state claim would overstate what was saved:
// a restore recreates the filesystem state, not a live process.
func (b *Backend) Snapshot(ctx context.Context, handle backend.Handle, kind string) (string, error) {
	if kind != "" && kind != "filesystem" {
		return "", fmt.Errorf(
			"cubesandbox captures the filesystem, so it takes a filesystem snapshot rather than %q", kind)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.CreateTimeout)
	defer cancel()
	raw, err := b.call(ctx, http.MethodPost,
		"/sandboxes/"+handle.SandboxID+"/snapshots", map[string]any{})
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
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err := b.call(ctx, http.MethodDelete, "/snapshots/"+snapshotID, nil)
	if err != nil && !isNotFound(err) {
		return err
	}
	return nil
}

// Preflight confirms the cluster is reachable.
func (b *Backend) Preflight(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	if _, err := b.call(ctx, http.MethodGet, "/health", nil); err != nil {
		return fmt.Errorf("cubesandbox gateway at %s is not answering: %w", b.cfg.Gateway, err)
	}
	return nil
}

func (b *Backend) call(ctx context.Context, method, path string, body any) ([]byte, error) {
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
	req, err := http.NewRequestWithContext(ctx, method, b.cfg.Gateway+path, payload)
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
