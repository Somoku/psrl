// Package opensandbox runs sandboxes on an OpenSandbox cluster.
//
// OpenSandbox speaks a versioned REST API at /v1: POST /sandboxes creates from
// an image or a snapshot, DELETE /sandboxes/{id} destroys, GET /sandboxes/{id}
// reports state, and /pause+/resume are explicit endpoints. Snapshots land at
// /sandboxes/{id}/snapshots; the endpoint for in-sandbox commands is fetched at
// /sandboxes/{id}/endpoints/{port}.
//
// Resources are stated as Kubernetes-style quantity strings ("500m", "256Mi"),
// not raw integers, so this adapter converts from the portable spec's numeric
// form. Provider mode only — OpenSandbox schedules on its own cluster and this
// service does cross-backend quota only.
//
// What this adapter does not do is reimplement anything underneath: the fsb
// (fast sandbox) runtime, the block-level image delivery, warm pools, and the
// snapshot store are OpenSandbox's, and they are the reason to run it at all.
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
	// execdPort is the standard port for the execd service inside a sandbox.
	// Commands arrive here; the adapter fetches the public endpoint at create
	// time so the SDK can reach it without a second round trip.
	execdPort = 44772
)

// Config is one OpenSandbox cluster.
type Config struct {
	// Gateway is the cluster entry point, typically https://host/v1.
	Gateway string
	// APIKey is the OPEN-SANDBOX-API-KEY header value. Empty skips the header.
	APIKey         string
	RequestTimeout time.Duration
	// CreateTimeout is separate because a cold start may need to pull layers or
	// restore a snapshot. The project's latency targets allow minutes; capping at
	// a coordination deadline reads a slow cluster as a fault.
	CreateTimeout time.Duration
}

// Backend is OpenSandbox as a sandbox backend.
type Backend struct {
	cfg  Config
	http *http.Client
}

// New returns an OpenSandbox backend. OpenSandbox has its own scheduler, so
// provider mode is the only shape this backend runs in.
func New(cfg Config) (*Backend, error) {
	if cfg.Gateway == "" {
		return nil, fmt.Errorf("opensandbox needs a gateway address")
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
func (b *Backend) Name() string { return "opensandbox" }

// Mode is always provider: OpenSandbox places sandboxes on its own cluster.
func (b *Backend) Mode() backend.SchedulingMode { return backend.SchedulingProvider }

// Capabilities declares what OpenSandbox offers.
//
// OpenSandbox's pause is in-place (freeze, not hibernate): the sandbox stays
// resident and compute is not released. Filesystem snapshots are supported for
// checkpoint/restore across nodes, template build, warm pools, image block
// delivery, volume mounts, and egress policy.
func (b *Backend) Capabilities() backend.Capabilities {
	return backend.Capabilities{
		Features: []string{
			"freeze",
			"filesystem_snapshot",
			"restore",
			"warm_pool",
			"image_on_demand",
			"image_block_delivery",
			"template_build",
			"volume",
			"egress_policy",
			"credential_injection",
			"isolation_runtime",
		},
		// No resume level: a snapshot captures the filesystem but not memory.
		// Reporting full_state would let a conformance run read a checkpoint as
		// proof a live process survived a move.
		ResumeLevel: "",
		PauseModes:  []string{"freeze"},
	}
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
// An "image" source becomes an ImageSpec; a "template" source is passed as the
// image URI — OpenSandbox resolves template IDs through the same image field.
// Resources are converted from the portable numeric form (CPUCount, MemoryMB)
// to Kubernetes quantity strings, which is the only resource form OpenSandbox
// accepts.
func (b *Backend) Create(
	ctx context.Context, _ string, spec backend.Spec, _ string,
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

	req := createRequest{
		ResourceLimits: resourceLimits(spec),
		Env:            spec.Env,
		Metadata:       spec.Metadata,
	}
	// "template" kind signals that the reference is a templateID, not an OCI
	// image. OpenSandbox resolves both through its image field; the caller
	// distinguishes them by how the reference is formatted.
	req.Image = &imageSpec{URI: spec.Source.Reference}

	applyOptions(&req, spec)

	createCtx, cancel := context.WithTimeout(ctx, b.cfg.CreateTimeout)
	defer cancel()
	raw, err := b.call(createCtx, http.MethodPost, "/sandboxes", req)
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
	agentAddress, agentHeaders := b.fetchEndpoint(ctx, info.ID)

	return backend.Created{
		Handle:       backend.Handle{Backend: b.Name(), SandboxID: info.ID},
		Capabilities: b.Capabilities(),
		Agent: backend.AgentEndpoint{
			Address: agentAddress,
			Headers: agentHeaders,
		},
	}, nil
}

// fetchEndpoint retrieves the public execd endpoint for a sandbox.
//
// A failure is not fatal: if the endpoint is unavailable the SDK falls back to
// the control path for commands, which is slower but functional. The endpoint
// cache on the cluster side makes subsequent fetches cheap.
func (b *Backend) fetchEndpoint(ctx context.Context, sandboxID string) (string, map[string]string) {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	path := fmt.Sprintf("/sandboxes/%s/endpoints/%d", sandboxID, execdPort)
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
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err := b.call(ctx, http.MethodDelete, "/sandboxes/"+handle.SandboxID, nil)
	if err != nil && !isNotFound(err) {
		return fmt.Errorf("opensandbox release %s: %w", handle.SandboxID, err)
	}
	return nil
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
// OpenSandbox's pause is an in-place freeze: the sandbox stays resident. A
// hibernate (write-and-release) is not offered; requesting one is an error
// rather than being served as a freeze, because a caller that expected its
// compute to be returned would otherwise be misled.
func (b *Backend) Pause(ctx context.Context, handle backend.Handle, mode string) error {
	if mode != "" && mode != "freeze" {
		return fmt.Errorf(
			"opensandbox keeps the sandbox resident on pause, so it freezes rather than %q", mode)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err := b.call(ctx, http.MethodPost, "/sandboxes/"+handle.SandboxID+"/pause", nil)
	return err
}

// Resume unfreezes a paused sandbox.
func (b *Backend) Resume(ctx context.Context, handle backend.Handle) error {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	_, err := b.call(ctx, http.MethodPost, "/sandboxes/"+handle.SandboxID+"/resume", nil)
	return err
}

type snapshotInfo struct {
	ID string `json:"id"`
}

// Snapshot captures the sandbox's filesystem.
//
// OpenSandbox snapshots capture the filesystem; the only kind this backend
// offers is "filesystem". A full_state claim would overstate what was saved:
// a restore recreates the filesystem state, not a live process.
func (b *Backend) Snapshot(ctx context.Context, handle backend.Handle, kind string) (string, error) {
	if kind != "" && kind != "filesystem" {
		return "", fmt.Errorf(
			"opensandbox captures the filesystem, so it takes a filesystem snapshot rather than %q", kind)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.CreateTimeout)
	defer cancel()
	raw, err := b.call(ctx, http.MethodPost,
		"/sandboxes/"+handle.SandboxID+"/snapshots", map[string]any{})
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
	// A minimal list is cheaper than a dedicated health endpoint and proves
	// both auth and routing, not just TCP connectivity.
	if _, err := b.call(ctx, http.MethodGet, "/sandboxes?pageSize=1", nil); err != nil {
		return fmt.Errorf("opensandbox gateway at %s is not answering: %w", b.cfg.Gateway, err)
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
