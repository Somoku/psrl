// Package dockerbackend runs sandboxes as containers on one node's daemon.
//
// It speaks the Engine API over the node's Unix socket rather than through a
// client library, for two reasons. The surface it needs is small -- create,
// start, inspect, remove, pause, commit -- and a vendored client would pull a
// dependency tree larger than the whole service for those six calls.
//
// The agent endpoint it reports is the daemon itself: a container has no process
// listening for commands, so this backend is the one case where the service's
// own node agent proxies command traffic. Every provider backend reports an
// in-sandbox agent instead and is not proxied.
package dockerbackend

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"path"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// The label every sandbox carries, so a sweep can find what this service owns
// without a local record. A container is the source of truth; a database that
// disagreed with the daemon would be worse than no database.
const (
	labelSandbox = "psrl.sandbox"
	labelOwner   = "psrl.owner"
	labelClass   = "psrl.resource_class"
	labelNode    = "psrl.node"
	// These carry what a restart cannot otherwise recover: how the sandbox is
	// executed, how long it may live, and which create produced it.
	labelExecMode    = "psrl.exec_mode"
	labelLifetime    = "psrl.lifetime_s"
	labelIdempotency = "psrl.idempotency_key"
	// Disk is charged by the node but not enforced by the Engine, so unlike
	// memory and CPU it cannot be read back from the container's limits.
	labelDisk = "psrl.disk_mb"
)

// DefaultPullTimeout is the image pull budget a node uses when its configuration
// states none. It is exported because a control plane that bounds a create over
// the node plane has to allow for the same pull before it gives up.
const DefaultPullTimeout = 10 * time.Minute

// Config is one node's Docker settings.
type Config struct {
	// Socket is the daemon's Unix socket.
	Socket string
	// APIVersion is pinned, because a daemon older than the service must fail
	// loudly at startup rather than on an endpoint that silently did not exist.
	APIVersion string
	NodeID     string
	OwnerID    string
	// Runtime selects a stronger isolation boundary when the node provides one.
	Runtime string
	// CgroupParent is the cgroup subtree every sandbox is created in, relative to the
	// daemon's own cgroup root. It is what makes the measured memory domain describe
	// sandboxes rather than everything the daemon's host happens to be doing, so an
	// empty value means the sampler has nothing narrower than the root to read.
	//
	// The cgroupfs driver places a container at <parent>/<id>; the systemd driver
	// treats the value as a slice name instead, which is why preflight refuses a
	// daemon that would interpret it differently.
	CgroupParent string
	// NetworkMode is the Docker network mode for new containers.
	// "host" removes per-container veth/netns creation and iptables rule churn,
	// which are the primary kernel-lock bottlenecks under concurrent creates.
	// Leave empty for the daemon default (bridge).
	NetworkMode string
	// MaxCreateConcurrency caps the number of concurrent /containers/create +
	// /start pairs in-flight to dockerd. It is the `max_create_concurrency`
	// deployment switch, and zero means unbounded.
	//
	// A daemon serializes parts of container setup in the kernel, so an unbounded
	// burst turns latency into timeouts rather than throughput. Size it from the
	// daemon rather than from the machine: a many-core node reports a GOMAXPROCS
	// far above what the create path can schedule at once, so leaving this unset on
	// a large node is what produced the create and exec timeouts it exists to stop.
	MaxCreateConcurrency int
	// RequestTimeout bounds one Engine call. A create that pulls an image can
	// legitimately exceed it, so the pull is a separate call with its own budget.
	RequestTimeout time.Duration
	PullTimeout    time.Duration
	// WarmPool, when enabled, keeps pre-created containers ready so a create
	// that matches the pool's spec claims one instantly rather than paying the
	// container-build cost. Zero-value Size disables the pool.
	WarmPool WarmPoolConfig
}

// Backend is the Docker runtime for one node.
type Backend struct {
	cfg  Config
	http *http.Client
	mode backend.SchedulingMode

	// createSem limits concurrent /containers/create + /start pairs in-flight to
	// dockerd. At high concurrency dockerd's internal worker pool saturates and all
	// requests queue on the same kernel locks (cgroup, netns, iptables), inflating
	// p95 far beyond p50. A bounded semaphore keeps the in-flight count at a level
	// dockerd can schedule without queueing, which flattens the latency distribution.
	// nil means unlimited (MaxCreateConcurrency == 0).
	createSem chan struct{}

	// shells holds the persistent shell sessions, one per sandbox. A sandbox
	// created with ExecMode "persistent" keeps a login shell open between calls;
	// one with ExecMode "one_shot" (or empty) creates a fresh process per call.
	shells *shellSessions

	// pool holds pre-created containers that are claimed on Create when the spec
	// matches. nil when the pool is disabled or not configured.
	pool *warmPool

	mu    sync.Mutex
	known map[string]string // sandbox id -> container id
}

// New returns a Docker backend bound to one node's daemon.
func New(cfg Config, mode backend.SchedulingMode) (*Backend, error) {
	if cfg.Socket == "" {
		cfg.Socket = "/var/run/docker.sock"
	}
	if cfg.APIVersion == "" {
		cfg.APIVersion = "v1.40"
	}
	if !strings.HasPrefix(cfg.APIVersion, "v") {
		cfg.APIVersion = "v" + cfg.APIVersion
	}
	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = 30 * time.Second
	}
	if cfg.PullTimeout <= 0 {
		cfg.PullTimeout = DefaultPullTimeout
	}
	if !mode.Valid() {
		return nil, fmt.Errorf("docker backend scheduling mode %q is not direct or provider", mode)
	}
	if cfg.MaxCreateConcurrency < 0 {
		return nil, fmt.Errorf(
			"docker backend max create concurrency %d is negative, and zero is what means unbounded",
			cfg.MaxCreateConcurrency)
	}
	var createSem chan struct{}
	if cfg.MaxCreateConcurrency > 0 {
		createSem = make(chan struct{}, cfg.MaxCreateConcurrency)
	}
	socket := cfg.Socket
	b := &Backend{
		cfg:       cfg,
		mode:      mode,
		createSem: createSem,
		http: &http.Client{
			Transport: &http.Transport{
				DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
					return (&net.Dialer{}).DialContext(ctx, "unix", socket)
				},
				MaxIdleConns:        64,
				MaxIdleConnsPerHost: 64,
			},
		},
		known:  map[string]string{},
		shells: newShellSessions(),
	}
	// Built after the backend exists, because the pool drives this backend's own
	// create and remove: a pool with its own transport would be a second way to
	// reach the daemon and could disagree with this one about concurrency.
	pool, err := newWarmPool(cfg.WarmPool,
		b.createPoolEntry,
		b.remove,
	)
	if err != nil {
		return nil, fmt.Errorf("docker warm pool: %w", err)
	}
	b.pool = pool
	return b, nil
}

// Name is the registry key for this backend.
func (b *Backend) Name() string { return "docker" }

// Mode says who places sandboxes for this backend.
func (b *Backend) Mode() backend.SchedulingMode { return b.mode }

// Capabilities declares what this node can do, with each conditional feature
// earned rather than assumed.
//
// A capability is declared only where the node can honour it: a stronger
// isolation boundary needs a configured runtime, and claiming one the node lacks
// would admit a spec and then run it with less isolation than it asked for.
func (b *Backend) Capabilities() backend.Capabilities {
	features := []string{"freeze", "host_mount", "filesystem_snapshot", "restore", "credential_injection"}
	if b.cfg.Runtime != "" {
		features = append(features, "isolation_runtime")
	}
	sort.Strings(features)
	// No resume level: a commit captures the writable layer and not memory, so a
	// restore elsewhere needs a published snapshot, which this backend does not do
	// on its own. Claiming full state would let a conformance run read a workspace
	// restore as proof a harness survived a move.
	return backend.Capabilities{Features: features, PauseModes: []string{"freeze"}}
}

// Nodes reports this backend's single node. A Docker daemon is per machine, so
// one backend instance is one node.
func (b *Backend) Nodes(context.Context) ([]string, error) {
	return []string{b.cfg.NodeID}, nil
}

// Headroom is reported by the node's own admission rather than guessed here, so
// this returns nothing and lets the node agent fill it in. Returning a daemon
// statistic instead would be a second, disagreeing account of the same envelope.
func (b *Backend) Headroom(context.Context, string) (map[string]backend.Resources, error) {
	return nil, nil
}

// RegisterBinding is a no-op: a container carries its node in a label, so there
// is no routing table to keep in step.
func (b *Backend) RegisterBinding(context.Context, string, string) error { return nil }

// Create starts one container and waits until it can run a command.
//
// When the warm pool holds a matching entry the container is claimed rather
// than built, which removes the per-container namespace and cgroup overhead
// from the create latency entirely. A spec that does not match the pool spec
// (different image, larger footprint, GPU request) falls through to a cold
// create so the pool never silently degrades a caller's resource constraints.
func (b *Backend) Create(ctx context.Context, nodeID string, spec backend.Spec, callback string) (backend.Created, error) {
	if spec.Source.Kind != "" && spec.Source.Kind != "image" {
		return backend.Created{}, fmt.Errorf("docker runs an image, so source kind %q cannot be served", spec.Source.Kind)
	}
	// Try the warm pool first. A cache hit skips image pull, container creation,
	// and the start call — the three steps whose latency scales with concurrency.
	if containerID, ok := b.pool.claim(ctx, spec); ok {
		return b.adoptWarm(containerID, spec, nodeID), nil
	}
	if err := b.ensureImage(ctx, spec.Source.Reference); err != nil {
		return backend.Created{}, err
	}
	// Acquire the create semaphore before touching the daemon. This bounds the
	// number of concurrent /containers/create + /start pairs in-flight, preventing
	// dockerd's internal queues from growing unboundedly and inflating p95 latency.
	if b.createSem != nil {
		select {
		case b.createSem <- struct{}{}:
			defer func() { <-b.createSem }()
		case <-ctx.Done():
			return backend.Created{}, ctx.Err()
		}
	}
	body := b.createBody(spec, nodeID)
	created, err := b.createNamed(ctx, body)
	if err != nil {
		return backend.Created{}, fmt.Errorf("docker create: %w", err)
	}
	if err := b.call(ctx, http.MethodPost, "/containers/"+created.ID+"/start", nil, nil); err != nil {
		// The container exists and holds resources, so it is removed here rather
		// than left for a sweep: the caller never saw a handle it could release.
		_ = b.remove(ctx, created.ID)
		return backend.Created{}, fmt.Errorf("docker start: %w", err)
	}
	b.mu.Lock()
	b.known[created.ID] = created.ID
	b.mu.Unlock()
	return backend.Created{
		Handle:       backend.Handle{Backend: b.Name(), SandboxID: created.ID, NodeID: nodeID},
		Capabilities: b.Capabilities(),
		// A container has no in-sandbox agent, so the node agent proxies commands
		// for it. Every provider backend reports a real address here instead.
		Agent: backend.AgentEndpoint{Address: ""},
	}, nil
}

// The prefixes every container this service creates carries.
//
// A warm-pool entry has to be distinguishable from a claimed sandbox, and the marker
// has to be changeable when the entry is adopted: the label set is written at create
// and immutable afterwards, so a label that said "inventory" would still say it about a
// live sandbox. That is the state that made a restarting node refuse to serve, and the
// state that would make it destroy a live sandbox if only the refusal were removed. A
// name is the one attribute the daemon lets us change after creation, so inventory is
// marked here and cleared by a rename at claim.
const (
	warmNamePrefix    = "psrl-warm-"
	sandboxNamePrefix = "psrl-sbx-"
)

// createdContainer is the reply to a create.
type createdContainer struct {
	ID       string   `json:"Id"`
	Warnings []string `json:"Warnings"`
}

// newContainerName returns a prefixed name with a random suffix.
func newContainerName(prefix string) string {
	var suffix [8]byte
	if _, err := rand.Read(suffix[:]); err != nil {
		// crypto/rand does not fail on Linux in practice. A timestamp keeps the
		// function total, and a duplicate is retried by the caller as a conflict.
		return prefix + strconv.FormatInt(time.Now().UnixNano(), 16)
	}
	return prefix + hex.EncodeToString(suffix[:])
}

// createNamed creates a container under a name this service chose.
//
// The name is not cosmetic: it is how pool inventory is told from a claimed sandbox,
// and how an operator finds the container behind a sandbox id. A conflict means the
// random suffix collided, which is cheap to retry and must not surface as a create
// failure.
func (b *Backend) createNamed(ctx context.Context, body map[string]any) (createdContainer, error) {
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		var created createdContainer
		err := b.call(ctx, http.MethodPost,
			"/containers/create?name="+url.QueryEscape(newContainerName(sandboxNamePrefix)), body, &created)
		if err == nil {
			return created, nil
		}
		lastErr = err
		var apiErr *apiError
		if !asAPIError(err, &apiErr) || apiErr.status != http.StatusConflict {
			return createdContainer{}, err
		}
	}
	return createdContainer{}, lastErr
}

func (b *Backend) createBody(spec backend.Spec, nodeID string) map[string]any {
	env := make([]string, 0, len(spec.Env))
	for key, value := range spec.Env {
		env = append(env, key+"="+value)
	}
	sort.Strings(env)

	labels := map[string]string{
		labelSandbox: "true",
		labelOwner:   b.cfg.OwnerID,
		labelClass:   spec.ResourceClass,
		labelNode:    nodeID,
	}
	if spec.ExecMode != "" {
		labels[labelExecMode] = spec.ExecMode
	}
	if spec.Lifetime > 0 {
		labels[labelLifetime] = strconv.FormatInt(int64(spec.Lifetime/time.Second), 10)
	}
	if spec.Resources.DiskMB > 0 {
		labels[labelDisk] = strconv.FormatInt(spec.Resources.DiskMB, 10)
	}
	if spec.IdempotencyKey != "" {
		labels[labelIdempotency] = spec.IdempotencyKey
	}
	for key, value := range spec.Metadata {
		labels[key] = value
	}

	hostConfig := map[string]any{
		// A sandbox is destroyed by this service, not by the daemon, so its exit
		// must leave a record a post mortem can read.
		"AutoRemove": false,
	}
	if b.cfg.NetworkMode != "" {
		// "host" removes per-container veth/netns creation and iptables churn,
		// eliminating the main kernel-lock bottleneck under concurrent creates.
		hostConfig["NetworkMode"] = b.cfg.NetworkMode
	}
	if spec.Resources.MemoryMB > 0 {
		hostConfig["Memory"] = spec.Resources.MemoryMB * 1024 * 1024
	}
	if spec.Resources.CPUCount > 0 {
		// Quota against the default 100ms period, which is how the Engine expresses
		// a fractional core.
		hostConfig["CpuPeriod"] = 100000
		hostConfig["CpuQuota"] = int64(spec.Resources.CPUCount * 100000)
	}
	if b.cfg.Runtime != "" {
		hostConfig["Runtime"] = b.cfg.Runtime
	}
	if b.cfg.CgroupParent != "" {
		hostConfig["CgroupParent"] = b.cfg.CgroupParent
	}
	if len(spec.AssignedGPUs) > 0 {
		hostConfig["DeviceRequests"] = []map[string]any{{
			"Driver":       "nvidia",
			"DeviceIDs":    deviceIDs(spec.AssignedGPUs),
			"Capabilities": [][]string{{"gpu"}},
		}}
		// Restrict the container to exactly its granted devices. Without this the
		// NVIDIA runtime exposes all devices and two concurrent sandboxes can
		// interfere with each other's workloads even though admission partitioned
		// the indices correctly.
		env = append(env, "CUDA_VISIBLE_DEVICES="+cudaVisibleDevices(spec.AssignedGPUs))
		sort.Strings(env)
	}

	body := map[string]any{
		"Image":      spec.Source.Reference,
		"Env":        env,
		"Labels":     labels,
		"HostConfig": hostConfig,
		// A sandbox waits for commands rather than running a workload of its own, so
		// it needs an entry process that does not exit.
		"Cmd": []string{"sleep", "infinity"},
	}
	if spec.Workdir != "" {
		body["WorkingDir"] = spec.Workdir
	}
	return body
}

func deviceIDs(indices []int32) []string {
	out := make([]string, len(indices))
	for i, index := range indices {
		out[i] = fmt.Sprint(index)
	}
	return out
}

// cudaVisibleDevices renders the CUDA_VISIBLE_DEVICES value for the granted
// indices. This restricts what the NVIDIA runtime exposes inside the container,
// so a sandbox that was admitted for device 2 cannot see device 0 or 1.
func cudaVisibleDevices(indices []int32) string {
	parts := make([]string, len(indices))
	for i, index := range indices {
		parts[i] = fmt.Sprint(index)
	}
	return strings.Join(parts, ",")
}

// ensureImage pulls an image the node does not have.
//
// The pull has its own budget, because its duration is the registry and the
// image rather than a round trip, and holding a create's deadline against it
// would report a slow registry as a node fault.
func (b *Backend) ensureImage(ctx context.Context, reference string) error {
	if reference == "" {
		return fmt.Errorf("docker needs an image reference")
	}
	inspectCtx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	if err := b.call(inspectCtx, http.MethodGet, "/images/"+reference+"/json", nil, nil); err == nil {
		return nil
	}
	pullCtx, cancelPull := context.WithTimeout(ctx, b.cfg.PullTimeout)
	defer cancelPull()
	return b.call(pullCtx, http.MethodPost, "/images/create?fromImage="+reference, nil, nil)
}

// Release destroys one sandbox and confirms it is gone.
//
// A removal that is merely requested is not a removal: until the daemon confirms
// it, the container still holds the memory its reservation covers, so returning
// the reservation here would over-commit the node.
func (b *Backend) Release(ctx context.Context, handle backend.Handle) error {
	// Close any persistent shell first, so its process exits with the container
	// rather than being orphaned against a container that is already gone.
	b.shells.drop(handle.SandboxID)
	if err := b.remove(ctx, handle.SandboxID); err != nil {
		return err
	}
	b.mu.Lock()
	delete(b.known, handle.SandboxID)
	b.mu.Unlock()
	return nil
}

func (b *Backend) remove(ctx context.Context, containerID string) error {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	err := b.call(ctx, http.MethodDelete, "/containers/"+containerID+"?force=true&v=true", nil, nil)
	if err == nil || isNotFound(err) {
		// Already gone is the outcome the caller wanted.
		return nil
	}
	return fmt.Errorf("docker remove %s: %w", containerID, err)
}

// Status reports a sandbox's portable state.
func (b *Backend) Status(ctx context.Context, handle backend.Handle) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	var inspected struct {
		State struct {
			Status   string `json:"Status"`
			Running  bool   `json:"Running"`
			Paused   bool   `json:"Paused"`
			OOMKill  bool   `json:"OOMKilled"`
			ExitCode int    `json:"ExitCode"`
		} `json:"State"`
	}
	if err := b.call(ctx, http.MethodGet, "/containers/"+handle.SandboxID+"/json", nil, &inspected); err != nil {
		if isNotFound(err) {
			return "terminated", nil
		}
		return "", err
	}
	switch {
	case inspected.State.Paused:
		return "paused", nil
	case inspected.State.Running:
		return "running", nil
	case inspected.State.Status == "exited" || inspected.State.Status == "dead":
		return "exited", nil
	default:
		return "unknown", nil
	}
}

// Pause freezes a container's process tree, keeping it resident on this node.
func (b *Backend) Pause(ctx context.Context, handle backend.Handle, mode string) error {
	if mode != "" && mode != "freeze" {
		return fmt.Errorf("docker keeps the container resident, so it offers a freeze rather than %q", mode)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	return b.call(ctx, http.MethodPost, "/containers/"+handle.SandboxID+"/pause", nil, nil)
}

// Resume unfreezes a paused container.
func (b *Backend) Resume(ctx context.Context, handle backend.Handle) error {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	return b.call(ctx, http.MethodPost, "/containers/"+handle.SandboxID+"/unpause", nil, nil)
}

// Snapshot commits the container's filesystem.
//
// Filesystem only: Docker has no supported memory checkpoint, so claiming a full
// state capture would promise a resume this backend cannot perform.
func (b *Backend) Snapshot(ctx context.Context, handle backend.Handle, kind string) (string, error) {
	if kind != "" && kind != "filesystem" {
		return "", fmt.Errorf("docker commits the writable layer, so it captures a filesystem snapshot rather than %q", kind)
	}
	ctx, cancel := context.WithTimeout(ctx, b.cfg.PullTimeout)
	defer cancel()
	tag := fmt.Sprintf("psrl-snapshot:%s-%d", handle.SandboxID[:min(12, len(handle.SandboxID))], time.Now().UnixNano())
	path := fmt.Sprintf("/commit?container=%s&repo=%s&tag=%s",
		handle.SandboxID, "psrl-snapshot", strings.SplitN(tag, ":", 2)[1])
	var committed struct {
		ID string `json:"Id"`
	}
	if err := b.call(ctx, http.MethodPost, path, nil, &committed); err != nil {
		return "", fmt.Errorf("docker commit: %w", err)
	}
	return committed.ID, nil
}

// ReadFile fetches one file from a sandbox via the Engine's archive endpoint.
//
// The shell-based approach (base64 | base64 -d) puts the entire file content
// on the command line, which fails with ENOMEM / "argument list too long" for
// files larger than ~96 KiB. The archive endpoint streams a tar containing the
// file without any shell involvement, so it works for arbitrarily large files.
func (b *Backend) ReadFile(ctx context.Context, handle backend.Handle, path string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	raw, err := b.callRaw(ctx, http.MethodGet,
		"/containers/"+handle.SandboxID+"/archive?path="+urlEncodePath(path), nil)
	if err != nil {
		return nil, fmt.Errorf("docker archive get %q: %w", path, err)
	}
	return extractTarSingle(raw)
}

// WriteFile uploads one file into a sandbox via the Engine's archive endpoint.
//
// Using PUT /containers/{id}/archive avoids the shell-argument-list limit that
// breaks the old printf-base64 approach for files larger than ~96 KiB (e.g.
// a large model patch). The data travels as a tar stream in the request body.
func (b *Backend) WriteFile(ctx context.Context, handle backend.Handle, path string, data []byte) error {
	tarBytes, err := singleFileTar(path, data)
	if err != nil {
		return fmt.Errorf("docker archive put %q: build tar: %w", path, err)
	}
	dir := parentDir(path)
	// Ensure the directory exists first so the archive upload lands cleanly.
	//
	// The command runs on the caller's context, not on the Engine budget that bounds
	// the upload below. How long a command takes is the command's own business, and
	// its bounds are the caller's own command timeout and the sandbox's lifetime.
	// Charging it to the request timeout makes a daemon too busy to start a process
	// report a stalled mkdir as a dead write, which is a fault in the wrong place.
	mkdirCmd := "mkdir -p " + shellescape(dir)
	code, out, execErr := b.execOneShot(ctx, handle, mkdirCmd, "", nil)
	if execErr != nil {
		return fmt.Errorf("docker archive put %q: mkdir: %w", path, execErr)
	}
	if code != 0 {
		return fmt.Errorf("docker archive put %q: mkdir exited %d: %s", path, code, out)
	}
	putCtx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	if err := b.putArchive(putCtx, handle.SandboxID, dir, tarBytes); err != nil {
		return fmt.Errorf("docker archive put %q: %w", path, err)
	}
	return nil
}

// putArchive calls PUT /containers/{id}/archive with a tar body.
func (b *Backend) putArchive(ctx context.Context, containerID, dir string, tarData []byte) error {
	url := "http://docker/" + b.cfg.APIVersion + "/containers/" + containerID + "/archive?path=" + urlEncodePath(dir)
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, url, bytes.NewReader(tarData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/x-tar")
	resp, err := b.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw := new(bytes.Buffer)
	_, _ = raw.ReadFrom(resp.Body)
	if resp.StatusCode >= 400 {
		return &apiError{status: resp.StatusCode, body: strings.TrimSpace(raw.String())}
	}
	return nil
}

// singleFileTar creates an in-memory tar archive containing one file.
// The entry name is the base name; the caller controls where it is written via
// the archive upload path parameter.
func singleFileTar(filePath string, data []byte) ([]byte, error) {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	name := path.Base(filePath)
	if err := tw.WriteHeader(&tar.Header{
		Typeflag: tar.TypeReg,
		Name:     name,
		Size:     int64(len(data)),
		Mode:     0o644,
	}); err != nil {
		return nil, err
	}
	if _, err := tw.Write(data); err != nil {
		return nil, err
	}
	if err := tw.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// extractTarSingle reads the single file from a tar that the archive endpoint
// returns for a non-directory path.
func extractTarSingle(tarData []byte) ([]byte, error) {
	tr := tar.NewReader(bytes.NewReader(tarData))
	_, err := tr.Next()
	if err != nil {
		return nil, fmt.Errorf("archive entry: %w", err)
	}
	return io.ReadAll(tr)
}

// urlEncodePath percent-encodes a filesystem path for use in a query string.
// Only the characters that would break a URL need encoding; / is left intact
// because Docker parses it as a path component, not as a delimiter.
func urlEncodePath(p string) string {
	// net/url.PathEscape encodes / as %2F, which Docker does not accept in the
	// path query parameter. Replace it back after encoding.
	return strings.ReplaceAll(pathEscape(p), "%2F", "/")
}

// pathEscape is net/url.PathEscape without the special treatment of /.
func pathEscape(s string) string {
	var buf strings.Builder
	const safe = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~:@!$&'()*+,;=/"
	for _, b := range []byte(s) {
		if strings.ContainsRune(safe, rune(b)) {
			buf.WriteByte(b)
		} else {
			fmt.Fprintf(&buf, "%%%02X", b)
		}
	}
	return buf.String()
}

// parentDir returns the directory component of a path, or "/" when there is
// none.
func parentDir(p string) string {
	d := path.Dir(p)
	if d == "" || d == "." {
		return "/"
	}
	return d
}

// shellescape wraps a string in single quotes, escaping any embedded single
// quotes. This is sufficient for mkdir paths that may contain spaces.
func shellescape(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "'\\''") + "'"
}

// DeleteSnapshot removes a committed image.
func (b *Backend) DeleteSnapshot(ctx context.Context, snapshotID string) error {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	err := b.call(ctx, http.MethodDelete, "/images/"+snapshotID+"?force=true", nil, nil)
	if err != nil && !isNotFound(err) {
		return err
	}
	return nil
}

// Exec runs one command in a sandbox and returns its result.
//
// This is here because a container has no in-sandbox agent to talk to. It is
// reached through the node agent rather than the control plane, so command
// traffic still stays node-local.
func (b *Backend) Exec(ctx context.Context, handle backend.Handle, command string, workdir string, env map[string]string) (int, string, error) {
	return b.execOneShot(ctx, handle, command, workdir, env)
}

// ExecWithMode runs a command under the strategy the spec asked for.
//
// A persistent shell keeps working directory, exported variables, and an
// activated environment alive between calls, which is what a multi-turn harness
// needs. One shot starts a process per command, which is cheaper and is right
// for a grader step that runs exactly once. The strategy is a property of the
// workload rather than of the backend, so it travels on the spec and the node
// passes it here.
func (b *Backend) ExecWithMode(
	ctx context.Context, handle backend.Handle, execMode, command, workdir string, env map[string]string,
) (int, string, error) {
	if execMode == "persistent" {
		return b.execPersistent(ctx, handle, command, workdir, env)
	}
	return b.execOneShot(ctx, handle, command, workdir, env)
}

// execOneShot starts a new process per command, holding no state between calls.
func (b *Backend) execOneShot(ctx context.Context, handle backend.Handle, command string, workdir string, env map[string]string) (int, string, error) {
	envList := make([]string, 0, len(env))
	for key, value := range env {
		envList = append(envList, key+"="+value)
	}
	sort.Strings(envList)
	create := map[string]any{
		"AttachStdout": true,
		"AttachStderr": true,
		"Cmd":          []string{"/bin/sh", "-c", command},
	}
	if workdir != "" {
		create["WorkingDir"] = workdir
	}
	if len(envList) > 0 {
		create["Env"] = envList
	}
	var created struct {
		ID string `json:"Id"`
	}
	if err := b.call(ctx, http.MethodPost, "/containers/"+handle.SandboxID+"/exec", create, &created); err != nil {
		return 0, "", fmt.Errorf("docker exec create: %w", err)
	}
	output, err := b.callRaw(ctx, http.MethodPost, "/exec/"+created.ID+"/start", map[string]any{"Detach": false, "Tty": false})
	if err != nil {
		return 0, "", fmt.Errorf("docker exec start: %w", err)
	}
	var inspected struct {
		ExitCode int  `json:"ExitCode"`
		Running  bool `json:"Running"`
	}
	if err := b.call(ctx, http.MethodGet, "/exec/"+created.ID+"/json", nil, &inspected); err != nil {
		return 0, "", fmt.Errorf("docker exec inspect: %w", err)
	}
	return inspected.ExitCode, demultiplex(output), nil
}

// demultiplex strips the Engine's stream framing.
//
// An attached exec without a TTY returns a framed stream: an eight-byte header
// per chunk, with the payload length in the last four bytes. Returning the raw
// bytes would put binary headers into a command's output.
func demultiplex(raw []byte) string {
	var out bytes.Buffer
	for len(raw) >= 8 {
		size := int(raw[4])<<24 | int(raw[5])<<16 | int(raw[6])<<8 | int(raw[7])
		raw = raw[8:]
		if size > len(raw) {
			size = len(raw)
		}
		out.Write(raw[:size])
		raw = raw[size:]
	}
	// A daemon that returned unframed bytes is older than the framing; keep them
	// rather than dropping a command's output.
	if out.Len() == 0 {
		return string(raw)
	}
	return out.String()
}

// Reconcile returns every sandbox this service owns on this node, with the facts
// a restarted node needs to charge for it.
//
// The daemon is the source of truth: a service that trusted its own list would
// leak every container it forgot across a restart. A warm pool entry is the one
// exception, for the reason on refuseRunningPoolEntries.
func (b *Backend) Reconcile(ctx context.Context) ([]backend.Reconciled, error) {
	listed, err := b.listOwned(ctx, labelSandbox+"=true", labelOwner+"="+b.cfg.OwnerID)
	if err != nil {
		return nil, err
	}
	found, err := b.inspectAll(ctx, listed)
	if err != nil {
		return nil, err
	}
	if err := refuseRunningPoolEntries(found); err != nil {
		return nil, err
	}
	return found, nil
}

// refuseRunningPoolEntries refuses to reconcile while a warm pool entry is running.
//
// A pool entry's class label is written at create and a container's labels are
// immutable, so a claimed entry still reads as pool inventory, and the claim
// itself was recorded only in the memory of the process that is gone. A running
// entry is therefore unattributable: charging it would be wrong, and destroying
// it could destroy a caller's live sandbox. An entry that has exited holds
// nothing and passes through, for the node to reclaim.
func refuseRunningPoolEntries(found []backend.Reconciled) error {
	running := 0
	for _, rec := range found {
		if rec.Class == warmPoolClass && rec.Running {
			running++
		}
	}
	if running == 0 {
		return nil
	}
	return fmt.Errorf(
		"%d running warm pool containers cannot be attributed to a caller after a restart: "+
			"drain the pool before restarting, or keep it disabled", running)
}

// FindByIdempotencyKey returns the sandbox a create with this key already made.
//
// A create whose response was lost leaves a container the caller never learned
// the id of, so a retry has to find it rather than make a second one.
func (b *Backend) FindByIdempotencyKey(
	ctx context.Context, key string,
) (backend.Reconciled, bool, error) {
	if key == "" {
		return backend.Reconciled{}, false, nil
	}
	listed, err := b.listOwned(ctx,
		labelSandbox+"=true", labelOwner+"="+b.cfg.OwnerID, labelIdempotency+"="+key)
	if err != nil {
		return backend.Reconciled{}, false, err
	}
	if len(listed) == 0 {
		return backend.Reconciled{}, false, nil
	}
	found, err := b.inspectOne(ctx, listed[0])
	if err != nil {
		return backend.Reconciled{}, false, err
	}
	return found, true, nil
}

// listOwned lists the container ids this service owns that match every label.
func (b *Backend) listOwned(ctx context.Context, labels ...string) ([]string, error) {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	filter, err := json.Marshal(map[string][]string{"label": labels})
	if err != nil {
		return nil, err
	}
	var listed []struct {
		ID string `json:"Id"`
	}
	path := "/containers/json?all=true&filters=" + urlEncode(string(filter))
	if err := b.call(ctx, http.MethodGet, path, nil, &listed); err != nil {
		return nil, err
	}
	out := make([]string, 0, len(listed))
	for _, container := range listed {
		out = append(out, container.ID)
	}
	return out, nil
}

// inspectAll reads every id back, bounded so a node holding many containers does
// not serialize the work into a slow startup.
func (b *Backend) inspectAll(ctx context.Context, ids []string) ([]backend.Reconciled, error) {
	const inspectors = 8
	out := make([]backend.Reconciled, len(ids))
	errs := make([]error, len(ids))
	sem := make(chan struct{}, inspectors)
	var wg sync.WaitGroup
	for i, id := range ids {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			out[i], errs[i] = b.inspectOne(ctx, id)
		}()
	}
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("inspect %s: %w", ids[i], err)
		}
	}
	return out, nil
}

// inspectOne rebuilds what one container holds from the daemon's own view of it.
func (b *Backend) inspectOne(ctx context.Context, id string) (backend.Reconciled, error) {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	var inspected struct {
		ID      string `json:"Id"`
		Created string `json:"Created"`
		State   struct {
			Running bool `json:"Running"`
		} `json:"State"`
		Config struct {
			Labels map[string]string `json:"Labels"`
			Env    []string          `json:"Env"`
		} `json:"Config"`
		HostConfig struct {
			Memory    int64 `json:"Memory"`
			CpuQuota  int64 `json:"CpuQuota"`
			CpuPeriod int64 `json:"CpuPeriod"`
		} `json:"HostConfig"`
	}
	if err := b.call(ctx, http.MethodGet, "/containers/"+id+"/json", nil, &inspected); err != nil {
		return backend.Reconciled{}, err
	}
	labels := inspected.Config.Labels
	rec := backend.Reconciled{
		Handle:   backend.Handle{Backend: b.Name(), SandboxID: id, NodeID: labels[labelNode]},
		Owner:    labels[backend.OwnerMetadataKey],
		Class:    labels[labelClass],
		ExecMode: labels[labelExecMode],
		Running:  inspected.State.Running,
		Resources: backend.Resources{
			MemoryMB: inspected.HostConfig.Memory / (1024 * 1024),
		},
	}
	// CPU is charged in millis and written as a quota against a fixed period, so
	// reading it back is the same division the create made.
	if inspected.HostConfig.CpuPeriod > 0 {
		rec.Resources.CPUCount = float64(inspected.HostConfig.CpuQuota) /
			float64(inspected.HostConfig.CpuPeriod)
	}
	if seconds, err := strconv.ParseInt(labels[labelLifetime], 10, 64); err == nil {
		rec.Lifetime = time.Duration(seconds) * time.Second
	}
	if diskMB, err := strconv.ParseInt(labels[labelDisk], 10, 64); err == nil {
		rec.Resources.DiskMB = diskMB
	}
	if created, err := time.Parse(time.RFC3339Nano, inspected.Created); err == nil {
		rec.CreatedAt = created
	}
	for _, entry := range inspected.Config.Env {
		if value, found := strings.CutPrefix(entry, "CUDA_VISIBLE_DEVICES="); found {
			rec.GPUIndices = parseDeviceList(value)
		}
	}
	return rec, nil
}

// parseDeviceList reads the comma-separated device indices a create wrote.
func parseDeviceList(value string) []int32 {
	if value == "" {
		return nil
	}
	out := make([]int32, 0, 1)
	for _, part := range strings.Split(value, ",") {
		if index, err := strconv.ParseInt(strings.TrimSpace(part), 10, 32); err == nil {
			out = append(out, int32(index))
		}
	}
	return out
}

// Start begins the warm pool's refill and sweep loops.
//
// It is separate from New because the pool creates containers, and construction
// has to be able to happen inside a preflight that is not allowed to change the
// node's state. A backend whose pool is disabled does nothing here.
func (b *Backend) Start(ctx context.Context) {
	b.pool.Start(ctx)
}

// Stop ends the pool's loops and destroys every entry it still holds, so a
// shutdown leaves no unowned container behind for the next start to reclaim.
func (b *Backend) Stop(ctx context.Context) {
	b.pool.Stop(ctx)
}

// WarmPool reports what the pool holds, for the metric hook. A backend with no
// pool reports a zero value rather than nothing, so the family is present either
// way and a reader does not have to special-case its absence.
func (b *Backend) WarmPool() WarmPoolReport {
	return b.pool.Snapshot()
}

// Preflight refuses a node whose daemon this service cannot drive.
//
// Every check here is something that would otherwise fail inside an episode,
// where it costs a sample and reads as a flaky rollout rather than as the
// configuration error it is. The pinned API version is compared against what
// the daemon will actually serve, and a configured runtime has to exist: both
// are declared as capabilities, and a capability the node cannot honour is
// worse than one it never claimed.
func (b *Backend) Preflight(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, b.cfg.RequestTimeout)
	defer cancel()
	var version struct {
		APIVersion    string `json:"ApiVersion"`
		MinAPIVersion string `json:"MinAPIVersion"`
		Version       string `json:"Version"`
	}
	// Unversioned, because the pinned version is what is being checked: asking
	// through the prefix makes a too-new pin look like a dead daemon.
	if err := b.callUnversioned(ctx, http.MethodGet, "/version", &version); err != nil {
		return fmt.Errorf("docker daemon at %s is not answering: %w", b.cfg.Socket, err)
	}
	if version.APIVersion == "" {
		return fmt.Errorf("docker daemon at %s reported no API version", b.cfg.Socket)
	}
	// The service pins an API version, so a daemon that will not serve it has to
	// fail here. Discovering it on the first create means finding out from an
	// endpoint that silently did not exist.
	if version.MinAPIVersion != "" && olderAPI(b.cfg.APIVersion, version.MinAPIVersion) {
		return fmt.Errorf(
			"docker daemon at %s serves API %s and newer, but this service is pinned to %s; "+
				"set backends[].api_version to %s or newer",
			b.cfg.Socket, version.MinAPIVersion, b.cfg.APIVersion, version.MinAPIVersion)
	}
	if olderAPI(version.APIVersion, b.cfg.APIVersion) {
		return fmt.Errorf(
			"docker daemon at %s serves API up to %s, but this service is pinned to %s; "+
				"upgrade the daemon or set backends[].api_version to %s",
			b.cfg.Socket, version.APIVersion, b.cfg.APIVersion, version.APIVersion)
	}
	if err := b.preflightCgroupDriver(ctx); err != nil {
		return err
	}
	return b.preflightRuntime(ctx)
}

// preflightCgroupDriver refuses a daemon that would not put a container in the
// configured subtree.
//
// Only the cgroupfs driver treats CgroupParent as a path. The systemd driver reads it as
// a slice name, so the same configuration would place containers somewhere else and the
// sampler would measure a domain they are not in -- a failure that otherwise shows up as
// a ceiling that never moves.
func (b *Backend) preflightCgroupDriver(ctx context.Context) error {
	if b.cfg.CgroupParent == "" {
		return nil
	}
	var info struct {
		CgroupDriver string `json:"CgroupDriver"`
	}
	if err := b.call(ctx, http.MethodGet, "/info", nil, &info); err != nil {
		return fmt.Errorf("docker daemon at %s did not report its cgroup driver: %w", b.cfg.Socket, err)
	}
	if info.CgroupDriver != "" && info.CgroupDriver != "cgroupfs" {
		return fmt.Errorf(
			"this backend sets cgroup_parent %q but the daemon at %s uses the %q cgroup driver, "+
				"which reads that value as a slice name rather than a path; containers would not land in "+
				"the sandbox subtree the sampler measures",
			b.cfg.CgroupParent, b.cfg.Socket, info.CgroupDriver)
	}
	return nil
}

// preflightRuntime refuses a configured runtime the daemon does not have.
//
// This backend declares isolation_runtime whenever a runtime is configured, and
// placement routes a spec that requires it here. If the runtime is absent, the
// daemon rejects the create -- so without this check the service advertises an
// isolation boundary it cannot provide and the refusal arrives per episode.
func (b *Backend) preflightRuntime(ctx context.Context) error {
	if b.cfg.Runtime == "" {
		return nil
	}
	var info struct {
		Runtimes map[string]struct {
			Path string `json:"path"`
		} `json:"Runtimes"`
	}
	if err := b.call(ctx, http.MethodGet, "/info", nil, &info); err != nil {
		return fmt.Errorf("docker daemon at %s did not report its runtimes: %w", b.cfg.Socket, err)
	}
	if _, has := info.Runtimes[b.cfg.Runtime]; !has {
		available := make([]string, 0, len(info.Runtimes))
		for name := range info.Runtimes {
			available = append(available, name)
		}
		sort.Strings(available)
		return fmt.Errorf(
			"docker runtime %q is not installed on this node (available: %v), but this backend "+
				"declares isolation_runtime with it, so a spec requiring stronger isolation would be "+
				"admitted and then run without it",
			b.cfg.Runtime, available)
	}
	return nil
}

// olderAPI reports whether a Docker API version precedes another.
//
// Compared field by field as integers rather than as strings, because the
// versions pass 1.9: "v1.10" sorts before "v1.9" lexically and after it
// numerically, and the wrong answer here refuses a daemon that would have worked.
func olderAPI(version, floor string) bool {
	left := apiParts(version)
	right := apiParts(floor)
	if left == nil || right == nil {
		// Not comparable. The only safe answer is "not older", so an unrecognised
		// version string never refuses a daemon that would have worked.
		return false
	}
	for i := 0; i < len(left) && i < len(right); i++ {
		if left[i] != right[i] {
			return left[i] < right[i]
		}
	}
	return len(left) < len(right)
}

func apiParts(version string) []int {
	fields := strings.Split(strings.TrimPrefix(version, "v"), ".")
	out := make([]int, 0, len(fields))
	for _, field := range fields {
		if field == "" {
			// An empty field is not a zero: "" would otherwise parse as version 0 and
			// compare older than everything.
			return nil
		}
		value := 0
		for i := 0; i < len(field); i++ {
			if field[i] < '0' || field[i] > '9' {
				// A version with a non-numeric field is not comparable, so the caller
				// gets the only safe answer: not older, and therefore not refused.
				return nil
			}
			value = value*10 + int(field[i]-'0')
		}
		out = append(out, value)
	}
	return out
}

func (b *Backend) call(ctx context.Context, method, path string, body any, out any) error {
	raw, err := b.callRaw(ctx, method, path, body)
	if err != nil {
		return err
	}
	if out == nil || len(raw) == 0 {
		return nil
	}
	return json.Unmarshal(raw, out)
}

// callUnversioned reaches an endpoint without the pinned API prefix.
//
// Only /version uses this, and it has to: a daemon rejects a URL carrying an
// API version it does not serve, so asking it through the prefix returns a 400
// that reads as "the daemon is not answering". The point of preflight is to say
// which setting is wrong, and that answer is in the reply this fetches.
func (b *Backend) callUnversioned(ctx context.Context, method, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, method, "http://docker"+path, bytes.NewReader(nil))
	if err != nil {
		return err
	}
	resp, err := b.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw := new(bytes.Buffer)
	if _, err := raw.ReadFrom(resp.Body); err != nil {
		return err
	}
	if resp.StatusCode >= 400 {
		return fmt.Errorf("docker API returned %d: %s", resp.StatusCode, strings.TrimSpace(raw.String()))
	}
	if out == nil || raw.Len() == 0 {
		return nil
	}
	return json.Unmarshal(raw.Bytes(), out)
}

func (b *Backend) callRaw(ctx context.Context, method, path string, body any) ([]byte, error) {
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
	url := "http://docker/" + b.cfg.APIVersion + path
	req, err := http.NewRequestWithContext(ctx, method, url, payload)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
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
	return fmt.Sprintf("docker API returned %d: %s", e.status, e.body)
}

func isNotFound(err error) bool {
	var apiErr *apiError
	if ok := asAPIError(err, &apiErr); ok {
		return apiErr.status == http.StatusNotFound
	}
	return false
}

func asAPIError(err error, target **apiError) bool {
	for err != nil {
		if typed, ok := err.(*apiError); ok {
			*target = typed
			return true
		}
		unwrapper, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = unwrapper.Unwrap()
	}
	return false
}

func urlEncode(value string) string {
	var out strings.Builder
	for _, r := range value {
		switch {
		case r == '{' || r == '}' || r == '[' || r == ']' || r == '"' || r == ',' ||
			r == ':' || r == '=' || r == ' ' || r == '\\' || r == '%':
			fmt.Fprintf(&out, "%%%02X", r)
		default:
			out.WriteRune(r)
		}
	}
	return out.String()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
