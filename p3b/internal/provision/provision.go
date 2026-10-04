// Package provision checks that the external services a backend type requires
// are reachable and minimally functional before sandboxd begins serving.
//
// No installation is ever performed here. If a required service is absent the
// check returns an actionable error that names the service, describes what is
// missing, and tells the operator what to do. Installing software is the
// operator's job; reporting clearly when it is missing is ours.
//
// The three backends that need an external service are:
//
//   - agentenv: requires a reachable AgentENV gateway (provider mode) or one
//     reachable node runtime per configured node (direct mode).
//
//   - opensandbox: in provider mode, a reachable OpenSandbox gateway. In direct
//     mode there is no OpenSandbox control plane at all -- this service drives the
//     container runtime and stages the agent itself -- so what is checked instead
//     is the runtime, the agent image, and the stage directory.
//
//   - cubesandbox: requires a reachable CubeSandbox gateway (provider mode) or
//     one reachable Cubelet per configured node (direct mode).
//
// The docker backend requires only a local Docker daemon, whose liveness is
// already verified by dockerbackend.Preflight.
//
// Each check is a plain HTTP probe against the service's health or list
// endpoint. The probe proves both TCP reachability and auth/routing (where an
// API key is configured), not just port open. A service that is reachable but
// refuses the probe with an auth error is reported separately from one that is
// simply down, so the operator can distinguish a wrong key from a missing
// service.
package provision

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	cubebox "psrl.dev/sandboxd/internal/backend/cubesandbox/cubeletpb/services/cubebox/v1"
)

const probeTimeout = 15 * time.Second

// Result is the outcome of one pre-flight probe.
type Result struct {
	// Name is a human-readable label for what was probed ("agentenv gateway",
	// "opensandbox node node-1").
	Name string
	// Err is nil when the probe passed.
	Err error
}

// CheckAgentEnvGateway probes an AgentENV gateway in provider mode.
//
// The gateway is considered ready when GET /health returns 2xx. A non-2xx
// response, a connection error, or a timeout all produce an error that
// includes the URL and a pointer to the installation guide.
func CheckAgentEnvGateway(ctx context.Context, gatewayURL, apiKey string) Result {
	label := "agentenv gateway"
	return probe(ctx, label, normalize(gatewayURL)+"/health", apiKey, "X-API-Key",
		"Install AgentENV and start the gateway, then verify with: curl "+gatewayURL+"/health")
}

// CheckAgentEnvNode probes one AgentENV node runtime in direct mode.
//
// Each node runtime serves the full sandbox API on its own port. The probe
// checks GET /health; a failure means that specific node is unreachable.
func CheckAgentEnvNode(ctx context.Context, nodeID, nodeURL, apiKey string) Result {
	label := "agentenv node " + nodeID
	return probe(ctx, label, normalize(nodeURL)+"/health", apiKey, "X-API-Key",
		"Ensure the AgentENV node runtime is running on "+nodeURL)
}

// CheckOpenSandboxGateway probes an OpenSandbox lifecycle server in provider mode.
//
// OpenSandbox exposes GET /health for connectivity and GET /v1/sandboxes for
// auth. We probe /health first (cheap) and then /v1/sandboxes?pageSize=1
// (proves the API key is accepted), matching the behaviour of the Go adapter's
// own Preflight.
func CheckOpenSandboxGateway(ctx context.Context, gatewayURL, apiKey string) Result {
	label := "opensandbox gateway"
	base := normalize(gatewayURL)

	// TCP + routing check.
	if r := probe(ctx, label, base+"/health", "", "",
		"Install opensandbox-server (pip install 'opensandbox-server==0.2.3') and start it, "+
			"then verify with: curl "+gatewayURL+"/health"); r.Err != nil {
		return r
	}
	// Auth check.
	return probe(ctx, label, base+"/v1/sandboxes?pageSize=1", apiKey, "OPEN-SANDBOX-API-KEY",
		"OpenSandbox is reachable but rejected the API key. "+
			"Check [server].api_key in ~/.sandbox.toml and set OPENSANDBOX_API_KEY accordingly.")
}

// CheckOpenSandboxNode probes one docker-mode opensandbox-server node in direct mode.
//
// Each per-node server exposes GET /health. A missing server produces an error
// that names the node and gives the install command.
func CheckOpenSandboxNode(ctx context.Context, nodeID, nodeURL, apiKey string) Result {
	label := "opensandbox node " + nodeID
	base := normalize(nodeURL)
	if r := probe(ctx, label, base+"/health", "", "",
		"opensandbox-server is not running on "+nodeURL+". "+
			"Install with: pip install 'opensandbox-server==0.2.3'\n"+
			"Configure with: opensandbox-server init-config ~/.sandbox.toml --example docker\n"+
			"Start with:     opensandbox-server --config ~/.sandbox.toml"); r.Err != nil {
		return r
	}
	if apiKey != "" {
		return probe(ctx, label, base+"/v1/sandboxes?pageSize=1", apiKey, "OPEN-SANDBOX-API-KEY",
			"opensandbox-server on "+nodeURL+" rejected the API key. "+
				"Check [server].api_key in ~/.sandbox.toml.")
	}
	return Result{Name: label}
}

// CheckOpenSandboxDirect verifies what a directly driven OpenSandbox needs.
//
// Direct mode runs no OpenSandbox control plane at all: this service drives the
// container runtime and stages OpenSandbox's agent into each sandbox itself. So
// there is no server to probe. What must hold instead is that the runtime is
// reachable, that the agent image is available, and that the stage directory can
// be written -- all three on this machine.
//
// The agent image is only checked for presence here. A pull is the operator's
// decision: doing it inside a preflight would turn a configuration check into a
// multi-minute network operation, and a deployment pipeline reading the exit code
// could not tell a slow registry from a wrong reference.
func CheckOpenSandboxDirect(ctx context.Context, dockerSocket, execdImage, stageDir string) Result {
	label := "opensandbox direct runtime"

	if execdImage == "" {
		return Result{Name: label, Err: fmt.Errorf(
			"%s: no agent image is configured\n"+
				"To fix: set execd_image on the opensandbox backend. The agent is staged from "+
				"that image into every sandbox, so direct mode cannot run without it.", label)}
	}
	if dockerSocket == "" {
		return Result{Name: label, Err: fmt.Errorf(
			"%s: no container runtime socket is configured\n"+
				"To fix: set socket on the opensandbox backend (for example "+
				"unix:///var/run/docker.sock). Direct mode drives the runtime itself.", label)}
	}

	// The runtime, first: nothing else is actionable if it is not answering.
	if err := pingDockerSocket(ctx, dockerSocket); err != nil {
		return Result{Name: label, Err: fmt.Errorf(
			"%s: the container runtime at %s is not answering: %w\n"+
				"To fix: start the Docker daemon and confirm this process may read its socket "+
				"(verify with: docker version).", label, dockerSocket, err)}
	}

	// The agent image. Absent is reported as actionable rather than fatal-sounding,
	// because one pull resolves it.
	present, err := dockerImagePresent(ctx, dockerSocket, execdImage)
	if err != nil {
		return Result{Name: label, Err: fmt.Errorf(
			"%s: could not ask the runtime about the agent image %s: %w", label, execdImage, err)}
	}
	if !present {
		return Result{Name: label, Err: fmt.Errorf(
			"%s: the agent image %s is not present on this host\n"+
				"To fix: docker pull %s", label, execdImage, execdImage)}
	}

	// The stage directory. It is written once at startup and then mounted
	// read-only into every sandbox, so an unwritable path fails every create.
	if stageDir == "" {
		stageDir = "/var/lib/sandboxd/opensandbox-agent"
	}
	if err := checkWritableDir(stageDir); err != nil {
		return Result{Name: label, Err: fmt.Errorf(
			"%s: the agent stage directory %s is not usable: %w\n"+
				"To fix: create it and make it writable by this process "+
				"(mkdir -p %s), or set stage_dir to a path that is.",
			label, stageDir, err, stageDir)}
	}

	return Result{Name: label}
}

// CheckCubeSandboxGateway probes a CubeSandbox CubeMaster gateway in provider mode.
func CheckCubeSandboxGateway(ctx context.Context, gatewayURL, apiKey string) Result {
	label := "cubesandbox gateway"
	return probe(ctx, label, normalize(gatewayURL)+"/health", apiKey, "X-Api-Key",
		"Ensure the CubeSandbox CubeMaster gateway is running and reachable at "+gatewayURL)
}

// CheckCubeSandboxNode probes one Cubelet node in direct mode.
//
// A Cubelet is probed over gRPC, not HTTP, and the distinction is not academic.
// Cubelet binds four listeners on independently configured addresses: an HTTP
// one, a TTRPC socket, a gRPC TCP port, and a gRPC unix socket. The address this
// backend is configured with is the gRPC TCP one, because that is where
// CubeboxMgr -- the service every lifecycle call uses -- is served. Cubelet's
// HTTP listener serves metrics and snhost, and has no /health route at all.
//
// So an HTTP GET /health against the configured address fails twice over: it
// asks for a path that does not exist, at a port speaking a different protocol.
// It would fail on a correctly deployed Cubelet, which makes it worse than no
// check: --preflight exists to catch a misconfiguration before the service
// serves, and a check that reports a working deployment as broken trains an
// operator to ignore it.
//
// The probe is a real CubeboxMgr call rather than the gRPC health service,
// because Cubelet does not register grpc.health.v1 (no health.NewServer anywhere
// in its tree). List with an empty request is the cheapest call that proves the
// service is routable and answering, which is the same thing the backend's own
// Preflight does.
func CheckCubeSandboxNode(ctx context.Context, nodeID, nodeURL, apiKey string) Result {
	label := "cubesandbox node " + nodeID
	hint := "Ensure the CubeSandbox Cubelet is running and its gRPC TCP endpoint is reachable at " +
		nodeURL + " (node " + nodeID + "). This is the grpc.tcp_address in the Cubelet " +
		"configuration, not its HTTP metrics port."

	target := grpcTarget(nodeURL)
	ctx, cancel := context.WithTimeout(ctx, probeTimeout)
	defer cancel()

	conn, err := grpc.NewClient(target, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return Result{Name: label, Err: fmt.Errorf(
			"%s: %s cannot be dialled: %w\nTo fix: %s", label, target, err, hint)}
	}
	defer conn.Close()

	if _, err := cubebox.NewCubeboxMgrClient(conn).List(
		ctx, &cubebox.ListCubeSandboxRequest{},
	); err != nil {
		return Result{Name: label, Err: fmt.Errorf(
			"%s is not answering CubeboxMgr at %s: %w\nTo fix: %s", label, target, err, hint)}
	}
	return Result{Name: label}
}

// grpcTarget strips a scheme a caller may have written out of habit.
//
// A gRPC target is a host:port authority. An "http://" prefix left in place is
// taken as part of the host name and fails to resolve, which reads as an
// unreachable node rather than a malformed address.
func grpcTarget(address string) string {
	for _, scheme := range []string{"http://", "https://", "grpc://"} {
		address = strings.TrimPrefix(address, scheme)
	}
	return strings.TrimSuffix(address, "/")
}

// probe fires one HTTP GET and maps the result to a Result.
func probe(ctx context.Context, label, url, apiKey, apiKeyHeader, hint string) Result {
	ctx, cancel := context.WithTimeout(ctx, probeTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return Result{Name: label, Err: fmt.Errorf("%s: could not build request for %s: %w", label, url, err)}
	}
	if apiKey != "" && apiKeyHeader != "" {
		req.Header.Set(apiKeyHeader, apiKey)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return Result{
			Name: label,
			Err: fmt.Errorf(
				"%s is not reachable at %s: %w\nTo fix: %s",
				label, url, err, hint,
			),
		}
	}
	defer resp.Body.Close()
	// Drain up to 512 bytes for the error message.
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return Result{
			Name: label,
			Err: fmt.Errorf(
				"%s at %s returned HTTP %d (auth failed): %s\nTo fix: %s",
				label, url, resp.StatusCode, strings.TrimSpace(string(body)), hint,
			),
		}
	}
	if resp.StatusCode >= 400 {
		return Result{
			Name: label,
			Err: fmt.Errorf(
				"%s at %s returned HTTP %d: %s\nTo fix: %s",
				label, url, resp.StatusCode, strings.TrimSpace(string(body)), hint,
			),
		}
	}
	return Result{Name: label}
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

// -- container runtime probes --------------------------------------------------

// dockerSocketClient dials a Docker daemon over a unix socket or a TCP address.
//
// A probe-local client rather than a shared one: a check must not be able to
// disturb a serving backend's connection pool, and it is discarded immediately.
func dockerSocketClient(socket string) *http.Client {
	transport := &http.Transport{}
	if path, found := strings.CutPrefix(socket, "unix://"); found {
		transport.DialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", path)
		}
	} else {
		host := strings.TrimPrefix(strings.TrimPrefix(socket, "tcp://"), "http://")
		transport.DialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "tcp", host)
		}
	}
	return &http.Client{Transport: transport}
}

// pingDockerSocket proves the daemon is answering, which is the cheapest useful
// check and the one every other runtime check depends on.
func pingDockerSocket(ctx context.Context, socket string) error {
	ctx, cancel := context.WithTimeout(ctx, probeTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://docker/_ping", nil)
	if err != nil {
		return err
	}
	resp, err := dockerSocketClient(socket).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode >= 400 {
		return fmt.Errorf("the daemon answered /_ping with HTTP %d", resp.StatusCode)
	}
	return nil
}

// dockerImagePresent reports whether an image is already on this host.
//
// A 404 is the answer "not present" rather than a fault, so it is distinguished
// from a transport failure: the operator's fix differs between a missing image
// and a daemon that cannot be reached.
func dockerImagePresent(ctx context.Context, socket, image string) (bool, error) {
	ctx, cancel := context.WithTimeout(ctx, probeTimeout)
	defer cancel()
	url := "http://docker/v1.43/images/" + image + "/json"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false, err
	}
	resp, err := dockerSocketClient(socket).Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 512))
	if resp.StatusCode == http.StatusNotFound {
		return false, nil
	}
	if resp.StatusCode >= 400 {
		return false, fmt.Errorf("the daemon answered HTTP %d", resp.StatusCode)
	}
	return true, nil
}

// checkWritableDir confirms a directory exists and can be written, creating it
// when it is absent.
//
// Creating here is not an installation: the directory is this service's own
// working state, not third-party software. What it must not do is leave a
// deployment to discover at the first create that the path was unusable.
func checkWritableDir(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	probe := filepath.Join(dir, ".write-probe")
	if err := os.WriteFile(probe, []byte("ok"), 0o644); err != nil {
		return err
	}
	return os.Remove(probe)
}
