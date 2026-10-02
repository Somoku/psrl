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
//     reachable node runtime per configured node (psrl mode).
//
//   - opensandbox: requires a reachable OpenSandbox lifecycle server (provider
//     mode) or one reachable docker-mode opensandbox-server per configured node
//     (psrl mode).
//
//   - cubesandbox: requires a reachable CubeSandbox gateway (provider mode) or
//     one reachable Cubelet per configured node (psrl mode).
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
	"net/http"
	"strings"
	"time"
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

// CheckAgentEnvNode probes one AgentENV node runtime in psrl mode.
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

// CheckOpenSandboxNode probes one docker-mode opensandbox-server node in psrl mode.
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

// CheckCubeSandboxGateway probes a CubeSandbox CubeMaster gateway in provider mode.
func CheckCubeSandboxGateway(ctx context.Context, gatewayURL, apiKey string) Result {
	label := "cubesandbox gateway"
	return probe(ctx, label, normalize(gatewayURL)+"/health", apiKey, "X-Api-Key",
		"Ensure the CubeSandbox CubeMaster gateway is running and reachable at "+gatewayURL)
}

// CheckCubeSandboxNode probes one Cubelet node in psrl mode.
//
// Each Cubelet serves the full sandbox API on its own port. The probe checks
// GET /health; a failure means that specific Cubelet is unreachable.
func CheckCubeSandboxNode(ctx context.Context, nodeID, nodeURL, apiKey string) Result {
	label := "cubesandbox node " + nodeID
	return probe(ctx, label, normalize(nodeURL)+"/health", apiKey, "X-Api-Key",
		"Ensure the CubeSandbox Cubelet is running and reachable at "+nodeURL+" (node "+nodeID+")")
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
