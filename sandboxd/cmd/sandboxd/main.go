// Command sandboxd is the sandbox service.
//
// One binary serves both roles. A single-machine deployment runs control and
// node together and the SDK reaches both over one socket; a fleet runs one
// control process and one node process per machine. Keeping it one binary means
// a node and its control plane cannot be built from different commits, which is
// the failure a split would invite.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/backend/agentenv"
	"psrl.dev/sandboxd/internal/backend/cubesandbox"
	"psrl.dev/sandboxd/internal/backend/dockerbackend"
	"psrl.dev/sandboxd/internal/backend/opensandbox"
	"psrl.dev/sandboxd/internal/monitor"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/placement"
	"psrl.dev/sandboxd/internal/quota"
	"psrl.dev/sandboxd/internal/server"
	"psrl.dev/sandboxd/internal/timing"
)

// Config is the whole deployment, stated once.
//
// An operator declares intent -- how long an episode takes, what each class is
// owed, which backends exist -- and every deadline follows from it. A knob that
// could be derived is not a knob.
type Config struct {
	// Listen is where the SDK reaches this process.
	Listen string `json:"listen"`
	NodeID string `json:"node_id"`

	Timing struct {
		EpisodeDeadlineS float64            `json:"episode_deadline_s"`
		NodeTTLS         float64            `json:"node_ttl_s"`
		RPCTimeoutS      float64            `json:"rpc_timeout_s"`
		Overrides        map[string]float64 `json:"overrides"`
	} `json:"timing"`

	Fleet struct {
		MemoryMB  int64 `json:"memory_mb"`
		CPUMillis int64 `json:"cpu_millis"`
		DiskMB    int64 `json:"disk_mb"`
	} `json:"fleet"`

	Node struct {
		MemoryMB        int64   `json:"memory_mb"`
		CPUMillis       int64   `json:"cpu_millis"`
		DiskMB          int64   `json:"disk_mb"`
		GPUIndices      []int32 `json:"gpu_indices"`
		LocalCPUCeiling float64 `json:"local_cpu_ceiling"`
		LocalMemCeiling float64 `json:"local_mem_ceiling"`
	} `json:"node"`

	Classes map[string]struct {
		Guaranteed float64 `json:"guaranteed_share"`
		Max        float64 `json:"max_share"`
	} `json:"classes"`

	Backends []BackendConfig `json:"backends"`

	DefaultBackend string `json:"default_backend"`
	OwnerID        string `json:"owner_id"`
}

// BackendConfig declares one backend. The fields a backend does not use are
// absent from its own configuration rather than ignored, so a deployment that
// sets a Docker socket on a microVM backend sees it do nothing and can tell.
type BackendConfig struct {
	Type string `json:"type"`
	// Mode is "psrl" or "provider": who chooses the node. It is a deployment
	// property rather than a backend capability, so the same backend can run
	// both ways in one fleet and a scheduler ablation holds everything else fixed.
	Mode string `json:"mode"`

	// Docker.
	Socket     string `json:"socket"`
	APIVersion string `json:"api_version"`
	Runtime    string `json:"runtime"`

	// AgentEnv. Nodes are required in psrl mode, where this service chooses
	// which one to call; a gateway is required in provider mode, where AgentEnv
	// chooses. Scheduler is where a binding is registered so the provider's own
	// routing keeps working while this service places.
	Gateway   string `json:"gateway"`
	Scheduler string `json:"scheduler"`
	APIKey    string `json:"api_key"`
	Nodes     []struct {
		NodeID  string `json:"node_id"`
		Address string `json:"address"`
	} `json:"nodes"`
}

func main() {
	configPath := flag.String("config", "", "path to the service configuration")
	check := flag.Bool("preflight", false,
		"validate the configuration and every backend it declares, then exit without serving")
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	// Preflight is its own mode rather than a flag the service honours while
	// starting, so a deployment can be validated from a pipeline before anything
	// is listening: the exit code is the whole answer.
	if *check {
		if err := preflight(*configPath, log); err != nil {
			log.Error("sandboxd preflight failed", "error", err)
			os.Exit(1)
		}
		log.Info("sandboxd preflight passed")
		return
	}
	if err := run(*configPath, log); err != nil {
		log.Error("sandboxd stopped", "error", err)
		os.Exit(1)
	}
}

// preflight validates a deployment without serving it.
//
// It builds exactly what run would build, in the same order, so a pass here
// means the same configuration starts. Building is most of the check: the timing
// contract asserts its own orderings, the quota shares have to sum below one,
// and every backend is reached and refused if it cannot be driven.
func preflight(configPath string, log *slog.Logger) error {
	cfg, err := load(configPath)
	if err != nil {
		return err
	}
	spans, err := timing.New(
		seconds(cfg.Timing.EpisodeDeadlineS),
		seconds(cfg.Timing.NodeTTLS),
		seconds(cfg.Timing.RPCTimeoutS),
		overrides(cfg.Timing.Overrides),
	)
	if err != nil {
		return err
	}
	log.Info("timing contract resolved", "spans", spans.Spans())

	// buildBackends preflights each one as it is built, which is the expensive
	// part of the check and the part that reaches the node.
	backends, err := buildBackends(cfg, log)
	if err != nil {
		return err
	}
	if _, err := backend.NewRegistry(backends, cfg.DefaultBackend); err != nil {
		return err
	}
	if _, err := quota.New(quota.Config{
		Total: quota.Amount{
			MemoryMB: cfg.Fleet.MemoryMB, CPUMillis: cfg.Fleet.CPUMillis, DiskMB: cfg.Fleet.DiskMB,
			Sandboxes: 1 << 20,
		},
		Classes:  quotaClasses(cfg),
		LeaseTTL: spans.CapacityLeaseTTL(),
	}); err != nil {
		return err
	}
	// The node envelope and its windows are only meaningful where a runtime is
	// installed here, which is the same condition run uses to build the agent.
	local := localBackends(backends)
	if len(local) == 0 {
		log.Info("no local runtime is declared, so this process would serve control only")
		return nil
	}
	gate, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{
			MemoryMB: cfg.Node.MemoryMB, CPUMillis: cfg.Node.CPUMillis, DiskMB: cfg.Node.DiskMB,
		},
		Classes:         nodeClasses(cfg),
		GPUIndices:      cfg.Node.GPUIndices,
		LocalCPUCeiling: cfg.Node.LocalCPUCeiling,
		LocalMemCeiling: cfg.Node.LocalMemCeiling,
		LeaseTTL:        spans.CapacityLeaseTTL(),
	})
	if err != nil {
		return err
	}
	if _, err := node.NewLifecycle(cfg.NodeID, gate, node.Windows{
		PauseWindow:   spans.PauseWindow(),
		ReapWindow:    spans.ReapWindow(),
		Lifetime:      spans.Lifetime(),
		SweepInterval: spans.SweepInterval(),
	}); err != nil {
		return err
	}
	log.Info("deployment is servable",
		"node", cfg.NodeID, "local_backends", backendNames(local), "all_backends", backendNames(backends))
	return nil
}

func run(configPath string, log *slog.Logger) error {
	cfg, err := load(configPath)
	if err != nil {
		return err
	}

	spans, err := timing.New(
		seconds(cfg.Timing.EpisodeDeadlineS),
		seconds(cfg.Timing.NodeTTLS),
		seconds(cfg.Timing.RPCTimeoutS),
		overrides(cfg.Timing.Overrides),
	)
	if err != nil {
		// A configuration whose deadlines are in an impossible order fails here
		// rather than leaking a node slot per episode.
		return err
	}
	log.Info("sandboxd timing resolved", "spans", spans.Spans())

	backends, err := buildBackends(cfg, log)
	if err != nil {
		return err
	}
	registry, err := backend.NewRegistry(backends, cfg.DefaultBackend)
	if err != nil {
		return err
	}

	// Only the runtimes installed on this machine get a node agent. A provider
	// backend runs on the provider's own hosts, so admitting it against this
	// node's envelope would charge this machine for memory it never spends.
	local := localBackends(backends)

	fleet := monitor.New(spans.NodeTTL)

	var (
		agent *server.Node
		life  *node.Lifecycle
		nodes server.NodeClient
	)
	if len(local) > 0 {
		gate, err := node.NewAdmission(node.Config{
			Envelope: node.Resources{
				MemoryMB: cfg.Node.MemoryMB, CPUMillis: cfg.Node.CPUMillis, DiskMB: cfg.Node.DiskMB,
			},
			Classes:         nodeClasses(cfg),
			GPUIndices:      cfg.Node.GPUIndices,
			LocalCPUCeiling: cfg.Node.LocalCPUCeiling,
			LocalMemCeiling: cfg.Node.LocalMemCeiling,
			LeaseTTL:        spans.CapacityLeaseTTL(),
		})
		if err != nil {
			return err
		}
		life, err = node.NewLifecycle(cfg.NodeID, gate, node.Windows{
			PauseWindow:   spans.PauseWindow(),
			ReapWindow:    spans.ReapWindow(),
			Lifetime:      spans.Lifetime(),
			SweepInterval: spans.SweepInterval(),
		})
		if err != nil {
			return err
		}
		agent, err = server.NewNode(server.NodeConfig{
			// Every runtime installed here shares one admission ledger: a microVM and
			// a container on this host draw on the same memory, so accounting them
			// apart would admit twice what the machine has.
			NodeID: cfg.NodeID, Admission: gate, Lifecycle: life, Backends: local,
		})
		if err != nil {
			return err
		}
		fleet.Report(agent.View())
		nodes = server.NewLocalNodeClient(agent)
	}

	ledger, err := quota.New(quota.Config{
		Total: quota.Amount{
			MemoryMB: cfg.Fleet.MemoryMB, CPUMillis: cfg.Fleet.CPUMillis, DiskMB: cfg.Fleet.DiskMB,
			Sandboxes: 1 << 20,
		},
		Classes:  quotaClasses(cfg),
		LeaseTTL: spans.CapacityLeaseTTL(),
	})
	if err != nil {
		return err
	}
	place, err := placement.New(placement.Config{
		NodeTTL:        spans.NodeTTL,
		ReservationTTL: spans.ReservationTTL(),
		SweepInterval:  spans.SweepInterval(),
	}, fleet)
	if err != nil {
		return err
	}
	control, err := server.NewControl(server.ControlConfig{
		Registry: registry, Ledger: ledger, Placement: place, Monitor: fleet,
		Nodes:          nodes,
		AcquireTimeout: spans.AcquireTimeout(),
	})
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if agent != nil {
		life.Start(ctx)
		defer life.Stop()
		// The node republishes itself on the contract's cadence, so placement reads
		// a view that is never more than a report old.
		go republish(ctx, agent, fleet, spans.LoadReportInterval())
	}
	go sweepReservations(ctx, place, spans.SweepInterval())

	listener, err := listen(cfg.Listen)
	if err != nil {
		return err
	}
	log.Info("sandboxd listening", "address", cfg.Listen, "node", cfg.NodeID, "backends", backendNames(backends))
	return server.NewJSONListener(control, agent, log).Serve(ctx, listener)
}

func republish(ctx context.Context, agent *server.Node, fleet *monitor.Monitor, every time.Duration) {
	ticker := time.NewTicker(every)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			fleet.Report(agent.View())
		}
	}
}

func sweepReservations(ctx context.Context, place *placement.Service, every time.Duration) {
	ticker := time.NewTicker(every)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			place.Sweep()
		}
	}
}

func buildBackends(cfg Config, log *slog.Logger) ([]backend.Backend, error) {
	if len(cfg.Backends) == 0 {
		return nil, fmt.Errorf("a sandbox service needs at least one backend")
	}
	owner := cfg.OwnerID
	if owner == "" {
		owner = "sandboxd-" + cfg.NodeID
	}
	var built []backend.Backend
	for _, declared := range cfg.Backends {
		mode := backend.SchedulingMode(declared.Mode)
		if declared.Mode == "" {
			mode = backend.SchedulingPSRL
		}
		var (
			b   backend.Backend
			err error
		)
		switch declared.Type {
		case "agentenv":
			nodes := make([]agentenv.NodeAddress, 0, len(declared.Nodes))
			for _, node := range declared.Nodes {
				nodes = append(nodes, agentenv.NodeAddress{NodeID: node.NodeID, Address: node.Address})
			}
			b, err = agentenv.New(agentenv.Config{
				Gateway:   declared.Gateway,
				Nodes:     nodes,
				Scheduler: declared.Scheduler,
				APIKey:    declared.APIKey,
			}, mode)
		case "cubesandbox":
			b, err = cubesandbox.New(cubesandbox.Config{
				Gateway: declared.Gateway,
				APIKey:  declared.APIKey,
			})
		case "opensandbox":
			b, err = opensandbox.New(opensandbox.Config{
				Gateway: declared.Gateway,
				APIKey:  declared.APIKey,
			})
		case "docker":
			b, err = dockerbackend.New(dockerbackend.Config{
				Socket:     declared.Socket,
				APIVersion: declared.APIVersion,
				NodeID:     cfg.NodeID,
				OwnerID:    owner,
				Runtime:    declared.Runtime,
			}, mode)
		default:
			return nil, fmt.Errorf("sandbox backend type %q is not built into this binary", declared.Type)
		}
		if err != nil {
			return nil, err
		}
		// Preflighted through the interface rather than per type, so a backend
		// added later cannot be wired in without its refusal. At startup rather
		// than at the first rollout: a daemon that is not answering is a
		// configuration error, and discovering it inside an episode costs a sample.
		if checker, can := b.(backend.Preflighter); can {
			if err := checker.Preflight(context.Background()); err != nil {
				return nil, fmt.Errorf("backend %q preflight: %w", declared.Type, err)
			}
		}
		log.Info("sandbox backend ready", "type", declared.Type, "mode", mode)
		built = append(built, b)
	}
	return built, nil
}

// localBackends are the runtimes installed on this machine, which are the ones
// this node's admission can speak for. A provider backend places on the
// provider's own hosts, so this node has no envelope to charge it against.
func localBackends(backends []backend.Backend) []backend.Backend {
	local := make([]backend.Backend, 0, len(backends))
	for _, b := range backends {
		if b.Mode() == backend.SchedulingPSRL {
			local = append(local, b)
		}
	}
	return local
}

func backendNames(backends []backend.Backend) []string {
	names := make([]string, 0, len(backends))
	for _, b := range backends {
		names = append(names, b.Name())
	}
	return names
}

func nodeClasses(cfg Config) map[string]node.ClassShare {
	out := make(map[string]node.ClassShare, len(cfg.Classes))
	for name, share := range cfg.Classes {
		out[name] = node.ClassShare{Guaranteed: share.Guaranteed, Max: share.Max}
	}
	return out
}

func quotaClasses(cfg Config) map[string]quota.ClassShare {
	out := make(map[string]quota.ClassShare, len(cfg.Classes))
	for name, share := range cfg.Classes {
		out[name] = quota.ClassShare{Guaranteed: share.Guaranteed, Max: share.Max}
	}
	return out
}

func load(path string) (Config, error) {
	if path == "" {
		return Config{}, fmt.Errorf("sandboxd needs -config")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return Config{}, fmt.Errorf("reading %s: %w", path, err)
	}
	if cfg.NodeID == "" {
		host, err := os.Hostname()
		if err != nil {
			return Config{}, fmt.Errorf("no node_id configured and the hostname is unreadable: %w", err)
		}
		cfg.NodeID = host
	}
	if cfg.Listen == "" {
		cfg.Listen = "unix:///run/sandboxd.sock"
	}
	return cfg, nil
}

func listen(address string) (net.Listener, error) {
	if path, found := trimPrefix(address, "unix://"); found {
		// A stale socket from a killed process would make bind fail, and the
		// service would refuse to start for a file nobody is listening on.
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return nil, err
		}
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return nil, err
		}
		return net.Listen("unix", path)
	}
	return net.Listen("tcp", address)
}

func trimPrefix(value, prefix string) (string, bool) {
	if len(value) >= len(prefix) && value[:len(prefix)] == prefix {
		return value[len(prefix):], true
	}
	return "", false
}

func seconds(value float64) time.Duration {
	return time.Duration(value * float64(time.Second))
}

func overrides(raw map[string]float64) map[string]time.Duration {
	if len(raw) == 0 {
		return nil
	}
	out := make(map[string]time.Duration, len(raw))
	for name, value := range raw {
		out[name] = seconds(value)
	}
	return out
}
