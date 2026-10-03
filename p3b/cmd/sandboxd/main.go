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
	"psrl.dev/sandboxd/internal/pressure"
	"psrl.dev/sandboxd/internal/provision"
	"psrl.dev/sandboxd/internal/quota"
	"psrl.dev/sandboxd/internal/server"
	"psrl.dev/sandboxd/internal/timing"
)

// Config is the whole deployment, stated once.
//
// An operator declares intent -- how long an episode takes, what each class is
// owed, which backends exist -- and every deadline follows from it. A knob that
// could be derived is not a knob.
//
// Role controls which planes this process runs:
//
//	"" or "combined"  — control plane + node plane on one process (default, backward-compatible)
//	"control"         — cluster-level control plane only; NodeListen is unused; FleetNodes names remote nodes
//	"node"            — node plane only; Listen serves the node-plane protocol; control fields are unused
//
// In "control" role the process connects to the remote nodes listed in FleetNodes.
// In "node" role the process serves the node-plane on NodeListen and the SDK can
// reach it through a co-located control process.
type Config struct {
	// Role is the deployment shape this process takes.
	Role string `json:"role"`

	// Listen is where the SDK reaches the control plane.
	Listen string `json:"listen"`
	// NodeListen is where the node-plane listens. Required for "node" role;
	// ignored in "control" role; defaults to listen+"-node" in "combined" if set.
	NodeListen string `json:"node_listen"`

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
		MemoryMB          int64   `json:"memory_mb"`
		CPUMillis         int64   `json:"cpu_millis"`
		DiskMB            int64   `json:"disk_mb"`
		GPUIndices        []int32 `json:"gpu_indices"`
		LocalCPUCeiling   float64 `json:"local_cpu_ceiling"`
		LocalMemCeiling   float64 `json:"local_mem_ceiling"`
		// Overcommit is the largest multiple of the declared envelope the node may
		// grant when measurement says reservations are overstated. Zero and one both
		// mean off. Only memory expands; the expansion withdraws proportionally as
		// measured utilization rises toward UtilizationTarget.
		Overcommit        float64 `json:"overcommit"`
		// UtilizationTarget is the measured memory fraction at which overcommit is
		// fully withdrawn. Must be set and below LocalMemCeiling when Overcommit > 1.
		UtilizationTarget float64 `json:"utilization_target"`
	} `json:"node"`

	Classes map[string]struct {
		Guaranteed float64 `json:"guaranteed_share"`
		Max        float64 `json:"max_share"`
	} `json:"classes"`

	Backends []BackendConfig `json:"backends"`

	// FleetNodes lists the remote node agents a "control" role process reaches.
	// Each entry names a node and the address of its node-plane listener.
	// Ignored in "node" and "combined" roles.
	FleetNodes []struct {
		NodeID  string `json:"node_id"`
		Address string `json:"address"`
	} `json:"fleet_nodes"`

	DefaultBackend string `json:"default_backend"`
	OwnerID        string `json:"owner_id"`
}

// BackendConfig declares one backend. The fields a backend does not use are
// absent from its own configuration rather than ignored, so a deployment that
// sets a Docker socket on a microVM backend sees it do nothing and can tell.
type BackendConfig struct {
	Type string `json:"type"`
	// Mode is "direct" or "provider": who chooses the node and who drives it. It is a deployment
	// property rather than a backend capability, so the same backend can run
	// both ways in one fleet and a scheduler ablation holds everything else fixed.
	Mode string `json:"mode"`

	// Container runtime, driven directly. Shared by the docker backend and by
	// any backend running in direct mode, because both drive the same daemon.
	Socket               string `json:"socket"`
	APIVersion           string `json:"api_version"`
	Runtime              string `json:"runtime"`
	NetworkMode          string `json:"network_mode"`
	MaxCreateConcurrency int    `json:"max_create_concurrency"`

	// OpenSandbox in direct mode. The agent image carries execd, bootstrap.sh,
	// and bubblewrap; it is staged once at startup into stage_dir and then
	// bind-mounted into every sandbox, so the per-sandbox cost is a mount rather
	// than a copy. The port range bounds where the agent is published when a
	// mapping is needed at all -- under network_mode "host" there is none.
	ExecdImage string `json:"execd_image"`
	StageDir   string `json:"stage_dir"`
	PortMin    int    `json:"port_min"`
	PortMax    int    `json:"port_max"`

	// AgentEnv and CubeSandbox. Nodes are required in direct mode, where this
	// service chooses which one to call; a gateway is required in provider mode,
	// where the provider chooses. Scheduler is where a binding is registered so
	// the provider's own routing keeps working while this service places.
	Gateway   string `json:"gateway"`
	Scheduler string `json:"scheduler"`
	APIKey    string `json:"api_key"`
	Nodes     []struct {
		NodeID  string `json:"node_id"`
		Address string `json:"address"`
	} `json:"nodes"`

	// WarmPool, for docker direct mode. A zero Size disables the pool.
	WarmPool struct {
		Image           string  `json:"image"`
		Size            int     `json:"size"`
		MemoryMB        int64   `json:"memory_mb"`
		CPUCount        float64 `json:"cpu_count"`
		EntryTTLS       float64 `json:"entry_ttl_s"`
		RefillIntervalS float64 `json:"refill_interval_s"`
	} `json:"warm_pool"`
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
		Overcommit:        cfg.Node.Overcommit,
		UtilizationTarget: cfg.Node.UtilizationTarget,
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
		return err
	}
	log.Info("sandboxd timing resolved", "spans", spans.Spans())

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	switch cfg.Role {
	case "node":
		return runNode(ctx, cfg, spans, log)
	case "control":
		return runControl(ctx, cfg, spans, log)
	default:
		// "" or "combined": both planes in one process — the backward-compatible shape.
		return runCombined(ctx, cfg, spans, log)
	}
}

// runNode starts a node-only process. It builds the local backends and admission,
// then serves the node-plane protocol on NodeListen. No control-plane gRPC is
// started; the SDK does not connect here.
func runNode(ctx context.Context, cfg Config, spans timing.Contract, log *slog.Logger) error {
	if cfg.NodeListen == "" {
		return fmt.Errorf("role \"node\" requires node_listen to be set")
	}
	backends, err := buildBackends(cfg, log)
	if err != nil {
		return err
	}
	local := localBackends(backends)
	if len(local) == 0 {
		return fmt.Errorf("role \"node\" requires at least one direct backend")
	}
	gate, err := node.NewAdmission(node.Config{
		Envelope:        node.Resources{MemoryMB: cfg.Node.MemoryMB, CPUMillis: cfg.Node.CPUMillis, DiskMB: cfg.Node.DiskMB},
		Classes:         nodeClasses(cfg),
		GPUIndices:      cfg.Node.GPUIndices,
		LocalCPUCeiling: cfg.Node.LocalCPUCeiling,
		LocalMemCeiling: cfg.Node.LocalMemCeiling,
		Overcommit:        cfg.Node.Overcommit,
		UtilizationTarget: cfg.Node.UtilizationTarget,
		LeaseTTL:        spans.CapacityLeaseTTL(),
	})
	if err != nil {
		return err
	}
	life, err := node.NewLifecycle(cfg.NodeID, gate, node.Windows{
		PauseWindow:   spans.PauseWindow(),
		ReapWindow:    spans.ReapWindow(),
		Lifetime:      spans.Lifetime(),
		SweepInterval: spans.SweepInterval(),
	})
	if err != nil {
		return err
	}
	agent, err := server.NewNode(server.NodeConfig{
		NodeID: cfg.NodeID, Admission: gate, Lifecycle: life, Backends: local,
		Pressure: nodePressure(log),
	})
	if err != nil {
		return err
	}
	life.Start(ctx)
	defer life.Stop()
	startBackends(ctx, local, log)
	defer stopBackends(local)

	listener, err := listen(cfg.NodeListen)
	if err != nil {
		return err
	}
	log.Info("sandboxd node plane listening", "address", cfg.NodeListen, "node", cfg.NodeID, "backends", backendNames(local))
	return server.NewNodeListener(agent, log).Serve(ctx, listener)
}

// runControl starts a control-only process. It connects to all fleet nodes from
// FleetNodes, builds a FleetNodeClient, starts the fleet poller, and serves the
// SDK socket as normal.
func runControl(ctx context.Context, cfg Config, spans timing.Contract, log *slog.Logger) error {
	if len(cfg.FleetNodes) == 0 {
		return fmt.Errorf("role \"control\" requires fleet_nodes to be set")
	}
	// Control-only processes carry no local backends — they route to fleet nodes.
	// The registry may be empty if all backends are on the remote nodes, but quota
	// and placement still need to be wired.
	backends, err := buildBackends(cfg, log)
	if err != nil && len(cfg.Backends) > 0 {
		// If backends are declared they must build. If none are declared, this is fine.
		return err
	}
	if backends == nil {
		backends = []backend.Backend{}
	}
	registry, err := backend.NewRegistry(backends, cfg.DefaultBackend)
	if err != nil {
		return err
	}

	fleet := monitor.New(spans.NodeTTL)

	rpcTimeout := spans.RPCTimeout
	remotes := make(map[string]*server.RemoteNodeClient, len(cfg.FleetNodes))
	for _, n := range cfg.FleetNodes {
		if n.NodeID == "" || n.Address == "" {
			return fmt.Errorf("each fleet_nodes entry needs both node_id and address")
		}
		remotes[n.NodeID] = server.NewRemoteNodeClient(n.Address, rpcTimeout)
		log.Info("fleet node registered", "node", n.NodeID, "address", n.Address)
	}
	fleetClient := server.NewFleetNodeClient(remotes)
	go fleetClient.PollFleet(ctx, fleet, spans.LoadReportInterval())

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
		Nodes:          fleetClient,
		AcquireTimeout: spans.AcquireTimeout(),
	})
	if err != nil {
		return err
	}

	go sweepReservations(ctx, place, spans.SweepInterval())

	listener, err := listen(cfg.Listen)
	if err != nil {
		return err
	}
	log.Info("sandboxd control plane listening", "address", cfg.Listen, "fleet_nodes", len(cfg.FleetNodes))
	// nil node: a control-only process forwards exec via FleetNodeClient, not a
	// local node. JSONListener.exec routes through control.nodes.ExecOn.
	return server.NewJSONListener(control, nil, log).Serve(ctx, listener)
}

// runCombined is the original single-process shape: control and node in one
// binary, reachable over one socket.
func runCombined(ctx context.Context, cfg Config, spans timing.Contract, log *slog.Logger) error {
	backends, err := buildBackends(cfg, log)
	if err != nil {
		return err
	}
	registry, err := backend.NewRegistry(backends, cfg.DefaultBackend)
	if err != nil {
		return err
	}

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
			Overcommit:        cfg.Node.Overcommit,
			UtilizationTarget: cfg.Node.UtilizationTarget,
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
			NodeID: cfg.NodeID, Admission: gate, Lifecycle: life, Backends: local,
			Pressure: nodePressure(log),
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

	if agent != nil {
		life.Start(ctx)
		defer life.Stop()
		// A backend with background work of its own -- the docker warm pool is the
		// one that has any -- is started here and drained on the way out. It is
		// reached through an interface rather than a type switch so a backend added
		// later gets the same treatment without this file naming it.
		startBackends(ctx, local, log)
		defer stopBackends(local)
		go republish(ctx, agent, fleet, spans.LoadReportInterval())

		// If node_listen is set in combined mode, also serve the node-plane
		// protocol so that a second control process (e.g. a dedicated gateway) can
		// reach this node without a local socket.
		if cfg.NodeListen != "" {
			nl, err := listen(cfg.NodeListen)
			if err != nil {
				return err
			}
			log.Info("sandboxd node plane also listening", "address", cfg.NodeListen)
			go func() { _ = server.NewNodeListener(agent, log).Serve(ctx, nl) }()
		}
	}
	go sweepReservations(ctx, place, spans.SweepInterval())

	listener, err := listen(cfg.Listen)
	if err != nil {
		return err
	}
	log.Info("sandboxd listening", "address", cfg.Listen, "node", cfg.NodeID, "backends", backendNames(backends))
	return server.NewJSONListener(control, agent, log).Serve(ctx, listener)
}

func republish(ctx context.Context, agent *server.Node, fleet *monitor.Monitor, every time.Duration) {	ticker := time.NewTicker(every)
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
			mode = backend.SchedulingDirect
		}

		// Provisioned before built, because a backend whose service is absent
		// fails to construct with the runtime's own error, and that error names a
		// socket or a status code rather than the thing an operator has to install.
		// This check names it.
		if result := provisionCheck(declared, mode); result.Err != nil {
			return nil, result.Err
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
			csNodes := make([]cubesandbox.NodeAddress, 0, len(declared.Nodes))
			for _, node := range declared.Nodes {
				csNodes = append(csNodes, cubesandbox.NodeAddress{NodeID: node.NodeID, Address: node.Address})
			}
			b, err = cubesandbox.New(cubesandbox.Config{
				Gateway:              declared.Gateway,
				Nodes:                csNodes,
				APIKey:               declared.APIKey,
				OwnerID:              owner,
				MaxCreateConcurrency: declared.MaxCreateConcurrency,
			}, mode)
		case "opensandbox":
			// Direct mode drives this machine's container runtime and stages the
			// agent into each sandbox, so it takes the runtime configuration rather
			// than a list of node addresses: one daemon is one node, the same shape
			// the docker backend has. Provider mode takes the gateway instead.
			b, err = opensandbox.New(opensandbox.Config{
				Gateway:              declared.Gateway,
				APIKey:               declared.APIKey,
				ExecdImage:           declared.ExecdImage,
				Socket:               declared.Socket,
				APIVersion:           declared.APIVersion,
				StageDir:             declared.StageDir,
				NodeID:               cfg.NodeID,
				OwnerID:              owner,
				Runtime:              declared.Runtime,
				NetworkMode:          declared.NetworkMode,
				MaxCreateConcurrency: declared.MaxCreateConcurrency,
				PortMin:              declared.PortMin,
				PortMax:              declared.PortMax,
			}, mode)
		case "docker":
			b, err = dockerbackend.New(dockerbackend.Config{
				Socket:               declared.Socket,
				APIVersion:           declared.APIVersion,
				NodeID:               cfg.NodeID,
				OwnerID:              owner,
				Runtime:              declared.Runtime,
				NetworkMode:          declared.NetworkMode,
				MaxCreateConcurrency: declared.MaxCreateConcurrency,
				WarmPool: dockerbackend.WarmPoolConfig{
					Image:          declared.WarmPool.Image,
					Size:           declared.WarmPool.Size,
					MemoryMB:       declared.WarmPool.MemoryMB,
					CPUCount:       declared.WarmPool.CPUCount,
					EntryTTL:       seconds(declared.WarmPool.EntryTTLS),
					RefillInterval: seconds(declared.WarmPool.RefillIntervalS),
				},
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

// provisionCheck confirms the external software a declared backend needs is
// present, before anything tries to use it.
//
// It never installs. A backend whose service is absent fails to construct
// anyway, but with the runtime's own error -- a refused socket, an HTTP status --
// which says what failed and not what to do about it. These checks name the
// missing thing and the command that provides it.
//
// The docker backend is absent from this switch because it needs nothing beyond
// the daemon, and its own Preflight already proves that.
func provisionCheck(declared BackendConfig, mode backend.SchedulingMode) provision.Result {
	ctx := context.Background()
	direct := mode == backend.SchedulingDirect

	switch declared.Type {
	case "opensandbox":
		if direct {
			// No OpenSandbox control plane is deployed in this shape: the runtime,
			// the agent image, and the stage directory are the whole dependency.
			return provision.CheckOpenSandboxDirect(ctx, declared.Socket, declared.ExecdImage, declared.StageDir)
		}
		return provision.CheckOpenSandboxGateway(ctx, declared.Gateway, declared.APIKey)

	case "agentenv":
		if direct {
			for _, node := range declared.Nodes {
				if result := provision.CheckAgentEnvNode(ctx, node.NodeID, node.Address, declared.APIKey); result.Err != nil {
					return result
				}
			}
			return provision.Result{Name: "agentenv nodes"}
		}
		return provision.CheckAgentEnvGateway(ctx, declared.Gateway, declared.APIKey)

	case "cubesandbox":
		if direct {
			for _, node := range declared.Nodes {
				if result := provision.CheckCubeSandboxNode(ctx, node.NodeID, node.Address, declared.APIKey); result.Err != nil {
					return result
				}
			}
			return provision.Result{Name: "cubesandbox nodes"}
		}
		return provision.CheckCubeSandboxGateway(ctx, declared.Gateway, declared.APIKey)
	}
	return provision.Result{Name: declared.Type}
}

// localBackends are the runtimes installed on this machine, which are the ones
// this node's admission can speak for. A provider backend places on the
// provider's own hosts, so this node has no envelope to charge it against.
func localBackends(backends []backend.Backend) []backend.Backend {
	local := make([]backend.Backend, 0, len(backends))
	for _, b := range backends {
		if b.Mode() == backend.SchedulingDirect {
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

// startBackends begins any background work a backend owns.
//
// Only a backend that has something to run implements this, so the assertion is
// an interface check rather than a flag: a backend with no pool and no loop is
// simply skipped.
func startBackends(ctx context.Context, backends []backend.Backend, log *slog.Logger) {
	for _, b := range backends {
		starter, can := b.(interface{ Start(context.Context) })
		if !can {
			continue
		}
		starter.Start(ctx)
		log.Info("backend background work started", "backend", b.Name())
	}
}

// stopBackends drains what startBackends began.
//
// The context is fresh rather than the service's, because the service's is
// already cancelled by the time a deferred stop runs and a drain that needs to
// reach the daemon would be cancelled before it could.
func stopBackends(backends []backend.Backend) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	for _, b := range backends {
		if stopper, can := b.(interface{ Stop(context.Context) }); can {
			stopper.Stop(ctx)
		}
	}
}

// nodePressure returns the Pressure function this node agent uses to read
// actual machine utilization from the cgroup hierarchy.
//
// When cgroup v2 is not available — a development machine, a container with a
// different hierarchy, or a host that did not configure it — the function falls
// back to the zero reading so the service starts rather than refusing. The log
// line is the only signal an operator has that the ceiling configuration is not
// being enforced.
func nodePressure(log *slog.Logger) func() node.Pressure {
	reader, err := pressure.NewReader()
	if err != nil {
		log.Warn("cgroup v2 pressure reading unavailable; utilisation ceilings will not be enforced",
			"reason", err)
		return func() node.Pressure { return node.Pressure{} }
	}
	return reader.Read
}
