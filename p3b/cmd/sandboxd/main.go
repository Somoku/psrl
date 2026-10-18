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
	"psrl.dev/sandboxd/internal/cgroup"
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
		NodeTTLS    float64 `json:"node_ttl_s"`
		RPCTimeoutS float64 `json:"rpc_timeout_s"`
	} `json:"timing"`

	// Fleet is the ceiling the cross-backend ledger admits against: the sum of
	// what the nodes in FleetNodes will grant, which is each node's declared
	// envelope times its overcommit. Stating the nodes' declared envelopes alone
	// understates it by the overcommit factor, and then the ledger rather than
	// the machines is what a burst queues behind.
	//
	// Over-declaring is not a way to raise the fleet's real capacity: a node
	// still refuses once its measured pressure crosses its ceiling, so the cost
	// of a total above what the nodes will grant is a round trip, not an OOM.
	// It is declared rather than derived from the nodes' reports because a
	// control plane that starts empty must admit against a fleet it cannot yet
	// see.
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
		// Overcommit is the largest multiple of the declared envelope the node may
		// grant when measurement says reservations are overstated. Zero and one both
		// mean off. Only memory expands; the expansion withdraws proportionally as
		// measured utilization rises toward UtilizationTarget.
		Overcommit float64 `json:"overcommit"`
		// UtilizationTarget is the measured memory fraction at which overcommit is
		// fully withdrawn. Must be set and below LocalMemCeiling when Overcommit > 1.
		UtilizationTarget float64 `json:"utilization_target"`
		// CgroupParent is the subtree Docker creates sandboxes into and the sampler
		// measures. Empty takes "psrl-sandboxes". The service creates it, bounds it,
		// and moves itself out of it before any container is created.
		//
		// The ceilings above are fractions of *this* subtree, not of the host or of
		// the pod this service runs in: measured on one of these nodes, a ceiling
		// applied to the pod was mostly about a trainer's page cache.
		CgroupParent string `json:"cgroup_parent"`
		// AdmissionReserve is the share of the envelope no running sandbox may spend, kept
		// so a wave that has not arrived yet has somewhere to land. Without it the first
		// wave takes the machine and grading -- or the next batch -- cannot be admitted
		// until an episode finishes. Zero disables it; unset takes 10%.
		AdmissionReserve float64 `json:"admission_reserve"`
		// SampleIntervalS is how often the domain is measured. Zero takes 5s. The
		// interval is also the window a CPU rate is averaged over.
		SampleIntervalS float64 `json:"sample_interval_s"`
		// SandboxMemorySafety is the share of the ancestor's memory limit kept free for
		// page cache growth and work that is not a sandbox. Zero takes 5%.
		SandboxMemorySafety float64 `json:"sandbox_memory_safety"`
		// Elastic prices a sandbox from what its class actually costs, and asks idle sandboxes
		// for their reclaimable memory when the node is near its budget. Disabled charges each
		// sandbox its declared maximum, which is the shape every deployment had before.
		Elastic struct {
			Enabled bool `json:"enabled"`
			// MinReserveMB is what a sandbox is charged for memory before its class has
			// any history; MinReserveCPUMillis is the same for CPU, in millicores.
			MinReserveMB        int64   `json:"min_reserve_mb"`
			MinReserveCPUMillis int64   `json:"min_reserve_cpu_millis"`
			Percentile          float64 `json:"percentile"`
			GrowthMargin        float64 `json:"growth_margin"`
			ShrinkAfter         int     `json:"shrink_after"`
			ShrinkStep          float64 `json:"shrink_step"`
			ShrinkAbove         float64 `json:"shrink_above"`
			RestoreBelow        float64 `json:"restore_below"`
			ReleaseMarginMB     int64   `json:"release_margin_mb"`
			MinLimitMB          int64   `json:"min_limit_mb"`
			// ReleaseMarginCPUMillis is what an idle sandbox's quota is left above what it
			// measured using, and MinCPUMillis is the lowest it may be throttled to.
			ReleaseMarginCPUMillis int64 `json:"release_margin_cpu_millis"`
			MinCPUMillis           int64 `json:"min_cpu_millis"`
			// GrowthHorizonS is how far ahead admission reserves a class's observed growth.
			GrowthHorizonS  float64 `json:"growth_horizon_s"`
			SampleIntervalS float64 `json:"sample_interval_s"`
			// StartLimitMB is the memory bound a sandbox is created under, and the level
			// its bound is never lowered below while it lives. Zero creates it at the
			// declared maximum, which is the shape this had before.
			StartLimitMB int64 `json:"start_limit_mb"`
			// CeilingLimitMB is the highest a sandbox's bound may be raised to. Zero uses
			// what the sandbox declared.
			CeilingLimitMB int64 `json:"ceiling_limit_mb"`
			// GrowAbove is the fraction of its bound a sandbox reaches before the bound is
			// raised, GrowFactor by how much, and EmergencyGrowFactor by how much on an
			// OOM. ShrinkBelow is the fraction it must fall under, ShrinkDwellS for how
			// long, and ShrinkFactor by how much it then comes down.
			GrowAbove            float64 `json:"grow_above"`
			GrowFactor           float64 `json:"grow_factor"`
			EmergencyGrowFactor  float64 `json:"emergency_grow_factor"`
			ShrinkBelow          float64 `json:"shrink_below"`
			ShrinkDwellS         float64 `json:"shrink_dwell_s"`
			ShrinkFactor         float64 `json:"shrink_factor"`
			SafetyFraction       float64 `json:"safety_fraction"`
			LimitSampleIntervalS float64 `json:"limit_sample_interval_s"`
		} `json:"elastic"`
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

// WarmPoolDeclared is a deployment's warm-pool declaration.
//
// The template fields are what every entry is built from, so they must describe the
// containers this deployment actually creates. They are compared exactly, because a
// label or a limit written at create cannot be changed afterwards -- with one exception:
// lifetime_s is the longest lifetime the deployment runs, and any request at or below it
// is served, since a lifetime is a backstop rather than a charge. State zero only if the
// callers state none too.
type WarmPoolDeclared struct {
	Image           string            `json:"image"`
	Size            int               `json:"size"`
	MemoryMB        int64             `json:"memory_mb"`
	CPUCount        float64           `json:"cpu_count"`
	DiskMB          int64             `json:"disk_mb"`
	Env             map[string]string `json:"env"`
	Workdir         string            `json:"workdir"`
	ResourceClass   string            `json:"resource_class"`
	ExecMode        string            `json:"exec_mode"`
	LifetimeS       float64           `json:"lifetime_s"`
	EntryTTLS       float64           `json:"entry_ttl_s"`
	RefillIntervalS float64           `json:"refill_interval_s"`
	// PauseAbove stops refill at this measured memory fraction; ReclaimAbove also
	// destroys idle entries. Zero takes 0.9 for both.
	PauseAbove   float64 `json:"pause_above"`
	ReclaimAbove float64 `json:"reclaim_above"`
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
	Socket      string `json:"socket"`
	APIVersion  string `json:"api_version"`
	Runtime     string `json:"runtime"`
	NetworkMode string `json:"network_mode"`
	// Bound on concurrent creates, because a daemon serializes parts of container
	// setup in the kernel and an unbounded burst turns latency into timeouts rather
	// than throughput. Zero leaves it unbounded, which is only right on a small
	// node: a many-core machine reports a GOMAXPROCS far above what the create path
	// can schedule at once, so size this from the daemon instead. Sixteen is the
	// value the OpenSandbox deployment documents. Applies to every backend that
	// drives a daemon: docker, opensandbox, and cubesandbox.
	MaxCreateConcurrency int `json:"max_create_concurrency"`
	// MountAllowlist is the host trees this node will share with a sandbox. Empty
	// means it shares none, and a spec that asks for a mount is refused rather than
	// created without it. A host mount is the one request in the contract that is
	// about the machine, so the node states what it is willing to give instead of
	// honouring any path a caller names.
	MountAllowlist []string `json:"mount_allowlist"`

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

	// WarmPool pre-builds sandboxes for a direct-mode backend, so a matching
	// create pays neither the container build nor (for opensandbox) the agent
	// readiness wait. A zero Size disables it.
	//
	// For opensandbox the size is also a standing claim on the port range, because
	// every entry holds a published host port: the backend refuses a pool above a
	// quarter of port_max-port_min rather than letting it starve cold creates.
	WarmPool WarmPoolDeclared `json:"warm_pool"`
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
		seconds(cfg.Timing.NodeTTLS),
		seconds(cfg.Timing.RPCTimeoutS),
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
		LeaseTTL: timing.OwnerLease,
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
	// The domain is resolved without creating anything: preflight runs against a
	// deployment that may not have started yet, and a check with side effects would
	// leave a subtree behind for a service that never served.
	parent := cfg.Node.CgroupParent
	if parent == "" {
		parent = cgroup.SandboxParent
	}
	domain, err := cgroup.Resolve()
	if err != nil {
		return fmt.Errorf("sandbox domain: %w", err)
	}
	budget, err := domain.SetupReadOnly(parent, cgroup.Request{
		MemoryMB:   cfg.Node.MemoryMB,
		Overcommit: cfg.Node.Overcommit,
		CPUMillis:  cfg.Node.CPUMillis,
		Safety:     cfg.Node.SandboxMemorySafety,
	})
	if err != nil {
		return fmt.Errorf("sandbox domain: %w", err)
	}
	log.Info("sandbox domain resolves",
		"domain", domain.String(), "subtree", parent,
		"limit_mib", budget.SubtreeLimitBytes/(1024*1024),
		"declared_mib", budget.DeclaredMB, "requested_mib", budget.RequestedMB,
		"ancestor_limit_mib", budget.AncestorLimitBytes/(1024*1024),
		"unreclaimable_mib", budget.UnreclaimableBytes/(1024*1024))
	if budget.Capped() {
		log.Warn("the sandbox subtree would be smaller than the declared envelope",
			"declared_mib", budget.RequestedMB, "envelope_mib", budget.DeclaredMB)
	}
	gate, err := node.NewAdmission(node.Config{
		Envelope:          nodeEnvelope(cfg, budget),
		Classes:           nodeClasses(cfg),
		GPUIndices:        cfg.Node.GPUIndices,
		LocalCPUCeiling:   cfg.Node.LocalCPUCeiling,
		LocalMemCeiling:   cfg.Node.LocalMemCeiling,
		Overcommit:        cfg.Node.Overcommit,
		UtilizationTarget: cfg.Node.UtilizationTarget,
		// A share of the envelope no sandbox may spend, so a later wave always has
		// somewhere to land.
		AdmissionReserve: cfg.Node.AdmissionReserve,
		LeaseTTL:         timing.OwnerLease,
	})
	if err != nil {
		return err
	}
	if _, err := node.NewLifecycle(cfg.NodeID, gate, timing.SweepInterval); err != nil {
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
		seconds(cfg.Timing.NodeTTLS),
		seconds(cfg.Timing.RPCTimeoutS),
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
	prepared, err := nodeDomain(ctx, cfg, log)
	if err != nil {
		return err
	}
	pressureFn := prepared.pressure
	gate, err := node.NewAdmission(node.Config{
		Envelope:          nodeEnvelope(cfg, prepared.budget),
		Classes:           nodeClasses(cfg),
		GPUIndices:        cfg.Node.GPUIndices,
		LocalCPUCeiling:   cfg.Node.LocalCPUCeiling,
		LocalMemCeiling:   cfg.Node.LocalMemCeiling,
		Overcommit:        cfg.Node.Overcommit,
		UtilizationTarget: cfg.Node.UtilizationTarget,
		// The node reclaims a running sandbox once its owner has been silent for
		// OwnerSilenceTTL, so this gate tracks the heartbeat rather than the
		// control plane's quota lease.
		LeaseTTL: timing.OwnerSilenceTTL,
	})
	if err != nil {
		return err
	}
	// Seed the gate with the sampler's first reading. Nothing else touches it until the
	// first admit or the first report, and until then an unseeded gate is an "unusable
	// sample": admission would refuse the node's first sandbox, and the warm pool would
	// spend its first refill doing nothing.
	gate.SetPressure(pressureFn())
	life, err := node.NewLifecycle(cfg.NodeID, gate, timing.SweepInterval)
	if err != nil {
		return err
	}
	// The sweep reads owner liveness from this gate: a caller that stops
	// heartbeating has its sandboxes reclaimed as orphans.
	life.SetOwnerLiveness(gate.OwnerAlive)
	if cfg.Node.Elastic.Enabled {
		log.Info("elastic admission enabled",
			"min_reserve_mb", cfg.Node.Elastic.MinReserveMB,
			"growth_margin", cfg.Node.Elastic.GrowthMargin,
			"shrink_above", cfg.Node.Elastic.ShrinkAbove,
			"restore_below", cfg.Node.Elastic.RestoreBelow)
	}
	agent, err := server.NewNode(server.NodeConfig{
		NodeID: cfg.NodeID, Admission: gate, Lifecycle: life, Backends: local,
		Pressure:        pressureFn,
		Elastic:         elasticConfig(cfg),
		Domain:          prepared.domain,
		SandboxParent:   prepared.parent,
		SandboxBudgetMB: prepared.budget.SubtreeLimitBytes / (1024 * 1024),
	})
	if err != nil {
		return err
	}
	// Rebuild this node's ledger from the daemon before it serves. A node that
	// cannot enumerate what it already runs does not know how much of its
	// envelope is spent, and admitting against that would hand out memory a live
	// sandbox is using.
	reconciled, err := life.Reconcile(ctx, local)
	if err != nil {
		return fmt.Errorf("node %q cannot account for its own runtime: %w", cfg.NodeID, err)
	}
	log.Info("node reconciled its runtime at startup",
		"node", cfg.NodeID, "adopted", reconciled.Adopted,
		"reclaimed", reconciled.Reclaimed, "failures", reconciled.Failures)
	life.Start(ctx)
	defer life.Stop()
	startBackends(ctx, local, gate, log)
	defer stopBackends(local)
	go elasticLoop(ctx, agent, seconds(cfg.Node.Elastic.SampleIntervalS), log)
	// The bounds half runs on its own, much shorter cadence: it is the one a burst
	// has to be visible to.
	go limitLoop(ctx, agent, seconds(cfg.Node.Elastic.LimitSampleIntervalS), log)

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
	// A create can outlast one RPC budget because the node may pull an image
	// before it answers, so the node-plane create bound carries the pull budget.
	createTimeout := dockerbackend.DefaultPullTimeout
	remotes := make(map[string]*server.RemoteNodeClient, len(cfg.FleetNodes))
	for _, n := range cfg.FleetNodes {
		if n.NodeID == "" || n.Address == "" {
			return fmt.Errorf("each fleet_nodes entry needs both node_id and address")
		}
		remotes[n.NodeID] = server.NewRemoteNodeClient(n.Address, rpcTimeout, createTimeout)
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
		LeaseTTL: timing.OwnerLease,
	})
	if err != nil {
		return err
	}
	place, err := placement.New(placement.Config{
		NodeTTL:        spans.NodeTTL,
		ReservationTTL: timing.ReservationTTL,
		SweepInterval:  timing.SweepInterval,
	}, fleet)
	if err != nil {
		return err
	}
	control, err := server.NewControl(server.ControlConfig{
		Registry: registry, Ledger: ledger, Placement: place, Monitor: fleet,
		Nodes: fleetClient, Logger: log,
	})
	if err != nil {
		return err
	}

	go sweepReservations(ctx, place, timing.SweepInterval)
	// The fleet's grants are repriced between reports, because that is when the nodes'
	// published prices move. Doing it on the admission path would put a walk of every
	// live grant in front of the request that is waiting for it.
	go repriceGrants(ctx, control, spans.LoadReportInterval())

	listener, err := listen(cfg.Listen)
	if err != nil {
		return err
	}
	log.Info("sandboxd control plane listening", "address", cfg.Listen, "fleet_nodes", len(cfg.FleetNodes))
	// nil node: a control-only process forwards exec via FleetNodeClient, not a
	// local node. The data-plane resolver lets a create reply name the node a
	// caller should send commands to, so they skip this process entirely.
	return server.NewJSONListener(control, nil, log).
		WithDataPlane(fleetClient.DataPlaneFor).
		Serve(ctx, listener)
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
		gate  *node.Admission
	)
	if len(local) > 0 {
		prepared, err := nodeDomain(ctx, cfg, log)
		if err != nil {
			return err
		}
		pressureFn := prepared.pressure
		gate, err = node.NewAdmission(node.Config{
			Envelope:          nodeEnvelope(cfg, prepared.budget),
			Classes:           nodeClasses(cfg),
			GPUIndices:        cfg.Node.GPUIndices,
			LocalCPUCeiling:   cfg.Node.LocalCPUCeiling,
			LocalMemCeiling:   cfg.Node.LocalMemCeiling,
			Overcommit:        cfg.Node.Overcommit,
			UtilizationTarget: cfg.Node.UtilizationTarget,
			// The node reclaims a running sandbox once its owner has been silent
			// for OwnerSilenceTTL, so this gate tracks the heartbeat rather than
			// the control plane's quota lease.
			LeaseTTL: timing.OwnerSilenceTTL,
		})
		if err != nil {
			return err
		}
		gate.SetPressure(pressureFn())
		life, err = node.NewLifecycle(cfg.NodeID, gate, timing.SweepInterval)
		if err != nil {
			return err
		}
		// The sweep reads owner liveness from this gate: a caller that stops
		// heartbeating has its sandboxes reclaimed as orphans.
		life.SetOwnerLiveness(gate.OwnerAlive)
		agent, err = server.NewNode(server.NodeConfig{
			NodeID: cfg.NodeID, Admission: gate, Lifecycle: life, Backends: local,
			Pressure:        pressureFn,
			Elastic:         elasticConfig(cfg),
			Domain:          prepared.domain,
			SandboxParent:   prepared.parent,
			SandboxBudgetMB: prepared.budget.SubtreeLimitBytes / (1024 * 1024),
		})
		if err != nil {
			return err
		}
		// Rebuild the ledger from the daemon before anything reads the node's
		// view or it starts admitting, so a restart does not report an empty node
		// while its containers are still holding memory.
		reconciled, err := life.Reconcile(ctx, local)
		if err != nil {
			return fmt.Errorf("node %q cannot account for its own runtime: %w", cfg.NodeID, err)
		}
		log.Info("node reconciled its runtime at startup",
			"node", cfg.NodeID, "adopted", reconciled.Adopted,
			"reclaimed", reconciled.Reclaimed, "failures", reconciled.Failures)
		fleet.Report(agent.View())
		nodes = server.NewLocalNodeClient(agent)
	}

	ledger, err := quota.New(quota.Config{
		Total: quota.Amount{
			MemoryMB: cfg.Fleet.MemoryMB, CPUMillis: cfg.Fleet.CPUMillis, DiskMB: cfg.Fleet.DiskMB,
			Sandboxes: 1 << 20,
		},
		Classes:  quotaClasses(cfg),
		LeaseTTL: timing.OwnerLease,
	})
	if err != nil {
		return err
	}
	place, err := placement.New(placement.Config{
		NodeTTL:        spans.NodeTTL,
		ReservationTTL: timing.ReservationTTL,
		SweepInterval:  timing.SweepInterval,
	}, fleet)
	if err != nil {
		return err
	}
	control, err := server.NewControl(server.ControlConfig{
		Registry: registry, Ledger: ledger, Placement: place, Monitor: fleet,
		Nodes: nodes, Logger: log,
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
		startBackends(ctx, local, gate, log)
		defer stopBackends(local)
		go elasticLoop(ctx, agent, seconds(cfg.Node.Elastic.SampleIntervalS), log)
		// The bounds half runs on its own, much shorter cadence: it is the one a burst
		// has to be visible to.
		go limitLoop(ctx, agent, seconds(cfg.Node.Elastic.LimitSampleIntervalS), log)
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
	go sweepReservations(ctx, place, timing.SweepInterval)
	// The fleet's grants are repriced between reports, because that is when the nodes'
	// published prices move. Doing it on the admission path would put a walk of every
	// live grant in front of the request that is waiting for it.
	go repriceGrants(ctx, control, spans.LoadReportInterval())

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
				WarmPool: opensandbox.WarmPoolConfig{
					Image:          declared.WarmPool.Image,
					Size:           declared.WarmPool.Size,
					MemoryMB:       declared.WarmPool.MemoryMB,
					CPUCount:       declared.WarmPool.CPUCount,
					EntryTTL:       seconds(declared.WarmPool.EntryTTLS),
					RefillInterval: seconds(declared.WarmPool.RefillIntervalS),
				},
			}, mode)
		case "docker":
			if declared.WarmPool.Size > 0 {
				log.Info("warm pool declared",
					"backend", "docker", "size", declared.WarmPool.Size,
					"image", declared.WarmPool.Image, "class", declared.WarmPool.ResourceClass,
					"memory_mb", declared.WarmPool.MemoryMB, "entry_ttl_s", declared.WarmPool.EntryTTLS)
			}
			b, err = dockerbackend.New(dockerbackend.Config{
				Socket:               declared.Socket,
				APIVersion:           declared.APIVersion,
				NodeID:               cfg.NodeID,
				OwnerID:              owner,
				Runtime:              declared.Runtime,
				NetworkMode:          declared.NetworkMode,
				MaxCreateConcurrency: declared.MaxCreateConcurrency,
				MountAllowlist:       declared.MountAllowlist,
				CgroupParent:         cfg.Node.CgroupParent,
				WarmPool: dockerbackend.WarmPoolConfig{
					Template:       warmPoolTemplate(declared.WarmPool),
					Size:           declared.WarmPool.Size,
					EntryTTL:       seconds(declared.WarmPool.EntryTTLS),
					RefillInterval: seconds(declared.WarmPool.RefillIntervalS),
					PauseAbove:     declared.WarmPool.PauseAbove,
					ReclaimAbove:   declared.WarmPool.ReclaimAbove,
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
		log.Info("sandbox backend ready", "type", declared.Type, "mode", mode,
			"max_create_concurrency", declared.MaxCreateConcurrency)
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

// elasticConfig turns the deployment's declaration into the controller's configuration.
func elasticConfig(cfg Config) server.ElasticConfig {
	declared := cfg.Node.Elastic
	estimator := node.FootprintConfig{
		Percentile:   declared.Percentile,
		GrowthMargin: declared.GrowthMargin,
		ShrinkAfter:  declared.ShrinkAfter,
		ShrinkStep:   declared.ShrinkStep,
	}
	memory := estimator
	memory.MinReserve = declared.MinReserveMB
	cpu := estimator
	cpu.MinReserve = declared.MinReserveCPUMillis
	return server.ElasticConfig{
		Enabled:                declared.Enabled,
		Footprint:              memory,
		CPUFootprint:           cpu,
		ShrinkAbove:            declared.ShrinkAbove,
		RestoreBelow:           declared.RestoreBelow,
		ReleaseMarginMB:        declared.ReleaseMarginMB,
		MinLimitMB:             declared.MinLimitMB,
		ReleaseMarginCPUMillis: declared.ReleaseMarginCPUMillis,
		MinCPUMillis:           declared.MinCPUMillis,
		GrowthHorizonS:         declared.GrowthHorizonS,
		StartLimitMB:           declared.StartLimitMB,
		CeilingLimitMB:         declared.CeilingLimitMB,
		GrowAbove:              declared.GrowAbove,
		GrowFactor:             declared.GrowFactor,
		EmergencyGrowFactor:    declared.EmergencyGrowFactor,
		ShrinkBelow:            declared.ShrinkBelow,
		ShrinkDwellS:           declared.ShrinkDwellS,
		ShrinkFactor:           declared.ShrinkFactor,
		SafetyFraction:         declared.SafetyFraction,
		LimitSampleIntervalS:   declared.LimitSampleIntervalS,
	}
}

// elasticLoop runs the pricing and reporting half of the controller.
//
// It keeps the slow cadence even though the limits half runs at 200ms, because what it
// decides is a statistic over a class -- what a sandbox of this class costs, how fast the
// class is climbing -- and a statistic taken five times a second is a different, noisier
// number rather than a fresher one.
func elasticLoop(ctx context.Context, agent *server.Node, interval time.Duration, log *slog.Logger) {
	if interval <= 0 {
		interval = seconds(0) // zero means the sampler's default
	}
	if interval <= 0 {
		interval = 5 * time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// The ledger is audited on this cadence because this is the node's only
			// place that looks at itself rather than at a request. A charge above the
			// envelope is not something a caller can report: from outside it is one
			// more refusal, indistinguishable from a node that is simply full.
			auditLedger(agent, log)
			report := agent.ElasticTick(ctx)
			changed := len(report.Throttled) + len(report.Released)
			if changed == 0 && report.OOMKills == 0 {
				continue
			}
			log.Warn("elastic admission changed sandbox limits",
				"observed_mib", report.ObservedMB, "budget_mib", report.BudgetMB,
				"reserved_mib", report.ReservedMB, "reserves_mib", report.ReservesMB,
				"observed_cpu_millis", report.ObservedCPUMillis, "budget_cpu_millis", report.BudgetCPUMillis,
				"reserved_cpu_millis", report.ReservedCPUMillis, "reserves_cpu_millis", report.ReservesCPUMillis,
				"throttled", len(report.Throttled), "released", len(report.Released),
				"oom_kills", report.OOMKills)
		}
	}
}

// limitLoop runs the bounds half of the controller, at the cadence a burst is visible on.
//
// It logs only when something happened, and only the counts: at five ticks a second a line
// per tick is not a log, it is the reason nobody reads the log.
func limitLoop(ctx context.Context, agent *server.Node, interval time.Duration, log *slog.Logger) {
	if interval <= 0 {
		interval = server.DefaultLimitSampleInterval()
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			report := agent.ElasticLimitsTick(ctx)
			if len(report.Grown) == 0 && len(report.Shrunk) == 0 && len(report.Preempted) == 0 &&
				len(report.OOMed) == 0 && len(report.Ended) == 0 {
				continue
			}
			log.Warn("elastic admission moved a sandbox memory bound",
				"grown", len(report.Grown), "shrunk", len(report.Shrunk),
				"preempted", len(report.Preempted), "oomed", len(report.OOMed),
				"ended", len(report.Ended),
				"bounds_mib", report.LimitSumMB, "spare_mib", report.SpareMB)
		}
	}
}

// auditLedger states whether this node is charging capacity it does not have.
//
// It is logged once per transition rather than per tick, so a node that stays broken is
// one line an operator can find rather than a line every five seconds that buries
// everything else.
var ledgerUnaccounted bool

func auditLedger(agent *server.Node, log *slog.Logger) {
	granted, envelope, accounted := agent.LedgerAudit()
	if accounted {
		if ledgerUnaccounted {
			ledgerUnaccounted = false
			log.Warn("the node's charge is inside its envelope again", "granted_memory_mb", granted.MemoryMB)
		}
		return
	}
	if ledgerUnaccounted {
		return
	}
	ledgerUnaccounted = true
	log.Error("the node is charging more than it has: every request will be refused until it is restarted",
		"granted_memory_mb", granted.MemoryMB, "envelope_memory_mb", envelope.MemoryMB,
		"granted_cpu_millis", granted.CPUMillis, "envelope_cpu_millis", envelope.CPUMillis)
}

// startBackends begins any background work a backend owns.
//
// Only a backend that has something to run implements this, so the assertion is
// an interface check rather than a flag: a backend with no pool and no loop is
// simply skipped.
func startBackends(ctx context.Context, backends []backend.Backend, gate *node.Admission, log *slog.Logger) {
	for _, b := range backends {
		// A backend that prefetches containers borrows the node's budget for them, and
		// the budget has to be installed before its loops start or the first refill
		// would build memory the node never agreed to hold.
		if budgeted, can := b.(backend.PoolBudgeted); can && gate != nil {
			budgeted.SetPoolBudget(gate.AsPoolBudget())
		}
		starter, can := b.(interface{ Start(context.Context) })
		if !can {
			continue
		}
		starter.Start(ctx)
		log.Info("backend background work started", "backend", b.Name())
	}
	if count, err := foreignContainers(ctx, backends); err != nil {
		log.Warn("could not count containers outside the sandbox subtree", "reason", err)
	} else if count > 0 {
		log.Warn("containers this service does not own are running beside its sandboxes",
			"containers", count,
			"reason", "they are outside the sandbox cgroup subtree, so the sampler does not measure them; "+
				"their memory shows up as capacity this node can never admit")
	}
}

// foreignContainers counts containers on the local daemons that this service does not own.
//
// They are outside the sandbox subtree by construction -- only containers this service
// creates are placed there -- so they are charged to the ancestor cgroup and invisible to
// every ceiling. One is a warning rather than a refusal: sharing a daemon is a deployment
// choice, and the operator needs to know what it costs.
func foreignContainers(ctx context.Context, backends []backend.Backend) (int, error) {
	total := 0
	for _, b := range backends {
		counter, can := b.(interface {
			ForeignContainers(context.Context) (int, error)
		})
		if !can {
			continue
		}
		count, err := counter.ForeignContainers(ctx)
		if err != nil {
			return total, err
		}
		total += count
	}
	return total, nil
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

// nodeDomain prepares the sandbox cgroup subtree and the sampler that measures it.
//
// It is one function because the two have to agree: the subtree's limit is resolved
// from the node's declared envelope and the ancestor's capacity, and the envelope
// admission uses is the resolved value. Deriving them separately is what lets a kernel
// bound and an admission bound drift apart, and the symptom of that drift is either a
// ceiling nothing can reach or an OOM admission promised it had room for.
//
// A deployment that configured no measured ceiling, no overcommit and no subtree loses
// nothing by running without a domain, so an unreadable cgroup warns instead of
// refusing to start. Every deployment that did configure one is refused: admitting
// against capacity it cannot see is the failure the domain exists to prevent.
// sandboxDomain is everything the node learns about its subtree: the tree itself, the
// budget it resolved, and the sampler that reads it.
type sandboxDomain struct {
	domain   cgroup.Domain
	parent   string
	budget   cgroup.Budget
	pressure func() node.Pressure
}

func nodeDomain(ctx context.Context, cfg Config, log *slog.Logger) (sandboxDomain, error) {
	parent := cfg.Node.CgroupParent
	if parent == "" {
		parent = cgroup.SandboxParent
	}
	request := cgroup.Request{
		MemoryMB:   cfg.Node.MemoryMB,
		Overcommit: cfg.Node.Overcommit,
		CPUMillis:  cfg.Node.CPUMillis,
		Safety:     cfg.Node.SandboxMemorySafety,
	}
	prepared, err := prepareDomain(ctx, parent, request, seconds(cfg.Node.SampleIntervalS), log)
	if err == nil {
		return prepared, nil
	}
	measured := cfg.Node.LocalCPUCeiling > 0 || cfg.Node.LocalMemCeiling > 0
	if measured || cfg.Node.Overcommit > 1 || cfg.Node.CgroupParent != "" || cfg.Node.Elastic.Enabled {
		return sandboxDomain{}, fmt.Errorf("sandbox domain: %w", err)
	}
	log.Warn("no sandbox cgroup domain; measured ceilings, overcommit and elasticity stay off", "reason", err)
	return sandboxDomain{
		parent:   parent,
		budget:   cgroup.Budget{DeclaredMB: cfg.Node.MemoryMB},
		pressure: func() node.Pressure { return node.Pressure{} },
	}, nil
}

// prepareDomain creates the subtree, moves this process out of it, and starts the
// sampler that reads it.
func prepareDomain(
	ctx context.Context, parent string, request cgroup.Request, interval time.Duration, log *slog.Logger,
) (sandboxDomain, error) {
	domain, err := cgroup.Resolve()
	if err != nil {
		return sandboxDomain{}, err
	}
	budget, err := domain.Setup(parent, cgroup.HostLeaf, request)
	if err != nil {
		return sandboxDomain{}, err
	}
	reader, err := pressure.NewDomainReader(parent)
	if err != nil {
		return sandboxDomain{}, err
	}
	sampler, err := pressure.NewSampler(reader, interval)
	if err != nil {
		return sandboxDomain{}, err
	}
	sampler.Start(ctx)
	log.Info("sandbox domain ready",
		"domain", domain.String(), "subtree", parent,
		"limit_mib", budget.SubtreeLimitBytes/(1024*1024),
		"declared_mib", budget.DeclaredMB, "requested_mib", budget.RequestedMB,
		"ancestor_limit_mib", budget.AncestorLimitBytes/(1024*1024),
		"unreclaimable_mib", budget.UnreclaimableBytes/(1024*1024),
		"cpu_millis", budget.CPUMillis, "sample_interval_s", sampler.Interval().Seconds())
	if budget.Capped() {
		log.Warn("the sandbox subtree is smaller than the declared envelope",
			"declared_mib", budget.RequestedMB, "envelope_mib", budget.DeclaredMB,
			"limit_mib", budget.SubtreeLimitBytes/(1024*1024),
			"ancestor_limit_mib", budget.AncestorLimitBytes/(1024*1024),
			"unreclaimable_mib", budget.UnreclaimableBytes/(1024*1024),
			"reason", "the ancestor cgroup cannot hold the declared envelope plus what it already holds")
	}
	return sandboxDomain{domain: domain, parent: parent, budget: budget, pressure: sampler.Pressure()}, nil
}

// nodeEnvelope is the envelope admission uses: the resolved one when a domain exists.
func nodeEnvelope(cfg Config, budget cgroup.Budget) node.Resources {
	memoryMB := cfg.Node.MemoryMB
	if budget.DeclaredMB > 0 {
		memoryMB = budget.DeclaredMB
	}
	return node.Resources{MemoryMB: memoryMB, CPUMillis: cfg.Node.CPUMillis, DiskMB: cfg.Node.DiskMB}
}

// warmPoolTemplate turns a deployment's pool declaration into the spec entries are built
// from.
func warmPoolTemplate(cfg WarmPoolDeclared) backend.Spec {
	return backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: cfg.Image},
		Resources:     backend.Resources{MemoryMB: cfg.MemoryMB, CPUCount: cfg.CPUCount, DiskMB: cfg.DiskMB},
		ResourceClass: cfg.ResourceClass,
		Env:           cfg.Env,
		Workdir:       cfg.Workdir,
		ExecMode:      cfg.ExecMode,
		Lifetime:      seconds(cfg.LifetimeS),
	}
}

// repriceGrants keeps the fleet's ledger honest about what the nodes measure.
//
// It runs on the report cadence for the same reason placement does: the fleet cannot know
// a class's price until a node publishes it, and a price it has not seen yet is not a
// reason to move anything.
func repriceGrants(ctx context.Context, control *server.Control, interval time.Duration) {
	if control == nil {
		return
	}
	if interval <= 0 {
		interval = 30 * time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			control.RepriceGrants()
		}
	}
}
