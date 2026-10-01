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
	"psrl.dev/sandboxd/internal/backend/dockerbackend"
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

	Backends []struct {
		Type string `json:"type"`
		// Mode is "psrl" or "provider": who chooses the node. It is a deployment
		// property rather than a backend capability, so the same backend can run
		// both ways in one fleet and a scheduler ablation holds everything else fixed.
		Mode       string `json:"mode"`
		Socket     string `json:"socket"`
		APIVersion string `json:"api_version"`
		Runtime    string `json:"runtime"`
	} `json:"backends"`

	DefaultBackend string `json:"default_backend"`
	OwnerID        string `json:"owner_id"`
}

func main() {
	configPath := flag.String("config", "", "path to the service configuration")
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	if err := run(*configPath, log); err != nil {
		log.Error("sandboxd stopped", "error", err)
		os.Exit(1)
	}
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
		NodeID: cfg.NodeID, Admission: gate, Lifecycle: life, Backend: backends[0],
	})
	if err != nil {
		return err
	}

	fleet := monitor.New(spans.NodeTTL)
	fleet.Report(agent.View())

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
		Nodes:          server.NewLocalNodeClient(agent),
		AcquireTimeout: spans.AcquireTimeout(),
	})
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	life.Start(ctx)
	defer life.Stop()

	// The node republishes itself on the contract's cadence, so placement reads a
	// view that is never more than a report old.
	go republish(ctx, agent, fleet, spans.LoadReportInterval())
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
		switch declared.Type {
		case "docker":
			b, err := dockerbackend.New(dockerbackend.Config{
				Socket:     declared.Socket,
				APIVersion: declared.APIVersion,
				NodeID:     cfg.NodeID,
				OwnerID:    owner,
				Runtime:    declared.Runtime,
			}, mode)
			if err != nil {
				return nil, err
			}
			// Preflight at startup rather than at the first rollout: a daemon that
			// is not answering is a configuration error, and discovering it inside
			// an episode costs a sample.
			if err := b.Preflight(context.Background()); err != nil {
				return nil, fmt.Errorf("backend %q preflight: %w", declared.Type, err)
			}
			log.Info("sandbox backend ready", "type", declared.Type, "mode", mode)
			built = append(built, b)
		default:
			return nil, fmt.Errorf("sandbox backend type %q is not built into this binary", declared.Type)
		}
	}
	return built, nil
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
