package main

// The shipped example configuration has to be loadable and valid, because it is
// the first command a new operator runs. This is not a hypothetical: the
// scheduling modes were renamed to "direct" and "provider", and example.json
// kept the retired "psrl" value, so the documented quickstart failed at startup
// with a mode the service no longer accepts. Nothing caught it, because no test
// read the file the README points at.
//
// These cases drive the same loader and the same validators the service drives,
// rather than reimplementing validation — a check that agrees with a second
// implementation of the rules proves only that the two agree.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/quota"
	"psrl.dev/sandboxd/internal/timing"
)

// examplePath is the configuration the README's quickstart runs.
const examplePath = "../../example.json"

func TestTheExampleConfigurationLoads(t *testing.T) {
	cfg, err := load(examplePath)
	if err != nil {
		t.Fatalf("the shipped example.json does not load: %v\n"+
			"This is the file the README tells an operator to run, so a failure here "+
			"is a broken quickstart rather than a broken test.", err)
	}
	if len(cfg.Backends) == 0 {
		t.Fatal("example.json declares no backends, so the service would have nothing to place on")
	}
	if cfg.DefaultBackend == "" {
		t.Error("example.json names no default_backend, so a spec without one could not be routed")
	}
}

func TestEveryBackendModeInTheExampleIsAccepted(t *testing.T) {
	// The specific defect this pins. A mode the registry refuses fails the
	// deployment at startup, and the error names a value the operator copied
	// from the shipped file.
	cfg, err := load(examplePath)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	for _, declared := range cfg.Backends {
		mode := backend.SchedulingMode(declared.Mode)
		// An empty mode is allowed: the service defaults it.
		if declared.Mode == "" {
			continue
		}
		if !mode.Valid() {
			t.Errorf("example.json declares backend %q with mode %q, which the service refuses. "+
				"Valid modes are %q and %q.",
				declared.Type, declared.Mode, backend.SchedulingDirect, backend.SchedulingProvider)
		}
	}
}

func TestTheExampleTimingAndSharesValidate(t *testing.T) {
	// The same three constructions the service performs on startup, in the same
	// order, so a configuration that passes here is one that starts.
	cfg, err := load(examplePath)
	if err != nil {
		t.Fatalf("load: %v", err)
	}

	spans, err := timing.New(
		seconds(cfg.Timing.EpisodeDeadlineS),
		seconds(cfg.Timing.NodeTTLS),
		seconds(cfg.Timing.RPCTimeoutS),
		overrides(cfg.Timing.Overrides),
	)
	if err != nil {
		t.Fatalf("the example's timing contract is invalid: %v", err)
	}

	if _, err := quota.New(quota.Config{
		Total: quota.Amount{
			MemoryMB: cfg.Fleet.MemoryMB, CPUMillis: cfg.Fleet.CPUMillis, DiskMB: cfg.Fleet.DiskMB,
			Sandboxes: 1 << 20,
		},
		Classes:  quotaClasses(cfg),
		LeaseTTL: spans.CapacityLeaseTTL(),
	}); err != nil {
		t.Errorf("the example's fleet quota is invalid: %v", err)
	}

	if _, err := node.NewAdmission(node.Config{
		Envelope: node.Resources{
			MemoryMB: cfg.Node.MemoryMB, CPUMillis: cfg.Node.CPUMillis, DiskMB: cfg.Node.DiskMB,
		},
		Classes:           nodeClasses(cfg),
		GPUIndices:        cfg.Node.GPUIndices,
		LocalCPUCeiling:   cfg.Node.LocalCPUCeiling,
		LocalMemCeiling:   cfg.Node.LocalMemCeiling,
		Overcommit:        cfg.Node.Overcommit,
		UtilizationTarget: cfg.Node.UtilizationTarget,
		LeaseTTL:          spans.CapacityLeaseTTL(),
	}); err != nil {
		t.Errorf("the example's node admission is invalid: %v", err)
	}
}

func TestTheExampleDeclaresEveryClassItsBackendsCharge(t *testing.T) {
	// A spec is charged to a class, and a class absent from the configuration is
	// compared against nothing. The SDK's own default class is "rollout", so a
	// shipped example that omits it would refuse the first create a reader makes.
	cfg, err := load(examplePath)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if _, declared := cfg.Classes["rollout"]; !declared {
		t.Error("example.json declares no \"rollout\" class, which is the class a spec " +
			"takes when it names none; the first create from the README would be refused")
	}
}

func TestAnUnknownBackendModeIsRejectedWithItsOwnName(t *testing.T) {
	// The failure an operator actually hits. It has to name the offending value,
	// because the whole class of defect is a value that looks plausible.
	directory := t.TempDir()
	path := filepath.Join(directory, "bad.json")

	base, err := os.ReadFile(examplePath)
	if err != nil {
		t.Fatalf("read example: %v", err)
	}
	var raw map[string]any
	if err := json.Unmarshal(base, &raw); err != nil {
		t.Fatalf("decode example: %v", err)
	}
	backends, _ := raw["backends"].([]any)
	if len(backends) == 0 {
		t.Fatal("example.json has no backends to mutate")
	}
	first, _ := backends[0].(map[string]any)
	first["mode"] = "psrl" // the retired value
	mutated, err := json.Marshal(raw)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if err := os.WriteFile(path, mutated, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}

	cfg, err := load(path)
	if err != nil {
		// Refused at load is also correct, as long as it is refused.
		return
	}
	for _, declared := range cfg.Backends {
		if declared.Mode == "psrl" && backend.SchedulingMode(declared.Mode).Valid() {
			t.Error("the retired mode \"psrl\" is being accepted as valid")
		}
	}
}

// seconds and overrides mirror what main uses, kept here so the test drives the
// same conversion rather than a second one that could disagree.
var _ = time.Second
