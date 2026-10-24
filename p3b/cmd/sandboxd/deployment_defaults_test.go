package main

import (
	"log/slog"
	"path/filepath"
	"strings"
	"testing"
)

func loadDeployment(t *testing.T, name string) Config {
	t.Helper()
	cfg, err := load(filepath.Join("..", "..", name))
	if err != nil {
		t.Fatalf("load %s: %v", name, err)
	}
	return cfg
}

func dockerDeclared(cfg Config) bool {
	for _, declared := range cfg.Backends {
		if declared.Type == "docker" && declared.Mode == "direct" {
			return true
		}
	}
	return false
}

// TestTheExampleDeploymentIsADockerCluster pins the shipped default: a control
// plane over node agents, every plane running the Docker backend. The examples
// are what `make cluster` and the README run, so a drift here is a default that
// silently became the single-process shape or a different runtime.
func TestTheExampleDeploymentIsADockerCluster(t *testing.T) {
	control := loadDeployment(t, "example_control.json")

	if control.Role != "control" {
		t.Fatalf("example_control.json role = %q, want control", control.Role)
	}
	if strings.HasPrefix(control.Listen, "unix://") || control.Listen == "" {
		t.Fatalf("the control plane must listen on host:port so every worker can reach it, got %q", control.Listen)
	}
	if len(control.FleetNodes) < 2 {
		t.Fatalf("the cluster example must name at least two nodes, got %d", len(control.FleetNodes))
	}
	if control.DefaultBackend != "docker" || !dockerDeclared(control) {
		t.Fatalf("the control plane must route to the docker backend, got default=%q backends=%v",
			control.DefaultBackend, control.Backends)
	}

	flock := map[string]bool{}
	for _, fleetNode := range control.FleetNodes {
		if fleetNode.NodeID == "" || fleetNode.Address == "" {
			t.Fatalf("fleet node entry %+v needs both node_id and address", fleetNode)
		}
		if strings.HasPrefix(fleetNode.Address, "unix://") {
			t.Fatalf("a fleet node is reached over TCP, got %q", fleetNode.Address)
		}
		flock[fleetNode.NodeID] = true
	}

	cgroupParents := map[string]bool{}
	for _, name := range []string{"example_node.json", "example_node_b.json"} {
		node := loadDeployment(t, name)
		if node.Role != "node" {
			t.Fatalf("%s role = %q, want node", name, node.Role)
		}
		if node.NodeListen == "" || strings.HasPrefix(node.NodeListen, "unix://") {
			t.Fatalf("%s must listen on host:port, got %q", name, node.NodeListen)
		}
		if !flock[node.NodeID] {
			t.Fatalf("%s node %q is not in the control plane's fleet_nodes", name, node.NodeID)
		}
		if node.DefaultBackend != "docker" || !dockerDeclared(node) {
			t.Fatalf("%s must run the docker backend, got default=%q backends=%v",
				name, node.DefaultBackend, node.Backends)
		}
		if node.Node.Overcommit < 1 {
			// Zero is not "off": the cgroup domain refuses a subtree smaller than
			// the envelope. One is the explicit no-overcommit value.
			t.Fatalf("%s overcommit = %v, want at least 1", name, node.Node.Overcommit)
		}
		if node.Node.CgroupParent == "" || cgroupParents[node.Node.CgroupParent] {
			t.Fatalf("%s cgroup_parent %q is empty or shared with another example node",
				name, node.Node.CgroupParent)
		}
		cgroupParents[node.Node.CgroupParent] = true
	}
}

// TestControlPreflightNeedsNoLocalDaemon is the property that makes a control
// host a control host: it declares a backend for routing and drives none of it.
// A preflight that reached a local Docker daemon would make the split deployment
// impossible to validate on a host that runs no sandboxes.
func TestControlPreflightNeedsNoLocalDaemon(t *testing.T) {
	if err := preflight(filepath.Join("..", "..", "example_control.json"), slog.Default()); err != nil {
		t.Fatalf("control preflight: %v", err)
	}
}
