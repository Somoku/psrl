package server

import (
	"context"
	"testing"
	"time"
)

func TestContractPublishesTheMechanismSpans(t *testing.T) {
	reply, err := (&JSONListener{}).contract(context.Background())
	if err != nil {
		t.Fatalf("contract: %v", err)
	}
	spans := reply.(map[string]any)
	// The client derives its heartbeat cadence from these; a missing or renamed
	// key silently returns the client to hardcoded values that can drift.
	if got := spans["owner_silence_ttl_s"]; got != 90.0 {
		t.Fatalf("owner_silence_ttl_s = %v, want 90", got)
	}
	if got := spans["owner_heartbeat_interval_s"]; got != 30.0 {
		t.Fatalf("owner_heartbeat_interval_s = %v, want 30", got)
	}
	if got := spans["runtime_probe_interval_s"]; got != 30.0 {
		t.Fatalf("runtime_probe_interval_s = %v, want 30", got)
	}
}

func TestDiagnosticsReportsRenewalAndFleetHealth(t *testing.T) {
	never := make(chan struct{})
	defer close(never)
	nodes := &renewClient{
		stubNodes: &stubNodes{},
		block:     map[string]<-chan struct{}{"node-b": never},
	}
	control, _ := newTestControl(t, nodes, time.Now)
	reportNode(t, control, "node-a")
	reportNode(t, control, "node-b")

	control.RenewOwner(context.Background(), "owner-1")
	diagnostics := control.Diagnostics()

	renew := diagnostics["renew"].(map[string]any)
	if renew["attempts"].(int64) != 1 {
		t.Fatalf("renew attempts = %v, want 1", renew["attempts"])
	}
	if renew["failures"].(int64) != 1 {
		t.Fatalf("renew failures = %v, want 1 (the stalled node)", renew["failures"])
	}
	if renew["nodes_targeted"].(int64) != 2 {
		t.Fatalf("nodes targeted = %v, want 2", renew["nodes_targeted"])
	}

	fleet := diagnostics["fleet"].(map[string]any)
	if fleet["live_nodes"].(int) != 2 {
		t.Fatalf("live nodes = %v, want 2", fleet["live_nodes"])
	}
	if fleet["stale_nodes"].(int) != 0 {
		t.Fatalf("stale nodes = %v, want 0", fleet["stale_nodes"])
	}
}
