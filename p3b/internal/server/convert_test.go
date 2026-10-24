package server

import (
	"reflect"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

func TestASpecSurvivesTheWireRoundTrip(t *testing.T) {
	// Every field the node enforces has to survive specToProto and specFromProto,
	// because the node reads the spec it is handed rather than the one the caller
	// wrote. A field dropped here is silently replaced by a default -- which is
	// how a caller's stated lifetime became the node's own.
	spec := backend.Spec{
		Source:              backend.Source{Kind: "image", Reference: "alpine:latest"},
		Resources:           backend.Resources{MemoryMB: 512, CPUCount: 0.75, DiskMB: 128, GPUCount: 1},
		ResourceClass:       "rollout",
		WorkflowID:          "workflow-7",
		IdempotencyKey:      "ticket-3",
		Env:                 map[string]string{"A": "1", "B": "2"},
		Metadata:            map[string]string{backend.OwnerMetadataKey: "worker-9"},
		Workdir:             "/workspace/repo",
		ExecMode:            "persistent",
		RequiredFeatures:    []string{"freeze", "resume_anywhere"},
		RequiredResume:      "full_state",
		Backend:             "docker",
		RequiredNodeLabel:   "pool=rl",
		ForbiddenNodeLabels: []string{"role=trainer"},
		Lifetime:            45 * time.Minute,
		BackendOptions:      map[string]map[string]string{"docker": {"network_mode": "host"}},
	}

	back := specFromProto(specToProto(spec))

	if !reflect.DeepEqual(back, spec) {
		t.Fatalf("the spec changed on the wire:\n got %+v\nwant %+v", back, spec)
	}
	if back.Lifetime != 45*time.Minute {
		t.Fatalf("lifetime arrived as %s, want 45m", back.Lifetime)
	}
}
