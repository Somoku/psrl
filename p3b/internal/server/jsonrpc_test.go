package server

import (
	"encoding/json"
	"testing"

	"google.golang.org/protobuf/proto"

	v1 "psrl.dev/sandboxd/api/v1"
)

// The SDK sends node labels as JSON, and placement filters on them. They have to
// reach the proto spec and survive a protobuf round-trip: a label that lives only
// in the Go struct is dropped the moment the spec is marshaled, and placement
// then silently ignores a pin the caller stated.
func TestJSONSpecCarriesNodeLabels(t *testing.T) {
	raw := []byte(`{
		"source": {"kind": "image", "reference": "img"},
		"resources": {"cpu_count": 2, "memory_mb": 1024},
		"resource_class": "rollout",
		"required_node_label": "trainer",
		"forbidden_node_labels": ["gpu-a", "gpu-b"]
	}`)
	var spec jsonSpec
	if err := json.Unmarshal(raw, &spec); err != nil {
		t.Fatalf("unmarshal json spec: %v", err)
	}

	out := spec.toProto()
	if out.GetRequiredNodeLabel() != "trainer" {
		t.Fatalf("required node label lost converting the JSON spec: %q", out.GetRequiredNodeLabel())
	}
	forbidden := out.GetForbiddenNodeLabels()
	if len(forbidden) != 2 || forbidden[0] != "gpu-a" || forbidden[1] != "gpu-b" {
		t.Fatalf("forbidden node labels lost converting the JSON spec: %v", forbidden)
	}

	encoded, err := proto.Marshal(out)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back v1.SandboxSpec
	if err := proto.Unmarshal(encoded, &back); err != nil {
		t.Fatalf("unmarshal proto: %v", err)
	}
	if back.GetRequiredNodeLabel() != "trainer" || len(back.GetForbiddenNodeLabels()) != 2 {
		t.Fatalf("node labels did not survive protobuf: required=%q forbidden=%v",
			back.GetRequiredNodeLabel(), back.GetForbiddenNodeLabels())
	}
}
