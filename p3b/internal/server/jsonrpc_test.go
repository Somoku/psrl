package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
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

func TestPreserveSessionLostKeepsTheTwoFailuresApart(t *testing.T) {
	// A lost session is a fact about the caller's own state and an unanswered node
	// is an infrastructure fault, so they cannot collapse into one another: the
	// caller retries the second and has to rebuild after the first.
	lost := fmt.Errorf("the shell exited: %w", backend.ErrSessionLost)
	if code := status.Code(preserveSessionLost(lost)); code != codes.NotFound {
		t.Errorf("a lost session became %v, want NotFound", code)
	}

	// A failure that already carries an attribution keeps it, rather than being
	// replaced by an equal-looking status that has lost the reason.
	attributed := status.Error(codes.NotFound, "sandbox is not held by this node")
	if got := preserveSessionLost(attributed); status.Code(got) != codes.NotFound {
		t.Errorf("an attributed failure became %v, want NotFound", status.Code(got))
	}

	plain := errors.New("connection refused")
	if code := status.Code(preserveSessionLost(plain)); code != codes.Unavailable {
		t.Errorf("a plain failure became %v, want Unavailable", code)
	}
}
