package server

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/node"
)

// typedBackend is a fake whose exec and file operations fail with a chosen
// sentinel, so a test can pin how the node turns a runtime fact into a caller
// classification.
type typedBackend struct {
	*fakeDirect
	execErr error
	readErr error
	status  string
}

func (b *typedBackend) Exec(
	context.Context, backend.Handle, string, string, string, map[string]string,
) (backend.ExecResult, error) {
	return backend.ExecResult{}, b.execErr
}

func (b *typedBackend) ReadFile(context.Context, backend.Handle, string) ([]byte, error) {
	return nil, b.readErr
}

func (b *typedBackend) WriteFile(context.Context, backend.Handle, string, []byte) error {
	return nil
}

func (b *typedBackend) Status(context.Context, backend.Handle) (string, error) {
	if b.status != "" {
		return b.status, nil
	}
	return "running", nil
}

func createOneSandbox(
	t *testing.T, hosted backend.Backend,
) (*Node, *node.Lifecycle, backend.Handle) {
	t.Helper()
	agent, _, life, spec := newNodeOverFake(t, hosted)
	created, err := agent.CreateOn(context.Background(), &v1.CreateOnRequest{
		LeaseId: reserveOne(t, agent, spec), Backend: "fake", Spec: spec,
	})
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	handle := backend.Handle{
		Backend:   created.GetHandle().GetBackend(),
		SandboxID: created.GetHandle().GetSandboxId(),
		NodeID:    "node-a",
	}
	return agent, life, handle
}

func TestExecOnAStoppedContainerIsSessionLostAndReleasesTheCharge(t *testing.T) {
	hosted := &typedBackend{fakeDirect: &fakeDirect{}, execErr: backend.ErrRuntimeGone, status: "oom"}
	agent, life, handle := createOneSandbox(t, hosted)

	_, err := agent.Exec(context.Background(), handle, "true", "", nil)
	var gone *sessionGoneError
	if !errors.As(err, &gone) {
		t.Fatalf("a stopped container must be reported as a lost session, got %v", err)
	}
	if gone.reason != "oom" {
		t.Fatalf("exit reason %q, want oom from the runtime's own status", gone.reason)
	}
	life.WaitCleanup()
	if len(hosted.release) != 1 {
		t.Fatalf("the backend saw %d releases, want 1", len(hosted.release))
	}
}

func TestReadOfAMissingPathIsNotALostSession(t *testing.T) {
	hosted := &typedBackend{fakeDirect: &fakeDirect{}, readErr: backend.ErrPathNotFound}
	agent, life, handle := createOneSandbox(t, hosted)

	_, err := agent.ReadBytes(context.Background(), handle, "/root/.psrl-harness/missing")
	var missing *pathNotFoundError
	if !errors.As(err, &missing) {
		t.Fatalf("a missing file must be its own error, got %v", err)
	}
	life.WaitCleanup()
	if len(hosted.release) != 0 {
		t.Fatal("a missing file must not destroy a healthy sandbox")
	}
	if code := classify(err).Code; code != "path_not_found" {
		t.Fatalf("wire code %q, want path_not_found", code)
	}
}

func TestReadFromAGoneContainerIsALostSession(t *testing.T) {
	hosted := &typedBackend{fakeDirect: &fakeDirect{}, readErr: fmt.Errorf("archive: %w", backend.ErrRuntimeGone)}
	agent, life, handle := createOneSandbox(t, hosted)

	_, err := agent.ReadBytes(context.Background(), handle, "/tmp/x")
	var gone *sessionGoneError
	if !errors.As(err, &gone) {
		t.Fatalf("a gone container must be a lost session, got %v", err)
	}
	life.WaitCleanup()
	if len(hosted.release) != 1 {
		t.Fatalf("the backend saw %d releases, want 1", len(hosted.release))
	}
}

func TestClassifyPreservesPathNotFoundAcrossTheNodePlane(t *testing.T) {
	original := &wireError{Code: "path_not_found", Message: "reading /x: missing"}
	acrossTheHop := wireErrorToStatus(original)

	if code := classify(acrossTheHop).Code; code != "path_not_found" {
		t.Fatalf("wire code after the node hop %q, want path_not_found", code)
	}
}

func TestPreserveSessionLostStillMapsTheBareSentinel(t *testing.T) {
	mapped := preserveSessionLost(fmt.Errorf("shell exited: %w", backend.ErrSessionLost))
	if status.Code(mapped) != codes.NotFound {
		t.Fatalf("a bare ErrSessionLost must map to NotFound, got %v", mapped)
	}
}
