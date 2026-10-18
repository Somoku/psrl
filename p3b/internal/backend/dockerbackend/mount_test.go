package dockerbackend

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"psrl.dev/sandboxd/internal/backend"
)

// newMountBackend returns a backend whose allowlist is one tree under a temp dir.
func newMountBackend(t *testing.T, allowlist ...string) *Backend {
	t.Helper()
	return &Backend{cfg: Config{MountAllowlist: allowlist}}
}

func TestAMountInsideTheAllowlistIsBound(t *testing.T) {
	root := t.TempDir()
	shared := filepath.Join(root, "shared")
	if err := os.MkdirAll(filepath.Join(shared, "harness"), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	backend1 := newMountBackend(t, shared)

	got, err := backend1.binds(backend.Spec{Mounts: []backend.Mount{
		{Source: filepath.Join(shared, "harness"), Target: "/opt/harness", ReadOnly: true},
	}})
	if err != nil {
		t.Fatalf("binds: %v", err)
	}
	if len(got) != 1 || got[0] != filepath.Join(shared, "harness")+":/opt/harness:ro" {
		t.Errorf("bind: got %v", got)
	}
}

func TestAMountOutsideTheAllowlistIsRefused(t *testing.T) {
	root := t.TempDir()
	shared := filepath.Join(root, "shared")
	if err := os.MkdirAll(shared, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	// A sibling whose name begins with the allowed one: a prefix comparison without the
	// separator would let it through.
	sneaky := filepath.Join(root, "shared-secrets")
	if err := os.MkdirAll(sneaky, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	backend1 := newMountBackend(t, shared)

	_, err := backend1.binds(backend.Spec{Mounts: []backend.Mount{{Source: sneaky, Target: "/opt/x"}}})
	if err == nil {
		t.Fatal("a sibling of an allowed tree was accepted")
	}
	if !strings.Contains(err.Error(), "outside this node's mount allowlist") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestASymlinkOutOfTheAllowlistIsRefused(t *testing.T) {
	root := t.TempDir()
	shared := filepath.Join(root, "shared")
	if err := os.MkdirAll(shared, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	outside := filepath.Join(root, "outside")
	if err := os.MkdirAll(outside, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	// A link that lives in the allowed tree and points out of it, which is the escape a
	// path comparison on the unresolved name would miss.
	if err := os.Symlink(outside, filepath.Join(shared, "escape")); err != nil {
		t.Skipf("symlinks unsupported here: %v", err)
	}
	backend1 := newMountBackend(t, shared)

	if _, err := backend1.binds(backend.Spec{Mounts: []backend.Mount{
		{Source: filepath.Join(shared, "escape"), Target: "/opt/x"},
	}}); err == nil {
		t.Fatal("a symlink out of the allowlist was accepted")
	}
}

func TestAMountThatDoesNotExistIsRefusedRatherThanCreated(t *testing.T) {
	// The daemon creates a missing bind source as a directory on the host and reports
	// success, so passing it through would hand back a sandbox that looks mounted and is
	// not.
	root := t.TempDir()
	backend1 := newMountBackend(t, root)

	_, err := backend1.binds(backend.Spec{Mounts: []backend.Mount{
		{Source: filepath.Join(root, "absent"), Target: "/opt/x"},
	}})
	if err == nil {
		t.Fatal("a missing mount source was accepted")
	}
	if !strings.Contains(err.Error(), "is not a path on this node") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestANodeWithNoAllowlistRefusesAMount(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(root, 0o755); err == nil {
		// An empty allowlist shares nothing, and the message says what to set.
		backend1 := newMountBackend(t)
		_, err := backend1.binds(backend.Spec{Mounts: []backend.Mount{{Source: root, Target: "/opt/x"}}})
		if err == nil || !strings.Contains(err.Error(), "mount_allowlist") {
			t.Errorf("expected a refusal naming the setting, got %v", err)
		}
	}
}

func TestHostMountIsAdvertisedOnlyWhenTheNodeSharesSomething(t *testing.T) {
	shared := t.TempDir()
	without := newMountBackend(t)
	if without.Capabilities().Supports("host_mount") {
		t.Error("a node that shares nothing must not advertise host_mount")
	}
	with := newMountBackend(t, shared)
	if !with.Capabilities().Supports("host_mount") {
		t.Error("a node with an allowlist must advertise host_mount")
	}
}
