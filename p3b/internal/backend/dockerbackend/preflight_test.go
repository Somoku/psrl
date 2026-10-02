//go:build integration

package dockerbackend

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// Preflight is the one check that runs before a deployment serves, so what it
// refuses and what it lets through are both worth pinning.

func TestAPIVersionsCompareNumericallyRatherThanAsText(t *testing.T) {
	// The string comparison is wrong exactly where Docker's versions actually
	// live: "v1.9" sorts after "v1.40" lexically, which would refuse a modern
	// daemon for being too old.
	cases := []struct {
		version, floor string
		older          bool
	}{
		{"v1.9", "v1.40", true},
		{"v1.40", "v1.9", false},
		{"v1.40", "v1.40", false},
		{"v1.41", "v1.40", false},
		{"v1.40", "v1.41", true},
		{"1.40", "v1.40", false},
		{"v1.4", "v1.40", true},
	}
	for _, c := range cases {
		if got := olderAPI(c.version, c.floor); got != c.older {
			t.Errorf("olderAPI(%q, %q) = %v, want %v", c.version, c.floor, got, c.older)
		}
	}
}

func TestAnUncomparableVersionNeverRefusesADaemon(t *testing.T) {
	// A version this parser does not understand is not evidence the daemon is too
	// old, and refusing on it would take down a deployment that worked.
	for _, version := range []string{"", "v1.40-beta", "latest", "v1.x"} {
		if olderAPI(version, "v1.40") {
			t.Errorf("olderAPI(%q, \"v1.40\") refused a version it cannot compare", version)
		}
		if olderAPI("v1.40", version) {
			t.Errorf("olderAPI(\"v1.40\", %q) refused a floor it cannot compare", version)
		}
	}
}

func TestPreflightRefusesARuntimeTheNodeDoesNotHave(t *testing.T) {
	// This backend declares isolation_runtime whenever a runtime is configured,
	// so an absent one means admitting a spec that asked for a stronger boundary
	// and then running it without one.
	socket := liveSocket(t)
	b, err := New(Config{
		Socket: socket, APIVersion: "v1.40", NodeID: "test-node",
		OwnerID:        fmt.Sprintf("sandboxd-preflight-%d", time.Now().UnixNano()),
		Runtime:        "runtime-that-is-not-installed",
		RequestTimeout: 30 * time.Second,
	}, backend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}

	err = b.Preflight(context.Background())

	if err == nil {
		t.Fatal("preflight must refuse a runtime the daemon does not have")
	}
	// The message has to name the setting, because that is the whole value of
	// failing here instead of on the first create.
	if !strings.Contains(err.Error(), "runtime-that-is-not-installed") {
		t.Fatalf("the refusal must name the runtime that was configured, got %v", err)
	}
	if !strings.Contains(err.Error(), "isolation_runtime") {
		t.Fatalf("the refusal must say what capability it would have falsified, got %v", err)
	}
}

func TestPreflightAcceptsTheRuntimeEveryDaemonHas(t *testing.T) {
	socket := liveSocket(t)
	b, err := New(Config{
		Socket: socket, APIVersion: "v1.40", NodeID: "test-node",
		OwnerID:        fmt.Sprintf("sandboxd-preflight-%d", time.Now().UnixNano()),
		Runtime:        "runc",
		RequestTimeout: 30 * time.Second,
	}, backend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}

	if err := b.Preflight(context.Background()); err != nil {
		t.Fatalf("runc is installed on every daemon, so preflight must pass: %v", err)
	}
}

func TestPreflightRefusesAnAPIVersionTheDaemonWillNotServe(t *testing.T) {
	// A version far beyond anything released: the daemon reports a lower ceiling,
	// and this has to fail at startup rather than on an endpoint that silently
	// did not exist.
	socket := liveSocket(t)
	b, err := New(Config{
		Socket: socket, APIVersion: "v9.99", NodeID: "test-node",
		OwnerID:        fmt.Sprintf("sandboxd-preflight-%d", time.Now().UnixNano()),
		RequestTimeout: 30 * time.Second,
	}, backend.SchedulingPSRL)
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}

	err = b.Preflight(context.Background())

	if err == nil {
		t.Fatal("preflight must refuse an API version the daemon does not serve")
	}
	if !strings.Contains(err.Error(), "api_version") {
		t.Fatalf("the refusal must name the setting to change, got %v", err)
	}
}

// liveSocket returns the node's Docker socket, skipping where there is none.
func liveSocket(t *testing.T) string {
	t.Helper()
	socket := os.Getenv("DOCKER_SOCKET")
	if socket == "" {
		socket = "/var/run/docker.sock"
	}
	if _, err := os.Stat(socket); err != nil {
		t.Skipf("no docker socket at %s", socket)
	}
	return socket
}
