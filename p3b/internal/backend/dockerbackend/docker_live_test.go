//go:build integration

package dockerbackend

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// These run against a real daemon, because the Engine API is the contract under
// test: a fake would only prove this package agrees with my model of Docker.
// They are skipped where no socket is present, so the suite still runs anywhere.

const testImage = "alpine:latest"

func liveBackend(t *testing.T) *Backend {
	t.Helper()
	socket := os.Getenv("DOCKER_SOCKET")
	if socket == "" {
		socket = "/var/run/docker.sock"
	}
	if _, err := os.Stat(socket); err != nil {
		t.Skipf("no docker socket at %s", socket)
	}
	b, err := New(Config{
		Socket:     socket,
		APIVersion: "v1.40",
		NodeID:     "test-node",
		// Scoped to this test run, so a sweep here can never see a production
		// sandbox and nothing this test makes is mistaken for one.
		OwnerID:        fmt.Sprintf("sandboxd-test-%d", time.Now().UnixNano()),
		RequestTimeout: 30 * time.Second,
		PullTimeout:    3 * time.Minute,
	}, backend.SchedulingDirect)
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}
	if err := b.Preflight(context.Background()); err != nil {
		t.Skipf("daemon not usable: %v", err)
	}
	return b
}

func spec() backend.Spec {
	return backend.Spec{
		Source:        backend.Source{Kind: "image", Reference: testImage},
		Resources:     backend.Resources{MemoryMB: 64, CPUCount: 0.5},
		ResourceClass: "rollout",
	}
}

func createOne(t *testing.T, b *Backend) backend.Created {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	created, err := b.Create(ctx, "test-node", spec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	t.Cleanup(func() {
		releaseCtx, releaseCancel := context.WithTimeout(context.Background(), time.Minute)
		defer releaseCancel()
		if err := b.Release(releaseCtx, created.Handle); err != nil {
			t.Errorf("cleanup release: %v", err)
		}
	})
	return created
}

func TestPreflightRejectsAMissingDaemon(t *testing.T) {
	b, err := New(Config{Socket: "/nonexistent/docker.sock", RequestTimeout: 2 * time.Second}, backend.SchedulingDirect)
	if err != nil {
		t.Fatalf("new backend: %v", err)
	}

	if err := b.Preflight(context.Background()); err == nil {
		t.Fatal("preflight must refuse a node whose daemon cannot be reached")
	}
}

func TestACreatedSandboxIsRunningAndRunsACommand(t *testing.T) {
	b := liveBackend(t)
	created := createOne(t, b)
	ctx := context.Background()

	status, err := b.Status(ctx, created.Handle)
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if status != "running" {
		t.Fatalf("status is %q, want running", status)
	}

	code, output, err := b.Exec(ctx, created.Handle, "echo hello-sandbox", "", nil)
	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	if code != 0 {
		t.Fatalf("exit code %d, want 0 (output %q)", code, output)
	}
	if !strings.Contains(output, "hello-sandbox") {
		t.Fatalf("output %q does not carry the command's stdout", output)
	}
}

func TestExecReportsANonZeroExitRatherThanAnError(t *testing.T) {
	// A failing command is a result, not a transport fault: reading it as an error
	// would make a task failure indistinguishable from a broken node.
	b := liveBackend(t)
	created := createOne(t, b)

	code, _, err := b.Exec(context.Background(), created.Handle, "exit 42", "", nil)

	if err != nil {
		t.Fatalf("a non-zero exit must not be an error: %v", err)
	}
	if code != 42 {
		t.Fatalf("exit code %d, want 42", code)
	}
}

func TestExecCarriesTheWorkingDirectoryAndEnvironment(t *testing.T) {
	b := liveBackend(t)
	created := createOne(t, b)

	_, output, err := b.Exec(context.Background(), created.Handle, "pwd && echo $SANDBOX_MARK", "/tmp",
		map[string]string{"SANDBOX_MARK": "carried"})

	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	if !strings.Contains(output, "/tmp") || !strings.Contains(output, "carried") {
		t.Fatalf("output %q is missing the working directory or the environment", output)
	}
}

func TestOutputIsDemultiplexedRatherThanFramed(t *testing.T) {
	// An attached exec returns a framed stream; passing the frames through would
	// put binary headers into a command's output.
	b := liveBackend(t)
	created := createOne(t, b)

	_, output, err := b.Exec(context.Background(), created.Handle, "printf 'clean'", "", nil)

	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	if output != "clean" {
		t.Fatalf("output is %q, want exactly \"clean\"", output)
	}
}

func TestAFreezeKeepsTheSandboxResidentAndResumable(t *testing.T) {
	b := liveBackend(t)
	created := createOne(t, b)
	ctx := context.Background()

	if err := b.Pause(ctx, created.Handle, "freeze"); err != nil {
		t.Fatalf("pause: %v", err)
	}
	status, err := b.Status(ctx, created.Handle)
	if err != nil {
		t.Fatalf("status while paused: %v", err)
	}
	if status != "paused" {
		t.Fatalf("status is %q, want paused", status)
	}

	if err := b.Resume(ctx, created.Handle); err != nil {
		t.Fatalf("resume: %v", err)
	}
	if code, _, err := b.Exec(ctx, created.Handle, "true", "", nil); err != nil || code != 0 {
		t.Fatalf("a resumed sandbox must run commands: code=%d err=%v", code, err)
	}
}

func TestAHibernationIsRefusedBecauseDockerKeepsTheContainerResident(t *testing.T) {
	// Claiming a hibernation would promise that compute was released, which a
	// freeze does not do.
	b := liveBackend(t)
	created := createOne(t, b)

	if err := b.Pause(context.Background(), created.Handle, "hibernate"); err == nil {
		t.Fatal("docker offers a freeze, so a hibernation must be refused")
	}
}

func TestASnapshotCommitsTheFilesystemAndCanBeDeleted(t *testing.T) {
	b := liveBackend(t)
	created := createOne(t, b)
	ctx := context.Background()
	if code, _, err := b.Exec(ctx, created.Handle, "echo marked > /tmp/mark", "", nil); err != nil || code != 0 {
		t.Fatalf("writing the marker: code=%d err=%v", code, err)
	}

	snapshotID, err := b.Snapshot(ctx, created.Handle, "filesystem")
	if err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	if snapshotID == "" {
		t.Fatal("a snapshot must return an id a restore can name")
	}

	if err := b.DeleteSnapshot(ctx, snapshotID); err != nil {
		t.Fatalf("delete snapshot: %v", err)
	}
}

func TestAFullStateSnapshotIsRefused(t *testing.T) {
	// Docker commits the writable layer and has no supported memory checkpoint, so
	// a full-state claim would let a caller read a workspace restore as proof that
	// a live process survived.
	b := liveBackend(t)
	created := createOne(t, b)

	if _, err := b.Snapshot(context.Background(), created.Handle, "full_state"); err == nil {
		t.Fatal("docker cannot capture memory, so full_state must be refused")
	}
}

func TestReleaseIsIdempotent(t *testing.T) {
	// A retried release is normal: the first may have timed out after the daemon
	// already removed the container.
	b := liveBackend(t)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	created, err := b.Create(ctx, "test-node", spec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	if err := b.Release(ctx, created.Handle); err != nil {
		t.Fatalf("first release: %v", err)
	}
	if err := b.Release(ctx, created.Handle); err != nil {
		t.Fatalf("a repeated release must be a no-op, got %v", err)
	}
}

func TestStatusOfAReleasedSandboxIsTerminatedRatherThanAnError(t *testing.T) {
	b := liveBackend(t)
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()
	created, err := b.Create(ctx, "test-node", spec(), "")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	if err := b.Release(ctx, created.Handle); err != nil {
		t.Fatalf("release: %v", err)
	}

	status, err := b.Status(ctx, created.Handle)

	if err != nil {
		t.Fatalf("a gone sandbox is a state, not an error: %v", err)
	}
	if status != "terminated" {
		t.Fatalf("status is %q, want terminated", status)
	}
}

func TestASweepFindsThisServiceOwnSandboxesFromTheDaemon(t *testing.T) {
	// The daemon is the source of truth: a service that trusted its own list would
	// leak every container it forgot across a restart.
	b := liveBackend(t)
	created := createOne(t, b)

	found, err := b.Sweep(context.Background())
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}

	var matched bool
	for _, handle := range found {
		if handle.SandboxID == created.Handle.SandboxID {
			matched = true
			if handle.NodeID != "test-node" {
				t.Fatalf("swept handle carries node %q, want test-node", handle.NodeID)
			}
		}
	}
	if !matched {
		t.Fatalf("sweep returned %d handles, none of them the live sandbox", len(found))
	}
}

func TestTheMemoryLimitIsAppliedToTheContainer(t *testing.T) {
	// An unenforced limit is worse than none: the node's accounting would believe
	// a bound the kernel is not keeping.
	b := liveBackend(t)
	created := createOne(t, b)

	_, output, err := b.Exec(context.Background(), created.Handle,
		"cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes", "", nil)

	if err != nil {
		t.Fatalf("exec: %v", err)
	}
	// 64MB was requested; the cgroup must report that rather than the host total.
	if !strings.Contains(output, "67108864") {
		t.Fatalf("cgroup reports %q, want the 64MB limit (67108864)", strings.TrimSpace(output))
	}
}

func TestConcurrentCreatesAndReleasesAllSucceed(t *testing.T) {
	// Creates arrive in bursts, so the backend has to hold up under them rather
	// than only in sequence.
	b := liveBackend(t)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	// Warm the image once, so the burst measures creates and not one pull.
	if err := b.ensureImage(ctx, testImage); err != nil {
		t.Fatalf("pull: %v", err)
	}

	const n = 12
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			created, err := b.Create(ctx, "test-node", spec(), "")
			if err != nil {
				errs <- fmt.Errorf("create: %w", err)
				return
			}
			if code, _, err := b.Exec(ctx, created.Handle, "true", "", nil); err != nil || code != 0 {
				errs <- fmt.Errorf("exec: code=%d err=%w", code, err)
			}
			if err := b.Release(ctx, created.Handle); err != nil {
				errs <- fmt.Errorf("release: %w", err)
			}
		}()
	}
	wg.Wait()
	close(errs)

	for err := range errs {
		t.Errorf("concurrent lifecycle: %v", err)
	}

	// Nothing of this run's may be left behind.
	left, err := b.Sweep(ctx)
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if len(left) != 0 {
		t.Fatalf("%d sandboxes survived the burst", len(left))
	}
}
