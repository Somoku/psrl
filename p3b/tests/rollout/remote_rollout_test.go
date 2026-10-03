//go:build rollout

// Package rollout verifies the cluster-management claim against two real sandboxd
// processes and a real container runtime.
//
// This is the test the design's step 2.4 exit criterion actually asks for. The
// in-process two-plane test in internal/server covers the wire protocol with a
// fake backend, which proves the framing and the routing but cannot prove the
// thing the step is about: that a control plane on one machine drives a container
// on another, and that the sandbox a caller gets back is indistinguishable from a
// local one.
//
// The difference is not cosmetic. Four classes of failure only appear with real
// processes and a real daemon:
//
//   - A node that builds its admission from its own cgroup hierarchy rather than
//     the control plane's. A fake backend has no pressure, so a node that read the
//     wrong hierarchy would pass the in-process test.
//   - Exec state. A persistent shell lives in the node process; a fake backend
//     returns canned output, so a shell that failed to survive between calls would
//     pass the in-process test.
//   - Lifecycle across a process boundary. A sandbox whose lease expires on the
//     node while the control plane still believes it holds it is a leak that an
//     in-process lifecycle sharing one clock cannot produce.
//   - Release reaching the daemon. A fake backend's Release is a map delete; a
//     container that stayed running after release is invisible to it.
//
// It is behind a build tag because it requires a Docker daemon and starts real
// processes. CI runs the untagged suites; this one runs on a node:
//
//	go test -tags rollout ./tests/rollout/ -v
//
// PSRL_ROLLOUT_IMAGE overrides the image (default alpine:latest). The image must
// already be present — pulling is not this test's job and a pull failure would
// report as a scheduling failure.
package rollout

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const (
	// nodeBootDeadline is how long a node process has to bind its listener. A node
	// that has to reach a daemon takes longer than one that does not, and a boot
	// that exceeds this is a configuration failure rather than a slow machine.
	nodeBootDeadline = 30 * time.Second
	// callDeadline bounds one node-plane call. Generous because a create pays for
	// container construction, which is the cost the whole design is about.
	callDeadline = 120 * time.Second
)

func rolloutImage() string {
	if image := os.Getenv("PSRL_ROLLOUT_IMAGE"); image != "" {
		return image
	}
	return "alpine:latest"
}

func dockerSocket() string {
	if socket := os.Getenv("DOCKER_HOST"); socket != "" {
		return strings.TrimPrefix(socket, "unix://")
	}
	return "/var/run/docker.sock"
}

// -- the node-plane protocol, spoken directly ----------------------------------
//
// The test speaks the wire format rather than importing RemoteNodeClient. Using
// the client would mean a framing bug in the client is invisible here, and this
// test exists to catch exactly the things a shared implementation hides.

// call sends one node-plane request and decodes the reply.
func call(t *testing.T, address, method string, payload map[string]any) map[string]any {
	t.Helper()
	conn, err := net.DialTimeout("tcp", address, 10*time.Second)
	if err != nil {
		t.Fatalf("dial %s for %s: %v", address, method, err)
	}
	defer conn.Close()
	if err := conn.SetDeadline(time.Now().Add(callDeadline)); err != nil {
		t.Fatalf("set deadline: %v", err)
	}

	request := map[string]any{"method": method}
	for key, value := range payload {
		request[key] = value
	}
	body, err := json.Marshal(request)
	if err != nil {
		t.Fatalf("marshal %s: %v", method, err)
	}
	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(body)))
	if _, err := conn.Write(append(header, body...)); err != nil {
		t.Fatalf("write %s: %v", method, err)
	}

	if _, err := io_ReadFull(conn, header); err != nil {
		t.Fatalf("read %s reply header: %v", method, err)
	}
	size := binary.BigEndian.Uint32(header)
	// A frame larger than this is a protocol desync rather than a large reply: the
	// node-plane carries control messages and exec output, not file transfers.
	if size > 64<<20 {
		t.Fatalf("%s reply claims %d bytes, which is a protocol desync", method, size)
	}
	reply := make([]byte, size)
	if _, err := io_ReadFull(conn, reply); err != nil {
		t.Fatalf("read %s reply body: %v", method, err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(reply, &decoded); err != nil {
		t.Fatalf("decode %s reply: %v (raw: %s)", method, err, truncate(string(reply)))
	}
	if errText, failed := decoded["error"].(string); failed && errText != "" {
		t.Fatalf("%s failed on the node: %s", method, errText)
	}
	return decoded
}

// io_ReadFull is io.ReadFull, named so the import list stays minimal and the
// framing is visibly the test's own rather than borrowed from the implementation.
func io_ReadFull(conn net.Conn, buf []byte) (int, error) {
	read := 0
	for read < len(buf) {
		n, err := conn.Read(buf[read:])
		read += n
		if err != nil {
			return read, err
		}
	}
	return read, nil
}

func truncate(s string) string {
	if len(s) <= 400 {
		return s
	}
	return s[:400] + "...(truncated)"
}

// -- bringing up a node process -------------------------------------------------

// nodeProcess is one running sandboxd in "node" role.
type nodeProcess struct {
	nodeID  string
	address string
	cmd     *exec.Cmd
	logPath string
}

// freePort asks the kernel for a port and releases it, so the config can name a
// port the node then binds. A race is possible in principle and has not been
// observed; the alternative is a fixed port, which collides with a previous run.
func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve a port: %v", err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// startNode writes a node-role configuration and starts sandboxd against it.
//
// Overcommit is enabled deliberately: the node reads its own cgroup hierarchy to
// resolve the effective envelope, and a node that found no hierarchy admits at the
// declared envelope. Both outcomes are correct, and the test asserts the node
// reports which one it reached rather than assuming either.
func startNode(t *testing.T, binary, workDir, nodeID string, overcommit float64) *nodeProcess {
	t.Helper()
	port := freePort(t)
	address := fmt.Sprintf("127.0.0.1:%d", port)

	config := map[string]any{
		"role":        "node",
		"node_listen": address,
		"node_id":     nodeID,
		"timing": map[string]any{
			"episode_deadline_s": 1800.0,
			"node_ttl_s":         120.0,
			"rpc_timeout_s":      60.0,
		},
		"node": map[string]any{
			"memory_mb":         4096,
			"cpu_millis":        4000,
			"disk_mb":           20000,
			"local_mem_ceiling": 0.95,
			"overcommit":        overcommit,
			// Below the ceiling, which the configuration validator enforces.
			"utilization_target": 0.85,
		},
		"classes": map[string]any{
			"default": map[string]any{"guaranteed_share": 0.5, "max_share": 1.0},
		},
		"backends": []any{
			map[string]any{
				"type":   "docker",
				"mode":   "direct",
				"socket": dockerSocket(),
			},
		},
		"default_backend": "docker",
		"owner_id":        "psrl-rollout-test",
	}
	if overcommit <= 1 {
		// An unset target with overcommit off is valid, and leaving the target in
		// would not be exercising the off path.
		delete(config["node"].(map[string]any), "utilization_target")
		delete(config["node"].(map[string]any), "overcommit")
	}

	configPath := filepath.Join(workDir, nodeID+".json")
	encoded, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		t.Fatalf("marshal config: %v", err)
	}
	if err := os.WriteFile(configPath, encoded, 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	logPath := filepath.Join(workDir, nodeID+".log")
	logFile, err := os.Create(logPath)
	if err != nil {
		t.Fatalf("create log: %v", err)
	}
	cmd := exec.Command(binary, "-config", configPath)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	if err := cmd.Start(); err != nil {
		t.Fatalf("start node %s: %v", nodeID, err)
	}

	proc := &nodeProcess{nodeID: nodeID, address: address, cmd: cmd, logPath: logPath}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		logFile.Close()
		// The log is the only account of a node that failed to boot, so it is
		// surfaced rather than discarded on failure.
		if t.Failed() {
			if contents, readErr := os.ReadFile(logPath); readErr == nil {
				t.Logf("--- %s log ---\n%s", nodeID, contents)
			}
		}
	})

	waitForListener(t, address, proc)
	return proc
}

// waitForListener blocks until the node answers, or fails with its log.
func waitForListener(t *testing.T, address string, proc *nodeProcess) {
	t.Helper()
	deadline := time.Now().Add(nodeBootDeadline)
	for time.Now().Before(deadline) {
		// A process that exited is a configuration failure, and waiting out the
		// deadline would report it as a timeout instead of showing the reason.
		if proc.cmd.ProcessState != nil && proc.cmd.ProcessState.Exited() {
			contents, _ := os.ReadFile(proc.logPath)
			t.Fatalf("node %s exited before binding %s:\n%s", proc.nodeID, address, contents)
		}
		conn, err := net.DialTimeout("tcp", address, time.Second)
		if err == nil {
			conn.Close()
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
	contents, _ := os.ReadFile(proc.logPath)
	t.Fatalf("node %s did not bind %s within %s:\n%s", proc.nodeID, address, nodeBootDeadline, contents)
}

// buildSandboxd compiles the service once for the whole suite.
func buildSandboxd(t *testing.T) string {
	t.Helper()
	outDir := t.TempDir()
	binary := filepath.Join(outDir, "sandboxd")
	// The test lives two directories below the module root.
	cmd := exec.Command("go", "build", "-o", binary, "./cmd/sandboxd")
	cmd.Dir = "../.."
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("build sandboxd: %v\n%s", err, output)
	}
	return binary
}

// -- the specs the test places --------------------------------------------------

func baseSpec(workflowID, execMode string) map[string]any {
	return map[string]any{
		"source_kind":      "image",
		"source_reference": rolloutImage(),
		"resources":        map[string]any{"memory_mb": 256, "cpu_count": 0.5},
		"resource_class":   "default",
		"workflow_id":      workflowID,
		"exec_mode":        execMode,
		"workdir":          "/",
	}
}

// place runs admit then create_on, returning the handle fields.
func place(t *testing.T, address string, spec map[string]any) (sandboxID, nodeID string) {
	t.Helper()
	admitted := call(t, address, "admit", map[string]any{"spec": spec})
	leaseID, _ := admitted["lease_id"].(string)
	if refusal, _ := admitted["refusal"].(string); refusal != "" {
		t.Fatalf("the node refused a request inside its envelope: %s", refusal)
	}
	if leaseID == "" {
		t.Fatalf("admit returned no lease: %v", admitted)
	}
	created := call(t, address, "create_on", map[string]any{
		"lease_id": leaseID,
		"backend":  "docker",
		"spec":     spec,
	})
	handle, ok := created["handle"].(map[string]any)
	if !ok {
		t.Fatalf("create_on returned no handle: %v", created)
	}
	sandboxID, _ = handle["sandbox_id"].(string)
	nodeID, _ = handle["node_id"].(string)
	if sandboxID == "" {
		t.Fatalf("create_on returned an empty sandbox id: %v", handle)
	}
	return sandboxID, nodeID
}

func handleOf(backendName, sandboxID, nodeID string) map[string]any {
	return map[string]any{"backend": backendName, "sandbox_id": sandboxID, "node_id": nodeID}
}

func execOn(t *testing.T, address, sandboxID, nodeID, command string) (int, string) {
	t.Helper()
	reply := call(t, address, "exec", map[string]any{
		"handle":  handleOf("docker", sandboxID, nodeID),
		"command": command,
	})
	code := 0
	if raw, present := reply["exit_code"].(float64); present {
		code = int(raw)
	}
	stdout, _ := reply["stdout"].(string)
	return code, stdout
}

func release(t *testing.T, address, sandboxID, nodeID string) {
	t.Helper()
	call(t, address, "release_on", map[string]any{"handle": handleOf("docker", sandboxID, nodeID)})
}

func statusOf(t *testing.T, address, sandboxID, nodeID string) string {
	t.Helper()
	reply := call(t, address, "status_on", map[string]any{"handle": handleOf("docker", sandboxID, nodeID)})
	state, _ := reply["status"].(string)
	return state
}

// containerExists asks the daemon directly.
//
// This is the assertion the in-process test cannot make: whether release actually
// reached the runtime, rather than whether the service's own map forgot the entry.
func containerExists(t *testing.T, sandboxID string) bool {
	t.Helper()
	cmd := exec.Command("docker", "inspect", "--format", "{{.State.Status}}", sandboxID)
	output, err := cmd.CombinedOutput()
	if err != nil {
		// inspect fails when the container is gone, which is the expected outcome
		// after a release.
		return false
	}
	return strings.TrimSpace(string(output)) != ""
}

// -- the tests ------------------------------------------------------------------

// TestARemoteSandboxRunsCommandsAndIsDestroyed is the step 2.4 exit criterion.
func TestARemoteSandboxRunsCommandsAndIsDestroyed(t *testing.T) {
	binary := buildSandboxd(t)
	node := startNode(t, binary, t.TempDir(), "rollout-node-1", 0)

	sandboxID, nodeID := place(t, node.address, baseSpec("wf-remote-basic", "one_shot"))
	if nodeID != "rollout-node-1" {
		t.Errorf("the handle names node %q, want rollout-node-1: a caller routes later calls on this field",
			nodeID)
	}
	t.Cleanup(func() { release(t, node.address, sandboxID, nodeID) })

	if state := statusOf(t, node.address, sandboxID, nodeID); state != "running" {
		t.Fatalf("a freshly created sandbox reports %q, want running", state)
	}

	code, stdout := execOn(t, node.address, sandboxID, nodeID, "echo remote-ok")
	if code != 0 {
		t.Errorf("exec exit code %d, want 0 (stdout: %q)", code, stdout)
	}
	if !strings.Contains(stdout, "remote-ok") {
		t.Errorf("exec stdout %q does not contain the echoed marker", stdout)
	}

	// A non-zero exit must cross the wire as a status rather than an error: a
	// grader reads the code, and a transport that turned it into a failure would
	// make every failed test look like a broken sandbox.
	code, _ = execOn(t, node.address, sandboxID, nodeID, "exit 7")
	if code != 7 {
		t.Errorf("a command exiting 7 reported %d: the exit code is data, not a transport failure", code)
	}

	if !containerExists(t, sandboxID) {
		t.Error("the daemon does not know the container the node reported creating")
	}
	release(t, node.address, sandboxID, nodeID)
	// The daemon's own view, not the service's: a release that updated the
	// accounting without reaching the runtime is a leak that grows per episode.
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) && containerExists(t, sandboxID) {
		time.Sleep(500 * time.Millisecond)
	}
	if containerExists(t, sandboxID) {
		t.Error("the container still exists after release, so the release did not reach the runtime")
	}
}

// TestAPersistentShellKeepsStateBetweenCalls is what one_shot cannot do.
//
// This is the exec-mode half of the design's step 1.1, verified where it matters:
// in a node process driving a real container, rather than against a fake that
// returns canned output.
func TestAPersistentShellKeepsStateBetweenCalls(t *testing.T) {
	binary := buildSandboxd(t)
	node := startNode(t, binary, t.TempDir(), "rollout-node-shell", 0)

	sandboxID, nodeID := place(t, node.address, baseSpec("wf-remote-shell", "persistent"))
	t.Cleanup(func() { release(t, node.address, sandboxID, nodeID) })

	// A working directory set in one call must still be in effect in the next,
	// which is the property a multi-turn harness depends on.
	if code, stdout := execOn(t, node.address, sandboxID, nodeID, "mkdir -p /tmp/persist && cd /tmp/persist"); code != 0 {
		t.Fatalf("the setup command failed with %d: %q", code, stdout)
	}
	code, stdout := execOn(t, node.address, sandboxID, nodeID, "pwd")
	if code != 0 {
		t.Fatalf("pwd failed with %d: %q", code, stdout)
	}
	if !strings.Contains(stdout, "/tmp/persist") {
		t.Errorf("pwd reports %q, want /tmp/persist: the shell did not keep its working directory, "+
			"so a harness turn cannot build on the previous one", strings.TrimSpace(stdout))
	}

	// An exported variable is the other half: a venv activation is an export, and
	// a shell that forgot it would make every subsequent command run against the
	// wrong interpreter.
	if code, stdout := execOn(t, node.address, sandboxID, nodeID, "export ROLLOUT_MARKER=kept"); code != 0 {
		t.Fatalf("export failed with %d: %q", code, stdout)
	}
	code, stdout = execOn(t, node.address, sandboxID, nodeID, "echo $ROLLOUT_MARKER")
	if !strings.Contains(stdout, "kept") {
		t.Errorf("the exported variable reads back as %q, want kept", strings.TrimSpace(stdout))
	}

	// The exit code must still be the command's own, not the shell's.
	if code, _ = execOn(t, node.address, sandboxID, nodeID, "exit 3"); code != 3 {
		t.Errorf("a persistent shell reported %d for a command exiting 3: the sentinel is not "+
			"carrying the status", code)
	}
	// And the shell must survive a command that ended non-zero.
	if code, stdout = execOn(t, node.address, sandboxID, nodeID, "echo still-alive"); code != 0 ||
		!strings.Contains(stdout, "still-alive") {
		t.Errorf("the shell did not survive a failed command: code %d, stdout %q", code, stdout)
	}
}

// TestOneShotDoesNotKeepState is the control for the previous test.
//
// Without it, a persistent shell that was silently the only mode would pass, and
// the mode selection the workloads rely on would be untested.
func TestOneShotDoesNotKeepState(t *testing.T) {
	binary := buildSandboxd(t)
	node := startNode(t, binary, t.TempDir(), "rollout-node-oneshot", 0)

	sandboxID, nodeID := place(t, node.address, baseSpec("wf-remote-oneshot", "one_shot"))
	t.Cleanup(func() { release(t, node.address, sandboxID, nodeID) })

	if code, _ := execOn(t, node.address, sandboxID, nodeID, "export ROLLOUT_MARKER=leaked"); code != 0 {
		t.Fatal("the export command itself failed")
	}
	_, stdout := execOn(t, node.address, sandboxID, nodeID, "echo [$ROLLOUT_MARKER]")
	if strings.Contains(stdout, "leaked") {
		t.Errorf("a one-shot exec kept state across calls (%q). Each call must start a fresh "+
			"process, or a grader step inherits whatever the rollout left behind", strings.TrimSpace(stdout))
	}
}

// TestTwoNodesHoldTheirOwnSandboxes verifies the fleet is actually partitioned.
func TestTwoNodesHoldTheirOwnSandboxes(t *testing.T) {
	binary := buildSandboxd(t)
	workDir := t.TempDir()
	first := startNode(t, binary, workDir, "rollout-node-a", 0)
	second := startNode(t, binary, workDir, "rollout-node-b", 0)

	firstID, firstNode := place(t, first.address, baseSpec("wf-fleet-a", "one_shot"))
	t.Cleanup(func() { release(t, first.address, firstID, firstNode) })
	secondID, secondNode := place(t, second.address, baseSpec("wf-fleet-b", "one_shot"))
	t.Cleanup(func() { release(t, second.address, secondID, secondNode) })

	if firstID == secondID {
		t.Fatal("two nodes produced the same sandbox id")
	}
	if firstNode != "rollout-node-a" || secondNode != "rollout-node-b" {
		t.Errorf("handles name %q and %q, want rollout-node-a and rollout-node-b", firstNode, secondNode)
	}

	// Each sandbox is reachable on its own node and only there. A node asked about
	// a sandbox it does not hold must say so rather than answering about a
	// different one, which is what makes the id-to-node mapping load-bearing.
	if state := statusOf(t, first.address, firstID, firstNode); state != "running" {
		t.Errorf("node a reports %q for its own sandbox, want running", state)
	}
	if state := statusOf(t, first.address, secondID, firstNode); state == "running" {
		t.Error("node a reports running for a sandbox that lives on node b")
	}

	// Writing on one node must not be visible on the other: the two sandboxes are
	// separate containers, and a shared mount would make an episode's work leak
	// into a sibling's.
	if code, _ := execOn(t, first.address, firstID, firstNode, "echo a > /tmp/which-node"); code != 0 {
		t.Fatal("writing a marker on node a failed")
	}
	if code, _ := execOn(t, second.address, secondID, secondNode, "echo b > /tmp/which-node"); code != 0 {
		t.Fatal("writing a marker on node b failed")
	}
	_, stdout := execOn(t, first.address, firstID, firstNode, "cat /tmp/which-node")
	if !strings.Contains(stdout, "a") || strings.Contains(stdout, "b") {
		t.Errorf("node a's sandbox reads %q, want its own marker: the two sandboxes share state",
			strings.TrimSpace(stdout))
	}
}

// TestTheNodeReportsWhatItWillAdmit verifies the report the fleet view is built from.
func TestTheNodeReportsWhatItWillAdmit(t *testing.T) {
	binary := buildSandboxd(t)
	node := startNode(t, binary, t.TempDir(), "rollout-node-report", 0)

	reported := call(t, node.address, "report", nil)
	view, ok := reported["view"].(map[string]any)
	if !ok {
		t.Fatalf("report returned no view: %v", reported)
	}
	if id, _ := view["node_id"].(string); id != "rollout-node-report" {
		t.Errorf("the report names node %q, want rollout-node-report", id)
	}
	// A node that reports no class headroom gives placement nothing to compare a
	// request against, so every placement decision would be a guess.
	headroom, present := view["class_headroom"].(map[string]any)
	if !present || len(headroom) == 0 {
		t.Error("the report carries no per-class headroom, so placement has nothing to compare against")
	}
	if _, declared := headroom["default"]; !declared {
		t.Errorf("the report omits the default class, which every request without an explicit "+
			"class is admitted under: %v", headroom)
	}
}

// TestOvercommitExpandsAdmissionOnALightlyLoadedNode is the step 4.2 criterion.
//
// The assertion is relative rather than absolute: the expansion depends on what
// the node's own cgroup hierarchy reports, so an absolute figure would encode this
// machine's memory usage into the test. What must hold on any node is that a node
// configured to overcommit reports at least as much headroom as the same node
// configured not to, and strictly more when its measurement is readable and below
// the target.
func TestOvercommitExpandsAdmissionOnALightlyLoadedNode(t *testing.T) {
	binary := buildSandboxd(t)
	workDir := t.TempDir()
	plain := startNode(t, binary, workDir, "rollout-node-plain", 0)
	expanded := startNode(t, binary, workDir, "rollout-node-overcommit", 2.0)

	plainHeadroom := defaultClassMemory(t, plain.address)
	expandedHeadroom := defaultClassMemory(t, expanded.address)

	if expandedHeadroom < plainHeadroom {
		t.Fatalf("the overcommitted node reports %d MB against the plain node's %d MB: "+
			"overcommit must never reduce what a node will admit", expandedHeadroom, plainHeadroom)
	}
	if expandedHeadroom == plainHeadroom {
		// Correct and expected on a node with no readable cgroup hierarchy, or one
		// already past the utilization target. Reported rather than failed, because
		// the design's own rule is that an unreadable measurement does not expand.
		t.Logf("overcommit resolved to no expansion (%d MB both ways). This is correct when the "+
			"node's cgroup hierarchy is unreadable or its memory use is already past the 0.85 "+
			"target; it means this machine cannot demonstrate the density gain.", plainHeadroom)
		return
	}
	t.Logf("overcommit expanded the admissible memory from %d MB to %d MB (%.2fx)",
		plainHeadroom, expandedHeadroom, float64(expandedHeadroom)/float64(plainHeadroom))

	// The expansion has to be usable, not merely reported: a request that the plain
	// node would refuse must be admitted here.
	spec := baseSpec("wf-overcommit", "one_shot")
	spec["resources"] = map[string]any{"memory_mb": plainHeadroom + 128, "cpu_count": 0.5}
	admitted := call(t, expanded.address, "admit", map[string]any{"spec": spec})
	if refusal, _ := admitted["refusal"].(string); refusal != "" {
		t.Errorf("the overcommitted node refused %d MB (%s) while reporting %d MB of headroom",
			plainHeadroom+128, refusal, expandedHeadroom)
	}
}

// defaultClassMemory reads the default class's admissible memory from a report.
func defaultClassMemory(t *testing.T, address string) int64 {
	t.Helper()
	reported := call(t, address, "report", nil)
	view, ok := reported["view"].(map[string]any)
	if !ok {
		t.Fatalf("report returned no view: %v", reported)
	}
	headroom, ok := view["class_headroom"].(map[string]any)
	if !ok {
		t.Fatalf("report returned no class headroom: %v", view)
	}
	class, ok := headroom["default"].(map[string]any)
	if !ok {
		t.Fatalf("report has no default class: %v", headroom)
	}
	memory, _ := class["memory_mb"].(float64)
	return int64(memory)
}

// TestAnOversizedRequestIsRefusedRatherThanFailingLate verifies admission is the gate.
//
// A request larger than the node can hold must be refused at admission, where the
// control plane can place it elsewhere. Letting it through and failing at create
// would spend container construction on a request that was never going to fit,
// and report a scheduling decision as a runtime fault.
func TestAnOversizedRequestIsRefusedRatherThanFailingLate(t *testing.T) {
	binary := buildSandboxd(t)
	node := startNode(t, binary, t.TempDir(), "rollout-node-refuse", 0)

	spec := baseSpec("wf-oversized", "one_shot")
	spec["resources"] = map[string]any{"memory_mb": 1 << 20, "cpu_count": 0.5}

	// Not through place(): a refusal is the expected outcome here, and place fails
	// the test on one.
	admitted := call(t, node.address, "admit", map[string]any{"spec": spec})
	refusal, _ := admitted["refusal"].(string)
	if refusal == "" {
		t.Fatalf("a request for 1 TB was admitted against a 4 GB node: %v", admitted)
	}
	if leaseID, _ := admitted["lease_id"].(string); leaseID != "" {
		t.Error("a refused request still returned a lease, which would be charged against the node")
	}
	t.Logf("the node refused an oversized request with %q, which is the reason the control "+
		"plane reads to place elsewhere", refusal)
}
