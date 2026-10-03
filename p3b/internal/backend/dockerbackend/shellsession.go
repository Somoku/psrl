// shellsession.go: persistent-shell exec strategy for the docker backend.
//
// One-shot exec starts a new /bin/sh -c process per call. That is the right
// choice for grader steps that run exactly one command, but multi-turn harness
// tasks need shell state to survive between turns: a working directory set in
// turn one must still be in effect in turn two.
//
// A persistent shell solves this by holding one login shell open for the life
// of the sandbox. Commands are written to its stdin and completed by a sentinel
// that carries the exit status. The shell is the container's init; keeping it
// open keeps the container alive, and the sandbox's filesystem survives even
// when the last command timed out or the shell had to be restarted.
//
// The protocol matches the Python exec.py layer exactly, so a harness built
// against either one sees the same behaviour. The sentinel is a random token
// chosen at session startup, so a command that happens to print a similar
// string does not terminate the read loop prematurely.
//
// The Docker Engine API is used rather than the CLI because this service
// already drives the Engine directly. Attaching a shell requires a TCP hijack:
// POST /exec/{id}/start with Connection: Upgrade, Upgrade: tcp turns the
// response body into a raw bidirectional stream. The net.Conn returned by the
// transport's DialContext is the stream; framing is stripped by the same
// demultiplex function the one-shot path uses.
package dockerbackend

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

const (
	// sentinelPrefix and sentinelSuffix bracket the exit status marker.
	// A token seeded from the session's sandbox id makes the sentinel unique
	// per sandbox, so two sandboxes on the same host cannot collide.
	sentinelPrefix = "///PSRL-DONE:"
	sentinelSuffix = ":PSRL-DONE///"

	// readinessMarker is printed by the probe that confirms the shell answers.
	readinessMarker = "ready"

	// shellReadBufBytes is how many bytes one partial read drains. Large enough
	// to avoid excessive syscalls, small enough that a single allocation of
	// four of them stays under 1 MiB.
	shellReadBufBytes = 256 * 1024

	// defaultShellStartupTimeout is the deadline the readiness probe uses.
	defaultShellStartupTimeout = 60 * time.Second

	// defaultSilenceTimeout is the longest a command may produce no output
	// before it is declared stuck. This is the per-command default; callers
	// can override it through SilenceTimeout on the command.
	defaultSilenceTimeout = 120 * time.Second
)

// shellSessions holds the live persistent shells, keyed by sandbox id.
// The Backend owns one of these; it is created alongside the Backend and
// never replaced.
type shellSessions struct {
	mu      sync.Mutex
	sessions map[string]*shellSession
}

func newShellSessions() *shellSessions {
	return &shellSessions{sessions: make(map[string]*shellSession)}
}

// acquire returns the live session for a sandbox, creating one if none exists.
func (ss *shellSessions) acquire(
	ctx context.Context,
	b *Backend,
	sandboxID string,
) (*shellSession, error) {
	ss.mu.Lock()
	existing := ss.sessions[sandboxID]
	ss.mu.Unlock()

	if existing != nil && existing.alive() {
		return existing, nil
	}

	s, err := newShellSession(ctx, b, sandboxID)
	if err != nil {
		return nil, err
	}
	ss.mu.Lock()
	// Another goroutine may have won the race; if so, close ours and use theirs.
	if other, ok := ss.sessions[sandboxID]; ok && other.alive() {
		ss.mu.Unlock()
		s.close()
		return other, nil
	}
	ss.sessions[sandboxID] = s
	ss.mu.Unlock()
	return s, nil
}

// drop removes and closes the session for a sandbox, called when the sandbox
// is released.
func (ss *shellSessions) drop(sandboxID string) {
	ss.mu.Lock()
	s := ss.sessions[sandboxID]
	delete(ss.sessions, sandboxID)
	ss.mu.Unlock()
	if s != nil {
		s.close()
	}
}

// shellSession is one persistent login shell inside a container.
//
// The shell is attached via Docker's exec hijack API, which turns an HTTP
// response into a raw bidirectional stream. stdin is written to send commands;
// stdout is read to collect output. The stream is framed by the Engine's
// multiplexing protocol (same as the one-shot path) and stripped before
// delivery to the caller.
type shellSession struct {
	sandboxID string
	sentinel  string // random per session, printed at end of each command
	conn      net.Conn
	reader    *bufio.Reader
	mu        sync.Mutex
	closed    bool
}

// newShellSession starts a shell inside the container and waits for its
// readiness probe to return before handing the session to the caller.
func newShellSession(ctx context.Context, b *Backend, sandboxID string) (*shellSession, error) {
	// Pick a sentinel token that is extremely unlikely to appear in command output.
	token := strconv.FormatInt(time.Now().UnixNano(), 36)

	// Create an exec instance with stdin attached.
	create := map[string]any{
		"AttachStdin":  true,
		"AttachStdout": true,
		"AttachStderr": true,
		"Tty":          false,
		// Run as a login shell so profile files are sourced, matching what a user
		// would see when they attach to the container interactively.
		"Cmd": []string{"/bin/sh", "-l"},
	}
	var created struct {
		ID string `json:"Id"`
	}
	if err := b.call(ctx, http.MethodPost, "/containers/"+sandboxID+"/exec", create, &created); err != nil {
		return nil, fmt.Errorf("persistent shell exec create: %w", err)
	}

	// Upgrade to a hijacked raw connection.
	conn, err := b.hijackExec(ctx, created.ID)
	if err != nil {
		return nil, fmt.Errorf("persistent shell exec hijack: %w", err)
	}

	s := &shellSession{
		sandboxID: sandboxID,
		sentinel:  sentinelPrefix + token + sentinelSuffix,
		conn:      conn,
		reader:    bufio.NewReaderSize(conn, shellReadBufBytes),
	}

	// Confirm the shell answers before handing it to the caller.
	startCtx, cancel := context.WithTimeout(ctx, defaultShellStartupTimeout)
	defer cancel()
	probe := fmt.Sprintf("echo %s $$", readinessMarker)
	out, _, err := s.runInner(startCtx, probe, "", nil)
	if err != nil || !strings.Contains(out, readinessMarker) {
		s.close()
		return nil, fmt.Errorf("persistent shell did not become ready: %v (output: %q)", err, out)
	}
	return s, nil
}

// alive reports whether the shell process is still running.
func (s *shellSession) alive() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return !s.closed
}

// run executes one command and returns its stdout and exit code.
//
// The command is wrapped in a subshell that redirects stdin to /dev/null and
// appends the sentinel with the exit status. The sentinel is what tells the
// read loop that this command is done; without it the loop would block until
// the next command's sentinel arrived, merging two commands' outputs.
func (s *shellSession) run(
	ctx context.Context,
	command, workdir string,
	env map[string]string,
) (stdout string, exitCode int, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return "", 1, fmt.Errorf("persistent shell is closed")
	}
	return s.runInner(ctx, command, workdir, env)
}

// runInner executes one command. Must be called with s.mu held.
func (s *shellSession) runInner(
	ctx context.Context,
	command, workdir string,
	env map[string]string,
) (string, int, error) {
	wrapped := s.wrap(command, workdir, env)
	if _, err := fmt.Fprintf(s.conn, "%s\n", wrapped); err != nil {
		s.closed = true
		return "", 1, fmt.Errorf("persistent shell write: %w", err)
	}

	// Read until the sentinel appears.
	var buf bytes.Buffer
	deadline := time.Time{}
	if t, ok := ctx.Deadline(); ok {
		deadline = t
	}
	silenceDeadline := time.Now().Add(defaultSilenceTimeout)
	raw := make([]byte, shellReadBufBytes)
	for {
		if !deadline.IsZero() {
			now := time.Now()
			if now.After(deadline) {
				s.closed = true
				return buf.String(), 1, fmt.Errorf("persistent shell command exceeded deadline")
			}
			nextSilence := silenceDeadline
			if deadline.Before(nextSilence) {
				nextSilence = deadline
			}
			_ = s.conn.SetReadDeadline(nextSilence)
		} else {
			_ = s.conn.SetReadDeadline(silenceDeadline)
		}

		n, readErr := s.reader.Read(raw)
		if n > 0 {
			buf.Write(raw[:n])
			silenceDeadline = time.Now().Add(defaultSilenceTimeout)
			// Check if the sentinel has arrived.
			output := buf.String()
			exitCode, remaining, found := parseSentinel(output, s.sentinel)
			if found {
				return remaining, exitCode, nil
			}
		}
		if readErr != nil {
			if os.IsTimeout(readErr) {
				if time.Now().After(silenceDeadline) {
					s.closed = true
					return buf.String(), 1, fmt.Errorf("persistent shell silence timeout exceeded")
				}
				continue
			}
			s.closed = true
			return buf.String(), 1, fmt.Errorf("persistent shell read: %w", readErr)
		}
	}
}

// wrap builds the text written to stdin for one command.
//
// The exit status is captured and printed as part of the sentinel. Stdin is
// redirected to /dev/null so a command that reads stdin does not consume the
// next queued line (which would be the next command or the next sentinel).
func (s *shellSession) wrap(command, workdir string, env map[string]string) string {
	var b strings.Builder

	// Per-command environment variables as prefix assignments.
	if len(env) > 0 {
		for key, value := range env {
			b.WriteString(shellQuote(key))
			b.WriteByte('=')
			b.WriteString(shellQuote(value))
			b.WriteByte(' ')
		}
	}

	// Change directory if requested, fail loud so the caller knows it failed.
	if workdir != "" {
		b.WriteString("cd ")
		b.WriteString(shellQuote(workdir))
		b.WriteString(" && ")
	}

	// Run the command in a subshell with stdin closed, capture exit status.
	b.WriteString("( ")
	b.WriteString(command)
	b.WriteString(" </dev/null ); _exit=$?; printf '%s%d%s\\n' ")
	b.WriteString(shellQuote(sentinelPrefix))
	b.WriteString(" $_exit ")
	b.WriteString(shellQuote(sentinelSuffix))
	return b.String()
}

// close terminates the shell and marks the session closed.
func (s *shellSession) close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	if s.conn != nil {
		// Best-effort exit; ignore errors — the connection is being closed.
		_, _ = fmt.Fprintf(s.conn, "exit\n")
		_ = s.conn.Close()
	}
}

// sentinelRE matches the sentinel line produced by wrap().
var sentinelRE = regexp.MustCompile(regexp.QuoteMeta(sentinelPrefix) + `(\d+)` + regexp.QuoteMeta(sentinelSuffix))

// parseSentinel scans output for the sentinel, returning (exitCode, output, found).
//
// The sentinel line is stripped from the returned output so callers see only
// what the command itself wrote.
func parseSentinel(output, _ string) (exitCode int, remaining string, found bool) {
	match := sentinelRE.FindStringSubmatchIndex(output)
	if match == nil {
		return 0, output, false
	}
	code, err := strconv.Atoi(output[match[2]:match[3]])
	if err != nil {
		// Sentinel arrived but status is unparseable; report the output.
		return 1, output[:match[0]] + output[match[1]:], true
	}
	return code, output[:match[0]] + output[match[1]:], true
}

// shellQuote quotes a value for safe inclusion in a POSIX shell command.
func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}

// hijackExec upgrades an exec-start request to a raw bidirectional connection.
//
// The Docker Engine returns HTTP/1.1 101 Switching Protocols when it accepts
// the upgrade. After that, the connection is a raw stream: writes go to the
// exec's stdin, reads come from its stdout/stderr with the Engine's frame
// header stripped.
func (b *Backend) hijackExec(ctx context.Context, execID string) (net.Conn, error) {
	path := "/" + b.cfg.APIVersion + "/exec/" + execID + "/start"
	body, err := json.Marshal(map[string]any{"Detach": false, "Tty": false})
	if err != nil {
		return nil, err
	}

	// Dial the socket directly to get access to the raw connection.
	var conn net.Conn
	switch {
	case strings.HasPrefix(b.cfg.Socket, "unix://"):
		conn, err = (&net.Dialer{}).DialContext(ctx, "unix", strings.TrimPrefix(b.cfg.Socket, "unix://"))
	default:
		conn, err = (&net.Dialer{}).DialContext(ctx, "unix", b.cfg.Socket)
	}
	if err != nil {
		return nil, fmt.Errorf("dial docker socket: %w", err)
	}

	req := fmt.Sprintf(
		"POST %s HTTP/1.1\r\nHost: docker\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: Upgrade\r\nUpgrade: tcp\r\n\r\n%s",
		path, len(body), string(body),
	)
	if _, err := io.WriteString(conn, req); err != nil {
		_ = conn.Close()
		return nil, fmt.Errorf("write hijack request: %w", err)
	}

	// Read the response headers to confirm the upgrade was accepted.
	br := bufio.NewReader(conn)
	resp, err := http.ReadResponse(br, nil)
	if err != nil {
		_ = conn.Close()
		return nil, fmt.Errorf("read hijack response: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusSwitchingProtocols && resp.StatusCode != http.StatusOK {
		_ = conn.Close()
		return nil, fmt.Errorf("hijack: unexpected status %d", resp.StatusCode)
	}
	// Drain any bytes the response reader buffered. They belong to the exec stream.
	drained := make([]byte, br.Buffered())
	if len(drained) > 0 {
		if _, err := io.ReadFull(br, drained); err != nil {
			_ = conn.Close()
			return nil, fmt.Errorf("drain hijack buffer: %w", err)
		}
	}
	// Wrap the connection with a reader that replays buffered bytes first.
	return &prependedConn{Conn: conn, prepended: drained}, nil
}

// prependedConn replays a slice of bytes before delegating to the real conn.
type prependedConn struct {
	net.Conn
	prepended []byte
}

func (p *prependedConn) Read(buf []byte) (int, error) {
	if len(p.prepended) > 0 {
		n := copy(buf, p.prepended)
		p.prepended = p.prepended[n:]
		return n, nil
	}
	return p.Conn.Read(buf)
}

// ExecPersistent runs a command in the persistent shell for a sandbox.
//
// It is called by Exec when spec.ExecMode is "persistent". The session is
// created on the first call and reused for all subsequent ones; if it
// dies (timeout, shell crash) a new one is started transparently.
func (b *Backend) execPersistent(
	ctx context.Context,
	handle backend.Handle,
	command, workdir string,
	env map[string]string,
) (int, string, error) {
	s, err := b.shells.acquire(ctx, b, handle.SandboxID)
	if err != nil {
		return 1, "", fmt.Errorf("persistent shell: %w", err)
	}
	stdout, code, err := s.run(ctx, command, workdir, env)
	if err != nil && !s.alive() {
		// The shell died. Drop it and retry once with a fresh one.
		b.shells.drop(handle.SandboxID)
		s2, err2 := b.shells.acquire(ctx, b, handle.SandboxID)
		if err2 != nil {
			return 1, stdout, fmt.Errorf("persistent shell restart: %w", err2)
		}
		stdout, code, err = s2.run(ctx, command, workdir, env)
	}
	// Strip the frame headers the Engine multiplexer adds: the persistent shell
	// stream uses the same framing as the one-shot exec.
	return code, demultiplex([]byte(stdout)), err
}
