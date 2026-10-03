// execd staging: the one-time preparation that makes direct mode possible.
//
// In direct mode this service drives the Docker daemon itself, so nothing else
// puts OpenSandbox's in-sandbox agent into a container. The agent is distributed
// as an OCI image whose root holds three artifacts:
//
//	/execd                  the Go agent binary, statically linked
//	/bootstrap.sh           the launcher that starts execd and the workload
//	/usr/local/bin/bwrap    bubblewrap, for isolated-session support
//
// They are extracted once, at startup, into a host directory that every sandbox
// then bind-mounts read-only. Extracting once and mounting is the whole reason
// direct mode is cheaper than going through a lifecycle server: that server
// copies the same bytes into every container it creates, so the cost is paid per
// sandbox and lands inside the create latency. A mount costs nothing per
// sandbox.
//
// Staging is idempotent and keyed by the image's digest, so a restart does not
// re-extract and an upgraded agent image does. The digest is recorded beside the
// artifacts; a mismatch re-stages, a match returns immediately.
package opensandbox

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Artifact paths inside the agent image, and where each lands under the stage
// directory. The in-container layout is flat because bootstrap.sh resolves its
// siblings relative to itself.
const (
	execdImagePath     = "/execd"
	bootstrapImagePath = "/bootstrap.sh"
	bwrapImagePath     = "/usr/local/bin/bwrap"

	stagedExecd     = "execd"
	stagedBootstrap = "bootstrap.sh"
	stagedBwrap     = "bwrap"

	// digestMarker records which image the staged artifacts came from, so a
	// restart can skip the work and an upgrade cannot be served stale bytes.
	digestMarker = ".image-digest"

	// SandboxAgentDir is where a sandbox finds the staged artifacts. bootstrap.sh
	// is invoked by absolute path, and it locates execd beside itself.
	SandboxAgentDir = "/opt/opensandbox"

	// stageTimeout bounds the whole extraction. A registry pull dominates it, so
	// it is generous: failing a deployment because a cold registry was slow would
	// be a worse outcome than waiting.
	stageTimeout = 10 * time.Minute
)

// stagedAgent is the result of staging: a host directory holding the artifacts,
// and the digest they came from.
type stagedAgent struct {
	// Dir is the host path to bind-mount into each sandbox.
	Dir string
	// Digest identifies the image the artifacts came from.
	Digest string
	// HasBwrap says whether bubblewrap was present. It is optional: an agent
	// image without it serves every route except isolated sessions, so its
	// absence narrows capabilities rather than failing the deployment.
	HasBwrap bool
}

// stageAgent extracts the agent artifacts from an image into dir.
//
// The extraction borrows the daemon rather than reimplementing an image reader:
// a container is created from the image (never started), its filesystem is read
// through the archive endpoint, and the container is removed. This is the same
// mechanism a lifecycle server uses, and it works for any image layout without
// this code needing to understand layers.
//
// A missing image is pulled first. A pull failure is reported as itself rather
// than as a staging failure, because the operator's fix differs: a missing image
// is a registry or a name problem, a failed extraction is an image-contents one.
func stageAgent(ctx context.Context, client *dockerClient, image, dir string) (stagedAgent, error) {
	ctx, cancel := context.WithTimeout(ctx, stageTimeout)
	defer cancel()

	digest, err := client.imageDigest(ctx, image)
	if err != nil {
		// Not present locally. Pull, then read the digest the pull produced.
		if pullErr := client.pullImage(ctx, image); pullErr != nil {
			return stagedAgent{}, fmt.Errorf(
				"the OpenSandbox agent image %q is neither present locally nor pullable: %w\n"+
					"Pull it manually with: docker pull %s", image, pullErr, image)
		}
		digest, err = client.imageDigest(ctx, image)
		if err != nil {
			return stagedAgent{}, fmt.Errorf("the agent image %q was pulled but cannot be inspected: %w", image, err)
		}
	}

	if staged, fresh := readStaged(dir, digest); fresh {
		return staged, nil
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return stagedAgent{}, fmt.Errorf("the agent stage directory %s is not writable: %w", dir, err)
	}

	// A container is the handle the archive endpoint needs. It is never started,
	// so nothing from the agent image runs on this host.
	containerID, err := client.createContainer(ctx, map[string]any{
		"Image": image,
		// Overridden so an image with a real entrypoint cannot be started by
		// accident by anything that lists containers and restarts them.
		"Entrypoint": []string{"/bin/true"},
		"Labels":     map[string]string{ownerLabel: "agent-stage"},
	}, "")
	if err != nil {
		return stagedAgent{}, fmt.Errorf("could not create a container to read the agent image %q: %w", image, err)
	}
	defer func() {
		// Best effort: a leaked stage container is inert (never started) and the
		// next run's label would find it, but leaving it is still untidy.
		removeCtx, removeCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer removeCancel()
		_ = client.removeContainer(removeCtx, containerID, true)
	}()

	required := []struct {
		in, out string
		mode    os.FileMode
	}{
		{execdImagePath, stagedExecd, 0o755},
		{bootstrapImagePath, stagedBootstrap, 0o755},
	}
	for _, artifact := range required {
		if err := client.copyOut(ctx, containerID, artifact.in, filepath.Join(dir, artifact.out), artifact.mode); err != nil {
			return stagedAgent{}, fmt.Errorf(
				"the agent image %q does not carry %s: %w\n"+
					"This image is not an OpenSandbox agent image; check the reference.", image, artifact.in, err)
		}
	}

	// Optional. Its absence costs isolated sessions and nothing else, so it must
	// not fail staging.
	hasBwrap := client.copyOut(ctx, containerID, bwrapImagePath, filepath.Join(dir, stagedBwrap), 0o755) == nil

	if err := os.WriteFile(filepath.Join(dir, digestMarker), []byte(digest), 0o644); err != nil {
		return stagedAgent{}, fmt.Errorf("could not record the staged agent digest in %s: %w", dir, err)
	}
	return stagedAgent{Dir: dir, Digest: digest, HasBwrap: hasBwrap}, nil
}

// readStaged reports whether dir already holds artifacts from this digest.
//
// Every required artifact must be present as well as the digest matching: a
// half-written stage directory from an interrupted run would otherwise be read
// as complete, and the sandbox would fail to start with a missing-file error
// far from its cause.
func readStaged(dir, digest string) (stagedAgent, bool) {
	recorded, err := os.ReadFile(filepath.Join(dir, digestMarker))
	if err != nil || strings.TrimSpace(string(recorded)) != digest {
		return stagedAgent{}, false
	}
	for _, name := range []string{stagedExecd, stagedBootstrap} {
		if info, err := os.Stat(filepath.Join(dir, name)); err != nil || info.Size() == 0 {
			return stagedAgent{}, false
		}
	}
	_, bwrapErr := os.Stat(filepath.Join(dir, stagedBwrap))
	return stagedAgent{Dir: dir, Digest: digest, HasBwrap: bwrapErr == nil}, true
}

// -- the Docker client this package drives in direct mode ---------------------

// dockerClient is the slice of the Engine API direct mode needs.
//
// It is this package's own client rather than a shared one, because the
// OpenSandbox integration is an independent path: a change made for the
// container backend must not silently alter how agent containers are created,
// and vice versa.
type dockerClient struct {
	http       *http.Client
	apiVersion string
}

const ownerLabel = "opensandbox.psrl/owner"

// newDockerClient dials a Docker daemon over a unix socket or a TCP address.
func newDockerClient(socket, apiVersion string) (*dockerClient, error) {
	if socket == "" {
		return nil, fmt.Errorf("a Docker socket is required to drive the runtime directly")
	}
	if apiVersion == "" {
		apiVersion = "v1.43"
	}
	transport := &http.Transport{MaxIdleConns: 128, MaxIdleConnsPerHost: 64, IdleConnTimeout: 90 * time.Second}
	if path, found := strings.CutPrefix(socket, "unix://"); found {
		transport.DialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", path)
		}
	} else {
		host := strings.TrimPrefix(strings.TrimPrefix(socket, "tcp://"), "http://")
		transport.DialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "tcp", host)
		}
	}
	return &dockerClient{http: &http.Client{Transport: transport}, apiVersion: apiVersion}, nil
}

func (d *dockerClient) url(path string) string {
	// The host is ignored by the unix dialer and supplied by the tcp one, so a
	// placeholder keeps one URL shape for both.
	return "http://docker" + "/" + d.apiVersion + path
}

// ping proves the daemon is answering, which is the cheapest useful preflight.
func (d *dockerClient) ping(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, d.url("/_ping"), nil)
	if err != nil {
		return err
	}
	resp, err := d.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode >= 400 {
		return fmt.Errorf("the Docker daemon answered /_ping with HTTP %d", resp.StatusCode)
	}
	return nil
}

// imageDigest returns an image's identity, or an error when it is absent.
func (d *dockerClient) imageDigest(ctx context.Context, image string) (string, error) {
	var inspected struct {
		ID       string   `json:"Id"`
		RepoTags []string `json:"RepoTags"`
	}
	if err := d.call(ctx, http.MethodGet, "/images/"+image+"/json", nil, &inspected); err != nil {
		return "", err
	}
	if inspected.ID == "" {
		return "", fmt.Errorf("the Docker daemon reported no id for image %q", image)
	}
	return inspected.ID, nil
}

// pullImage fetches an image and waits for the pull to finish.
//
// The endpoint streams progress and the stream ending is what says the pull is
// done, so the body is drained rather than ignored. A pull reports its failures
// inside that stream as well as by status code, so the tail is inspected.
func (d *dockerClient) pullImage(ctx context.Context, image string) error {
	name, tag := image, "latest"
	// A digest reference carries its own separator and must not be split on ":".
	if at := strings.LastIndex(image, "@"); at >= 0 {
		name, tag = image[:at], image[at+1:]
	} else if colon := strings.LastIndex(image, ":"); colon > strings.LastIndex(image, "/") {
		name, tag = image[:colon], image[colon+1:]
	}
	path := fmt.Sprintf("/images/create?fromImage=%s&tag=%s", name, tag)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, d.url(path), nil)
	if err != nil {
		return err
	}
	resp, err := d.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode >= 400 {
		return fmt.Errorf("docker pull returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	// The stream carries per-layer objects; an "error" key anywhere is a failure
	// even though the status was 200.
	if bytes.Contains(body, []byte(`"error"`)) {
		return fmt.Errorf("docker pull of %q failed: %s", image, tailLine(body))
	}
	return nil
}

// createContainer creates one container and returns its id.
func (d *dockerClient) createContainer(ctx context.Context, body map[string]any, name string) (string, error) {
	path := "/containers/create"
	if name != "" {
		path += "?name=" + name
	}
	var created struct {
		ID       string   `json:"Id"`
		Warnings []string `json:"Warnings"`
	}
	if err := d.call(ctx, http.MethodPost, path, body, &created); err != nil {
		return "", err
	}
	if created.ID == "" {
		return "", fmt.Errorf("the Docker daemon created a container but returned no id")
	}
	return created.ID, nil
}

// startContainer starts a created container.
func (d *dockerClient) startContainer(ctx context.Context, id string) error {
	return d.call(ctx, http.MethodPost, "/containers/"+id+"/start", nil, nil)
}

// removeContainer deletes a container, optionally killing it first.
func (d *dockerClient) removeContainer(ctx context.Context, id string, force bool) error {
	path := "/containers/" + id + "?v=1"
	if force {
		path += "&force=1"
	}
	err := d.call(ctx, http.MethodDelete, path, nil, nil)
	if err != nil && isDockerNotFound(err) {
		// Already gone is the outcome the caller wanted.
		return nil
	}
	return err
}

// copyOut extracts one file from a container's filesystem to a host path.
//
// The archive endpoint returns a tar stream even for a single file, so the entry
// is located rather than assumed: a path that resolves to a symlink or a
// directory yields a different entry name than the one requested.
func (d *dockerClient) copyOut(ctx context.Context, containerID, inPath, outPath string, mode os.FileMode) error {
	path := fmt.Sprintf("/containers/%s/archive?path=%s", containerID, inPath)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, d.url(path), nil)
	if err != nil {
		return err
	}
	resp, err := d.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("reading %s returned HTTP %d: %s", inPath, resp.StatusCode, strings.TrimSpace(string(body)))
	}

	reader := tar.NewReader(resp.Body)
	for {
		header, err := reader.Next()
		if err == io.EOF {
			return fmt.Errorf("the archive for %s carried no regular file", inPath)
		}
		if err != nil {
			return fmt.Errorf("reading the archive for %s: %w", inPath, err)
		}
		if header.Typeflag != tar.TypeReg {
			continue
		}
		target, err := os.OpenFile(outPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
		if err != nil {
			return fmt.Errorf("writing %s: %w", outPath, err)
		}
		//nolint:gosec // The source is an image this deployment configured.
		if _, err := io.Copy(target, reader); err != nil {
			target.Close()
			return fmt.Errorf("writing %s: %w", outPath, err)
		}
		if err := target.Close(); err != nil {
			return fmt.Errorf("writing %s: %w", outPath, err)
		}
		// Explicit chmod: the create mode is masked by umask, and execd has to be
		// executable by whatever uid the sandbox runs as.
		return os.Chmod(outPath, mode)
	}
}

// call sends one Engine API request, decoding into out when it is not nil.
func (d *dockerClient) call(ctx context.Context, method, path string, body any, out any) error {
	var payload io.Reader = http.NoBody
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return err
		}
		payload = bytes.NewReader(encoded)
	}
	req, err := http.NewRequestWithContext(ctx, method, d.url(path), payload)
	if err != nil {
		return err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := d.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode >= 400 {
		return &dockerError{status: resp.StatusCode, body: strings.TrimSpace(string(raw))}
	}
	if out != nil && len(raw) > 0 {
		return json.Unmarshal(raw, out)
	}
	return nil
}

type dockerError struct {
	status int
	body   string
}

func (e *dockerError) Error() string {
	return fmt.Sprintf("the Docker daemon returned %d: %s", e.status, e.body)
}

func (e *dockerError) Status() int { return e.status }

func isDockerNotFound(err error) bool {
	var typed *dockerError
	for err != nil {
		if candidate, ok := err.(*dockerError); ok {
			typed = candidate
			break
		}
		unwrapper, ok := err.(interface{ Unwrap() error })
		if !ok {
			return false
		}
		err = unwrapper.Unwrap()
	}
	return typed != nil && typed.status == http.StatusNotFound
}

// tailLine returns the last non-empty line, which is where a streamed failure
// reports itself.
func tailLine(raw []byte) string {
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if line := strings.TrimSpace(lines[i]); line != "" {
			return line
		}
	}
	return ""
}
