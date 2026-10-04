// parity.go: the direct-mode features that close the gap with provider mode.
//
// # What the gap actually was, and what of it is closable
//
// Provider mode wraps CubeMaster and declared four features direct mode did not:
// volume, egress_policy, warm_pool, and template_build, plus freeze. Those were
// not an oversight -- the adapter was right to omit what it could not serve --
// but three of them were omitted for different reasons, and only one reason is
// permanent. Reading CubeboxMgr's own service definition separates them:
//
//   - volume IS closable, and is closed here. RunCubeSandboxRequest carries
//     Volumes and each ContainerConfig carries VolumeMounts, so a Cubelet mounts
//     storage natively. What CubeMaster adds above that is a named-volume
//     registry, not the mount itself, and a spec that names a plugin driver
//     reaches the same cubecow plumbing either way.
//
//   - template_build and warm_pool are NOT closable and remain absent. Both are
//     CubeMaster APIs with no CubeboxMgr equivalent: there is no RPC that builds
//     a template, and a warm pool is a pool of sandboxes CubeMaster holds across
//     nodes. p3b has its own warm pool for the backends it drives directly, which
//     is the right place for that concern, not a claim that Cubelet provides one.
//
//   - freeze is NOT closable, and this is the one worth stating plainly because
//     it is the most tempting to fake. CubeboxMgr has no Pause or Resume RPC at
//     all -- the full service is Create, Destroy, List, Update, Exec, AppSnapshot,
//     CommitSandbox, RollbackSandbox, CleanupTemplate, and five storage/snapshot
//     inspection calls. A pause is a CubeMaster operation above the node.
//     Declaring freeze here would let the reclaimer pause-on-idle and believe it
//     had released compute it had not, which is worse than the feature's absence:
//     the node's accounting would drift from reality with nothing to detect it.
//
// # What direct mode gains that provider mode never had
//
// Three CubeboxMgr RPCs were going unused, and each closes a real gap:
//
//   - Exec, as a detached run. Its response carries no stdout and no exit status,
//     so it is an attach primitive rather than a request-response exec; exposed
//     here as RunDetached, named for what it actually does. See the method for
//     why a synchronous exec cannot be built on it.
//
//   - RollbackSandbox. A restore in place, from a snapshot CommitSandbox made.
//     Without it a snapshot was write-only through this adapter.
//
//   - GetStorageMetrics. Node-local cubecow usage, which is what makes a
//     density decision measurable rather than assumed.
package cubesandbox

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"psrl.dev/sandboxd/internal/backend"
	cubebox "psrl.dev/sandboxd/internal/backend/cubesandbox/cubeletpb/services/cubebox/v1"
	volumeplugin "psrl.dev/sandboxd/internal/backend/cubesandbox/cubeletpb/services/volumeplugin/v1"
)

// Volume-related backend option keys.
//
// The portable spec has no volume field at this layer, so a volume request
// arrives through this backend's own namespaced options. The format is one
// option per volume, which keeps it declarative and order-independent:
//
//	volume.<name> = <target>[:ro][:driver=<driver>][:size=<size>]
//
// Examples:
//
//	volume.workspace = /workspace
//	volume.dataset   = /data:ro:driver=cubecow
//	volume.scratch   = /scratch:size=4Gi
//
// A name is part of the key rather than the value because Cubelet keys the mount
// to the volume by name, and two volumes sharing a name would silently collide.
const volumeOptionPrefix = "volume."

// volumeRequest is one parsed volume option.
type volumeRequest struct {
	name     string
	target   string
	readOnly bool
	// driver names a cubecow volume plugin. Empty means an emptyDir, which is the
	// node-local scratch a sandbox gets without any storage backend configured.
	driver string
	// sizeLimit is only meaningful for an emptyDir; a plugin volume is sized by
	// the driver.
	sizeLimit string
}

// parseVolumeOptions reads every volume.* option into a request.
//
// A malformed option is an error rather than a skip. A typo in a volume spec
// would otherwise start a sandbox whose storage is silently missing, and the task
// inside it would fail for a reason that looks like its own bug.
func parseVolumeOptions(spec backend.Spec) ([]volumeRequest, error) {
	options, named := spec.Options("cubesandbox")
	if !named {
		return nil, nil
	}
	requests := make([]volumeRequest, 0, len(options))
	for key, value := range options {
		if !strings.HasPrefix(key, volumeOptionPrefix) {
			continue
		}
		name := strings.TrimPrefix(key, volumeOptionPrefix)
		if name == "" {
			return nil, fmt.Errorf("a cubesandbox volume option needs a name: %q has none", key)
		}
		parsed, err := parseVolumeValue(name, value)
		if err != nil {
			return nil, err
		}
		requests = append(requests, parsed)
	}
	// Sorted so one spec always produces one request body. An unordered map would
	// make two identical specs differ on the wire, which defeats idempotency and
	// makes a diff of two creates unreadable.
	sort.Slice(requests, func(i, j int) bool { return requests[i].name < requests[j].name })
	return requests, nil
}

func parseVolumeValue(name, value string) (volumeRequest, error) {
	parts := strings.Split(value, ":")
	if len(parts) == 0 || strings.TrimSpace(parts[0]) == "" {
		return volumeRequest{}, fmt.Errorf(
			"cubesandbox volume %q needs a container path, for example %q",
			name, "/workspace")
	}
	request := volumeRequest{name: name, target: strings.TrimSpace(parts[0])}
	if !strings.HasPrefix(request.target, "/") {
		return volumeRequest{}, fmt.Errorf(
			"cubesandbox volume %q target %q must be an absolute path", name, request.target)
	}
	for _, attribute := range parts[1:] {
		attribute = strings.TrimSpace(attribute)
		switch {
		case attribute == "":
			continue
		case attribute == "ro":
			request.readOnly = true
		case attribute == "rw":
			request.readOnly = false
		case strings.HasPrefix(attribute, "driver="):
			request.driver = strings.TrimPrefix(attribute, "driver=")
		case strings.HasPrefix(attribute, "size="):
			request.sizeLimit = strings.TrimPrefix(attribute, "size=")
		default:
			return volumeRequest{}, fmt.Errorf(
				"cubesandbox volume %q has an unknown attribute %q; expected ro, rw, "+
					"driver=<name>, or size=<quantity>", name, attribute)
		}
	}
	return request, nil
}

// volumesFor turns parsed requests into the Cubelet volume and mount pair.
//
// Both halves are needed and they are keyed to each other by name: a volume with
// no mount is storage the sandbox cannot see, and a mount with no volume fails
// the create.
func volumesFor(requests []volumeRequest) ([]*cubebox.Volume, []*cubebox.VolumeMounts) {
	if len(requests) == 0 {
		return nil, nil
	}
	volumes := make([]*cubebox.Volume, 0, len(requests))
	mounts := make([]*cubebox.VolumeMounts, 0, len(requests))
	for _, request := range requests {
		source := &cubebox.VolumeSource{}
		if request.driver != "" {
			// A plugin volume is the shape CubeMaster's named volumes resolve to, so a
			// spec naming a driver reaches the same cubecow plumbing either way.
			source.PluginVolume = &volumeplugin.PluginVolumeSource{Driver: request.driver}
		} else {
			// No driver: node-local scratch. This is what a sandbox gets when the
			// deployment has no storage backend, and it is still a real volume.
			empty := &cubebox.EmptyDirVolumeSource{}
			if request.sizeLimit != "" {
				empty.SizeLimit = request.sizeLimit
			}
			source.EmptyDir = empty
		}
		volumes = append(volumes, &cubebox.Volume{Name: request.name, VolumeSource: source})
		mounts = append(mounts, &cubebox.VolumeMounts{
			Name:          request.name,
			ContainerPath: request.target,
			Readonly:      request.readOnly,
		})
	}
	return volumes, mounts
}

// -- exec ----------------------------------------------------------------------

// RunDetached starts a command in a sandbox and does not wait for it.
//
// # Why this is not an Exec
//
// CubeboxMgr's Exec RPC cannot return output. Its response message carries only
// a request id and a result code -- no stdout, no stderr, no exit status -- and
// the request carries a Terminal flag and an argv. That shape is an attach
// primitive: Cubelet starts the process and the caller is expected to reach it
// over a stream, which is how cubecli drives an interactive session.
//
// So a synchronous exec cannot be built on this RPC, and pretending otherwise
// would be the worst option available: a caller would receive exit code 0 and
// empty output for a command that failed, and a grader reading that would score
// a broken task as a passing one. The method is therefore named for what it does.
//
// A caller that needs command output from this backend has two honest routes:
// use provider mode, where the sandbox's envd agent is the command path and the
// create reply hands back its endpoint; or read the result out of the sandbox
// filesystem, which is what a harness that writes its own trajectory already
// does.
func (b *Backend) RunDetached(
	ctx context.Context, handle backend.Handle, command, workdir string, env map[string]string,
) error {
	if b.mode != backend.SchedulingDirect {
		return fmt.Errorf(
			"cubesandbox in provider mode runs commands through the sandbox's own agent, " +
				"not through this service; use the agent endpoint from the create reply")
	}
	return b.direct.runDetached(ctx, handle, command, workdir, env)
}

// runDetached issues one ExecCubeSandbox RPC and reports only whether Cubelet
// accepted it.
func (r *directRuntime) runDetached(
	ctx context.Context, handle backend.Handle, command, workdir string, env map[string]string,
) error {
	node, err := r.node(handle.NodeID)
	if err != nil {
		return err
	}
	// A shell, so a caller may send a pipeline or a conditional rather than only an
	// argv -- the same thing every other backend's exec accepts.
	request := &cubebox.ExecCubeSandboxRequest{
		RequestID:   newRequestID(),
		SandboxId:   handle.SandboxID,
		ContainerId: mainContainer,
		Args:        []string{"/bin/sh", "-c", command},
		// Cwd is a real field on this RPC, so the working directory is set rather
		// than prepended as a cd: a chdir that failed inside the command string
		// would run the command in the wrong place instead of failing.
		Cwd: workdir,
		Env: envStrings(env),
	}

	callCtx, cancel := context.WithTimeout(ctx, r.requestTimeout)
	defer cancel()
	reply, err := node.client.Exec(callCtx, request)
	if err != nil {
		return fmt.Errorf("cubesandbox exec on %s: %w", handle.SandboxID, err)
	}
	if err := retError(reply.GetRet()); err != nil {
		return fmt.Errorf("cubesandbox exec on %s: %w", handle.SandboxID, err)
	}
	return nil
}

// envStrings renders an environment map as KEY=VALUE in a stable order.
//
// Stable because two identical specs must produce one request body: an
// unordered map would make a diff of two creates unreadable and defeat
// idempotency on the wire.
func envStrings(env map[string]string) []string {
	if len(env) == 0 {
		return nil
	}
	names := make([]string, 0, len(env))
	for name := range env {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]string, 0, len(names))
	for _, name := range names {
		out = append(out, name+"="+env[name])
	}
	return out
}

// -- restore -------------------------------------------------------------------

// Restore rolls a running sandbox back to a snapshot it took earlier.
//
// Without this a snapshot was write-only through this adapter: CommitSandbox
// produced an id that nothing could consume. The RPC restores in place rather
// than creating a new sandbox, which is why it takes a handle and not a spec.
//
// Provider mode has no equivalent here on purpose: CubeMaster restores by
// creating a sandbox from a snapshot id, which is the create path, so routing a
// restore through this method would be two ways to do one thing.
func (b *Backend) Restore(ctx context.Context, handle backend.Handle, snapshotID string) error {
	if b.mode != backend.SchedulingDirect {
		return fmt.Errorf(
			"cubesandbox in provider mode restores by creating a sandbox from the snapshot id, " +
				"so pass it as the spec source rather than restoring in place")
	}
	if snapshotID == "" {
		return fmt.Errorf("cubesandbox restore needs a snapshot id")
	}
	return b.direct.restore(ctx, handle, snapshotID)
}

func (r *directRuntime) restore(ctx context.Context, handle backend.Handle, snapshotID string) error {
	node, err := r.node(handle.NodeID)
	if err != nil {
		return err
	}
	// The create timeout, not the request timeout: a rollback rewrites the
	// sandbox's rootfs from a snapshot, which is image-sized work rather than a
	// control call.
	callCtx, cancel := context.WithTimeout(ctx, r.createTimeout)
	defer cancel()
	reply, err := node.client.RollbackSandbox(callCtx, &cubebox.RollbackSandboxRequest{
		RequestID:  newRequestID(),
		SandboxID:  handle.SandboxID,
		SnapshotID: snapshotID,
	})
	if err != nil {
		return fmt.Errorf("cubesandbox restore %s to %s: %w", handle.SandboxID, snapshotID, err)
	}
	if err := retError(reply.GetRet()); err != nil {
		return fmt.Errorf("cubesandbox restore %s to %s: %w", handle.SandboxID, snapshotID, err)
	}
	return nil
}

// -- storage metrics -----------------------------------------------------------

// StorageMetrics reports node-local cubecow usage, per node.
//
// This is what makes a density decision measurable rather than assumed. The
// backend's own Headroom deliberately returns nothing -- node admission tracks
// every grant exactly and reading the provider's snapshot too would be two
// disagreeing accounts of one envelope -- but storage is the dimension admission
// does not track, because a sandbox's disk growth is not a reservation. So this
// is reported for the metric hook rather than fed back into placement.
//
// A node that does not answer is skipped rather than failing the call: this is
// diagnostic, and one unreachable Cubelet must not deny an operator the view of
// the rest of the fleet.
func (b *Backend) StorageMetrics(ctx context.Context) map[string]map[string]uint64 {
	if b.mode != backend.SchedulingDirect {
		return nil
	}
	return b.direct.storageMetrics(ctx)
}

func (r *directRuntime) storageMetrics(ctx context.Context) map[string]map[string]uint64 {
	r.mu.RLock()
	nodes := make([]*cubeletConn, 0, len(r.nodes))
	for _, node := range r.nodes {
		nodes = append(nodes, node)
	}
	r.mu.RUnlock()

	out := make(map[string]map[string]uint64, len(nodes))
	for _, node := range nodes {
		callCtx, cancel := context.WithTimeout(ctx, r.requestTimeout)
		reply, err := node.client.GetStorageMetrics(callCtx, &cubebox.GetStorageMetricsRequest{
			RequestID: newRequestID(),
		})
		cancel()
		if err != nil || retError(reply.GetRet()) != nil {
			continue
		}
		if metrics := reply.GetMetrics(); len(metrics) > 0 {
			out[node.nodeID] = metrics
		}
	}
	return out
}
