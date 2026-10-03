// Package server wires the service's domain packages onto the wire contract.
//
// It holds no policy of its own. Every decision -- which backend, which node,
// whether a class may grow -- belongs to the package that owns it, and this is
// the adapter that carries those decisions across a socket. Keeping it a pure
// adapter is what stops a second control plane growing here.
package server

import (
	v1 "psrl.dev/sandboxd/api/v1"
	"psrl.dev/sandboxd/internal/backend"
	"psrl.dev/sandboxd/internal/node"
	"psrl.dev/sandboxd/internal/placement"
	"psrl.dev/sandboxd/internal/quota"
)

// specFromProto converts a caller's request into the portable spec.
//
// An absent resource is absent rather than zero: a caller that states no memory
// requirement must not be compared against a limit as though it asked for none.
func specFromProto(in *v1.SandboxSpec) backend.Spec {
	if in == nil {
		return backend.Spec{}
	}
	spec := backend.Spec{
		ResourceClass:       orDefault(in.GetResourceClass(), "default"),
		WorkflowID:          in.GetWorkflowId(),
		IdempotencyKey:      in.GetIdempotencyKey(),
		Env:                 in.GetEnv(),
		Metadata:            in.GetMetadata(),
		Workdir:             in.GetWorkdir(),
		ExecMode:            execModeName(in.GetExecMode()),
		RequiredResume:      resumeLevelName(in.GetRequiredResumeLevel()),
		Backend:             in.GetBackend(),
		RequiredNodeLabel:   in.GetRequiredNodeLabel(),
		ForbiddenNodeLabels: in.GetForbiddenNodeLabels(),
	}
	if source := in.GetSource(); source != nil {
		spec.Source = backend.Source{Kind: sourceKindName(source.GetKind()), Reference: source.GetReference()}
	}
	if resources := in.GetResources(); resources != nil {
		spec.Resources = backend.Resources{
			CPUCount: resources.GetCpuCount(),
			MemoryMB: resources.GetMemoryMb(),
			DiskMB:   resources.GetDiskMb(),
			GPUCount: resources.GetGpuCount(),
		}
	}
	for _, feature := range in.GetRequiredFeatures() {
		spec.RequiredFeatures = append(spec.RequiredFeatures, featureName(feature))
	}
	if options := in.GetBackendOptions(); len(options) > 0 {
		spec.BackendOptions = make(map[string]map[string]string, len(options))
		for name, values := range options {
			spec.BackendOptions[name] = values.GetValues()
		}
	}
	return spec
}

// placementRequest turns a spec into what placement compares.
//
// The footprint travels with the request because placement cannot decide
// whether a sandbox fits without knowing how big it is, and the class travels
// with it because headroom is per class.
func placementRequest(spec backend.Spec, backendName, ownerID string) placement.Request {
	req := placement.Request{
		Backend:          backendName,
		RequiredFeatures: spec.RequiredFeatures,
		RequiredResume:   spec.RequiredResume,
		GPUCount:         spec.Resources.GPUCount,
		OwnerID:          ownerID,
		ResourceClass:    spec.ResourceClass,
		RequiredLabel:    spec.RequiredNodeLabel,
		ForbiddenLabels:  spec.ForbiddenNodeLabels,
		Footprint: placement.Headroom{
			MemoryMB:  spec.Resources.MemoryMB,
			CPUMillis: int64(spec.Resources.CPUCount * 1000),
			GPUCount:  spec.Resources.GPUCount,
			DiskMB:    spec.Resources.DiskMB,
		},
	}
	// A source that already names a digest is exact, and one that names a tag is
	// what a caller usually has, so both travel and placement trusts the digest.
	if spec.Source.Kind == "image" && spec.Source.Reference != "" {
		if containsAt(spec.Source.Reference) {
			req.ImageDigests = []string{spec.Source.Reference}
		} else {
			req.ImageReferences = []string{spec.Source.Reference}
		}
	}
	return req
}

func quotaAmount(spec backend.Spec) quota.Amount {
	return quota.Amount{
		MemoryMB:  spec.Resources.MemoryMB,
		CPUMillis: int64(spec.Resources.CPUCount * 1000),
		GPUCount:  spec.Resources.GPUCount,
		DiskMB:    spec.Resources.DiskMB,
		// Every sandbox counts as one against a backend that can only be held to a
		// concurrency, which is all a managed provider sells.
		Sandboxes: 1,
	}
}

func nodeResources(spec backend.Spec) node.Resources {
	return node.Resources{
		MemoryMB:  spec.Resources.MemoryMB,
		CPUMillis: int64(spec.Resources.CPUCount * 1000),
		GPUCount:  spec.Resources.GPUCount,
		DiskMB:    spec.Resources.DiskMB,
	}
}

func createdToProto(created backend.Created) *v1.CreateResponse {
	out := &v1.CreateResponse{
		Handle: &v1.SandboxHandle{
			Backend:   created.Handle.Backend,
			SandboxId: created.Handle.SandboxID,
			NodeId:    created.Handle.NodeID,
		},
		Capabilities: capabilitiesToProto(created.Capabilities),
		Agent: &v1.AgentEndpoint{
			Address:           created.Agent.Address,
			Headers:           created.Agent.Headers,
			CallbackHostAlias: created.Agent.CallbackHostAlias,
			CallbackPort:      created.Agent.CallbackPort,
		},
		WarmStart: created.WarmStart,
	}
	return out
}

func capabilitiesToProto(capabilities backend.Capabilities) *v1.Capabilities {
	out := &v1.Capabilities{ResumeLevel: resumeLevelValue(capabilities.ResumeLevel)}
	for _, feature := range capabilities.Features {
		out.Features = append(out.Features, featureValue(feature))
	}
	for _, mode := range capabilities.PauseModes {
		out.PauseModes = append(out.PauseModes, pauseModeValue(mode))
	}
	return out
}

// capabilitiesFromProto is the inverse of capabilitiesToProto.
//
// It exists because reading back only the resume level silently dropped the
// feature list and the pause modes on every create that crossed the node
// boundary. The caller then saw a backend that declared nothing it could do, so
// a capability check at the SDK -- "does this sandbox support a full-state
// snapshot?" -- answered no for a backend whose whole purpose is that it
// answers yes.
func capabilitiesFromProto(capabilities *v1.Capabilities) backend.Capabilities {
	out := backend.Capabilities{ResumeLevel: resumeLevelName(capabilities.GetResumeLevel())}
	for _, feature := range capabilities.GetFeatures() {
		if name := featureName(feature); name != "" {
			out.Features = append(out.Features, name)
		}
	}
	for _, mode := range capabilities.GetPauseModes() {
		if name := pauseModeName(mode); name != "" {
			out.PauseModes = append(out.PauseModes, name)
		}
	}
	return out
}

func headroomToProto(r node.Resources) *v1.Headroom {
	return &v1.Headroom{MemoryMb: r.MemoryMB, CpuMillis: r.CPUMillis, GpuCount: r.GPUCount, DiskMb: r.DiskMB}
}

func quotaHeadroomToProto(a quota.Amount) *v1.Headroom {
	return &v1.Headroom{MemoryMb: a.MemoryMB, CpuMillis: a.CPUMillis, GpuCount: a.GPUCount, DiskMb: a.DiskMB}
}

func placementHeadroom(r node.Resources) placement.Headroom {
	return placement.Headroom{MemoryMB: r.MemoryMB, CPUMillis: r.CPUMillis, GPUCount: r.GPUCount, DiskMB: r.DiskMB}
}

// Enum names are mapped explicitly rather than by string munging, so a new value
// has to be handled deliberately instead of silently becoming "unspecified".

func featureName(f v1.Feature) string {
	switch f {
	case v1.Feature_FREEZE:
		return "freeze"
	case v1.Feature_HIBERNATE:
		return "hibernate"
	case v1.Feature_FILESYSTEM_SNAPSHOT:
		return "filesystem_snapshot"
	case v1.Feature_FULL_STATE_SNAPSHOT:
		return "full_state_snapshot"
	case v1.Feature_RESTORE:
		return "restore"
	case v1.Feature_NATIVE_FORK:
		return "native_fork"
	case v1.Feature_HOST_MOUNT:
		return "host_mount"
	case v1.Feature_RESUME_ANYWHERE:
		return "resume_anywhere"
	case v1.Feature_WARM_POOL:
		return "warm_pool"
	case v1.Feature_IMAGE_ON_DEMAND:
		return "image_on_demand"
	case v1.Feature_IMAGE_BLOCK_DELIVERY:
		return "image_block_delivery"
	case v1.Feature_TEMPLATE_BUILD:
		return "template_build"
	case v1.Feature_VOLUME:
		return "volume"
	case v1.Feature_EGRESS_POLICY:
		return "egress_policy"
	case v1.Feature_CREDENTIAL_INJECTION:
		return "credential_injection"
	case v1.Feature_ISOLATION_RUNTIME:
		return "isolation_runtime"
	default:
		return ""
	}
}

var featureValues = map[string]v1.Feature{
	"freeze":               v1.Feature_FREEZE,
	"hibernate":            v1.Feature_HIBERNATE,
	"filesystem_snapshot":  v1.Feature_FILESYSTEM_SNAPSHOT,
	"full_state_snapshot":  v1.Feature_FULL_STATE_SNAPSHOT,
	"restore":              v1.Feature_RESTORE,
	"native_fork":          v1.Feature_NATIVE_FORK,
	"host_mount":           v1.Feature_HOST_MOUNT,
	"resume_anywhere":      v1.Feature_RESUME_ANYWHERE,
	"warm_pool":            v1.Feature_WARM_POOL,
	"image_on_demand":      v1.Feature_IMAGE_ON_DEMAND,
	"image_block_delivery": v1.Feature_IMAGE_BLOCK_DELIVERY,
	"template_build":       v1.Feature_TEMPLATE_BUILD,
	"volume":               v1.Feature_VOLUME,
	"egress_policy":        v1.Feature_EGRESS_POLICY,
	"credential_injection": v1.Feature_CREDENTIAL_INJECTION,
	"isolation_runtime":    v1.Feature_ISOLATION_RUNTIME,
}

func featureValue(name string) v1.Feature { return featureValues[name] }

func resumeLevelName(level v1.ResumeLevel) string {
	switch level {
	case v1.ResumeLevel_FILESYSTEM:
		return "filesystem"
	case v1.ResumeLevel_FULL_STATE:
		return "full_state"
	default:
		return ""
	}
}

func resumeLevelValue(name string) v1.ResumeLevel {
	switch name {
	case "filesystem":
		return v1.ResumeLevel_FILESYSTEM
	case "full_state":
		return v1.ResumeLevel_FULL_STATE
	default:
		return v1.ResumeLevel_RESUME_LEVEL_UNSPECIFIED
	}
}

func pauseModeName(mode v1.PauseMode) string {
	switch mode {
	case v1.PauseMode_PAUSE_FREEZE:
		return "freeze"
	case v1.PauseMode_PAUSE_HIBERNATE:
		return "hibernate"
	default:
		return ""
	}
}

func pauseModeValue(name string) v1.PauseMode {
	switch name {
	case "freeze":
		return v1.PauseMode_PAUSE_FREEZE
	case "hibernate":
		return v1.PauseMode_PAUSE_HIBERNATE
	default:
		return v1.PauseMode_PAUSE_MODE_UNSPECIFIED
	}
}

func snapshotKindName(kind v1.SnapshotKind) string {
	switch kind {
	case v1.SnapshotKind_SNAPSHOT_FILESYSTEM:
		return "filesystem"
	case v1.SnapshotKind_SNAPSHOT_FULL_STATE:
		return "full_state"
	default:
		return ""
	}
}

func sourceKindName(kind v1.Source_Kind) string {
	switch kind {
	case v1.Source_IMAGE:
		return "image"
	case v1.Source_TEMPLATE:
		return "template"
	default:
		return ""
	}
}

func execModeName(mode v1.ExecMode) string {
	switch mode {
	case v1.ExecMode_PERSISTENT:
		return "persistent"
	case v1.ExecMode_ONE_SHOT:
		return "one_shot"
	default:
		return ""
	}
}

func statusValue(name string) v1.SandboxStatus {
	switch name {
	case "running":
		return v1.SandboxStatus_RUNNING
	case "paused":
		return v1.SandboxStatus_PAUSED
	case "exited":
		return v1.SandboxStatus_EXITED
	case "terminated":
		return v1.SandboxStatus_TERMINATED
	default:
		return v1.SandboxStatus_SANDBOX_STATUS_UNSPECIFIED
	}
}

func orDefault(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}

func containsAt(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] == '@' {
			return true
		}
	}
	return false
}
