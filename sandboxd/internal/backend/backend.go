// Package backend is the contract a sandbox runtime implements.
//
// It is deliberately narrow. Everything a backend must provide is here;
// everything optional is a separate interface it may also implement, so a
// backend that cannot pause writes no pause method rather than one that returns
// an error.
//
// One method deserves its own explanation. AgentEndpoint is what makes the
// service affordable: an episode issues one create and dozens of commands, so
// command traffic goes straight from the caller to the agent inside the sandbox.
// A backend that could not report that address would force every command through
// the control plane, which adds a hop and a serialization per command without
// making a decision.
package backend

import (
	"context"
	"fmt"
	"time"
)

// SchedulingMode says who chooses the node a sandbox lands on.
//
// It is a deployment property rather than a backend capability: the same backend
// can run both ways in one fleet, which is what makes a scheduler ablation
// possible with every other variable held fixed.
type SchedulingMode string

const (
	// SchedulingPSRL means this service places sandboxes itself, against its own
	// fleet view and its own node admission, and the backend's control plane is
	// not started. Used where the backend's own scheduler has nothing to add.
	SchedulingPSRL SchedulingMode = "psrl"
	// SchedulingProvider means the backend's control plane chooses the node and
	// this service does cross-backend quota only. Used where the backend has a real
	// scheduler whose decisions would be lost by overriding it.
	SchedulingProvider SchedulingMode = "provider"
)

// Valid reports whether a mode is one of the two shapes.
func (m SchedulingMode) Valid() bool {
	return m == SchedulingPSRL || m == SchedulingProvider
}

// Source is where a sandbox's filesystem comes from.
type Source struct {
	// Kind is "image" or "template".
	Kind      string
	Reference string
}

// Resources is a portable resource request. A zero field states no requirement,
// which is not the same as requesting nothing.
type Resources struct {
	CPUCount float64
	MemoryMB int64
	DiskMB   int64
	GPUCount int32
}

// Spec is the portable creation request.
type Spec struct {
	Source    Source
	Resources Resources

	ResourceClass  string
	WorkflowID     string
	IdempotencyKey string

	Env      map[string]string
	Metadata map[string]string
	Workdir  string
	// ExecMode is "persistent" or "one_shot". Empty takes the backend's default.
	ExecMode string

	RequiredFeatures []string
	RequiredResume   string
	Backend          string

	// AssignedGPUs are the device indices node admission granted. Written by the
	// service and never by a caller, which is why it is not part of a spec's
	// identity.
	AssignedGPUs []int32

	// BackendOptions is tuning only one backend understands, keyed by backend
	// name. It never takes part in routing and is not part of a spec's identity,
	// so a provider's own knobs stay reachable without making the portable spec
	// unportable.
	BackendOptions map[string]map[string]string
}

// Options returns this backend's own options from a spec, and whether the spec
// named any.
func (s Spec) Options(backend string) (map[string]string, bool) {
	options, named := s.BackendOptions[backend]
	return options, named
}

// Handle identifies one live sandbox.
type Handle struct {
	Backend   string
	SandboxID string
	NodeID    string
}

// AgentEndpoint is where a caller reaches the agent inside a sandbox.
type AgentEndpoint struct {
	// Address is the agent's base URL, with a scheme.
	Address string
	// Headers the agent requires. Carried rather than reconstructed, because a
	// resumed sandbox can move and can require different ones.
	Headers map[string]string
	// CallbackHostAlias and CallbackPort are how this sandbox reaches the caller's
	// own server, when the hosting node rewrites loopback URLs at all.
	CallbackHostAlias string
	CallbackPort      int32
}

// Capabilities is what a backend actually grants, which is not always what a
// deployment declared.
type Capabilities struct {
	Features []string
	// ResumeLevel is reported only with the resume_anywhere feature, so the two
	// cannot contradict each other.
	ResumeLevel string
	// PauseModes are the pause semantics on offer: a freeze keeps the sandbox
	// resident, a hibernation releases its compute.
	PauseModes []string
}

// Supports reports whether a feature is implemented with its declared semantics.
func (c Capabilities) Supports(feature string) bool {
	for _, have := range c.Features {
		if have == feature {
			return true
		}
	}
	return false
}

// Missing returns the required features this backend does not have.
func (c Capabilities) Missing(required []string) []string {
	var missing []string
	for _, feature := range required {
		if !c.Supports(feature) {
			missing = append(missing, feature)
		}
	}
	return missing
}

// Created is what a backend returns for a new sandbox.
type Created struct {
	Handle       Handle
	Capabilities Capabilities
	Agent        AgentEndpoint
	// WarmStart reports that this sandbox came from a pool rather than a cold
	// create, which is the only way a caller learns the claim was warm.
	WarmStart bool
}

// Backend is the required contract. Every backend implements all of it.
type Backend interface {
	Name() string
	Mode() SchedulingMode
	Capabilities() Capabilities

	// Create provisions one sandbox. A node id is empty when the backend's own
	// control plane chooses, and names the node when this service does.
	Create(ctx context.Context, nodeID string, spec Spec, callback string) (Created, error)
	Release(ctx context.Context, handle Handle) error
	Status(ctx context.Context, handle Handle) (string, error)
}

// NodeScheduled is implemented by a backend this service can place against. A
// backend without it runs in provider mode only.
type NodeScheduled interface {
	// Nodes returns the backend's node ids, for the service's own placement.
	Nodes(ctx context.Context) ([]string, error)
	// Headroom returns what one node could grant each class right now.
	Headroom(ctx context.Context, nodeID string) (map[string]Resources, error)
	// RegisterBinding tells the backend where a sandbox landed, so its own routing
	// keeps working while this service does the placing. A backend with no such
	// concept implements it as a no-op.
	RegisterBinding(ctx context.Context, nodeID string, sandboxID string) error
}

// Stateful is implemented by a backend that can pause, resume, or snapshot.
type Stateful interface {
	Pause(ctx context.Context, handle Handle, mode string) error
	Resume(ctx context.Context, handle Handle) error
	Snapshot(ctx context.Context, handle Handle, kind string) (string, error)
	DeleteSnapshot(ctx context.Context, snapshotID string) error
}

// Deployer is implemented by a backend this service can install on a node, so a
// named backend deploys with no per-node manual step.
type Deployer interface {
	// Preflight refuses a node that cannot run this backend, at deploy time
	// rather than at the first rollout. A kernel too old or a missing device
	// surfaces here, where it is a configuration error, instead of inside an
	// episode where it is a lost sample.
	Preflight(ctx context.Context, nodeID string) error
	Deploy(ctx context.Context, nodeID string) error
	HealthCheck(ctx context.Context, nodeID string) error
	Teardown(ctx context.Context, nodeID string) error
}

// Registry resolves a spec to the backend that will serve it.
type Registry struct {
	backends map[string]Backend
	order    []string
	fallback string
}

// NewRegistry returns a registry over the configured backends. The default is
// tried first, so a deployment with one backend never pays for a search.
func NewRegistry(backends []Backend, defaultBackend string) (*Registry, error) {
	if len(backends) == 0 {
		return nil, fmt.Errorf("a sandbox service needs at least one backend")
	}
	byName := make(map[string]Backend, len(backends))
	order := make([]string, 0, len(backends))
	for _, b := range backends {
		if _, duplicate := byName[b.Name()]; duplicate {
			return nil, fmt.Errorf("sandbox backend %q is configured twice; a backend is keyed by its name", b.Name())
		}
		if !b.Mode().Valid() {
			return nil, fmt.Errorf("sandbox backend %q declares scheduling mode %q, want psrl or provider", b.Name(), b.Mode())
		}
		byName[b.Name()] = b
		order = append(order, b.Name())
	}
	if defaultBackend == "" {
		defaultBackend = order[0]
	}
	if _, known := byName[defaultBackend]; !known {
		return nil, fmt.Errorf("default sandbox backend %q is not configured", defaultBackend)
	}
	// The default first, then the rest in declaration order, so selection is
	// deterministic and a one-backend deployment short-circuits.
	ordered := append([]string{defaultBackend}, filter(order, defaultBackend)...)
	return &Registry{backends: byName, order: ordered, fallback: defaultBackend}, nil
}

// Get returns one backend by name.
func (r *Registry) Get(name string) (Backend, error) {
	if name == "" {
		return r.backends[r.fallback], nil
	}
	b, known := r.backends[name]
	if !known {
		return nil, fmt.Errorf("sandbox backend %q is not configured (configured: %v)", name, r.order)
	}
	return b, nil
}

// All returns every configured backend, in selection order.
func (r *Registry) All() []Backend {
	out := make([]Backend, 0, len(r.order))
	for _, name := range r.order {
		out = append(out, r.backends[name])
	}
	return out
}

// Select resolves a spec to a backend by capability, honouring a pin.
//
// Admission refuses rather than degrades: a caller that needs a live process to
// survive a move must never be handed a workspace-only resume and told it is the
// same thing.
func (r *Registry) Select(spec Spec) (Backend, error) {
	if spec.Backend != "" {
		return r.Get(spec.Backend)
	}
	reasons := make(map[string]string, len(r.order))
	for _, name := range r.order {
		b := r.backends[name]
		capabilities := b.Capabilities()
		if missing := capabilities.Missing(spec.RequiredFeatures); len(missing) > 0 {
			reasons[name] = fmt.Sprintf("missing features %v", missing)
			continue
		}
		if spec.RequiredResume != "" && !resumeSatisfies(capabilities.ResumeLevel, spec.RequiredResume) {
			reasons[name] = fmt.Sprintf("resumes at %q, which does not satisfy %q",
				capabilities.ResumeLevel, spec.RequiredResume)
			continue
		}
		return b, nil
	}
	return nil, fmt.Errorf(
		"no configured sandbox backend can serve this spec (%v). Declare the requirement on a backend "+
			"that supports it, or pin one", reasons)
}

var resumeRank = map[string]int{"": 0, "filesystem": 1, "full_state": 2}

func resumeSatisfies(have, need string) bool { return resumeRank[have] >= resumeRank[need] }

func filter(names []string, drop string) []string {
	out := make([]string, 0, len(names))
	for _, name := range names {
		if name != drop {
			out = append(out, name)
		}
	}
	return out
}

// RetryAfter is how long a caller should wait before asking again, when a
// backend reports its own backpressure rather than a failure.
type RetryAfter struct {
	After time.Duration
}

func (r RetryAfter) Error() string {
	return fmt.Sprintf("the backend is applying backpressure; retry after %s", r.After)
}
