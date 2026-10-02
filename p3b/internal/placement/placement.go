// Package placement chooses a node for one sandbox and protects the choice.
//
// Selection is capability match, then room, then balance, then image locality.
// The order is the design rather than a preference: a capability and room are
// requirements, and locality is an optimisation. Ranking locality first lets a
// per-task image draw a whole batch onto whichever node happens to hold it, and
// with no room check those requests reserve a node that can admit a fraction of
// them, so the rest queue invisibly for an episode's length.
//
// Placement decides; it does not collect. It reads one fleet view from a monitor
// and overlays the decisions it has made since that view was taken, because a
// reservation is charged here before the node can report it. Without that
// overlay a burst of concurrent requests all read the same view and all choose
// the same node.
package placement

import (
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"
)

// ErrNoCandidate means no live node could ever satisfy the request. It is a
// deployment fault and waiting will not fix it.
var ErrNoCandidate = errors.New("no sandbox node satisfies this request")

// ErrExhausted means a capable node exists but none has room right now. It is
// separate from ErrNoCandidate because the caller's recourse differs: this one
// clears itself as sandboxes are released.
var ErrExhausted = errors.New("every capable sandbox node is full")

// Headroom is a resource vector, in the units a node's envelope is denominated
// in, so no conversion is needed to compare a request against it.
type Headroom struct {
	MemoryMB  int64
	CPUMillis int64
	GPUCount  int32
	DiskMB    int64
}

// Sub returns the difference, clamped at zero.
func (h Headroom) Sub(o Headroom) Headroom {
	return Headroom{
		MemoryMB:  maxI64(h.MemoryMB-o.MemoryMB, 0),
		CPUMillis: maxI64(h.CPUMillis-o.CPUMillis, 0),
		GPUCount:  maxI32(h.GPUCount-o.GPUCount, 0),
		DiskMB:    maxI64(h.DiskMB-o.DiskMB, 0),
	}
}

// Add returns the sum.
func (h Headroom) Add(o Headroom) Headroom {
	return Headroom{h.MemoryMB + o.MemoryMB, h.CPUMillis + o.CPUMillis, h.GPUCount + o.GPUCount, h.DiskMB + o.DiskMB}
}

// BackendCapability is one runtime a node hosts, with what it can actually
// grant.
//
// Capability is per runtime rather than per node: one machine can run a
// container daemon and a microVM node side by side, and only one of them can
// carry a live process to another host. A node-level set would let a request
// needing a full-state resume land on a node that hosts such a runtime and then
// be served by its container daemon.
type BackendCapability struct {
	Name        string
	Features    map[string]struct{}
	ResumeLevel string
	HostMounts  bool
}

// Supports reports whether this runtime offers a feature.
func (b BackendCapability) Supports(feature string) bool {
	_, has := b.Features[feature]
	return has
}

// NodeView is one node's state, as the monitor last collected it.
type NodeView struct {
	NodeID string
	SeenAt time.Time

	// Backends are the runtimes this node hosts. A node hosting none can serve
	// nothing, so it is filtered out rather than read as unconstrained.
	Backends []BackendCapability

	// What each class could be granted now. A request is compared against its own
	// class, never against the envelope remainder.
	//
	// Node level rather than per runtime, because the runtimes share one machine:
	// a microVM and a container on the same host draw on the same memory, and
	// accounting them apart would admit twice what the node has.
	ClassHeadroom map[string]Headroom
	Envelope      Headroom

	LiveSandboxes int
	CPUUsedPct    float64
	MemUsedPct    float64

	ImageDigests    map[string]struct{}
	ImageReferences map[string]struct{}
	GPUFree         int32
	Labels          map[string]struct{}
	// A node that cannot destroy what it holds still holds its memory, so
	// admitting against it would over-commit the host.
	Draining bool
}

// Request is what a caller needs from a node.
type Request struct {
	Backend           string
	RequiredFeatures  []string
	RequiredResume    string
	RequiresHostMount bool
	GPUCount          int32
	RequiredLabel     string
	ImageDigests      []string
	ImageReferences   []string
	OwnerID           string

	// What the sandbox will cost the node that takes it, and which class pays.
	// Zero means the caller stated no footprint, which is not filtered on: a
	// request whose size is unknown cannot be compared against a remainder.
	Footprint     Headroom
	ResourceClass string
}

// Decision is one chosen node and the reservation protecting it.
type Decision struct {
	NodeID        string
	Backend       string
	ReservationID string
}

type reservation struct {
	id        string
	nodeID    string
	owner     string
	amount    Headroom
	renewedAt time.Time
}

// Monitor supplies the fleet view. Placement pulls rather than being pushed, so
// several replicas share one view and none depends on having seen every report.
//
// Version lets a reader skip the pull when nothing has changed. Choose is on the
// create path and a fleet view is hundreds of nodes, so copying one per decision
// is pure garbage: at 160 nodes it put three quarters of the service's time in
// the collector.
type Monitor interface {
	Fleet() []NodeView
	Version() uint64
}

// Config bounds the service's own clocks.
type Config struct {
	NodeTTL        time.Duration
	ReservationTTL time.Duration
	SweepInterval  time.Duration
	// BalanceBuckets is how finely utilisation is compared before locality is
	// allowed to decide. Coarse enough that similarly loaded nodes are a tie,
	// which is what leaves locality a real say without letting it override a node
	// that is genuinely fuller.
	BalanceBuckets int
}

// Validate refuses clocks the service cannot honour.
func (c Config) Validate() error {
	if c.NodeTTL <= 0 || c.ReservationTTL <= 0 || c.SweepInterval <= 0 {
		return fmt.Errorf("placement TTLs and sweep interval must be greater than zero")
	}
	if c.ReservationTTL >= c.NodeTTL {
		return fmt.Errorf(
			"placement reservation TTL (%s) must be shorter than the node TTL (%s), or a drained node "+
				"leaves reservations pointing at nothing", c.ReservationTTL, c.NodeTTL)
	}
	if c.SweepInterval >= c.ReservationTTL {
		return fmt.Errorf(
			"placement sweep interval (%s) must be shorter than the reservation TTL (%s), or a dead "+
				"owner's slot is freed later than its TTL promises", c.SweepInterval, c.ReservationTTL)
	}
	return nil
}

// Service picks nodes and holds the reservations that protect them.
type Service struct {
	mu      sync.Mutex
	cfg     Config
	monitor Monitor
	// pending is what this replica has promised since the last fleet view. It is
	// per node, and it is cleared for a node as soon as a fresher view arrives,
	// because by then the node's own accounting includes the grant.
	pending  map[string]Headroom
	viewedAt map[string]time.Time
	reserved map[string]*reservation
	now      func() time.Time

	// The last fleet view and the version it came from. Re-pulled only when the
	// monitor reports a newer one.
	fleet        []NodeView
	fleetVersion uint64
	fleetValid   bool
	// Scratch index slices, reused across decisions. Indices rather than values:
	// a NodeView is large, and sorting them by value spent a third of the service's
	// time copying structs.
	capable, roomy []int
	// matched is the runtime each candidate node was matched on, for this decision
	// only. Reused rather than allocated per create, which is on the hot path.
	matched map[string]string
	// Per-candidate utilisation, computed once per decision rather than on every
	// comparison the sort makes.
	scores []candidateScore

	decisions, noCandidate, exhausted, swept int64
	localityAsked, localityHits              int64
}

// New returns a placement service over a validated configuration.
func New(cfg Config, monitor Monitor) (*Service, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.BalanceBuckets <= 0 {
		cfg.BalanceBuckets = 10
	}
	return &Service{
		cfg:      cfg,
		monitor:  monitor,
		pending:  map[string]Headroom{},
		viewedAt: map[string]time.Time{},
		reserved: map[string]*reservation{},
		matched:  map[string]string{},
		now:      time.Now,
	}, nil
}

// SetClock replaces the service's clock, for a test that drives time directly.
func (s *Service) SetClock(now func() time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.now = now
}

// Choose picks a node that can actually hold the request and reserves it there.
func (s *Service) Choose(req Request) (Decision, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	s.sweepLocked(now)

	fleet := s.fleetLocked()
	live := 0
	capable, roomy := s.capable[:0], s.roomy[:0]
	matched := s.matched
	for i := range fleet {
		view := &fleet[i]
		if now.Sub(view.SeenAt) > s.cfg.NodeTTL {
			continue
		}
		s.refreshPendingLocked(view)
		live++
		runtime, ok := s.satisfies(view, req)
		if !ok {
			continue
		}
		// The matched runtime is remembered per node, so the winner provisions on
		// the one its capabilities were actually checked against.
		matched[view.NodeID] = runtime
		capable = append(capable, i)
		if s.canHold(view, req) {
			roomy = append(roomy, i)
		}
	}
	s.capable, s.roomy = capable, roomy
	if len(capable) == 0 {
		s.noCandidate++
		return Decision{}, fmt.Errorf("%w (live nodes: %d, backend: %q, features: %v): %s",
			ErrNoCandidate, live, req.Backend, req.RequiredFeatures,
			"declare the requirement on a backend that provides it, or add a node that does")
	}
	if len(roomy) == 0 {
		s.exhausted++
		return Decision{}, fmt.Errorf("%w: the request needs %dMB and %dmcpu, and %d capable node(s) have "+
			"no room. This clears as sandboxes are released",
			ErrExhausted, req.Footprint.MemoryMB, req.Footprint.CPUMillis, len(capable))
	}

	chosen := &fleet[s.best(fleet, roomy, req)]
	id := fmt.Sprintf("rsv-%d-%s", now.UnixNano(), chosen.NodeID)
	s.reserved[id] = &reservation{
		id: id, nodeID: chosen.NodeID, owner: req.OwnerID,
		amount: req.Footprint, renewedAt: now,
	}
	// Charge the promise now, so the next caller in this same burst sees the room
	// go even though the node cannot report it until the grant lands.
	s.pending[chosen.NodeID] = s.pending[chosen.NodeID].Add(req.Footprint)
	s.decisions++
	s.noteLocality(chosen, req)
	return Decision{NodeID: chosen.NodeID, Backend: matched[chosen.NodeID], ReservationID: id}, nil
}

// fleetLocked returns the cached fleet view, pulling only when it has changed.
func (s *Service) fleetLocked() []NodeView {
	if version := s.monitor.Version(); !s.fleetValid || version != s.fleetVersion {
		s.fleet = s.monitor.Fleet()
		s.fleetVersion = version
		s.fleetValid = true
	}
	return s.fleet
}

// refreshPendingLocked drops the promises a fresher node view already accounts for.
func (s *Service) refreshPendingLocked(view *NodeView) {
	if last, seen := s.viewedAt[view.NodeID]; !seen || view.SeenAt.After(last) {
		s.viewedAt[view.NodeID] = view.SeenAt
		delete(s.pending, view.NodeID)
	}
}

// satisfies returns the runtime on this node that can host a request, and
// whether one exists.
//
// Deliberately strict. A runtime that cannot bind a host path, or resumes at a
// weaker level, is not a slower candidate; it is a wrong one. The match is per
// runtime because that is where capability lives: a node hosting both a microVM
// and a container daemon satisfies a full-state resume only through the former,
// and the chosen name travels in the decision so the node provisions on the
// runtime that was actually matched.
func (s *Service) satisfies(view *NodeView, req Request) (string, bool) {
	if view.Draining {
		return "", false
	}
	if req.GPUCount > view.GPUFree {
		return "", false
	}
	if req.RequiredLabel != "" {
		if _, has := view.Labels[req.RequiredLabel]; !has {
			return "", false
		}
	}
	for _, hosted := range view.Backends {
		if req.Backend != "" && hosted.Name != req.Backend {
			continue
		}
		if !capable(hosted, req) {
			continue
		}
		return hosted.Name, true
	}
	return "", false
}

// capable reports whether one runtime meets a request's stated requirements.
func capable(hosted BackendCapability, req Request) bool {
	for _, feature := range req.RequiredFeatures {
		if !hosted.Supports(feature) {
			return false
		}
	}
	if req.RequiredResume != "" && !resumeSatisfies(hosted.ResumeLevel, req.RequiredResume) {
		return false
	}
	if req.RequiresHostMount && !hosted.HostMounts {
		return false
	}
	return true
}

// canHold reports whether a node still has room for one request, net of what
// this replica has already promised it.
func (s *Service) canHold(view *NodeView, req Request) bool {
	free := s.freeFor(view, req.ResourceClass)
	if req.Footprint.MemoryMB > 0 && req.Footprint.MemoryMB > free.MemoryMB {
		return false
	}
	if req.Footprint.CPUMillis > 0 && req.Footprint.CPUMillis > free.CPUMillis {
		return false
	}
	if req.Footprint.DiskMB > 0 && req.Footprint.DiskMB > free.DiskMB {
		return false
	}
	return true
}

// freeFor returns what one class could still get on a node, net of promises.
//
// A node that reports no class headroom declared no envelope, which means it
// schedules its own capacity: this service has no basis to refuse it, so it is
// reported as unbounded rather than as full.
func (s *Service) freeFor(view *NodeView, class string) Headroom {
	reported, declared := view.ClassHeadroom[class]
	if !declared {
		if len(view.ClassHeadroom) == 0 {
			return Headroom{MemoryMB: 1 << 62, CPUMillis: 1 << 62, GPUCount: 1 << 30, DiskMB: 1 << 62}
		}
		reported = view.Envelope
	}
	return reported.Sub(s.pending[view.NodeID])
}

type candidateScore struct {
	index    int
	bucket   int
	locality int
	load     int
}

// better reports whether one candidate outranks another.
//
// Utilisation is bucketed rather than compared exactly, so locality still decides
// between nodes that are similarly loaded, which is what makes it an optimisation
// rather than a tie-break that never fires. The node id is last so two equally
// good nodes do not alternate, which would make placement irreproducible.
func better(a, b candidateScore, fleet []NodeView) bool {
	if a.bucket != b.bucket {
		return a.bucket < b.bucket
	}
	if a.locality != b.locality {
		return a.locality > b.locality
	}
	if a.load != b.load {
		return a.load < b.load
	}
	return fleet[a.index].NodeID < fleet[b.index].NodeID
}

// best returns the index of the winning candidate.
//
// A linear scan rather than a sort: only the winner is used, so ordering the rest
// is work nobody reads. Each candidate is scored once, because utilisation walks
// a map and a comparison sort would recompute it O(n log n) times.
func (s *Service) best(fleet []NodeView, candidates []int, req Request) int {
	scores := s.scores[:0]
	for _, index := range candidates {
		view := &fleet[index]
		utilisation := s.utilisation(view, req.ResourceClass)
		scores = append(scores, candidateScore{
			index:    index,
			bucket:   int(utilisation * float64(s.cfg.BalanceBuckets)),
			locality: localityScore(view, req),
			load:     view.LiveSandboxes,
		})
	}
	s.scores = scores
	winner := scores[0]
	for _, score := range scores[1:] {
		if better(score, winner, fleet) {
			winner = score
		}
	}
	return winner.index
}

// utilisation returns how full a node is, as the worse of its two dimensions.
//
// A fraction rather than an absolute, so two nodes of different sizes order by
// how loaded they are rather than by how large they are.
func (s *Service) utilisation(view *NodeView, class string) float64 {
	free := s.freeFor(view, class)
	worst := 0.0
	if view.Envelope.MemoryMB > 0 {
		worst = maxF(worst, 1-float64(free.MemoryMB)/float64(view.Envelope.MemoryMB))
	}
	if view.Envelope.CPUMillis > 0 {
		worst = maxF(worst, 1-float64(free.CPUMillis)/float64(view.Envelope.CPUMillis))
	}
	return worst
}

// localityScore returns how well a node already holds the images a request names.
// A digest match is worth more than a reference match, because a digest is exact
// and a tag can be moved.
func localityScore(view *NodeView, req Request) int {
	score := 0
	for _, digest := range req.ImageDigests {
		if _, has := view.ImageDigests[digest]; has {
			score += 2
		}
	}
	for _, ref := range req.ImageReferences {
		if _, has := view.ImageReferences[ref]; has {
			score++
		}
	}
	return score
}

func (s *Service) noteLocality(view *NodeView, req Request) {
	if len(req.ImageDigests) == 0 && len(req.ImageReferences) == 0 {
		return
	}
	s.localityAsked++
	if localityScore(view, req) > 0 {
		s.localityHits++
	}
}

// Renew says a reservation's owner is still holding it.
func (s *Service) Renew(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if r, ok := s.reserved[id]; ok {
		r.renewedAt = s.now()
	}
}

// Release returns a reservation whose sandbox ran and finished.
func (s *Service) Release(id string) { s.retire(id) }

// Cancel withdraws a reservation whose provision never happened. Counted apart
// from a release, so a fleet losing slots to failed provisions is visible.
func (s *Service) Cancel(id string) { s.retire(id) }

func (s *Service) retire(id string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.reserved[id]
	if !ok {
		return
	}
	delete(s.reserved, id)
	s.pending[r.nodeID] = s.pending[r.nodeID].Sub(r.amount)
}

// Sweep drops reservations whose owner stopped renewing, returning their ids.
func (s *Service) Sweep() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sweepLocked(s.now())
}

func (s *Service) sweepLocked(now time.Time) []string {
	var expired []string
	for id, r := range s.reserved {
		if now.Sub(r.renewedAt) > s.cfg.ReservationTTL {
			expired = append(expired, id)
		}
	}
	sort.Strings(expired)
	for _, id := range expired {
		r := s.reserved[id]
		delete(s.reserved, id)
		s.pending[r.nodeID] = s.pending[r.nodeID].Sub(r.amount)
	}
	s.swept += int64(len(expired))
	return expired
}

// Report is the service's point-in-time accounting for the metric hook.
type Report struct {
	Nodes             int
	DrainedNodes      int
	ReservationsOpen  int
	ReservationsSwept int64
	Decisions         int64
	NoCandidate       int64
	CapacityExhausted int64
	LocalityHitRatio  float64
}

// Snapshot returns the service's counters. The planes report and never log.
func (s *Service) Snapshot() Report {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := s.now()
	nodes, drained := 0, 0
	fleet := s.fleetLocked()
	for i := range fleet {
		view := &fleet[i]
		nodes++
		if view.Draining || now.Sub(view.SeenAt) > s.cfg.NodeTTL {
			drained++
		}
	}
	ratio := 0.0
	if s.localityAsked > 0 {
		ratio = float64(s.localityHits) / float64(s.localityAsked)
	}
	return Report{
		Nodes: nodes, DrainedNodes: drained,
		ReservationsOpen: len(s.reserved), ReservationsSwept: s.swept,
		Decisions: s.decisions, NoCandidate: s.noCandidate,
		CapacityExhausted: s.exhausted, LocalityHitRatio: ratio,
	}
}

var resumeRank = map[string]int{"": 0, "filesystem": 1, "full_state": 2}

func resumeSatisfies(have, need string) bool { return resumeRank[have] >= resumeRank[need] }

func maxI64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func maxI32(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}

func maxF(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
