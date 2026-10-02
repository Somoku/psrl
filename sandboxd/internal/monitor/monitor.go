// Package monitor holds one view of the fleet and serves it to placement.
//
// Collection and decision are separate on purpose. Placement reads a view; it
// does not accumulate one from whatever heartbeat happened to arrive. That is
// what lets several placement replicas share a consistent picture, and what
// keeps a node's liveness from being a side effect of the last create that
// touched it.
//
// The view is a cache, never a ledger. On restart it rebuilds from the nodes
// themselves, so losing it costs one report interval and nothing else.
package monitor

import (
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"psrl.dev/sandboxd/internal/placement"
)

// Monitor collects node reports and serves the fleet view.
//
// Safe for concurrent use: nodes report from many goroutines while placement
// replicas read.
type Monitor struct {
	mu      sync.RWMutex
	nodes   map[string]placement.NodeView
	nodeTTL time.Duration
	now     func() time.Time

	// view is the materialised fleet, rebuilt only when a report changes it, so a
	// reader on the create path never pays for the copy. version lets that reader
	// skip the read lock entirely when nothing has moved.
	view    []placement.NodeView
	version atomic.Uint64

	reports atomic.Int64
}

// New returns a monitor that drains a node from the fleet after nodeTTL of
// silence.
func New(nodeTTL time.Duration) *Monitor {
	return &Monitor{
		nodes:   map[string]placement.NodeView{},
		nodeTTL: nodeTTL,
		now:     time.Now,
	}
}

// SetClock replaces the monitor's clock, for a test that drives time directly.
func (m *Monitor) SetClock(now func() time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.now = now
}

// Report records one node's view of itself.
//
// A node reports its own liveness rather than having it inferred, because
// liveness belongs to the node: one worker dying must not drain a node that
// other workers are still using.
func (m *Monitor) Report(view placement.NodeView) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if view.SeenAt.IsZero() {
		view.SeenAt = m.now()
	}
	m.nodes[view.NodeID] = view
	m.rebuildLocked()
}

// Forget drops a node, for an agent that is shutting down cleanly. Waiting for
// its TTL would send work to a node that has already said it is leaving.
func (m *Monitor) Forget(nodeID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, known := m.nodes[nodeID]; !known {
		return
	}
	delete(m.nodes, nodeID)
	m.rebuildLocked()
}

// rebuildLocked materialises the fleet in a stable order.
//
// Sorted by node id so two placement replicas reading the same version see the
// same order, which is what makes a tie-break reproducible across replicas.
func (m *Monitor) rebuildLocked() {
	view := make([]placement.NodeView, 0, len(m.nodes))
	for _, node := range m.nodes {
		view = append(view, node)
	}
	sort.Slice(view, func(i, j int) bool { return view[i].NodeID < view[j].NodeID })
	m.view = view
	m.version.Add(1)
	m.reports.Add(1)
}

// Fleet returns the current view. The slice is shared and must not be mutated by
// a caller; placement treats it as read-only and copies nothing.
func (m *Monitor) Fleet() []placement.NodeView {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.view
}

// Version changes whenever the fleet does, so a reader can skip an unchanged pull.
func (m *Monitor) Version() uint64 { return m.version.Load() }

// Report is the monitor's accounting for the caller's metric hook.
type Report struct {
	Nodes        int
	LiveNodes    int
	DrainedNodes int
	Reports      int64
}

// Snapshot returns what the monitor sees. The planes report and never log.
func (m *Monitor) Snapshot() Report {
	m.mu.RLock()
	defer m.mu.RUnlock()
	now := m.now()
	live := 0
	for _, node := range m.nodes {
		if now.Sub(node.SeenAt) <= m.nodeTTL && !node.Draining {
			live++
		}
	}
	return Report{
		Nodes:        len(m.nodes),
		LiveNodes:    live,
		DrainedNodes: len(m.nodes) - live,
		Reports:      m.reports.Load(),
	}
}
