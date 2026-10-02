// Package timing derives every deadline the service enforces from three
// declared values.
//
// An operator can estimate one number: how long an episode takes. Everything
// else follows from it, so a deployment states an intent and the code derives
// the rest. The ratios are properties of the mechanism rather than of a
// deployment, which is why they are constants here and not knobs.
//
// Every ordering is asserted in one place. Before this, the orderings lived in
// two files that did not know about each other, so the cross-level ones -- a
// node reporting less often than its own TTL, a sweep slower than the lease it
// enforces -- were comments rather than checks.
//
// It mirrors psrl/sandbox/timing.py deliberately: while both implementations
// coexist they must derive the same spans from the same inputs, or a node and
// its control plane would disagree about when a sandbox is idle.
package timing

import (
	"fmt"
	"time"
)

// Ratios. A pause window shorter than an episode would pause a running episode,
// and a lifetime close to the reap window would fire before the reaper ran.
const (
	pauseEpisodes       = 2
	reapPauseWindows    = 3
	lifetimeReapWindows = 4
	// A report, a renewal, and a sweep each run several times inside the deadline
	// they protect, so one lost round trip never costs the thing it protected.
	reportsPerTTL  = 4
	renewalsPerTTL = 3
	sweepsPerTTL   = 4
	// A reservation is held by a caller and a node record by the fleet, so a
	// reservation must expire first or a drained node leaves reservations
	// pointing at nothing.
	reservationTTLNumerator   = 1
	reservationTTLDenominator = 2
	// Crash recovery is two phases, and both must finish inside the capacity lease.
	lifecycleHalf = 2
)

// Defaults for the two values that describe how fast a node answers rather than
// how long a workload runs.
const (
	DefaultNodeTTL    = 120 * time.Second
	DefaultRPCTimeout = 60 * time.Second
)

// Contract is every deadline the service enforces.
type Contract struct {
	// EpisodeDeadline is the only value a deployment must state.
	EpisodeDeadline time.Duration
	NodeTTL         time.Duration
	RPCTimeout      time.Duration

	// Overrides replace a derived span. An override is validated exactly like a
	// derived one, so one that inverts an ordering is refused rather than honoured.
	Overrides map[string]time.Duration
}

// New returns a validated contract.
func New(episodeDeadline, nodeTTL, rpcTimeout time.Duration, overrides map[string]time.Duration) (Contract, error) {
	if nodeTTL <= 0 {
		nodeTTL = DefaultNodeTTL
	}
	if rpcTimeout <= 0 {
		rpcTimeout = DefaultRPCTimeout
	}
	c := Contract{EpisodeDeadline: episodeDeadline, NodeTTL: nodeTTL, RPCTimeout: rpcTimeout, Overrides: overrides}
	if err := c.Validate(); err != nil {
		return Contract{}, err
	}
	return c, nil
}

func (c Contract) value(name string, derived time.Duration) time.Duration {
	if override, named := c.Overrides[name]; named && override > 0 {
		return override
	}
	return derived
}

// PauseWindow is the idle span after which a sandbox releases its compute.
func (c Contract) PauseWindow() time.Duration {
	return c.value("pause_window", c.EpisodeDeadline*pauseEpisodes)
}

// ReapWindow is the idle span after which a sandbox is destroyed instead.
func (c Contract) ReapWindow() time.Duration {
	return c.value("reap_window", c.PauseWindow()*reapPauseWindows)
}

// Lifetime bounds a sandbox that is never idle.
func (c Contract) Lifetime() time.Duration {
	return c.value("lifetime", c.ReapWindow()*lifetimeReapWindows)
}

// AcquireTimeout is how long admission may queue before the wait is the fault.
func (c Contract) AcquireTimeout() time.Duration {
	return c.value("acquire_timeout", c.EpisodeDeadline)
}

// LoadReportInterval is how often a node publishes its view.
func (c Contract) LoadReportInterval() time.Duration {
	return c.value("load_report_interval", c.NodeTTL/reportsPerTTL)
}

// MonitorPullInterval is how often the fleet view is collected.
func (c Contract) MonitorPullInterval() time.Duration {
	return c.value("monitor_pull_interval", c.NodeTTL/reportsPerTTL)
}

// ReservationTTL is how long a placement reservation survives unrenewed.
func (c Contract) ReservationTTL() time.Duration {
	return c.value("reservation_ttl", c.NodeTTL*reservationTTLNumerator/reservationTTLDenominator)
}

// ReservationRenewInterval is how often an owner says it still holds its
// reservations.
func (c Contract) ReservationRenewInterval() time.Duration {
	return c.value("reservation_renew_interval", c.ReservationTTL()/renewalsPerTTL)
}

// CapacityLeaseTTL is how long a grant outlives its owner's silence. It covers
// the pause window, because a paused sandbox still holds the memory its
// reservation covers.
func (c Contract) CapacityLeaseTTL() time.Duration {
	return c.value("capacity_lease_ttl", c.PauseWindow())
}

// OwnerHeartbeatInterval is how often an owner refreshes its capacity lease.
func (c Contract) OwnerHeartbeatInterval() time.Duration {
	return c.value("owner_heartbeat_interval", c.CapacityLeaseTTL()/renewalsPerTTL)
}

// SweepInterval is the reclamation cadence, faster than the shortest span it
// enforces.
func (c Contract) SweepInterval() time.Duration {
	shortest := c.PauseWindow()
	if ttl := c.ReservationTTL(); ttl < shortest {
		shortest = ttl
	}
	if lease := c.CapacityLeaseTTL(); lease < shortest {
		shortest = lease
	}
	derived := shortest / sweepsPerTTL
	if derived < time.Second {
		derived = time.Second
	}
	return c.value("sweep_interval", derived)
}

// LifecycleLeaseTTL and LifecycleGCInterval are the two phases of crash
// recovery, which must complete inside the capacity lease.
func (c Contract) LifecycleLeaseTTL() time.Duration {
	return c.value("lifecycle_lease_ttl", c.CapacityLeaseTTL()/lifecycleHalf)
}

// LifecycleGCInterval is how often crash recovery sweeps.
func (c Contract) LifecycleGCInterval() time.Duration {
	return c.value("lifecycle_gc_interval", c.LifecycleLeaseTTL()/lifecycleHalf)
}

// Validate refuses a contract whose spans are in an order the code cannot
// honour. Each rule is something the service relies on, and a violation
// produces a slow leak or a sandbox reclaimed while in use -- neither of which
// points back at the configuration.
func (c Contract) Validate() error {
	if c.EpisodeDeadline <= 0 {
		return fmt.Errorf("timing requires an episode deadline greater than zero")
	}
	if c.NodeTTL <= 0 || c.RPCTimeout <= 0 {
		return fmt.Errorf("timing node TTL and RPC timeout must be greater than zero")
	}
	checks := []struct {
		ok  bool
		why string
	}{
		{c.PauseWindow() < c.ReapWindow(), fmt.Sprintf(
			"pause window (%s) must be shorter than the reap window (%s), or a sandbox is destroyed before "+
				"it is ever paused", c.PauseWindow(), c.ReapWindow())},
		{c.ReapWindow() < c.Lifetime(), fmt.Sprintf(
			"reap window (%s) must be shorter than the lifetime (%s), or the backstop fires first and the "+
				"reaper never runs", c.ReapWindow(), c.Lifetime())},
		{c.CapacityLeaseTTL() >= c.PauseWindow(), fmt.Sprintf(
			"capacity lease (%s) must cover the pause window (%s), or a paused sandbox loses the reservation "+
				"holding its memory", c.CapacityLeaseTTL(), c.PauseWindow())},
		{c.OwnerHeartbeatInterval() < c.CapacityLeaseTTL(), fmt.Sprintf(
			"owner heartbeat (%s) must be shorter than the capacity lease (%s), or a live owner loses its "+
				"reservation", c.OwnerHeartbeatInterval(), c.CapacityLeaseTTL())},
		{c.LoadReportInterval()*3 < c.NodeTTL, fmt.Sprintf(
			"load report interval (%s) must fit several times into the node TTL (%s), or one missed report "+
				"drains a live node", c.LoadReportInterval(), c.NodeTTL)},
		{c.ReservationTTL() < c.NodeTTL, fmt.Sprintf(
			"reservation TTL (%s) must be shorter than the node TTL (%s), or a drained node leaves "+
				"reservations pointing at nothing", c.ReservationTTL(), c.NodeTTL)},
		{c.ReservationRenewInterval() < c.ReservationTTL(), fmt.Sprintf(
			"reservation renewal (%s) must be shorter than the reservation TTL (%s), or every reservation "+
				"expires under renewal", c.ReservationRenewInterval(), c.ReservationTTL())},
		{c.SweepInterval() < c.PauseWindow() && c.SweepInterval() < c.ReservationTTL(), fmt.Sprintf(
			"sweep interval (%s) must be shorter than the shortest span it enforces, or a window is reported "+
				"long after it elapsed", c.SweepInterval())},
		{c.LifecycleLeaseTTL()+c.LifecycleGCInterval() < c.CapacityLeaseTTL(), fmt.Sprintf(
			"crash recovery (lease %s plus sweep %s) must complete inside the capacity lease (%s), or a dead "+
				"owner's containers outlive the reservation that protected the node",
			c.LifecycleLeaseTTL(), c.LifecycleGCInterval(), c.CapacityLeaseTTL())},
	}
	for _, check := range checks {
		if !check.ok {
			return fmt.Errorf("timing: %s", check.why)
		}
	}
	return nil
}

// Spans returns every resolved deadline, for an operator dump or a metric hook.
func (c Contract) Spans() map[string]time.Duration {
	return map[string]time.Duration{
		"pause_window":               c.PauseWindow(),
		"reap_window":                c.ReapWindow(),
		"lifetime":                   c.Lifetime(),
		"acquire_timeout":            c.AcquireTimeout(),
		"load_report_interval":       c.LoadReportInterval(),
		"monitor_pull_interval":      c.MonitorPullInterval(),
		"reservation_ttl":            c.ReservationTTL(),
		"reservation_renew_interval": c.ReservationRenewInterval(),
		"capacity_lease_ttl":         c.CapacityLeaseTTL(),
		"owner_heartbeat_interval":   c.OwnerHeartbeatInterval(),
		"sweep_interval":             c.SweepInterval(),
		"lifecycle_lease_ttl":        c.LifecycleLeaseTTL(),
		"lifecycle_gc_interval":      c.LifecycleGCInterval(),
	}
}
