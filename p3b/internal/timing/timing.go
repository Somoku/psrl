// Package timing holds the fleet-liveness spans the service enforces.
//
// Episode-scale deadlines are gone. A sandbox's own lifetime is stated per create
// and travels with the spec, and a queue wait is answered by admission -- either
// the request fits now, it will fit when other sandboxes are released, or it can
// never fit and is refused -- rather than by a deadline that guesses. So the
// ratios that once turned "one episode" into a pause window, a reap window, and a
// lifetime have nothing left to derive from.
//
// What is left is how fast the fleet notices that a node or an owner has gone
// away. Those describe how fast the mechanism answers, not how long a workload
// runs, so they are constants here rather than knobs a deployment states.
package timing

import (
	"fmt"
	"time"
)

// Defaults for the two values that describe how fast a node answers rather than
// how long a workload runs.
const (
	DefaultNodeTTL    = 120 * time.Second
	DefaultRPCTimeout = 60 * time.Second
)

// The mechanism's own fault-detection spans.
const (
	// OwnerLease bounds a grant whose owner stopped renewing. It is constant rather
	// than derived, and deliberately larger than any sandbox lifetime, so a long
	// episode cannot lose the reservation that holds its memory. A sandbox is
	// reclaimed by its own lifetime; this only bounds a dead owner's accounting.
	OwnerLease = 4 * time.Hour
	// ReservationTTL outlives the gap between a placement promise and the node's
	// next report, which carries the grant. Provisioning renews it, so it does not
	// have to cover a cold image pull.
	ReservationTTL = 60 * time.Second
	// SweepInterval drives the reclamation sweep, the lease expiry, and placement's
	// reservation sweep. It is faster than the shortest span it enforces.
	SweepInterval = 15 * time.Second

	// reportsPerTTL is how many reports fit inside a node TTL, so one missed report
	// never drains a live node.
	reportsPerTTL = 4
)

// Contract is the fleet-liveness configuration.
//
// Everything episode-scale that used to be derived here is stated where it
// belongs: a sandbox's lifetime on its create, and a queue wait at admission.
type Contract struct {
	NodeTTL    time.Duration
	RPCTimeout time.Duration
}

// New returns a validated contract.
func New(nodeTTL, rpcTimeout time.Duration) (Contract, error) {
	if nodeTTL <= 0 {
		nodeTTL = DefaultNodeTTL
	}
	if rpcTimeout <= 0 {
		rpcTimeout = DefaultRPCTimeout
	}
	c := Contract{NodeTTL: nodeTTL, RPCTimeout: rpcTimeout}
	if err := c.Validate(); err != nil {
		return Contract{}, err
	}
	return c, nil
}

// Validate refuses a contract whose spans are in an order the code cannot honour.
func (c Contract) Validate() error {
	if c.NodeTTL <= 0 || c.RPCTimeout <= 0 {
		return fmt.Errorf("timing node TTL and RPC timeout must be greater than zero")
	}
	return nil
}

// LoadReportInterval is how often a node publishes its view, several times per
// TTL so one lost report never drains a live node.
func (c Contract) LoadReportInterval() time.Duration {
	return c.NodeTTL / reportsPerTTL
}

// Spans returns every resolved span, for an operator dump or a metric hook.
func (c Contract) Spans() map[string]time.Duration {
	return map[string]time.Duration{
		"node_ttl":            c.NodeTTL,
		"rpc_timeout":         c.RPCTimeout,
		"load_report_interval": c.LoadReportInterval(),
		"owner_lease":         OwnerLease,
		"reservation_ttl":     ReservationTTL,
		"sweep_interval":      SweepInterval,
	}
}
