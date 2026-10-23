package node

import (
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// allocState is where one allocation is in its life.
//
// Every transition happens under the lifecycle's lock, so an illegal one is a
// state assertion rather than a race.
type allocState string

const (
	// allocReserved is a grant with no runtime object yet: admission has charged
	// for it and no create has produced a container.
	allocReserved allocState = "reserved"
	// allocCreating is a create in flight, or one whose outcome is not known yet.
	allocCreating allocState = "creating"
	// allocResident is a runtime object that exists and is charged for.
	allocResident allocState = "resident"
	// allocDestroying is a destroy in flight, so a second release on the same
	// record does nothing rather than issuing a second destroy.
	allocDestroying allocState = "destroying"
	// allocCleanupFailed is a destroy the runtime refused. The object may still
	// exist, so the record stays charged and is retried after a backoff. It is
	// also what puts the node in draining.
	allocCleanupFailed allocState = "cleanup_failed"
)

// allocation is one sandbox end to end: the envelope it holds and the runtime
// object it holds that envelope against.
//
// One record rather than a grant here and a container there, because two copies
// of one fact drift. That drift is what let a grant be returned while its
// container was still holding memory.
type allocation struct {
	leaseID   string
	sandboxID string
	backend   backend.Backend
	handle    backend.Handle
	class     string
	owner     string
	execMode  string
	grant     Grant
	state     allocState

	createdAt time.Time
	// lifetime is the absolute span this sandbox may exist, stated on its create.
	lifetime time.Duration
	// lastActivity is stamped on both command boundaries, for a report that has
	// to tell a sandbox running a long command from one sitting unused.
	lastActivity time.Time
	inFlight     int

	// pendingReason is why the sweep decided to remove this record, kept until
	// the removal confirms so the record can be attributed when it is dropped.
	pendingReason ExitReason
	// pendingExitCode and pendingFinishedAt are the runtime's attribution for
	// the stop, carried from the decision to removeLocked so the caller that
	// discovered the loss can report it before cleanup runs.
	pendingExitCode   int
	pendingCodeKnown  bool
	pendingFinishedAt time.Time
	pendingStateError string
	// cleanupAttempts and nextCleanupAt pace the retries of a refused removal.
	cleanupAttempts int
	nextCleanupAt   time.Time

	lastError string
}

// holdsRuntime reports whether this record owns a container that must be
// destroyed before its budget can come back.
func (a *allocation) holdsRuntime() bool {
	switch a.state {
	case allocResident, allocDestroying, allocCleanupFailed:
		return true
	default:
		return false
	}
}
