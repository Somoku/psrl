package node

import (
	"context"
	"sync"
	"time"
)

// Cleanup is how a node destroys sandboxes it has decided to reclaim.
//
// Destruction is the one part of reclamation that talks to a runtime, and a
// runtime can be slow, wedged, or unreachable. Doing it on the sweep would let
// one stuck daemon hold up every other reclamation, and doing it on the create
// path would let a create burst starve reclamation. So the sweep decides and
// this carries the decision out, on its own workers and its own budget.
//
// A destruction that fails is not forgotten: the record stays charged, because
// a container the runtime refused to remove still holds its memory. Retrying is
// also safe, since removing a container that is already gone is what the
// backend already treats as success.
type Cleanup struct {
	life    *Lifecycle
	queue   chan string
	workers chan struct{}

	// pending counts decisions enqueued but not yet carried out, so a test can
	// wait for the effects of a sweep rather than sleeping on it.
	pending sync.WaitGroup

	mu      sync.Mutex
	started bool
	stop    chan struct{}
	stopped sync.WaitGroup
}

// CleanupConfig is what an operator states about reclamation's own concurrency.
type CleanupConfig struct {
	// Workers is how many destructions may be in flight at once. It is separate
	// from the backend's create concurrency on purpose: the two contend for the
	// same daemon, and sharing one budget would let either starve the other.
	Workers int
	// QueueDepth bounds how many decisions may wait for a worker. A sweep that
	// finds more than this leaves the rest for its next pass, which is why the
	// queue is bounded rather than unbounded.
	QueueDepth int
	// Backoff is the first retry delay after a failed destruction. It doubles per
	// attempt up to MaxBackoff.
	Backoff    time.Duration
	MaxBackoff time.Duration
	// AttemptTimeout bounds one destruction. A removal is a local operation, so
	// it has no reason to wait as long as an image pull.
	AttemptTimeout time.Duration
}

// DefaultCleanupConfig is the reclamation concurrency a node runs with.
func DefaultCleanupConfig() CleanupConfig {
	return CleanupConfig{
		Workers:        8,
		QueueDepth:     1024,
		Backoff:        2 * time.Second,
		MaxBackoff:     30 * time.Second,
		AttemptTimeout: 60 * time.Second,
	}
}

func (c CleanupConfig) validated() CleanupConfig {
	defaults := DefaultCleanupConfig()
	if c.Workers <= 0 {
		c.Workers = defaults.Workers
	}
	if c.QueueDepth <= 0 {
		c.QueueDepth = defaults.QueueDepth
	}
	if c.Backoff <= 0 {
		c.Backoff = defaults.Backoff
	}
	if c.MaxBackoff < c.Backoff {
		c.MaxBackoff = defaults.MaxBackoff
	}
	if c.AttemptTimeout <= 0 {
		c.AttemptTimeout = defaults.AttemptTimeout
	}
	return c
}

// newCleanup returns the reclamation workers for one lifecycle.
func newCleanup(life *Lifecycle, cfg CleanupConfig) *Cleanup {
	cfg = cfg.validated()
	return &Cleanup{
		life:    life,
		queue:   make(chan string, cfg.QueueDepth),
		workers: make(chan struct{}, cfg.Workers),
	}
}

// backoffAfter is how long to wait before the given attempt number.
func (c *Cleanup) backoffAfter(attempts int) time.Duration {
	cfg := c.life.cleanupCfg
	delay := cfg.Backoff
	for i := 1; i < attempts; i++ {
		delay *= 2
		if delay >= cfg.MaxBackoff {
			return cfg.MaxBackoff
		}
	}
	return delay
}

// Start runs the workers until ctx ends or Stop is called.
//
// Starting is optional: a sweep that decides with no workers running starts
// them itself, because a decision has to be carried out whether or not anyone
// remembered to call this.
func (c *Cleanup) Start(ctx context.Context) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.startLocked(ctx)
}

// ensureStarted starts the dispatcher if nothing has.
func (c *Cleanup) ensureStarted() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.startLocked(context.Background())
}

func (c *Cleanup) startLocked(ctx context.Context) {
	if c.started {
		return
	}
	c.started = true
	stop := make(chan struct{})
	c.stop = stop

	c.stopped.Add(1)
	go func() {
		defer c.stopped.Done()
		for {
			select {
			case <-stop:
				return
			case <-ctx.Done():
				return
			case leaseID := <-c.queue:
				c.workers <- struct{}{}
				go func(id string) {
					defer func() { <-c.workers }()
					defer c.pending.Done()
					c.destroy(id)
				}(leaseID)
			}
		}
	}()
}

// Stop ends the workers, leaving records they had not started still charged and
// still queued for the next sweep.
func (c *Cleanup) Stop() {
	c.mu.Lock()
	stop := c.stop
	c.stop = nil
	c.started = false
	c.mu.Unlock()
	if stop == nil {
		return
	}
	close(stop)
	c.stopped.Wait()
}

// WaitIdle blocks until every destruction enqueued so far has finished.
//
// It is for a test that has stopped driving the sweep: it waits on decisions
// already made, and a sweep that enqueues more while it waits would keep it
// waiting rather than make it wrong.
func (c *Cleanup) WaitIdle() {
	c.pending.Wait()
}

// enqueue hands one decision to the workers, reporting whether it was accepted.
//
// A false answer means the queue is full: the record stays charged and is
// decided again by a later sweep, which is the correct outcome for a bound
// rather than a lost container.
func (c *Cleanup) enqueue(leaseID string) bool {
	c.ensureStarted()
	c.pending.Add(1)
	select {
	case c.queue <- leaseID:
		return true
	default:
		c.pending.Done()
		return false
	}
}

// destroy carries out one decision.
//
// The sweep moved the record to destroying when it decided, which is what keeps
// a concurrent release from issuing a second removal of the same container, and
// this is the only consumer of that decision.
func (c *Cleanup) destroy(leaseID string) {
	c.life.mu.Lock()
	held, known := c.life.allocations[leaseID]
	if !known || !held.holdsRuntime() {
		c.life.mu.Unlock()
		return
	}
	handle, hosted := held.handle, held.backend
	c.life.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), c.life.cleanupCfg.AttemptTimeout)
	err := hosted.Release(ctx, handle)
	cancel()

	if err == nil {
		c.life.finishDestroy(leaseID, true, "")
		return
	}
	c.life.finishDestroy(leaseID, false, err.Error())
}

// finishDestroy records the outcome of one attempt and, on success, returns the
// record's budget.
//
// The budget comes back only here, after a runtime confirmed the object is
// gone. That is the whole invariant: a failed removal keeps its charge, so one
// stuck container cannot become a host out of memory.
func (l *Lifecycle) finishDestroy(leaseID string, destroyed bool, failure string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	current, stillHeld := l.allocations[leaseID]
	if !stillHeld {
		return
	}
	if destroyed {
		l.clearCleanupFailureLocked(current)
		l.removeLocked(current, current.pendingReason)
		return
	}
	current.cleanupAttempts++
	current.nextCleanupAt = l.now().Add(l.cleanup.backoffAfter(current.cleanupAttempts))
	current.lastError = failure
	current.state = allocCleanupFailed
	l.failures++
	l.noteCleanupFailureLocked()
}

// noteCleanupFailureLocked marks the node as draining while it holds a container
// it could not destroy.
//
// Admitting against that memory would over-commit the host, so the node refuses
// new work until it is gone. This is the one place draining is set.
func (l *Lifecycle) noteCleanupFailureLocked() {
	if l.draining {
		return
	}
	l.draining = true
	if l.admission != nil {
		l.admission.SetDraining(true)
	}
}

// clearCleanupFailureLocked ends draining once no failed removal is left.
func (l *Lifecycle) clearCleanupFailureLocked(current *allocation) {
	if !l.draining {
		return
	}
	for _, held := range l.allocations {
		if held.state == allocCleanupFailed && held != current {
			return
		}
	}
	l.draining = false
	if l.admission != nil {
		l.admission.SetDraining(false)
	}
}
