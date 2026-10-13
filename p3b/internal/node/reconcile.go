package node

import (
	"context"
	"fmt"
	"time"

	"psrl.dev/sandboxd/internal/backend"
)

// reconciledLeasePrefix marks a lease the node did not reserve but rebuilt from
// a runtime object. It keeps those leases distinguishable from the ones
// admission handed out, which is what a post mortem needs to tell a container
// that outlived a restart from one this process created.
const reconciledLeasePrefix = "reconciled-"

// reconcileReleaseTimeout bounds destroying one container found at startup.
// A create may legitimately spend the image pull budget, but a removal is a
// local operation and has no reason to wait that long.
const reconcileReleaseTimeout = 30 * time.Second

// ReconcileReport is what one startup pass found and did.
type ReconcileReport struct {
	// Adopted are runtime objects still holding memory, now charged and owned.
	Adopted int
	// Reclaimed are runtime objects that no longer hold memory and were removed.
	Reclaimed int
	// Failures are removals the backend refused. They leave litter rather than a
	// double charge, so they do not stop the node from serving.
	Failures int
}

// Reconcile rebuilds this node's ledger from the runtime objects it finds.
//
// The daemon is the source of truth. A node that trusted only its own records
// would leak every container it forgot across a restart and, worse, would not
// know how much of its envelope was already spent, so its next admission would
// hand out memory a live container is using.
//
// A container that still holds memory is charged and owned exactly as its
// create charged it, so the sweep owns it from here. One that has exited holds
// nothing, so it is removed rather than charged.
//
// It must run before the node serves. A node that cannot enumerate its own
// runtime cannot know what it already holds, and starting to admit anyway is
// how one restart turns into a host out of memory.
func (l *Lifecycle) Reconcile(ctx context.Context, backends []backend.Backend) (ReconcileReport, error) {
	var report ReconcileReport
	for _, b := range backends {
		reconciler, can := b.(backend.Reconciler)
		if !can {
			// A backend whose objects live on an external platform has no local
			// daemon to enumerate, and nothing this node created by hand.
			continue
		}
		found, err := reconciler.Reconcile(ctx)
		if err != nil {
			return report, fmt.Errorf("enumerate %s: %w", b.Name(), err)
		}
		for _, rec := range found {
			if !rec.Running {
				releaseCtx, cancel := context.WithTimeout(ctx, reconcileReleaseTimeout)
				err := b.Release(releaseCtx, rec.Handle)
				cancel()
				if err != nil {
					report.Failures++
					continue
				}
				report.Reclaimed++
				continue
			}
			if err := l.AdoptReconciled(rec, b); err != nil {
				return report, err
			}
			report.Adopted++
		}
	}
	return report, nil
}

// AdoptReconciled records a runtime object this node found rather than created.
//
// The charge skips the headroom check, because the memory is already spent
// whether the envelope has room for it or not. A charge that doubles up on a
// device index is refused: two records claiming one GPU is a fault, and letting
// both proceed would put two sandboxes on one device that each believe they own
// it.
func (l *Lifecycle) AdoptReconciled(rec backend.Reconciled, b backend.Backend) error {
	grant := Grant{
		LeaseID:    reconciledLeasePrefix + rec.Handle.SandboxID,
		Class:      rec.Class,
		Resources:  nodeResourcesFrom(rec.Resources),
		GPUIndices: append([]int32(nil), rec.GPUIndices...),
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	if err := l.admission.Recharge(grant, rec.Owner); err != nil {
		return fmt.Errorf("charge %s from the daemon: %w", rec.Handle.SandboxID, err)
	}
	now := l.now()
	createdAt := rec.CreatedAt
	if createdAt.IsZero() {
		// A container the daemon would not date is treated as new, so its
		// lifetime is measured from this restart rather than from zero.
		createdAt = now
	}
	lifetime := rec.Lifetime
	if lifetime <= 0 {
		lifetime = DefaultLifetime
	}
	held := &allocation{
		leaseID: grant.LeaseID, sandboxID: rec.Handle.SandboxID, backend: b,
		handle: rec.Handle, class: rec.Class, owner: rec.Owner, execMode: rec.ExecMode,
		grant: grant, state: allocResident, createdAt: createdAt,
		lifetime: lifetime, lastActivity: now,
	}
	l.allocations[held.leaseID] = held
	l.bySandbox[held.sandboxID] = held
	return nil
}

// nodeResourcesFrom converts a backend's portable request into this node's
// accounting units. The two differ only in how CPU is stated: the backend
// speaks fractional cores, a node charges milliseconds.
func nodeResourcesFrom(r backend.Resources) Resources {
	return Resources{
		MemoryMB:  r.MemoryMB,
		CPUMillis: int64(r.CPUCount * 1000),
		GPUCount:  r.GPUCount,
		DiskMB:    r.DiskMB,
	}
}
