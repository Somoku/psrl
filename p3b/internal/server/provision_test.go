package server

import (
	"errors"
	"testing"
)

// A create carries no client deadline, so a caller has to be able to ask whether
// its request reached the service and is being worked on. The ticket is its
// idempotency key, and these are the states a create moves through.
func TestAProvisionIsObservableByTicket(t *testing.T) {
	c := &Control{provisions: map[string]*provisionRecord{}}

	if _, known := c.ProvisionState("t1"); known {
		t.Fatal("an unseen ticket must read as unknown, not as a failure")
	}

	c.beginProvision("t1")
	if state, known := c.ProvisionState("t1"); !known || state.State != "provisioning" {
		t.Fatalf("a registered create must read as provisioning: %+v known=%v", state, known)
	}

	c.finishProvision("t1", nil)
	if state, _ := c.ProvisionState("t1"); state.State != "created" {
		t.Fatalf("a successful create must report created, got %q", state.State)
	}
}

func TestAFailedProvisionIsReportedAsFailed(t *testing.T) {
	c := &Control{provisions: map[string]*provisionRecord{}}

	c.beginProvision("t1")
	c.finishProvision("t1", errors.New("the class is full"))

	if state, _ := c.ProvisionState("t1"); state.State != "failed" {
		t.Fatalf("a failed create must report failed, got %q", state.State)
	}
}

func TestProvisionRecordsAgeOut(t *testing.T) {
	// A finished record only has to outlive the caller's polling, not the run.
	c := &Control{provisions: map[string]*provisionRecord{}}

	c.beginProvision("stale")
	c.finishProvision("stale", nil)
	// Backdate it past retention, then a new create prunes it.
	c.provisions["stale"].updatedAt = c.provisions["stale"].updatedAt.Add(-2 * provisionRetention)
	c.beginProvision("fresh")

	if _, known := c.ProvisionState("stale"); known {
		t.Fatal("a record past its retention must be pruned")
	}
	if _, known := c.ProvisionState("fresh"); !known {
		t.Fatal("the create that pruned must itself be recorded")
	}
}
