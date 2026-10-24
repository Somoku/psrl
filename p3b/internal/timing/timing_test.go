package timing

import (
	"testing"
	"time"
)

func TestTheContractResolvesItsSpans(t *testing.T) {
	c, err := New(0, 0)
	if err != nil {
		t.Fatalf("new contract: %v", err)
	}

	for name, span := range c.Spans() {
		if span <= 0 {
			t.Fatalf("span %q resolved to %s", name, span)
		}
	}
}

func TestTheDefaultsFillInWhenADeploymentOmitsThem(t *testing.T) {
	c, err := New(0, 0)
	if err != nil {
		t.Fatalf("new contract: %v", err)
	}

	if c.NodeTTL != DefaultNodeTTL {
		t.Fatalf("node TTL is %s, want the default %s", c.NodeTTL, DefaultNodeTTL)
	}
	if c.RPCTimeout != DefaultRPCTimeout {
		t.Fatalf("RPC timeout is %s, want the default %s", c.RPCTimeout, DefaultRPCTimeout)
	}
}

func TestLivenessSpansStayOrderedForEveryNodeTTL(t *testing.T) {
	for _, ttl := range []time.Duration{
		2 * time.Minute, 10 * time.Minute, time.Hour,
	} {
		c, err := New(ttl, 0)
		if err != nil {
			t.Fatalf("node TTL %s: %v", ttl, err)
		}
		// A reservation only has to outlive the gap to the node's next report, which
		// is within one TTL; the sweep must be faster than the reservation it expires.
		if ReservationTTL >= c.NodeTTL {
			t.Fatalf("node TTL %s is not longer than the reservation TTL %s", ttl, ReservationTTL)
		}
		if SweepInterval >= ReservationTTL {
			t.Fatalf("sweep interval %s is not shorter than the reservation TTL %s", SweepInterval, ReservationTTL)
		}
		if c.LoadReportInterval()*3 >= c.NodeTTL {
			t.Fatalf("node TTL %s reports every %s, so one miss drains the node", ttl, c.LoadReportInterval())
		}
	}
}

func TestTheDefaultNodeTTLOutlivesAReservation(t *testing.T) {
	// Placement refuses a node TTL at or below the reservation TTL, so the default
	// has to satisfy that or every default deployment fails to start.
	if ReservationTTL >= DefaultNodeTTL {
		t.Fatalf("default node TTL %s is not longer than the reservation TTL %s", DefaultNodeTTL, ReservationTTL)
	}
	if SweepInterval >= ReservationTTL {
		t.Fatalf("sweep interval %s is not shorter than the reservation TTL %s", SweepInterval, ReservationTTL)
	}
}

func TestTheOwnerLeaseOutlivesAnySandbox(t *testing.T) {
	// The owner lease is a dead-owner backstop, not a sandbox lifetime, so it has to
	// exceed any lifetime a deployment states. The default lifetime is the floor.
	if OwnerLease < 4*time.Hour {
		t.Fatalf("owner lease %s is too short to cover a default-lifetime sandbox", OwnerLease)
	}
}

func TestAValidatedContractRefusesAnImpossibleTTL(t *testing.T) {
	if err := (Contract{}).Validate(); err == nil {
		t.Fatal("a contract with no TTL cannot bound anything")
	}
}
