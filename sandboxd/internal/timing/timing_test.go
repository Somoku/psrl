package timing

import (
	"testing"
	"time"
)

func TestOneDeclaredValueResolvesEverySpan(t *testing.T) {
	c, err := New(30*time.Minute, 0, 0, nil)
	if err != nil {
		t.Fatalf("new contract: %v", err)
	}

	for name, span := range c.Spans() {
		if span <= 0 {
			t.Fatalf("span %q resolved to %s", name, span)
		}
	}
}

// These are the same numbers psrl/sandbox/timing.py derives. While both
// implementations coexist they must agree, or a node and its control plane would
// disagree about when a sandbox is idle.
func TestTheDerivedSpansMatchThePythonContract(t *testing.T) {
	c, err := New(1800*time.Second, 120*time.Second, 60*time.Second, nil)
	if err != nil {
		t.Fatalf("new contract: %v", err)
	}

	want := map[string]time.Duration{
		"pause_window":               3600 * time.Second,
		"reap_window":                10800 * time.Second,
		"lifetime":                   43200 * time.Second,
		"acquire_timeout":            1800 * time.Second,
		"load_report_interval":       30 * time.Second,
		"monitor_pull_interval":      30 * time.Second,
		"reservation_ttl":            60 * time.Second,
		"reservation_renew_interval": 20 * time.Second,
		"capacity_lease_ttl":         3600 * time.Second,
		"owner_heartbeat_interval":   1200 * time.Second,
		"sweep_interval":             15 * time.Second,
		"lifecycle_lease_ttl":        1800 * time.Second,
		"lifecycle_gc_interval":      900 * time.Second,
	}
	got := c.Spans()
	for name, expected := range want {
		if got[name] != expected {
			t.Errorf("span %q is %s, want %s (the Python contract's value)", name, got[name], expected)
		}
	}
}

func TestTheIdleWindowsStayOrderedForEveryEpisodeLength(t *testing.T) {
	for _, deadline := range []time.Duration{
		time.Second, 10 * time.Second, time.Minute, 10 * time.Minute,
		30 * time.Minute, 2 * time.Hour, 24 * time.Hour,
	} {
		c, err := New(deadline, 0, 0, nil)
		if err != nil {
			t.Fatalf("deadline %s: %v", deadline, err)
		}
		if !(c.PauseWindow() < c.ReapWindow() && c.ReapWindow() < c.Lifetime()) {
			t.Fatalf("deadline %s produced %s/%s/%s", deadline, c.PauseWindow(), c.ReapWindow(), c.Lifetime())
		}
	}
}

func TestLivenessSpansStayOrderedForEveryNodeTTL(t *testing.T) {
	for _, ttl := range []time.Duration{
		5 * time.Second, 30 * time.Second, 2 * time.Minute, 10 * time.Minute, time.Hour,
	} {
		c, err := New(30*time.Minute, ttl, 0, nil)
		if err != nil {
			t.Fatalf("node TTL %s: %v", ttl, err)
		}
		if c.ReservationRenewInterval() >= c.ReservationTTL() || c.ReservationTTL() >= c.NodeTTL {
			t.Fatalf("node TTL %s produced renew=%s ttl=%s", ttl, c.ReservationRenewInterval(), c.ReservationTTL())
		}
		if c.LoadReportInterval()*3 >= c.NodeTTL {
			t.Fatalf("node TTL %s reports every %s, so one miss drains the node", ttl, c.LoadReportInterval())
		}
	}
}

func TestCrashRecoveryCompletesInsideTheCapacityLease(t *testing.T) {
	c, err := New(30*time.Minute, 0, 0, nil)
	if err != nil {
		t.Fatalf("new contract: %v", err)
	}

	if c.LifecycleLeaseTTL()+c.LifecycleGCInterval() >= c.CapacityLeaseTTL() {
		t.Fatalf("recovery (%s + %s) does not fit the lease (%s)",
			c.LifecycleLeaseTTL(), c.LifecycleGCInterval(), c.CapacityLeaseTTL())
	}
}

func TestAPausedSandboxKeepsTheReservationHoldingItsMemory(t *testing.T) {
	c, err := New(30*time.Minute, 0, 0, nil)
	if err != nil {
		t.Fatalf("new contract: %v", err)
	}

	if c.CapacityLeaseTTL() < c.PauseWindow() {
		t.Fatalf("lease %s does not cover the pause window %s", c.CapacityLeaseTTL(), c.PauseWindow())
	}
}

func TestAnOverrideReplacesADerivedSpan(t *testing.T) {
	c, err := New(time.Minute, 0, 0, map[string]time.Duration{
		"pause_window": 30 * time.Second,
		"reap_window":  2 * time.Minute,
	})
	if err != nil {
		t.Fatalf("new contract: %v", err)
	}

	if c.PauseWindow() != 30*time.Second {
		t.Fatalf("pause window is %s, want the override", c.PauseWindow())
	}
	// Derived from the override rather than from the deadline.
	if c.Lifetime() != 8*time.Minute {
		t.Fatalf("lifetime is %s, want 8m derived from the overridden reap window", c.Lifetime())
	}
}

func TestAnOverrideThatInvertsAnOrderingIsRefused(t *testing.T) {
	_, err := New(30*time.Minute, 0, 0, map[string]time.Duration{"reap_window": time.Minute})

	if err == nil {
		t.Fatal("a reap window before the pause window destroys before pausing")
	}
}

func TestAnEpisodeDeadlineIsRequired(t *testing.T) {
	if _, err := New(0, 0, 0, nil); err == nil {
		t.Fatal("a contract without an episode deadline cannot derive anything")
	}
}

func TestTheNodeTTLDefaultIsUsedWhenADeploymentOmitsIt(t *testing.T) {
	c, err := New(30*time.Minute, 0, 0, nil)
	if err != nil {
		t.Fatalf("new contract: %v", err)
	}

	if c.NodeTTL != DefaultNodeTTL {
		t.Fatalf("node TTL is %s, want the default %s", c.NodeTTL, DefaultNodeTTL)
	}
}

func TestTheSweepNeverFallsBelowASecond(t *testing.T) {
	// A sweep faster than a second wakes the node for nothing.
	c, err := New(time.Second, 5*time.Second, 0, nil)
	if err != nil {
		t.Fatalf("new contract: %v", err)
	}

	if c.SweepInterval() < time.Second {
		t.Fatalf("sweep interval is %s", c.SweepInterval())
	}
}
