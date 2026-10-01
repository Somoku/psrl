package monitor

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"psrl.dev/sandboxd/internal/placement"
)

func view(id string) placement.NodeView {
	return placement.NodeView{NodeID: id, Backend: "docker", SeenAt: time.Now()}
}

func TestAReportBecomesVisibleInTheFleet(t *testing.T) {
	m := New(2 * time.Minute)

	m.Report(view("a"))

	fleet := m.Fleet()
	if len(fleet) != 1 || fleet[0].NodeID != "a" {
		t.Fatalf("fleet is %v, want one node 'a'", fleet)
	}
}

func TestTheFleetIsOrderedSoReplicasAgree(t *testing.T) {
	// Two placement replicas reading the same version must see the same order, or
	// a tie-break is not reproducible across them.
	m := New(2 * time.Minute)
	for _, id := range []string{"c", "a", "b"} {
		m.Report(view(id))
	}

	fleet := m.Fleet()

	for i, want := range []string{"a", "b", "c"} {
		if fleet[i].NodeID != want {
			t.Fatalf("fleet[%d] is %q, want %q", i, fleet[i].NodeID, want)
		}
	}
}

func TestTheVersionChangesOnlyWhenTheFleetDoes(t *testing.T) {
	// A reader on the create path skips the pull on an unchanged version, so the
	// version must move for a report and stay put otherwise.
	m := New(2 * time.Minute)
	before := m.Version()

	m.Report(view("a"))
	after := m.Version()
	unchanged := m.Version()

	if after == before {
		t.Fatal("a report must advance the version")
	}
	if unchanged != after {
		t.Fatal("reading must not advance the version")
	}
}

func TestANodeThatSaysItIsLeavingIsDroppedImmediately(t *testing.T) {
	// Waiting for its TTL would keep sending work to a node that already said it
	// is going away.
	m := New(2 * time.Minute)
	m.Report(view("a"))

	m.Forget("a")

	if fleet := m.Fleet(); len(fleet) != 0 {
		t.Fatalf("fleet is %v, want empty", fleet)
	}
}

func TestForgettingAnUnknownNodeDoesNotDisturbTheVersion(t *testing.T) {
	m := New(2 * time.Minute)
	m.Report(view("a"))
	version := m.Version()

	m.Forget("ghost")

	if m.Version() != version {
		t.Fatal("forgetting a node that was never there must not invalidate a reader's cache")
	}
}

func TestASilentNodeCountsAsDrained(t *testing.T) {
	m := New(time.Minute)
	clock := time.Now()
	m.SetClock(func() time.Time { return clock })
	m.Report(view("a"))

	clock = clock.Add(5 * time.Minute)

	report := m.Snapshot()
	if report.LiveNodes != 0 || report.DrainedNodes != 1 {
		t.Fatalf("snapshot is %+v, want one drained node", report)
	}
}

func TestADrainingNodeIsNotCountedLive(t *testing.T) {
	m := New(time.Minute)
	draining := view("a")
	draining.Draining = true
	m.Report(draining)

	if report := m.Snapshot(); report.LiveNodes != 0 {
		t.Fatalf("a draining node is not live: %+v", report)
	}
}

func TestARepeatedReportReplacesRatherThanAccumulates(t *testing.T) {
	m := New(2 * time.Minute)
	m.Report(view("a"))
	m.Report(view("a"))

	if fleet := m.Fleet(); len(fleet) != 1 {
		t.Fatalf("fleet holds %d entries for one node", len(fleet))
	}
}

func TestConcurrentReportsAndReadsStayConsistent(t *testing.T) {
	// Nodes report from many goroutines while placement replicas read, so the
	// view a reader holds must always be a whole fleet rather than a torn one.
	m := New(2 * time.Minute)
	var wg sync.WaitGroup
	stop := make(chan struct{})

	for r := 0; r < 8; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					for _, node := range m.Fleet() {
						if node.NodeID == "" {
							panic("read a torn node view")
						}
					}
				}
			}
		}()
	}
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				m.Report(view(fmt.Sprintf("node-%d-%d", w, i%10)))
			}
		}(w)
	}

	time.Sleep(50 * time.Millisecond)
	close(stop)
	wg.Wait()

	if got := len(m.Fleet()); got != 80 {
		t.Fatalf("fleet holds %d nodes, want 80", got)
	}
}

func BenchmarkFleetRead(b *testing.B) {
	m := New(2 * time.Minute)
	for i := 0; i < 160; i++ {
		m.Report(view(fmt.Sprintf("node-%03d", i)))
	}
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if m.Version() != 0 {
				_ = m.Fleet()
			}
		}
	})
}
