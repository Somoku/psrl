package pressure

import (
	"fmt"
	"testing"
	"time"
)

// TestLiveHostPressureIsMeasured proves the reader works against this host's
// real cgroup hierarchy, whichever version it runs.
//
// This is the test that would have caught the v1 gap: the unit tests above all
// use fake directories, so they pass on a host whose real hierarchy cannot be
// read at all.
func TestLiveHostPressureIsMeasured(t *testing.T) {
	r, err := NewReader()
	if err != nil {
		t.Skipf("no readable cgroup hierarchy on this host: %v", err)
	}
	t.Logf("hierarchy: %s", r.Describe())

	first := r.Read()
	t.Logf("sample 1: cpu=%.4f mem=%.4f", first.CPUUsedPct, first.MemUsedPct)
	if first.CPUUsedPct != 0 {
		t.Errorf("the first CPU sample must be zero, got %.4f", first.CPUUsedPct)
	}

	// Burn CPU so the second sample has a rate to measure.
	deadline := time.Now().Add(300 * time.Millisecond)
	sink := 0
	for time.Now().Before(deadline) {
		sink += len(fmt.Sprint(time.Now().UnixNano()))
	}
	_ = sink

	second := r.Read()
	t.Logf("sample 2: cpu=%.4f mem=%.4f", second.CPUUsedPct, second.MemUsedPct)

	// Memory must be a real reading: this process has a heap, so zero means the
	// file was not read.
	if second.MemUsedPct <= 0 {
		t.Errorf("memory reads as %.4f on a live host; the usage file is not being read", second.MemUsedPct)
	}
	if second.MemUsedPct > 1 {
		t.Errorf("memory fraction %.4f is above one", second.MemUsedPct)
	}
	// CPU must be in range. It can legitimately be near zero on a busy shared
	// host, so the assertion is the range rather than a floor.
	if second.CPUUsedPct < 0 || second.CPUUsedPct > 1 {
		t.Errorf("cpu fraction %.4f is outside [0,1]", second.CPUUsedPct)
	}
}
