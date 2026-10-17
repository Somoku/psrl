package pressure

import (
	"context"
	"errors"
	"sync"
	"time"

	"psrl.dev/sandboxd/internal/node"
)

// DefaultSampleInterval is how often the sampler reads the domain.
//
// A CPU fraction is a rate between two reads, so the interval is also the window the
// rate is averaged over. Five seconds is long enough that the number describes work
// rather than scheduler noise, and short enough that a node that fills up is noticed
// before the next create.
const DefaultSampleInterval = 5 * time.Second

// staleIntervals is how many missed reads make a sample unusable. Three intervals
// tolerates one slow read without turning a healthy node into an invalid one.
const staleIntervals = 3

// errNoReader is what a sampler built without a reader reports.
var errNoReader = errors.New("pressure: a sampler needs a reader")

// Sampler reads one domain on a cadence and publishes the latest sample.
//
// The cadence matters as much as the domain. Reading on demand means the CPU rate is
// differenced over whatever interval happens to separate two callers, and callers are
// RPC handlers: two reads a microsecond apart produce a ratio of two nearly equal
// numbers, which is noise rather than load. One ticker gives every consumer the same
// measurement, and makes "how old is this?" a question with an answer.
type Sampler struct {
	reader *Reader
	every  time.Duration
	stale  time.Duration

	mu   sync.RWMutex
	last node.Pressure
}

// NewSampler returns a sampler over a reader.
func NewSampler(reader *Reader, every time.Duration) (*Sampler, error) {
	if reader == nil {
		return nil, errNoReader
	}
	if every <= 0 {
		every = DefaultSampleInterval
	}
	return &Sampler{reader: reader, every: every, stale: every * staleIntervals}, nil
}

// Start takes one sample synchronously and then samples on its own cadence until the
// context is cancelled.
//
// The first read happens here rather than on the first tick so a node that starts
// serving immediately has a memory level to decide with; the CPU rate is zero until a
// second read exists, which is the honest answer for an interval that has not elapsed.
func (s *Sampler) Start(ctx context.Context) {
	s.publish(s.reader.Read())
	go func() {
		ticker := time.NewTicker(s.every)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				s.publish(s.reader.Read())
			}
		}
	}()
}

// publish replaces the cached sample.
func (s *Sampler) publish(sample node.Pressure) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.last = sample
}

// Sample returns the latest sample, marked invalid when it is too old to describe the
// machine now.
//
// An old sample is not stale in the sense of being roughly right: a node whose sampler
// stopped has no idea what it is doing, and admission must treat that as unknown rather
// than as the last thing it happened to see.
func (s *Sampler) Sample() node.Pressure {
	s.mu.RLock()
	sample := s.last
	s.mu.RUnlock()
	if sample.Valid && time.Since(sample.SampledAt) > s.stale {
		sample.Valid = false
	}
	return sample
}

// Pressure returns the sampler as the function the node takes.
//
// It is a cache read, so the node may call it from its report path and from admission
// without changing what is measured.
func (s *Sampler) Pressure() func() node.Pressure {
	return s.Sample
}

// Interval reports the sampling cadence, for the startup log.
func (s *Sampler) Interval() time.Duration { return s.every }
