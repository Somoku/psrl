package server

import (
	"testing"

	"psrl.dev/sandboxd/internal/backend"
)

// TestOwnerOfNamesTheCallerTheReclaimerKeyedOn pins the identity a node reclaims
// a sandbox by.
//
// Getting this wrong is not a cosmetic bug: ownership is what a silence budget
// measures, so an owner nobody renews reads as a dead caller and every sandbox
// it holds is reclaimed while it is still working. That happened once, because
// ownership was read from `workflow_id` — a phase-reservation key the caller
// usually leaves unset — while the caller stated its owner in metadata.
func TestOwnerOfNamesTheCallerTheReclaimerKeyedOn(t *testing.T) {
	tests := []struct {
		name string
		spec backend.Spec
		want string
	}{
		{
			name: "the owner the caller states",
			spec: backend.Spec{Metadata: map[string]string{OwnerMetadataKey: "w0-host-1-abcd"}},
			want: "w0-host-1-abcd",
		},
		{
			name: "the stated owner wins over a workflow",
			spec: backend.Spec{
				WorkflowID: "phase-1",
				Metadata:   map[string]string{OwnerMetadataKey: "w0-host-1-abcd"},
			},
			want: "w0-host-1-abcd",
		},
		{
			name: "a workflow stands in when no owner is stated",
			spec: backend.Spec{WorkflowID: "phase-1"},
			want: "phase-1",
		},
		{
			name: "neither stated",
			spec: backend.Spec{},
			want: "anonymous",
		},
		{
			name: "an empty owner is not a name",
			spec: backend.Spec{Metadata: map[string]string{OwnerMetadataKey: ""}},
			want: "anonymous",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := ownerOf(test.spec); got != test.want {
				t.Errorf("ownerOf(%+v) = %q, want %q", test.spec, got, test.want)
			}
		})
	}
}
