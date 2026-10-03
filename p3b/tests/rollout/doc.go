// Package rollout contains end-to-end tests that verify the cluster-management
// claim against real sandboxd processes and a real container runtime.
//
// Run with:
//
//	go test -tags rollout ./tests/rollout/ -v -timeout 10m
//
// Requires Docker and builds the sandboxd binary from source. Set
// PSRL_ROLLOUT_IMAGE to override the default alpine:latest test image (the image
// must already be present — the test does not pull).
package rollout
