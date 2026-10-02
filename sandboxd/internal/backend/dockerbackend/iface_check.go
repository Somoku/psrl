package dockerbackend

import "psrl.dev/sandboxd/internal/backend"

var (
	_ backend.Backend       = (*Backend)(nil)
	_ backend.NodeScheduled = (*Backend)(nil)
	_ backend.Stateful      = (*Backend)(nil)
)
