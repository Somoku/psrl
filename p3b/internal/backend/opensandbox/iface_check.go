package opensandbox

import "psrl.dev/sandboxd/internal/backend"

var (
	_ backend.Backend     = (*Backend)(nil)
	_ backend.Stateful    = (*Backend)(nil)
	_ backend.Preflighter = (*Backend)(nil)
)
