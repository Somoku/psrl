package pressure

import "runtime"

// hostCores is how many cores a CPU rate is a fraction of.
//
// This is the machine's count rather than the cgroup's quota, because the
// ceilings it feeds are about the host: a node whose cgroup is quota'd to two
// cores on a 64-core machine is not under host pressure when it saturates them,
// and refusing work there would idle the machine.
func hostCores() int {
	return runtime.NumCPU()
}
