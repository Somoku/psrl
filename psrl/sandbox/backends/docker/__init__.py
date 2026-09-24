"""The Docker backend, one module per concern.

`_target_` in the configuration names this package, so `DockerBackend` is re-exported
here. Everything else is reached through the module that owns it.

| Module | Owns |
|---|---|
| `backend` | The `SandboxBackend` itself: create, prepare, capabilities, and the container config |
| `session` | One container: command lifecycle, stop detection, exit classification, destruction |
| `engine` | Two bounded HTTP pools, the Docker protocol, frame decoding, archive streaming |
| `exec` | The command strategies: one shot, and the persistent shell with its sentinel |
| `events` | One shared container event stream, with reconnect gap recovery |
| `lifecycle` | Worker heartbeat, node collector supervision, startup reclamation |
| `policy` | Typed security, workload policy, and disk admission configuration |
| `devices` | GPU passthrough without the NVIDIA toolkit, and the visibility environment |
| `egress` | The per-sandbox egress allowlist, programmed in the host firewall |
| `pool` | Prepared containers held off the critical path |
| `cli` | Docker CLI housekeeping: force removal, label sweeps, image pruning |
| `text` | Observation truncation that keeps both ends of a long output |
"""

from psrl.sandbox.backends.docker.backend import DockerBackend

__all__ = ["DockerBackend"]
