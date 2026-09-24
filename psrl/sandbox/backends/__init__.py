"""The sandbox backends, one module or package per provider.

`_target_` in the configuration names a backend class through this module, so every
backend is re-exported here. A backend's own modules stay behind its own package, and the
config dataclasses are reached through the package that owns them.
"""

from psrl.sandbox.backends.docker import DockerBackend
from psrl.sandbox.backends.e2b import AgentEnvBackend, CubeSandboxBackend, E2BBackend
from psrl.sandbox.backends.opensandbox import OpenSandboxBackend

__all__ = [
    "AgentEnvBackend",
    "CubeSandboxBackend",
    "DockerBackend",
    "E2BBackend",
    "OpenSandboxBackend",
]
