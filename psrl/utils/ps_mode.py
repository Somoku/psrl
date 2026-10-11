"""What each ``ps_mode`` means, in one place (items D3, T2).

The modes were tested inline as ``ps_mode == "nixl_cpu" or ps_mode == "nixl_gpu"``, a
phrase repeated twelve times across six files. That shape makes adding a mode a search
problem: `nixl_direct` was accepted by config validation and by the trainer's worker
setup, yet still rejected by the other ten guards, so the mode existed without being
usable. Nothing reported that -- each guard failed on its own terms, far from the
decision that introduced the mode.

The predicates below name the *question* each guard was actually asking. `uses_nixl` is
"does the transport go over NIXL", which is true of direct supply; `has_ps_data_plane`
is "do PS workers hold the weights", which is not. Most of the old sites meant the first
and were written as if they meant the second.

Modes:

* ``cpu`` / ``cpu_ref`` -- weights live in the PSManager actor, moved by Ray object
  transfer. No NIXL.
* ``nixl_cpu`` / ``nixl_gpu`` -- PS workers hold the weights in host or device memory and
  rollouts read them over RDMA.
* ``nixl_direct`` -- no PS data plane: rollouts read the train ranks (design 3.3). The
  meta server still runs, because the handshake and comm plan are metadata.
"""

from __future__ import annotations

#: Every accepted value, so an unknown mode fails loudly rather than silently taking a
#: default branch.
ALL_MODES: frozenset[str] = frozenset(
    {"cpu", "cpu_ref", "nixl_cpu", "nixl_gpu", "nixl_direct"}
)

#: Modes whose weight transport is NIXL RDMA.
_NIXL_MODES: frozenset[str] = frozenset({"nixl_cpu", "nixl_gpu", "nixl_direct"})

#: Modes in which PS workers hold the weights. `nixl_direct` is deliberately absent.
_PS_DATA_PLANE_MODES: frozenset[str] = frozenset({"cpu", "cpu_ref", "nixl_cpu", "nixl_gpu"})


def validate_mode(mode: str) -> str:
    """Return ``mode``, or raise if it is not a known one."""
    if mode not in ALL_MODES:
        raise ValueError(
            f"unknown ps_mode {mode!r}; expected one of {sorted(ALL_MODES)}"
        )
    return mode


def uses_nixl(mode: str) -> bool:
    """Whether weights move over NIXL rather than Ray object transfer.

    This is what most of the old `nixl_cpu or nixl_gpu` guards meant: they gate
    registration, the handshake, and the pull path, all of which direct supply also needs.
    """
    return validate_mode(mode) in _NIXL_MODES


def has_ps_data_plane(mode: str) -> bool:
    """Whether PS workers hold the weights.

    False for `nixl_direct`, which is the whole point of it. Use this -- not `uses_nixl`
    -- when the question is "is there a PS to read from or push to", because those are
    the only sites that must behave differently under direct supply.
    """
    return validate_mode(mode) in _PS_DATA_PLANE_MODES


def requires_ps_workers(mode: str) -> bool:
    """Whether PS worker actors must be created.

    Distinct from `has_ps_data_plane` only in intent: this one reads naturally at a
    worker-group construction site, and keeping the two separate means a future mode that
    stores weights somewhere other than a worker group does not have to lie about either.
    """
    return has_ps_data_plane(mode) and uses_nixl(mode)
