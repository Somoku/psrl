"""Resource pools with fractional GPU bundles, owned by psrl rather than by verl.

`colocate_validate_and_train` puts a train worker and a validate worker on the same
physical GPU, and it does so by asking Ray for *unequal* fractions of it -- 0.9 for train
and 0.1 for validate. The inequality is the mechanism, not a tuning choice: two bundles of
0.5 would let Ray satisfy a second train worker from the same card, which is exactly what
the split exists to prevent.

This used to live as a patch inside the vendored verl checkout, which exposed it as
`RayResourcePool(resource_num_per_bundle=...)`. Upstream verl has no such parameter -- its
`get_placement_groups` hardcodes one whole device per bundle -- so when psrl moved to a
newer verl the call site kept passing an argument the signature no longer accepted, and
the run died with a `TypeError` from `init_workers()`.

Owning it here is the fix rather than re-patching verl. The fraction is a consequence of a
psrl placement decision, so psrl is where it belongs; keeping it out of the vendored
checkout also means it survives the next verl bump instead of being a change that has to
be re-applied and silently rots when it is not.
"""

from __future__ import annotations

import logging

import ray
from ray.util.placement_group import placement_group
from verl.plugin.platform import get_platform
from verl.single_controller.ray import (
    RayResourcePool,
    RayWorkerGroup,
    SubRayResourcePool,
)
from verl.single_controller.ray.base import sort_placement_group_by_node_ip

logger = logging.getLogger(__name__)


class PSRL_RayResourcePool(RayResourcePool):
    """A `RayResourcePool` whose bundles may request a fraction of a GPU.

    Args:
        resource_num_per_bundle: GPUs requested per bundle. ``1.0`` reproduces verl's
            behaviour exactly; a smaller value is what makes colocation possible.
    """

    def __init__(self, *args, resource_num_per_bundle: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resource_num_per_bundle = resource_num_per_bundle

    def get_placement_groups(self, strategy="STRICT_PACK", name=None, device_name="cuda"):
        """As verl's, except that a bundle asks for `resource_num_per_bundle` devices.

        Reproduced rather than delegated: the fraction has to be in the bundle dict that
        `placement_group` is built from, and verl offers no seam at that point.
        """
        if self.pgs is not None:
            return self.pgs

        pg_name_prefix = name or (
            f"{self.name_prefix}verl_group_"
            f"{'_'.join([str(count) for count in self._store])}:"
        )

        current_platform = get_platform()
        if device_name != current_platform.device_name:
            logger.warning(
                f"Requested device {device_name} does not match current platform "
                f"device {current_platform.device_name}"
            )
        device_name = current_platform.ray_resource_name()

        bundle = {"CPU": self.max_colocate_count}
        if self.use_gpu:
            bundle[device_name] = self.resource_num_per_bundle
            if self.accelerator_type is not None:
                bundle[self.accelerator_type] = 1e-4
        pg_scheme = [
            [bundle.copy() for _ in range(process_count)] for process_count in self._store
        ]

        lifetime = "detached" if self.detached else None

        pgs = [
            placement_group(
                bundles=bundles,
                strategy=strategy,
                name=pg_name_prefix + str(idx),
                lifetime=lifetime,
            )
            for idx, bundles in enumerate(pg_scheme)
        ]

        ray.get([pg.ready() for pg in pgs])

        self.pgs = sort_placement_group_by_node_ip(pgs)
        return pgs


class PSRL_SubRayResourcePool(SubRayResourcePool):
    """A subpool that carries `resource_num_per_bundle` without re-deriving it.

    A subpool is handed its parent's placement groups, so it never builds bundles and the
    fraction is never applied here. It still has to be *stored*: the elastic path reads it
    back off the shared pool when constructing the next subgroup, so dropping it would
    turn a fractional parent into whole-GPU children.
    """

    def __init__(self, *args, resource_num_per_bundle: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resource_num_per_bundle = resource_num_per_bundle


class _FractionalPoolView:
    """A read-only view of a pool that reports a scaled `max_colocate_count`.

    The fraction has to be applied in two places, and verl exposes a seam for only one of
    them. Asking Ray for `GPU: 0.9` bundles is not enough -- the *worker* must also request
    0.9, or Ray rejects it with "resource request {'GPU': 1.0} cannot fit into any bundles
    for the placement group, [{'GPU': 0.9}, ...]". verl derives that request as
    ``1 / resource_pool.max_colocate_count``.

    So rather than reproduce the 75-line `_create_worker` to change one line of it, this
    reports ``max_colocate_count / fraction``, which makes verl's own arithmetic land on
    ``fraction / max_colocate_count`` -- the value the removed parameter used to produce
    directly.

    A view instead of assigning to the real pool: the attribute is read by other code for
    its true purpose (it is also the bundle's CPU count), so a temporary mutation would be
    visible to anything else touching the pool during worker creation.
    """

    __slots__ = ("_pool", "_fraction")

    def __init__(self, pool, fraction: float) -> None:
        self._pool = pool
        self._fraction = fraction

    def __getattr__(self, name):
        return getattr(self._pool, name)

    @property
    def max_colocate_count(self) -> float:
        return self._pool.max_colocate_count / self._fraction


class PSRL_RayWorkerGroup(RayWorkerGroup):
    """A worker group whose actors request the same GPU fraction as their bundle.

    Without this the pool and the workers disagree: bundles are fractional while every
    actor still asks for a whole device, so placement fails for the *first* worker and the
    error names Ray's bundle list rather than the fraction that caused it.
    """

    def _create_worker(self, rank, pg_idx, pg, local_rank, resource_pool, *args, **kwargs):
        fraction = getattr(resource_pool, "resource_num_per_bundle", 1.0)
        if fraction != 1.0:
            resource_pool = _FractionalPoolView(resource_pool, fraction)
        return super()._create_worker(
            rank, pg_idx, pg, local_rank, resource_pool, *args, **kwargs
        )
