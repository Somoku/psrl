"""Accelerator passthrough and per-sandbox egress policy for Docker sandboxes.

Both are node-level concerns that belong with the backend rather than with the
session: one decides what devices a container may see, and the other decides what
network it may reach.
"""

from __future__ import annotations

import logging

psrl_logger = logging.getLogger(__file__)

# Control devices a CUDA workload needs. The `--gpus` flag is avoided because it
# requires the NVIDIA container runtime, which managed node pools often lack.
NVIDIA_CONTROL_DEVICES: tuple[str, ...] = (
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
)

DEFAULT_DRIVER_LIB_GLOBS: tuple[str, ...] = (
    "/usr/lib64/libnvidia-ml.so.1",
    "/usr/lib64/libcuda.so.1",
)

DEFAULT_NVIDIA_SMI_PATH = "/usr/bin/nvidia-smi"

# The environment variable a CUDA runtime reads to pick devices. Setting it turns
# "the device is visible" into "this sandbox uses exactly what admission granted".
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
NVIDIA_VISIBLE_DEVICES = "NVIDIA_VISIBLE_DEVICES"

# Docker section key holding per-container resource limits, which is where a
# device-request tuple is expressed on the Engine API.
_RESOURCES_KEY = "DeviceRequests"


def gpu_device_arguments(
    indices: tuple[int, ...],
    *,
    driver_libs: tuple[str, ...] = DEFAULT_DRIVER_LIB_GLOBS,
    nvidia_smi_path: str = DEFAULT_NVIDIA_SMI_PATH,
    exists=None,
) -> list[dict[str, object]]:
    """
    Build the bind mounts and devices that give a container accelerator access.

    Args:
        indices (tuple[int, ...]): Device indices to expose.
        driver_libs (tuple[str, ...]): Host driver libraries to mount read only.
            A path that does not exist on this host is skipped.
        nvidia_smi_path (str): Host `nvidia-smi` path, skipped when absent.
        exists: Existence predicate, injectable so a test needs no devices.

    Returns:
        list[dict[str, object]]: Mount specifications for the container config.
    """
    if not indices:
        return []
    detector = exists or _exists
    mounts: list[dict[str, object]] = []
    for index in indices:
        mounts.append({"Type": "bind", "Source": f"/dev/nvidia{index}", "Target": f"/dev/nvidia{index}"})
    for control_device in NVIDIA_CONTROL_DEVICES:
        if detector(control_device):
            mounts.append({"Type": "bind", "Source": control_device, "Target": control_device})
        else:
            psrl_logger.warning(f"GPU control device {control_device!r} is absent, skipping passthrough.")
    for lib_path in driver_libs:
        if detector(lib_path):
            mounts.append({"Type": "bind", "Source": lib_path, "Target": lib_path, "ReadOnly": True})
    if detector(nvidia_smi_path):
        mounts.append({"Type": "bind", "Source": nvidia_smi_path, "Target": nvidia_smi_path, "ReadOnly": True})
    return mounts


def gpu_visibility_env(indices: tuple[int, ...]) -> dict[str, str]:
    """
    Return the environment that pins a workload to its granted devices.

    Without this a sandbox sees every device the container can open, which means
    two sandboxes admitted against different devices can both use the first one
    and neither matches the envelope it was charged against.
    """
    if not indices:
        return {}
    joined = ",".join(str(index) for index in indices)
    return {CUDA_VISIBLE_DEVICES: joined, NVIDIA_VISIBLE_DEVICES: joined}


def _exists(path: str) -> bool:
    import os

    return os.path.exists(path)


def resolve_device_mounts(
    indices: tuple[int, ...],
    *,
    policy_mounts: tuple[dict[str, object], ...] = (),
    existing: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    """
    Merge device mounts into a container's mount list without duplicating a path.

    A duplicated bind mount makes the Engine API reject the create, which would
    surface as an unrelated provisioning failure.

    Args:
        indices (tuple[int, ...]): Granted device indices.
        policy_mounts (tuple[dict[str, object], ...]): Device mounts from policy.
        existing (list[dict[str, object]] | None): Mounts already requested.

    Returns:
        tuple[list[dict[str, object]], list[str]]: The merged mounts and the target
            paths that were added.
    """
    mounts = list(existing or [])
    targets = {str(mount.get("Target")) for mount in mounts}
    added: list[str] = []
    for mount in (*policy_mounts, *gpu_device_arguments(indices)):
        target = str(mount.get("Target"))
        if target in targets:
            continue
        mounts.append(dict(mount))
        targets.add(target)
        added.append(target)
    return mounts, added
