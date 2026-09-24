"""Accelerator passthrough, which is what makes a device grant real.

A sandbox that can open a device it was not granted uses capacity the node never
charged for, and two sandboxes can then fight over one card while the envelope
reports both as accounted.
"""

from __future__ import annotations

import pytest
from psrl.sandbox.backends.docker.devices import (
    NVIDIA_CONTROL_DEVICES,
    gpu_device_arguments,
    gpu_visibility_env,
    resolve_device_mounts,
)

pytestmark = pytest.mark.cpu_test


def _present(*paths: str):
    return lambda path: path in paths


def test_no_devices_produces_no_mounts_and_no_visibility() -> None:
    assert gpu_device_arguments(()) == []
    assert gpu_visibility_env(()) == {}


def test_each_granted_device_is_bound_by_index() -> None:
    mounts = gpu_device_arguments((2,), exists=_present("/dev/nvidia2", *NVIDIA_CONTROL_DEVICES))

    targets = [mount["Target"] for mount in mounts]
    assert "/dev/nvidia2" in targets
    assert "/dev/nvidia0" not in targets


def test_the_control_devices_a_cuda_workload_needs_are_included() -> None:
    mounts = gpu_device_arguments((0,), exists=_present("/dev/nvidia0", *NVIDIA_CONTROL_DEVICES))

    targets = {mount["Target"] for mount in mounts}
    assert set(NVIDIA_CONTROL_DEVICES) <= targets


def test_an_absent_control_device_is_skipped_rather_than_faked() -> None:
    # Binds have to name a path that exists, or the daemon rejects the create.
    mounts = gpu_device_arguments((0,), exists=_present("/dev/nvidia0"))
    targets = {mount["Target"] for mount in mounts}

    assert targets == {"/dev/nvidia0"}


def test_the_driver_libraries_are_mounted_read_only() -> None:
    mounts = gpu_device_arguments(
        (0,),
        driver_libs=("/usr/lib64/libcuda.so.1",),
        exists=_present("/dev/nvidia0", "/usr/lib64/libcuda.so.1"),
    )

    mounted = {mount["Target"]: mount for mount in mounts}
    assert mounted["/usr/lib64/libcuda.so.1"]["ReadOnly"] is True


def test_the_visibility_variable_pins_the_workload_to_its_grant() -> None:
    env = gpu_visibility_env((0, 3))

    assert env["CUDA_VISIBLE_DEVICES"] == "0,3"
    assert env["NVIDIA_VISIBLE_DEVICES"] == "0,3"


def test_a_mount_already_requested_is_not_duplicated() -> None:
    # A duplicated bind mount makes the daemon reject the create, which surfaces as
    # an unrelated provisioning failure.
    existing = [{"Type": "bind", "Source": "/dev/nvidia0", "Target": "/dev/nvidia0"}]

    mounts, added = resolve_device_mounts((0,), existing=existing)

    assert added == []
    assert len(mounts) == 1


def test_devices_are_added_alongside_an_ordinary_bind_mount() -> None:
    existing = [{"Type": "bind", "Source": "/tmp/x", "Target": "/work"}]

    mounts, added = resolve_device_mounts((1,), existing=existing)

    assert "/dev/nvidia1" in added
    assert {mount["Target"] for mount in mounts} >= {"/work", "/dev/nvidia1"}
