"""Save and load per-rank Megatron checkpoints with plain tensors.

Sharded wrappers are replaced with payloads before `torch.save` to avoid DCP
metadata gathering. The saved topology, optimizer mode, and architecture must
match between save and load.
"""

import json
import logging
import os

import torch
from megatron.core import mpu
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedBase,
)

logger = logging.getLogger(__name__)

_METADATA_FILE = "parallel_config.json"
_METADATA_VERSION = 2
_CHECKPOINT_FORMAT = "per_rank_plain_tensors"


def _parallel_world_size(getter_name: str) -> int:
    """Read one parallel-world-size getter from `mpu`, defaulting to 1 when absent."""
    getter = getattr(mpu, getter_name, None)
    if getter is None:
        return 1
    value = getter()
    return 1 if value is None else int(value)


def _build_metadata(extra_metadata: dict | None = None) -> dict:
    """Build the versioned checkpoint metadata, including the current topology."""
    metadata = {
        "metadata_version": _METADATA_VERSION,
        "format": _CHECKPOINT_FORMAT,
        "topology": {
            "world_size": torch.distributed.get_world_size(),
            "tp_size": _parallel_world_size("get_tensor_model_parallel_world_size"),
            "pp_size": _parallel_world_size("get_pipeline_model_parallel_world_size"),
            "vpp_size": _parallel_world_size("get_virtual_pipeline_model_parallel_world_size"),
            "cp_size": _parallel_world_size("get_context_parallel_world_size"),
            "ep_size": _parallel_world_size("get_expert_model_parallel_world_size"),
            "etp_size": _parallel_world_size("get_expert_tensor_parallel_world_size"),
            "dp_size": _parallel_world_size("get_data_parallel_world_size"),
        },
    }
    if extra_metadata:
        for key in ("optimizer_mode", "architecture"):
            if key in extra_metadata:
                metadata[key] = extra_metadata[key]
    return metadata


def _validate_metadata(saved: dict, expected: dict) -> None:
    """Fail if the saved checkpoint does not match the running topology and model."""
    if saved.get("metadata_version") != _METADATA_VERSION:
        raise ValueError(
            f"Unsupported per-rank checkpoint metadata version {saved.get('metadata_version')!r}, "
            f"expected {_METADATA_VERSION}."
        )
    if saved.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(
            f"Unsupported per-rank checkpoint format {saved.get('format')!r}, expected {_CHECKPOINT_FORMAT!r}."
        )

    mismatches = []
    saved_topology = saved.get("topology", {})
    for key, current_value in expected["topology"].items():
        saved_value = saved_topology.get(key)
        if saved_value != current_value:
            mismatches.append(f"{key}: saved={saved_value!r}, current={current_value!r}")
    for key in ("optimizer_mode", "architecture"):
        current_value = expected.get(key)
        saved_value = saved.get(key)
        if current_value is not None and saved_value != current_value:
            mismatches.append(f"{key}: saved={saved_value!r}, current={current_value!r}")
    if mismatches:
        raise ValueError(
            "Per-rank checkpoints require identical topology and optimizer/model configuration. "
            f"{', '.join(mismatches)}."
        )


def _assert_no_sharded_objects(obj, _path="root"):
    """Recursively assert that *obj* contains no Megatron checkpoint wrappers.

    Raises ``AssertionError`` with a descriptive path on the first violation.
    Call after ``_extract_plain_state_dict`` (save path) and after
    ``_unwrap_sharded_state_dict`` (load path) to catch any missed wrappers.
    """
    if isinstance(obj, (ShardedBase, LocalNonpersistentObject)):
        raise AssertionError(
            f"Unexpected sharded wrapper at {_path!r}: {type(obj).__name__}. "
            f"All wrappers should have been extracted before this point."
        )
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_sharded_objects(v, f"{_path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_no_sharded_objects(v, f"{_path}[{i}]")


def _extract_plain_state_dict(obj):
    """Replace Megatron checkpoint wrappers with their payloads before saving."""
    if isinstance(obj, dict):
        return {k: _extract_plain_state_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_extract_plain_state_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_extract_plain_state_dict(v) for v in obj)
    # `ShardedTensorFactory` subclasses `ShardedBase`, so this branch handles both.
    if isinstance(obj, ShardedBase):
        return obj.data
    if isinstance(obj, LocalNonpersistentObject):
        return obj.obj
    return obj


def _unwrap_sharded_state_dict(obj):
    """Unwrap Megatron checkpoint wrappers from a loaded state dict.

    Lists of sharded tensors are joined defensively for checkpoints whose tensor payload was
    migrated from the old factory-expanded representation.
    """
    if isinstance(obj, dict):
        return {k: _unwrap_sharded_state_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        # Dense factory outputs use shard lists that merge with `torch.cat`.
        if obj and all(isinstance(x, ShardedBase) for x in obj):
            return torch.cat([x.data for x in obj])
        return [_unwrap_sharded_state_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_unwrap_sharded_state_dict(v) for v in obj)
    if isinstance(obj, ShardedBase):
        return obj.data
    if isinstance(obj, LocalNonpersistentObject):
        return obj.obj
    return obj


def save_megatron_checkpoint(
    sharded_state_dict,
    ckpt_path,
    async_save: bool = False,
    metadata: dict | None = None,
):
    """Save a Megatron sharded state dict using per-rank ``torch.save``.

    Args:
        sharded_state_dict: Megatron sharded state dict (may contain
            ``ShardedTensorFactory``, ``ShardedTensor``, ``ShardedObject``,
            ``LocalNonpersistentObject``).
        ckpt_path (str): Directory to save checkpoint files into.
        async_save (bool): Must be False because per-rank saves are synchronous.
        metadata (dict | None): Optimizer mode and architecture supplied by the checkpoint manager.
    """
    if async_save:
        raise ValueError("Per-rank Megatron checkpoints do not support async_save.")

    plain_state_dict = _extract_plain_state_dict(sharded_state_dict)
    _assert_no_sharded_objects(plain_state_dict)

    os.makedirs(ckpt_path, exist_ok=True)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    save_path = os.path.join(ckpt_path, f"rank_{rank}.pt")
    torch.save(plain_state_dict, save_path)
    assert os.path.exists(save_path), f"torch.save appeared to succeed, but path={save_path!r} was not found on disk."

    if rank == 0:
        checkpoint_metadata = _build_metadata(metadata)
        if checkpoint_metadata["topology"]["world_size"] != world_size:
            raise RuntimeError("Distributed world size changed while saving a per-rank checkpoint.")
        with open(os.path.join(ckpt_path, _METADATA_FILE), "w") as f:
            json.dump(checkpoint_metadata, f, indent=2)

    torch.distributed.barrier()
    logger.info("[Rank %d] Saved per-rank checkpoint to %s", rank, save_path)
    return None


def load_megatron_checkpoint(
    sharded_state_dict,
    ckpt_dir,
    expected_metadata: dict | None = None,
):  # noqa: ARG001
    """Load a per-rank Megatron checkpoint from *ckpt_dir*.

    Validates the metadata stored in ``parallel_config.json`` and asserts that the loaded state
    dict contains no residual Megatron wrappers.

    Args:
        sharded_state_dict: Unused and retained for API compatibility with
            ``load_dist_checkpointing``.
        ckpt_dir (str): Directory containing ``rank_<N>.pt`` files and
            ``parallel_config.json``.
        expected_metadata (dict | None): Optimizer mode and architecture expected by the manager.

    Returns:
        State dict with plain tensors and objects (no Megatron wrappers).
    """
    rank = torch.distributed.get_rank()
    rank_path = os.path.join(ckpt_dir, f"rank_{rank}.pt")

    if not os.path.exists(rank_path):
        raise FileNotFoundError(
            f"Per-rank checkpoint not found at {rank_path!r}. "
            "Only the per-rank format saved by save_megatron_checkpoint is supported."
        )

    metadata_path = os.path.join(ckpt_dir, _METADATA_FILE)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path!r}. Checkpoint directory may be corrupt or incomplete."
        )

    with open(metadata_path) as f:
        metadata = json.load(f)

    _validate_metadata(metadata, _build_metadata(expected_metadata))

    raw_state_dict = torch.load(rank_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_sharded_state_dict(raw_state_dict)
    _assert_no_sharded_objects(state_dict)

    logger.info("[Rank %d] Loaded per-rank checkpoint from %s", rank, rank_path)
    return state_dict
