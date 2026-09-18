import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from transformers import AutoModel
from verl.utils.fsdp_utils import get_fsdp_wrap_policy


def get_model_sharding(fsdp_model: FSDP) -> dict[str, dict]:
    """
    Print the local sharded state dictionary on rank zero.

    Args:
        fsdp_model (FSDP): Model whose sharding metadata is inspected.
    """
    with FSDP.state_dict_type(
        fsdp_model,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
    ):
        sharded_sd = fsdp_model.state_dict()

    for name, stensor in sharded_sd.items():
        if dist.get_rank() == 0:
            print(name, stensor)


def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def print_model_param_stats(model: torch.nn.Module, description: str):
    rank = dist.get_rank()
    total_params = 0
    meta_params = 0
    cpu_params = 0
    gpu_params = 0
    other_params = 0

    print(f"\n[Rank {rank}] {description}")
    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        dev_type = param.device.type
        if dev_type == "meta":
            meta_params += numel
        elif dev_type == "cpu":
            cpu_params += numel
        elif dev_type == "cuda":
            gpu_params += numel
        else:
            other_params += numel

    print(f"  • Total parameters (logical): {total_params:,d}")
    print(f"  • Meta tensor parameters: {meta_params:,d}")
    print(f"  • CPU parameters: {cpu_params:,d}")
    print(f"  • CUDA parameters: {gpu_params:,d}")
    if other_params > 0:
        print(f"  • Other device parameters: {other_params:,d}")
    # print(model)


def auto_wrap(module, recurse, nonwrapped_numel):
    print(f"Wrapping module: {module.__class__.__name__}, recurse: {recurse}, nonwrapped_numel: {nonwrapped_numel}")
    return isinstance(module, torch.nn.Linear)


def load_and_shard_model():
    """Load the model and wrap it with FSDP."""
    model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B")
    model = FSDP(
        model,
        # auto_wrap_policy=auto_wrap,
        auto_wrap_policy=get_fsdp_wrap_policy(module=model, config=None, is_lora=False),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        sync_module_states=False,
        device_mesh=init_device_mesh("cuda", mesh_shape=(2,)),
    )
    print_model_param_stats(model, "After FSDP initialization")
    get_model_sharding(model)
    return model


if __name__ == "__main__":
    import os

    setup()

    model = load_and_shard_model()

    # Training step omitted.
    # train(model, ...)

    rank = dist.get_rank()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(True, True)):
        full_state_dict = model.state_dict()
        # print(f"[Rank {rank}]: {full_state_dict}")
        if rank == 0:
            torch.save(full_state_dict, "aggregated_model.pt")
    dist.barrier()
