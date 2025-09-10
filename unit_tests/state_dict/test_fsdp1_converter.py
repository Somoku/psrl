"""
Example usage of vLLM to HuggingFace state dict converter.

This example shows how to convert a vLLM model's state dict to HuggingFace format
using the new class-based API.
"""
import os
import torch
import time
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._traversal_utils import _get_fsdp_handles 
from torch.distributed.fsdp.api import ShardingStrategy, StateDictType, FullStateDictConfig, ShardedStateDictConfig
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

from psrl.utils.converter import convert_fsdp_inplace

def auto_wrap(module, recurse, nonwrapped_numel):
    print(f"Wrapping module: {module.__class__.__name__}, recurse: {recurse}, nonwrapped_numel: {nonwrapped_numel}")
    return isinstance(module, torch.nn.Linear)

def example_with_real_model():
    """Example with a real vLLM model instance, now supports distributed torchrun."""
    import torch.distributed as dist
    from vllm import LLM
    
    # Read distributed info from environment variables
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    psrl_workspace = os.environ.get("PSRL_WORKSPACE", "./psrl_workspace")
    print(f"rank: {rank}, world_size: {world_size}, psrl_workspace: {psrl_workspace}")
    
    # Initialize torch distributed
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    model = AutoModelForCausalLM.from_pretrained(f"{psrl_workspace}/models/Qwen2.5-0.5B-Instruct")
    fsdp_model = FSDP(
        model,
        # auto_wrap_policy=auto_wrap,
        auto_wrap_policy=get_fsdp_wrap_policy(module=model, config=None, is_lora=False),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=rank,
        sync_module_states=False,
        device_mesh=init_device_mesh("cuda", mesh_shape=(world_size,))
    )

    # Get the model class
    model_class = type(fsdp_model)
    print(f"[rank{rank}] model_class: {model_class}")
    print(f"[rank{rank}] fsdp_model: {fsdp_model}")

    # Get the state dict
    start_time = time.time()
    with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
        fsdp_state_dict = fsdp_model.state_dict()
    end_time = time.time()
    print(f"[rank{rank}] State dict loaded in {end_time - start_time:.2f} seconds")
    for name, param in fsdp_state_dict.items():
    # for name, param in fsdp_model.named_parameters():
        print(f"[rank{rank}] {name}: {param}")

    # Convert to HuggingFace format
    hf_state_dict, sharding = convert_fsdp_inplace("fsdp", fsdp_model)

    # Save the converted state dict for each rank
    print("=" * 50)
    for name in hf_state_dict.keys():
        print(f"[rank{rank}] {name}: {hf_state_dict[name].shape}, {hf_state_dict[name].sum()}, {sharding[name]}")
    # torch.save(hf_state_dict, f"converted_model_rank{rank}.pth")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    print("Real model conversion example...")
    example_with_real_model() 