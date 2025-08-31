"""
Example usage of vLLM to HuggingFace state dict converter.

This example shows how to convert a vLLM model's state dict to HuggingFace format
using the new class-based API.
"""
import os
import torch
from torch.nn import Parameter
from typing import Dict

from psrl.utils.state_dict import convert_vllm_inplace, create_parameter_mapping

def example_with_real_model():
    """Example with a real vLLM model instance, now supports distributed torchrun."""
    import torch.distributed as dist
    from vllm import LLM
    
    # Read distributed info from environment variables
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    print(f"rank: {rank}, world_size: {world_size}")
    
    # Initialize torch distributed
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    model_path = "/jizhicfs/lhy/models/Qwen2.5-0.5B-Instruct"
    llm = LLM(
        model=model_path,
        tensor_parallel_size=world_size,
        distributed_executor_backend="external_launcher",
        seed=0,
    )
    vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    # for module_prefix, module in vllm_model.named_modules():
        # print(f"[rank{rank}] {module_prefix}: {module.__class__.__name__}, {module}")

    # Get the model class
    model_class = type(vllm_model)
    print(f"[rank{rank}] model_class: {model_class}")
    print(f"[rank{rank}] vllm_model: {vllm_model}")

    # Get the state dict
    vllm_state_dict = vllm_model.state_dict()
    for name, param in vllm_state_dict.items():
        print(f"[rank{rank}] {name}: {param.shape}")

    # Convert to HuggingFace format
    param_mapping = create_parameter_mapping(model_class, model_path)
    hf_state_dict, sharding = convert_vllm_inplace(param_mapping, vllm_model, tp_rank=rank)

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