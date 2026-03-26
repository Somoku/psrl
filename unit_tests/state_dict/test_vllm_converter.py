"""
Example usage of vLLM to HuggingFace state dict converter.

This example shows how to convert a vLLM model's state dict to HuggingFace format
using the new class-based API.
"""

import os

from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.vllm_converter import convert_vllm_inplace


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

    model_path = f"{psrl_workspace}/models/Qwen2.5-0.5B-Instruct"
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
    for name in hf_state_dict:
        print(f"[rank{rank}] {name}: {hf_state_dict[name].shape}, {hf_state_dict[name].sum()}, {sharding[name]}")
    # torch.save(hf_state_dict, f"converted_model_rank{rank}.pth")

    dist.barrier()
    dist.destroy_process_group()


def test_new_api_no_parameter_mapping():
    """Test that SupportsWeightLayoutSpec models convert without any ParameterMapping.

    Requires: GPU, torchrun, checkpoint at $PSRL_WORKSPACE/models/Qwen2.5-0.5B-Instruct
    """
    import torch
    import torch.distributed as dist
    from vllm import LLM
    from vllm.model_executor.models.interfaces import SupportsWeightLayoutSpec
    from psrl.utils.converter.vllm_converter import convert_vllm_inplace

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    psrl_workspace = os.environ.get("PSRL_WORKSPACE", "./psrl_workspace")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    model_path = f"{psrl_workspace}/models/Qwen2.5-0.5B-Instruct"
    llm = LLM(
        model=model_path,
        tensor_parallel_size=world_size,
        distributed_executor_backend="external_launcher",
        seed=0,
    )
    vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # Verify the model implements SupportsWeightLayoutSpec
    assert isinstance(vllm_model, SupportsWeightLayoutSpec), \
        f"{type(vllm_model).__name__} must implement SupportsWeightLayoutSpec"

    # New API: no parameter_mapping needed
    hf_state_dict, sharding = convert_vllm_inplace(vllm_model, tp_rank=rank)

    # Verify output shape/completeness
    assert len(hf_state_dict) > 0, "Converted state dict must be non-empty"
    for name, tensor in hf_state_dict.items():
        assert isinstance(tensor, torch.Tensor), f"Expected tensor for {name}"
        assert name in sharding, f"Missing sharding entry for {name}"

    # Verify packed params have been decomposed
    param_names = set(hf_state_dict.keys())
    assert any("q_proj" in n for n in param_names), "q_proj must appear after QKV decomposition"
    assert any("k_proj" in n for n in param_names), "k_proj must appear after QKV decomposition"
    assert any("v_proj" in n for n in param_names), "v_proj must appear after QKV decomposition"
    assert not any("qkv_proj" in n for n in param_names), \
        "qkv_proj must NOT appear — must be split into q/k/v"
    assert any("gate_proj" in n for n in param_names), "gate_proj must appear after gate_up split"
    assert not any("gate_up_proj" in n for n in param_names), \
        "gate_up_proj must NOT appear — must be split"

    print(f"[rank{rank}] PASS: {len(hf_state_dict)} parameters converted")
    for name in sorted(hf_state_dict.keys())[:5]:
        print(f"[rank{rank}]   {name}: {hf_state_dict[name].shape}, sharding={sharding[name]}")

    dist.barrier()
    dist.destroy_process_group()


def test_spec_consistency():
    """Verify the model's spec stacked_params are consistent with its state dict keys.

    Requires: GPU, torchrun, checkpoint at $PSRL_WORKSPACE/models/Qwen2.5-0.5B-Instruct
    """
    import torch.distributed as dist
    from vllm import LLM

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    psrl_workspace = os.environ.get("PSRL_WORKSPACE", "./psrl_workspace")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    model_path = f"{psrl_workspace}/models/Qwen2.5-0.5B-Instruct"
    llm = LLM(
        model=model_path,
        tensor_parallel_size=world_size,
        distributed_executor_backend="external_launcher",
        seed=0,
    )
    vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    spec = vllm_model.get_weight_layout_spec()

    # Every packed suffix in spec must appear in at least one actual parameter name
    state_dict_keys = set(vllm_model.state_dict().keys())
    for packed_suffix, hf_suffix, shard_id in spec.stacked_params:
        found = any(packed_suffix in k for k in state_dict_keys)
        assert found, \
            f"Spec entry '{packed_suffix}' not found in model state dict — spec may be stale"

    print(f"[rank{rank}] PASS: spec consistency verified, {len(spec.stacked_params)} stacked entries")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "consistency":
        test_spec_consistency()
    else:
        test_new_api_no_parameter_mapping()
