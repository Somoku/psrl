import os

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from transformers import AutoConfig, AutoModelForCausalLM


def get_init_weight_context_manager(use_meta_tensor: bool = True):
    """
    Select rank-aware model weight initialization.

    Args:
        use_meta_tensor (bool): Whether nonzero ranks should create meta tensors.

    Returns:
        Callable: A factory for CPU or meta-device initialization.
    """
    cpu_init_weights = lambda: torch.device("cpu")
    if use_meta_tensor:
        rank = dist.get_rank()
        if rank == 0:
            return cpu_init_weights
        else:
            return init_empty_weights
    else:
        return cpu_init_weights


def print_model_param_stats(model: torch.nn.Module, description: str):
    """
    Print parameter counts by device for the current rank.

    Args:
        model (torch.nn.Module): Model whose parameters are counted.
        description (str): Label for the reported state.
    """
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


def fsdp2_load_full_state_dict(model: torch.nn.Module, full_state: dict, cpu_offload: CPUOffloadPolicy = None):
    """
    Load and shard rank zero's full state dictionary across FSDP2 ranks.

    Nonzero ranks allocate empty CUDA storage before receiving broadcast shards.
    Buffers require a separate broadcast because they are absent from the state dictionary.

    Args:
        model (torch.nn.Module): Wrapped model that receives the state.
        full_state (dict): Full state dictionary populated on rank zero.
        cpu_offload (CPUOffloadPolicy | None): Optional post-load offload policy.
    """
    rank = dist.get_rank()
    local_cuda = torch.cuda.current_device()

    model = model.to(device=local_cuda, non_blocking=True) if rank == 0 else model.to_empty(device=local_cuda)

    """
    for name, param in model.named_parameters():
        print(f"[Rank {rank}]: before set_model_state_dict, {name}, {param}, {param.shape}")
    """

    cpu_offload_enabled = cpu_offload is not None
    options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload_enabled, broadcast_from_rank0=True)
    set_model_state_dict(model, full_state, options=options)

    # Buffers are absent from the state dictionary and require a separate broadcast.
    for _, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    if cpu_offload_enabled:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(local_cuda)


def apply_fsdp2_wrapper(model: torch.nn.Module, fsdp_config: dict, config: AutoConfig):
    """
    Apply FSDP2 to transformer, embedding, and root modules.

    Args:
        model (torch.nn.Module): Model to wrap.
        fsdp_config (dict): Wrapper options with the following structure:
            {
                "mp_policy": MixedPrecisionPolicy(...),
                "cpu_offload": CPUOffloadPolicy(...) or None,
                "reshard_after_forward": True/False
            }
        config (AutoConfig): Model configuration.
    """
    default_no_split = getattr(model, "_no_split_modules", None)
    wrap_cls_names = fsdp_config.get("wrap_policy", {}).get("transformer_layer_cls_to_wrap", default_no_split)
    if isinstance(wrap_cls_names, str):
        wrap_cls_names = [wrap_cls_names]
    assert wrap_cls_names and wrap_cls_names[0] is not None
    # print(f"wrap_cls_names is {wrap_cls_names}")

    modules_to_wrap = []
    for name, subm in model.named_modules():
        if subm.__class__.__name__ in wrap_cls_names or (
            isinstance(subm, torch.nn.Embedding) and not getattr(model.config, "tie_word_embeddings", False)
        ):
            modules_to_wrap.append(subm)

    for subm in modules_to_wrap:
        fully_shard(subm, **fsdp_config)

    fully_shard(model, **fsdp_config)


def main():
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    mesh = init_device_mesh("cuda", mesh_shape=(2,))

    if rank == 0:
        print(f"[GLOBAL] Running the FSDP2 demo on {world_size} GPUs.\n")

    pretrained_name = "../../models/Qwen2.5-0.5B-Instruct"
    torch_dtype = torch.float16

    config = AutoConfig.from_pretrained(pretrained_name)
    config.torch_dtype = torch_dtype

    use_meta = True
    init_context = get_init_weight_context_manager(use_meta_tensor=use_meta)

    with init_context():
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_name,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=False,
            low_cpu_mem_usage=False,
        )
    print_model_param_stats(model, "After initialization on CPU or meta devices")

    mp_policy = MixedPrecisionPolicy(param_dtype=torch_dtype, reduce_dtype=torch.float32, cast_forward_inputs=True)
    cpu_offload = None  # CPUOffloadPolicy(pin_memory=True)

    fsdp_kwargs = {
        "mesh": mesh,
        "offload_policy": cpu_offload,
        "mp_policy": mp_policy,
        "reshard_after_forward": False,
    }
    apply_fsdp2_wrapper(model, fsdp_kwargs, config)
    print_model_param_stats(model, "After applying FSDP2 on empty CUDA tensors")

    full_state = model.state_dict()  # populated only on rank zero
    fsdp2_load_full_state_dict(model, full_state, cpu_offload)

    print_model_param_stats(
        model,
        "After loading each rank's local state-dict shard",
    )

    dist.barrier()
    if rank == 0:
        print("\n[GLOBAL] All ranks completed the FSDP2 load checks.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
