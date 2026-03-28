# fsdp2_demo.py

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

# 注意：FSDP2 API
from transformers import AutoConfig, AutoModelForCausalLM


def get_init_weight_context_manager(use_meta_tensor: bool = True):
    """
    返回一个 context manager，用于在 from_pretrained() 时决定：
     - rank 0：用 CPU 直接加载完整权重；
     - rank != 0：用 init_empty_weights()，创建 meta tensor（不分配内存）。
    如果 use_meta_tensor=False，则所有 rank 都用 CPU 来加载真正权重。
    """
    cpu_init_weights = lambda: torch.device("cpu")
    if use_meta_tensor:
        rank = dist.get_rank()
        if rank == 0:
            # rank 0：直接在 CPU 上分配
            return cpu_init_weights
        else:
            # 其他 rank：meta tensor
            return init_empty_weights
    else:
        return cpu_init_weights


def print_model_param_stats(model: torch.nn.Module, description: str):
    """
    遍历 model.named_parameters()，统计并打印当前 rank 上：
     - 总参数量 total_params
     - device='meta' 的参数量 meta_params
     - device='cpu' 的参数量 cpu_params
     - device.startswith('cuda') 的参数量 gpu_params（如果存在）
    """
    rank = dist.get_rank()
    total_params = 0
    meta_params = 0
    cpu_params = 0
    gpu_params = 0
    other_params = 0

    print(f"\n[Rank {rank}] —— {description} ——")
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

    print(f"  • 总参数量（logical）: {total_params:,d}")
    print(f"  • meta tensor 上参数: {meta_params:,d}")
    print(f"  • cpu device 上参数: {cpu_params:,d}")
    print(f"  • cuda device 上参数: {gpu_params:,d}")
    if other_params > 0:
        print(f"  • 其他 device 上参数: {other_params:,d}")


def fsdp2_load_full_state_dict(model: torch.nn.Module, full_state: dict, cpu_offload: CPUOffloadPolicy = None):
    """
    把 rank 0 上的「完整 full_state」广播 & 切片给每个 FSDP2 wrapper 后的模型。
    1. 在 rank 0 上，model.to(cuda:LOCAL_RANK)，其内部 FSDP FlatParameter 都变成 EmptyTensor(指向 GPU)，
       然后调用 set_model_state_dict(..., broadcast_from_rank0=True)，监督把完整权重发给每张卡。
    2. 在 rank != 0 上，先让 model.to_empty(cuda:LOCAL_RANK)，即让 FSDP FlatParameter 都是空占位，
       然后同样用 set_model_state_dict(...)，但广播来源是 rank 0，非 0 会接收对应切片。
    3. 如果 cpu_offload=True，则最后再把整个模型先送回 CPU，然后将 buffers 手动搬回本地 GPU。
    """
    rank = dist.get_rank()
    local_cuda = torch.cuda.current_device()

    # 1. 首先把 wrapper 后的 model 全部转到 GPU 并让参数成为 EmptyTensor（指向本地 GPU），
    #    这样 set_model_state_dict 才能 “in-place fill” 进去每张卡对应的切片。
    model = model.to(device=local_cuda, non_blocking=True) if rank == 0 else model.to_empty(device=local_cuda)

    """
    for name, param in model.named_parameters():
        print(f"[Rank {rank}]: before set_model_state_dict, {name}, {param}, {param.shape}")
    """

    # 2. 调用 FSDP2 的 set_model_state_dict，将 full_state_dict 切片并加载到各卡。
    cpu_offload_enabled = cpu_offload is not None
    options = StateDictOptions(full_state_dict=True, cpu_offload=cpu_offload_enabled, broadcast_from_rank0=True)
    # 内部会自动把 full_state（只有 rank 0 有真正数据）广播到其他 rank，
    # 并让每个 rank 只收到自己本地 shard。
    set_model_state_dict(model, full_state, options=options)

    # 3. buffers（如 rotary_emb）不在 state_dict 里，需要手动广播一遍：
    for _, buf in model.named_buffers():
        dist.broadcast(buf, src=0)

    # 4. 如果启用了 cpu_offload，就把模型先搬回 CPU，buffer 再搬回 GPU。
    if cpu_offload_enabled:
        model.to("cpu", non_blocking=True)
        for buf in model.buffers():
            buf.data = buf.data.to(local_cuda)


def apply_fsdp2_wrapper(model: torch.nn.Module, fsdp_config: dict, config: AutoConfig):
    """
    把原始模型的“Transformer 层”和“Embedding”分别 wrap 成 FSDP2。
    fsdp_config: dict 包含 {
        "mp_policy": MixedPrecisionPolicy(...),
        "cpu_offload": CPUOffloadPolicy(...) or None,
        "reshard_after_forward": True/False
    }
    """
    # 1. 找出需要 wrap 的子模块列表
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

    # 2. 先 wrap 各个 Transformer 层、Embedding
    for subm in modules_to_wrap:
        fully_shard(subm, **fsdp_config)

    # 3. 最后把整棵树的 root module 也 wrap 一遍
    fully_shard(model, **fsdp_config)


def main():
    # ——1. 初始化分布式
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    mesh = init_device_mesh("cuda", mesh_shape=(2,))

    if rank == 0:
        print(f"[GLOBAL] world_size = {world_size}, 使用 {world_size} 张 GPU 进行 FSDP2 演示\n")

    # ——2. 设定模型名称与 dtype
    pretrained_name = "../../models/Qwen2.5-0.5B-Instruct"
    torch_dtype = torch.float16

    # ——3. 构造 Hugging Face Config（其余 config 可自由改）
    config = AutoConfig.from_pretrained(pretrained_name)
    config.torch_dtype = torch_dtype

    # ——4. 第一阶段：Raw Model 加载（CPU + meta 占位）
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
    # 打印：第一阶段加载完后，各 rank 上的参数分布
    print_model_param_stats(model, "第一阶段：from_pretrained() 后（CPU 或 meta）")

    # ——5. 第二阶段：Apply FSDP2 Wrapper
    #    配置 MixedPrecision + CPUOffload（可根据需要调整）
    mp_policy = MixedPrecisionPolicy(param_dtype=torch_dtype, reduce_dtype=torch.float32, cast_forward_inputs=True)
    # 这里示例让 actor 不 offload，其他角色 offload，本文就统一设 None
    cpu_offload = None  # CPUOffloadPolicy(pin_memory=True)  # 若要 offload，可启用这一行

    fsdp_kwargs = {
        "mesh": mesh,
        "offload_policy": cpu_offload,
        "mp_policy": mp_policy,
        "reshard_after_forward": False,  # 只是示例，真实训练可置 True
    }
    # Wrap
    apply_fsdp2_wrapper(model, fsdp_kwargs, config)
    # wrap 过后，此时所有参数都在“EmptyTensor, device=cuda:local_rank”上（占位）
    print_model_param_stats(model, "第二阶段：FSDP2 Wrapper 之后（所有 FlatParam 都在 GPU EmptyTensor）")

    # ——6. 第三阶段：准备好 full_state_dict，然后广播&切片加载
    #    首先让 rank 0 得到“完整”CPU state_dict；其他 rank 得到 meta state_dict（不含实际数据）
    full_state = model.state_dict()  # rank 0: CPU 或者 GPU（取决于加载时机），其他 rank: meta
    # 再调用自定义的 fsdp2_load_full_state_dict
    fsdp2_load_full_state_dict(model, full_state, cpu_offload)

    # 打印：第三阶段加载完毕后，各 rank 上对应自己的切片参数分布
    print_model_param_stats(
        model,
        "第三阶段：set_model_state_dict + 切片加载完成后（各 rank 仅保留本地 shard）",
    )

    # ——7. Barrier 同步，并 exit
    dist.barrier()
    if rank == 0:
        print("\n[GLOBAL] 所有 rank 完成各阶段检查，FSDP2 load 演示结束。")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
