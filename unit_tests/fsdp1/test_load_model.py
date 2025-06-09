import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, StateDictType, FullStateDictConfig
from transformers import AutoModel, AutoConfig
import os

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

    
def auto_wrap(module, recurse, nonwrapped_numel):
    print(f"Wrapping module: {module.__class__.__name__}, recurse: {recurse}, nonwrapped_numel: {nonwrapped_numel}")
    return isinstance(module, torch.nn.Linear)

def load_and_shard_model():
    """加载并分片模型"""
    # 创建FSDP包裹的模型
    model = FSDP(
        AutoModel.from_pretrained("bert-base-uncased"),
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        sync_module_states=False
    )
    print_model_param_stats(model, "FSDP初始化后")  # 阶段1: meta设备初始化 [[4]]

    return model

if __name__ == "__main__":
    import os
    setup()
    
    # 加载并分片模型
    model = load_and_shard_model()
    
    # 训练代码（此处省略）
    # train(model, ...)
    
    # 保存聚合后的模型
    rank = dist.get_rank()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(True, True)):
        full_state_dict = model.state_dict()
        print(f"[Rank {rank}]: {full_state_dict}")
        if rank == 0:
            torch.save(full_state_dict, "aggregated_model.pt")
    dist.barrier()
