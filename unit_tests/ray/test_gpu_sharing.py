import time

import ray
import torch

# 初始化Ray
ray.init()

# 获取当前节点ID
current_node_id = ray.get_runtime_context().get_node_id()
print(f"Current Node ID: {current_node_id}")


@ray.remote(
    num_cpus=0,  # 共享CPU（不独占）
    num_gpus=0.5,  # 共享50%的GPU资源（小数表共享）
    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=current_node_id,  # 绑定到当前节点
        soft=False,
    ),
)
class GPUWorker:
    def __init__(self, id):
        self.id = id
        # 显存分配量（单位：字节）
        self.size = 1024 * 1024 * 1024  # 1GB

    def allocate_memory(self):
        device = torch.device("cuda")
        # 分配显存（每个元素4字节）
        _ = torch.zeros(self.size // 4, dtype=torch.float32, device=device)
        allocated = torch.cuda.memory_allocated(device)  # 当前显存占用
        return {
            "id": self.id,
            "allocated_mb": allocated / (1024 * 1024),
            "max_memory_mb": torch.cuda.get_device_properties(device).total_memory / (1024 * 1024),
        }


# 创建两个共享同一GPU的Actor
actor1 = GPUWorker.remote(id=1)
actor2 = GPUWorker.remote(id=2)

# 并行分配显存
result_ref1 = actor1.allocate_memory.remote()
result_ref2 = actor2.allocate_memory.remote()
results = ray.get([result_ref1, result_ref2])

# 打印结果
for res in results:
    print(f"Actor {res['id']}: Allocated {res['allocated_mb']:.2f} MB, GPU Total: {res['max_memory_mb']:.2f} MB")

time.sleep(100)

# 总显存占用 = Actor1占用 + Actor2占用
total_used = sum(res["allocated_mb"] for res in results)
print(f"\nTotal GPU Memory Used: {total_used:.2f} MB")
