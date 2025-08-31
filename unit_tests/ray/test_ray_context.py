import ray

@ray.remote
class MyActor:
    def __init__(self):
        # 在 __init__ 里或任意方法里都可以调用
        self.node_id = ray.get_runtime_context().get_node_id()

    def get_node_id(self):
        # 返回16进制字符串形式的 NodeID
        return self.node_id

# 启动 Ray
ray.init()

# 创建 Actor
actor = MyActor.remote()

# 从 Driver 端获取该 Actor 的节点 ID
node_id = ray.get(actor.get_node_id.remote())
print(f"Actor 所在的节点 ID：{node_id}")
