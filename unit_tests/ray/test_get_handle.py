import ray
from ray import get_runtime_context

# 定义一个简单的 Actor
@ray.remote
class MyActor:
    def __init__(self):
        self.value = 42
        
    def get_val(self):
        return self.value

    def say_hello(self):
        self.value += 1
        return "Hello from MyActor!"

    # 获取当前actor句柄的方法
    def get_current_actor_handle(self):
        return get_runtime_context().current_actor


# 启动 Ray
ray.init(ignore_reinit_error=True)

# 创建 MyActor 的实例
actor_handle = MyActor.remote()
# 随便再创建另一个
another_actor_handle = MyActor.remote()

# 获取当前 actor 的句柄，通过调用 actor 的方法获取
actor_handle_outside = ray.get(actor_handle.get_current_actor_handle.remote())

# 验证返回的句柄，并调用其方法
if actor_handle_outside:
    print("Successfully retrieved the actor handle!")
    
    # 通过返回的句柄调用actor方法
    result = ray.get(actor_handle_outside.say_hello.remote())
    print("Actor says:", result)

# 也可以直接通过 actor_handle 调用
result_direct = ray.get(actor_handle.say_hello.remote())
print("Actor says (via direct handle):", result_direct)

print(f"Actor value = {ray.get(actor_handle.get_val.remote())}")
print(f"Another actor value = {ray.get(another_actor_handle.get_val.remote())}")

# 关闭 Ray
ray.shutdown()
