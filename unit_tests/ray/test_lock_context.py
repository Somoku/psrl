import ray
import time
import asyncio
from collections import deque

# --------------------------------------------
# 1. 定义 add_lock 装饰器，将锁逻辑注入到任意类
# --------------------------------------------
def add_lock(cls):
    """
    给被装饰类注入：
      - self._locked = False
      - self._waiters = deque()
      - async def acquire(self)
      - async def release(self)
    """
    original_init = getattr(cls, "__init__", None)

    def __init__(self, *args, **kwargs):
        # 先执行原来的 __init__（如果存在）
        if original_init is not None:
            original_init(self, *args, **kwargs)
        else:
            super(cls, self).__init__(*args, **kwargs)
        # 注入锁状态字段
        self._locked = False
        self._waiters = deque()

    async def acquire(self):
        # 如果没人占用，就直接拿到锁
        if not self._locked:
            self._locked = True
            return
        # 否则创建一个 Future，挂到队列里等待唤醒
        fut = asyncio.get_event_loop().create_future()
        self._waiters.append(fut)
        await fut
        # 被唤醒后，相当于拿到锁了

    async def release(self):
        # 如果有等待者，就把锁传给下一个
        if self._waiters:
            fut = self._waiters.popleft()
            fut.set_result(None)
        else:
            # 没人等，直接释放锁标志
            self._locked = False

    # 将新的 __init__、acquire、release 绑定到原类
    setattr(cls, "__init__", __init__)
    setattr(cls, "acquire", acquire)
    setattr(cls, "release", release)
    return cls


# --------------------------------------------
# 2. 定义 RayLock 同步上下文，用于在 Driver 里对 actor 上锁／解锁
# --------------------------------------------
class RayLock:
    def __init__(self, actor_handle):
        self._actor = actor_handle

    def __enter__(self):
        # 直接用 ray.get() 等待 acquire 完成
        ray.get(self._actor.acquire.remote())

    def __exit__(self, exc_type, exc, tb):
        ray.get(self._actor.release.remote())


# --------------------------------------------
# 3. 定义一个简单的 CounterActor，用来验证“读-改-写”在无锁 vs 有锁 下的差异
# --------------------------------------------
@ray.remote
@add_lock
class CounterActor:
    def __init__(self):
        # 外部计数器初始为 0
        self.count = 0

    def read(self) -> int:
        # 读取当前计数
        return self.count

    def write(self, value: int):
        # 将计数设为指定值
        self.count = value


# --------------------------------------------
# 4. 定义两个 worker：一个不加锁，一个加锁
# --------------------------------------------

@ray.remote
def worker_no_lock(counter: ray.actor.ActorHandle, work_id: int):
    """
    无锁版本：
      1. 先读取 count
      2. 本地 sleep 一小会儿（模拟计算/延迟），以便其他 worker 插入
      3. 再写入 count + 1
    """
    # 第一步：读当前值
    curr = ray.get(counter.read.remote())
    # 模拟一些处理时间
    time.sleep(0.1)
    # 第二步：写回 curr + 1
    ray.get(counter.write.remote(curr + 1))
    return f"worker_no_lock-{work_id} done"


@ray.remote
def worker_with_lock(counter: ray.actor.ActorHandle, work_id: int):
    """
    有锁版本：
      在读取和写入之间加 RayLock，保证不会被别的 worker 插队。
    """
    with RayLock(counter):
        curr = ray.get(counter.read.remote())
        time.sleep(0.1)
        ray.get(counter.write.remote(curr + 1))
    return f"worker_with_lock-{work_id} done"


# --------------------------------------------
# 5. 主程序：分别运行 N 个无锁任务和 N 个有锁任务，观察最终结果
# --------------------------------------------
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    NUM_WORKERS = 5

    print("=========== 无锁测试 ===========")
    # 创建一个新的 CounterActor（无锁条件下，多个 worker 可能会发生读-写丢失）
    counter1 = CounterActor.remote()

    # 并行启动 NUM_WORKERS 个无锁 worker
    futures_no_lock = [
        worker_no_lock.remote(counter1, i)
        for i in range(NUM_WORKERS)
    ]
    # 等待所有无锁 worker 完成
    results_no_lock = ray.get(futures_no_lock)
    for r in results_no_lock:
        print(r)

    # 读最终值
    final_no_lock = ray.get(counter1.read.remote())
    print(f"无锁版本，最终 count 应该 < {NUM_WORKERS}，实际 count = {final_no_lock}")
    print()

    print("=========== 有锁测试 ===========")
    # 创建一个新的 CounterActor（装饰后自带锁）
    counter2 = CounterActor.remote()

    # 并行启动 NUM_WORKERS 个有锁 worker
    futures_with_lock = [
        worker_with_lock.remote(counter2, i)
        for i in range(NUM_WORKERS)
    ]
    results_with_lock = ray.get(futures_with_lock)
    for r in results_with_lock:
        print(r)

    final_with_lock = ray.get(counter2.read.remote())
    print(f"有锁版本，最终 count 应该 == {NUM_WORKERS}，实际 count = {final_with_lock}")

    ray.shutdown()
