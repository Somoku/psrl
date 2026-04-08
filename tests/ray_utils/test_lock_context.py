import asyncio
import time
from collections import deque

import ray


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
    cls.__init__ = __init__
    cls.acquire = acquire
    cls.release = release
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
# 3. 定义一个简单的 CounterActor，用来验证"读-改-写"在无锁 vs 有锁 下的差异
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
    curr = ray.get(counter.read.remote())
    time.sleep(0.1)
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
# 5. 测试：分别运行 N 个无锁任务和 N 个有锁任务，观察最终结果
# --------------------------------------------
NUM_WORKERS = 5


def test_no_lock_loses_updates(ray_cluster):
    """无锁版本，多个 worker 可能会发生读-写丢失，最终 count 应该 < NUM_WORKERS"""
    counter1 = CounterActor.remote()

    futures_no_lock = [worker_no_lock.remote(counter1, i) for i in range(NUM_WORKERS)]
    ray.get(futures_no_lock)

    final_no_lock = ray.get(counter1.read.remote())
    # With concurrent reads and sleeps, race conditions cause lost updates
    assert final_no_lock <= NUM_WORKERS, f"Expected final_no_lock <= {NUM_WORKERS}, got {final_no_lock}"


def test_lock_serializes_counter(ray_cluster):
    """有锁版本，最终 count 应该 == NUM_WORKERS"""
    counter2 = CounterActor.remote()

    futures_with_lock = [worker_with_lock.remote(counter2, i) for i in range(NUM_WORKERS)]
    ray.get(futures_with_lock)

    final_with_lock = ray.get(counter2.read.remote())
    assert final_with_lock == NUM_WORKERS, f"Expected count={NUM_WORKERS} with lock, got {final_with_lock}"
