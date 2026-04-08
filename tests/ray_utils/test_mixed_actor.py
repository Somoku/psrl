# mixed_actor_example.py

import asyncio
import time

import ray


@ray.remote
class Worker:
    async def compute(self, x: int) -> int:
        """
        一个简单的异步方法：打印开始 -> sleep 2s -> 打印结束 -> 返回 x * x
        """
        print(f"[{time.strftime('%X')}] Worker.compute({x}) start")
        await asyncio.sleep(2)
        print(f"[{time.strftime('%X')}] Worker.compute({x}) end")
        return x * x


@ray.remote
class MixedActor:
    def __init__(self, worker_handle):
        """
        MixedActor 的构造函数里保存一个 Worker 的引用
        """
        self.worker = worker_handle

    async def async_method(self, x: int) -> int:
        """
        一个异步方法，内部使用 await 来获取 Worker.compute 的结果。
        由于使用 await，事件循环不会被阻塞，Ray 会在后台调度其他协程/方法。
        """
        print(f"[{time.strftime('%X')}] MixedActor.async_method({x}) start")
        result = await self.worker.compute.remote(x)
        print(f"[{time.strftime('%X')}] MixedActor.async_method({x}) got result = {result}")
        return result

    def sync_method(self, x: int) -> int:
        """
        一个同步方法，直接调用 ray.get 来阻塞地获取 Worker.compute 的结果。
        由于 MixedActor 是一个 AsyncActor（因为它有 async_method），
        这个同步方法实际上会在同一个 asyncio 事件循环线程里执行，
        ray.get 会"冻结"事件循环，直到结果返回为止。
        """
        print(f"[{time.strftime('%X')}] MixedActor.sync_method({x}) start (blocking ray.get)")
        result = ray.get(self.worker.compute.remote(x))
        print(f"[{time.strftime('%X')}] MixedActor.sync_method({x}) got result = {result}")
        return result


def test_mixed_actor_sync_method(ray_cluster):
    """测试调用 sync_method"""
    w = Worker.remote()
    a = MixedActor.remote(w)

    print(f"[{time.strftime('%X')}] --> 调用 MixedActor.sync_method(3)")
    ref_sync = a.sync_method.remote(3)
    res_sync = ray.get(ref_sync)
    print(f"[{time.strftime('%X')}] <-- MixedActor.sync_method(3) 返回：{res_sync}")
    assert res_sync == 9


def test_mixed_actor_async_method(ray_cluster):
    """测试调用 async_method"""
    w = Worker.remote()
    a = MixedActor.remote(w)

    print(f"[{time.strftime('%X')}] --> 调用 MixedActor.async_method(5)")
    ref_async = a.async_method.remote(5)
    res_async = ray.get(ref_async)
    print(f"[{time.strftime('%X')}] <-- MixedActor.async_method(5) 返回：{res_async}")
    assert res_async == 25


def test_mixed_concurrent_calls(ray_cluster):
    """混合并发调用：先后几乎同时发出多个请求，观察同步调用会"冻结"事件循环"""
    w = Worker.remote()
    a = MixedActor.remote(w)

    print(f"[{time.strftime('%X')}] --> 并发调用 MixedActor.async_method(7) 和 MixedActor.sync_method(9)")
    ref1 = a.sync_method.remote(9)
    ref2 = a.async_method.remote(7)
    res1 = ray.get(ref1)
    print(f"[{time.strftime('%X')}] <-- MixedActor.sync_method(9) 返回：{res1}")
    res2 = ray.get(ref2)
    print(f"[{time.strftime('%X')}] <-- MixedActor.async_method(7) 返回：{res2}")
    assert res1 == 81  # 9 * 9
    assert res2 == 49  # 7 * 7
