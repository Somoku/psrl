import ray
import time
import asyncio

# --------------------------------------------
# 定义一个简单的 VersionedActor，用来验证版本等待功能
# --------------------------------------------
@ray.remote
class VersionedActor:
    def __init__(self):
        # 初始化版本为 0，并准备一个 dict 存放等待者列表
        self.version = 0
        # key: 目标版本号；value: list of asyncio.Future
        self._version_waiters = {}

    async def wait_for_version(self, target_version: int):
        """
        如果当前 self.version == target_version，立即返回。
        否则，新建一个 Future 挂到 _version_waiters[target_version]，
        并 await 该 Future，直到 set_version 被调用并符合条件时唤醒。
        """
        if self.version == target_version:
            return  # 版本已匹配，直接返回

        # 否则，自行创建一个 Future，挂在对应目标版本的等待列表里
        fut = asyncio.get_event_loop().create_future()
        if target_version not in self._version_waiters:
            self._version_waiters[target_version] = []
        self._version_waiters[target_version].append(fut)
        await fut
        # 被唤醒后，就可以返回了

    def set_version(self, new_version: int):
        """
        将内部 self.version 更新为 new_version，并检查是否有 waiters
        等待该版本。如果有，把对应 Future 全部唤醒。
        """
        self.version = new_version
        # 如果有等待者列表，全部唤醒
        if new_version in self._version_waiters:
            for fut in self._version_waiters[new_version]:
                # 唤醒所有等待该版本的 Future
                if not fut.done():
                    fut.set_result(None)
            # 唤醒完成后，删除该 key
            del self._version_waiters[new_version]

    def get_version(self) -> int:
        """返回当前版本号，纯读方法。"""
        return self.version


# --------------------------------------------
# 在 Driver 里验证 wait_for_version / set_version 的行为
# --------------------------------------------
@ray.remote
def version_waiter(actor_handle, target_ver: int, worker_id: int):
    """
    调用 actor_handle.wait_for_version(target_ver)：
      - 如果版本已匹配，立刻返回
      - 否则阻塞，直到 set_version 将版本更新为 target_ver
    """
    print(f"[{time.strftime('%X')}] <-- waiter {worker_id} start waiting for version {target_ver}")
    ray.get(actor_handle.wait_for_version.remote(target_ver))
    print(f"[{time.strftime('%X')}] <-- waiter {worker_id} finish waiting for version {target_ver}")
    return f"[waiter {worker_id}] unblocked: version == {target_ver}"

@ray.remote
def version_setter(actor_handle, new_version: int, delay_s: float):
    """
    sleep 一段时间，然后调用 actor_handle.set_version(new_version)。
    """
    time.sleep(delay_s)
    ray.get(actor_handle.set_version.remote(new_version))
    return f"[setter] set version to {new_version}"


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # 创建一个 VersionedActor 实例
    versioned = VersionedActor.remote()

    # 同时启动几个 waiter，等待不同的 target_version
    waiter_futures = [
        version_waiter.remote(versioned, 1, worker_id=1),
        version_waiter.remote(versioned, 2, worker_id=2),
        version_waiter.remote(versioned, 0, worker_id=3),  # 版本 0 已默认匹配
    ]

    # 启动两个 setter：先延迟 1 秒后把版本设为 1，再延迟 2 秒后把版本设为 2
    setter_futures = [
        version_setter.remote(versioned, 1, delay_s=1.0),
        version_setter.remote(versioned, 2, delay_s=2.0),
    ]

    # 等候所有 setter 完成
    setter_results = ray.get(setter_futures)
    for res in setter_results:
        print(res)

    # 等候所有 waiter 完成
    waiter_results = ray.get(waiter_futures)
    for res in waiter_results:
        print(res)

    # 最后输出当前版本，应该等于 2
    final_version = ray.get(versioned.get_version.remote())
    print(f"Final version on actor = {final_version}")

    ray.shutdown()
