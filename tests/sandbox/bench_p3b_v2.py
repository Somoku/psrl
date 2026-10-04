"""Concurrency benchmark v2: two-node fleet, 多倍并发规模.

节点 250 和节点 97 各自运行一个独立的 sandboxd 服务，两者是独立的单节点服务而非
联邦 fleet。本 benchmark 的两节点模式是：同时向两个节点发出请求，模拟两个 worker
组同时跑 rollout 的场景，统计合并后的吞吐量和延迟分布。

测试分三个阶段：
  1. 单节点预热 — 确认服务和 alpine 镜像均已就绪
  2. 单节点扩展测试 — 8 / 16 / 32 / 64 / 128 并发
  3. 双节点联合测试 — 两节点各承担 N/2，总并发 N = 32 / 64 / 128 / 256

每个 sandbox：256 MB，1 vCPU，资源类 rollout，执行一条 echo 命令后释放。
"""
import asyncio
import statistics
import sys
import time
from typing import NamedTuple

from sandboxd import Resources, SandboxClient, SandboxSpec, Source

NODE_250 = "unix:///run/sandboxd.sock"        # local; run this script on node 250
NODE_97_HOST = "28.49.198.97"                  # reached via TCP when testing remotely
NODE_97_TCP = "28.49.198.97:7777"              # TCP listener on node 97 (added below)

IMAGE = "alpine:latest"
MEM_MB = 256
CPU = 1.0
CONCURRENCY_SINGLE = [8, 16, 32, 64, 128]
CONCURRENCY_DUAL   = [32, 64, 128, 256]


class Result(NamedTuple):
    n_requested: int
    n_ok: int
    n_err: int
    elapsed_s: float
    create_times: list[float]
    errors: list[str]

    @property
    def throughput(self):
        return round(self.n_ok / self.elapsed_s, 2) if self.elapsed_s else 0

    def pct(self, p):
        if not self.create_times:
            return 0
        idx = int(len(self.create_times) * p)
        return round(sorted(self.create_times)[min(idx, len(self.create_times)-1)], 0)


async def one_sandbox(client: SandboxClient, label: str) -> float:
    t0 = time.perf_counter()
    spec = SandboxSpec(
        source=Source.image(IMAGE),
        resources=Resources(memory_mb=MEM_MB, cpu_count=CPU),
        resource_class="rollout",
    )
    async with await client.create(spec) as sb:
        ms = (time.perf_counter() - t0) * 1000
        result = await sb.exec("echo bench-ok")
        assert result.exit_code == 0
    return ms


async def run_burst(endpoint: str, n: int) -> Result:
    client = SandboxClient(endpoint)
    t_wall = time.perf_counter()
    raw = await asyncio.gather(
        *[one_sandbox(client, f"sb-{i}") for i in range(n)],
        return_exceptions=True
    )
    elapsed = time.perf_counter() - t_wall
    await client.close()

    ok = [r for r in raw if isinstance(r, float)]
    errs = [str(r) for r in raw if not isinstance(r, float)]
    return Result(n, len(ok), len(errs), round(elapsed, 2), ok, errs)


async def run_dual(ep1: str, ep2: str, total_n: int) -> Result:
    """Split total_n sandboxes evenly between two endpoints, gather concurrently."""
    half = total_n // 2
    rest = total_n - half
    t_wall = time.perf_counter()
    (r1, r2) = await asyncio.gather(run_burst(ep1, half), run_burst(ep2, rest))
    elapsed = time.perf_counter() - t_wall
    all_times = r1.create_times + r2.create_times
    all_errs  = r1.errors + r2.errors
    return Result(total_n, r1.n_ok + r2.n_ok, r1.n_err + r2.n_err,
                  round(elapsed, 2), all_times, all_errs)


def print_row(label, r: Result):
    tag = "OK" if r.n_err == 0 else f"{r.n_err} ERR"
    print(f"{label:>20}  {r.n_ok:>4}/{r.n_requested:<4} {r.elapsed_s:>8.1f}s "
          f"{r.throughput:>10.2f}/s  {r.pct(0.5):>8.0f}ms  {r.pct(0.95):>9.0f}ms  [{tag}]")


async def main():
    # ── Phase 0: warm up node 250 ──────────────────────────────────────────────
    print("\n=== Phase 0: warmup ===")
    w = await run_burst(NODE_250, 4)
    print(f"warmup: {w.n_ok}/4 ok, p50={w.pct(0.5):.0f}ms")

    # ── Phase 1: single-node scaling ──────────────────────────────────────────
    print(f"\n=== Phase 1: single node (node-250, {IMAGE}, {MEM_MB} MB/sandbox) ===")
    print(f"{'concurrency':>20}  {'OK/N':>9} {'elapsed':>9} {'throughput':>11}  {'p50':>9}  {'p95':>10}  notes")
    print("-" * 85)
    for n in CONCURRENCY_SINGLE:
        r = await run_burst(NODE_250, n)
        print_row(f"n={n}", r)
        if r.errors:
            print(f"  errors: {r.errors[:3]}")

    # ── Phase 2: dual-node scaling ─────────────────────────────────────────────
    # The node-97 sandboxd listens on the unix socket locally; to reach it from
    # node-250 we need a TCP forward. Check if one is running.
    print(f"\n=== Phase 2: dual-node (node-250 + node-97 via TCP, {IMAGE}, {MEM_MB} MB/sandbox) ===")
    import socket as _socket
    try:
        s = _socket.create_connection((NODE_97_HOST, 7777), timeout=3)
        s.close()
        ep2 = NODE_97_TCP
        print(f"node-97 TCP forward reachable at {NODE_97_TCP}")
    except Exception as e:
        print(f"node-97 TCP forward not reachable ({e})")
        print("Skipping dual-node phase. Start a TCP forward on node-97 with:")
        print("  socat TCP-LISTEN:7777,fork,reuseaddr UNIX-CONNECT:/run/sandboxd.sock &")
        print(f"{'concurrency':>20}  {'OK/N':>9} {'elapsed':>9} {'throughput':>11}  {'p50':>9}  {'p95':>10}  notes")
        return

    print(f"{'concurrency':>20}  {'OK/N':>9} {'elapsed':>9} {'throughput':>11}  {'p50':>9}  {'p95':>10}  notes")
    print("-" * 85)
    for n in CONCURRENCY_DUAL:
        r = await run_dual(NODE_250, ep2, n)
        print_row(f"n={n} (2×{n//2})", r)
        if r.errors:
            print(f"  errors: {r.errors[:3]}")

    print("\nNote: dual-node throughput is the combined rate; p50/p95 are over all sandboxes.")
    print("The two nodes run independent fleet ledgers (not a single federated fleet).")


asyncio.run(main())
