"""Where the time in a stage-boundary offload goes, and how it scales to a full node.

A single-process run of `test_stage_offload_gpu.py` measured D2H 50.9 GB/s but H2D only
36.7 GB/s. The restore direction also reallocates, so this separates three things:

* the device reallocation itself (`resize_` -> `cudaMalloc` after `empty_cache`);
* the copy, with the pinned buffer on the GPU's own NUMA node versus the other one;
* eight ranks switching at once, which is what production does -- every rank on a node
  crosses the stage boundary together, so the figure that matters is per-rank bandwidth
  under contention for host memory and the PCIe root complexes.

    python tests/nixl/bench_stage_offload.py --gib 8 --procs 8

Pinned buffers are touched by a thread bound to the target NUMA node, so first-touch
places them there; `numactl` is not assumed to be installed.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.multiprocessing as mp

GiB = 1 << 30


def _numa_cpus(node: int) -> set[int]:
    path = f"/sys/devices/system/node/node{node}/cpulist"
    cpus: set[int] = set()
    with open(path) as f:
        for part in f.read().strip().split(","):
            lo, _, hi = part.partition("-")
            cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def _gpu_numa(device: int) -> int:
    props = torch.cuda.get_device_properties(device)
    bus = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    try:
        with open(f"/sys/bus/pci/devices/{bus}/numa_node") as f:
            return max(0, int(f.read().strip()))
    except OSError:
        return 0


def _run(device: int, gib: int, numa: str, out, barrier) -> None:
    torch.cuda.set_device(device)
    gpu_node = _gpu_numa(device)
    node = gpu_node if numa == "local" else 1 - gpu_node
    os.sched_setaffinity(0, _numa_cpus(node))

    nbytes = gib * GiB
    gpu = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    gpu.random_(0, 255)
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    host.zero_()  # first touch from the bound CPUs
    storage = gpu.untyped_storage()
    torch.cuda.synchronize()

    def timed(fn):
        barrier.wait()
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - start

    # Warm once so the first-copy driver setup is not in the figures.
    host.copy_(gpu, non_blocking=True)
    gpu.copy_(host, non_blocking=True)

    d2h = timed(lambda: host.copy_(gpu, non_blocking=True))
    h2d = timed(lambda: gpu.copy_(host, non_blocking=True))

    storage.resize_(0)
    torch.cuda.empty_cache()
    realloc = timed(lambda: storage.resize_(nbytes))
    # Reallocation without empty_cache: what a caching allocator that kept the block pays.
    storage.resize_(0)
    realloc_cached = timed(lambda: storage.resize_(nbytes))

    out.put(
        {
            "device": device,
            "gpu_numa": gpu_node,
            "host_numa": node,
            "d2h_gbps": nbytes / d2h / 1e9,
            "h2d_gbps": nbytes / h2d / 1e9,
            "realloc_s": realloc,
            "realloc_cached_s": realloc_cached,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gib", type=int, default=8)
    parser.add_argument("--procs", type=int, default=8)
    args = parser.parse_args()

    ctx = mp.get_context("spawn")
    results = {}
    for numa in ("local", "remote"):
        for procs in sorted({1, args.procs}):
            out, barrier = ctx.Queue(), ctx.Barrier(procs)
            workers = [ctx.Process(target=_run, args=(d, args.gib, numa, out, barrier)) for d in range(procs)]
            for w in workers:
                w.start()
            rows = [out.get(timeout=600) for _ in workers]
            for w in workers:
                w.join()
            rows.sort(key=lambda r: r["device"])
            label = f"{numa}_numa_{procs}proc"
            results[label] = rows
            d2h = [r["d2h_gbps"] for r in rows]
            h2d = [r["h2d_gbps"] for r in rows]
            print(
                f"{label:22s} per-rank D2H min/avg {min(d2h):5.1f}/{sum(d2h) / len(d2h):5.1f} GB/s  "
                f"H2D min/avg {min(h2d):5.1f}/{sum(h2d) / len(h2d):5.1f} GB/s  "
                f"node D2H {sum(d2h):6.1f}  H2D {sum(h2d):6.1f}  "
                f"realloc {max(r['realloc_s'] for r in rows) * 1e3:6.1f} ms "
                f"(cached {max(r['realloc_cached_s'] for r in rows) * 1e3:5.2f} ms)",
                flush=True,
            )
    out_path = os.path.join(os.path.dirname(__file__), "bench_stage_offload.result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
