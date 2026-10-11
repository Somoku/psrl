"""Is a local strided scatter cheap enough to replace strided RDMA? (plan item S0)

This is the gate on the whole P2 plan (`docs/design/descriptor_cost_plan.md`). P2 trades
descriptors for a local copy: instead of transferring a shard as 4096 strided descriptors,
transfer it as one contiguous block into a landing buffer and scatter it locally into the
strided parameter. Measured descriptor cost is ~2.1-4us each, so the 4096-row shards on
Qwen3-8B cost ~1.24s per client -- against which a local copy of the same bytes should be
single-digit milliseconds.

**Should be.** The write pattern is 4096 rows of ~2KB into a strided destination, which is
not what HBM is good at, and a peak-bandwidth estimate is exactly the kind of reasoning
that has been wrong three times on this problem. So this measures it on one GPU in
minutes, before any of P2 is written.

The comparison that matters is scatter time against *descriptor* time, not against the
transfer: the transfer is already nearly free over NVLink (measured wait_peer = 0.008s).

Run:

    python tests/nixl/bench_strided_scatter.py

Writes `bench_strided_scatter.result.json` next to this file.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

#: Measured cost of one NIXL transfer descriptor, from `bench_cross_node_bw.py --tensors`
#: (0.026ms at 1 descriptor vs 16.942ms at 4096, at a fixed 2 GiB payload).
US_PER_DESCRIPTOR = 2.1

#: Qwen3-8B row-parallel shapes. `o_proj` is [hidden, hidden]; `down_proj` is
#: [hidden, intermediate]. Both are sharded on the *input* (inner) dimension, which is
#: what turns them strided -- the row count is the untouched output dimension.
SHAPES = (
    ("o_proj  cell", 4096, 1024),
    ("down_proj cell", 4096, 3072),
    ("o_proj  70B", 8192, 2048),
)
REPEATS = 10


def _time(fn, repeats=REPEATS):
    """Median wall time of ``fn`` in ms, with the GPU synchronised around each call."""
    samples = []
    for _ in range(repeats + 2):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples[2:])  # drop warm-up


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")

    results = []
    for label, rows, cell_cols in SHAPES:
        # The parameter as vLLM holds it: the full local tensor, of which this client owns
        # `cell_cols` columns per mesh cell. Two cells for gen TP=2 against train TP=4.
        param = torch.zeros(rows, cell_cols * 2, dtype=torch.bfloat16, device="cuda")
        destination = param[:, 0:cell_cols]
        landing = torch.empty(rows * cell_cols, dtype=torch.bfloat16, device="cuda")
        landing.uniform_(-1, 1)
        payload_bytes = landing.numel() * landing.element_size()

        scatter_ms = _time(lambda: destination.copy_(landing.view(rows, cell_cols)))
        # A contiguous copy of the same bytes, as the floor: how much of the scatter cost
        # is the strided destination rather than the bytes themselves.
        flat_dst = torch.empty_like(landing)
        flat_ms = _time(lambda: flat_dst.copy_(landing))

        descriptor_ms = rows * US_PER_DESCRIPTOR / 1e3
        results.append(
            {
                "shape": label,
                "rows": rows,
                "cell_cols": cell_cols,
                "payload_mib": payload_bytes / 2**20,
                "scatter_ms": scatter_ms,
                "contiguous_copy_ms": flat_ms,
                "strided_penalty": scatter_ms / flat_ms if flat_ms else None,
                "scatter_gbps": payload_bytes / (scatter_ms / 1e3) / 1e9,
                "descriptor_ms_if_strided_rdma": descriptor_ms,
                "speedup_vs_descriptors": descriptor_ms / scatter_ms if scatter_ms else None,
            }
        )
        r = results[-1]
        print(
            f"{label:16s} rows={rows:5d} {r['payload_mib']:7.1f} MiB  "
            f"scatter {scatter_ms:7.3f} ms ({r['scatter_gbps']:6.1f} GB/s, "
            f"{r['strided_penalty']:4.1f}x a flat copy)  "
            f"vs {descriptor_ms:7.3f} ms of descriptors  "
            f"-> {r['speedup_vs_descriptors']:7.1f}x"
        )

    # The gate: P2 is only worth building if the scatter is far cheaper than the
    # descriptors it removes. 10x is the margin, so the conclusion survives a scatter
    # that turns out slower on another GPU or a different shape.
    worst = min(r["speedup_vs_descriptors"] for r in results)
    verdict = {
        "worst_speedup_vs_descriptors": worst,
        "gate_margin": 10.0,
        "p2_is_worth_building": worst >= 10.0,
    }
    print(f"\nverdict: {verdict}")

    out = Path(__file__).with_name("bench_strided_scatter.result.json")
    out.write_text(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "us_per_descriptor": US_PER_DESCRIPTOR,
                "repeats": REPEATS,
                "shapes": results,
                "verdict": verdict,
            },
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
