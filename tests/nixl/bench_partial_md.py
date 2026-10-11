"""How much metadata a join actually costs, full versus partial (item E4).

Every member of the cluster loads the metadata of every joining member, so join cost is
`O(members x md_bytes)` and `md_bytes` grows with the joiner's *registered region count*.
At 10k-GPU scale that product is what decides whether elasticity is usable, so the design
proposes splitting the exchange: connection info for everyone, descriptors only for the
peers that will actually read.

The design quotes ~406 B/region, 883 B for a partial of one region out of 4096, and 401 B
for conn-info-only. Those numbers are the justification for the whole item, and they came
from a different machine and NIXL build -- so measure them here before implementing
against them.

Run on one GPU:

    python tests/nixl/bench_partial_md.py

Writes `bench_partial_md.result.json` next to this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import nixl._bindings as nixlBind
import torch
from nixl._api import nixl_agent, nixl_agent_config

REGION_COUNTS = (1, 16, 256, 1024, 4096)
REGION_ELEMS = 1024  # bf16 -> 2KB per region, small so the sweep stays quick


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")

    rows = []
    for count in REGION_COUNTS:
        agent = nixl_agent(f"md-bench-{count}", nixl_agent_config(False, True, 0))
        tensors = [
            torch.zeros(REGION_ELEMS, dtype=torch.bfloat16, device="cuda")
            for _ in range(count)
        ]
        reg = agent.register_memory(tensors)
        assert reg, f"register_memory failed for {count} regions"

        full = len(agent.get_agent_metadata())
        # Conn-info only: an empty descriptor list, which is what every member needs.
        # Built directly rather than via `get_reg_descs([])`, which indexes `descs[0]` to
        # infer the element type and so cannot express "no descriptors".
        empty = nixlBind.nixlRegDList(nixlBind.VRAM_SEG, [])
        conn_only = len(agent.get_partial_agent_metadata(empty, True))
        # One region's descriptors, which is what an actual reader needs.
        one_region = len(
            agent.get_partial_agent_metadata(agent.get_reg_descs([tensors[0]]), False)
        )

        rows.append(
            {
                "regions": count,
                "full_md_bytes": full,
                "conn_only_bytes": conn_only,
                "one_region_bytes": one_region,
                "bytes_per_region": (full - conn_only) / count if count else None,
                "conn_only_compression": full / conn_only if conn_only else None,
            }
        )
        print(
            f"regions={count:5d}  full={full:8d}B  conn_only={conn_only:5d}B  "
            f"one_region={one_region:6d}B  "
            f"per_region={rows[-1]['bytes_per_region']:7.1f}B  "
            f"compression={rows[-1]['conn_only_compression']:8.1f}x"
        )
        del reg, tensors, agent

    # The claim E4 rests on: conn-info cost must not grow with region count, or the
    # two-stage split buys nothing at scale.
    conn_sizes = {r["conn_only_bytes"] for r in rows}
    verdict = {
        "conn_only_is_constant": max(conn_sizes) - min(conn_sizes) < 64,
        "conn_only_range_bytes": [min(conn_sizes), max(conn_sizes)],
        "max_compression": max(r["conn_only_compression"] for r in rows),
    }
    print(f"\nverdict: {verdict}")

    out = Path(__file__).with_name("bench_partial_md.result.json")
    out.write_text(
        json.dumps(
            {"gpu": torch.cuda.get_device_name(0), "sweep": rows, "verdict": verdict},
            indent=2,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
