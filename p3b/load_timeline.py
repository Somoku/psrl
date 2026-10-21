#!/usr/bin/env python3
"""Record a p3b cluster's load over time, for verifying scheduling behaviour.

There is no point tuning an admission controller you cannot see. This samples the control
plane's fleet view on a fixed cadence and writes one row per sample to a TSV, plus a
header naming every column, so a run can be replayed and argued about afterwards.

What it records, and why each part matters:

- **Per node**: envelope (what the node will admit against, *after* overcommit),
  per-class headroom, live sandboxes, and measured CPU/memory. The gap between the
  envelope and the headroom is the reservations; the gap between the reservations and
  the measurement is the packing error, which is the thing under test.
- **Cluster**: the totals, and the counters that say what the fleet *refused* --
  capacity exhausted vs no capable node, which are different failures and must not be
  read as one number.
- **Balance**: the spread of load across nodes. A cluster that is half idle on one node
  while another refuses is a placement defect, not a capacity one, and it is invisible
  in the totals.

Written as TSV with a leading header comment so it opens in anything, and so a diff
between two runs is a diff of numbers rather than of structure.
"""

from __future__ import annotations

import argparse, asyncio, sys, time
from sandboxd import SandboxClient


def _num(d, *keys, default=0.0):
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return default


def sample(reply: dict, ts: float) -> list[tuple]:
    nodes = reply.get("nodes", []) or []
    rows = []
    tot_mem_head = tot_cpu_head = 0.0
    tot_mem_env = tot_cpu_env = 0.0
    lives = []
    for n in nodes:
        nid = n.get("node_id", "?")
        env = n.get("envelope") or {}
        mem_env = _num(env, "memory_mb", "memory")
        cpu_env = _num(env, "cpu_millis", "cpu")
        # Headroom is what the envelope has left after the node's promises. The
        # envelope object carries the total, not the remainder: reading a
        # `memory_headroom_mb` key that does not exist silently reported the
        # envelope as headroom, which made a full node look idle.
        granted = n.get("granted") or {}
        mem_granted = _num(granted, "memory_mb", "memory")
        cpu_granted = _num(granted, "cpu_millis", "cpu")
        mem_head = max(mem_env - mem_granted, 0.0)
        cpu_head = max(cpu_env - cpu_granted, 0.0)
        class_head = n.get("class_headroom") or {}
        for cls, hr in class_head.items():
            rows.append((ts, nid, f"class:{cls}",
                         _num(hr, "memory_mb", "memory"), _num(hr, "cpu_millis", "cpu"),
                         "", _num(n, "live_sandboxes"),
                         _num(n, "cpu_used_pct"), _num(n, "mem_used_pct")))
        rows.append((ts, nid, "node", mem_head, cpu_head,
                     f"{mem_env}/{cpu_env}", _num(n, "live_sandboxes"),
                     _num(n, "cpu_used_pct"), _num(n, "mem_used_pct")))
        tot_mem_head += mem_head
        tot_cpu_head += cpu_head
        tot_mem_env += mem_env
        tot_cpu_env += cpu_env
        lives.append(_num(n, "live_sandboxes"))
    spread = (max(lives) - min(lives)) if lives else 0.0
    rows.append((ts, "cluster", "total", tot_mem_head, tot_cpu_head,
                 f"{tot_mem_env}/{tot_cpu_env}", sum(lives), spread, 0.0))
    rows.append((ts, "cluster", "refusals",
                 _num(reply, "capacity_exhausted"), _num(reply, "no_candidate"),
                 "", _num(reply, "reservations_open"),
                 _num(reply, "locality_hit_ratio"), _num(reply, "drained_nodes")))
    return rows


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default="28.49.198.250:9000")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--out", default="/home/psrl/p3b/load_timeline.tsv")
    ap.add_argument("--duration", type=float, default=0.0, help="0 = until interrupted")
    args = ap.parse_args()

    client = SandboxClient(args.endpoint)
    with open(args.out, "w") as fh:
        fh.write("# p3b load timeline\n")
        fh.write("# ts\tiso\tnode\tscope\tmemory_headroom_mb\tcpu_headroom_millis\t"
                 "envelope\tlive_sandboxes\tcpu_used_pct\tmem_used_pct\n")
        started = time.time()
        while True:
            try:
                reply = await client.fleet_detail()
            except Exception as exc:
                print(f"sample failed: {exc!r}", file=sys.stderr)
                await asyncio.sleep(args.interval)
                continue
            now = time.time()
            for row in sample(reply, now):
                ts, nid, scope = row[0], row[1], row[2]
                fh.write(f"{ts:.3f}\t{time.strftime('%H:%M:%S', time.localtime(ts))}\t"
                         f"{nid}\t{scope}\t" + "\t".join(f"{v:g}" if isinstance(v, float) else str(v)
                                                          for v in row[3:]) + "\n")
            fh.flush()
            if args.duration and now - started >= args.duration:
                break
            await asyncio.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
