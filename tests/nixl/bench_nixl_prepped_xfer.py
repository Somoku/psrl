"""Micro-benchmark for the two NIXL APIs the weight-transport design depends on.

PSRL has never called ``prep_xfer_dlist`` / ``make_prepped_xfer`` nor the Nx5 strided
descriptor form, so the design (``docs/design/weight_transport.md`` landing item 0)
cannot assume how they behave. This benchmark answers four questions before the
large refactor starts:

Q1  Does ``prep_xfer_dlist`` accept PSRL-scale descriptor counts, and what does the
    one-time preparation cost?
Q2  Is ``make_prepped_xfer`` actually cheaper per submission than ``initialize_xfer``
    (the design claims it removes an ``O(#tensors)`` per-step cost)?
Q3  Does the Nx5 strided form work, and how many plain descriptors does one strided
    descriptor replace?
Q4  Do prepped handles survive index reordering, i.e. can one handle serve the
    chunk-offset staggering in the design?

Single process, single GPU, two in-process agents. That is enough for the
API-behaviour questions; bandwidth belongs to the multi-node e2e tests.

Run::

    python tests/nixl/bench_nixl_prepped_xfer.py
    python tests/nixl/bench_nixl_prepped_xfer.py --counts 128,1024 --json out.json

**Budget at least 60s per agent on a shared container host.** ``nixl_agent()`` spends
that time inside ``uct_tcp_query_devices``, which calls ``ucs_netif_is_active()`` --
and therefore ``getifaddrs()`` -- once per network interface. On a Docker host with
~460 interfaces (hundreds of ``br-*`` bridges and ``veth*`` pairs from other
containers) ``getifaddrs()`` costs ~69ms, so enumeration alone is ~32s per agent and
this benchmark needs two. ``UCX_TLS``/``UCX_NET_DEVICES`` do not help because
enumeration precedes filtering. It looks exactly like a hang; it is not. Verify with
``ls /sys/class/net | wc -l`` before concluding NIXL is broken.

Findings on H20 / NIXL 1.5.0 built from ``third_party/nixl`` with UCX
(see ``bench_nixl_prepped_xfer.result.json``):

================  ==========  =============  ==========  =========
descriptors       prep (ms)   prepped (ms)   init (ms)   speedup
================  ==========  =============  ==========  =========
1024              0.005       0.003          0.030       10.0x
4096              0.019       0.006          0.113       18.8x
16384             0.074       0.019          0.440       23.2x
================  ==========  =============  ==========  =========

* ``make_prepped_xfer`` is 10-23x cheaper than ``initialize_xfer`` and the gap widens
  with descriptor count, because the prepped path only ships indices while
  ``initialize_xfer`` re-serializes the whole descriptor list each time.
* **Indices must be an int32 ndarray.** Passing a Python list costs ~15x more
  (0.088 ms vs 0.006 ms at 4096 descriptors) -- pybind11 converts list elements one
  by one, which silently reintroduces the per-descriptor cost the prepped path exists
  to remove. An earlier run of this benchmark passed lists and wrongly concluded that
  prepped submission was slower than ``initialize_xfer``.
* **Nx5 strided needs NIXL >= 1.5.0.** The 1.2.0 wheel pinned by
  ``scripts/install_nixl.sh`` has no such overload. With 1.5.0 one strided descriptor
  replaces all ``count`` plain ones, which drops preparation to 0.002 ms and
  submission to 0.001 ms regardless of count.
* Prepped handles accept arbitrary index order and subsets, so one handle per client
  can serve every chunk-offset pattern the design needs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

try:
    import nixl._bindings as nixl_bindings
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError as exc:  # pragma: no cover - environment guard
    print(f"NIXL is not importable: {exc}", file=sys.stderr)
    sys.exit(1)


# PSRL registers one descriptor per (key, shard). These counts bracket the real range:
# a dense 7B lands near 300, Qwen3-30B-A3B near 3k, and an HF-expanded Kimi-K2 would
# reach 140k -- which is exactly the case MoE grouping (design 3.5.5) exists to avoid.
DEFAULT_COUNTS = (256, 1024, 4096, 16384)
DEFAULT_SLICE_BYTES = 64 * 1024
WARMUP_ITERS = 3
MEASURE_ITERS = 10


@dataclass
class Measurement:
    """Timing summary in milliseconds."""

    label: str
    count: int
    median_ms: float
    p90_ms: float
    per_desc_us: float

    @staticmethod
    def of(label: str, count: int, samples_s: list[float]) -> Measurement:
        ordered = sorted(samples_s)
        median_s = statistics.median(ordered)
        p90_s = ordered[min(int(len(ordered) * 0.9), len(ordered) - 1)]
        return Measurement(
            label=label,
            count=count,
            median_ms=median_s * 1e3,
            p90_ms=p90_s * 1e3,
            per_desc_us=(median_s / count) * 1e6 if count else 0.0,
        )


@dataclass
class Report:
    gpu: str
    plugins: list[str]
    prep: list[Measurement] = field(default_factory=list)
    submit_prepped: list[Measurement] = field(default_factory=list)
    submit_initialize: list[Measurement] = field(default_factory=list)
    strided: dict = field(default_factory=dict)
    reorder: dict = field(default_factory=dict)


def _time_it(fn, iters: int = MEASURE_ITERS, warmup: int = WARMUP_ITERS) -> list[float]:
    """Return per-iteration wall times in seconds, warmup excluded."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return samples


class Harness:
    """Owns the agent, one registered GPU buffer, and the descriptor lists over it."""

    def __init__(self, count: int, slice_bytes: int):
        self.count = count
        self.slice_bytes = slice_bytes
        self.agent_name = f"bench-{count}"
        # progress_thread is off: it only matters for completion polling, and leaving it
        # on makes short-lived agents slow to tear down.
        self.agent = nixl_agent(self.agent_name, nixl_agent_config(enable_prog_thread=False))

        # One contiguous allocation carved into `count` equal slices mirrors how PSRL
        # registers Megatron's DDP buffer: many descriptors over few allocations.
        self.buf = torch.zeros(count * slice_bytes, dtype=torch.uint8, device="cuda")
        base = self.buf.data_ptr()
        device_id = self.buf.device.index

        reg = np.zeros((1, 3), dtype=np.uint64)
        reg[0] = (base, count * slice_bytes, device_id)
        self.reg_descs = self.agent.register_memory(nixl_bindings.nixlRegDList(nixl_bindings.VRAM_SEG, reg))

        self.plain = np.zeros((count, 3), dtype=np.uint64)
        for i in range(count):
            self.plain[i] = (base + i * slice_bytes, slice_bytes, device_id)

        # Nx5 strided: (addr, len, dev_id, stride, num). The whole slice set collapses
        # into a single row because the slices are evenly spaced by construction.
        self.strided = np.zeros((1, 5), dtype=np.uint64)
        self.strided[0] = (base, slice_bytes, device_id, slice_bytes, count)

        self.xfer_descs = self.agent.get_xfer_descs(self.plain, mem_type="VRAM")
        # NIXL rejects an agent as its own remote ("remote agent name same as local"),
        # so pair two agents in-process. The peer registers the same buffer, which is
        # all `prep_xfer_dlist` needs to resolve remote descriptors.
        self.peer_name = f"{self.agent_name}-peer"
        self.peer = nixl_agent(self.peer_name, nixl_agent_config(enable_prog_thread=False))
        self.peer_reg = self.peer.register_memory(nixl_bindings.nixlRegDList(nixl_bindings.VRAM_SEG, reg))
        self.agent.add_remote_agent(self.peer.get_agent_metadata())
        self.peer.add_remote_agent(self.agent.get_agent_metadata())

    def close(self) -> None:
        for agent, name in ((self.agent, self.peer_name), (self.peer, self.agent_name)):
            try:
                agent.remove_remote_agent(name)
            except Exception:
                pass
        for agent, descs in ((self.agent, self.reg_descs), (self.peer, self.peer_reg)):
            try:
                agent.deregister_memory(descs)
            except Exception:
                pass
        del self.peer
        del self.buf
        torch.cuda.empty_cache()

    # --- Q1 -------------------------------------------------------------------
    def prep_handle(self, descs=None, remote: bool = False):
        """Prepare a dlist. ``remote=True`` describes the peer's side of a transfer."""
        return self.agent.prep_xfer_dlist(
            self.peer_name if remote else "NIXL_INIT_AGENT",
            descs if descs is not None else self.plain,
            mem_type="VRAM",
        )

    def measure_prep(self) -> Measurement:
        handles = []

        def once():
            handles.append(self.prep_handle())

        samples = _time_it(once)
        for h in handles:
            self.agent.release_dlist_handle(h)
        return Measurement.of("prep_xfer_dlist", self.count, samples)

    # --- Q2 -------------------------------------------------------------------
    def measure_submit_prepped(self) -> Measurement:
        """Cost of composing a transfer from an already-prepared dlist."""
        local = self.prep_handle()
        remote = self.prep_handle(remote=True)
        # int32 ndarray, NOT a Python list: pybind11 converts a list element by
        # element, which costs ~15x more than handing over a typed buffer.
        indices = np.arange(self.count, dtype=np.int32)
        made = []

        def once():
            made.append(self.agent.make_prepped_xfer("WRITE", local, indices, remote, indices))

        try:
            samples = _time_it(once)
        finally:
            for h in made:
                try:
                    self.agent.release_xfer_handle(h)
                except Exception:
                    pass
            self.agent.release_dlist_handle(local)
            self.agent.release_dlist_handle(remote)
        return Measurement.of("make_prepped_xfer", self.count, samples)

    def measure_submit_initialize(self) -> Measurement:
        """Baseline: what PSRL does today, one initialize_xfer per submission."""
        made = []

        def once():
            made.append(self.agent.initialize_xfer("WRITE", self.xfer_descs, self.xfer_descs, self.peer_name))

        try:
            samples = _time_it(once)
        finally:
            for h in made:
                try:
                    self.agent.release_xfer_handle(h)
                except Exception:
                    pass
        return Measurement.of("initialize_xfer", self.count, samples)

    # --- Q3 -------------------------------------------------------------------
    def check_strided(self) -> dict:
        """Measure the Nx5 strided form, or report that this build lacks it.

        One strided row ``(addr, len, dev_id, stride, count)`` stands in for ``count``
        plain rows. NIXL >= 1.5.0 routes such an array through ``to_stride_dlist``
        (``nixl_bindings.cpp:148``); the 1.2.0 wheel pinned by
        ``scripts/install_nixl.sh`` has no such overload and only accepts Nx3.
        """
        out: dict = {"count": self.count}

        # nixlXferDList is Nx3-only in every build -- strided goes through
        # prep_xfer_dlist instead. Record it so the asymmetry is not mistaken for
        # a missing feature.
        try:
            nixl_bindings.nixlXferDList(nixl_bindings.VRAM_SEG, self.strided)
            out["xferdlist_accepts_nx5"] = True
        except Exception as exc:
            out["xferdlist_accepts_nx5"] = False
            out["xferdlist_error"] = str(exc)

        try:
            local = self.prep_handle(self.strided)
            remote = self.prep_handle(self.strided, remote=True)
        except Exception as exc:
            out["prep_accepts_nx5"] = False
            out["strided_error"] = f"{type(exc).__name__}: {exc}"
            out["verdict"] = "this NIXL build lacks the Nx5 overload (needs >= 1.5.0); see scripts/install_nixl.sh"
            return out

        out["prep_accepts_nx5"] = True
        out["compression_ratio"] = self.count  # count plain rows -> 1 strided row
        made = []
        try:
            prep_samples = _time_it(lambda: self._prep_and_drop(self.strided))
            out["strided_prep_ms"] = statistics.median(prep_samples) * 1e3

            one = np.arange(1, dtype=np.int32)
            submit = _time_it(lambda: made.append(self.agent.make_prepped_xfer("WRITE", local, one, remote, one)))
            out["strided_submit_ms"] = statistics.median(submit) * 1e3
        finally:
            for h in made:
                try:
                    self.agent.release_xfer_handle(h)
                except Exception:
                    pass
            self.agent.release_dlist_handle(local)
            self.agent.release_dlist_handle(remote)
        return out

    def _prep_and_drop(self, descs) -> None:
        self.agent.release_dlist_handle(self.prep_handle(descs))

    # --- Q4 -------------------------------------------------------------------
    def check_reorder(self) -> dict:
        """A prepped handle must tolerate arbitrary index order and subsets.

        The design staggers each instance's starting chunk (3.5.2.3) and submits
        per-commit-group subsets, so one handle has to serve many index patterns.
        """
        local = self.prep_handle()
        remote = self.prep_handle(remote=True)
        out: dict = {}
        offset = self.count // 2
        patterns = {
            "identity": np.arange(self.count, dtype=np.int32),
            "rotated": np.roll(np.arange(self.count, dtype=np.int32), -offset),
            "subset_head": np.arange(min(8, self.count), dtype=np.int32),
            "reversed": np.arange(self.count, dtype=np.int32)[::-1].copy(),
        }
        try:
            for name, idx in patterns.items():
                try:
                    h = self.agent.make_prepped_xfer("WRITE", local, idx, remote, idx)
                    self.agent.release_xfer_handle(h)
                    out[name] = "ok"
                except Exception as exc:
                    out[name] = f"FAILED: {exc}"
        finally:
            self.agent.release_dlist_handle(local)
            self.agent.release_dlist_handle(remote)
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--counts",
        default=",".join(str(c) for c in DEFAULT_COUNTS),
        help="Comma-separated descriptor counts to sweep.",
    )
    parser.add_argument("--slice-bytes", type=int, default=DEFAULT_SLICE_BYTES)
    parser.add_argument("--json", help="Write the raw report to this path.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this benchmark.", file=sys.stderr)
        return 1

    counts = [int(c) for c in args.counts.split(",") if c.strip()]
    probe = nixl_agent("probe", nixl_agent_config(enable_prog_thread=False))
    report = Report(gpu=torch.cuda.get_device_name(0), plugins=list(probe.get_plugin_list()))
    del probe

    print(f"GPU      : {report.gpu}")
    print(f"plugins  : {', '.join(report.plugins)}")
    print(f"slice    : {args.slice_bytes} B\n")

    header = f"{'count':>8} {'prep(ms)':>10} {'prepped(ms)':>12} {'init(ms)':>10} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for count in counts:
        h = Harness(count, args.slice_bytes)
        try:
            prep = h.measure_prep()
            prepped = h.measure_submit_prepped()
            init = h.measure_submit_initialize()
            report.prep.append(prep)
            report.submit_prepped.append(prepped)
            report.submit_initialize.append(init)

            speedup = init.median_ms / prepped.median_ms if prepped.median_ms else float("inf")
            print(
                f"{count:>8} {prep.median_ms:>10.3f} {prepped.median_ms:>12.3f} "
                f"{init.median_ms:>10.3f} {speedup:>7.2f}x"
            )

            if count == counts[-1]:
                report.strided = h.check_strided()
                report.reorder = h.check_reorder()
        finally:
            h.close()

    print("\nQ3 strided (Nx5)")
    for k, v in report.strided.items():
        print(f"  {k:<26} {v}")

    print("\nQ4 prepped handle index reuse")
    for k, v in report.reorder.items():
        print(f"  {k:<26} {v}")

    if args.json:
        payload = {
            "gpu": report.gpu,
            "plugins": report.plugins,
            "slice_bytes": args.slice_bytes,
            "prep": [asdict(m) for m in report.prep],
            "submit_prepped": [asdict(m) for m in report.submit_prepped],
            "submit_initialize": [asdict(m) for m in report.submit_initialize],
            "strided": report.strided,
            "reorder": report.reorder,
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")

    failures = [k for k, v in report.reorder.items() if v != "ok"]
    if failures:
        print(f"\nFAIL: prepped handle rejected index patterns: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
