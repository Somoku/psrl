"""`RolloutScheduler` must accept every call the vLLM engine makes to its parent class.

vLLM 0.30 changed `Scheduler.schedule` to take `throttle_prefills`. The subclass still
declared `schedule(self)`, which nothing noticed until a live engine called it: the engine
core died on its first scheduling step with `TypeError: schedule() takes 1 positional
argument but 2 were given`, after weights had already been pulled. Comparing signatures
needs no GPU and no engine, so a vLLM bump that moves one now fails here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test

REPO = Path(__file__).resolve().parents[2]
VLLM_SCHED = REPO / "third_party" / "vllm-0.30.0" / "vllm" / "v1" / "core" / "sched"
SUBCLASS = REPO / "psrl" / "workers" / "gen" / "rollout_scheduler.py"


def _signatures(path: Path) -> dict[tuple[str, str], list[str]]:
    out = {}
    for cls in (n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)):
        for fn in cls.body:
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = fn.args
                names = [x.arg for x in a.posonlyargs + a.args]
                names += ["*" + a.vararg.arg] if a.vararg else []
                names += [x.arg for x in a.kwonlyargs]
                names += ["**" + a.kwarg.arg] if a.kwarg else []
                out[(cls.name, fn.name)] = names
    return out


@pytest.mark.skipif(not VLLM_SCHED.is_dir(), reason="vendored vLLM source not present")
def test_every_override_matches_the_parent_signature():
    parents: dict[str, tuple[str, list[str]]] = {}
    # AsyncScheduler overrides Scheduler, so it wins where both define a method.
    for filename in ("async_scheduler.py", "scheduler.py"):
        for (cls, name), params in _signatures(VLLM_SCHED / filename).items():
            parents.setdefault(name, (cls, params))

    mismatches = []
    for (cls, name), params in _signatures(SUBCLASS).items():
        if cls != "RolloutScheduler" or name not in parents:
            continue
        parent_cls, parent_params = parents[name]
        if params != parent_params:
            mismatches.append(f"{name}: override {params} vs {parent_cls}.{name} {parent_params}")

    assert not mismatches, "RolloutScheduler diverges from the vendored vLLM scheduler:\n" + "\n".join(mismatches)


def test_schedule_forwards_throttle_prefills():
    """The argument must reach the parent, not merely be accepted."""
    source = SUBCLASS.read_text()
    assert "super().schedule(throttle_prefills)" in source
