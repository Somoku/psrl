"""The generation dump must survive its own async boundary.

`_dump_generations` submits `self._write_generations` to a `ThreadPoolExecutor` with the
sample lists and the current step. A `@staticmethod` decorator on a function that still
declares `self` makes `self` an ordinary parameter, so the executor's positional call is
one short and the future raises `TypeError: missing 1 required positional argument:
'global_steps'`. Nothing exercised this path until `trainer.validation_data_dir` was set;
the trainer then died after validation had already finished.

The check is an AST comparison rather than a call so it needs no Ray, no vLLM, and no
checkpoint: the property is that the number of values submitted matches the number of
parameters the method will bind.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test

TRAINER = Path(__file__).resolve().parents[2] / "psrl" / "trainer" / "ppo" / "ray_trainer.py"


def _class_method(path: Path, class_name: str) -> ast.ClassDef:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} not found in {path}")


def _find(node: ast.AST, name: str) -> ast.FunctionDef:
    for child in ast.walk(node):
        if isinstance(child, ast.FunctionDef) and child.name == name:
            return child
    raise AssertionError(f"{name} not found")


def _submit_args(function: ast.FunctionDef, attribute: str) -> int:
    """Number of positional values passed to `self.<attribute>` inside `function`."""
    for call in ast.walk(function):
        if not isinstance(call, ast.Call):
            continue
        if isinstance(call.func, ast.Attribute) and call.func.attr == "submit":
            target = call.args[0]
            if isinstance(target, ast.Attribute) and target.attr == attribute:
                return len(call.args) - 1
    raise AssertionError(f"no submit of {attribute} found in {function.name}")


def test_write_generations_is_callable_through_its_submission():
    cls = _class_method(TRAINER, "PSRL_RayPPOTrainer")
    write = _find(cls, "_write_generations")
    dump = _find(cls, "_dump_generations")

    decorators = {d.id for d in write.decorator_list if isinstance(d, ast.Name)}
    assert "staticmethod" not in decorators, (
        "_write_generations declares `self`; decorating it staticmethod shifts every "
        "parameter by one and the submitted call loses `global_steps`"
    )

    bound = len(write.args.args) - 1  # minus `self`
    assert _submit_args(dump, "_write_generations") == bound, (
        "the executor submits a different number of values than the method binds"
    )
