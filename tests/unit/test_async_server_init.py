"""The async vLLM server must signal readiness after its worker RPC returns.

`nixl_protocol` waits on `_is_init_nixl_client`, which `init_nixl_client` sets after the
collective RPC. A helper inserted before that line once left the `set()` after a `return`
inside the helper: unreachable, so every handshake waited out its 1800 s budget with the
servers healthy and nothing logged. An AST check catches both the missing signal and
statements that can never run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test

SERVER = Path(__file__).resolve().parents[2] / "psrl" / "workers" / "gen" / "vllm_async_server.py"


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(SERVER.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {SERVER}")


def _sets_event(function: ast.FunctionDef, attribute: str) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        target = node.func.value
        if (
            node.func.attr == "set"
            and isinstance(target, ast.Attribute)
            and target.attr == attribute
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            return True
    return False


def test_init_nixl_client_signals_readiness():
    function = _function("init_nixl_client")
    assert _sets_event(function, "_is_init_nixl_client"), (
        "init_nixl_client must set _is_init_nixl_client; nixl_protocol waits on it, and a "
        "missing signal turns every handshake into a 30-minute timeout"
    )


def _unreachable_statements(node: ast.AST) -> list[ast.stmt]:
    """Statements that follow a `return` in the same block."""
    dead: list[ast.stmt] = []
    for _, value in ast.iter_fields(node):
        bodies = value if isinstance(value, list) else [value]
        for body in bodies:
            if not isinstance(body, list) or not all(isinstance(item, ast.stmt) for item in body):
                continue
            for index, statement in enumerate(body):
                if isinstance(statement, ast.Return) and index + 1 < len(body):
                    dead.extend(body[index + 1 :])
                dead.extend(_unreachable_statements(statement))
    return dead


def test_no_statement_follows_a_return():
    tree = ast.parse(SERVER.read_text())
    dead = _unreachable_statements(tree)
    assert not dead, "unreachable statements after a return: " + ", ".join(
        f"line {statement.lineno}" for statement in dead
    )
