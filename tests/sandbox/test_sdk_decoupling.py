"""The import boundary between PSRL and the p3b service.

p3b exists to be an independently deployable service that PSRL calls, not a
library PSRL hosts. That claim is only worth something if it is enforced, and
until now it was a docstring promise in `psrl/sandbox/backends/p3b.py` with
nothing checking it. A promise of that shape decays quietly: the first import
that crosses the line still works, still passes every other test, and leaves
the coupling for whoever next tries to deploy the service on its own.

Three boundaries, each with its own failure if it breaks:

1. The bridge may import only the standalone SDK and PSRL's abstract core. An
   import from elsewhere in `psrl.sandbox` would pull the in-process path --
   the Docker engine client, the local lifecycle, the capacity coordinator --
   into a backend whose whole point is that the service owns those decisions.

2. The SDK may not import PSRL at all. If it did, `pip install sandboxd` would
   drag the trainer in, and the SDK could not be used by anything other than
   PSRL.

3. There may be exactly one copy of the SDK. There were two for a while
   (`psrl/pysandbox/` and `p3b/sdk/sandboxd/`), logic-identical but with
   separate import paths, and the end-to-end suite exercised the copy rather
   than the artifact that ships. A second copy does not announce itself when it
   starts to drift.

These are read as source rather than by importing, so a violation is reported
as the offending line rather than as whatever the import happened to break.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BRIDGE = REPO / "psrl" / "sandbox" / "backends" / "p3b.py"
SDK = REPO / "p3b" / "sdk" / "sandboxd"


def imported_modules(path: Path) -> set[str]:
    """Every module name this file imports, including inside functions.

    `ast` rather than a regex: a nested import inside a method is still an
    import, and a comment that mentions a module name is not.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_the_bridge_imports_nothing_from_the_in_process_path() -> None:
    # The one module it may take from PSRL is `core`, which holds the abstract
    # interfaces every backend implements. Anything else is the in-process path.
    offenders = sorted(
        name
        for name in imported_modules(BRIDGE)
        if name.startswith("psrl.sandbox") and name != "psrl.sandbox.core"
    )
    assert not offenders, (
        f"{BRIDGE.relative_to(REPO)} imports {offenders} from psrl.sandbox. "
        "It may import only psrl.sandbox.core (the abstract interfaces) and the "
        "standalone sandboxd SDK; anything else pulls the in-process sandbox path "
        "into a backend whose decisions belong to the service."
    )


def test_the_bridge_uses_the_standalone_sdk() -> None:
    # The positive half: it has to go through the SDK, not reach past it.
    imports = imported_modules(BRIDGE)
    assert any(name == "sandboxd" or name.startswith("sandboxd.") for name in imports), (
        f"{BRIDGE.relative_to(REPO)} does not import the sandboxd SDK, so it is not "
        "talking to the service through its published surface."
    )


def test_the_sdk_does_not_import_psrl() -> None:
    # A single violation here makes `pip install sandboxd` depend on the trainer.
    violations: list[str] = []
    for module in sorted(SDK.glob("*.py")):
        for name in sorted(imported_modules(module)):
            if name == "psrl" or name.startswith("psrl."):
                violations.append(f"{module.relative_to(REPO)} imports {name}")
    assert not violations, (
        "The SDK must be installable on its own, so it cannot import PSRL:\n  "
        + "\n  ".join(violations)
    )


def test_there_is_exactly_one_copy_of_the_sdk() -> None:
    # A vendored copy inside PSRL is what this guards against. It was real:
    # psrl/pysandbox/ shadowed the shipped package and the live suite tested it.
    vendored = REPO / "psrl" / "pysandbox"
    assert not vendored.exists(), (
        f"{vendored.relative_to(REPO)} exists again. The SDK ships from "
        "p3b/sdk/sandboxd; a second copy inside PSRL drifts from it silently and "
        "makes the end-to-end test exercise something other than what is deployed."
    )

    # And the one that does exist has to be the installable one.
    assert (SDK / "__init__.py").is_file(), (
        f"{SDK.relative_to(REPO)} is missing, so there is no SDK to install."
    )
    assert (SDK.parent / "pyproject.toml").is_file(), (
        "p3b/sdk has no pyproject.toml, so `pip install p3b/sdk` cannot work."
    )


def test_nothing_in_the_tree_imports_the_removed_copy() -> None:
    # Catches a stale import left behind anywhere, including in a test or a
    # benchmark, which is how the previous copy stayed alive.
    # Assembled rather than written literally, so this file's own assertion
    # messages -- which have to name the module to be useful -- are not matches.
    removed = "psrl" + ".pysandbox"
    offenders: list[str] = []
    for path in REPO.glob("**/*.py"):
        parts = set(path.parts)
        if "third_party" in parts or ".git" in parts or "__pycache__" in parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if removed in text:
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, (
        "These files still reference the removed psrl.pysandbox copy; they should "
        "import `sandboxd`:\n  " + "\n  ".join(sorted(offenders))
    )
