"""Isolate verl's logger from root-logger changes, without depending on verl for it.

verl's own `configure_verl_logging` exists in some checkouts and not others, and psrl
calls it on every `TaskRunner.run`. Importing it from verl made psrl's startup depend on
which verl checkout happens to be on `PYTHONPATH`: the symbol is absent from the deployed
(patched) verl, so `main_ppo` failed at import time with an `ImportError` naming verl.

There is nothing verl-specific in the body -- it configures the `"verl"` logger using only
the standard library -- so owning it here removes the dependency rather than working
around it.
"""

from __future__ import annotations

import logging
import os
import sys


def configure_verl_logging() -> None:
    """Give the ``verl`` logger its own handler and detach it from the root logger.

    `propagate = False` is the point of this: without it, anything that reconfigures the
    root logger (Ray, vLLM, and Megatron all do) silently changes verl's output format and
    level mid-run.
    """
    level = os.getenv("VERL_LOGGING_LEVEL", "INFO")

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s:%(message)s"))

    verl_logger = logging.getLogger("verl")
    verl_logger.handlers = [handler]
    verl_logger.setLevel(level)
    verl_logger.propagate = False
