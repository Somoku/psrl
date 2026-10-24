"""CUDA toolchain facts that decide whether a kernel can be JIT-compiled."""

from __future__ import annotations

import os
import re
import shutil
import subprocess

# DeepGEMM compiles its kernels on first use and refuses an older toolchain, even
# when the package itself imports. vLLM's `has_deep_gemm()` checks the package only,
# so a host can pass that check and still fail when the first FP8 layer loads.
DEEP_GEMM_MIN_NVCC = (12, 9)

_NVCC_RELEASE = re.compile(r"release (\d+)\.(\d+)")


def parse_nvcc_version(output: str) -> tuple[int, int] | None:
    """Return the ``(major, minor)`` CUDA release named in ``nvcc --version`` output."""
    match = _NVCC_RELEASE.search(output)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def nvcc_version() -> tuple[int, int] | None:
    """Ask the CUDA toolkit this process would use for its version."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            candidate = os.path.join(cuda_home, "bin", "nvcc")
            if os.path.exists(candidate):
                nvcc = candidate
    if nvcc is None:
        return None
    try:
        completed = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_nvcc_version(completed.stdout)


def deep_gemm_jit_available() -> bool:
    """Whether the CUDA toolchain can compile DeepGEMM's JIT kernels."""
    version = nvcc_version()
    return version is not None and version >= DEEP_GEMM_MIN_NVCC
