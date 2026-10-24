"""The DeepGEMM toolchain probe reads the release NVCC actually prints."""

from __future__ import annotations

import pytest

from psrl.utils import cuda_toolchain


def test_parse_nvcc_release():
    output = (
        "nvcc: NVIDIA (R) Cuda compiler driver\n"
        "Cuda compilation tools, release 12.8, V12.8.93\n"
        "Build cuda_12.8.r12.8/compiler.35583870_0\n"
    )

    assert cuda_toolchain.parse_nvcc_version(output) == (12, 8)


def test_parse_nvcc_double_digit_minor():
    assert cuda_toolchain.parse_nvcc_version("release 12.10, V12.10.1") == (12, 10)


def test_parse_nvcc_rejects_unrelated_output():
    assert cuda_toolchain.parse_nvcc_version("command not found") is None


@pytest.mark.parametrize(
    ("version", "available"),
    [((12, 8), False), ((12, 9), True), ((13, 0), True), (None, False)],
)
def test_deep_gemm_requires_the_jit_toolchain(monkeypatch, version, available):
    monkeypatch.setattr(cuda_toolchain, "nvcc_version", lambda: version)

    assert cuda_toolchain.deep_gemm_jit_available() is available
