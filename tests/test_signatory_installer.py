"""Protect the native extension's PyTorch ABI requirement."""

import runpy
from pathlib import Path

import pytest

check_torch_version = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/install_signatory.py")
)["check_torch_version"]


@pytest.mark.parametrize("version", ["2.13.0", "2.13.0+cpu", "2.13.0+cu130"])
def test_accepts_pinned_torch_with_local_build_suffix(version):
    check_torch_version(version)


@pytest.mark.parametrize(
    "version", ["1.9.0", "2.12.0", "2.13.0rc1", "2.13.1", "2.15.0"]
)
def test_rejects_torch_versions_project_install_would_replace(version):
    with pytest.raises(RuntimeError, match="Install the pinned PyTorch"):
        check_torch_version(version)
