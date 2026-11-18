#!/usr/bin/env python3
"""Build Signatory 1.2.6.1.9.0 for this Python and installed PyTorch."""

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

VERSION = "1.2.6.1.9.0"
TORCH_VERSION = "2.13.0"


def check_torch_version(installed):
    from packaging.version import Version

    if Version(installed).public != TORCH_VERSION:
        raise RuntimeError(
            f"Requires torch=={TORCH_VERSION} to match setup.py; found {installed}. "
            "Install the pinned PyTorch before building Signatory."
        )


def require_prerequisites():
    try:
        for package in ("pip", "setuptools", "wheel", "packaging"):
            importlib.metadata.version(package)
        import torch
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError(
            f"Missing build prerequisite: {exc}; install pip, packaging, "
            f"setuptools, wheel, and torch=={TORCH_VERSION}"
        ) from exc
    check_torch_version(torch.__version__)
    return torch


def mac_build_environment(tempdir):
    prefix = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
    omp = Path(prefix)
    if not (omp / "include").is_dir() or not (omp / "lib/libomp.dylib").exists():
        raise RuntimeError(f"Homebrew libomp is incomplete at {omp}")
    clang = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
    sdk = subprocess.check_output(["xcrun", "--show-sdk-path"], text=True).strip()
    wrapper = Path(tempdir) / "clang-openmp"
    wrapper.write_text(
        '#!/bin/bash\nargs=()\nfor arg in "$@"; do\n'
        "  if [[ $arg == -fopenmp ]]; then args+=( -Xpreprocessor -fopenmp ); "
        'else args+=("$arg"); fi\ndone\n'
        f"exec {clang!r} -isysroot {sdk!r} -I{str(omp / 'include')!r} \"${{args[@]}}\"\n"
    )
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env["CC"] = str(wrapper)
    env["CXX"] = str(wrapper)
    env["CFLAGS"] = f"{env.get('CFLAGS', '')} -I{omp / 'include'}".strip()
    env["LDFLAGS"] = f"{env.get('LDFLAGS', '')} -L{omp / 'lib'} -lomp".strip()
    env["ARCHFLAGS"] = f"{env.get('ARCHFLAGS', '')} -arch {platform.machine()}".strip()
    return env


def patch_macos_extensions():
    dist = importlib.metadata.distribution("signatory")
    extensions = [
        Path(dist.locate_file(f)) for f in (dist.files or []) if str(f).endswith(".so")
    ]
    if not extensions:
        raise RuntimeError(
            "Signatory installed, but its extension module could not be located"
        )
    destination = "@loader_path/../torch/lib/libomp.dylib"
    patched = 0
    for extension in extensions:
        dependencies = subprocess.check_output(
            ["otool", "-L", str(extension)], text=True
        )
        for line in dependencies.splitlines()[1:]:
            dependency = line.strip().split(" (", 1)[0]
            if dependency.endswith("/libomp.dylib") and dependency.startswith("/"):
                subprocess.run(
                    [
                        "install_name_tool",
                        "-change",
                        dependency,
                        destination,
                        str(extension),
                    ],
                    check=True,
                )
                subprocess.run(
                    ["codesign", "--force", "--sign", "-", str(extension)], check=True
                )
                patched += 1
    if not patched:
        raise RuntimeError(
            "No absolute libomp.dylib dependency found in the built Signatory extensions"
        )


def smoke_test():
    code = (
        "import torch, signatory; "
        "x=torch.randn(2,3,2,requires_grad=True); "
        "y=signatory.signature(x, depth=2, basepoint=True); y.sum().backward(); "
        "assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def main():
    import argparse

    argparse.ArgumentParser(description=__doc__).parse_args()
    torch = require_prerequisites()
    macos = sys.platform == "darwin"
    if macos:
        required = ("brew", "xcrun", "otool", "install_name_tool", "codesign")
        missing = [tool for tool in required if not shutil.which(tool)]
        if missing:
            raise RuntimeError("Missing macOS build tools: " + ", ".join(missing))
        bundled_omp = Path(torch.__file__).resolve().parent / "lib/libomp.dylib"
        if not bundled_omp.is_file():
            raise RuntimeError(
                f"Required bundled OpenMP runtime not found: {bundled_omp}"
            )
    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-cache-dir",
        "--no-build-isolation",
        "--no-binary",
        "signatory",
        "--force-reinstall",
        f"signatory=={VERSION}",
    ]
    if macos:
        with tempfile.TemporaryDirectory(prefix="signatory-clang-") as tempdir:
            env = mac_build_environment(tempdir)
            subprocess.run(args, env=env, check=True)
        patch_macos_extensions()
    else:
        subprocess.run(args, check=True)
        print("Linux build path is generic and has not been tested by this helper.")
    smoke_test()
    print("Signatory build and forward/backward smoke test succeeded.")
    print(
        "Rebuild after changing Python or PyTorch; the extension targets their current ABI."
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f"install_signatory: {exc}")
