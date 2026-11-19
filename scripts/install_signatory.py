#!/usr/bin/env python3
"""Build Signatory 1.2.6.1.9.0 for this Python and installed PyTorch."""

import hashlib
import json
import tarfile
import urllib.request
import urllib.error
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


SDIST_SHA256 = "c9ab17df6286688c1c7b12c0e906031a1fe7681dfca551876c1ff616e12807d8"
SDIST_URL = f"https://files.pythonhosted.org/packages/source/s/signatory/signatory-{VERSION}.tar.gz"


def patch_linux_source(source, manifest=None):
    """Apply exact versioned edits atomically after checking every input/output."""
    manifest = Path(manifest) if manifest is not None else Path(__file__).parent / "patches" / f"signatory-{VERSION}-linux-autograd-v2.json"
    changes = []
    for relative, spec in json.loads(manifest.read_text()).items():
        path = Path(source) / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise RuntimeError(f"Unexpected Signatory source: {relative}")
        text = path.read_text()
        for edit in spec["edits"]:
            if text.count(edit["old"]) != 1:
                raise RuntimeError(f"Signatory patch conflict: {relative}")
            text = text.replace(edit["old"], edit["new"])
        if hashlib.sha256(text.encode()).hexdigest() != spec["patched_sha256"]:
            raise RuntimeError(f"Signatory patch output mismatch: {relative}")
        changes.append((path, text))
    for path, text in changes:
        path.write_text(text)


def build_linux_source(args):
    if not args or args[-1] != f"signatory=={VERSION}":
        raise RuntimeError("Unexpected Signatory source-build arguments")
    if not hasattr(tarfile, "data_filter"):
        raise RuntimeError("Safe extraction requires Python >=3.10.12 or >=3.11.4")
    try:
        _build_linux_source(args)
    except (OSError, urllib.error.URLError, tarfile.TarError) as exc:
        raise RuntimeError(f"Signatory source download/extraction/build file error: {exc}") from exc


def _build_linux_source(args):
    with tempfile.TemporaryDirectory(prefix="signatory-linux-") as tempdir:
        archive = Path(tempdir) / "source.tar.gz"
        urllib.request.urlretrieve(SDIST_URL, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SDIST_SHA256:
            raise RuntimeError("Signatory source archive SHA256 mismatch")
        with tarfile.open(archive) as package:
            package.extractall(tempdir, filter="data")
        source = Path(tempdir) / f"signatory-{VERSION}"
        # Reviewed invariant: detached inputs are read-only; in-place writes
        # target private allocations only. No returned shared storage requires
        # shared version tracking (the basepoint gradient's private base is
        # unreachable). Python callers create views outside this guard.
        # Raw OpenMP workers do not inherit this thread-local guard.
        # Any future source change requires a new alias/mutation review;
        # whole-file hashes deliberately fail closed rather than adapting.
        patch_linux_source(source)
        subprocess.run([*args[:-1], str(source)], check=True)


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
    elif sys.platform.startswith("linux"):
        build_linux_source(args)
        print("Linux detached-native dispatcher guards applied to verified fresh source.")
    else:
        subprocess.run(args, check=True)
        print("Unpatched source-build path uses the active PyTorch headers and system toolchain.")
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
