#!/usr/bin/env python3
"""Benchmark and numerically validate path signatures against Signatory.

The local implementation is loaded by extracting the exact ``_path_signature``
function AST from scripts/local_path_signature.py and executing it with only a
``torch`` namespace. This intentionally avoids importing the package's unrelated
sigkernel dependency, which is unavailable in some legacy Signatory environments.
"""

import argparse
import ast
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

np = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("signatory", "local"))
    parser.add_argument(
        "--output", type=Path, help="output .npz path (timings go beside it as .json)"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        help="reuse exact inputs/cotangents from a reference .npz",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("REF.npz", "CAND.npz"),
        help="compare saved outputs instead of running a benchmark",
    )
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("--threads and --repeats must be positive")
    if not args.compare and (not args.backend or not args.output):
        parser.error("--backend and --output are required unless --compare is used")
    return args


def load_local_signature(torch):
    source = Path(__file__).resolve().with_name("local_path_signature.py")
    tree = ast.parse(source.read_text(), filename=str(source))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_path_signature"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(isolated), str(source), "exec"), namespace)
    return namespace["_path_signature"]


def make_cases():
    rng = np.random.RandomState(1729)
    cases = []
    # The first shape/transformation mirrors tests/test_traj_kernels.py: a 1-D
    # trajectory mapped to cos/sin channels. Its gradient is with respect to the
    # original 1-D input, through that transformation.
    for name, shape, depth, transform in (
        ("traj_128x25x2", (128, 25, 1), 3, "cos_sin"),
        ("representative_100x10x2", (100, 10, 2), 4, None),
        ("channels_20x25x3", (20, 25, 3), 3, None),
        ("high_channel_stress_8x10x7", (8, 10, 7), 4, None),
    ):
        for dtype in (np.float32, np.float64):
            x = rng.standard_normal(shape).astype(dtype)
            upstream_rng = np.random.RandomState(8675309)
            # Cotangent dimension is determined from depth and transformed channels.
            channels = 2 if transform else shape[-1]
            out_size = sum(channels**d for d in range(1, depth + 1))
            cotangent = upstream_rng.standard_normal((shape[0], out_size)).astype(dtype)
            cases.append(
                {
                    "name": name + ("_f32" if dtype == np.float32 else "_f64"),
                    "input": x,
                    "cotangent": cotangent,
                    "depth": depth,
                    "transform": transform,
                    "kind": "main",
                }
            )

    # Small correctness-only cases exercise identity/degenerate paths and strides.
    edge_rng = np.random.RandomState(314159)
    raw = edge_rng.standard_normal((2, 3, 5)).astype(np.float64)
    cases.extend(
        [
            {
                "name": "single_point",
                "input": edge_rng.standard_normal((2, 1, 2)).astype(np.float64),
                "depth": 3,
                "transform": None,
                "kind": "edge",
            },
            {
                "name": "constant_path",
                "input": np.repeat(edge_rng.standard_normal((2, 1, 2)), 5, axis=1),
                "depth": 3,
                "transform": None,
                "kind": "edge",
            },
            {
                "name": "noncontiguous_path",
                "input": raw.transpose(0, 2, 1),
                "depth": 2,
                "transform": None,
                "kind": "edge",
            },
        ]
    )
    for case in cases:
        if "cotangent" not in case:
            channels = case["input"].shape[-1]
            size = sum(channels**d for d in range(1, case["depth"] + 1))
            case["cotangent"] = (
                np.random.RandomState(8675309)
                .standard_normal((case["input"].shape[0], size))
                .astype(case["input"].dtype)
            )
    return cases


def run_backend(args, torch):
    if args.backend == "local":
        signature = load_local_signature(torch)
    else:
        import signatory

        signature = lambda path, depth: signatory.signature(path, depth, basepoint=True)

    torch.set_num_threads(args.threads)
    cases = make_cases()
    if args.inputs:
        with np.load(args.inputs, allow_pickle=False) as reference:
            for case in cases:
                # Keep the noncontiguous test's strides while sharing identical
                # floating-point inputs across NumPy versions/platforms.
                shared = np.empty_like(case["input"], order="K")
                shared[...] = reference[case["name"] + "__input"]
                case["input"] = shared
                case["cotangent"] = reference[case["name"] + "__cotangent"].copy()
    arrays = {}
    timing_cases = {}
    for case in cases:
        name, depth = case["name"], case["depth"]
        x_np = case["input"]
        # Preserve the deliberately non-contiguous edge input's strides.
        x = torch.from_numpy(x_np).requires_grad_(True)

        def path_from(value):
            return (
                torch.cat((value.cos(), value.sin()), dim=-1)
                if case["transform"] == "cos_sin"
                else value
            )

        cotangent = torch.from_numpy(case["cotangent"].copy())
        with torch.enable_grad():
            output = signature(path_from(x), depth)
            gradient = torch.autograd.grad((output * cotangent).sum(), x)[0]
        output_np, gradient_np = (
            output.detach().cpu().numpy(),
            gradient.detach().cpu().numpy(),
        )
        if not np.isfinite(output_np).all() or not np.isfinite(gradient_np).all():
            raise ValueError("non-finite signature or VJP in case " + name)
        arrays[name + "__signature"] = output_np
        arrays[name + "__vjp"] = gradient_np
        arrays[name + "__input"] = x_np
        arrays[name + "__cotangent"] = case["cotangent"]
        meta = {
            "depth": depth,
            "transform": case["transform"],
            "kind": case["kind"],
            "input_shape": list(x_np.shape),
            "input_dtype": str(x_np.dtype),
            "input_strides": list(x_np.strides),
        }
        arrays[name + "__metadata"] = np.asarray(json.dumps(meta, sort_keys=True))

        if case["kind"] == "main":

            def forward():
                with torch.no_grad():
                    return signature(path_from(x), depth)

            def forward_backward():
                leaf = x.detach().requires_grad_(True)
                result = signature(path_from(leaf), depth)
                torch.autograd.grad((result * cotangent).sum(), leaf)

            def measure(fn):
                for _ in range(3):
                    fn()
                samples = []
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    fn()
                    samples.append(time.perf_counter() - start)
                return {
                    "median_seconds": statistics.median(samples),
                    "p95_seconds": float(np.percentile(samples, 95)),
                    "samples": len(samples),
                }

            timing_cases[name] = {
                "forward_no_grad": measure(forward),
                "forward_backward": measure(forward_backward),
            }
        print("computed", name, file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    timing_path = args.output.with_suffix(".json")
    timing = {
        "backend": args.backend,
        "threads": args.threads,
        "repeats": args.repeats,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cases": timing_cases,
    }
    if args.backend == "signatory":
        from importlib.metadata import version

        timing["signatory"] = version("signatory")
    timing_path.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n")
    print("wrote", args.output, "and", timing_path)


def compare_files(ref_path, cand_path):
    rtol, atol = 1e-4, 1e-5
    failures = 0
    with np.load(ref_path, allow_pickle=False) as ref, np.load(
        cand_path, allow_pickle=False
    ) as cand:
        ref_keys, cand_keys = set(ref.files), set(cand.files)
        if ref_keys != cand_keys:
            print(
                "NPZ keys differ; missing in candidate:",
                sorted(ref_keys - cand_keys),
                "extra:",
                sorted(cand_keys - ref_keys),
            )
            return 1
        for key in sorted(ref_keys):
            a, b = ref[key], cand[key]
            if (
                key.endswith("__metadata")
                or key.endswith("__input")
                or key.endswith("__cotangent")
            ):
                if not np.array_equal(a, b):
                    print(key, "reproducibility metadata/input differs")
                    failures += 1
                continue
            # Tolerances are selected by the original input dtype in companion metadata.
            prefix = key.rsplit("__", 1)[0]
            meta = json.loads(str(ref[prefix + "__metadata"]))
            rt, at = (1e-9, 1e-10) if meta["input_dtype"] == "float64" else (rtol, atol)
            if a.shape != b.shape:
                print(key, "shape differs:", a.shape, b.shape)
                failures += 1
                continue
            delta = np.abs(a - b)
            max_abs = float(delta.max(initial=0))
            rel_norm = float(
                np.linalg.norm((a - b).ravel())
                / max(np.linalg.norm(a.ravel()), np.finfo(float).tiny)
            )
            passed = np.isfinite(b).all() and np.allclose(a, b, rtol=rt, atol=at)
            print(
                "{}: {} max_abs={:.6g} scaled_relative_norm={:.6g} (rtol={}, atol={})".format(
                    key, "PASS" if passed else "FAIL", max_abs, rel_norm, rt, at
                )
            )
            failures += not passed
    return int(failures != 0)


def main():
    global np
    args = parse_args()
    # Set this before importing torch: Signatory's legacy build uses libomp even
    # when torch itself reports that its OpenMP backend is unavailable.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import torch

    if args.compare:
        return compare_files(*args.compare)
    run_backend(args, torch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
