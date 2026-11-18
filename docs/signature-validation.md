# Signatory replacement validation

> Historical evaluation of the uncompiled candidate. Production now uses rebuilt
> native Signatory; see [compiled validation](signatory-compiled-validation.md).

## Conclusion

The local implementation agrees closely with Signatory mathematically, but is **not a performance-equivalent replacement**. Do not accept the security migration as a transparent research-workload upgrade on this evidence alone.

All signature-value comparisons passed. All float64 gradient comparisons passed. One float32 stress workload failed the predeclared elementwise gradient tolerance; that failure is retained rather than relaxing the tolerance.

## Environment and methodology

- CPU: Apple M4 Max, arm64, 48 GiB RAM; macOS 26.7.1. No GPU measurements.
- Reference: Python 3.9.6, PyTorch 1.9.0, Signatory distribution `1.2.6.1.9.0`, NumPy 1.26.4.
- Candidate: Python 3.11.10, PyTorch 2.13.0, NumPy 2.4.6.
- Control: the same local function on the reference Python/PyTorch environment, to distinguish implementation overhead from upgrading PyTorch.
- Signatory was compiled from its original PyPI source, without algorithm changes. Apple Clang used the existing Homebrew libomp; a compiler wrapper translated `-fopenmp` to `-Xpreprocessor -fopenmp`. Build flags included `ARCHFLAGS='-arch arm64'`, libomp include/library paths, and `-lomp`.
- Runs were sequential, with both `OMP_NUM_THREADS` and PyTorch intra-op threads fixed to 1 or 4. Each timed operation had 3 warmups and 30 samples; results report median wall time, with p95 retained in JSON.
- Forward-only uses `no_grad`. Forward+backward differentiates a deterministic random cotangent contraction with respect to the input path. Timings include the cos/sin transform where applicable, not disk I/O or full SVGD/planning iterations.
- At measurement time, the benchmark extracted the exact `_path_signature` function AST from `src/kernels/_traj_kernels.py`. That historical implementation is now preserved in `scripts/local_path_signature.py` to avoid importing unrelated dependencies into the legacy environment.
- Candidate inputs and cotangents are loaded from the reference NPZ, not regenerated. A shared seed alone produced last-bit float64 differences between NumPy versions; those initial runs were superseded by exact-input reruns.

## Numerical results

Eleven cases: four shapes in both float32 and float64, plus float64 single-point, constant-path, and noncontiguous-input cases. All use `basepoint=True`. Gradients are vector-Jacobian products (VJPs), including differentiation through cos/sin for the trajectory case.

Declared tolerances:

- float32: `rtol=1e-4, atol=1e-5`
- float64: `rtol=1e-9, atol=1e-10`

Against the upgraded candidate, **21 of 22 value/VJP comparisons pass**, at both thread counts. The old-PyTorch control has the same single failing case. All reference inputs, cotangents, and case metadata match exactly.

The failure is the float32 VJP for synthetic `(8, 10, 7)`, depth 4: 3 of 560 elements exceed tolerance. Maximum absolute difference across the entire VJP is `2.13623e-4`, and relative L2 error is `1.11599e-6` (one thread). The three failing component pairs (reference, candidate) are:

- `0.29942322`, `0.29948425`
- `-0.30834007`, `-0.30841446`
- `-0.07436752`, `-0.07432175`

A diagnostic promoting these **same float32 inputs and cotangents** to float64 gives maximum VJP difference `4.26326e-13` between implementations. Relative to that float64 Signatory result:

| Float32 implementation | Max absolute VJP error | Relative L2 error |
|---|---:|---:|
| Signatory | 2.17772e-4 | 1.00806e-6 |
| Local | 8.52134e-5 | 5.06972e-7 |

This supports floating-point accumulation/cancellation as the cause, rather than a signature-formula mismatch. It does **not** establish that the differences are harmless to long optimization runs.

## Performance

Float32, one CPU thread, medians in milliseconds. Shapes are `(batch, path length, signature channels)`.

| Workload | Depth | Signatory forward | Local forward | Slowdown | Signatory forward+backward | Local forward+backward | Slowdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| `(128,25,2)`, cos/sin trajectory | 3 | 0.125 | 0.999 | 8.0x | 1.354 | 3.236 | 2.4x |
| `(100,10,2)`, synthetic | 4 | 0.073 | 0.585 | 8.0x | 0.627 | 1.919 | 3.1x |
| `(20,25,3)`, synthetic | 3 | 0.043 | 0.861 | 20.1x | 0.287 | 2.941 | 10.2x |
| `(8,10,7)`, synthetic stress | 4 | 0.077 | 1.198 | 15.5x | 0.393 | 3.222 | 8.2x |

Only the cos/sin workload is directly grounded in `tests/test_traj_kernels.py`; the others explore batch/channel/depth variation and are not claims about actual robot experiment configurations. In particular, `SignatureKernel` uses the separate `sigkernel` implementation, not the replaced Signatory call.

With four threads, upgraded-local forward slowdowns are 5.3–20.8x and forward+backward slowdowns are 2.8–6.3x across these float32 cases. More threads do not remove the regression. Float64 timings and p95 measurements are in the raw JSON files.

On the **same PyTorch 1.9 environment**, the one-thread trajectory local implementation takes 1.053 ms forward and 2.887 ms forward+backward, versus Signatory's 0.125 and 1.354 ms. The implementation itself therefore accounts for most of this regression; it is not merely the PyTorch upgrade.

## Reproduction and artifacts

Harness: `scripts/validate_path_signature.py`.

Existing temporary environments:

- `/tmp/sigsvgd-signatory-reference`
- `/tmp/sigsvgd-security-venv`

For example, from the repository root:

```sh
REF=/tmp/sigsvgd-signatory-reference/bin/python
NEW=/tmp/sigsvgd-security-venv/bin/python
OUT=/tmp/sigsvgd-signature-validation

$REF scripts/validate_path_signature.py --backend signatory \
  --threads 1 --repeats 30 --output "$OUT/signatory-t1.npz"
$NEW scripts/validate_path_signature.py --backend local \
  --threads 1 --repeats 30 --inputs "$OUT/signatory-t1.npz" \
  --output "$OUT/local-new-t1.npz"
$NEW scripts/validate_path_signature.py \
  --compare "$OUT/signatory-t1.npz" "$OUT/local-new-t1.npz"
```

The comparison intentionally exits 1 for the documented float32 stress failure. Repeat with `--threads 4`; use `$REF --backend local` for the old-PyTorch control. The harness sets `OMP_NUM_THREADS` before importing PyTorch.

Raw timing JSON and comparison text are retained in `docs/signature-validation/`. NPZ arrays and the same-input float64 diagnostic are in `/tmp/sigsvgd-signature-validation/` (temporary; regenerate using the harness if removed).

## Remaining validation

No CUDA/MPS, peak-memory, full optimizer convergence, or research-result reproduction measurements were performed. A single seed and modest workload matrix do not prove general equivalence. The replacement remains uncommitted; these results argue for investigating a maintained/ported compiled implementation before accepting the performance trade-off.
