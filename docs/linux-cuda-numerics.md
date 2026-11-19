# Linux CUDA numerical diagnostic

Diagnostic scripts and all-element aggregate metrics are in
`signature-validation/chunking-diagnostic/`. No production code or tolerance changed.
Exact float32 inputs and cotangents came from
`/tmp/sigsvgd-linux-validation/legacy-cpu-t1.npz`; basepoint=True matches
`validate_path_signature.py`. TF32 was disabled. All CUDA calls really executed;
there is no availability skip.

The four main shapes were run on candidate CPU threads 1 and 24, CUDA default,
and CUDA with `_signature_batch_trick` returning None (diagnostic only).
The same float32 inputs/cotangents were promoted to float64 for CPU t1 reference;
this is a higher-precision proxy, not an exact oracle. For the trigonometric
trajectory case, cos/sin are also evaluated in float64. Existing legacy CPU/GPU
and original candidate GPU artifacts supply the old/new baseline comparisons.
No BF16 measurements or conclusions are involved.

## Findings

Original directional allclose thresholds remain rtol=1e-4, atol=1e-5.
The comparison follows `np.allclose(reference, candidate)`, scaling the
tolerance by the candidate (second argument).
Stress VJP (560 elements), compared with old CPU:

| variant | failing elements | max absolute difference |
|---|---:|---:|
| candidate CPU t1 | 0 | 1.526e-5 |
| candidate CPU t24 | 2 | 3.014e-4 |
| candidate CUDA default | 2 | 2.518e-4 |
| candidate CUDA no-chunk | 3 | 2.441e-4 |
| legacy CUDA | 3 | 2.365e-4 |

Candidate default versus legacy GPU still fails one element (max abs 6.866e-5).
Disabling chunking also fails one versus legacy GPU, and does **not** repair the
strict CPU baseline comparison. Therefore changing production chunking is not
supported by this experiment, even on correctness grounds alone.

Source `_signature_batch_trick` uses CPU thread count or CUDA threshold 2048,
with mult=min(round(threshold/batch), floor(stream/3), round(sqrt(stream))).
For stress this gives no split at CPU t1 and mult=3 at CPU t24 and CUDA.
CPU t24 changes both thread count and chunking, including Chen products,
so it does not isolate grouping alone. The six signature/VJP arrays for the
other three shapes remain bit-identical between CPU t1 and t24 (those shapes
remain unsplit), an indirect control consistent with a chunking contribution.
CUDA no-chunk still differs from CPU t1, so chunking
alone does not explain the CUDA differences; backend arithmetic also matters.
This is bounded evidence, not attribution to a particular compiler instruction.
Candidate and legacy default GPU signatures are bit-identical across all four
shapes. Their old/new VJP differences arise in backward computation, not from
a changed chunk configuration.

Against promoted float64, stress VJP candidate default has zero failing elements,
max abs 5.244e-5 and L2 error 2.997e-4; legacy GPU has one failing element,
max abs 6.142e-5 and L2 error 3.463e-4. Legacy CPU itself fails three elements; candidate CPU t1 and t24 each also
fail three stress VJP elements against the float64 proxy.
Thus the stress result supports a baseline floating-point limitation rather than
uniform degradation in the candidate. It does not establish numerical equivalence.

**Counterexamples to “candidate is never worse”:** channels VJP max absolute error
is 7.700e-6 versus legacy GPU 7.636e-6; representative VJP L2 error is
8.258e-5 versus 8.245e-5; trajectory VJP max error is 2.197e-6 versus
2.100e-6 (and slightly larger L2). All pass the fixed high-precision comparison.
Candidate CUDA passes all eight signature/VJP float64-proxy comparisons, but
these aggregate facts cannot guarantee elementwise dominance or other inputs.

## Reproduce

From repository root, using the existing candidate environment:

```sh
ROOT=/tmp/sigsvgd-linux-validation
# If baseline artifacts are absent, regenerate them first (includes benchmarks):
# OUT="$ROOT" bash docs/signature-validation/linux-reproduce.sh
P=docs/signature-validation/chunking-diagnostic
PY=/home/coder/.venvs/sigsvgd-candidate/bin/python
$PY $P/run.py gpu --input "$ROOT/legacy-cpu-t1.npz" --output "$ROOT/chunking-gpu.npz"
$PY $P/run.py cpu --input "$ROOT/legacy-cpu-t1.npz" --output "$ROOT/chunking-cpu.npz"
$PY $P/summarize.py --root "$ROOT"
```

The summarizer requires existing legacy CPU/GPU and candidate GPU original NPZ
artifacts in that directory. Metadata JSON records actual versions/device.
Future runs also record input path and SHA-256; historical metadata has not
been backfilled with fields that were not captured at execution time.
