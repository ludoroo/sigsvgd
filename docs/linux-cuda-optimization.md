# Linux optimization: adopted v2, performance in the requested ballpark

Linux builds now apply four `at::AutoDispatchBelowADInplaceOrView` guards to
Signatory 1.2.6.1.9.0: signature forward/backward and multi-combine
forward/backward. macOS remains unpatched. No arithmetic, tolerance, compiler
flag, frame-pointer or architecture changes are adopted.

The user clarified that performance need only be “ballpark,” not exact parity.
On the measured NVIDIA L4 workloads, final CUDA forward medians are **2.1–4.7%**
slower than legacy and forward+backward medians **6.8–9.4%** slower. All-run
p95 ratios are within 10%. This meets that clarified performance goal, not an
unconditional numerical/performance-equivalence claim. The inherited float32
stress VJP failure remains: [numerical report](linux-cuda-numerics.md).

## Final CUDA evidence

Core 2, one torch/OMP thread, 20 warmups, **500 samples per workload/phase/run**,
serial legacy/broad/broad/legacy. Initial same-variant p95 drift reached 8.01%,
triggering the predeclared single reverse broad/legacy/legacy/broad repeat after
30 seconds idle. Reverse maximum p95 drift was 1.81%. **All eight runs remain
included**; there was no best-run selection or further repeat. Ratios use the
arithmetic mean of the four per-run quantiles per variant, not pooled quantiles.

| Workload | Forward median gap | Forward p95 gap | Forward+backward median gap | Forward+backward p95 gap |
|---|---:|---:|---:|---:|
| trajectory | 2.1% | 4.0% | 6.8% | 7.6% |
| representative | 4.3% | 3.4% | 8.1% | 7.9% |
| channels | 4.3% | 5.0% | 9.4% | 9.7% |
| stress | 4.7% | 5.9% | 8.5% | 9.5% |

**Timed build:** the isolated broad v2/cu126 overlay, not the literal
installer-built binary. Its patched C++ source hashes match the actual installer
v2 manifest exactly. Independently built binaries need not be byte-identical;
that source identity is not a claim that the installer binary itself was timed.
The actual installer-built cu126 binary passed the tests and exact canonical
checks below. Before/after timed native hashes match.

Compact initial/reverse/all-run summaries, native hashes, source-identity
checks, host inventory, original measurement manifest and reproducer are in
[`signature-validation/optimization/broad-v2/`](signature-validation/optimization/broad-v2/).
The manifest hashes raw files retained under `/tmp/sigsvgd-broad-cu126`; raw sample
arrays are not tracked. The reproducer refuses overwrite and requires the
specified interpreters, overlays and canonical input. Run it only on a released,
idle GPU, with a fresh output root; reverse only under its declared drift rule.

## Installer and reviewed safety scope

`python scripts/install_signatory.py` downloads a fixed PyPI URL directly from
`files.pythonhosted.org`, bypassing pip mirror selection for source retrieval.
Archive SHA256 is checked:
`c9ab17df6286688c1c7b12c0e906031a1fe7681dfca551876c1ff616e12807d8`.
The sole active patch is
`scripts/patches/signatory-1.2.6.1.9.0-linux-autograd-v2.json`. Whole-file input
hashes, exact unique anchors and patched output hashes fail closed before writes.
Pip builds against active Python/torch, without dependency resolution, build
isolation or cached/prebuilt Signatory wheels. Safe tar extraction needs Python
3.10.12+, 3.11.4+, or newer supported Python. Download/file failures are explicit.

Detached native inputs are read-only; in-place writes target private allocations.
No returned shared storage requires shared version tracking: the basepoint
gradient's private base is unreachable. Python-created views are outside the
guard. Raw OpenMP workers do not inherit this thread-local guard. Future source
changes require renewed alias/mutation review, enforced by exact hashes.

**Private `_version`, `_base` and view flags are not preserved.** The 253-record
broad/narrow comparison retains 2,470 differences: 2,278 private metadata fields
and 192 error-message version-number offsets, no public differences. Independent
review d8680bad passed public storage/alias mutation and native-branch/Path/
logsignature checks. Native malformed-call restoration is not independently
established: previous malformed calls crashed both builds, so are not safe tests.

## Validation

- Fresh actual installer v2/cu126 and isolated v2/cu130: **97 passed, 1 skipped**
  each across the complete `tests/` suite.
- Public tests include values and versions of callers/cotangents/outputs,
  gradcheck, initial plus basepoint, direct multi-combine, saved-gradient alias
  mutation, and isolation between returned gradients. A 48-case CPU matrix
  covers stream, inverse, scalar term, false/true/tensor basepoint and optional
  tensor initial. Values are checked because versions alone cannot detect native
  writes under the broad guard.
- Installer tests cover wrong archive/source/output hashes, patch conflicts,
  validation-before-write, Linux/macOS routing and extraction/download errors.
- Installer-built canonical NPZs: **55 keys exactly equal** to original unmodified
  control separately on CPU1, CPU4 and CUDA1.
- Legacy fixed-tolerance comparisons: **22/22 CPU1, 22/22 CPU4, 21/22 CUDA1**;
  only the documented stress VJP fails. Tolerances are unchanged.

Full robotics/example collection has known unavailable optional dependencies;
other GPUs, whole-planner performance and convergence are not validated.

## Optional measured Linux GPU runtime

The installer preserves the torch 2.13.0 pin and never silently selects CUDA.
The final measured runtime is same-version cu126, not a torch downgrade:

```sh
python -m pip install 'torch==2.13.0+cu126' --index-url https://download.pytorch.org/whl/cu126
python scripts/install_signatory.py
```

Use an isolated environment and rebuild after changing torch or Python. Actual
wheel-index checks found no torch 2.13 cu128 wheel. Changing CUDA libraries
together supports a runtime contribution, not a proven individual BLAS cause.

## Historical evidence

The original broad rejection used strict private-metadata and short-run tail
gates. It is superseded by public-contract review and the final all-run protocol
under the clarified performance target; its data remain in the
[broad report](linux-cuda-guard-experiment.md). The
[narrow report](linux-cuda-autograd-guard-experiment.md) retains E1's strict
720-offset rejection. E2 narrow guards and E3 frame-pointer experiments are not
current production choices. Earlier timing/build records remain in
`signature-validation/optimization/`; their worse gaps are historical, not the
current result.

CPU1/CPU4 proposal measurements compare **broad versus narrow cu126**, whereas
final CUDA measurements above compare **broad cu126 versus legacy**. Do not mix
those baselines. CPU broad/narrow medians improved; detailed proposal statistics
remain tracked. No cross-device or universal no-regression claim is made.
