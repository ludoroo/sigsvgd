# Compiled Signatory on patched PyTorch

## Outcome

The original **Signatory 1.2.6.1.9.0 C++ source builds unchanged against PyTorch 2.13.0**. No signature or backward algorithm port was necessary. Production `PathSigKernel` now uses this rebuilt native implementation instead of the slower pure-PyTorch candidate.

The earlier assumption that the package's Torch-1.9 version suffix required keeping that runtime was too strong: binary builds are ABI-specific, but this source remains build-compatible with the tested patched runtime.

## Build compatibility

`scripts/install_signatory.py` builds from source using the current interpreter and PyTorch, bypasses cached/prebuilt Signatory wheels, and runs a forward/backward smoke check. On macOS it uses Apple Clang plus Homebrew OpenMP headers.

Initially linking Homebrew's `libomp` caused a duplicate-OpenMP abort because modern PyTorch already bundles a runtime. The installer relinks the extension to `@loader_path/../torch/lib/libomp.dylib` and ad-hoc signs the modified extension. It does **not** suppress the error using `KMP_DUPLICATE_LIB_OK`.

`setup.py` and `environment.yaml` pin PyTorch 2.13.0 to make native ABI upgrades deliberate. Rebuild Signatory after changing PyTorch or Python. This version is outside all 12 original open Dependabot ranges; maintaining this source-build path does not guarantee immunity from future advisories.

## Correctness

Same inputs, random cotangents, shapes, dtypes, and tolerances as the [earlier validation](signature-validation.md). Reference: original Signatory on PyTorch 1.9. Candidate: the same Signatory source rebuilt on PyTorch 2.13.

**22/22 signature-value and gradient comparisons pass at both 1 and 4 threads.** This includes the high-channel float32 stress gradient that failed for the pure-PyTorch candidate. All five project signature/kernel tests also pass on the rebuilt backend. Eight additional installer tests verify accepting the pinned PyTorch (including local build suffixes) and rejecting other ABI versions: 13 tests pass in total.

For directly supplied paths, signature outputs were bit-identical in the one-thread comparison. The cos/sin transformed trajectory has small differences from PyTorch's transcendental operations; it still passes the original tolerances. Gradients pass without tolerance relaxation.

## CPU performance

Apple M4 Max, macOS 26.7.1, float32. Median milliseconds after three warmups: original 30 samples, rebuilt 50 samples. Sequential processes, fixed intra-op/OpenMP threads, otherwise same methodology as the earlier report. These are microbenchmarks, not whole-planner or convergence measurements.

One thread:

| Workload (batch × length × channels) | Depth | Original forward | Rebuilt forward | Original forward+backward | Rebuilt forward+backward |
|---|---:|---:|---:|---:|---:|
| Trajectory cos/sin `128×25×2` | 3 | 0.125 | 0.114 | 1.354 | 1.270 |
| Synthetic `100×10×2` | 4 | 0.073 | 0.067 | 0.627 | 0.621 |
| Synthetic `20×25×3` | 3 | 0.043 | 0.043 | 0.287 | 0.301 |
| Stress `8×10×7` | 4 | 0.077 | 0.080 | 0.393 | 0.402 |

These measurements were repeated after building with the reproducible installer. The repository's trajectory workload is about 6% faster including backward, versus 2.4x slower for the uncompiled candidate. The synthetic one-thread backward cases range from about 1% faster to 5% slower. An initial manual-build run showed 12–30% slowdowns on some synthetic cases, so small differences should not be treated as reliable speedups without longer repeated measurements.

Four threads:

| Workload | Original forward+backward | Rebuilt forward+backward |
|---|---:|---:|
| Trajectory cos/sin | 1.142 | 1.003 |
| Synthetic `100×10×2` | 0.538 | 0.434 |
| Synthetic `20×25×3` | 0.475 | 0.454 |
| Stress `8×10×7` | 0.594 | 0.567 |

The rebuild is modestly faster in these four-thread backward measurements. Thread scheduling, differing PyTorch threading backends, and Python versions affect results; these are observed timings, not guarantees. Float64 and p95 results are retained in JSON.

## Reproduce

After the documented environment/prerequisite installation:

```sh
python scripts/install_signatory.py
MPLBACKEND=Agg python -m pytest tests -q
python scripts/validate_path_signature.py --backend signatory \
  --threads 1 --repeats 50 \
  --inputs /tmp/sigsvgd-signature-validation/signatory-t1.npz \
  --output /tmp/sigsvgd-signature-validation/signatory-modern-t1.npz
python scripts/validate_path_signature.py --compare \
  /tmp/sigsvgd-signature-validation/signatory-t1.npz \
  /tmp/sigsvgd-signature-validation/signatory-modern-t1.npz
```

Repeat with thread count 4 and matching reference file. The legacy reference generation/environment is described in the earlier report. `--backend local` now loads the historical pure-PyTorch candidate from `scripts/local_path_signature.py`; production no longer uses that implementation.

Raw results: `docs/signature-validation/signatory-modern-t{1,4}.json` and `comparison-signatory-modern-t{1,4}.txt`. NPZ arrays remain in `/tmp/sigsvgd-signature-validation/` and are regenerable.

## Limits

The results above describe Apple Silicon CPU only. See [Linux/CUDA validation](linux-cuda-validation.md) for the separate Linux L4 run, which reports performance regressions and a float32 stress VJP tolerance failure. A direct native Signatory call on an Apple MPS tensor segfaulted; the project's wrapper now rejects non-CPU/non-CUDA devices before entering native code. Full robotic experiments and long optimization convergence remain unverified. Linux CUDA allocator peaks are recorded separately; CPU and whole-process peak memory remain unmeasured. Original Signatory is not a newly maintained upstream library; rebuilding it solves this compatibility/performance problem but leaves ongoing dependency maintenance with this project.
