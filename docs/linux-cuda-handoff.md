# Cloud-agent handoff: Linux and NVIDIA CUDA validation

## Task

Work on repository `https://github.com/ludoroo/sigsvgd`, branch
`fix-dependabot-native-signatory`. Validate the security upgrade on Linux CPU and
an actual NVIDIA GPU. Fix concrete compatibility issues while preserving native
Signatory performance and signature/gradient semantics. Do not downgrade PyTorch
to vulnerable versions in the project. Do not push or provision paid compute
without explicit authorization.

Read `README.md`, `docs/signatory-compiled-validation.md`, and the historical
`docs/signature-validation.md` before changing code.

## Current state

- Original project pinned `torch==1.9.0` and `signatory==1.2.6.1.9.0`.
- Twelve open Dependabot alerts concerned PyTorch; 2.13.0 was outside every
  reported vulnerable range when checked. Recheck current advisories if needed.
- Project now pins `torch==2.13.0` and retains original Signatory source. Its
  `1.9.0` package-version suffix does not prevent rebuilding against 2.13.0.
- `scripts/install_signatory.py` forces a fresh source build against the current
  interpreter/PyTorch, rejects versions other than the project pin (accepting
  local CUDA/CPU suffixes), and runs a CPU forward/backward smoke check.
- On macOS, the unchanged upstream C++ builds successfully. A duplicate OpenMP
  crash was fixed by relinking the extension to PyTorch's bundled libomp; no
  unsafe `KMP_DUPLICATE_LIB_OK` workaround is used.
- macOS CPU: 22/22 reference value/VJP comparisons pass at both 1 and 4 threads.
  Native performance is near the old stack: trajectory forward+backward about
  1.27 ms vs 1.35 ms originally. The discarded pure-PyTorch replacement was much
  slower and had one float32 stress VJP mismatch.
- Production `_path_signature` in `src/kernels/_traj_kernels.py` calls native
  Signatory. `scripts/local_path_signature.py` is ONLY the historical benchmark
  candidate; do not accidentally switch production back to it.
- Direct Signatory on Apple MPS **segfaults**. The project wrapper now rejects
  devices other than CPU/CUDA before native execution. CUDA is permitted but
  remains unverified; this guard is not a claim of CUDA support validation.
- Fifteen tests pass locally on Python 3.11 / PyTorch 2.13. Linux will skip the
  MPS-only rejection test if that device is unavailable.
- Linux has NOT been tested. A local Podman VM was started to inspect availability
  and then stopped without running Linux tests. There is no local NVIDIA GPU.
- Safe checkpoint loading, packaging fixes, and updated install documentation are
  also on this branch. Full robotics install on macOS was blocked by PyBullet's
  compiler failure; don't mistake that for a validated robotics workflow.

## 1. Verify actual resources

Record OS/architecture, Python/compiler versions, CPU/thread count, NVIDIA model,
driver, `nvidia-smi`, PyTorch version, `torch.version.cuda`, and
`torch.cuda.is_available()`. Execute and synchronize a small CUDA tensor operation.
A CUDA-tagged wheel or CPU-only cloud container is not evidence of a usable GPU.
If no GPU is available, complete Linux CPU validation and explicitly mark CUDA
blocked. Do not claim skipped GPU tests passed.

## 2. Build the candidate in an isolated environment

Use Python 3.11, a C++ toolchain compatible with the installed PyTorch headers,
and sufficient memory for compilation. Install `torch==2.13.0` (CUDA-capable build
appropriate to the host), pip, setuptools, wheel, packaging, NumPy, Cython, and
pytest, then run:

```sh
python scripts/install_signatory.py
```

The installer currently has a generic, unverified Linux path. Diagnose actual
build/import failures; keep macOS-specific flags and dylib changes off Linux.
Never reuse a Signatory wheel compiled against another PyTorch/Python ABI.

For core project tests, install the remaining dependencies from `setup.py` if
feasible. Native Signatory-only benchmark runs do not require the robotics stack.
A smaller core-test environment also needs `gpytorch==1.11`, SciPy, matplotlib,
seaborn, pandas, plotly, altair, PyYAML, tqdm, and the pinned sigkernel Git source
listed in `setup.py` (preinstall NumPy/Cython and use `--no-build-isolation`).
Use `MPLBACKEND=Agg python -m pytest tests -q` and report missing dependencies
separately rather than silently excluding failing tests.

## 3. Validate CPU and CUDA numerics

- Build a separate reference environment using Python 3.9, `torch==1.9.0`,
  `numpy<2`, and `signatory==1.2.6.1.9.0`, freshly source-built against that torch.
  **Do not use the project installer for the legacy reference:** it intentionally
  rejects vulnerable PyTorch. Keep this old runtime isolated, use synthetic
  inputs only, and never load untrusted pickle/checkpoint files in it.
- `scripts/validate_path_signature.py` already generates deterministic cases and
  random cotangents and compares saved NPZ results. Current harness is CPU-only.
  Extend it with an explicit CPU/CUDA device option, fail clearly if CUDA was
  requested but unavailable, move both inputs and cotangents to that device,
  preserve the noncontiguous test, and synchronize the GPU before/after timing.
- Generate reference NPZ files on the cloud host. The macOS `/tmp` artifacts do
  not exist in a fresh checkout; only timing JSON and comparison text are tracked.
- Use `--inputs reference.npz` for candidate runs: identical random seeds alone
  produced last-bit differences between NumPy versions. Keep `allow_pickle=False`.
- Check CPU vs reference, CUDA vs the CPU reference, and CUDA vs candidate CPU.
  Include float32/float64, single-point/constant/noncontiguous paths, and gradients
  through the cos/sin trajectory transform.
- Preserve existing tolerances (`rtol=1e-4, atol=1e-5` float32;
  `rtol=1e-9, atol=1e-10` float64). Report failures; do not silently loosen them.
- Add durable behavior-focused GPU tests, skipped only when CUDA is unavailable,
  exercising the project's wrapper/kernel as well as direct Signatory calls.

## 4. Benchmark honestly

Benchmark original and rebuilt native Signatory on the **same host/device** when
possible, with warmups, repeated forward-only and forward+backward measurements,
fixed CPU thread counts, and CUDA synchronization. Retain median/p95 timings.
Measure GPU peak memory if feasible, resetting counters between cases.

Use the existing four workload shapes/depths, especially the repository's
`128×25×2`, depth-3 cos/sin trajectory case. Distinguish synthetic stress workloads
from actual experiment configurations. Avoid concurrent benchmarks.

PyTorch 1.9 may not support the GPU's compute capability. If old-stack CUDA cannot
run, preserve a CPU reference for correctness and report candidate CUDA timings,
but explicitly state that old-vs-new GPU performance equivalence is unverified.
Do not compare old CPU timing with new GPU timing as evidence of equivalence.

## 5. Deliverables

- Reproducible Linux build/run commands and any minimal installer fixes.
- GPU-capable validation harness and focused tests.
- `docs/linux-cuda-validation.md` with exact environments, numerical outcomes,
  performance tables, limitations, and raw result artifacts (no model datasets).
- Clear separation of tested CPU, tested CUDA, and blocked/skipped work.
- Focused commits; no push without explicit permission.

Do not expand into unrelated research algorithm rewrites or claim full robotics
or optimizer-convergence validation from signature microbenchmarks alone.
