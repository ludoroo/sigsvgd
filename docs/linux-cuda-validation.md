# Linux CPU and NVIDIA CUDA validation

## Assessment

The rebuilt native implementation runs on Linux CPU and NVIDIA L4, but **performance equivalence is not established: the candidate is slower**. CPU old/new comparisons pass 22/22 at both thread counts. All four comparisons involving CUDA fail the float32 high-channel stress VJP at the unchanged tolerance; each passes the remaining 21/22 checks, including all values and float64 VJPs. Even old CPU versus old CUDA fails. Do not describe this as general numerical equivalence or a transparent research-workload upgrade.

Float32 tolerance is rtol=1e-4, atol=1e-5; float64 is 1e-9, 1e-10. Both TF32 switches are off. Maximum stress VJP differences are 0.000251770 (old CPU/new CUDA), 0.000251770 (new CPU/new CUDA), 0.000236511 (old CPU/old CUDA), and 0.0000686646 (old CUDA/new CUDA). These are consistent with backend accumulation differences but do not prove harmless optimization behavior. No promoted-input float64 diagnostic was performed in this run.

## Environment and reproduction

Host: Ubuntu 22.04, Linux 6.8 x86_64, AMD EPYC 7R13, 32 logical CPUs; NVIDIA L4 23034 MiB, driver 590.48.01. Exact host/compiler inventory and package freezes are in `signature-validation/linux-host.txt` and `linux-{candidate,reference}-freeze.txt`. Candidate uses Python 3.11.17 / torch 2.13.0+cu130; isolated synthetic-input-only reference uses Python 3.9.25 / torch 1.9.0+cu111. Both use freshly source-built Signatory 1.2.6.1.9.0. The initially attempted PyTorch 1.9 cu102 wheel could not execute on L4 (no kernel image); cu111 successfully executes forward/backward. These are different CUDA runtimes, not an isolated compiler-only comparison.

### Environment setup

The captured compiler is `g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0`.
Install `uv`, Git and the system C++ toolchain first. From the checkout root:

```sh
uv python install 3.11.17 3.9.25
uv venv --seed --python 3.11.17 "$HOME/.venvs/sigsvgd-candidate"
uv venv --seed --python 3.9.25 "$HOME/.venvs/sigsvgd-reference"
NEW="$HOME/.venvs/sigsvgd-candidate/bin/python"
REF="$HOME/.venvs/sigsvgd-reference/bin/python"
"$NEW" -m pip install 'torch==2.13.0'
"$NEW" -m pip install setuptools wheel packaging 'numpy==2.4.6' cython pytest
# Forces an uncached source build against the active candidate torch headers.
"$NEW" scripts/install_signatory.py

# Vulnerable legacy runtime: synthetic inputs only; never load untrusted files.
"$REF" -m pip install 'setuptools<70' wheel packaging 'numpy<2'
"$REF" -m pip install 'torch==1.9.0+cu111' \
  --find-links https://download.pytorch.org/whl/torch_stable.html
"$REF" -m pip install --no-deps --no-cache-dir --no-build-isolation \
  --no-binary signatory --force-reinstall 'signatory==1.2.6.1.9.0'

# Core project tests, not the complete robotics stack.
"$NEW" -m pip install 'gpytorch==1.11' scipy matplotlib seaborn pandas \
  plotly altair PyYAML tqdm
"$NEW" -m pip install --no-build-isolation \
  'sigkernel @ git+https://github.com/crispitagorico/sigkernel@3b2373982e12b3d499a80228311a04debcc1bea1'
```

The source-build command used internally by the candidate installer is
`python -m pip install --no-deps --no-cache-dir --no-build-isolation --no-binary signatory --force-reinstall signatory==1.2.6.1.9.0`.
The installer additionally checks the torch pin and CPU forward/backward.
The freezes were captured with `python -m pip freeze --all` (including setuptools) in each environment and record the package versions; unpinned dependency resolution
in these setup commands may change. Verify the resulting torch CUDA suffixes
against the captured JSON before comparing results on another host.

Run from repository root: `bash docs/signature-validation/linux-reproduce.sh`. The script defaults to the checkout containing it and `$HOME/.venvs` interpreters; override `REPO`, `REF`, `NEW`, `OUT`, or `ART` as needed. It includes affinity, sample counts, alternating repeats 3/4, comparisons and exit-code capture. Benchmark failures stop immediately; expected comparison exits are 0 for CPU pairs and 1 for CUDA pairs. The pytest log was refreshed with `-rs`, and repeat3/4 comparison logs were captured from existing NPZ files without rerunning timings. The comparator requires exactly 22 result lines: CPU must have 22 PASS and exit 0; CUDA must have 21 PASS, only the documented stress VJP FAIL, and exit 1. It rejects tracebacks, metadata mismatches, empty logs and extra output. This checks reproduction of documented failures, **not acceptance success**. Artifacts default to `$OUT/artifacts`, avoiding overwriting tracked evidence. CPU uses taskset core 2 at one thread and cores 2-5 at four threads, 50 samples; CUDA uses core 2 and one CPU thread, 200 synchronized samples. Three warmups precede timing. Runs are sequential. Canonical `/tmp/sigsvgd-linux-validation/legacy-cpu-t1.npz` supplies identical inputs/cotangents to every other run. NPZ files are temporary, not tracked. JSON and text logs are retained here with exit codes. Comparisons intentionally return 1 for the failures above.

Timing excludes disk I/O and full planning, includes cos/sin trajectory transformation and random-cotangent backward. Only trajectory 128×25×2 depth 3 is grounded in the repository test; other workloads are synthetic. Full pytest: **25 passed, 1 skipped**, exit 0 (including all 10 parametrized GPU value/VJP tests); see linux-pytest.txt. The unavailable MPS test is skipped, not a GPU success claim. Actual GPU execution evidence is the NVIDIA L4 inventory in `linux-host.txt` (`nvidia-smi`), CUDA timing JSON recording `cuda_available: true`, `device_name: NVIDIA L4`, and allocator memory, plus those 10 passing GPU tests. A CUDA wheel tag alone would not establish GPU execution.

## Timings

Float32 milliseconds, median / p95. Initial sequential measurements. JSON contains only main-workload timings and allocator memory (including float64). Numerical comparisons, including edge cases, are in comparison text logs and the temporary NPZ arrays, not JSON.

| Device/threads | Workload | Old forward | New forward | Old forward+backward | New forward+backward |
|---|---|---:|---:|---:|---:|
| cpu/1 | channels_20x25x3_f32 | 0.154 / 0.182 | 0.206 / 0.260 | 0.706 / 0.719 | 0.859 / 0.893 |
| cpu/1 | high_channel_stress_8x10x7_f32 | 0.215 / 0.247 | 0.247 / 0.271 | 0.902 / 0.956 | 1.021 / 1.073 |
| cpu/1 | representative_100x10x2_f32 | 0.202 / 0.228 | 0.226 / 0.245 | 1.221 / 1.292 | 1.404 / 1.508 |
| cpu/1 | traj_128x25x2_f32 | 0.331 / 0.353 | 0.360 / 0.394 | 2.347 / 2.418 | 2.527 / 2.582 |
| cpu/4 | channels_20x25x3_f32 | 0.156 / 0.180 | 0.191 / 0.211 | 0.550 / 0.604 | 0.674 / 0.753 |
| cpu/4 | high_channel_stress_8x10x7_f32 | 0.135 / 0.155 | 0.161 / 0.187 | 0.685 / 0.718 | 0.801 / 0.839 |
| cpu/4 | representative_100x10x2_f32 | 0.203 / 0.225 | 0.226 / 0.244 | 0.721 / 1.001 | 0.828 / 0.908 |
| cpu/4 | traj_128x25x2_f32 | 0.323 / 0.340 | 0.367 / 0.384 | 1.273 / 1.343 | 1.394 / 1.467 |
| cuda/1 | channels_20x25x3_f32 | 0.556 / 0.572 | 0.723 / 0.750 | 2.131 / 2.204 | 3.000 / 3.099 |
| cuda/1 | high_channel_stress_8x10x7_f32 | 0.679 / 0.696 | 0.891 / 0.914 | 2.504 / 2.566 | 3.483 / 3.601 |
| cuda/1 | representative_100x10x2_f32 | 0.683 / 0.708 | 0.892 / 0.941 | 2.504 / 2.560 | 3.484 / 3.591 |
| cuda/1 | traj_128x25x2_f32 | 0.580 / 0.598 | 0.758 / 0.796 | 2.261 / 2.314 | 3.106 / 3.193 |

## Repeat measurements and limits

Material slowdowns triggered sequential repeats in alternating order: repeat3 candidate then reference, repeat4 reference then candidate for each device/thread pair. See matching JSON and run logs. Repeat1/2 used an incorrect reference interpreter path; those incomplete attempts are excluded from the deliverables (retained only under `/tmp/sigsvgd-linux-validation/excluded-artifacts`). No performance-equivalence claim is made.

`peak_allocated_bytes` includes the live allocation baseline, not just incremental operation memory. `peak_reserved_bytes` includes allocator cache retained from previous cases and is **not case-specific**, even though peak counters are reset. Neither metric measures total process memory, driver/context allocations, CPU RSS, or whole-planner peak memory. CPU memory is not measured. No full robotics installation/workflow, optimizer convergence, multi-seed stability, MPS support, other GPUs, or security advisory re-audit was established here. The fixed-tolerance failure remains unresolved; no algorithms or tolerances were changed.

Across float32 workloads, candidate/reference forward+backward median ratios in
repeat3 are 1.07–1.21× (CPU t1), 1.17–1.26× (CPU t4), and 1.39–1.42× (CUDA).
In reverse-order repeat4 they are 1.08–1.21×, 1.11–1.25×, and 1.36–1.40×.
The regression persists across order reversal.

## Profiling follow-up

See [bounded installed-runtime profiling](linux-cuda-profiling.md) for overhead attribution, raw summaries and limitations.
