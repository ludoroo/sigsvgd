# Bounded installed-Signatory overhead investigation

The corrected, regenerated runs confirm a regression: candidate/reference
forward+backward medians are 1.07–1.22× CPU and 1.36–1.43× CUDA.
All profiling uses the validation harness's basepoint=True; earlier missing-
basepoint measurements were discarded. No algorithm or tolerance was changed.

## Interpretation and limits

Most measured CUDA regression is inside native host calls (see split deltas
below), consistent with native ATen dispatch/launch costs rather than solely
Python/autograd. Native timers contain dispatch, launch and incidental waits;
no synchronization occurs inside a native-call timer. Residual includes Python,
autograd, transformation and instrumentation overhead. This does not identify
a specific dispatcher, flag or runtime component as the cause.

Both Kineto captures contain actual GPU kernels. Reprocessing the existing
traces case-insensitively (`Kernel` in torch 1.9, `kernel` in modern torch)
shows no observed kernel-duration regression. These instrumented sums across
different profiler/runtime versions do not establish a precise GPU speedup.
On CUDA, together with split results, native host dispatch/launch overhead is the
dominant observed bottleneck. On CPU, both native and residual time contribute;
native time includes computation, and residual growth can exceed native growth.
CUDA event spans include host launch gaps, not just compute.
CPU operator counts/self times are instrumented evidence, not wall timings.
Four small add/multiply/clone/sin controls supply narrow overhead measurements,
not a universal per-launch estimate. No unsupported extrapolation is made.

Installed Python source hashes match. JSON records actual native binary hashes,
readelf DW_AT_producer flags, ldd and loaded /proc/self/maps library paths.
Both native builds report GCC 11.4/O3; reference C++14 versus candidate C++20,
frame-pointer and protection flags differ. Python/torch/CUDA also differ.
These are confounders, not demonstrated causes. No rebuilds, flag changes,
installed-environment changes or production changes were made.

## Reproduce and artifacts

Run from checkout root with the installed environments documented in
[validation](linux-cuda-validation.md). For each device `cpu cuda`, run timing
in order `candidate reference reference candidate`, labelling rounds 1/2:

```sh
taskset -c 2 /home/coder/.venvs/sigsvgd-candidate/bin/python \
  scripts/profile_signatory_overhead.py \
  --inputs /tmp/sigsvgd-linux-validation/legacy-cpu-t1.npz \
  --device cuda --mode timing --output /tmp/candidate-cuda-timing.json
```

Substitute reference interpreter as appropriate. Run modes `split`, `profile`,
`microbench` separately for each interpreter/device, sequentially. Defaults are
20 warmups/100 repeats; profile captures three iterations after 20 warmups.
Timing uses synchronized wall measurements and a **separate** CUDA event-span
loop. Input SHA256 is recorded, so substituted data can be detected.

Tracked raw sample arrays and summaries are in `signature-validation/profiling/`;
traces are `/tmp/signatory-{torch_version}-{device}-{workload}-{phase}.json`,
not tracked. All modes were regenerated after fixing basepoint and enabling Kineto. No other
benchmarks were active. Full `pytest -q` was attempted but collection fails on
three robotics examples (missing optional dependencies, including pybullet).
`pytest tests -q`: 30 passed, 1 skipped. No temporary probe tests retained.
Numerical failures documented in validation remain unresolved and unchanged.


## Corrected basepoint=True measurements

All artifacts were regenerated with basepoint=True, explicit no-grad forward, OMP_NUM_THREADS=1 before torch import and one torch thread. Metadata records actual Python/CUDA/device/TF32 settings. Timing rounds form ABBA; values below are forward+backward milliseconds (round 1 / round 2). Native/residual deltas are candidate minus reference per call, from the separate instrumented split run.

|Device|Workload|Reference wall|Candidate wall|Native delta|Residual delta|
|---|---|---:|---:|---:|---:|
|cpu|traj_128x25x2_f32|2.337 / 2.350|2.517 / 2.519|0.075|0.072|
|cpu|representative_100x10x2_f32|1.213 / 1.214|1.370 / 1.380|0.025|0.065|
|cpu|channels_20x25x3_f32|0.702 / 0.702|0.853 / 0.857|0.069|0.065|
|cpu|high_channel_stress_8x10x7_f32|0.893 / 0.890|1.021 / 1.016|0.044|0.067|
|cuda|traj_128x25x2_f32|2.312 / 2.252|3.146 / 3.191|0.655|0.188|
|cuda|representative_100x10x2_f32|2.563 / 2.497|3.513 / 3.572|0.728|0.202|
|cuda|channels_20x25x3_f32|2.177 / 2.120|3.033 / 3.010|0.727|0.165|
|cuda|high_channel_stress_8x10x7_f32|2.556 / 2.495|3.542 / 3.491|0.746|0.208|

### Kineto kernel activities

Existing traces were reprocessed; no timings were rerun. Counts and duration sums cover three instrumented forward+backward iterations.

|Workload|Reference kernels / ms|Candidate kernels / ms|
|---|---:|---:|
|traj_128x25x2_f32|786 / 1.427|789 / 1.365|
|representative_100x10x2_f32|870 / 1.506|855 / 1.455|
|channels_20x25x3_f32|759 / 1.314|750 / 1.273|
|high_channel_stress_8x10x7_f32|870 / 1.542|855 / 1.477|

### ATen count evidence

Counts below sum recorded CPU operator events over three forward+backward iterations. Categories are exact names, nested events are not independent work.

|Device|Environment|Workload|empty|copy_|mul|add_|narrow|
|---|---|---|---:|---:|---:|---:|---:|
|cpu|reference|traj_128x25x2_f32|30|18|18|3|54|
|cpu|reference|representative_100x10x2_f32|27|18|15|3|60|
|cpu|reference|channels_20x25x3_f32|27|18|12|3|48|
|cpu|reference|high_channel_stress_8x10x7_f32|27|18|15|3|60|
|cpu|candidate|traj_128x25x2_f32|24|27|18|6|54|
|cpu|candidate|representative_100x10x2_f32|24|33|15|3|60|
|cpu|candidate|channels_20x25x3_f32|24|27|12|3|48|
|cpu|candidate|high_channel_stress_8x10x7_f32|24|33|15|3|60|
|cuda|reference|traj_128x25x2_f32|99|108|54|150|243|
|cuda|reference|representative_100x10x2_f32|150|135|42|132|312|
|cuda|reference|channels_20x25x3_f32|96|108|48|150|237|
|cuda|reference|high_channel_stress_8x10x7_f32|150|135|42|132|312|
|cuda|candidate|traj_128x25x2_f32|39|216|54|153|243|
|cuda|candidate|representative_100x10x2_f32|63|294|42|138|312|
|cuda|candidate|channels_20x25x3_f32|39|216|48|150|237|
|cuda|candidate|high_channel_stress_8x10x7_f32|63|294|42|138|312|

### Small-operation controls

Median synchronized wall microseconds, not inferred launch or compute costs.
CUDA values include synchronization overhead; allocator/cache and dispatcher
behavior also contribute. These controls support investigating host overhead,
but cannot be multiplied by signature operator counts to predict its cost.

|Device|Operation|Reference µs|Candidate µs|
|---|---|---:|---:|
|cpu|add|1.941|2.150|
|cpu|mul|1.940|2.090|
|cpu|clone|1.910|2.260|
|cpu|sin|11.531|10.135|
|cuda|add|11.596|17.176|
|cuda|mul|11.436|16.765|
|cuda|clone|14.910|18.311|
|cuda|sin|10.581|15.051|

### Provenance and exact output naming

`readelf` (binutils) and `ldd` (libc tooling) must be on PATH. Native provenance
comes from installed extension binaries, never repository CMake configuration.
Historical measurements did not record measurement-time script/harness hashes
or explicit per-workload configuration fields. This is a provenance limitation;
new runs record both. Profile artifacts record the reprocessor hash, trace
hashes and pre-edit source hashes captured immediately before this category fix;
those pre-edit hashes are not claimed to have been captured at measurement time.
No new measurement hashes were backfilled into historical timing artifacts.

The following reproduces exact artifact names without overwriting tracked data.
Profile mode does overwrite the fixed `/tmp/signatory-…json` trace paths, so copy
those traces elsewhere first if preserving the recorded trace hashes is needed:

```sh
OUT=/tmp/signatory-profile-reproduction
mkdir -p "$OUT"
for device in cpu cuda; do
  for round in 1 2; do
    order="candidate reference"
    [ "$round" = 1 ] || order="reference candidate"
    for env in $order; do
      taskset -c 2 /home/coder/.venvs/sigsvgd-$env/bin/python \
        scripts/profile_signatory_overhead.py \
        --inputs /tmp/sigsvgd-linux-validation/legacy-cpu-t1.npz \
        --device "$device" --mode timing \
        --output "$OUT/$env-$device-timing-$round.json" || exit
    done
  done
  for env in candidate reference; do
    for mode in split profile microbench; do
      taskset -c 2 /home/coder/.venvs/sigsvgd-$env/bin/python \
        scripts/profile_signatory_overhead.py \
        --inputs /tmp/sigsvgd-linux-validation/legacy-cpu-t1.npz \
        --device "$device" --mode "$mode" \
        --output "$OUT/$env-$device-$mode.json" || exit
    done
  done
done
```
