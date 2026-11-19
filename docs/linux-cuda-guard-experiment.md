# Dispatcher guard experiment: REJECTED, not adopted

The detached-backward guard improves CUDA median timing but fails the
observable-metadata and no-p95-regression gates. **Do not adopt this patch.**
No production source, installer, algorithm, tolerance or installed environment
was changed.

## Experiment and results

Two fresh Signatory 1.2.6.1.9.0 wheels were built against candidate torch from
the same sdist. The patch adds only named, function-scope
`at::AutoDispatchBelowADInplaceOrView` guards after the specified detach
operations in `signature_backward` and `signature_combine_backward`. It is not
InferenceMode. The guard invariant is: **never write in-place through an
externally owned alias within the scope**.

Sequential control/patched/patched/control measurements used identical
canonical inputs. Timing, split and profile runs were separate. Raw evidence,
source patch, hashes and scripts are in
[guard-experiment](signature-validation/guard-experiment/).

- CUDA float32 forward+backward medians improved **10.61–13.19%** across all
  four shapes and both pairs. CPU1 forward+backward medians improved
  **0.21–5.83%**, with no median regression in that phase. This statement does
  not cover forward-only timing or all CPU4 results.
- The tail gate fails: reverse-pair CPU1 trajectory forward+backward p95 rose
  **2.5587 → 2.8038 ms (+9.58%)**. CPU4 also has p95 regressions, including
  first-pair channels float64 backward **+3.36%**. Tiny CUDA forward-only p95
  regressions occur too. See `tails.txt`; these are bounded observations, not
  significance claims.
- CPU4 ABBA used cores **2–5**, four torch/OMP threads, **50 samples** per
  case. Its NPZ outputs were **not compared exactly**. The separate canonical
  CPU4 numerical runs were pinned to core 2 and are not CPU4 speed evidence.
- Every NPZ key/array from canonical control and patch CPU1, CPU4 and CUDA1
  matched the unchanged installed candidate using `np.array_equal`. Legacy
  checks retain exactly the documented CUDA float32 stress VJP failure at the
  unchanged tolerance (21/22 pass); CPU comparisons pass 22/22.
- All eight profile pairs per device have equal CPU-op and kernel-name
  sequences. CUDA captures have nonempty kernel activities. Traces and their
  preserved pre-experiment baseline remain under
  `/tmp/sigsvgd-guard-experiment/`; raw summary JSON is tracked.

The preceding copy investigation is closed: nested `baddbmm_` `copy_` is ATen
`result.copy_(self)`, an alias no-op without CUDA runtime children, measured at
roughly 0.04–0.08 µs. No copy/arithmetic replacement was attempted because an
arithmetic replacement changes rounding.

## Comparative safety: observable differences

Deterministic `safety.py` records **253 records**: 192 signature configurations,
48 combine configurations, six gradchecks, two path-mutation observations,
two native-throw status records, two restoration observations and one
four-task/two-thread smoke record. Signature configurations cover CPU/CUDA,
lengths 2/4, stream, false/true/tensor basepoint, inverse, absent/tensor initial
and scalar term. Combine covers one/two/three distinct or aliased inputs.
Records include caller/output/gradoutput values and versions before/after
backward, repeated gradients, accumulation, inference/view/base metadata,
create_graph and second-backward behavior, gradfn classes and mutation errors.
Signature, Path and logsignature gradchecks pass on both devices/overlays.

There are **1,694 field differences**: 1,566 gradient-version fields, 64 view
fields and 64 base-presence fields. Repeated observations of the same gradient
are not independent bugs. No recorded value, caller before/after version,
exception, inference-flag or gradfn-class differences were found. Differences
are retained, not normalized away.

`alias_demo.py` demonstrates relevance: the inverse/tensor-basepoint gradient
in control is a view with an exposed `_base`; mutating that base changes the
returned gradient. Patch returns a non-view without an exposed base (version
0 versus 2 in the example). This externally observable bookkeeping/alias
change fails equivalence even though ordinary gradient values match.

## Checks and limits

Both overlays passed `pytest tests -q`: **30 passed, 1 skipped**. Full pytest
collection failed on three robotics examples with missing optional dependencies.
Candidate freeze and candidate/reference native hashes match before/after;
a new reference freeze was not captured.

The historical initial probe incorrectly assumed original-path mutation must
raise. It failed identically in both overlays; current comparative observations
do not make that assumption. The malformed native depth-zero backward
segfaulted **both** overlays. It is excluded from the active safety loop;
**native-throw guard restoration remains unverified**. Subsequent ordinary
autograd and multithread observations do not establish exception-path safety.
This is not broad autograd, optimizer or robotics-workload validation.

## Reproduce into a fresh directory

Requires the documented interpreters/checkout and canonical legacy/candidate
NPZs from [validation](linux-cuda-validation.md). Download the exact sdist:

```sh
mkdir -p /tmp/signatory-src
curl -fL https://files.pythonhosted.org/packages/source/s/signatory/signatory-1.2.6.1.9.0.tar.gz \
  -o /tmp/signatory-src/signatory-1.2.6.1.9.0.tar.gz
python3 docs/signature-validation/guard-experiment/reproduce.py /tmp/guard-fresh-run
```

`build.py` verifies SHA256
`c9ab17df6286688c1c7b12c0e906031a1fe7681dfca551876c1ff616e12807d8`
before extraction and rejects existing variant directories. The entrypoint
requires a nonexistent destination, rewrites output roots, builds overlays,
asserts imports, runs sequential measurements/comparisons/safety/alias demos,
and regenerates summaries. Historical pytest/freeze logs are not automatically
regenerated. Console logs use `.txt`; measured JSON remains unchanged.

Before each profile capture, the runner clears only the eight fixed paths for
the current torch/device, then copies exactly the eight paths identified by the
produced profile JSON. Finish checks require all eight pairs per device,
nonempty CPU events and CUDA kernels, and equal sequences. Legacy checks
require exactly 22 results and the expected failure/status. Safety review
requires nonzero differences limited to version/view/base categories; alias
demos must reproduce the expected base difference. These are diagnostic gates
for reproducing a **rejected** experiment, not adoption approval.

Helper checks passed against existing artifacts in scratch output, including
injected empty/missing trace failures and zero-safety-difference rejection.
Python compilation and diff whitespace checks passed. Benchmarks were not rerun
for the helper/documentation corrections.
