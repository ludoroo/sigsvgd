> Historical decision superseded: four broad v2 Linux guards are now adopted
> after independent public alias/storage/mutation review. Private version/view/
> base fields are not preserved. Original rejection evidence below is retained;
> see [current status](linux-cuda-optimization.md).

# Autograd-only dispatcher guard: REJECTED

The isolated narrower guard fails this experiment's conservative exact-metadata
gate. No production edits, installation into existing environments or adoption.
See the [previous broad-guard experiment](linux-cuda-guard-experiment.md) and
[profiling investigation](linux-cuda-profiling.md).

Fresh source SHA256 was verified as
`c9ab17df6286688c1c7b12c0e906031a1fe7681dfca551876c1ff616e12807d8`.
A fresh wheel was built in `/tmp/sigsvgd-autograd-guard` using candidate Python,
MAX_JOBS=1, pip wheel --no-cache-dir --no-deps --no-build-isolation,
and installed with --target into its isolated overlay. The only source changes
are two inserted `at::AutoDispatchBelowAutograd autograd_only_guard;` lines at
the prior detach anchors. The installed torch header confirms this guard excludes
`c10::autograd_dispatch_keyset`, not ADInplaceOrView.

The reused control overlay passed wheel RECORD SHA256 validation and all four
known wheel/native hashes from the previous experiment. Both fresh safety runs
asserted overlay import and used PYTHONDONTWRITEBYTECODE=1, sequentially.

## Stop gate

All 253 records were compared recursively without normalization. All six
gradchecks pass in each overlay. There are **720 field differences**, so the
required zero-difference gate fails. Fresh control versus previous control has
zero differences; fresh control versus previous broad guard has exactly 1,694,
confirming comparator sensitivity. An example is the initial-signature gradient
version: control 0, narrower guard 1 (also repeated/accumulated observations).
See compact [measured summary](signature-validation/autograd-guard-experiment/summary.json).

These are observed differences in private `Tensor._version` metadata, not
wrong gradient values: all recorded numerical values, view flags and base
presence match. The narrower guard eliminates the broad guard’s recorded
view/base differences, while introducing different version increments. No
public-behavior failure or numerical-correctness failure has been proven.
Rejection here means failure of the predeclared strict zero-field-difference
gate, not evidence of incorrect gradients.

Per that strict stop rule, no subsequent canonical NPZ, legacy,
alias-demo, pytest, trace, split, ABBA or timing runs were performed. There is no
performance or viability claim: **reject**. No empty trace result is accepted
as evidence; this experiment produced no traces.

## Reproduction and evidence

Scripts, exact patch and raw SHA256 manifest are in
[autograd-guard-experiment](signature-validation/autograd-guard-experiment/).
Full historical build logs, wheels, sources and safety JSON remain under
`/tmp/sigsvgd-autograd-guard`; large arrays are not tracked. Historical
measurement reused the integrity-verified previous control binary. The updated
reproduction builds **both** control and new guard from the same source into a
fresh root; rebuilt wheels need not be byte-identical to historical wheels.
It does not modify the previous experiment or measured raw files.

Requires candidate Python with its existing torch/CUDA toolchain, available
CUDA GPU (safety covers both CPU and CUDA), compiler/build dependencies and
the tracked `docs/signature-validation/guard-experiment/{control,patched}-safety.json`
files for deterministic-control and sensitivity checks. Missing prerequisites
fail closed rather than skipping checks. Run from repository root:

```sh
mkdir -p /tmp/signatory-src
curl -fL https://files.pythonhosted.org/packages/source/s/signatory/signatory-1.2.6.1.9.0.tar.gz \
  -o /tmp/signatory-src/signatory-1.2.6.1.9.0.tar.gz
PYTHONDONTWRITEBYTECODE=1 /home/coder/.venvs/sigsvgd-candidate/bin/python \
  docs/signature-validation/autograd-guard-experiment/reproduce.py \
  /tmp/sigsvgd-autograd-guard-reproduction
```

Destination must not exist. `build.py FRESH_ROOT` can also be run separately;
it verifies the source hash, builds sequentially and installs only into target
overlays. `reproduce.py` asserts overlay imports, runs both complete safety
observations with bytecode disabled and checks the **exact** tracked summary
(720 version-only differences), historical control equality (0) and broad
sensitivity (1,694). It writes full differences under the fresh root and stops;
it does not run performance measurements after the failed strict gate.

To inspect existing results independently:

```sh
python3 docs/signature-validation/autograd-guard-experiment/compare.py \
  /tmp/sigsvgd-autograd-guard/control-safety.json \
  /tmp/sigsvgd-autograd-guard/patched-safety.json \
  --diff-output /tmp/autograd-guard-differences.json
```

Comparator exit codes: 0 equal, 1 different, 2 invalid input. It checks record
count, all six named gradchecks, duplicate/missing keys and unequal list lengths
without assertions, including under `python -O`. No normalization is performed.

## Limitations / protocol deviation

Before/after environment freezes and installed-native snapshots were **not
captured**. Thus unchanged environments cannot be independently certified by
this run's before/after evidence, although all installation commands targeted
the isolated overlay. This is an incomplete protocol run, not adoption evidence.
Native-throw restoration remains untested: prior malformed depth-zero native
calls crashed both variants and were not rerun. Ordinary safety observations
do not establish exception-path restoration or broad workload compatibility.

## Helper checks after review

Without rebuilding or rerunning GPU observations, the comparator reproduced the
exact tracked summary and historical 0/1,694 counts in normal and `-O` modes.
Missing top-level/nested keys, failed gradcheck, unequal list length and malformed
JSON each returned exit 2 in both modes. All helper scripts compiled in memory;
`git diff --check` passed. Full fresh-build reproduction remains unexecuted.
Temporary malformed-input fixtures were removed automatically; no measured raw
evidence was changed.
