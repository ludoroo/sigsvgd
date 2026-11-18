# Download resources

```sh
make -j
```

# Install

Python 3.10 or newer is required; Python 3.11 is recommended. PyTorch is pinned to
**2.13.0**, which addresses the current Dependabot advisories. Signatory is rebuilt
from source against that version. Do not reuse the original PyTorch 1.9 environment.

Create a fresh environment using `conda` and the requirements in `environment.yaml`,
then install the local project:

```sh
conda env create -f environment.yaml
conda activate sigsvgd
python -m pip install setuptools wheel packaging numpy cython
# macOS only: install OpenMP headers (also requires Xcode Command Line Tools).
# brew install libomp
python scripts/install_signatory.py
python -m pip install --no-build-isolation -e .
```

The pinned `sigkernel` source dependency needs NumPy and Cython available at build
time, hence the separate build-prerequisite installation and `--no-build-isolation`.
PyTorch is installed from PyPI rather than the discontinued PyTorch conda channel.
The conda environment includes PyBullet; building it from PyPI on Apple Silicon
may fail with newer Apple compilers.

`PathSigKernel` continues to use compiled Signatory. Although its package version
is `1.2.6.1.9.0`, the source builds against PyTorch 2.13.0; the suffix is not a
runtime requirement. The installer forces a fresh source build and, on macOS,
links the extension to PyTorch's bundled OpenMP runtime to avoid duplicate-runtime
crashes. It never enables `KMP_DUPLICATE_LIB_OK`.

Re-run `scripts/install_signatory.py` after any Python or PyTorch change; do not
reuse an extension built against another PyTorch ABI. The PyTorch pin makes that
upgrade an explicit maintenance step. The installer was validated on Apple Silicon;
Linux and CUDA builds remain unverified. Apple MPS tensors are rejected because
upstream Signatory crashes on that unsupported device. See [compiled validation results](docs/signatory-compiled-validation.md)
for numerical and timing comparisons with the original stack.

The robotics examples also use the Git dependencies declared in `setup.py` and
resources downloaded above. GPU and full robotics experiments need separate
validation on their target hardware.

## Saved experiment files

Loaders use PyTorch's restricted `weights_only=True` mode. Tensor/state-dict files
remain supported. Legacy files containing NumPy arrays or arbitrary Python objects
may be rejected. Do not work around this by automatically falling back to unsafe
pickle loading: migrate only files you trust, in an isolated environment, to plain
tensors and primitive containers first. Never load untrusted experiment files or
checkpoints with `weights_only=False`.

## Tests

```sh
python -m pip install pytest
MPLBACKEND=Agg python -m pytest tests -q
```

# Misc.

- Rebuttal experimental seeds are located at https://github.com/lubaroli/stein_mpc/releases/tag/rebuttal-experiment-result
