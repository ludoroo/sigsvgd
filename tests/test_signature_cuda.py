"""Small GPU parity regressions, not performance benchmarks."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import signatory

from src.kernels import PathSigKernel, GaussianKernel
from src.kernels._traj_kernels import _path_signature


@pytest.fixture(autouse=True)
def disable_tf32():
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('backend', ['direct', 'wrapper', 'chunked', 'kernel', 'kernel_chunked'])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_values_and_path_vjp(dtype, backend):
    torch.manual_seed(42)
    # Signatory's CUDA batch trick uses min(round(2048 / batch),
    # length // 3, round(sqrt(length))) chunks: length 25 selects five,
    # whereas length 5 does not chunk. Call the whole public operation.
    length = 25 if backend in ('chunked', 'kernel_chunked') else 5
    source = torch.randn(4, 2, length, dtype=dtype).transpose(1, 2) * 0.1
    def evaluate(device):
        x = source.to(device).detach().requires_grad_(True)
        assert not x.is_contiguous()
        assert x.stride() == source.stride()
        if backend in ('kernel', 'kernel_chunked'):
            kernel = PathSigKernel(static_kernel=GaussianKernel(bandwidth_fn=lambda _: 2.0))
            y = x.detach().flip(0)
            out = kernel(x, y, depth=3, compute_grad=False)
        elif backend == 'direct':
            out = signatory.signature(x, 3, basepoint=True)
        else:
            out = _path_signature(x, 3)
        cotangent = torch.arange(out.numel(), dtype=dtype, device=device).reshape(out.shape) / out.numel()
        grad = torch.autograd.grad((out * cotangent).sum(), x)[0]
        return out.detach().cpu(), grad.cpu()
    cpu, gpu = evaluate('cpu'), evaluate('cuda')
    for a, b in zip(cpu, gpu):
        assert torch.allclose(a, b, rtol=1e-4 if dtype == torch.float32 else 1e-9,
                              atol=1e-5 if dtype == torch.float32 else 1e-10)


def test_cuda_unavailable_cli(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/validate_path_signature.py'
    result = subprocess.run([sys.executable, str(script), '--device', 'cuda',
                             '--backend', 'signatory', '--output', str(tmp_path / 'out.npz')],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True)
    assert result.returncode != 0
    assert 'CUDA requested but unavailable' in result.stderr
    assert not (tmp_path / 'out.npz').exists()
