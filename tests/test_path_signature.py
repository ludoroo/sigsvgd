import pytest
import torch

from src.kernels import GaussianKernel, PathSigKernel
from src.kernels._traj_kernels import _path_signature


def test_one_dimensional_signature_with_basepoint():
    path = torch.tensor([[[2.0], [5.0]]], dtype=torch.double)
    signature = _path_signature(path, depth=3)
    expected = torch.tensor([[5.0, 5.0**2 / 2, 5.0**3 / 6]], dtype=torch.double)
    torch.testing.assert_close(signature, expected)


def test_straight_line_signature_is_tensor_exponential():
    displacement = torch.tensor([[1.5, -2.0]], dtype=torch.double)
    path = torch.stack((torch.zeros_like(displacement), displacement), dim=1)
    signature = _path_signature(path, depth=3)
    level1 = displacement
    level2 = torch.einsum("bi,bj->bij", displacement, displacement).flatten(1) / 2
    level3 = (
        torch.einsum(
            "bi,bj,bk->bijk", displacement, displacement, displacement
        ).flatten(1)
        / 6
    )
    torch.testing.assert_close(signature, torch.cat((level1, level2, level3), dim=1))


def test_signature_obeys_chen_identity_and_preserves_path_order():
    origin = torch.zeros(1, 1, 2, dtype=torch.double)
    x = torch.tensor([[[1.0, 0.0]]], dtype=torch.double)
    y = torch.tensor([[[0.0, 1.0]]], dtype=torch.double)
    path_xy = torch.cat((origin, x, x + y), dim=1)
    path_yx = torch.cat((origin, y, y + x), dim=1)
    sig_xy = _path_signature(path_xy, depth=2)
    sig_yx = _path_signature(path_yx, depth=2)
    assert not torch.allclose(sig_xy, sig_yx)

    sig_x = _path_signature(torch.cat((origin, x), dim=1), depth=2)
    sig_y = _path_signature(torch.cat((origin, y), dim=1), depth=2)
    chen_level2 = (
        sig_x[:, 2:]
        + sig_y[:, 2:]
        + torch.einsum("bi,bj->bij", sig_x[:, :2], sig_y[:, :2]).flatten(1)
    )
    torch.testing.assert_close(
        sig_xy, torch.cat((sig_x[:, :2] + sig_y[:, :2], chen_level2), dim=1)
    )


def test_path_signature_kernel_preserves_values_and_path_gradients():
    path = torch.tensor(
        [[[2.0], [3.0]], [[-1.0], [2.0]]], dtype=torch.double, requires_grad=True
    )
    other = torch.tensor([[[0.0], [1.0]], [[3.0], [-2.0]]], dtype=torch.double)
    kernel = PathSigKernel(static_kernel=GaussianKernel(bandwidth_fn=lambda _: 2.0))
    actual = kernel(path, other, depth=2, compute_grad=False)
    # In one dimension, the origin-based signature depends only on the endpoint.
    x, y = path[:, -1], other[:, -1]
    x_sig = torch.cat((x, x.square() / 2), dim=1)
    y_sig = torch.cat((y, y.square() / 2), dim=1)
    expected = (-(x_sig[:, None] - y_sig[None]).square().sum(-1) / 8).exp()
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), path, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), path)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("device", ["meta", "mps"])
def test_signature_rejects_unsupported_devices_without_calling_native_code(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS hardware unavailable")
    path = torch.empty(2, 3, 2, device=device)
    with pytest.raises(ValueError, match="requires CPU or CUDA"):
        _path_signature(path, depth=2)


def test_signature_passes_gradcheck():
    path = torch.randn(1, 3, 2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p: _path_signature(p, depth=3), (path,))
