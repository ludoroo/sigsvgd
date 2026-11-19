"""Public contracts preserved by the Linux native dispatcher optimization."""
import pytest
import torch
import signatory


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('create_graph', [False, True])
def test_saved_gradient_alias_mutation_and_external_versions(device, create_graph):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    x = torch.linspace(-.3, .5, 16, device=device, dtype=torch.double).reshape(2,4,2).requires_grad_()
    initial = torch.zeros(2,14, device=device, dtype=torch.double, requires_grad=True)
    versions = (x._version, initial._version)
    y = signatory.signature(x, 3, initial=initial, basepoint=True)
    cotangent = torch.ones_like(y)
    cv = cotangent._version
    owned = [x, initial, y, cotangent]
    snapshots = [t.detach().clone() for t in owned]
    g, gi = torch.autograd.grad(y, (x,initial), cotangent, create_graph=create_graph)
    assert (x._version, initial._version, cotangent._version) == (*versions,cv)
    assert torch.autograd.gradcheck(lambda a,b: signatory.signature(a, 3, initial=b, basepoint=True), (x,initial))
    assert all(torch.equal(t, saved) for t,saved in zip(owned,snapshots))
    assert torch.isfinite(g).all()
    z = torch.ones_like(g, requires_grad=True)
    loss = (z*g).sum()
    alias = g.detach().transpose(0,1)
    alias.add_(1)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        loss.backward()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_combine_values_gradients_and_versions(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    paths = [torch.linspace(-.2,.4,12,device=device,dtype=torch.double).reshape(2,3,2) + i*.03 for i in range(3)]
    xs = [signatory.signature(x,2).detach().requires_grad_() for x in paths]
    versions = [x._version for x in xs]
    fn = lambda *args: signatory.multi_signature_combine(list(args),2,2)
    y = fn(*xs)
    pairwise = signatory.signature_combine(signatory.signature_combine(xs[0],xs[1],2,2),xs[2],2,2)
    torch.testing.assert_close(y,pairwise,rtol=1e-9,atol=1e-10)
    assert torch.autograd.gradcheck(fn,tuple(xs))
    cotangent = torch.ones_like(y)
    cv = cotangent._version
    snapshots = [t.detach().clone() for t in [*xs,y,cotangent]]
    grads = torch.autograd.grad(y,xs,cotangent)
    assert all(torch.equal(t,saved) for t,saved in zip([*xs,y,cotangent],snapshots))
    assert all(torch.isfinite(g).all() for g in grads)
    assert [x._version for x in xs] == versions
    assert cotangent._version == cv


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_output_mutation_is_detected_before_backward(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    x = torch.randn(2,4,2,device=device,requires_grad=True)
    y = signatory.signature(x,3,basepoint=True)
    with torch.no_grad():
        y.add_(1)
    with pytest.raises(RuntimeError,match='modified by an inplace operation'):
        y.sum().backward()


@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_returned_gradients_do_not_mutate_other_returns_or_callers(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    x = torch.randn(2,4,2,device=device,dtype=torch.double,requires_grad=True)
    base = torch.randn(2,2,device=device,dtype=torch.double,requires_grad=True)
    initial = torch.randn(2,14,device=device,dtype=torch.double,requires_grad=True)
    y = signatory.signature(x,3,basepoint=base,initial=initial,inverse=True)
    go = torch.ones_like(y)
    grads = torch.autograd.grad(y,(x,base,initial),go)
    owned = [x,base,initial,y,go]
    originals = [t.detach().clone() for t in owned]
    saved = [g.clone() for g in grads]
    for i,g in enumerate(grads):
        g.detach().transpose(0,1).add_(1)
        assert all(torch.equal(t,v) for t,v in zip(owned,originals))
        for j,other in enumerate(grads):
            if j > i:
                assert torch.equal(other,saved[j])


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('inverse', [False, True])
@pytest.mark.parametrize('scalar_term', [False, True])
@pytest.mark.parametrize('basepoint', [False, True, 'tensor'])
@pytest.mark.parametrize('with_initial', [False, True])
def test_cpu_native_branches_preserve_owned_values(stream, inverse, scalar_term, basepoint, with_initial):
    x = torch.linspace(-.2,.5,16,dtype=torch.double).reshape(2,4,2).requires_grad_()
    base = torch.full((2,2),.1,dtype=torch.double,requires_grad=True) if basepoint == 'tensor' else basepoint
    initial = torch.full((2,14 + int(scalar_term)),.05,dtype=torch.double,requires_grad=True) if with_initial else None
    inputs = [x] + ([base] if torch.is_tensor(base) else []) + ([initial] if initial is not None else [])
    saved_inputs = [t.detach().clone() for t in inputs]
    versions = [t._version for t in inputs]
    y = signatory.signature(x,3,stream=stream,inverse=inverse,scalar_term=scalar_term,basepoint=base,initial=initial)
    assert all(torch.equal(t,v) for t,v in zip(inputs,saved_inputs))
    cotangent = torch.linspace(.1,.8,y.numel(),dtype=y.dtype).reshape(y.shape)
    saved_y, saved_cotangent = y.detach().clone(), cotangent.clone()
    cv = cotangent._version
    grads = torch.autograd.grad(y,inputs,cotangent)
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(torch.equal(t,v) for t,v in zip(inputs,saved_inputs))
    assert [t._version for t in inputs] == versions
    assert torch.equal(y,saved_y)
    assert torch.equal(cotangent,saved_cotangent)
    assert cotangent._version == cv
