import torch,signatory,os
assert signatory.__file__.startswith(os.environ['PYTHONPATH']+'/')
torch.set_num_threads(1)
for device in ['cpu','cuda']:
 x=torch.randn(2,4,2,dtype=torch.double,device=device,requires_grad=True)
 assert torch.autograd.gradcheck(lambda x:signatory.signature(x,2,basepoint=True),(x,))
 y=signatory.signature(x,2,basepoint=True);y.sum().backward(retain_graph=True);g=x.grad.clone();y.sum().backward();assert torch.equal(x.grad,2*g);assert not x.grad.is_inference()
 z=torch.randn(2,4,2,dtype=torch.double,device=device,requires_grad=True);out=signatory.signature(z,2)
 with torch.no_grad():z.add_(1)
 try:out.sum().backward()
 except RuntimeError:pass
 else:raise AssertionError('mutation not detected')
 a=torch.randn(3,requires_grad=True,device=device);assert not (a*a).is_inference();(a*a).sum().backward();assert torch.equal(a.grad,2*a)
 print(device,'gradcheck, accumulation, normal gradients, path mutation detection, guard restoration PASS')
