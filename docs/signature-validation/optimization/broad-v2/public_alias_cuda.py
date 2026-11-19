import torch,signatory,json,itertools
torch.set_default_device('cuda');torch.set_num_threads(1)
r={}
for stream,inv,initial in itertools.product([False,True],repeat=3):
 x=torch.arange(8,dtype=torch.double).reshape(1,4,2).div(10).requires_grad_();b=torch.zeros(1,2,dtype=torch.double,requires_grad=True);i=torch.ones(1,6,dtype=torch.double,requires_grad=True)
 ins=[x,b]+([i] if initial else [])
 y=signatory.signature(x,2,basepoint=b,initial=i if initial else None,stream=stream,inverse=inv)
 gs=torch.autograd.grad(y.sum(),ins,retain_graph=True)
 for j,g in enumerate(gs):
  # Save the public returned gradient for a separate ordinary autograd computation.
  a=torch.ones_like(g,requires_grad=True);loss=(a*g).sum();alias=g.detach().view(-1)
  with torch.no_grad():alias.add_(1)
  try:loss.backward();status='accepted'
  except RuntimeError:status='invalidated'
  r[str((stream,inv,initial,j))]=status
  # User-created views must share mutation/version tracking with their source.
  v=g.view(-1);a=torch.ones_like(v,requires_grad=True);loss=(a*v).sum()
  with torch.no_grad():g.add_(1)
  try:loss.backward();status='accepted'
  except RuntimeError:status='invalidated'
  r[str((stream,inv,initial,j,'view'))]=status
print(json.dumps(r,sort_keys=True,indent=2))
