"""Public saved-gradient mutation and accumulation check for an explicit overlay."""
import torch,sys,json
torch.set_default_device("cuda")
sys.path.insert(0,sys.argv[1]);import signatory
assert signatory.__file__.startswith(sys.argv[1]+"/"), signatory.__file__
r={}
with torch._C._AutoDispatchBelowAutograd():
 x=torch.empty(2);r['empty']=x._version
 for name,fn in [('zero_',lambda:x.zero_()),('copy_',lambda:x.copy_(torch.ones(2))),('add_',lambda:x.add_(1)),('clone',lambda:x.clone())]:
  y=fn();r[name]=[x._version,y._version]
r['cases']=[]
for cg in (False,True):
 torch.manual_seed(4);x=torch.randn(2,4,2,dtype=torch.double,requires_grad=True)
 y=signatory.signature(x,3);g,=torch.autograd.grad(y.sum(),x,create_graph=cg)
 z=torch.randn_like(g,requires_grad=True);loss=(z*g).sum();alias=g.detach().transpose(0,1);v=g._version;alias.add_(1)
 try:loss.backward();raised=False
 except RuntimeError as e:raised='modified by an inplace operation' in str(e)
 assert raised
 r['cases'].append({'create_graph':cg,'version':v,'requires_grad':g.requires_grad,'gradient_before_mutation':(g-1).tolist(),'alias_check':raised})
torch.manual_seed(4);x=torch.randn(2,4,2,dtype=torch.double,requires_grad=True)
signatory.signature(x,3).sum().backward();first=x.grad.clone();signatory.signature(x,3).sum().backward();assert torch.allclose(x.grad,2*first)
r['accumulation']=True
print(json.dumps(r))
