import torch,signatory,json,os
assert signatory.__file__.startswith(os.environ['PYTHONPATH']+'/')
x=torch.tensor([[[.1,.2],[.3,.5]]],dtype=torch.double,requires_grad=True);b=torch.zeros(1,2,dtype=torch.double,requires_grad=True)
y=signatory.signature(x,2,basepoint=b,inverse=True);g=torch.autograd.grad(y.sum(),b)[0]
r=dict(view=g._is_view(),version=g._version,base=g._base is not None)
if g._base is not None:
 before=g.clone();g._base.add_(1);r['base_mutation_changes_returned_gradient']=not torch.equal(g,before)
else:r['base_mutation_changes_returned_gradient']='no base exposed'
print(json.dumps(r))
