import torch,signatory,json,itertools,concurrent.futures
torch.set_default_device('cuda');torch.set_num_threads(1)
def case(cfg):
 stream,inv,initial,target=cfg
 x=torch.arange(8,dtype=torch.double).reshape(1,4,2).div(10).requires_grad_();b=torch.zeros(1,2,dtype=torch.double,requires_grad=True);i=torch.ones(1,6,dtype=torch.double,requires_grad=True)
 ins=[x,b]+([i] if initial else [])
 y=signatory.signature(x,2,basepoint=b,initial=i if initial else None,stream=stream,inverse=inv)
 gs=torch.autograd.grad(y.sum(),ins,retain_graph=True);out={'inference':[g.is_inference() for g in [y]+list(gs)],'values':[g.tolist() for g in gs]}
 # Observe real overlap by mutation, not storage bounds or private view metadata.
 overlaps=[]
 for g in gs:
  before=[h.clone() for h in gs]
  with torch.no_grad():g.add_(.125)
  overlaps.append([not torch.equal(h,v) for h,v in zip(gs,before)])
 out['overlap']=overlaps
 ts=ins+[y]
 if target<len(ts):
  with torch.no_grad():ts[target].detach().add_(.125)
  try:out['mutation']=[g.tolist() for g in torch.autograd.grad(y.sum(),ins)]
  except RuntimeError as e:out['mutation']='RuntimeError'
 return out
cfgs=list(itertools.product([False,True],[False,True],[False,True],range(4)))
r={str(c):case(c) for c in cfgs}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:r['threads']=list(p.map(case,cfgs[:4]))
print(json.dumps(r,sort_keys=True,indent=2))
