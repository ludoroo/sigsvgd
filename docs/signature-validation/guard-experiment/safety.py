"""Comparative observations, not assumptions about unsupported differentiation."""
import torch,signatory,os,json,itertools,concurrent.futures
assert signatory.__file__.startswith(os.environ['PYTHONPATH']+'/')
torch.set_num_threads(1);torch.manual_seed(724)
def snap(x):
 if x is None:return None
 return dict(value=x.detach().cpu().tolist(),version=x._version,view=x._is_view(),base=x._base is not None,inference=x.is_inference(),requires_grad=x.requires_grad,gradfn=type(x.grad_fn).__name__ if x.grad_fn else None)
def attempt(f):
 try:return f()
 except Exception as e:return dict(error=type(e).__name__,message=str(e))
def observe(fn,inputs):
 y=fn();go=torch.linspace(.1,.7,y.numel(),device=y.device,dtype=y.dtype).reshape(y.shape).requires_grad_();ts=inputs+[y,go]
 result={'before':list(map(snap,ts))}
 gs=torch.autograd.grad(y,inputs,go,retain_graph=True,allow_unused=True);result['grad']=list(map(snap,gs));result['after']=list(map(snap,ts))
 result['repeat']=list(map(snap,torch.autograd.grad(y,inputs,go,retain_graph=True,allow_unused=True)))
 y.backward(go,retain_graph=True);result['accum1']=[snap(x.grad) for x in inputs];y.backward(go,retain_graph=True);result['accum2']=[snap(x.grad) for x in inputs]
 cg=torch.autograd.grad(y,inputs,go,create_graph=True,retain_graph=True,allow_unused=True);result['create_graph']=list(map(snap,cg));result['second']=attempt(lambda:list(map(snap,torch.autograd.grad(sum(g.sum() for g in cg if g is not None),inputs,allow_unused=True))))
 with torch.no_grad():y.add_(.125)
 result['output_mutation']=attempt(lambda:list(map(snap,torch.autograd.grad(y,inputs,go,retain_graph=True,allow_unused=True))))
 return result
results={}
for dev in ['cpu','cuda']:
 def leaf(shape):return torch.linspace(-.3,.5,torch.tensor(shape).prod().item(),device=dev,dtype=torch.double).reshape(shape).clone().requires_grad_()
 for length,stream,bp,inv,initial,scalar in itertools.product([2,4],[False,True],['false','true','tensor'],[False,True],[False,True],[False,True]):
  x=leaf((1,length,2));b=leaf((1,2)) if bp=='tensor' else bp=='true';ini=leaf((1,7 if scalar else 6)) if initial else None;inputs=[x]+([b] if torch.is_tensor(b) else [])+([ini] if initial else [])
  key=f'{dev}/signature/{length}/{stream}/{bp}/{inv}/{initial}/{scalar}'
  results[key]=attempt(lambda:observe(lambda:signatory.signature(x,2,stream=stream,basepoint=b,inverse=inv,initial=ini,scalar_term=scalar),inputs))
 for n,alias,inv,scalar in itertools.product([1,2,3],[False,True],[False,True],[False,True]):
  a=leaf((1,7 if scalar else 6));xs=[a]*n if alias else [leaf(a.shape) for _ in range(n)];unique=list(dict.fromkeys(xs));results[f'{dev}/combine/{n}/{alias}/{inv}/{scalar}']=attempt(lambda:observe(lambda:signatory.multi_signature_combine(xs,2,2,inverse=inv,scalar_term=scalar),unique))
 for name,fn in [('signature',lambda x:signatory.signature(x,2,basepoint=True)),('logsignature',lambda x:signatory.logsignature(x,2,basepoint=True)),('Path',lambda x:signatory.Path(x,2,basepoint=True).signature())]:
  results[f'{dev}/gradcheck/{name}']=attempt(lambda:torch.autograd.gradcheck(fn,(leaf((1,3,2)),)))
 # Compare original-path mutation rather than assuming path is saved.
 x=leaf((1,3,2));y=signatory.signature(x,2)
 with torch.no_grad():x.add_(1)
 results[f'{dev}/path_mutation']=attempt(lambda:snap(torch.autograd.grad(y.sum(),x)[0]))
 results[f'{dev}/native_throw']='not run in-process: depth=0 malformed native call segfaulted both variants; no guard restoration inference'
 z=leaf((2,));results[f'{dev}/restoration']=snap(torch.autograd.grad((z*z).sum(),z)[0])
def smoke(i):
 x=torch.full((1,3,2),.1+i*.01,dtype=torch.double,requires_grad=True);return snap(torch.autograd.grad(signatory.signature(x,2).sum(),x)[0])
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results['threads']=list(pool.map(smoke,range(4)))
open(os.environ['SAFETY_OUT'],'w').write(json.dumps(results,indent=2)+'\n')
