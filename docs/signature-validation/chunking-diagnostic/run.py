"""Diagnostic only; run GPU first, then CPU, then summarize. No production patch."""
import argparse, hashlib, json, importlib
from pathlib import Path
import numpy as np
import torch
import signatory
p=argparse.ArgumentParser(); p.add_argument('mode', choices=['gpu','cpu']); p.add_argument('--input',default='/tmp/sigsvgd-linux-validation/legacy-cpu-t1.npz'); p.add_argument('--output',required=True); a=p.parse_args()
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
data=np.load(a.input); out={}; sm=importlib.import_module('signatory.signature_module'); original=sm._signature_batch_trick
configs=[('cuda_default','cuda',1,False,False),('cuda_nochunk','cuda',1,True,False)] if a.mode=='gpu' else [('cpu_t1','cpu',1,False,False),('cpu_t24','cpu',24,False,False),('cpu_f64','cpu',1,False,True)]
for label,device,threads,disable,double in configs:
 torch.set_num_threads(threads); sm._signature_batch_trick=(lambda *args,**kwargs:None) if disable else original
 for key in data.files:
  if not key.endswith('_f32__input'): continue
  name=key[:-7]; depth=3 if name.startswith(('traj','channels')) else 4
  x=torch.tensor(data[key],device=device,dtype=torch.float64 if double else torch.float32,requires_grad=True)
  cot=torch.tensor(data[name+'__cotangent'],device=device,dtype=x.dtype)
  path=torch.cat((x.cos(),x.sin()),-1) if name.startswith('traj') else x
  y=signatory.signature(path,depth,basepoint=True); g=torch.autograd.grad((y*cot).sum(),x)[0]
  for kind,v in [('signature',y),('vjp',g)]: out[label+'::'+name+'__'+kind]=v.detach().cpu().numpy()
np.savez(a.output,**out)
Path(a.output+'.json').write_text(json.dumps({'input_path':str(Path(a.input).resolve()),'input_sha256':hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),'torch':torch.__version__,'signatory':signatory.__version__,'device':torch.cuda.get_device_name() if a.mode=='gpu' else 'cpu','tf32':False,'configs':configs},indent=2))
print('DONE',a.mode,flush=True)
