"""All-element, directional original tolerance checks; float64 is an oracle proxy."""
import argparse, json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path('/tmp/sigsvgd-linux-validation')); args=p.parse_args()
root=args.root; arrays={}
for f in ['chunking-gpu','chunking-cpu']:
 with np.load(root/(f+'.npz')) as z: arrays.update({k:z[k] for k in z.files})
for label,file in [('old_cpu','legacy-cpu-t1'),('old_gpu','legacy-cuda-t1'),('original_gpu','candidate-cuda-t1')]:
 with np.load(root/(file+'.npz')) as z:
  arrays.update({label+'::'+k:z[k] for k in z.files if '_f32__' in k and k.endswith(('__vjp','__signature'))})
def metric(a,b):
 d=np.abs(a.astype(float)-b.astype(float)); tol=1e-5+1e-4*np.abs(b)
 return dict(failed=int(np.count_nonzero(d>tol)),total=a.size,max_abs=float(d.max()),max_tolerance_ratio=float((d/tol).max()),l2_error=float(np.linalg.norm(d)))
rows=[]
for k in arrays:
 if not k.startswith('cpu_f64::'):continue
 name=k.split('::')[1]; ref=arrays[k]
 for label in ['cpu_t1','cpu_t24','cuda_default','cuda_nochunk','old_cpu','old_gpu','original_gpu']:
  v=arrays[label+'::'+name]
  rows.append(dict(case=name,variant=label,vs_cpu64=metric(ref,v),vs_old_cpu=metric(arrays['old_cpu::'+name],v),vs_old_gpu=metric(arrays['old_gpu::'+name],v)))
Path(__file__).with_name('summary.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:
 if 'stress' in r['case']:print(r)
