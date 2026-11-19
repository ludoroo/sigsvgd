import pathlib,json,collections,subprocess,hashlib
from trace_checks import check_pairs
r=pathlib.Path('/tmp/sigsvgd-guard-experiment')
out=check_pairs(r)
(r/'sequences.txt').write_text('\n'.join(out)+'\n')
for v in ['control','patched']:
 for d,t in [('cpu',1),('cpu',4),('cuda',1)]:
  p=subprocess.run(['/home/coder/.venvs/sigsvgd-candidate/bin/python','/home/coder/sigsvgd/scripts/validate_path_signature.py','--compare','/tmp/sigsvgd-linux-validation/legacy-cpu-t1.npz',str(r/f'{v}-{d}-{t}.npz')],capture_output=True,text=True)
  rows=[line for line in p.stdout.splitlines() if ': PASS ' in line or ': FAIL ' in line]
  failures=[line.split(':')[0] for line in rows if ': FAIL ' in line]
  assert len(rows)==22 and p.returncode==(1 if d=='cuda' else 0) and failures==(['high_channel_stress_8x10x7_f32__vjp'] if d=='cuda' else []) and not p.stderr, (p.returncode,p.stdout,p.stderr)
  (r/f'{v}-{d}-{t}-legacy.txt').write_text(p.stdout+p.stderr+f'\nexit={p.returncode}\n')
patch=''
for f in ['signature.cpp','tensor_algebra_ops.cpp']:
 p=subprocess.run(['diff','-u','--label','a/src/'+f,'--label','b/src/'+f,str(r/'control/signatory-1.2.6.1.9.0/src'/f),str(r/'patched/signatory-1.2.6.1.9.0/src'/f)],capture_output=True,text=True);patch+=p.stdout
(r/'source.patch').write_text(patch)
files=[*r.glob('*/wheels/*.whl'),*r.glob('*/overlay/signatory/*.so')]
(r/'build-sha256.txt').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p) for p in files)+'\n')

import numpy as np
exact=[]
for d,t in [('cpu',1),('cpu',4),('cuda',1)]:
 a=np.load(f'/tmp/sigsvgd-linux-validation/candidate-{d}-t{t}.npz')
 for v in ['control','patched']:
  b=np.load(r/f'{v}-{d}-{t}.npz');assert set(a.files)==set(b.files)
  bad=[k for k in a.files if not np.array_equal(a[k],b[k])];assert not bad,bad
  exact.append(f'{v} {d}/{t}: all {len(a.files)} NPZ arrays exactly equal')
(r/'exact-summary.txt').write_text('\n'.join(exact)+'\n')
