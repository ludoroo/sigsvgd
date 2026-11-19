import os,subprocess,pathlib
r=pathlib.Path('/tmp/sigsvgd-guard-experiment');py='/home/coder/.venvs/sigsvgd-candidate/bin/python'
for i,v in enumerate(['control','patched','patched','control']):
 env=dict(os.environ,PYTHONPATH=str(r/v/'overlay'),OMP_NUM_THREADS='4')
 with (r/f'cpu4-{i}-{v}.txt').open('w') as log:subprocess.run(['taskset','-c','2-5',py,str(r/'run.py'),env['PYTHONPATH'],'/home/coder/sigsvgd/scripts/validate_path_signature.py','--backend','signatory','--inputs','/tmp/sigsvgd-linux-validation/legacy-cpu-t1.npz','--threads','4','--repeats','50','--output',str(r/f'cpu4-{i}-{v}.npz')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
