import os,pathlib,subprocess,runpy,sys,shutil,json
from trace_checks import paths, CASES, PHASES
R=pathlib.Path('/tmp/sigsvgd-guard-experiment'); REPO=pathlib.Path('/home/coder/sigsvgd'); PY='/home/coder/.venvs/sigsvgd-candidate/bin/python'; INPUT='/tmp/sigsvgd-linux-validation/legacy-cpu-t1.npz'
if len(sys.argv)>1:
 overlay=sys.argv[1]; import signatory
 assert str(pathlib.Path(signatory.__file__).resolve()).startswith(overlay+'/'),signatory.__file__
 script=sys.argv[2];sys.path.insert(0,str(pathlib.Path(script).parent));sys.argv=sys.argv[2:];runpy.run_path(script,run_name='__main__');sys.exit()
def run(v,script,args,log):
 overlay=str(R/v/'overlay');env=dict(os.environ,PYTHONPATH=overlay,OMP_NUM_THREADS='1')
 with (R/log).open('w') as f:subprocess.run(['taskset','-c','2',PY,__file__,overlay,str(REPO/script),*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=REPO)
if __name__=='__main__':
 backup=R/'baseline-traces';backup.mkdir(exist_ok=True)
 for p in pathlib.Path('/tmp').glob('signatory-*.json'):shutil.copy2(p,backup/p.name)
 for device in ['cpu','cuda']:
  for i,v in enumerate(['control','patched','patched','control']):
   run(v,'scripts/profile_signatory_overhead.py',['--inputs',INPUT,'--device',device,'--mode','timing','--output',str(R/f'{device}-{i}-{v}.json')],f'{device}-{i}-{v}.txt')
 for v in ['control','patched']:
  for device in ['cpu','cuda']:
   for mode in ['split','profile']:
    if mode=='profile':
     version=subprocess.check_output([PY,'-c','import torch; print(torch.__version__)'],text=True).strip()
     for case in CASES:
      for phase in PHASES:pathlib.Path(f'/tmp/signatory-{version}-{device}-{case}-{phase}.json').unlink(missing_ok=True)
    run(v,'scripts/profile_signatory_overhead.py',['--inputs',INPUT,'--device',device,'--mode',mode,'--output',str(R/f'{v}-{device}-{mode}.json')],f'{v}-{device}-{mode}.txt')
    if mode=='profile':
     d=R/f'{v}-{device}-traces';d.mkdir(exist_ok=True)
     profile=json.loads((R/f'{v}-{device}-profile.json').read_text())
     for p in paths(profile):
      assert p.is_file(),p
      shutil.copy2(p,d/p.name)
  for device,threads in [('cpu',1),('cpu',4),('cuda',1)]:
   run(v,'scripts/validate_path_signature.py',['--backend','signatory','--inputs',INPUT,'--device',device,'--threads',str(threads),'--repeats','15','--output',str(R/f'{v}-{device}-{threads}.npz')],f'{v}-{device}-{threads}.txt')
