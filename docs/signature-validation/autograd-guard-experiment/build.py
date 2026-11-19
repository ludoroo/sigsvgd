import pathlib,subprocess,tarfile,hashlib,os,sys
r=pathlib.Path(sys.argv[1]).resolve(); py='/home/coder/.venvs/sigsvgd-candidate/bin/python'; s=pathlib.Path('/tmp/signatory-src/signatory-1.2.6.1.9.0.tar.gz')
r.mkdir(parents=True,exist_ok=False)
if hashlib.sha256(s.read_bytes()).hexdigest() != 'c9ab17df6286688c1c7b12c0e906031a1fe7681dfca551876c1ff616e12807d8': raise ValueError('source SHA256 mismatch')
(r/'sdist.sha256').write_text(hashlib.sha256(s.read_bytes()).hexdigest()+'  '+str(s)+'\n')
for variant in ['control','patched']:
 d=r/variant; d.mkdir(exist_ok=False)
 with tarfile.open(s) as t:t.extractall(d,filter='data')
 src=d/'signatory-1.2.6.1.9.0'
 if variant=='patched':
  for f,needle in [('signature.cpp','        path_increments = path_increments.detach();'),('tensor_algebra_ops.cpp','        for (auto& elem : sigtensors) {\n            elem = elem.detach();\n        }')]:
   p=src/'src'/f; text=p.read_text(); 
   if text.count(needle)!=1: raise ValueError('detach anchor mismatch')
   p.write_text(text.replace(needle,needle+'\n        at::AutoDispatchBelowAutograd autograd_only_guard;'))
 env=dict(os.environ,MAX_JOBS='1',PYTHONDONTWRITEBYTECODE='1')
 with (d/'build.txt').open('w') as log:subprocess.run([py,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-cache-dir','-w',str(d/'wheels'),str(src)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run([py,'-m','pip','install','--no-deps','--target',str(d/'overlay'),str(next((d/'wheels').glob('*.whl')))],env=env,check=True)
