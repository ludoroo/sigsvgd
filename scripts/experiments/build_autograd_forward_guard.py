"""Build the reviewed v2 four-guard isolated experiment wheel; never installs into the interpreter environment."""
import pathlib,tarfile,hashlib,subprocess,os,sys
r=pathlib.Path(sys.argv[1]).resolve();r.mkdir(exist_ok=False)
s=pathlib.Path('/tmp/signatory-src/signatory-1.2.6.1.9.0.tar.gz');assert hashlib.sha256(s.read_bytes()).hexdigest()=='c9ab17df6286688c1c7b12c0e906031a1fe7681dfca551876c1ff616e12807d8'
with tarfile.open(s) as t:t.extractall(r,filter='data')
src=r/'signatory-1.2.6.1.9.0'
for file,anchors in [('signature.cpp',['        initial_value = initial_value.detach();','        path_increments = path_increments.detach();']),('tensor_algebra_ops.cpp',['        // Actually do the computation\n\n        torch::Tensor out;','        for (auto& elem : sigtensors) {\n            elem = elem.detach();\n        }'])]:
 p=src/'src'/file;text=p.read_text()
 for anchor in anchors:
  assert text.count(anchor)==1,(file,anchor,text.count(anchor));text=text.replace(anchor,anchor+'\n        at::AutoDispatchBelowADInplaceOrView autograd_only_guard;')
 p.write_text(text)
py=sys.executable;env=dict(os.environ,MAX_JOBS='1',PYTHONDONTWRITEBYTECODE='1')
with (r/'build.txt').open('w') as f:subprocess.run([py,'-m','pip','wheel','--no-deps','--no-build-isolation','--no-cache-dir','-w',str(r/'wheels'),str(src)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run([py,'-m','pip','install','--no-deps','--target',str(r/'overlay'),str(next((r/'wheels').glob('*.whl')))],env=env,check=True)
