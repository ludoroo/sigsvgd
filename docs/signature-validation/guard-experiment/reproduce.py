"""Fresh-only reproducer: python3 reproduce.py /tmp/guard-fresh-run.
Requires documented interpreters, checkout, canonical NPZ and downloaded sdist.
"""
import pathlib,sys,subprocess,os
out=pathlib.Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
source=pathlib.Path(__file__).parent
for name in ['build.py','run.py','finish.py','cpu4.py','safety.py','review_results.py','alias_demo.py','trace_checks.py']:
 text=(source/name).read_text().replace('/tmp/sigsvgd-guard-experiment',str(out));(out/name).write_text(text)
py='/home/coder/.venvs/sigsvgd-candidate/bin/python'
def run(name):subprocess.run([py,str(out/name)],check=True)
run('build.py');run('run.py');run('finish.py');run('cpu4.py')
for v in ['control','patched']:
 env=dict(os.environ,PYTHONPATH=str(out/v/'overlay'),SAFETY_OUT=str(out/f'{v}-safety.json'))
 subprocess.run([py,str(out/'safety.py')],env=env,check=True)
 alias=subprocess.check_output([py,str(out/'alias_demo.py')],env=env,text=True)
 (out/f'{v}-alias.txt').write_text(alias)
 import json
 observed=json.loads(alias)
 assert observed['view']==(v=='control') and observed['base']==(v=='control')
 assert observed['base_mutation_changes_returned_gradient']==(True if v=='control' else 'no base exposed')
run('review_results.py')
