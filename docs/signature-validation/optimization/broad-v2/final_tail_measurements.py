"""Run ONLY after explicit parent GPU release. All runs retained; no selection.
Usage: python final_tail_measurements.py [--reverse]
First run LBBL; reverse BLLB only if first same-variant p95 drift exceeds 5%.
"""
import pathlib,subprocess,os,json,statistics,sys,datetime,platform
r=pathlib.Path('/tmp/sigsvgd-broad-cu126');reverse='--reverse' in sys.argv
seq=['broad','legacy','legacy','broad'] if reverse else ['legacy','broad','broad','legacy'];tag='reverse' if reverse else 'initial'
paths=[r/f'tail500-{tag}-{j}-{v}.json' for j,v in enumerate(seq)]
if any(p.exists() for p in paths):raise RuntimeError('Refusing overwrite: all observations must be retained')
if reverse:
 previous=json.loads((r/'tail500-initial-summary.json').read_text())
 if not any(abs(row[k])>.05 for row in previous for k in ['legacy_p95_drift','broad_p95_drift']):raise RuntimeError('Reverse run not warranted by prespecified drift rule')
 time=__import__('time');time.sleep(30)
for j,v in enumerate(seq):
 py='/tmp/sigsvgd-cu126-env/bin/python' if v=='broad' else '/home/coder/.venvs/sigsvgd-reference/bin/python'
 overlay=str(r/'overlay') if v=='broad' else '/home/coder/.venvs/sigsvgd-reference/lib/python3.9/site-packages'
 with paths[j].with_suffix('.log').open('w') as f:subprocess.run(['taskset','-c','2',py,'/tmp/sigsvgd-guard-experiment/run.py',overlay,'/home/coder/sigsvgd/scripts/profile_signatory_overhead.py','--inputs','/tmp/sigsvgd-linux-validation/legacy-cpu-t1.npz','--device','cuda','--mode','timing','--warmups','20','--repeats','500','--output',str(paths[j])],env=dict(os.environ,PYTHONPATH=overlay,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
data=[json.loads(p.read_text()) for p in paths];rows=[]
for case in data[0]['cases']:
 for phase in data[0]['cases'][case]:
  values={v:[d['cases'][case][phase] for d,w in zip(data,seq) if w==v] for v in ['legacy','broad']}
  row=dict(case=case,phase=phase)
  for metric in ['median_ms','p95_ms']:
   l=[d[metric] for d in values['legacy']];b=[d[metric] for d in values['broad']]
   row[metric+'_ratio']=statistics.mean(b)/statistics.mean(l)
   row['legacy_'+metric.split('_')[0]+'_drift']=l[1]/l[0]-1
   row['broad_'+metric.split('_')[0]+'_drift']=b[1]/b[0]-1
   row['legacy_'+metric]=l;row['broad_'+metric]=b
  rows.append(row)
(r/f'tail500-{tag}-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
