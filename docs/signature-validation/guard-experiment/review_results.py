import pathlib,json,collections
r=pathlib.Path('/tmp/sigsvgd-guard-experiment');lines=[]
for dev in ['cpu','cuda','cpu4']:
 for c,p in [(0,1),(3,2)]:
  a=json.loads((r/f'{dev}-{c}-control.json').read_text())['cases'];b=json.loads((r/f'{dev}-{p}-patched.json').read_text())['cases']
  for k in a:
   for phase in (['forward_no_grad','forward_backward'] if dev=='cpu4' else ['forward','forward_backward']):
    for metric in (['median_seconds','p95_seconds'] if dev=='cpu4' else ['median_ms','p95_ms']):
     x=a[k][phase][metric];y=b[k][phase][metric];lines.append(f'{dev} pair {c}/{p} {k} {phase} {metric}: {x:.8g} -> {y:.8g} improvement={(1-y/x)*100:.2f}%')
(r/'tails.txt').write_text('\n'.join(lines)+'\n')
a=json.loads((r/'control-safety.json').read_text());b=json.loads((r/'patched-safety.json').read_text());diff=[]
def walk(x,y,p):
 if type(x)!=type(y):diff.append((p,x,y))
 elif isinstance(x,dict):
  assert x.keys()==y.keys()
  for k in sorted(x):walk(x[k],y[k],p+'/'+k)
 elif isinstance(x,list):
  assert len(x)==len(y)
  for i,(v,w) in enumerate(zip(x,y)):walk(v,w,p+'/'+str(i))
 elif x!=y:diff.append((p,x,y))
walk(a,b,'');assert diff and {p.rsplit('/',1)[-1] for p,x,y in diff} <= {'version','base','view'}, 'unexpected safety differences';(r/'safety-differences.json').write_text(json.dumps(diff,indent=2)+'\n')
print('cases',len(a),'differences',len(diff),collections.Counter(p.rsplit('/',1)[-1] for p,x,y in diff))
