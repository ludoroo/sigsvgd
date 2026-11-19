"""Rebuild both overlays and reproduce the rejected metadata result sequentially."""
import json
import os
from pathlib import Path
import subprocess
import sys
from compare import compare

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PY = '/home/coder/.venvs/sigsvgd-candidate/bin/python'
OLD = HERE.parent / 'guard-experiment'
root = Path(sys.argv[1]).resolve()
if root.exists():
    raise SystemExit('destination must not exist')
# Use the tracked broad-experiment safety records; fail rather than skip checks.
for name in ('control-safety.json', 'patched-safety.json'):
    if not (OLD/name).is_file():
        raise SystemExit(f'required historical evidence missing: {OLD/name}')
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
subprocess.run([PY, '-c', 'import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 2)'], env=env, check=True)
subprocess.run([PY, str(HERE/'build.py'), str(root)], env=env, check=True)
for variant in ('control', 'patched'):
    overlay = root/variant/'overlay'
    runenv = dict(env, PYTHONPATH=str(overlay), SAFETY_OUT=str(root/f'{variant}-safety.json'))
    # safety.py additionally asserts the overlay import in the actual test process.
    subprocess.run([PY, '-c', 'import signatory, os; from pathlib import Path; '
                    'p=Path(signatory.__file__).resolve(); '
                    'p.relative_to(Path(os.environ["PYTHONPATH"]).resolve())'], env=runenv, check=True)
    with (root/f'{variant}-safety.txt').open('w') as log:
        subprocess.run([PY, str(HERE.parent/'guard-experiment/safety.py')],
                       env=runenv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
expected = json.loads((HERE/'summary.json').read_text())
summary, ds = compare(root/'control-safety.json', root/'patched-safety.json')
if summary != expected:
    raise SystemExit(f'unexpected result: {summary}')
(root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
(root/'safety-differences.json').write_text(json.dumps(ds, indent=2)+'\n')
control, _ = compare(root/'control-safety.json', OLD/'control-safety.json')
broad, _ = compare(root/'control-safety.json', OLD/'patched-safety.json')
if control['field_differences'] != 0 or broad['field_differences'] != 1694:
    raise SystemExit(f'historical checks failed: {control}, {broad}')
print('Reproduced 720 version-only differences; strict exact-metadata gate FAILED. Stop; no performance runs.')
