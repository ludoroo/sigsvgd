"""Strict trace inventory and sequence checks for this bounded experiment."""
import json
from pathlib import Path
CASES = {'traj_128x25x2_f32', 'representative_100x10x2_f32',
         'channels_20x25x3_f32', 'high_channel_stress_8x10x7_f32'}
PHASES = {'forward', 'forward_backward'}
def paths(profile):
    assert set(profile['cases']) == CASES
    result = []
    for case, phases in profile['cases'].items():
        assert set(phases) == PHASES
        for phase, record in phases.items():
            p = Path(record['trace'])
            assert p == Path('/tmp') / f"signatory-{profile['torch']}-{profile['device']}-{case}-{phase}.json"
            result.append(p)
    assert len(result) == len(set(result)) == 8
    return result

def check_pairs(root):
    lines = []
    for device in ['cpu', 'cuda']:
        profiles = {v: json.loads((root/f'{v}-{device}-profile.json').read_text()) for v in ['control', 'patched']}
        inventory = {v: paths(p) for v, p in profiles.items()}
        assert {p.name for p in inventory['control']} == {p.name for p in inventory['patched']}
        for expected in inventory['control']:
            pair = [root/f'{v}-{device}-traces'/expected.name for v in ['control', 'patched']]
            assert all(p.is_file() for p in pair), pair
            def seq(p, category):
                events = json.loads(p.read_text())['traceEvents']
                return [e.get('name') for e in events if str(e.get('cat', '')).lower() == category]
            for category in ['cpu_op', 'kernel']:
                a, b = [seq(p, category) for p in pair]
                if category == 'cpu_op' or device == 'cuda':
                    assert a and b, (pair, category)
                assert a == b, (pair, category)
            lines.append(f'{expected.name}: CPU op and kernel sequences equal')
    assert len(lines) == 16
    return lines
