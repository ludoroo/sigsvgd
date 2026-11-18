"""Exercise the profiling CLI without relying on timing thresholds."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_profile_cli(tmp_path, device):
    pytest.importorskip('signatory')
    inputs = tmp_path / 'inputs.npz'
    np.savez(inputs, placeholder=np.zeros(1))
    output = tmp_path / 'result.json'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    result = subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'scripts/profile_signatory_overhead.py'),
                    '--inputs', str(inputs), '--output', str(output), '--device', device,
                    '--mode', 'microbench', '--warmups', '0', '--repeats', '2'],
                   env=env, capture_output=True, text=True)
    data = json.loads(output.read_text())
    assert data['device'] == device
    if device == 'cuda':
        assert result.returncode == 2
        assert 'CUDA unavailable' in result.stderr
        assert data['unavailable'] == 'CUDA unavailable'
    else:
        assert result.returncode == 0, result.stderr
        assert set(data['cases']) == {'add', 'mul', 'clone', 'sin'}
        assert all(len(x['samples_ms']) == 2 for x in data['cases'].values())


def test_workload_matches_validation():
    import importlib.util
    torch = pytest.importorskip('torch')
    signatory = pytest.importorskip('signatory')
    scripts = Path(__file__).parents[1] / 'scripts'
    def load(name):
        spec = importlib.util.spec_from_file_location(name, scripts / (name+'.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    harness = load('validate_path_signature')
    harness.np = np
    profile = load('profile_signatory_overhead')
    # Mirrors the public validation harness path, basepoint and cotangent contraction.
    for case in harness.make_cases():
        if not case['name'].endswith('_f32'):
            continue
        x = torch.from_numpy(case['input'].copy()).requires_grad_(True)
        cot = torch.from_numpy(case['cotangent'].copy())
        path = torch.cat((x.cos(), x.sin()), dim=-1) if case['transform'] == 'cos_sin' else x
        expected = signatory.signature(path, case['depth'], basepoint=True)
        gradient = torch.autograd.grad((expected*cot).sum(), x)[0]
        actual, grad = profile.workload(torch, signatory, x, cot, case, True)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(grad, gradient)
        forward, grad = profile.workload(torch, signatory, x, cot, case, False)
        with torch.no_grad():
            expected_forward = signatory.signature(path.detach(), case['depth'], basepoint=True)
        torch.testing.assert_close(forward, expected_forward)
        assert not forward.requires_grad and grad is None


def test_kernel_categories():
    import runpy
    summary = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/profile_signatory_overhead.py'))['kernel_summary']
    events = [{'cat': 'Kernel', 'dur': 2}, {'cat': 'kernel', 'dur': 3},
              {'cat': 'cpu_to_cuda', 'dur': 100}, {'cat': 'gpu_memcpy', 'dur': 10}]
    assert summary({'traceEvents': events}) == dict(cuda_activity_status='available', kernel_count=2, kernel_duration_us=5)
    assert summary(events)['kernel_count'] == 2
    assert summary([])['kernel_count'] == 0


def test_split_cli(tmp_path):
    import runpy
    pytest.importorskip('signatory')
    scripts = Path(__file__).parents[1] / 'scripts'
    harness = runpy.run_path(str(scripts / 'validate_path_signature.py'))
    harness['make_cases'].__globals__['np'] = np
    cases = [c for c in harness['make_cases']() if c['name'].endswith('_f32')]
    inputs = tmp_path / 'inputs.npz'
    np.savez(inputs, **{c['name']+'__'+key: c[source] for c in cases
                      for key, source in [('input', 'input'), ('cotangent', 'cotangent')]})
    output = tmp_path / 'split.json'
    subprocess.run([sys.executable, str(scripts / 'profile_signatory_overhead.py'),
                    '--inputs', str(inputs), '--output', str(output), '--mode', 'split',
                    '--warmups', '0', '--repeats', '1'], check=True)
    data = json.loads(output.read_text())
    assert set(data['cases']) == {c['name'] for c in cases}
    for result in data['cases'].values():
        assert set(result) == {'forward', 'forward_backward'}
        assert {'signature_checkargs', 'signature_forward', 'signature_backward'} <= set(result['forward_backward']['native_host_ms'])
        assert {'host_total_ms', 'residual_host_ms'} <= set(result['forward_backward'])
    assert data['workload_configuration']['basepoint'] is True
    assert len(data['measurement_sources']) == 2
