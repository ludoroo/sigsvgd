"""Bounded installed-Signatory profiling; CUDA event spans are not compute time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import platform
import sys


def kernel_summary(trace_data):
    """Kineto category spelling differs between torch 1.9 and modern torch."""
    activities = trace_data.get('traceEvents', []) if isinstance(trace_data, dict) else trace_data
    kernels = [e for e in activities if str(e.get('cat', '')).lower() == 'kernel']
    return dict(cuda_activity_status='available' if kernels else 'unavailable or no kernels captured',
                kernel_count=len(kernels), kernel_duration_us=sum(e.get('dur', 0) for e in kernels))


def workload(torch, signatory, x, cotangent, case, backward):
    with torch.set_grad_enabled(backward):
        leaf = x.detach().requires_grad_(backward)
        path = torch.cat([leaf.cos(), leaf.sin()], dim=-1) if case["transform"] == "cos_sin" else leaf
        y = signatory.signature(path, case["depth"], basepoint=True)
        grad = torch.autograd.grad((y*cotangent).sum(), leaf)[0] if backward else None
        return y, grad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--mode', choices=['timing', 'split', 'profile', 'microbench'], default='timing')
    p.add_argument('--warmups', type=int, default=20)
    p.add_argument('--repeats', type=int, default=100)
    a = p.parse_args()
    if not 0 <= a.warmups <= 100 or not 1 <= a.repeats <= 1000:
        p.error('bounded warmups 0..100 and repeats 1..1000 required')
    os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np
    import torch
    import signatory
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(signatory.__file__).parent
    out = dict(python=platform.python_version(), cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available(), device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32, threads=torch.get_num_threads(), omp_num_threads=os.environ['OMP_NUM_THREADS'], torch=torch.__version__, device=a.device, mode=a.mode,
               affinity=sorted(os.sched_getaffinity(0)), warmups=a.warmups,
               repeats=a.repeats, inputs_sha256=hashlib.sha256(Path(a.inputs).read_bytes()).hexdigest(),
               source_hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')}, cases={})
    out['measurement_sources'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__), Path(__file__).with_name('validate_path_signature.py')]}
    out['workload_configuration'] = dict(basepoint=True, forward_no_grad=True)
    out['native_provenance'] = {}
    for f in root.glob('*.so'):
        dwarf = subprocess.run(['readelf', '--debug-dump=info', str(f)], capture_output=True, text=True)
        out['native_provenance'][str(f)] = dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
            producers=sorted(set(x.strip() for x in dwarf.stdout.splitlines() if 'DW_AT_producer' in x)),
            ldd=subprocess.check_output(['ldd', str(f)], text=True))
    if a.device == 'cuda' and not torch.cuda.is_available():
        out['unavailable'] = 'CUDA unavailable'
    else:
        sync = torch.cuda.synchronize if a.device == 'cuda' else lambda: None
        def measure(fn):
            for _ in range(a.warmups):
                fn()
            sync()
            samples = []
            for _ in range(a.repeats):
                sync()
                t = time.perf_counter()
                fn()
                sync()
                samples.append((time.perf_counter()-t)*1000)
            return dict(samples_ms=samples, median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples,95)))
        if a.mode == 'microbench':
            x = torch.ones(128, 25, 2, device=a.device)
            for name, fn in [('add', lambda: x+x), ('mul', lambda: x*x), ('clone', x.clone), ('sin', x.sin)]:
                out['cases'][name] = measure(fn)
        else:
            import validate_path_signature as harness
            harness.np = np
            make_cases = harness.make_cases
            with np.load(a.inputs, allow_pickle=False) as data:
                for case in make_cases():
                    name = case['name']
                    if not name.endswith('_f32'):
                        continue
                    x = torch.from_numpy(data[name+'__input'].copy()).to(a.device)
                    cot = torch.from_numpy(data[name+'__cotangent'].copy()).to(a.device)
                    def fn(backward=False):
                        return workload(torch, signatory, x, cot, case, backward)
                    out['workload_configuration'][name] = dict(depth=case['depth'], transform=case['transform'], basepoint=True)
                    results = {}
                    for backward in [False, True]:
                        call = lambda: fn(backward)
                        key = 'forward_backward' if backward else 'forward'
                        if a.mode == 'timing':
                            results[key] = measure(call)
                            if a.device == 'cuda':
                                spans = []
                                for _ in range(a.repeats):
                                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                    sync(); s.record(); call(); e.record(); e.synchronize()
                                    spans.append(s.elapsed_time(e))
                                results[key]['event_span_including_host_gaps_ms'] = spans
                        elif a.mode == 'split':
                            totals = {}
                            originals = {}
                            try:
                                for n in ['signature_checkargs','signature_forward','signature_backward','signature_combine_forward','signature_combine_backward']:
                                    original = getattr(signatory.impl,n)
                                    originals[n] = original
                                    def wrapped(*args, _n=n, _fn=original, **kw):
                                        t = time.perf_counter()
                                        try:
                                            return _fn(*args, **kw)
                                        finally:
                                            totals[_n] = totals.get(_n,0)+(time.perf_counter()-t)*1000
                                    setattr(signatory.impl,n,wrapped)
                                # warmups excluded from split accounting
                                for _ in range(a.warmups): call()
                                sync(); totals.clear()
                                t = time.perf_counter()
                                for _ in range(a.repeats): call()
                                host = (time.perf_counter()-t)*1000
                                sync()
                                results[key] = dict(native_host_ms=totals, host_total_ms=host, residual_host_ms=host-sum(totals.values()))
                            finally:
                                for n, original in originals.items(): setattr(signatory.impl,n,original)
                        else:
                            for _ in range(a.warmups): call()
                            sync()
                            activities_requested = [torch.profiler.ProfilerActivity.CPU]
                            if a.device == 'cuda':
                                activities_requested.append(torch.profiler.ProfilerActivity.CUDA)
                            with torch.profiler.profile(activities=activities_requested) as prof:
                                for _ in range(3): call()
                                sync()
                            trace = '/tmp/signatory-{}-{}-{}-{}.json'.format(torch.__version__,a.device,name,key)
                            prof.export_chrome_trace(trace)
                            trace_data = json.loads(Path(trace).read_text())
                            results[key] = dict(trace=trace, iterations=3, **kernel_summary(trace_data), operators=[dict(name=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total) for e in prof.key_averages()])
                    out['cases'][name] = results
    out['runtime_library_paths'] = sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '.so' in line))
    Path(a.output).write_text(json.dumps(out, indent=2)+'\n')
    if 'unavailable' in out:
        print(out['unavailable'], file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
