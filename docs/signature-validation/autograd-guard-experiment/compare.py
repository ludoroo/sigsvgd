"""Exact safety comparison: exit 0 equal, 1 different, 2 invalid input."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def differences(a, b, path=''):
    if type(a) is not type(b):
        return [(path, a, b)]
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError(f'{path}: unequal keys')
        return [d for k in a for d in differences(a[k], b[k], path+'/'+k)]
    if isinstance(a, list):
        if len(a) != len(b):
            raise ValueError(f'{path}: unequal list lengths')
        return [d for i, (x, y) in enumerate(zip(a, b))
                for d in differences(x, y, path+'/'+str(i))]
    return [] if a == b else [(path, a, b)]


def read_records(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate key: {key}')
            result[key] = value
        return result
    records = json.loads(Path(path).read_text(), object_pairs_hook=unique,
                        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    if not isinstance(records, dict) or len(records) != 253:
        raise ValueError('expected 253 safety records')
    expected = {f'{device}/gradcheck/{name}' for device in ('cpu', 'cuda')
                for name in ('signature', 'logsignature', 'Path')}
    actual = {k for k in records if '/gradcheck/' in k}
    if actual != expected or any(records[k] is not True for k in expected):
        raise ValueError('expected all six named gradchecks to pass')
    return records


def compare(a, b):
    ds = differences(read_records(a), read_records(b))
    summary = {'records': 253, 'gradchecks_each': 6,
               'field_differences': len(ds),
               'fields': dict(Counter(p.rsplit('/', 1)[-1] for p, _, _ in ds))}
    return summary, ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('control')
    parser.add_argument('candidate')
    parser.add_argument('--diff-output', type=Path)
    args = parser.parse_args()
    try:
        summary, ds = compare(args.control, args.candidate)
        if args.diff_output:
            args.diff_output.write_text(json.dumps(ds, indent=2, allow_nan=False)+'\n')
        print(json.dumps(summary, indent=2))
    except (OSError, ValueError, TypeError) as exc:
        print(f'Invalid safety comparison: {exc}', file=sys.stderr)
        sys.exit(2)
    sys.exit(bool(ds))
