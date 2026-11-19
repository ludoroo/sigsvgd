"""Protect the native extension's PyTorch ABI requirement."""

import runpy
from pathlib import Path

import pytest

check_torch_version = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/install_signatory.py")
)["check_torch_version"]


@pytest.mark.parametrize("version", ["2.13.0", "2.13.0+cpu", "2.13.0+cu130"])
def test_accepts_pinned_torch_with_local_build_suffix(version):
    check_torch_version(version)


@pytest.mark.parametrize(
    "version", ["1.9.0", "2.12.0", "2.13.0rc1", "2.13.1", "2.15.0"]
)
def test_rejects_torch_versions_project_install_would_replace(version):
    with pytest.raises(RuntimeError, match="Install the pinned PyTorch"):
        check_torch_version(version)


def test_linux_patch_rejects_unexpected_source(tmp_path):
    api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/install_signatory.py'))
    with pytest.raises(RuntimeError, match='source'):
        api['patch_linux_source'](tmp_path)


def test_patch_validates_all_files_before_writing(tmp_path):
    import hashlib
    import json
    api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/install_signatory.py'))
    patch = api['patch_linux_source']
    (tmp_path / 'patches').mkdir()
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'a').write_text('anchor\n')
    spec = {'sha256': hashlib.sha256(b'anchor\n').hexdigest(),
            'patched_sha256': hashlib.sha256(b'anchor\nguard\n').hexdigest(),
            'edits': [{'old': 'anchor\n', 'new': 'anchor\nguard\n'}]}
    manifest = tmp_path / 'patches/signatory-1.2.6.1.9.0-linux-autograd-v2.json'
    manifest.write_text(json.dumps({'a': spec, 'missing': spec}))
    with pytest.raises(RuntimeError, match='source'):
        patch(source, manifest)
    assert (source / 'a').read_text() == 'anchor\n'
    manifest.write_text(json.dumps({'a': spec}))
    patch(source, manifest)
    assert (source / 'a').read_text() == 'anchor\nguard\n'
    with pytest.raises(RuntimeError, match='source'):
        patch(source, manifest)
    (source / 'a').write_text('anchor\n')
    spec['edits'][0]['old'] = 'absent'
    manifest.write_text(json.dumps({'a': spec}))
    with pytest.raises(RuntimeError, match='conflict'):
        patch(source, manifest)


def test_linux_patch_contains_only_four_reviewed_guards():
    import json
    manifest = Path(__file__).resolve().parents[1] / 'scripts/patches/signatory-1.2.6.1.9.0-linux-autograd-v2.json'
    spec = json.loads(manifest.read_text())
    assert set(spec) == {'src/signature.cpp', 'src/tensor_algebra_ops.cpp'}
    for item in spec.values():
        assert len(item['edits']) == 2
        for edit in item['edits']:
            assert edit['new'] == edit['old'] + '        at::AutoDispatchBelowADInplaceOrView autograd_only_guard;\n'


def test_linux_download_hash_mismatch_does_not_build(monkeypatch):
    api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/install_signatory.py'))
    monkeypatch.setattr(api['urllib'].request, 'urlretrieve', lambda url, path: Path(path).write_bytes(b'wrong archive'))
    with pytest.raises(RuntimeError, match='SHA256 mismatch'):
        api['build_linux_source'](['pip', 'signatory==1.2.6.1.9.0'])


def test_macos_keeps_original_source_build(monkeypatch, tmp_path):
    from types import SimpleNamespace
    api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/install_signatory.py'))
    main = api['main']
    g = main.__globals__
    lib = tmp_path / 'lib'
    lib.mkdir()
    (lib / 'libomp.dylib').touch()
    monkeypatch.setattr(api['sys'], 'platform', 'darwin')
    monkeypatch.setattr(api['sys'], 'argv', ['install_signatory.py'])
    monkeypatch.setattr(api['shutil'], 'which', lambda tool: '/mock/' + tool)
    calls = []
    monkeypatch.setitem(g, 'require_prerequisites', lambda: SimpleNamespace(__file__=str(tmp_path / '__init__.py')))
    monkeypatch.setitem(g, 'mac_build_environment', lambda root: {'MAC': 'unchanged'})
    monkeypatch.setitem(g, 'patch_macos_extensions', lambda: calls.append('mac-patch'))
    monkeypatch.setitem(g, 'smoke_test', lambda: None)
    monkeypatch.setitem(g, 'build_linux_source', lambda args: pytest.fail('Linux patch used on macOS'))
    monkeypatch.setattr(api['subprocess'], 'run', lambda args, **kw: calls.append((args,kw)))
    main()
    assert calls[0][0][-1] == 'signatory==1.2.6.1.9.0'
    assert calls[0][1]['env'] == {'MAC': 'unchanged'}
    assert calls[1] == 'mac-patch'


def test_patch_output_hash_mismatch(tmp_path):
    import hashlib
    import json
    api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/install_signatory.py'))
    (tmp_path/'source').write_text('original')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'source': {'sha256': hashlib.sha256(b'original').hexdigest(), 'patched_sha256': 'wrong', 'edits': [{'old':'original','new':'changed'}]}}))
    with pytest.raises(RuntimeError,match='output mismatch'):
        api['patch_linux_source'](tmp_path,manifest)
    assert (tmp_path/'source').read_text() == 'original'


def test_linux_routes_to_verified_builder(monkeypatch):
    api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/install_signatory.py'))
    g = api['main'].__globals__
    monkeypatch.setattr(api['sys'],'platform','linux')
    monkeypatch.setattr(api['sys'],'argv',['installer'])
    calls=[]
    monkeypatch.setitem(g,'require_prerequisites',lambda: None)
    monkeypatch.setitem(g,'smoke_test',lambda: None)
    monkeypatch.setitem(g,'build_linux_source',lambda args: calls.append(args))
    api['main']()
    assert len(calls)==1 and calls[0][-1]=='signatory==1.2.6.1.9.0'


def test_safe_extraction_prerequisite_and_argument_validation(monkeypatch):
    api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/install_signatory.py'))
    with pytest.raises(RuntimeError,match='arguments'):
        api['build_linux_source']([])
    monkeypatch.delattr(api['tarfile'],'data_filter')
    with pytest.raises(RuntimeError,match='3.10.12'):
        api['build_linux_source'](['signatory==1.2.6.1.9.0'])


def test_download_error_is_actionable(monkeypatch):
    api = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/install_signatory.py'))
    def fail(*args):
        raise OSError('network unavailable')
    monkeypatch.setattr(api['urllib'].request,'urlretrieve',fail)
    with pytest.raises(RuntimeError,match='Signatory source download/extraction/build file error.*network unavailable'):
        api['build_linux_source'](['signatory==1.2.6.1.9.0'])
