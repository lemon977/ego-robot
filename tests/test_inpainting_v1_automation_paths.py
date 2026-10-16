"""Read-only physical alias adaptation cannot loosen evidence SHA checks."""
from pathlib import Path
from types import SimpleNamespace
import pytest

from chaoyang.inpainting_candidate import automation_paths as paths


def test_scoped_alias_check_preserves_exact_reference_and_restores(monkeypatch, tmp_path):
    original_ref = {'path': '/logical/old/image.png', 'bytes': 9, 'sha256': 'a'*64}
    native_calls = []
    def verify(reference, project_only=False):
        assert reference is original_ref and project_only is False
        native_calls.append(reference)
        return dict(reference)
    native = SimpleNamespace(_verify_exact_ref=verify, marker='unchanged')
    module = SimpleNamespace(audit=native)
    monkeypatch.setattr(paths, 'resolve_input_path', lambda value, root: tmp_path / 'image.png')
    with paths.seed_evidence_alias_reader(module, tmp_path):
        assert module.audit.marker == 'unchanged'
        assert module.audit._verify_exact_ref(original_ref, project_only=True) == original_ref
    assert module.audit is native and native_calls == [original_ref]


def test_outside_project_rejected_before_native_read(monkeypatch, tmp_path):
    native = SimpleNamespace(_verify_exact_ref=lambda *a, **k: pytest.fail('must not read outside'))
    module = SimpleNamespace(audit=native)
    monkeypatch.setattr(paths, 'resolve_input_path', lambda value, root: tmp_path.parent / 'other.png')
    with pytest.raises(ValueError, match='OUTSIDE_PHYSICAL_PROJECT'):
        with paths.seed_evidence_alias_reader(module, tmp_path):
            module.audit._verify_exact_ref({'path':'/outside'}, project_only=True)
    assert module.audit is native


def test_native_sha_error_is_not_swallowed(monkeypatch, tmp_path):
    def reject(*args, **kwargs):
        raise ValueError('EVIDENCE_REFERENCE_DRIFT')
    native = SimpleNamespace(_verify_exact_ref=reject)
    module = SimpleNamespace(audit=native)
    monkeypatch.setattr(paths, 'resolve_input_path', lambda value, root: tmp_path / 'data')
    with pytest.raises(ValueError, match='EVIDENCE_REFERENCE_DRIFT'):
        with paths.seed_evidence_alias_reader(module, tmp_path):
            module.audit._verify_exact_ref({'path':'/signed'}, project_only=True)
    assert module.audit is native
