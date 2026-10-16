"""Terminal-seal regression tests; no NAS reads or actual deletion."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def maintenance():
    path = Path(__file__).parents[2] / 'scripts/maintenance/relocate_pick_place_original199_v1.py'
    spec = importlib.util.spec_from_file_location('relocation_terminal_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def terminal_fixture(module, tmp_path, monkeypatch, *, deleted=0, drift=False):
    runtime = module.Runtime.__new__(module.Runtime)
    runtime.inputs = [dict(path=str(tmp_path/'input.json'), bytes=1, sha256='0'*64)]
    runtime.out = tmp_path
    runtime.writer = dict(pid=123, startticks=456)
    runtime.progress = dict(deleted_files=deleted, deleted_bytes=deleted*10)
    runtime.lock = (tmp_path/'lock').open('a+')
    captured = []
    monkeypatch.setattr(module, 'once', lambda path, value: captured.append((path, dict(value))))
    monkeypatch.setattr(module.H, 'pin', lambda path: dict(path=str(path), bytes=0, sha256='1'*64))
    monkeypatch.setattr(module, 'identity_metadata', lambda path: dict(size=2, inode=3))
    def verify(ref):
        if drift:
            raise RuntimeError('TEST_INPUT_SHA_DRIFT')
    monkeypatch.setattr(module.H, 'verify', verify)
    return runtime, captured


def test_success_still_verifies_inputs(maintenance, tmp_path, monkeypatch):
    runtime, captured = terminal_fixture(maintenance, tmp_path, monkeypatch)
    assert runtime.finish(dict(status='PASSED_TEST')) == 0
    assert captured[0][1]['status'] == 'PASSED_TEST'
    assert runtime.lock.closed


def test_success_turns_into_stopped_on_input_drift(maintenance, tmp_path, monkeypatch):
    runtime, captured = terminal_fixture(maintenance, tmp_path, monkeypatch, drift=True)
    assert runtime.finish(dict(status='PASSED_TEST')) == 2
    result = captured[0][1]
    assert result['status'] == 'STOPPED'
    assert result['pre_finish_status'] == 'PASSED_TEST'
    assert result['input_verification_failures'][0]['actual_metadata']['size'] == 2


def test_stopped_result_is_not_swallowed(maintenance, tmp_path, monkeypatch):
    runtime, captured = terminal_fixture(maintenance, tmp_path, monkeypatch, drift=True)
    assert runtime.finish(dict(status='STOPPED', failure='ORIGINAL_FAILURE')) == 2
    assert captured[0][1]['failure'] == 'ORIGINAL_FAILURE'
    assert captured[0][1]['status'] == 'STOPPED'


def test_partial_deletion_journal_survives_input_drift(maintenance, tmp_path, monkeypatch):
    runtime, captured = terminal_fixture(maintenance, tmp_path, monkeypatch, deleted=1, drift=True)
    journal = tmp_path/'DELETED_FILES_V1.jsonl'
    journal.write_text('{"status":"UNLINK_INTENT"}\n')
    assert runtime.finish(dict(status='STOPPED', failure='ORIGINAL_FAILURE')) == 2
    result = captured[0][1]
    assert result['status'] == 'PARTIAL_STOPPED'
    assert result['deleted_files'] == 1
    assert result['deletion_journal']['path'] == str(journal)
    assert journal.exists()


def test_changed_reader_cannot_reuse_old_validation(maintenance, monkeypatch):
    paths = maintenance.READER_FILES
    refs = [dict(path=str(p), bytes=10, sha256='1'*64) for p in paths]
    packet = dict(phase='R0_READONLY_VALIDATION', deletion_allowed=False, inputs=refs)
    validation = dict(status='PASSED_READONLY_STORAGE_VALIDATION',
        task_packet=dict(path='/unused/packet.json',bytes=0,sha256='0'*64))
    monkeypatch.setattr(maintenance.H,'verify',lambda ref: None)
    monkeypatch.setattr(maintenance.H,'load',lambda path: packet)
    monkeypatch.setattr(maintenance.H,'pin',lambda path: dict(path=str(path),bytes=10,sha256='2'*64))
    with pytest.raises(RuntimeError, match='CURRENT_STORAGE_READER_VERSION_NOT_VALIDATED'):
        maintenance.require_validated_reader_version(validation)


def test_matching_reader_version_is_accepted(maintenance, monkeypatch):
    refs = [dict(path=str(p),bytes=10,sha256='1'*64) for p in maintenance.READER_FILES]
    packet = dict(phase='R0_READONLY_VALIDATION', deletion_allowed=False, inputs=refs)
    monkeypatch.setattr(maintenance.H,'verify',lambda ref: None)
    monkeypatch.setattr(maintenance.H,'load',lambda path: packet)
    monkeypatch.setattr(maintenance.H,'pin',lambda path: next(r for r in refs if r['path']==str(path)))
    maintenance.require_validated_reader_version(dict(status='PASSED_READONLY_STORAGE_VALIDATION',
        task_packet=dict(path='/unused/packet.json',bytes=0,sha256='0'*64)))


@pytest.mark.parametrize('changed', [None, 'unregistered', 'wrong_phase', 'wrong_validation', 'deleted'])
def test_failed_cleanup_navigation_is_evidence_not_delete_authority(maintenance, monkeypatch, changed):
    result_ref = dict(path='/unused/C0/RESULT.json', bytes=10, sha256='1'*64)
    validation_ref = dict(path='/unused/R0/RESULT.json', bytes=20, sha256='2'*64)
    packet_ref = dict(path='/unused/C0/TASK_PACKET.json', bytes=30, sha256='3'*64)
    result = dict(status='STOPPED', failure='FINITE_DEADLINE', task_packet=packet_ref)
    packet = dict(phase='C0_RELOCATION_PROOF', deletion_allowed=False, inputs=[validation_ref])
    runtime = type('TestRuntime', (), {})()
    runtime.inputs = [result_ref]
    if changed == 'unregistered':
        runtime.inputs = []
    elif changed == 'wrong_phase':
        packet['phase'] = 'C1_EXACT_UNLINK'
    elif changed == 'wrong_validation':
        packet['inputs'] = []
    elif changed == 'deleted':
        result['deleted_files'] = 1
    monkeypatch.setattr(maintenance.H, 'verify', lambda ref: None)
    monkeypatch.setattr(maintenance.H, 'load', lambda path: result if str(path) == result_ref['path'] else packet)
    if changed is not None:
        with pytest.raises(RuntimeError, match='CLEANUP_STATUS_'):
            maintenance.cleanup_status_receipt(runtime, Path(result_ref['path']), validation_ref)
    else:
        status = maintenance.cleanup_status_receipt(runtime, Path(result_ref['path']), validation_ref)
        assert status['status'] == 'BLOCKED_REFERENCE_PROOF'
        assert status['locally_deleted_files'] == 0
        assert status['result'] == result_ref
        assert status['failure'] == 'FINITE_DEADLINE'
