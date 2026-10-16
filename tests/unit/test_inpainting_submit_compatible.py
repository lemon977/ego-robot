"""Synthetic attach checks: no publisher, GPU inventory, or model execution."""
from copy import deepcopy
import importlib.util
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chaoyang.governance import inpainting_candidate_registration as registration
from chaoyang.humanego_ict import project_mount_paths as mounts
from chaoyang.inpainting_candidate import automation as auto
from chaoyang.inpainting_candidate.contracts import STAGES, file_ref
from tests.test_inpainting_v1_automation_flow import project, save


@pytest.fixture
def attach(project, monkeypatch):
    request = auto.build_v1_request(project_root=project.root, session=project.raw / '050')
    request_ref = save(project.root / 'saved_request.json', request)
    code = save(project.root / 'frozen_code.json', {'synthetic': True})
    resume_path = project.root / 'explicit_resume.json'
    resume_ref = save(resume_path, {'schema': 'inpainting-v1-resume-candidates-v1', 'candidates': []})
    context = {'output_root': str(project.root / 'finite_submissions')}
    seen = []

    def compatible_spec(root, req, ref, *, execution_context, **extra):
        seen.append(extra)
        return {'task_id': 'synthetic_v1_compatible', 'mode': 'Automation',
                'runtime_code': [code], 'output_dir': str(project.root / 'sealed_output'),
                'component_signatures': {'050': {stage: stage for stage in STAGES}},
                'inputs': {'request': ref, **extra}, 'reuse_stages': {},
                'resume_plan': {'decisions': [{'session_id': '050', 'synthetic': True}]}}

    contexts = []

    def finite_context(root, *, wall_seconds, repair_rounds):
        contexts.append((root, wall_seconds, repair_rounds))
        return context

    builder = SimpleNamespace(_execution_context=lambda root, value: value,
                              temporal_execution_context=finite_context,
                              compatible_spec=compatible_spec)
    original = importlib.util.spec_from_file_location

    class Loader:
        def create_module(self, spec):
            return builder

        def exec_module(self, module):
            pass

    def definition(name, location, *args, **kwargs):
        if name == '_v1_compatible_spec_builder':
            return ModuleSpec(name, Loader())
        return original(name, location, *args, **kwargs)

    monkeypatch.setattr(importlib.util, 'spec_from_file_location', definition)
    monkeypatch.setattr(mounts, 'PROJECT_ALIASES', (project.root, project.root))
    monkeypatch.setattr(mounts, 'resolve_input_path', lambda path, root: Path(path))
    monkeypatch.setattr(registration, '_live', lambda *args: True)
    monkeypatch.setattr(registration.subprocess, 'run',
                        lambda *args, **kwargs: pytest.fail('attach launched a publisher'))
    spec = compatible_spec(project.root, request, request_ref, execution_context=context)
    packet = {**spec, 'run_signature': 'a' * 64,
              'sessions': [{'session_key': '050', 'native_frames': 3}],
              'writer': {'pid': 123, 'proc_start_ticks': 456}}
    packet_path = project.root / 'tasks/current' / spec['task_id'] / 'TASK_PACKET.json'
    packet_ref = save(packet_path, packet)
    state_ref = save(project.root / 'STATE.json', {'tasks': [], 'next_task': None})
    index_path = project.root / 'INDEX.json'
    index_ref = save(index_path, {'task_packets': []})
    save(project.root / 'docs/governance/CURRENT_STATUS_RECEIPT.json', {'files': {
        'task_state': state_ref, 'task_packet_index_current': index_ref}})
    model = {'schema': 'inpainting-v1-mco-stage-result-v1', 'mode': 'Automation',
             'task_id': spec['task_id'], 'run_signature': packet['run_signature'],
             'status': 'SEALED_COMPLETE', 'robot_video_produced': True,
             'sessions': [{'session_id': '050', 'session_key': '050', 'native_frames': 3,
                           'native_output_frames': 3, 'completed_frame_indices': [0, 1, 2],
                           'execution_complete': True, 'robot_video_produced': True}]}
    model_path = Path(spec['output_dir']) / 'MODEL_RESULT.json'
    result_path = Path(spec['output_dir']) / 'RESULT.json'

    def write_result(**overrides):
        value = {'task_id': spec['task_id'], 'run_signature': packet['run_signature'],
                 'status': 'PASSED', 'model_result_consumable': True,
                 'model_result': save(model_path, model)}
        value.update(overrides)
        save(result_path, value)
        return value

    def set_active():
        save(index_path, {'task_packets': [{'task_id': spec['task_id'],
                                           'execution_allowed': True,
                                           'packet_sha256': packet_ref['sha256']}]})

    def submit(*, resume=False):
        return registration.submit_v1_compatible_request(request, repo_root=project.root,
            execution_context=context, resume_manifest_path=resume_path if resume else None)

    write_result()
    return SimpleNamespace(submit=submit, write_result=write_result, set_active=set_active,
                           model=model, model_path=model_path, result_path=result_path,
                           packet=packet, packet_path=packet_path, resume_ref=resume_ref,
                           seen=seen, root=project.root, request=request, contexts=contexts)


@pytest.mark.parametrize('status', ['RUNNING', 'CLAIMED', 'WAIT_GPU_RESOURCE', 'PARTIAL',
                                  'SEALED_PARTIAL', 'SEALED_COMPLETE', None])
def test_nonterminal_or_unsealed_result_cannot_attach(attach, status):
    attach.write_result(status=status)
    with pytest.raises(registration.RegistrationError, match='NOT_TERMINAL'):
        attach.submit()


@pytest.mark.parametrize('status', ['FAILED', 'FAILED_RUNTIME_FINAL', 'TIMEOUT',
                                  'TIMED_OUT', 'DEADLINE_EXCEEDED'])
def test_failed_terminal_attaches_without_claiming_robot_even_with_consumable_flag(attach, status):
    attach.write_result(status=status, model_result_consumable=True,
                        model_result={'path': '/must/not/read/partial_model.json'})
    result = attach.submit()
    assert result['status'] == 'ATTACHED_TO_TERMINAL_V1_COMPATIBLE'
    assert result['terminal_status'] == status
    assert result['execution_started_here'] is False
    assert result['robot_video_produced'] is False


@pytest.mark.parametrize('change', ['missing_ref', 'missing_file', 'wrong_sha', 'wrong_task',
                                  'wrong_signature', 'partial', 'wrong_mode', 'incomplete',
                                  'wrong_frames', 'missing_frame', 'duplicate_session', 'no_sessions',
                                  'not_consumable'])
def test_passed_result_requires_bound_sealed_complete_model(attach, change):
    model = deepcopy(attach.model)
    overrides = {}
    if change == 'missing_ref':
        overrides['model_result'] = None
    elif change == 'wrong_sha':
        overrides['model_result'] = {**file_ref(attach.model_path), 'sha256': 'f' * 64}
    elif change == 'wrong_task':
        model['task_id'] = 'another_task'
    elif change == 'wrong_signature':
        model['run_signature'] = 'b' * 64
    elif change == 'partial':
        model['status'] = 'PARTIAL'
    elif change == 'wrong_mode':
        model['mode'] = 'Clean'
    elif change == 'incomplete':
        model['sessions'][0]['execution_complete'] = False
    elif change == 'wrong_frames':
        model['sessions'][0]['native_frames'] = 2
    elif change == 'missing_frame':
        model['sessions'][0]['completed_frame_indices'] = [0, 1]
        model['sessions'][0]['native_output_frames'] = 2
    elif change == 'duplicate_session':
        model['sessions'].append(deepcopy(model['sessions'][0]))
    elif change == 'no_sessions':
        model['sessions'] = []
    elif change == 'not_consumable':
        overrides['model_result_consumable'] = False
    attach.model.clear()
    attach.model.update(model)
    attach.write_result(**overrides)
    if change == 'missing_file':
        # Remove no data: redirect the declared reference to a never-created path.
        value = json.loads(attach.result_path.read_text())
        value['model_result']['path'] = str(attach.root / 'never_created' / 'MODEL_RESULT.json')
        save(attach.result_path, value)
    with pytest.raises(registration.RegistrationError, match='TERMINAL_MODEL'):
        attach.submit()


def test_verified_model_can_report_robot_only_when_model_declares_it(attach):
    result = attach.submit()
    assert result['robot_video_produced'] is True
    attach.model['robot_video_produced'] = False
    attach.write_result()
    assert attach.submit()['robot_video_produced'] is False


@pytest.mark.parametrize('active', [True, False])
def test_explicit_resume_manifest_repeat_attach_does_not_launch_or_rewrite(attach, active):
    if active:
        attach.set_active()
    before = {path: path.read_bytes() for path in
              (attach.packet_path, attach.result_path, attach.model_path)}
    first = attach.submit(resume=True)
    second = attach.submit(resume=True)
    expected = 'ATTACHED_TO_ACTIVE_V1_COMPATIBLE' if active else 'ATTACHED_TO_TERMINAL_V1_COMPATIBLE'
    assert first['status'] == second['status'] == expected
    assert first['execution_started_here'] is second['execution_started_here'] is False
    assert first['resume_manifest'] == attach.resume_ref
    assert attach.seen[-1]['resume_manifest'] == attach.resume_ref
    assert all(path.read_bytes() == value for path, value in before.items())


def test_default_single_round_gets_a_new_finite_context_without_old_deadline(attach):
    result = registration.submit_v1_compatible_request(attach.request, repo_root=attach.root,
                                                     wall_seconds=120)
    assert result['status'] == 'ATTACHED_TO_TERMINAL_V1_COMPATIBLE'
    assert attach.contexts == [(attach.root, 120, 1)]
    assert result['execution_started_here'] is False
