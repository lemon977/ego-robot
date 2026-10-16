"""CPU checks for opt-in submission, frozen deadlines and round Mask receipts."""
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import importlib.util
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from chaoyang.inpainting_candidate import automation as auto, entry, v1_mco_worker
from chaoyang.inpainting_candidate.contracts import STAGES, canonical_sha, file_ref
from chaoyang.pick_place_1008.visual import effective_clean_masks
from tests.test_inpainting_v1_automation_flow import project, save
from tests.unit.test_inpainting_v1_mask_causal_wiring import causal_fixture

SINGLE_ROUND_RECIPE = Path('inpainting_optimization/incremental_v1_8h/DEFAULT_SINGLE_ROUND_EXECUTION_RECIPE.json')


@pytest.fixture
def specs(project, monkeypatch):
    source = Path(__file__).resolve().parents[2] / 'scripts/maintenance/inpainting_v1_automation_specs.py'
    definition = importlib.util.spec_from_file_location('_test_temporal_specs', source)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    from chaoyang.governance import inpainting_candidate_registration as registration
    code = save(project.root / 'code.json', {'synthetic': True})
    save(project.root / 'scripts/maintenance/inpainting_v1_automation_specs.py', {})
    save(project.root / 'scripts/humanego_ict/production_resume_runner_v1.py', {})
    monkeypatch.setattr(registration, 'mco_route_contract', lambda root: {
        'inpainting_original_v1_mco_candidate_v1': {'code_closure': [code]}})
    monkeypatch.setattr(module, 'component_closures', lambda root: {s: [code] for s in STAGES})
    load = module.load_json

    def read(path):
        if Path(path).name == 'MANIFEST.json':
            return {**load(path), 'models': {'weights': [], 'clean_SETTINGS': {}}}
        if Path(path).name == 'FROZEN_INPUTS.json':
            return {'robot_assets': []}
        return load(path)

    monkeypatch.setattr(module, 'load_json', read)
    save(project.root / 'inpainting_optimization/temporal_repair_3h/EXECUTION_RECIPE.json', {
        'schema': 'inpainting-v1-temporal-repair-execution-recipe-v1',
        'recipe_id': 'SYNTHETIC_TWO_ROUND', 'repair_rounds': 2, 'plan_revision': 'SYNTHETIC_R2',
        'objective': 'Synthetic finite candidate', 'output_root': '_run/new_candidates',
        'review_root': 'review/new_candidates', 'wall_seconds_max': 3600})
    save(project.root / SINGLE_ROUND_RECIPE, {
        'schema': 'inpainting-v1-temporal-repair-execution-recipe-v1',
        'recipe_id': 'SYNTHETIC_SINGLE_ROUND', 'repair_rounds': 1, 'plan_revision': 'SYNTHETIC_R1',
        'objective': 'Synthetic finite single-round compatible candidate, not exact original V1 replay',
        'output_root': '_run/inpainting_incremental_8h_20261016/default_compatible',
        'review_root': 'review/inpainting_optimization/INCREMENTAL_V1_8H/DEFAULT_AUTOMATION',
        'wall_seconds_max': 3600, 'default_baseline': 'INPAINTING_RAW_TO_ROBOT_V1',
        'default_baseline_changed': False, 'candidate_adopted': False})
    save(project.root / 'inpainting_optimization/residual_1h/EXECUTION_RECIPE.json', {
        'schema': 'inpainting-v1-temporal-repair-execution-recipe-v1',
        'recipe_id': 'SYNTHETIC_THREE_ROUND', 'repair_rounds': 3, 'plan_revision': 'SYNTHETIC_R3',
        'objective': 'Synthetic finite three-round compatible candidate',
        'output_root': '_run/new_three_round_candidates', 'review_root': 'review/new_three_round_candidates',
        'wall_seconds_max': 3600})
    return module


def test_default_request_remains_original_and_two_round_option_is_signed(project):
    one = auto.build_v1_request(project_root=project.root, session=project.raw / '050')
    assert one == auto.build_v1_request(project_root=project.root, session=project.raw / '050', repair_rounds=1)
    two = entry.build_request(project_root=project.root, session=project.raw / '050',
                              mode='v1-compatible', repair_rounds=2)
    assert 'repair_rounds' not in one and two['repair_rounds'] == 2
    assert one['submission_input_sha256'] != two['submission_input_sha256']
    auto.verify_v1_request(one, project_root=project.root)
    two['repair_rounds'] = 1
    with pytest.raises(auto.AutomationError, match='SIGNATURE'):
        auto.verify_v1_request(two, project_root=project.root)


@pytest.mark.parametrize('value', [0, 4, True, '2', 2.0])
def test_invalid_rounds_rejected_before_raw_read(project, value):
    with pytest.raises(auto.AutomationError, match='REPAIR_ROUNDS'):
        auto.build_v1_request(project_root=project.root, session=project.raw / '050', repair_rounds=value)


def test_raw_auto_cannot_take_second_repair_round(project):
    with pytest.raises(entry.EntryError, match='V1_COMPATIBLE'):
        entry.build_request(project_root=project.root, session=project.raw / '050', repair_rounds=2)


def test_only_mask_consumers_change_when_rounds_change(project):
    one = auto.build_v1_request(project_root=project.root, session=project.raw / '050')
    two = auto.build_v1_request(project_root=project.root, session=project.raw / '050', repair_rounds=2)
    code = {stage: [{'same_frozen_code': stage}] for stage in STAGES}
    a, b = auto.component_signatures(one, code), auto.component_signatures(two, code)
    assert {s for s in STAGES if a['050'][s] != b['050'][s]} == {'Mask', 'Clean', 'Composite'}
    with pytest.raises(auto.AutomationError, match='CONFIG_DRIFT'):
        auto.component_signatures(two, code, stage_config={'Mask': {'repair_rounds': 1}})


def test_cli_forwards_two_rounds_and_wall_limit(project, monkeypatch, capsys):
    from chaoyang.cli import inpainting_submit
    from chaoyang import paths
    from chaoyang.governance import inpainting_candidate_registration as registration
    seen = []
    monkeypatch.setattr(paths, 'get_paths', lambda: SimpleNamespace(repo_root=project.root))
    monkeypatch.setattr(registration, 'submit_v1_compatible_request',
        lambda req, **kw: seen.append((req, kw)) or {'status': 'SYNTHETIC_PREVIEW'})
    assert inpainting_submit.main(['--session', str(project.raw / '050'), '--mode', 'v1-compatible',
        '--repair-rounds', '2', '--wall-seconds', '120', '--preview']) == 0
    assert seen[0][0]['repair_rounds'] == 2 and seen[0][1]['wall_seconds'] == 120
    assert seen[0][1]['preview'] is True
    assert json.loads(capsys.readouterr().out)['status'] == 'SYNTHETIC_PREVIEW'


@pytest.mark.parametrize('arguments', [[], ['--mode', 'raw-auto'], ['--mode', 'v1-compatible', '--wall-seconds', '3601']])
def test_cli_rejects_invalid_mode_or_wall_without_reading(arguments):
    from chaoyang.cli import inpainting_submit
    with pytest.raises(SystemExit):
        inpainting_submit.main(['--session', '/not/read', '--repair-rounds', '2', *arguments])


def test_new_context_has_stable_identity_and_does_not_read_old_deadline(project, specs, monkeypatch):
    two = auto.build_v1_request(project_root=project.root, session=project.raw / '050', repair_rounds=2)
    req_ref = save(project.root / 'REQUEST.json', two)
    monkeypatch.setattr(specs, 'deadline', lambda root: pytest.fail('old eight-hour deadline read'))
    first_context = specs.temporal_execution_context(project.root, wall_seconds=120)
    second_context = {**first_context, 'global_deadline_at':
        (datetime.fromisoformat(first_context['global_deadline_at']) + timedelta(seconds=30)).isoformat()}
    first = specs.compatible_spec(project.root, two, req_ref, execution_context=first_context)
    second = specs.compatible_spec(project.root, two, req_ref, execution_context=second_context)
    assert first['task_id'] == second['task_id']
    assert first['global_deadline_at'] != second['global_deadline_at']
    assert first['budgets']['wall_seconds_max'] == 120
    assert first['component_config']['Mask']['repair_rounds'] == 2
    assert Path(first['output_dir']).is_relative_to(project.root / '_run/new_candidates')
    assert all('bootstrap/launch.py' not in ref['path'] for ref in first['runtime_code'])
    assert 'execution_recipe' in first['inputs']


def test_single_round_context_ignores_ended_legacy_deadline_and_is_clock_stable(project, specs, monkeypatch):
    legacy_path=project.root/'inpainting_optimization/eight_hour_v1/AUTOMATION_STATE.json'
    ended=save(legacy_path,{'status':'COMPLETE_NOT_ADOPTED',
        'global_deadline_at':'2026-10-14T20:54:45.469204+00:00'})
    with pytest.raises(ValueError,match='TIMEBOX_EXPIRED'):
        specs.deadline(project.root)
    request=auto.build_v1_request(project_root=project.root,session=project.raw/'050')
    assert 'repair_rounds'not in request
    request_ref=save(project.root/'REQUEST.json',request)
    monkeypatch.setattr(specs,'deadline',lambda root:pytest.fail('ended historical deadline consumed'))
    first_context=specs.temporal_execution_context(project.root,wall_seconds=120,repair_rounds=1)
    later={**first_context,'global_deadline_at':
        (datetime.fromisoformat(first_context['global_deadline_at'])+timedelta(seconds=30)).isoformat()}
    first=specs.compatible_spec(project.root,request,request_ref,execution_context=first_context)
    second=specs.compatible_spec(project.root,request,request_ref,execution_context=later)
    assert first['task_id']==second['task_id']and first['component_signatures']==second['component_signatures']
    assert first['global_deadline_at']!=second['global_deadline_at']
    assert first['component_config']['Mask']['repair_rounds']==1
    assert first_context['execution_recipe']==file_ref(project.root/SINGLE_ROUND_RECIPE)
    assert Path(first['output_dir']).is_relative_to(project.root/'_run/inpainting_incremental_8h_20261016/default_compatible')
    assert Path(first['review_dir']).is_relative_to(project.root/'review/inpainting_optimization/INCREMENTAL_V1_8H/DEFAULT_AUTOMATION')
    assert first['budgets']['wall_seconds_max']==120 and first['config']['human_points_added']==0
    assert first['config']['session_specific_branches']is False
    assert file_ref(legacy_path)==ended


def test_single_round_wrapper_context_preserves_all_existing_component_signatures(project,specs,monkeypatch):
    request=auto.build_v1_request(project_root=project.root,session=project.raw/'050')
    ref=save(project.root/'REQUEST.json',request)
    # Historical wrapper comparison is synthetic only: no old Task or state
    # deadline is reopened or modified by this test.
    save(project.root/'_run/inpainting_v1_automation_20261014/bootstrap/launch.py',{'synthetic':True})
    monkeypatch.setattr(specs,'deadline',lambda root:(datetime.now(timezone.utc)+timedelta(seconds=120)).isoformat())
    legacy=specs.compatible_spec(project.root,request,ref)
    context=specs.temporal_execution_context(project.root,wall_seconds=120,repair_rounds=1)
    fresh=specs.compatible_spec(project.root,request,ref,execution_context=context)
    assert legacy['component_signatures']==fresh['component_signatures']
    assert legacy['component_code']==fresh['component_code']and legacy['component_weights']==fresh['component_weights']
    assert legacy['component_config']==fresh['component_config']
    assert fresh['component_config']['Clean']['pixel_propagation']=='FROZEN'
    assert 'execution_recipe'not in fresh['component_config']['Clean']
    assert legacy['task_id']!=fresh['task_id']


def test_round_recipes_are_isolated_and_r1_navigation_does_not_change_r2_r3(project,specs):
    expected={1:SINGLE_ROUND_RECIPE,2:Path('inpainting_optimization/temporal_repair_3h/EXECUTION_RECIPE.json'),
              3:Path('inpainting_optimization/residual_1h/EXECUTION_RECIPE.json')}
    requests={rounds:auto.build_v1_request(project_root=project.root,session=project.raw/'050',repair_rounds=rounds)
              for rounds in expected}
    refs={rounds:save(project.root/('REQUEST_%d.json'%rounds),request)for rounds,request in requests.items()}
    contexts={rounds:specs.temporal_execution_context(project.root,wall_seconds=120,repair_rounds=rounds)
              for rounds in expected}
    before={rounds:specs.compatible_spec(project.root,requests[rounds],refs[rounds],execution_context=context)
            for rounds,context in contexts.items()}
    assert specs.temporal_execution_context(project.root,wall_seconds=120)['execution_recipe']==file_ref(project.root/expected[2])
    assert len({value['task_id']for value in before.values()})==3
    assert len({value['output_dir']for value in before.values()})==3
    for rounds,relative in expected.items():
        assert specs._repair_recipe_path(project.root,rounds)==project.root/relative
        assert contexts[rounds]['execution_recipe']==file_ref(project.root/relative)
        assert before[rounds]['component_config']['Mask']['repair_rounds']==rounds
    for left,right in [(1,2),(1,3),(2,3)]:
        a,b=before[left]['component_signatures']['050'],before[right]['component_signatures']['050']
        assert {stage for stage in STAGES if a[stage]!=b[stage]}=={'Mask','Clean','Composite'}
    old_refs={rounds:file_ref(project.root/expected[rounds])for rounds in (2,3)}
    first=specs.load_json(project.root/SINGLE_ROUND_RECIPE)
    save(project.root/SINGLE_ROUND_RECIPE,{**first,'objective':'Synthetic R1 navigation-only revision'})
    for rounds in (2,3):
        after=specs.compatible_spec(project.root,requests[rounds],refs[rounds],execution_context=contexts[rounds])
        assert after['task_id']==before[rounds]['task_id']
        assert after['component_signatures']==before[rounds]['component_signatures']
        assert file_ref(project.root/expected[rounds])==old_refs[rounds]
    changed=specs.compatible_spec(project.root,requests[1],refs[1],execution_context=
        specs.temporal_execution_context(project.root,wall_seconds=120,repair_rounds=1))
    assert changed['task_id']!=before[1]['task_id']
    assert changed['component_signatures']==before[1]['component_signatures']


@pytest.mark.parametrize('request_round,recipe_round',[(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)])
def test_round_request_cannot_borrow_another_round_recipe(project,specs,request_round,recipe_round):
    request=auto.build_v1_request(project_root=project.root,session=project.raw/'050',repair_rounds=request_round)
    ref=save(project.root/'REQUEST.json',request)
    context=specs.temporal_execution_context(project.root,wall_seconds=120,repair_rounds=recipe_round)
    with pytest.raises(ValueError,match='REQUEST_EXECUTION_RECIPE_ROUNDS_DRIFT'):
        specs.compatible_spec(project.root,request,ref,execution_context=context)


@pytest.mark.parametrize('value',[0,3601,True,1.0,-1])
def test_single_round_recipe_enforces_existing_finite_budget(project,specs,value):
    with pytest.raises(ValueError,match='WALL_BUDGET'):
        specs.temporal_execution_context(project.root,repair_rounds=1,wall_seconds=value)


@pytest.mark.parametrize('value',[0,4,True,1.0,'1'])
def test_round_recipe_path_rejects_non_integer_or_unsupported_rounds(project,specs,value):
    with pytest.raises(ValueError,match='FINITE_RECIPE_ROUNDS_REQUIRED'):
        specs._repair_recipe_path(project.root,value)


def test_single_round_recipe_requires_exact_path_and_live_deadline(project,specs):
    context=specs.temporal_execution_context(project.root,repair_rounds=1,wall_seconds=120)
    copy=save(project.root/'COPIED_RECIPE.json',specs.load_json(project.root/SINGLE_ROUND_RECIPE))
    with pytest.raises(ValueError,match='FINITE_EXECUTION_RECIPE_REQUIRED'):
        specs._execution_context(project.root,{**context,'execution_recipe':copy})
    with pytest.raises(ValueError,match='TIMEBOX_EXPIRED'):
        specs._execution_context(project.root,{**context,'global_deadline_at':'2026-10-14T20:54:45.469204+00:00'})


def test_single_round_real_recipe_is_candidate_only_and_has_no_algorithm_overrides():
    path=Path(__file__).resolve().parents[2]/SINGLE_ROUND_RECIPE
    recipe=json.loads(path.read_text())
    assert recipe['schema']=='inpainting-v1-temporal-repair-execution-recipe-v1'
    assert recipe['repair_rounds']==1 and recipe['wall_seconds_max']==3600
    assert recipe['output_root']=='_run/inpainting_incremental_8h_20261016/default_compatible'
    assert recipe['review_root']=='review/inpainting_optimization/INCREMENTAL_V1_8H/DEFAULT_AUTOMATION'
    assert recipe['default_baseline']=='INPAINTING_RAW_TO_ROBOT_V1'
    assert recipe['default_baseline_changed']is recipe['candidate_adopted']is False
    assert recipe['manual_points_added']==0 and recipe['original_v1_seeds_allowed']is True
    assert recipe['training_authorized']is recipe['new_weights_authorized']is recipe['raw_auto_proved']is False
    assert not any(key in recipe for key in ['Mask','Clean','model_weights','settings','glove_text_prompt','pixel_propagation'])


def _provide_builder(monkeypatch, specs):
    original = importlib.util.spec_from_file_location

    class Loader:
        def create_module(self, spec):
            return specs
        def exec_module(self, module):
            pass

    def definition(name, location, *args, **kwargs):
        if name == '_v1_compatible_spec_builder':
            return ModuleSpec(name, Loader())
        return original(name, location, *args, **kwargs)

    monkeypatch.setattr(importlib.util, 'spec_from_file_location', definition)


def test_mount_alias_request_rebuild_keeps_opt_in_rounds(project, monkeypatch):
    from chaoyang.governance import inpainting_candidate_registration as registration
    from chaoyang.humanego_ict import project_mount_paths as mounts
    request = auto.build_v1_request(project_root=project.root, session=project.raw / '050', repair_rounds=2)
    alias = project.root / 'alias'
    alias.mkdir()
    monkeypatch.setattr(mounts, 'PROJECT_ALIASES', (project.root, alias))
    monkeypatch.setattr(mounts, 'resolve_input_path', lambda path, root: Path(path))
    seen = []

    def rebuild(**kwargs):
        seen.append(kwargs)
        raise RuntimeError('SYNTHETIC_REBUILD_CAPTURED')

    monkeypatch.setattr(auto, 'build_v1_request', rebuild)
    with pytest.raises(RuntimeError, match='REBUILD_CAPTURED'):
        registration.submit_v1_compatible_request(request, repo_root=project.root, preview=True)
    assert seen[0]['repair_rounds'] == 2


@pytest.mark.parametrize('active', [True, False])
def test_repeat_submission_across_clock_attaches_without_second_writer(project, specs, monkeypatch, active):
    from chaoyang.governance import inpainting_candidate_registration as registration
    from chaoyang.humanego_ict import project_mount_paths as mounts
    _provide_builder(monkeypatch, specs)
    monkeypatch.setattr(mounts, 'PROJECT_ALIASES', (project.root, project.root))
    monkeypatch.setattr(mounts, 'resolve_input_path', lambda path, root: Path(path))
    monkeypatch.setattr(registration, '_live', lambda *args: True)
    monkeypatch.setattr(registration.subprocess, 'run', lambda *args, **kwargs: pytest.fail('second publisher started'))
    request = auto.build_v1_request(project_root=project.root, session=project.raw / '050', repair_rounds=2)
    request_ref = save(project.root / 'saved_request.json', request)
    context = specs.temporal_execution_context(project.root, wall_seconds=120)
    spec = specs.compatible_spec(project.root, request, request_ref, execution_context=context)
    result_ref = save(Path(spec['output_dir']) / 'RESULT.json', {
        'task_id': spec['task_id'], 'run_signature': 'a' * 64, 'status': 'FAILED_RUNTIME_FINAL'})
    packet = {**spec, 'run_signature': 'a' * 64, 'inputs': {'request': request_ref},
              'writer': {'pid': 123, 'proc_start_ticks': 456}}
    packet_path = project.root / 'tasks/current' / spec['task_id'] / 'TASK_PACKET.json'
    packet_ref = save(packet_path, packet)
    state_ref = save(project.root / 'STATE.json', {'tasks': [], 'next_task': None})
    index_ref = save(project.root / 'INDEX.json', {'task_packets': [{
        'task_id': spec['task_id'], 'execution_allowed': True, 'packet_sha256': packet_ref['sha256']}] if active else []})
    save(project.root / 'docs/governance/CURRENT_STATUS_RECEIPT.json', {'files': {
        'task_state': state_ref, 'task_packet_index_current': index_ref}})
    first = registration.submit_v1_compatible_request(request, repo_root=project.root, execution_context=context)
    later = {**context, 'global_deadline_at':
        (datetime.fromisoformat(context['global_deadline_at']) + timedelta(seconds=30)).isoformat()}
    second = registration.submit_v1_compatible_request(request, repo_root=project.root, execution_context=later)
    expected = 'ATTACHED_TO_ACTIVE_V1_COMPATIBLE' if active else 'ATTACHED_TO_TERMINAL_V1_COMPATIBLE'
    assert first['status'] == second['status'] == expected
    assert first['task_id'] == second['task_id'] == spec['task_id']
    assert first['execution_started_here'] is second['execution_started_here'] is False
    assert first['robot_video_produced'] is second['robot_video_produced'] is False
    assert json.loads((Path(spec['output_dir']) / 'RESULT.json').read_text())['run_signature'] == 'a' * 64


def test_stable_execute_spec_delegates_to_existing_mco_publisher(project, specs, monkeypatch):
    from chaoyang.humanego_ict import governance_mount_compat as compat
    original = importlib.util.spec_from_file_location
    seen = []
    publisher = ModuleType('_synthetic_existing_mco')
    publisher.main = lambda argv: seen.append(argv) or 19
    monkeypatch.setattr(compat, 'install_governance_mount_compat', lambda root: seen.append(root))

    class Loader:
        def create_module(self, spec):
            return publisher
        def exec_module(self, module):
            pass

    def definition(name, location, *args, **kwargs):
        if name == '_existing_inpainting_stable_publisher':
            assert Path(location) == project.root / 'scripts/maintenance/inpainting_v1_mco_task.py'
            return ModuleSpec(name, Loader())
        return original(name, location, *args, **kwargs)

    monkeypatch.setattr(importlib.util, 'spec_from_file_location', definition)
    reference = save(project.root / 'EXECUTE_SPEC.json', {'synthetic': True})
    assert specs.execute_spec(reference['path'], root=project.root) == 19
    assert seen == [project.root, ['--spec', reference['path']]]


def test_mask_worker_seals_round_one_and_final_arrays(project, monkeypatch):
    from chaoyang.inpainting_candidate import automation_mask, automation_paths, automation_worker
    from chaoyang.pick_place_1008 import visual, mask_seeds
    shape = (2, 8, 8)
    rgb = np.zeros((*shape, 3), np.uint8)
    base = {role: np.zeros(shape, bool) for role in
            ('core', 'sam_core', 'removal', 'equipment', 'protected_visible', 'unknown')}
    base['metadata'] = {'point_tracking': {'status': 'COMPLETE'}}
    round_one = {k: v.copy() for k, v in base.items() if isinstance(v, np.ndarray)}
    final = {k: v.copy() for k, v in round_one.items()}
    for role in ('core', 'sam_core', 'removal'):
        round_one[role][:, 2, 2] = True
        final[role][:, 2, 2:4] = True
    round_one['effective_removal'], _ = effective_clean_masks(round_one['removal'], round_one['protected_visible'])
    final['effective_removal'], _ = effective_clean_masks(final['removal'], final['protected_visible'])
    final['metadata'] = {}
    seed = {'synthetic_seed': True}
    monkeypatch.setattr(automation_paths, 'seed_evidence_alias_reader', lambda *args: nullcontext())
    monkeypatch.setattr(mask_seeds, 'load_visible_seed', lambda *args: {'seed_reference': seed, 'objects': []})
    monkeypatch.setattr(visual, 'build_visual_masks', lambda *args, **kwargs: base)
    seen = []

    def repair(**kwargs):
        seen.append(kwargs)
        return final, {'input_bindings': {}}, [], np.zeros((2, 2, 1, 2)), round_one

    monkeypatch.setattr(automation_mask, 'repair_current_masks', repair)
    packet = {'project_root': str(project.root), 'component_config': {'Mask': {'repair_rounds': 2}},
              'component_weights': {'Mask': [{'path': '/frozen/sam3.1_multiplex.pt', 'sha256': 'b' * 64}]}}
    builder = automation_worker.V1Builders(packet, lambda: None)
    builder.sam_budget = SimpleNamespace(total=0, receipt=lambda: {'synthetic': True})
    monkeypatch.setattr(builder, '_session', lambda upstream: SimpleNamespace())
    monkeypatch.setattr(builder, '_rgb', lambda upstream: rgb)
    monkeypatch.setattr(builder, '_mapping', lambda upstream: [])
    monkeypatch.setattr(builder, '_models', lambda stage: {'sam': object()})
    root = project.root / 'mask_output'
    receipt = builder.mask(row={'session_id': '050', 'native_frames': 2, 'seed': seed,
        'raw_inputs': {'raw_vst': {'sha256': 'a' * 64}}}, output_dir=root, signature='c' * 64,
        upstream={'Source': {'synthetic': True}}, request={'repair_rounds': 2,
            'parent_inputs': {'mask_seeds': {'path': '/frozen/explicit_original_seeds'}}})
    assert seen[0]['repair_rounds'] == 2 and seen[0]['return_round_masks'] is True
    assert receipt['round_1_masks'] == file_ref(root / 'ROUND_1_MASK_ARRAYS.npz')
    assert receipt['final_masks'] == file_ref(root / 'MASK_ARRAYS.npz')
    with np.load(root / 'ROUND_1_MASK_ARRAYS.npz') as arrays:
        assert np.array_equal(arrays['removal'], round_one['removal'])
    with np.load(root / 'MASK_ARRAYS.npz') as arrays:
        assert np.array_equal(arrays['removal'], final['removal'])
    report = json.loads((root / 'MASK_REPAIR.json').read_text())
    assert report['round_1_masks'] == receipt['round_1_masks']
    assert report['final_masks'] == receipt['final_masks']


@pytest.mark.parametrize('value', [0, 3601, True])
def test_new_recipe_rejects_invalid_finite_budget(project, specs, value):
    with pytest.raises(ValueError, match='WALL_BUDGET'):
        specs.temporal_execution_context(project.root, wall_seconds=value)


def _make_successor_base(fixture):
    base = {k: v.copy() for k, v in fixture.base.items()}
    for role in ('core', 'sam_core', 'removal'):
        base[role][:, 2:3, 2:3] = True
    base['effective_removal'], _ = effective_clean_masks(base['removal'], base['protected_visible'])
    np.savez_compressed(fixture.root / 'ROUND1.npz', **base)
    _replace_base(fixture, file_ref(fixture.root / 'ROUND1.npz'))
    return base


def _replace_base(fixture, reference):
    packet = fixture.packet
    receipt = json.loads((fixture.root / 'MASK_RECEIPT.json').read_text())
    old = packet['inputs']['frozen_base_masks']
    receipt['outputs'] = [reference if ref == old else ref for ref in receipt['outputs']]
    packet['inputs']['frozen_base_masks'] = reference
    packet['inputs']['frozen_mask_receipt'] = fixture.save_json('MASK_RECEIPT.json', receipt)


def test_successor_base_requires_explicit_opt_in(causal_fixture):
    fixture = causal_fixture
    _make_successor_base(fixture)
    with pytest.raises(ValueError, match='BASE_NOT_EXACT_ORIGINAL'):
        v1_mco_worker._require_mask_causal_inputs(fixture.packet, fixture.original)
    fixture.packet['config']['base_mask_kind'] = 'SEALED_V1_MASK_SUCCESSOR'
    base, repaired, refs = v1_mco_worker._require_mask_causal_inputs(fixture.packet, fixture.original)
    assert np.array_equal(base['removal'][:, 2:3, 2:3], True * np.ones_like(base['removal'][:, 2:3, 2:3]))
    assert np.array_equal(repaired['removal'], fixture.repaired['removal'])
    assert refs['frozen_mask_receipt'] == fixture.packet['inputs']['frozen_mask_receipt']
    assert fixture.loads == []


@pytest.mark.parametrize('role', ['equipment', 'protected_visible', 'unknown', 'cup_observed', 'semantic_conflict', 'removal'])
def test_successor_base_preserves_original_support_before_model_load(causal_fixture, role):
    fixture = causal_fixture
    base = _make_successor_base(fixture)
    fixture.packet['config']['base_mask_kind'] = 'SEALED_V1_MASK_SUCCESSOR'
    if role == 'removal':
        base[role][:] = False
    else:
        base[role][0, 0, 0] ^= True
    np.savez_compressed(fixture.root / 'ROUND1.npz', **base)
    _replace_base(fixture, file_ref(fixture.root / 'ROUND1.npz'))
    with pytest.raises(ValueError, match='MCO_SUCCESSOR_'):
        v1_mco_worker.execute_v1_mco_stage(fixture.packet)
    assert fixture.loads == []


def test_successor_comparison_still_runs_three_original_propainter_calls(causal_fixture):
    fixture = causal_fixture
    _make_successor_base(fixture)
    fixture.packet['config']['base_mask_kind'] = 'SEALED_V1_MASK_SUCCESSOR'
    assert v1_mco_worker.execute_v1_mco_stage(fixture.packet) == 0
    assert len(fixture.loads) == 1 and fixture.parent.model.pixel_calls == 3


def test_successor_repair_cannot_remove_round_one_support(causal_fixture):
    fixture = causal_fixture
    _make_successor_base(fixture)
    fixture.packet['config']['base_mask_kind'] = 'SEALED_V1_MASK_SUCCESSOR'
    fixture.repaired['removal'][:, 2:3, 2:3] = False
    np.savez_compressed(fixture.root / 'REPAIRED.npz', **fixture.repaired)
    packet = fixture.packet
    old = packet['inputs']['frozen_repaired_masks']
    new = file_ref(fixture.root / 'REPAIRED.npz')
    receipt = json.loads((fixture.root / 'MASK_RECEIPT.json').read_text())
    receipt['outputs'] = [new if ref == old else ref for ref in receipt['outputs']]
    packet['inputs']['frozen_repaired_masks'] = new
    packet['inputs']['frozen_mask_receipt'] = fixture.save_json('MASK_RECEIPT.json', receipt)
    with pytest.raises(ValueError, match='SUPPORT_REMOVED'):
        v1_mco_worker.execute_v1_mco_stage(packet)
    assert fixture.loads == []


def test_successor_base_still_rejects_partial_sealed_receipt(causal_fixture):
    fixture = causal_fixture
    _make_successor_base(fixture)
    fixture.packet['config']['base_mask_kind'] = 'SEALED_V1_MASK_SUCCESSOR'
    receipt = json.loads((fixture.root / 'MASK_RECEIPT.json').read_text())
    receipt['status'] = 'PARTIAL'
    fixture.packet['inputs']['frozen_mask_receipt'] = fixture.save_json('MASK_RECEIPT.json', receipt)
    with pytest.raises(ValueError, match='SEALED_MASK_RECEIPT'):
        v1_mco_worker.execute_v1_mco_stage(fixture.packet)
    assert fixture.loads == []
