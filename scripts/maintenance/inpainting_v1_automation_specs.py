"""Bounded new specs for the approved V1-compatible experiment, not a runner."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

from chaoyang.inpainting_candidate.contracts import canonical_sha, file_ref, verify_ref
from chaoyang.humanego_ict.project_mount_paths import resolve_input_path


def local_ref(reference, root):
    """Bind the identical bytes to this task's physical execution mount."""
    verify_ref(reference)
    path = resolve_input_path(reference['path'], root)
    return {**reference, 'path': str(path)}


def _clean_work_for_selected_upstreams(root, candidates, verified_plan, selected, signatures):
    """Choose a verified base Clean for the final plan, not a different full DAG.

    plan_resume has already verified every declared parent/catalog receipt,
    including failed parents' explicit sealed prefixes. Enumerating those
    catalogs preserves candidates discarded only by its maximum-DAG choice.
    """
    from chaoyang.inpainting_candidate.automation import read_reference
    parents = candidates['parents'] if isinstance(candidates, dict) else candidates
    result = {}
    for proof in verified_plan['parents']:
        if proof['model_result'] is None:
            catalog = parents[proof['candidate_index']].get('stage_receipts', {})
        else:
            model = read_reference(proof['model_result'])
            catalog = {row.get('session_id', row.get('session_key')): row['stage_receipts']
                       for row in model['sessions']}
        for sid, stages in catalog.items():
            upstream = selected.get(sid, {})
            if sid in result or 'Clean' not in stages or not {'Source', 'Mask'} <= upstream.keys():
                continue
            receipt = read_reference(stages['Clean'])
            if receipt['component_signature'] != signatures.get(sid):
                continue
            if all(local_ref(receipt['upstream'][stage], root) == local_ref(upstream[stage], root)
                   for stage in ('Source', 'Mask')):
                result[sid] = {'receipt': stages['Clean'], 'expected_signature': signatures[sid]}
    return result


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def deadline(root):
    state = load_json(root / 'inpainting_optimization/eight_hour_v1/AUTOMATION_STATE.json')
    limit = datetime.fromisoformat(state['global_deadline_at'])
    if limit <= datetime.now(timezone.utc):
        raise ValueError('AUTOMATION_APPROVED_TIMEBOX_EXPIRED')
    return limit.isoformat()


def _repair_recipe_path(root, repair_rounds):
    if type(repair_rounds) is not int or repair_rounds not in (1, 2, 3):
        raise ValueError('AUTOMATION_FINITE_RECIPE_ROUNDS_REQUIRED')
    if repair_rounds == 1:
        return Path(root) / 'inpainting_optimization/incremental_v1_8h/DEFAULT_SINGLE_ROUND_EXECUTION_RECIPE.json'
    folder = 'temporal_repair_3h' if repair_rounds == 2 else 'residual_1h'
    return Path(root) / 'inpainting_optimization' / folder / 'EXECUTION_RECIPE.json'


def temporal_execution_context(root, *, wall_seconds=3600, repair_rounds=2):
    """Freeze a new opt-in Task recipe and its own bounded submission deadline."""
    root = Path(root).resolve(strict=True)
    recipe_ref = file_ref(_repair_recipe_path(root, repair_rounds))
    recipe = load_json(recipe_ref['path'])
    if (recipe.get('schema') != 'inpainting-v1-temporal-repair-execution-recipe-v1'
            or recipe.get('repair_rounds') != repair_rounds
            or type(wall_seconds) is not int or not 0 < wall_seconds <= 3600
            or type(recipe.get('wall_seconds_max')) is not int
            or wall_seconds > recipe['wall_seconds_max']):
        raise ValueError('AUTOMATION_TEMPORAL_EXECUTION_RECIPE_OR_WALL_BUDGET')
    context = {key: recipe[key] for key in ('plan_revision', 'objective', 'output_root', 'review_root')}
    return {**context, 'schema': 'inpainting-v1-finite-execution-context-v1',
        'global_deadline_at': (datetime.now(timezone.utc) + timedelta(seconds=wall_seconds)).isoformat(),
        'wall_seconds_max': wall_seconds, 'execution_recipe': recipe_ref}


def clean_reference_execution_context(root, *, wall_seconds=3600):
    root=Path(root).resolve(strict=True)
    ref=file_ref(root/'inpainting_optimization/clean_context_8h/EXECUTION_RECIPE.json');recipe=load_json(ref['path'])
    if (recipe.get('schema')!='inpainting-v1-clean-reference-execution-recipe-v1'
        or recipe.get('clean_reference_policy')!='EXCLUDE_UNCERTAIN_EQUIPMENT_RGB'
        or type(wall_seconds)is not int or not 0<wall_seconds<=recipe['wall_seconds_max']):
        raise ValueError('AUTOMATION_CLEAN_REFERENCE_RECIPE_REQUIRED')
    return {**{k:recipe[k]for k in ['plan_revision','objective','output_root','review_root']},
        'schema':'inpainting-v1-finite-execution-context-v1','wall_seconds_max':wall_seconds,
        'global_deadline_at':(datetime.now(timezone.utc)+timedelta(seconds=wall_seconds)).isoformat(),'execution_recipe':ref}


def _execution_context(root, value):
    from chaoyang.inpainting_candidate.automation import project_local
    if (value.get('schema') != 'inpainting-v1-finite-execution-context-v1'
            or type(value.get('wall_seconds_max')) is not int
            or not 0 < value['wall_seconds_max'] <= 3600):
        raise ValueError('AUTOMATION_FINITE_EXECUTION_CONTEXT_REQUIRED')
    verify_ref(value['execution_recipe'])
    recipe = load_json(value['execution_recipe']['path'])
    if recipe.get('schema')=='inpainting-v1-clean-reference-execution-recipe-v1':
        if (not Path(value['execution_recipe']['path']).resolve(strict=True).samefile(
                root/'inpainting_optimization/clean_context_8h/EXECUTION_RECIPE.json')
            or recipe.get('clean_reference_policy')!='EXCLUDE_UNCERTAIN_EQUIPMENT_RGB'
            or any(value.get(k)!=recipe.get(k)for k in ['plan_revision','objective'])
            or any((Path(value[k])if Path(value[k]).is_absolute()else root/value[k]).resolve()!=
                   (Path(recipe[k])if Path(recipe[k]).is_absolute()else root/recipe[k]).resolve()for k in ['output_root','review_root'])
            or value['wall_seconds_max']>recipe['wall_seconds_max']):
            raise ValueError('AUTOMATION_CLEAN_REFERENCE_RECIPE_DRIFT')
        limit=datetime.fromisoformat(value['global_deadline_at'])
        if limit.tzinfo is None or limit<=datetime.now(timezone.utc):raise ValueError('AUTOMATION_APPROVED_TIMEBOX_EXPIRED')
        result=dict(value)
        for key,area in [('output_root','_run'),('review_root','review')]:
            path=Path(value[key]);resolved=(path if path.is_absolute()else root/path).resolve()
            if not project_local(resolved,root)or not resolved.is_relative_to((root/area).resolve())or resolved==(root/area).resolve():
                raise ValueError('AUTOMATION_FINITE_PROJECT_OUTPUT_REQUIRED')
            result[key]=str(resolved)
        return result
    if not Path(value['execution_recipe']['path']).resolve(strict=True).samefile(
            _repair_recipe_path(root, recipe.get('repair_rounds'))):
        raise ValueError('AUTOMATION_FINITE_EXECUTION_RECIPE_REQUIRED')
    if (recipe.get('schema') != 'inpainting-v1-temporal-repair-execution-recipe-v1'
            or type(recipe.get('repair_rounds')) is not int or recipe.get('repair_rounds') not in (1, 2, 3)
            or any(value.get(k) != recipe.get(k) for k in ('plan_revision', 'objective'))
            or any((Path(value[k]) if Path(value[k]).is_absolute() else root / value[k]).resolve()
                   != (Path(recipe[k]) if Path(recipe[k]).is_absolute() else root / recipe[k]).resolve()
                   for k in ('output_root', 'review_root'))
            or value['wall_seconds_max'] > recipe.get('wall_seconds_max', 0)):
        raise ValueError('AUTOMATION_FINITE_EXECUTION_RECIPE_DRIFT')
    limit = datetime.fromisoformat(value['global_deadline_at'])
    if limit.tzinfo is None or limit <= datetime.now(timezone.utc):
        raise ValueError('AUTOMATION_APPROVED_TIMEBOX_EXPIRED')
    result = dict(value)
    for key, area in (('output_root', '_run'), ('review_root', 'review')):
        path = Path(value[key])
        path = path if path.is_absolute() else root / path
        resolved = path.resolve()
        if not project_local(resolved, root) or not resolved.is_relative_to((root / area).resolve()) or resolved == (root / area).resolve():
            raise ValueError('AUTOMATION_FINITE_PROJECT_OUTPUT_REQUIRED')
        result[key] = str(path)
    return result


def base_spec(root, task_id, mode, sessions, model_weights, inputs, config, *, execution_context=None):
    from chaoyang.governance.inpainting_candidate_registration import mco_route_contract
    route = mco_route_contract(root)['inpainting_original_v1_mco_candidate_v1']
    context = _execution_context(root, execution_context) if execution_context is not None else None
    spec = {
        'spec_schema': 'inpainting-v1-mco-stage-spec-v1', 'task_id': task_id,
        'plan_revision': 'INPAINTING_V1_AUTOMATION_EIGHT_HOUR_20261014',
        'objective': 'Approved eight-hour V1 experiment; independent finite package, no adoption',
        'mode': mode, 'global_deadline_at': context['global_deadline_at'] if context else deadline(root),
        'output_dir': str(root / '_run/inpainting_v1_automation_20261014' / task_id),
        'review_dir': str(root / 'review/inpainting_optimization/EIGHT_HOUR_AUTOMATION' / task_id),
        'sessions': sessions, 'inputs': inputs,
        'runtime_code': [*route['code_closure'], file_ref(root / ('scripts/maintenance/inpainting_v1_automation_specs.py'
            if context else '_run/inpainting_v1_automation_20261014/bootstrap/launch.py'))],
        'model_weights': model_weights,
        'calibration': {'status': 'FROZEN_V1_VISUAL_PRIOR_NOT_METRIC_QUALIFIED'},
        'schema': {'component': 'inpainting-candidate-component-v1', 'result': 'inpainting-v1-mco-stage-result-v1'},
        'config': config,
        'infrastructure': file_ref(root / 'scripts/humanego_ict/production_resume_runner_v1.py'),
        'read_set': ['AGENTS.md', 'inpainting_v1/MANIFEST.json',
                     'inpainting_optimization/eight_hour_v1/EIGHT_HOUR_AUTOMATION_PLAN_ZH.md',
                     'This exact spec, its frozen input references and runtime closure only'],
        'budgets': {'wall_seconds_max': 3600, 'cpu_threads_per_worker': 2, 'gpu_count': 1,
                    'model_calls_max': 2, 'render_calls_max': 0, 'training_steps_max': 0,
                    'output_bytes_estimate': 536870912},
    }
    if context is not None:
        spec.update(plan_revision=context['plan_revision'], objective=context['objective'],
            global_deadline_at=context['global_deadline_at'],
            output_dir=str(Path(context['output_root']) / task_id),
            review_dir=str(Path(context['review_root']) / task_id),
            inputs={**inputs, 'execution_recipe': context['execution_recipe']})
        spec['budgets']['wall_seconds_max'] = context['wall_seconds_max']
        spec['read_set'] = ['AGENTS.md', 'inpainting_v1/MANIFEST.json',
            context['execution_recipe']['path'], 'This exact spec and its frozen input/runtime references only']
    return spec


def clean_pair(root, *, session_id='050', task_id=None):
    """Fresh A/B fixed-mask input, historical result never used as A."""
    parent = load_json(root / 'inpainting_v1/MANIFEST.json')
    frozen = load_json(root / 'inpainting_optimization/eight_hour_v1/FROZEN_INPUTS.json')
    rows = {r['session_key']: r for r in frozen['sessions']}
    row = dict(rows[session_id])
    result = root / '_run/inpainting_v1_mco_8h_20261013/spread_mask_dev_a01/MODEL_RESULT.json'
    masks = result.parent / 'sessions' / session_id / 'Mask/MASK_ARRAYS.npz'
    row.update(mask_input=file_ref(masks), mask_result=file_ref(result),
               review_anchors=([0,28,64,77,91,96,119,120,121,122,123,124,125,126,127]
                               if session_id == '050' else [0,65,103,104,114,119,132,204,237]))
    source = root / '_run/inpainting_generalization_g0_publication_20261013/attempt_0001/sessions' / session_id / 'Source/RAW_RGB.npz'
    frame_map = (root / '_run/inpainting_v1_mco_8h_20261013/spread_render050_a01/sessions/050/Boundary/FRAME_MAP.json'
                 if session_id == '050' else root / '_run/inpainting_v1_mco_8h_20261013/mask_render_dev_a01/sessions/001/Boundary/FRAME_MAP.json')
    if not frame_map.is_file():
        # Source mapping is authoritative, not a synthesized FPS index.
        frame_map = source.parent / 'FRAME_MAP.json'
    weights = [local_ref(ref, root) for ref in parent['models']['weights'] if 'propainter' in ref['path'].lower()]
    spec = base_spec(root, task_id or 'inpainting_v1_mco_paired_clean' + session_id + '_20261014_a01',
        'Clean', [row], weights,
        {'parent_v1': file_ref(root / 'inpainting_v1/MANIFEST.json'),
         'storage_binding': frozen['storage_binding'], 'frozen_source_raw': file_ref(source),
         'frozen_frame_map': file_ref(frame_map), 'mask_arrays': row['mask_input'], 'mask_result': row['mask_result']},
        {'parent': 'INPAINTING_RAW_TO_ROBOT_V1', 'paired_clean': True,
         'pixel_propagation': 'BYPASS', 'human_points_added': 0, 'session_specific_branches': False})
    return spec


def component_closures(root):
    """Stage-local algorithm dependencies, separate from the operation fence."""
    common = ['src/chaoyang/inpainting_candidate/automation.py',
              'src/chaoyang/inpainting_candidate/automation_worker.py',
              'src/chaoyang/inpainting_candidate/contracts.py']
    source = ['src/chaoyang/pick_place_1008/source.py', 'src/chaoyang/pick_place_1008/camera.py',
              'src/chaoyang/pick_place_1008/installation_fixed_k.py', 'src/chaoyang/pick_place_1008/audit.py',
              'src/chaoyang/inpainting_candidate/automation_paths.py']
    motion = ['src/chaoyang/pick_place_1008/pipeline.py', 'src/chaoyang/pick_place_1008/retarget.py']
    # Only list files present in the frozen V1 closure, not guessed variants.
    motion = [p for p in motion if (root / p).is_file()]
    shared_robot = [str(p.relative_to(root)) for p in (root / 'src/chaoyang/exact78_baseline').glob('*.py')
                    if p.name in {'robot_model.py', 'retarget.py', 'pipeline.py', 'render.py'}]
    mask = ['src/chaoyang/pick_place_1008/visual.py', 'src/chaoyang/pick_place_1008/mask_seeds.py',
            'src/chaoyang/pick_place_1008/sam_point_adapter.py', 'src/chaoyang/exact78_baseline/mask.py',
            'src/chaoyang/inpainting_candidate/v1_mask_repair.py',
            'src/chaoyang/inpainting_candidate/automatic_masks.py',
            'src/chaoyang/inpainting_candidate/model_worker.py',
            'src/chaoyang/inpainting_candidate/model_stages.py']
    if (root / 'src/chaoyang/inpainting_candidate/automation_mask.py').is_file():
        mask.append('src/chaoyang/inpainting_candidate/automation_mask.py')
    mask.append('src/chaoyang/inpainting_candidate/automation_paths.py')
    mask.extend(['src/chaoyang/inpainting_candidate/automatic_v1_seeds.py',
                 'src/chaoyang/pipeline/sam31_compat_adapter_v1.py'])
    clean = ['src/chaoyang/exact78_baseline/clean.py', 'src/chaoyang/pick_place_1008/visual.py',
             'src/chaoyang/inpainting_candidate/model_stages.py',
             'src/chaoyang/inpainting_candidate/v1_clean_ablation.py']
    render = shared_robot
    composite = ['src/chaoyang/pick_place_1008/visual.py', 'src/chaoyang/exact78_baseline/render.py',
                 'src/chaoyang/inpainting_candidate/source_frames.py']
    groups = {'Source': source, 'Motion': motion + shared_robot, 'Mask': mask,
              'Clean': clean, 'Render': render, 'Composite': composite}
    vendor = {'Mask': ('vendor/SAM3/sam3',), 'Clean': ('vendor/ProPainter/model', 'vendor/ProPainter/RAFT', 'vendor/ProPainter/utils')}
    for stage, directories in vendor.items():
        for directory in directories:
            groups[stage] += [str(p.relative_to(root)) for p in (root / directory).rglob('*.py') if '__pycache__' not in p.parts]
    return {stage: [file_ref(root / name) for name in sorted(set(common + names))]
            for stage, names in groups.items()}


def compatible_spec(root, request, request_ref, *, reuse_stages=None, execution_context=None, clean_reference_policy=None,
                    resume_manifest=None,clean_write_policy=None):
    from chaoyang.inpainting_candidate.automation import verify_v1_request, component_signatures
    verify_v1_request(request, project_root=root)
    if execution_context is not None:
        context = _execution_context(root, execution_context)
        recipe=load_json(context['execution_recipe']['path'])
        if clean_reference_policy is not None:
            if (recipe.get('clean_reference_policy')!=clean_reference_policy or clean_reference_policy!='EXCLUDE_UNCERTAIN_EQUIPMENT_RGB'
                or request.get('repair_rounds',1)not in recipe.get('allowed_repair_rounds',[])):
                raise ValueError('AUTOMATION_CLEAN_REFERENCE_REQUEST_RECIPE_DRIFT')
        elif recipe.get('repair_rounds') != request.get('repair_rounds', 1):
            raise ValueError('AUTOMATION_REQUEST_EXECUTION_RECIPE_ROUNDS_DRIFT')
    parent = load_json(root / 'inpainting_v1/MANIFEST.json')
    weights = [local_ref(ref, root) for ref in parent['models']['weights']]
    components = component_closures(root)
    weight_groups = {'Mask': [r for r in weights if 'sam3' in r['path'].lower()],
                     'Clean': [r for r in weights if 'propainter' in r['path'].lower()]}
    frozen_assets = load_json(root / 'inpainting_optimization/eight_hour_v1/FROZEN_INPUTS.json')['robot_assets']
    config = {'Mask': {'repair_prompt_policy': 'SPREAD_MAX_GAP_THREE', 'original_v1_seeds_allowed': True,
                       'glove_text_prompt': 'a gloved hand', 'repair_rounds': request.get('repair_rounds', 1)},
              'Clean': {'pixel_propagation': 'FROZEN', 'settings': parent['models']['clean_SETTINGS']},
              'Motion': {'robot_assets': frozen_assets}, 'Render': {'robot_assets': frozen_assets},
              'Composite': {'display_policy': 'ORIGINAL_V1_OBJECT_FIRST_FALLBACK', 'hidden_geometry': 'UNKNOWN'}}
    if clean_reference_policy is not None:
        if execution_context is None or clean_reference_policy!='EXCLUDE_UNCERTAIN_EQUIPMENT_RGB':
            raise ValueError('AUTOMATION_CLEAN_REFERENCE_EXPLICIT_CONTEXT_REQUIRED')
        config['Clean'].update(reference_policy=clean_reference_policy,execution_recipe=context['execution_recipe'])
        extra=['src/chaoyang/inpainting_candidate/v1_context_builder.py','src/chaoyang/inpainting_candidate/v1_context_exclusion.py',
               'src/chaoyang/inpainting_candidate/v1_equipment_rgb.py','src/chaoyang/exact78_baseline/mask.py',
               'src/chaoyang/inpainting_candidate/source_frames.py']
        components['Clean'] += [file_ref(root/name)for name in extra if str(root/name)not in {r['path']for r in components['Clean']}]
        components['Composite'] += [file_ref(root/'src/chaoyang/inpainting_candidate/v1_context_builder.py')]
    base_clean_signatures={};base_components={};base_config={}
    if clean_write_policy is not None:
        from copy import deepcopy
        if clean_reference_policy!='EXCLUDE_UNCERTAIN_EQUIPMENT_RGB'or clean_write_policy not in (
                'POISSON_WORKING_GRADIENT_RAW_FIXED_BOUNDARY','POISSON_RGB_UNCERTAIN_BOUNDARY_VETO'):
            raise ValueError('AUTOMATION_BOUNDARY_EXPLICIT_CONTEXT_POLICY_REQUIRED')
        base_components=deepcopy(components);base_config=deepcopy(config)
        base_clean_signatures={sid:s['Clean']for sid,s in component_signatures(request,components,weight_groups,config).items()}
        config['Clean']['write_boundary_policy']=clean_write_policy
        for name in ['src/chaoyang/inpainting_candidate/v1_boundary_builder.py','src/chaoyang/inpainting_candidate/v1_context_seam.py']:
            components['Clean'].append(file_ref(root/name))
        components['Composite'].append(file_ref(root/'src/chaoyang/inpainting_candidate/v1_boundary_builder.py'))
    signatures = component_signatures(request, components, weight_groups, config)
    resume_plan=None;clean_work_reuse={}
    if resume_manifest is not None:
        if reuse_stages is not None:raise ValueError('AUTOMATION_RESUME_AND_EXPLICIT_REUSE_CONFLICT')
        from chaoyang.inpainting_candidate.v1_resume_plan import plan_resume
        resume_manifest=local_ref(resume_manifest,root)
        resume_candidates=load_json(resume_manifest['path'])
        resume_plan=plan_resume(project_root=root,request=request,component_code=components,
            component_weights=weight_groups,component_config=config,candidates=resume_candidates)
        reuse_stages=resume_plan['reuse_stages']
        if clean_write_policy is not None:
            work_plan=plan_resume(project_root=root,request=request,component_code=base_components,
                component_weights=weight_groups,component_config=base_config,candidates=resume_candidates)
            clean_work_reuse=_clean_work_for_selected_upstreams(root,resume_candidates,work_plan,
                reuse_stages,base_clean_signatures)
    if clean_reference_policy is not None and reuse_stages is None:
        from chaoyang.inpainting_candidate.automation import verify_component,read_reference
        cache_ref=recipe['stage_cache'];verify_ref(cache_ref);cache=load_json(cache_ref['path']);reuse_stages={}
        if cache.get('schema')!='inpainting-context-explicit-stage-cache-v1':raise ValueError('AUTOMATION_CONTEXT_CACHE_SCHEMA')
        for row in request['sessions']:
            for stored in cache['sessions']:
                if stored['session_id']!=row['session_id']:continue
                terminal=read_reference(stored['parent_result'])
                parent_model=read_reference(stored['parent_model'])
                if terminal.get('status')!='PASSED'or terminal.get('model_result')!=stored['parent_model']:
                    raise ValueError('AUTOMATION_CONTEXT_CACHE_PARENT_NOT_TERMINAL')
                for stage,ref in stored['stages'].items():
                    if stage not in ['Source','Motion','Mask','Render']:raise ValueError('AUTOMATION_CONTEXT_CACHE_STAGE')
                    receipt=read_reference(ref)
                    if receipt['component_signature']==signatures[row['session_id']][stage]:
                        verify_component(ref,receipt['component_signature'],row,stage=stage,repo_root=root)
                        reuse_stages.setdefault(row['session_id'],{})[stage]=ref
    from chaoyang.governance.inpainting_candidate_registration import mco_route_contract
    orchestration_runtime=[*mco_route_contract(root)['inpainting_original_v1_mco_candidate_v1']['code_closure'],
                           file_ref(root/'scripts/maintenance/inpainting_v1_automation_specs.py')]
    # New orchestration bytes require a new Task, not a second algorithm cache.
    # Component signatures deliberately exclude this navigation/publisher layer.
    identity_input = {'request': request, 'components': signatures, 'reuse_stages': reuse_stages or {},
                      'orchestration_runtime':orchestration_runtime}
    if resume_manifest is not None:identity_input['resume_manifest']=resume_manifest
    if clean_write_policy is not None:identity_input.update(base_clean_signatures=base_clean_signatures,clean_work_reuse=clean_work_reuse)
    if execution_context is not None:
        identity_input['execution_context'] = {k: v for k, v in _execution_context(root, execution_context).items()
                                               if k != 'global_deadline_at'}
    identity = canonical_sha(identity_input)
    task_id = 'inpainting_v1_mco_compat_' + identity[:24]
    sessions = [{'session_key': r['session_id'], 'native_frames': r['native_frames']} for r in request['sessions']]
    spec = base_spec(root, task_id, 'Automation', sessions, weights,
        {'parent_v1': file_ref(root / 'inpainting_v1/MANIFEST.json'), 'request': request_ref},
        {'mode': 'v1-compatible', 'three_point_repair': True, 'human_points_added': 0,
         'session_specific_branches': False}, execution_context=execution_context)
    n = len(sessions)
    spec['budgets'].update(sam_api_calls_max=320*n, propainter_calls_max=n,
        model_calls_max=321*n, render_calls_max=2*sum(r['native_frames'] for r in sessions))
    spec.update(component_code=components, component_weights=weight_groups, component_config=config,
                component_signatures=signatures, reuse_stages=reuse_stages or {})
    if resume_manifest is not None:
        spec['inputs']['resume_manifest']=resume_manifest
        spec['resume_plan']=resume_plan
    if clean_write_policy is not None:
        spec.update(base_clean_signatures=base_clean_signatures,clean_work_reuse=clean_work_reuse)
    return spec


def execute_spec(path, *, root=None):
    """Use the existing MCO publisher and its process-local mount adaptation."""
    import importlib
    import importlib.util
    from types import SimpleNamespace
    from chaoyang.humanego_ict.governance_mount_compat import install_governance_mount_compat, bind_publisher_module

    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    install_governance_mount_compat(root)
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(root.resolve(strict=True)):
        raise ValueError('PROJECT_ONLY_SPEC_INPUT')
    source = root / 'scripts/maintenance/inpainting_v1_mco_task.py'
    definition = importlib.util.spec_from_file_location('_existing_inpainting_stable_publisher', source)
    publisher = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(publisher)
    base_util = importlib.util

    class BoundInfraLoader:
        def __init__(self, original):
            self.original = original
        def __getattr__(self, name):
            return getattr(self.original, name)
        def create_module(self, spec):
            return self.original.create_module(spec)
        def exec_module(self, module):
            self.original.exec_module(module)
            bind_publisher_module(module)

    def scoped_spec(name, location, *args, **kwargs):
        result = base_util.spec_from_file_location(name, location, *args, **kwargs)
        if Path(location) == root / 'scripts/humanego_ict/production_resume_runner_v1.py':
            result.loader = BoundInfraLoader(result.loader)
        return result

    publisher.importlib = SimpleNamespace(util=SimpleNamespace(
        spec_from_file_location=scoped_spec, module_from_spec=base_util.module_from_spec),
        invalidate_caches=importlib.invalidate_caches, reload=importlib.reload, import_module=importlib.import_module)
    return publisher.main(['--spec', str(path)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--clean-pair', choices=['050', '001'])
    action.add_argument('--execute-spec', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.execute_spec is not None:
        if args.output is not None:
            parser.error('--output is supported only with --clean-pair')
        return execute_spec(args.execute_spec)
    if args.output is None:
        parser.error('--clean-pair requires --output')
    from chaoyang.governance.common import REPO_ROOT
    from chaoyang.inpainting_candidate.automation import write_once_json
    if not args.output.absolute().is_relative_to(REPO_ROOT):
        raise ValueError('PROJECT_ONLY_SPEC_OUTPUT')
    spec = clean_pair(REPO_ROOT, session_id=args.clean_pair)
    print(json.dumps(write_once_json(args.output, spec)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
