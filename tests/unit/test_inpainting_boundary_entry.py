"""Synthetic CPU adapter tests; no production input, PP model, or renderer runs."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.inpainting_candidate import v1_boundary_builder as module
from chaoyang.inpainting_candidate import v1_context_builder, v1_context_seam
from chaoyang.inpainting_candidate.automation import read_reference, verify_component
from chaoyang.inpainting_candidate.automation_worker import V1Builders, _arrays, _outputs, _save_arrays
from chaoyang.inpainting_candidate.contracts import SCHEMA, file_ref
from tests.test_inpainting_v1_automation_flow import save


@pytest.fixture
def scene(tmp_path, monkeypatch):
    from PIL import Image
    from chaoyang.inpainting_candidate import source_frames, v1_equipment_rgb
    from chaoyang.pick_place_1008 import visual

    y, x = np.mgrid[:16, :24]
    frame = np.stack((70 + x, 80 + y, 60 + x + y), axis=-1).astype(np.uint8)
    raw = np.stack((frame, frame + np.uint8(3)))
    masks = {name: np.zeros(raw.shape[:3], bool) for name in
             ('core', 'sam_core', 'equipment', 'removal', 'effective_removal',
              'protected_visible', 'cup_observed', 'unknown')}
    masks['effective_removal'][:, 4:12, 5:18] = True
    masks['protected_visible'][:, 7:9, 10:12] = True
    masks['effective_removal'][masks['protected_visible']] = False
    masks['removal'][:] = masks['effective_removal']
    masks['core'][:] = masks['effective_removal']
    masks['unknown'][:, 6:10, 9:13] = True
    working_effective = np.zeros_like(masks['effective_removal'])
    working_effective[:, 2:14, 2:22] = True
    working_effective[masks['protected_visible']] = False
    uncertain = np.zeros_like(working_effective)
    uncertain[:, 2:5, 2:7] = True  # U is deliberately not the expanded working mask W.
    work = raw.copy()
    work[working_effective] = np.clip(work[working_effective].astype(int) + 25, 0, 255)
    work[:, 5:7, 6:9] += np.uint8(20)
    Image.fromarray(raw[0]).save(tmp_path / 'seed.png')
    seed = save(tmp_path / 'seed.json', {'session_id': 'S', 'raw_frame_index': 0,
                                        'image': file_ref(tmp_path / 'seed.png'), 'objects': []})
    mapping = [{'session_id': 'S', 'native_frame_index': i, 'output_frame': i} for i in range(len(raw))]
    source = tmp_path / 'Source'
    source.mkdir()
    _save_arrays(source / 'RAW_RGB.npz', rgb=raw)
    save(source / 'FRAME_MAP.json', mapping)
    source_ref = save(source / 'RECEIPT.json', {'schema': SCHEMA, 'stage': 'Source',
        'outputs': [file_ref(source / 'RAW_RGB.npz'), file_ref(source / 'FRAME_MAP.json')]})
    mask_root = tmp_path / 'Mask'
    mask_root.mkdir()
    _save_arrays(mask_root / 'MASK_ARRAYS.npz', **masks)
    report = save(mask_root / 'MASK_REPORT.json', {'seed': seed, 'metadata': {}})
    mask_ref = save(mask_root / 'RECEIPT.json', {'schema': SCHEMA, 'stage': 'Mask',
        'outputs': [file_ref(mask_root / 'MASK_ARRAYS.npz'), report]})
    upstream = {'Source': source_ref, 'Mask': mask_ref}
    packet = {'project_root': str(tmp_path), 'component_config': {'Clean': {
        'reference_policy': v1_context_builder.POLICY, 'write_boundary_policy': module.RAW_FIXED_BOUNDARY}},
        'base_clean_signatures': {'S': 'b' * 64}, 'budgets': {'propainter_calls_max': 1}}
    row = {'session_id': 'S', 'native_frames': len(raw), 'seed': seed}
    painter_calls = []

    class Painter:
        settings = {'synthetic': True}
        last_validation = {'status': 'PASS', 'synthetic': True}

        def __init__(self, parent):
            assert parent is None

        def inpaint(self, actual_raw, removal, protected):
            painter_calls.append('SYNTHETIC_PP')
            np.testing.assert_array_equal(actual_raw, raw)
            np.testing.assert_array_equal(removal, working_effective)
            np.testing.assert_array_equal(protected, masks['protected_visible'])
            return work.copy()

    def extend(actual_raw, current, seed, **kwargs):
        result = {name: value.copy() if isinstance(value, np.ndarray) else deepcopy(value)
                  for name, value in current.items()}
        result['removal'] = working_effective.copy()
        result['effective_removal'] = working_effective.copy()
        return result, uncertain.copy(), {'synthetic': True}

    def video(path, rgb, mapping):
        Path(path).write_bytes(b'SYNTHETIC_VIDEO_NOT_VISUAL_EVIDENCE')
        return {'video': file_ref(path)}

    monkeypatch.setattr(visual, 'ProtectedProPainter', Painter)
    monkeypatch.setattr(v1_equipment_rgb, 'extend', extend)
    monkeypatch.setattr(source_frames, 'write_review_video', video)

    def builder(kind=module.BoundaryContextBuilders, value=None):
        instance = kind(packet if value is None else value, lambda: None)
        monkeypatch.setattr(instance, '_models', lambda stage: {'propainter': None})
        return instance

    def cache(name='BaseClean'):
        instance = builder(v1_context_builder.ContextExclusionBuilders)
        instance.clean(row=row, output_dir=tmp_path / name, signature='b' * 64,
                       upstream=upstream, request={})
        reference = file_ref(tmp_path / name / 'RECEIPT.json')
        painter_calls.clear()
        packet['clean_work_reuse'] = {'S': {'receipt': reference, 'expected_signature': 'b' * 64}}
        return reference

    return SimpleNamespace(root=tmp_path, packet=packet, raw=raw, masks=masks, work=work,
        uncertain=uncertain, working_effective=working_effective, mapping=mapping,
        row=row, upstream=upstream, builder=builder, cache=cache, painter_calls=painter_calls)


def run_clean(scene, name='Clean'):
    builder = scene.builder()
    result = builder.clean(row=scene.row, output_dir=scene.root / name, signature='c' * 64,
                           upstream=scene.upstream, request={})
    return builder, result, file_ref(scene.root / name / 'RECEIPT.json')


def assert_fixed_legacy(scene, ref):
    verify_component(ref, 'c' * 64, scene.row, stage='Clean', repo_root=scene.root)
    outputs = _outputs(ref)
    clean = _arrays(outputs['CLEAN_RGB.npz'])
    assert set(clean) == {'rgb', 'effective_removal'}
    np.testing.assert_array_equal(clean['effective_removal'], scene.masks['effective_removal'])
    changed = np.any(clean['rgb'] != scene.raw, axis=-1)
    assert changed.any()  # Not a no-op producer disguised as an adapter test.
    assert not np.any(changed & scene.masks['protected_visible'])
    assert not np.any(changed & ~scene.masks['effective_removal'])
    np.testing.assert_array_equal(_arrays(outputs['WORKING_CLEAN_RGB.npz'])['rgb'], scene.work)
    context = _arrays(outputs['CONTEXT_WORKING_MASKS.npz'])
    np.testing.assert_array_equal(context['rgb_uncertain_support'], scene.uncertain)
    report = read_reference(outputs['CLEAN_REPORT.json'])
    assert report['primary_clean_variant'] == module.PRIMARY_VARIANT
    assert report['hidden'] == 'UNKNOWN' and report['candidate_adopted'] is False
    assert report['human_review'] == 'PENDING' and report['training_eligible'] is False
    assert report['control_ground_truth'] is False and report['raw_inside_E_used'] is False
    assert report['actual_inputs'] == read_reference(ref)['actual_inputs']
    assert report['validation']['outside_delete_changed_pixels_per_frame'] == [0, 0]
    assert report['validation']['protected_changed_pixels_per_frame'] == [0, 0]
    return report


def test_cached_work_is_verified_legacy_compatible_and_zero_pp(scene, monkeypatch):
    base = scene.cache()
    scene.packet['budgets']['propainter_calls_max'] = 0
    old_sha = file_ref(base['path'])
    builder = scene.builder()
    monkeypatch.setattr(builder, '_models', lambda stage: pytest.fail('CACHED_WORK_MUST_NOT_LOAD_MODELS'))
    builder.clean(row=scene.row, output_dir=scene.root / 'Clean', signature='c' * 64,
                  upstream=scene.upstream, request={})
    ref = file_ref(scene.root / 'Clean/RECEIPT.json')
    report = assert_fixed_legacy(scene, ref)
    assert builder.clean_calls == 0 and not scene.painter_calls
    assert report['propainter_calls'] == 0 and report['work_cache']['consumed'] is True
    assert report['actual_inputs']['base_clean_receipt'] == base
    assert file_ref(base['path']) == old_sha and not (scene.root / 'Clean/Work').exists()


def test_uncached_work_uses_parent_once_in_sealed_subdirectory(scene):
    builder, receipt, ref = run_clean(scene)
    report = assert_fixed_legacy(scene, ref)
    assert builder.clean_calls == 1 and scene.painter_calls == ['SYNTHETIC_PP']
    assert report['propainter_calls'] == receipt['propainter_calls'] == 1
    assert report['work_cache'] == {'requested': False, 'consumed': False, 'reason': 'NOT_REQUESTED'}
    work_ref = report['actual_inputs']['base_clean_receipt']
    assert Path(work_ref['path']) == scene.root / 'Clean/Work/RECEIPT.json'
    parent = verify_component(work_ref, 'b' * 64, scene.row, stage='Clean', repo_root=scene.root)
    assert parent['primary_clean_variant'] == 'CONTEXT_EXCLUSION'


@pytest.mark.parametrize('failure', ['upstream', 'partial', 'signature', 'sha', 'session', 'native',
                                    'write_mask', 'uncertain_protected', 'outside_work', 'protected_work', 'seed'])
def test_invalid_cached_work_is_rejected_and_never_used_as_donor(scene, monkeypatch, failure):
    base = scene.cache()
    receipt = read_reference(base)
    cached_work = _outputs(base)['WORKING_CLEAN_RGB.npz']['path']
    if failure == 'upstream':
        wrong = save(scene.root / 'OtherMask.json', {'synthetic_wrong_upstream': True})
        receipt['upstream']['Mask'] = wrong
    elif failure == 'partial':
        receipt['status'] = 'PARTIAL'
    elif failure == 'signature':
        scene.packet['clean_work_reuse']['S']['expected_signature'] = 'd' * 64
    elif failure == 'sha':
        Path(cached_work).write_bytes(b'NOT_THE_SIGNED_ARCHIVE')
    elif failure == 'session':
        receipt['session_id'] = 'OTHER'
    elif failure == 'native':
        receipt['native_frames'] += 1
    elif failure in ('write_mask', 'uncertain_protected', 'outside_work', 'protected_work'):
        outputs = _outputs(base)
        ref = outputs['WORKING_CLEAN_RGB.npz' if failure in ('outside_work', 'protected_work')
                      else 'CONTEXT_WORKING_MASKS.npz']
        arrays = _arrays(ref)
        if failure == 'write_mask':
            arrays['write_effective_removal'][:] = False
        elif failure == 'uncertain_protected':
            arrays['rgb_uncertain_support'][scene.masks['protected_visible']] = True
        elif failure == 'outside_work':
            arrays['rgb'][:, 0, 0] ^= np.uint8(1)
        else:
            arrays['rgb'][scene.masks['protected_visible']] ^= np.uint8(1)
        with Path(ref['path']).open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        receipt['outputs'] = [file_ref(r['path']) for r in receipt['outputs']]
    elif failure == 'seed':
        ref = _outputs(base)['CONTEXT_REFERENCE_AUDIT.json']
        audit = read_reference(ref)
        audit['seed'] = save(scene.root / 'OtherSeed.json', {'synthetic_wrong_seed': True})
        save(Path(ref['path']), audit)
        receipt['outputs'] = [file_ref(r['path']) for r in receipt['outputs']]
    if failure not in ('signature', 'sha'):
        scene.packet['clean_work_reuse']['S']['receipt'] = save(Path(base['path']), receipt)
    original = module._arrays
    if failure not in ('write_mask', 'uncertain_protected', 'outside_work', 'protected_work', 'seed'):
        def no_bad_rgb(ref):
            assert ref['path'] != cached_work, 'INVALID_CACHE_RGB_WAS_CONSUMED'
            return original(ref)
        monkeypatch.setattr(module, '_arrays', no_bad_rgb)
    builder, receipt, ref = run_clean(scene)
    report = assert_fixed_legacy(scene, ref)
    assert builder.clean_calls == 1 and scene.painter_calls == ['SYNTHETIC_PP']
    assert report['work_cache']['requested'] is True and report['work_cache']['consumed'] is False
    assert report['work_cache']['reason']
    assert '/Clean/Work/' in report['actual_inputs']['work_outputs']['WORKING_CLEAN_RGB.npz']['path']


@pytest.mark.parametrize('cached', [False, True])
def test_real_veto_helper_integrates_with_cached_and_uncached_work(scene, cached):
    if cached:
        scene.cache()
        scene.packet['budgets']['propainter_calls_max'] = 0
    scene.packet['component_config']['Clean']['write_boundary_policy'] = module.RGB_UNCERTAIN_BOUNDARY_VETO
    builder, receipt, ref = run_clean(scene)
    report = assert_fixed_legacy(scene, ref)
    expected, classic = [], []
    for frame in range(scene.row['native_frames']):
        arguments = (scene.raw[frame], scene.work[frame], scene.masks['effective_removal'][frame],
                     scene.masks['protected_visible'][frame])
        pixels, diagnostic = v1_context_seam.reconstruct_fixed_boundary(
            *arguments, raw_boundary_allowed=~scene.uncertain[frame])
        assert diagnostic['raw_boundary_veto_pairs'] > 0
        expected.append(pixels)
        classic.append(v1_context_seam.reconstruct_fixed_boundary(*arguments)[0])
    actual = _arrays(_outputs(ref)['CLEAN_RGB.npz'])['rgb']
    np.testing.assert_array_equal(actual, np.stack(expected))
    assert np.any(actual != np.stack(classic))
    assert all(row['raw_boundary_veto_pairs'] > 0 for row in report['validation']['frames'])
    assert builder.clean_calls == receipt['propainter_calls'] == (0 if cached else 1)
    assert report['work_cache']['consumed'] is cached


@pytest.mark.parametrize('policy', [module.RAW_FIXED_BOUNDARY, module.RGB_UNCERTAIN_BOUNDARY_VETO])
def test_boundary_veto_uses_exact_not_u_not_not_working_mask(scene, monkeypatch, policy):
    scene.cache()
    scene.packet['component_config']['Clean']['write_boundary_policy'] = policy
    original = v1_context_seam.reconstruct_fixed_boundary
    seen = []

    def solve(raw, work, effective, protected, **kwargs):
        seen.append(kwargs)
        # Helper numerics are tested separately; this test proves adapter routing.
        return original(raw, work, effective, protected)

    monkeypatch.setattr(v1_context_seam, 'reconstruct_fixed_boundary', solve)
    _, receipt, ref = run_clean(scene)
    report = assert_fixed_legacy(scene, ref)
    assert len(seen) == scene.row['native_frames']
    for frame, kwargs in enumerate(seen):
        if policy == module.RAW_FIXED_BOUNDARY:
            assert kwargs == {}
        else:
            assert set(kwargs) == {'raw_boundary_allowed'}
            np.testing.assert_array_equal(kwargs['raw_boundary_allowed'], ~scene.uncertain[frame])
            assert not np.array_equal(kwargs['raw_boundary_allowed'], ~scene.working_effective[frame])
    assert report['write_boundary_policy'] == receipt['write_boundary_policy'] == policy


@pytest.mark.parametrize('field,value', [('write_boundary_policy', 'IMPLICIT_OR_UNKNOWN'), ('reference_policy', 'OTHER')])
def test_policy_is_explicit_before_any_work_or_output(scene, field, value):
    scene.packet['component_config']['Clean'][field] = value
    with pytest.raises(ValueError, match='POLICY_REQUIRED'):
        run_clean(scene)
    assert not scene.painter_calls and not (scene.root / 'Clean').exists()


def test_base_signature_is_required_before_work(scene):
    scene.packet.pop('base_clean_signatures')
    with pytest.raises(ValueError, match='BASE_CLEAN_SIGNATURE_REQUIRED'):
        run_clean(scene)
    assert not scene.painter_calls and not (scene.root / 'Clean').exists()


def test_rejected_cache_cannot_bypass_zero_pp_budget(scene):
    scene.cache()
    scene.packet['clean_work_reuse']['S']['expected_signature'] = 'd' * 64
    scene.packet['budgets']['propainter_calls_max'] = 0
    with pytest.raises(ValueError, match='PROPAINTER_CALL_BUDGET_EXHAUSTED'):
        run_clean(scene)
    assert not scene.painter_calls
    assert not (scene.root / 'Clean/RECEIPT.json').exists()
    assert not (scene.root / 'Clean/Work/RECEIPT.json').exists()


@pytest.mark.parametrize('failure', ['dtype', 'shape'])
def test_helper_cannot_silently_cast_or_broadcast_wrong_rgb(scene, monkeypatch, failure):
    scene.cache()

    def broken(raw, work, effective, protected):
        return (raw.astype(float) if failure == 'dtype' else raw[0]), {'status': 'CLAIMED_PASS'}

    monkeypatch.setattr(v1_context_seam, 'reconstruct_fixed_boundary', broken)
    with pytest.raises(ValueError, match='HELPER_NATIVE_RGB_REQUIRED'):
        run_clean(scene)
    assert not (scene.root / 'Clean/RECEIPT.json').exists()
    assert not (scene.root / 'Clean/CLEAN_RGB.npz').exists()


@pytest.mark.parametrize('domain', ['outside_E', 'protected'])
def test_adapter_independently_rejects_helper_pixel_domain_drift(scene, monkeypatch, domain):
    scene.cache()

    def broken(raw, work, effective, protected):
        output = raw.copy()
        y, x = np.argwhere(protected if domain == 'protected' else ~effective)[0]
        output[y, x] ^= np.uint8(1)
        return output, {'status': 'CLAIMED_PASS'}

    monkeypatch.setattr(v1_context_seam, 'reconstruct_fixed_boundary', broken)
    with pytest.raises(ValueError, match='FIXED_WRITE_INVARIANT'):
        run_clean(scene)
    assert not (scene.root / 'Clean/CLEAN_RGB.npz').exists()
    assert not (scene.root / 'Clean/RECEIPT.json').exists()


def test_original_composite_consumes_legacy_arrays_same_pixels_only_metadata_changes(scene):
    scene.cache()
    builder, _, clean_ref = run_clean(scene)
    motion = scene.root / 'Motion'
    motion.mkdir()
    _save_arrays(motion / 'actions.npz', qpc_velocity_violation=np.zeros((2, 2), bool),
                 qpc_acceleration_violation=np.ones((2, 2), bool))
    save(motion / 'motion.json', {'physical_contact_ground_truth': False})
    motion_ref = save(motion / 'RECEIPT.json', {'outputs': [file_ref(motion / name)
                                                             for name in ('actions.npz', 'motion.json')]})
    render = scene.root / 'Render'
    render.mkdir()
    rgb = np.full_like(scene.raw, 155)
    alpha = np.zeros(scene.raw.shape[:3], bool)
    alpha[:, 3:13, 3:20] = True
    _save_arrays(render / 'ROBOT_BUFFERS.npz', ego_rgb=rgb, ego_alpha=alpha, external_rgb=rgb, external_alpha=alpha)
    save(render / 'FRAME_MAP.json', scene.mapping)
    render_ref = save(render / 'RECEIPT.json', {'outputs': [file_ref(render / name)
                                                              for name in ('ROBOT_BUFFERS.npz', 'FRAME_MAP.json')]})
    upstream = {**scene.upstream, 'Clean': clean_ref, 'Render': render_ref, 'Motion': motion_ref}
    original_writer = V1Builders.composite.__globals__['write_once_json']
    builder.composite(row=scene.row, output_dir=scene.root / 'BoundaryComposite', signature='e' * 64,
                      upstream=upstream, request={})
    original = V1Builders(scene.packet, lambda: None)
    original.composite(row=scene.row, output_dir=scene.root / 'OriginalComposite', signature='e' * 64,
                       upstream=upstream, request={})
    assert V1Builders.composite.__globals__['write_once_json'] is original_writer
    for name in ('FINAL_RGB.npz',):
        with np.load(scene.root / 'BoundaryComposite' / name) as a, np.load(scene.root / 'OriginalComposite' / name) as b:
            np.testing.assert_array_equal(a['rgb'], b['rgb'])
    boundary = json.loads((scene.root / 'BoundaryComposite/SESSION.json').read_text())
    legacy = json.loads((scene.root / 'OriginalComposite/SESSION.json').read_text())
    assert boundary['primary_clean_variant'] == module.PRIMARY_VARIANT
    assert legacy['primary_clean_variant'] == 'A'
    assert boundary['write_boundary_policy'] == module.RAW_FIXED_BOUNDARY
    assert boundary['clean_write_boundary_policy'] == module.RAW_FIXED_BOUNDARY
    assert boundary['stage_receipts'] == legacy['stage_receipts']
    assert boundary['human_review'] == 'PENDING' and boundary['training_eligible'] is False
    assert (scene.root / 'BoundaryComposite/FRAME_STATUS.json').read_bytes() == (scene.root / 'OriginalComposite/FRAME_STATUS.json').read_bytes()
    assert builder.render_calls == 0 and builder.clean_calls == 0

