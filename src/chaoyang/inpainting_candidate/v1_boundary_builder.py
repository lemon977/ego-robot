"""Opt-in fixed-E boundary adapter; the frozen context/V1 builders stay unchanged.

Working pixels remain an unadopted visual hypothesis, not hidden scene truth.
Only an explicitly signed, sealed same-upstream context Clean may avoid PP.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import FunctionType
from zipfile import BadZipFile

import numpy as np

from .automation import read_reference, require, verify_component, verify_reference, write_once_json
from .automation_worker import V1Builders, _arrays, _outputs, _save_arrays
from .contracts import file_ref
from .v1_context_builder import ContextExclusionBuilders, POLICY

RAW_FIXED_BOUNDARY = 'POISSON_WORKING_GRADIENT_RAW_FIXED_BOUNDARY'
RGB_UNCERTAIN_BOUNDARY_VETO = 'POISSON_RGB_UNCERTAIN_BOUNDARY_VETO'
BOUNDARY_POLICIES = frozenset({RAW_FIXED_BOUNDARY, RGB_UNCERTAIN_BOUNDARY_VETO})
PRIMARY_VARIANT = 'CONTEXT_EXCLUSION_BOUNDARY'


def _same_reference(left, right):
    """Preserve byte/path identity while allowing proven aliases of one file."""
    a, b = verify_reference(left), verify_reference(right)
    return (left['bytes'] == right['bytes'] and left['sha256'] == right['sha256']
            and a.samefile(b))


class BoundaryContextBuilders(ContextExclusionBuilders):
    def _boundary_policy(self):
        config = self.packet.get('component_config', {}).get('Clean', {})
        require(config.get('reference_policy') == POLICY, 'BOUNDARY_CONTEXT_REFERENCE_POLICY_REQUIRED')
        policy = config.get('write_boundary_policy')
        require(isinstance(policy, str) and policy in BOUNDARY_POLICIES, 'BOUNDARY_EXPLICIT_WRITE_POLICY_REQUIRED')
        return policy

    def _work_inputs(self, reference, signature, row, upstream, raw, masks):
        receipt = verify_component(reference, signature, row, stage='Clean', repo_root=self.repo)
        require(receipt.get('reference_policy') == POLICY
                and receipt.get('primary_clean_variant') == 'CONTEXT_EXCLUSION',
                'BOUNDARY_BASE_CONTEXT_CLEAN_REQUIRED')
        previous = receipt.get('upstream', {})
        require(set(previous) == {'Source', 'Mask'} and all(
            _same_reference(previous[stage], upstream[stage]) for stage in ('Source', 'Mask')),
            'BOUNDARY_WORK_UPSTREAM_BYTE_REFERENCE_MISMATCH')
        outputs = _outputs(reference)
        names = {'WORKING_CLEAN_RGB.npz', 'CONTEXT_WORKING_MASKS.npz',
                 'CONTEXT_REFERENCE_AUDIT.json', 'CLEAN_REPORT.json'}
        require(names <= outputs.keys(), 'BOUNDARY_CONTEXT_WORK_OUTPUTS_REQUIRED')
        # Do not load cached RGB/masks until sealed identity and ancestry pass.
        work = _arrays(outputs['WORKING_CLEAN_RGB.npz'])['rgb']
        working = _arrays(outputs['CONTEXT_WORKING_MASKS.npz'])
        required = {'removal', 'effective_removal', 'write_effective_removal', 'rgb_uncertain_support'}
        require(work.shape == raw.shape and work.dtype == np.uint8
                and required <= working.keys()
                and all(value.dtype == np.bool_ and value.shape == raw.shape[:3]
                        for value in working.values())
                and np.array_equal(working['write_effective_removal'], masks['effective_removal'])
                and not np.any(masks['removal'] & ~working['removal'])
                and not np.any(masks['effective_removal'] & ~working['effective_removal'])
                and not np.any(working['effective_removal'] & masks['protected_visible']),
                'BOUNDARY_CONTEXT_WORK_PIXEL_DOMAIN_DRIFT')
        require(not np.any(working['rgb_uncertain_support'] & masks['protected_visible']),
                'BOUNDARY_WORK_UNCERTAINTY_PROTECTION_CONFLICT')
        # W is the effective (already PP-dilated) working write domain, not
        # the smaller removal input. These are producer invariants, not a
        # visual-quality threshold or enlarged deletion permission.
        changed = np.any(work != raw, axis=-1)
        require(not np.any(changed & (~working['effective_removal'] | masks['protected_visible'])),
                'BOUNDARY_WORK_OUTSIDE_EFFECTIVE_MASK_DRIFT')
        audit = read_reference(outputs['CONTEXT_REFERENCE_AUDIT.json'])
        require(_same_reference(audit['source_receipt'], upstream['Source'])
                and _same_reference(audit['mask_receipt'], upstream['Mask'])
                and _same_reference(audit['seed'], row['seed'])
                and audit.get('reference_policy') == POLICY, 'BOUNDARY_WORK_AUDIT_UPSTREAM_DRIFT')
        report = read_reference(outputs['CLEAN_REPORT.json'])
        require(report.get('primary_clean_variant') == 'CONTEXT_EXCLUSION'
                and report.get('reference_policy') == POLICY, 'BOUNDARY_WORK_REPORT_POLICY_DRIFT')
        return work, working, audit, report, {name: outputs[name] for name in sorted(names)}

    def clean(self, *, row, output_dir, signature, upstream, request):
        policy = self._boundary_policy()
        base = self.packet.get('base_clean_signatures', {}).get(row['session_id'])
        require(isinstance(base, str) and len(base) == 64
                and all(c in '0123456789abcdef' for c in base), 'BOUNDARY_BASE_CLEAN_SIGNATURE_REQUIRED')
        root = self._start(output_dir)
        raw = self._rgb(upstream)
        source_outputs, mask_outputs = _outputs(upstream['Source']), _outputs(upstream['Mask'])
        masks = _arrays(mask_outputs['MASK_ARRAYS.npz'])
        effective, protected = masks['effective_removal'], masks['protected_visible']
        require(raw.dtype == np.uint8 and raw.ndim == 4 and raw.shape[-1] == 3
                and len(raw) == row['native_frames']
                and all(v.dtype == np.bool_ and v.shape == raw.shape[:3]
                        for v in (effective, protected, masks['removal']))
                and not np.any(effective & protected), 'BOUNDARY_CURRENT_FIXED_PIXEL_DOMAIN_REQUIRED')
        mapping = self._mapping(upstream)
        require(isinstance(mapping, list) and len(mapping) == len(raw), 'BOUNDARY_SOURCE_FRAME_MAP_REQUIRED')
        before = self.clean_calls
        cache = self.packet.get('clean_work_reuse', {}).get(row['session_id'])
        cache_audit = {'requested': cache is not None, 'consumed': False, 'reason': 'NOT_REQUESTED'}
        selected = None
        if cache is not None:
            try:
                require(isinstance(cache, Mapping) and set(cache) == {'receipt', 'expected_signature'}
                        and cache['expected_signature'] == base, 'BOUNDARY_CACHE_BASE_SIGNATURE_MISMATCH')
                selected = self._work_inputs(cache['receipt'], base, row, upstream, raw, masks)
                work_ref = cache['receipt']
                cache_audit = {'requested': True, 'consumed': True, 'reason': 'VERIFIED_SEALED_SAME_UPSTREAM',
                               'receipt': work_ref, 'expected_signature': base}
            except (ValueError, OSError, KeyError, TypeError, EOFError, BadZipFile) as error:
                # Rejected cache bytes never become the working donor.
                cache_audit = {'requested': True, 'consumed': False, 'reason': str(error), 'candidate': cache}
        if selected is None:
            ContextExclusionBuilders.clean(self, row=row, output_dir=root / 'Work', signature=base,
                                           upstream=upstream, request=request)
            work_ref = file_ref(root / 'Work' / 'RECEIPT.json')
            selected = self._work_inputs(work_ref, base, row, upstream, raw, masks)
        work, working, audit, work_report, work_outputs = selected
        from .v1_context_seam import reconstruct_fixed_boundary
        clean, statistics = raw.copy(), []
        for frame in range(len(raw)):
            self.authority()
            kwargs = ({'raw_boundary_allowed': ~working['rgb_uncertain_support'][frame]}
                      if policy == RGB_UNCERTAIN_BOUNDARY_VETO else {})
            pixels, diagnosis = reconstruct_fixed_boundary(
                raw[frame], work[frame], effective[frame], protected[frame], **kwargs)
            require(isinstance(pixels, np.ndarray) and pixels.dtype == np.uint8
                    and pixels.shape == raw[frame].shape, 'BOUNDARY_HELPER_NATIVE_RGB_REQUIRED')
            clean[frame] = pixels
            statistics.append({'native_frame_index': frame, **diagnosis})
        changed = np.any(clean != raw, axis=-1)
        protected_changes = np.count_nonzero(changed & protected, axis=(1, 2)).tolist()
        outside_changes = np.count_nonzero(changed & ~effective, axis=(1, 2)).tolist()
        require(clean.dtype == np.uint8 and clean.shape == raw.shape
                and not any(protected_changes) and not any(outside_changes), 'BOUNDARY_FINAL_FIXED_WRITE_INVARIANT')
        calls = self.clean_calls - before
        require(calls == (0 if cache_audit['consumed'] else 1), 'BOUNDARY_BASE_PROPAINTER_CALL_COUNT_DRIFT')
        inputs = {'source_receipt': upstream['Source'], 'mask_receipt': upstream['Mask'],
                  'source_raw': source_outputs['RAW_RGB.npz'], 'source_frame_map': source_outputs['FRAME_MAP.json'],
                  'mask_arrays': mask_outputs['MASK_ARRAYS.npz'], 'base_clean_receipt': work_ref,
                  'base_clean_signature': base, 'work_outputs': work_outputs}
        validation = {'status': 'PASS', 'protected_changed_pixels_per_frame': protected_changes,
                      'outside_delete_changed_pixels_per_frame': outside_changes,
                      'write_domain': 'UNCHANGED_SEALED_MASK_STAGE_EFFECTIVE_REMOVAL', 'frames': statistics}
        _save_arrays(root / 'CLEAN_RGB.npz', rgb=clean, effective_removal=effective)
        _save_arrays(root / 'EFFECTIVE_MASKS.npz', effective_removal=effective)
        _save_arrays(root / 'WORKING_CLEAN_RGB.npz', rgb=work)
        _save_arrays(root / 'CONTEXT_WORKING_MASKS.npz', **working)
        write_once_json(root / 'CONTEXT_REFERENCE_AUDIT.json', audit)
        write_once_json(root / 'WORKING_CLEAN_REPORT.json', work_report)
        write_once_json(root / 'FRAME_MAP.json', mapping)
        write_once_json(root / 'CLEAN_VALIDATION.json', validation)
        report = {'validation': validation, 'propainter_calls': calls, 'paired_clean': False,
                  'primary_clean_variant': PRIMARY_VARIANT, 'reference_policy': POLICY,
                  'write_boundary_policy': policy, 'raw_boundary_allowed': 'NOT_RGB_UNCERTAIN_SUPPORT'
                  if policy == RGB_UNCERTAIN_BOUNDARY_VETO else 'ALL_OUTSIDE_E',
                  'actual_inputs': inputs, 'work_cache': cache_audit, 'working_source_reused': cache_audit['consumed'],
                  'historical_clean_consumed': False, 'final_clean_rgb_consumed': False, 'raw_inside_E_used': False,
                  'hidden': 'UNKNOWN', 'human_review': 'PENDING', 'candidate_adopted': False,
                  'training_eligible': False, 'control_ground_truth': False}
        write_once_json(root / 'CLEAN_REPORT.json', report)
        from .source_frames import write_review_video
        write_review_video(root / 'clean.mp4', clean, mapping)
        return self._seal('Clean', row, root, signature, upstream,
                          {'propainter_calls': calls, 'sam_api_calls': 0, 'reference_policy': POLICY,
                           'write_boundary_policy': policy, 'primary_clean_variant': PRIMARY_VARIANT,
                           'actual_inputs': inputs, 'work_cache': cache_audit,
                           'final_write_mask': 'UNCHANGED_SEALED_MASK_STAGE', 'raw_inside_E_used': False,
                           'hidden': 'UNKNOWN', 'new_manual_points': 0, 'candidate_adopted': False})

    def composite(self, **kwargs):
        policy = self._boundary_policy()
        original = V1Builders.composite
        namespace = dict(original.__globals__)
        writer = namespace['write_once_json']

        def scoped_writer(path, value):
            if Path(path).name == 'SESSION.json':
                value = {**value, 'primary_clean_variant': PRIMARY_VARIANT,
                         'clean_reference_policy': POLICY, 'clean_write_boundary_policy': policy,
                         'write_boundary_policy': policy}
            return writer(path, value)

        namespace['write_once_json'] = scoped_writer
        clone = FunctionType(original.__code__, namespace, original.__name__,
                             original.__defaults__, original.__closure__)
        return clone(self, **kwargs)


__all__ = ['BoundaryContextBuilders', 'RAW_FIXED_BOUNDARY', 'RGB_UNCERTAIN_BOUNDARY_VETO']
