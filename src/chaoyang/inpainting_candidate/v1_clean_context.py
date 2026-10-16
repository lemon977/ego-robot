"""Finite offline Clean context experiments; no model load or execution grant.

The frozen ProtectedProPainter body is reused verbatim in an isolated globals
namespace. Only its reference-index provider differs; vendor modules, settings,
neighbors, model objects and the original function globals remain untouched.
"""
from __future__ import annotations

from copy import deepcopy
from types import FunctionType, MethodType
from typing import Any, Callable

import numpy as np

LOCAL = 'FROZEN_LOCAL_REFERENCES'
GLOBAL_EQUAL = 'FULL_SESSION_EQUAL_COUNT'
REVERSE = 'REVERSE_NATIVE_TIME_ORDER'
TRACE = 'TRACE_UNCHANGED_ORIGINAL_PATH'
FULL_PIXEL = 'FULL_NATIVE_IMAGE_PROPAGATION_CONTEXT'
EXPERIMENTS = (GLOBAL_EQUAL, REVERSE)  # The observer is diagnostic, not a new recipe.


def reference_indices(mid, neighbors, length, stride, ref_num, *, policy=GLOBAL_EQUAL):
    from chaoyang.exact78_baseline.clean import _get_ref_index
    if (type(length) is not int or length < 2 or type(mid) is not int
            or not 0 <= mid < length or type(stride) is not int or stride <= 0
            or type(ref_num) is not int or ref_num < -1
            or not isinstance(neighbors, list) or len(neighbors) != len(set(neighbors))
            or mid not in neighbors or any(type(i) is not int or not 0 <= i < length for i in neighbors)):
        raise ValueError('CLEAN_CONTEXT_REFERENCE_NATIVE_DOMAIN')
    original = _get_ref_index(mid, neighbors, length, stride, ref_num)
    if policy == LOCAL:
        return list(original)
    if policy != GLOBAL_EQUAL:
        raise ValueError('CLEAN_CONTEXT_POLICY_NOT_REGISTERED')
    if not original:
        return []
    excluded = set(neighbors)
    candidates = [i for i in range(length) if i not in excluded]
    count = len(original)
    if len(candidates) < count:
        raise ValueError('CLEAN_CONTEXT_CANNOT_PRESERVE_REFERENCE_COUNT')
    # Equal-population bin centers, integer-only; no ranking, score or threshold.
    chosen = [candidates[((2*j+1)*len(candidates))//(2*count)] for j in range(count)]
    if len(set(chosen)) != count or any(i in excluded for i in chosen):
        raise ValueError('CLEAN_CONTEXT_REFERENCE_COUNT_OR_NEIGHBOR_DRIFT')
    return chosen


def make_context_painter(parent: Any, policy: str = GLOBAL_EQUAL):
    from chaoyang.pick_place_1008.visual import ProtectedProPainter
    from chaoyang.exact78_baseline.clean import _get_ref_index
    if policy not in (LOCAL, GLOBAL_EQUAL):
        raise ValueError('CLEAN_CONTEXT_POLICY_NOT_REGISTERED')
    painter = ProtectedProPainter(parent)
    log = {'policy': policy, 'single_change': 'REFERENCE_TEMPORAL_CANDIDATE_RANGE_ONLY',
           'original_body_reused': True, 'original_globals_modified': False,
           'same_reference_count': True, 'neighbors_changed': False,
           'weights_settings_propagation_unchanged': True, 'time_mode': 'OFFLINE_NONCAUSAL',
           'reference_registration': 'NOT_ASSERTED', 'calls': []}

    def select(mid, neighbors, length, stride, ref_num):
        old = _get_ref_index(mid, neighbors, length, stride, ref_num)
        selected = reference_indices(mid, neighbors, length, stride, ref_num, policy=policy)
        log['calls'].append({'mid': mid, 'neighbors': list(neighbors), 'length': length,
            'stride': stride, 'ref_num': ref_num, 'original_references': list(old),
            'actual_references': list(selected), 'same_count': len(selected) == len(old),
            'changed': list(selected) != list(old)})
        return selected

    original = ProtectedProPainter.inpaint
    namespace = dict(original.__globals__)
    namespace['_get_ref_index'] = select
    isolated = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    isolated.__kwdefaults__ = original.__kwdefaults__
    painter.inpaint = MethodType(isolated, painter)
    return painter, log


def run_context_clean_pair(parent, raw, removal, protected, *, experiment: str,
                           authority: Callable, on_call: Callable, backend_before_load=None):
    from chaoyang.pick_place_1008.visual import effective_clean_masks
    from .v1_clean_pair import (array_signature, runtime_backend, require_same_backend,
                               model_identity, RandomState)
    from .v1_mask_clean_causal import _reproducibility
    if experiment not in (*EXPERIMENTS, TRACE, FULL_PIXEL) or not callable(authority) or not callable(on_call):
        raise ValueError('CLEAN_CONTEXT_FINITE_EXPERIMENT_AND_AUTHORITY_REQUIRED')
    if (not isinstance(raw, np.ndarray) or raw.dtype != np.uint8 or raw.ndim != 4
            or raw.shape[-1] != 3 or len(raw) < 2
            or any(not isinstance(m, np.ndarray) or m.dtype != bool or m.shape != raw.shape[:3]
                   for m in (removal, protected))):
        raise ValueError('CLEAN_CONTEXT_NATIVE_RGB_MASK_DOMAIN')
    effective, _ = effective_clean_masks(removal, protected)
    fixed = {'raw': raw, 'removal': removal, 'protected': protected, 'effective': effective}
    signatures = {k: array_signature(v) for k,v in fixed.items()}
    backend = runtime_backend(parent.torch, parent.device)
    if backend_before_load is not None:
        require_same_backend(backend_before_load, backend)
    identity = model_identity(parent)
    rng = RandomState(parent.torch, parent.device)
    initial_rng = rng.metadata()
    outputs, validations, observations = {}, {}, []
    observer=None;stage_observers={}

    def check():
        require_same_backend(backend, runtime_backend(parent.torch, parent.device))
        if model_identity(parent) != identity:
            raise ValueError('CLEAN_CONTEXT_PARENT_MODEL_MUTATED')
        if {k:array_signature(v) for k,v in fixed.items()} != signatures:
            raise ValueError('CLEAN_CONTEXT_FIXED_INPUT_MUTATED')

    for variant in ('A','B','A_repeat'):
        reverse = variant == 'B' and experiment == REVERSE
        policy = GLOBAL_EQUAL if variant == 'B' and experiment == GLOBAL_EQUAL else LOCAL
        actual_parent=parent
        if experiment==TRACE and variant=='B':
            from .v1_clean_observer import observed_parent
            actual_parent,observer=observed_parent(parent,effective)
        if experiment==FULL_PIXEL and variant in ('A','B'):
            from .v1_clean_observer import observed_parent
            actual_parent,stage_observers[variant]=observed_parent(parent,effective,
                pixel_context_length=len(raw) if variant=='B' else None)
        if experiment==FULL_PIXEL and variant=='B':
            from .v1_full_pixel_context import make_full_pixel_painter
            painter,log=make_full_pixel_painter(actual_parent)
        else:
            painter, log = make_context_painter(actual_parent, policy)
        authority(); check(); on_call(variant); authority(); check(); rng.restore()
        if RandomState(parent.torch, parent.device).metadata() != initial_rng:
            raise ValueError('CLEAN_CONTEXT_RANDOM_STATE_RESET_FAILED')
        inputs = [v[::-1].copy() if reverse else v for v in (raw, removal, protected)]
        output = painter.inpaint(*inputs)
        authority(); check()
        validation = deepcopy(painter.last_validation)
        actual_effective = painter.effective_masks
        if reverse:
            output = output[::-1].copy()
            actual_effective = actual_effective[::-1].copy()
            validation = {k:(v[::-1] if isinstance(v,list) and len(v)==len(raw) else v)
                          for k,v in validation.items()}
        if output.shape != raw.shape or output.dtype != np.uint8 or not np.array_equal(actual_effective,effective):
            raise ValueError('CLEAN_CONTEXT_NATIVE_OUTPUT_OR_MASK_DRIFT')
        changed = np.any(output != raw,axis=-1)
        p = np.count_nonzero(changed & protected,axis=(1,2)).astype(int).tolist()
        e = np.count_nonzero(changed & ~effective,axis=(1,2)).astype(int).tolist()
        if (validation.get('status') != 'PASS' or any(p) or any(e)
                or validation.get('protected_changed_pixels_per_frame') != p
                or validation.get('outside_delete_changed_pixels_per_frame') != e):
            raise ValueError('CLEAN_CONTEXT_ORIGINAL_PIXEL_PROTECTION_FAILED')
        validation.update(independent_protected_changed_pixels_per_frame=p,
                          independent_outside_effective_changed_pixels_per_frame=e)
        outputs[variant], validations[variant] = output,validation
        observations.append({'variant':variant,'reference_context':log,'traversal':'REVERSED_THEN_RESTORED' if reverse else 'NATIVE',
            'internal_to_native_frame_indices':list(range(len(raw)-1,-1,-1)) if reverse else list(range(len(raw))),
            'output':array_signature(output),'backend':backend,'model_identity_sha256':identity['sha256']})
    difference = np.any(outputs['A'] != outputs['B'],axis=-1)
    if experiment==TRACE and (difference.any() or not np.array_equal(outputs['A'],outputs['A_repeat'])):
        raise ValueError('PP_TRACE_OBSERVER_CHANGED_OUTPUT_OR_UNSTABLE_BASELINE')
    if experiment==FULL_PIXEL and not np.array_equal(outputs['A'],outputs['A_repeat']):
        raise ValueError('FULL_PIXEL_CONTEXT_BASELINE_OBSERVER_OR_REPRODUCIBILITY_FAIL')
    receipt = {'schema':'inpainting-v1-clean-context-pair-v1','status':'PAIRED_COMPLETE',
        'experiment':experiment,'model_calls':3,'single_loaded_model':True,'single_change':experiment,
        'same_mask_weights_settings_precision_backend_and_rng':True,'actual_inputs':signatures,
        'actual_backend':backend,'model_identity':identity,'random_state_initial':initial_rng,
        'calls':observations,'A_repeat_reproducibility':_reproducibility(outputs['A'],outputs['A_repeat']),
        'A_B_difference_pixels_per_frame':np.count_nonzero(difference,axis=(1,2)).astype(int).tolist(),
        'native_frames':len(raw),'model_loads_by_pair':0,'human_review':'PENDING',
        'hidden_regions':'UNKNOWN','quality_pass':False,'candidate_adopted':False,
        'training_eligible':False,'control_ground_truth':False}
    trace={} if observer is None else {'observation':observer.metadata(),'observation_arrays':observer.arrays}
    if stage_observers:
        trace['stage_observations']={v:{'metadata':o.metadata(),'arrays':o.arrays} for v,o in stage_observers.items()}
        receipt.update(same_mask_weights_settings_precision_backend_and_rng=False,
            same_mask_weights_precision_backend_and_rng=True,settings_dictionary_unchanged=True,
            actual_changed_model_control='IMAGE_PIXEL_PROPAGATION_CONTEXT_LENGTH_ONLY',
            original_pixel_context_control=min(100,parent.settings['subvideo_length']),candidate_pixel_context_control=len(raw))
    return {**outputs,'validations':validations,'receipt':receipt,'rng_arrays':rng.arrays(),**trace}
