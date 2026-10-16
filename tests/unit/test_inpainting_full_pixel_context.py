import numpy as np
import torch
from chaoyang.inpainting_candidate.v1_clean_context import run_context_clean_pair,FULL_PIXEL
from chaoyang.inpainting_candidate.v1_full_pixel_context import make_full_pixel_painter
from chaoyang.pick_place_1008.visual import ProtectedProPainter
from tests.test_inpainting_v1_clean_pair import synthetic_parent,fixture_arrays


def test_only_isolated_frozen_assign_changes():
    original=ProtectedProPainter.inpaint.__code__;global_ref=ProtectedProPainter.inpaint.__globals__['_get_ref_index']
    parent=synthetic_parent();before=dict(parent.settings);painter,log=make_full_pixel_painter(parent)
    assert log['edited_ast_assignments']==1 and log['flow_chunking_unchanged']
    assert log['neural_reference_indices_unchanged'] and parent.settings==before
    assert painter.inpaint.__func__.__code__ is not original and ProtectedProPainter.inpaint.__code__ is original
    assert ProtectedProPainter.inpaint.__globals__['_get_ref_index'] is global_ref


def test_real_body_full_native_pixel_context_preserves_nn_refs_masks_and_repeat(monkeypatch):
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    result=run_context_clean_pair(synthetic_parent(),*fixture_arrays(128),experiment=FULL_PIXEL,
        authority=lambda:None,on_call=lambda v:None)
    assert result['receipt']['A_repeat_reproducibility']['exact_equal']
    obs=result['stage_observations']
    a=[r for r in obs['A']['metadata']['records'] if r['stage']=='PIXEL_PROPAGATION']
    b=[r for r in obs['B']['metadata']['records'] if r['stage']=='PIXEL_PROPAGATION']
    assert [(r['native_start'],r['native_end_exclusive']) for r in a]==[(0,90),(70,128)]
    assert [(r['native_start'],r['native_end_exclusive']) for r in b]==[(0,128)]
    for v in ('A','B'):
        assert all(s['mask']['mismatch_vs_frozen_effective_pixels']==0
                   for r in obs[v]['metadata']['records'] for s in r['samples'])
    calls=result['receipt']['calls']
    assert [r['actual_references'] for r in calls[0]['reference_context']['calls']]==[
        r['actual_references'] for r in calls[1]['reference_context']['calls']]
    assert not result['receipt']['same_mask_weights_settings_precision_backend_and_rng']
    assert result['receipt']['same_mask_weights_precision_backend_and_rng']
