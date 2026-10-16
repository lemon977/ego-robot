"""CPU synthetic evidence only; never weights, CUDA or quality approval."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from chaoyang.inpainting_candidate import v1_clean_context as context
from chaoyang.pick_place_1008.visual import ProtectedProPainter
from chaoyang.exact78_baseline.clean import _get_ref_index
from tests.test_inpainting_v1_clean_pair import synthetic_parent,fixture_arrays

@pytest.mark.parametrize('length,mid',[(128,0),(128,64),(128,125),(238,0),(238,230),(20,10),(4,0)])
def test_full_context_preserves_count_and_excludes_neighbors(length,mid):
 neighbors=list(range(max(0,mid-5),min(length,mid+6)))
 count=8 if length>80 else -1
 old=_get_ref_index(mid,neighbors,length,10,count)
 new=context.reference_indices(mid,neighbors,length,10,count)
 assert len(new)==len(old) and new==sorted(set(new))
 assert not set(new)&set(neighbors) and all(0<=i<length for i in new)
 assert context.reference_indices(mid,neighbors,length,10,count,policy=context.LOCAL)==old
 if length==128 and mid==0:
  assert old==[10,20,30] and new==[26,67,107]

def test_isolated_body_reuses_code_without_modifying_original_globals():
 before=ProtectedProPainter.inpaint.__globals__['_get_ref_index']
 parent=synthetic_parent();painter,log=context.make_context_painter(parent)
 assert painter.inpaint.__func__.__code__ is ProtectedProPainter.inpaint.__code__
 assert painter.inpaint.__func__.__globals__ is not ProtectedProPainter.inpaint.__globals__
 assert ProtectedProPainter.inpaint.__globals__['_get_ref_index'] is before
 assert painter.parent is parent and log['original_globals_modified'] is False

@pytest.mark.parametrize('experiment',context.EXPERIMENTS)
def test_three_calls_same_masks_rng_and_native_output(monkeypatch,experiment):
 monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
 parent=synthetic_parent();arrays=fixture_arrays(128);before=[v.copy() for v in arrays];calls=[]
 result=context.run_context_clean_pair(parent,*arrays,experiment=experiment,authority=lambda:None,on_call=calls.append)
 assert calls==['A','B','A_repeat']
 assert result['receipt']['A_repeat_reproducibility']['exact_equal']
 assert result['receipt']['same_mask_weights_settings_precision_backend_and_rng']
 assert all(np.array_equal(a,b) for a,b in zip(arrays,before))
 assert all(result[k].shape==arrays[0].shape for k in ['A','B','A_repeat'])
 for v in result['validations'].values():
  assert v['status']=='PASS' and not any(v['protected_changed_pixels_per_frame'])
  assert not any(v['outside_delete_changed_pixels_per_frame'])
 log=result['receipt']['calls'][1]
 assert all(c['same_count'] for c in log['reference_context']['calls'])
 if experiment==context.REVERSE:
  assert log['internal_to_native_frame_indices']==list(range(127,-1,-1))
 else:
  assert any(c['changed'] for c in log['reference_context']['calls'])

def test_backend_drift_rejected_before_any_call():
 from chaoyang.inpainting_candidate.v1_clean_pair import runtime_backend
 parent=synthetic_parent();backend=deepcopy(runtime_backend(torch,'cpu'));backend['flags']['cudnn_benchmark']=not backend['flags']['cudnn_benchmark']
 with pytest.raises(ValueError,match='BACKEND_DRIFT'):
  context.run_context_clean_pair(parent,*fixture_arrays(4),experiment=context.GLOBAL_EQUAL,
   authority=lambda:None,on_call=lambda v:pytest.fail('must not infer'),backend_before_load=backend)
 assert not parent.raft_records
