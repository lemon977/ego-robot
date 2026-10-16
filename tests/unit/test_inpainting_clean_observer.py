import numpy as np
import pytest
import torch
from chaoyang.inpainting_candidate.v1_clean_observer import OriginalPathObserver
from tests.unit.test_inpainting_v1_mask_causal_wiring import causal_fixture


class Model:
    def img_propagation(self,masked,flows,masks,interpolation):
        self.pixel_result=(masked.reshape(-1,*masked.shape[2:]),masks.reshape(-1,*masks.shape[2:]))
        return self.pixel_result
    def __call__(self,frames,flows,masks,updated,count):
        self.forward_result=frames[:,:count].reshape(-1,*frames.shape[2:])
        return self.forward_result


def inputs():
    e=np.zeros((4,8,8),bool);e[:,2:6,2:6]=True
    mask=torch.from_numpy(e.astype(np.float32))[None,:,None]
    image=torch.full((1,4,3,8,8),.5)*(1-mask)
    return e,mask,image


def observer(model,e):return OriginalPathObserver(model,e,{'subvideo_length':80,'neighbor_length':10,'ref_stride':10})


def test_observer_does_not_modify_tensors_returns_or_rng():
    e,mask,image=inputs();before=image.clone();before_mask=mask.clone();rng=torch.get_rng_state().clone()
    model=Model();o=observer(model,e)
    returned=o.img_propagation(image,None,mask,'nearest')
    assert returned is model.pixel_result
    result=o(image,None,mask,mask,4)
    assert result is model.forward_result and torch.equal(before,image) and torch.equal(before_mask,mask)
    assert torch.equal(rng,torch.get_rng_state())
    meta=o.metadata();assert meta['pixel_calls']==meta['forward_calls']==1
    sample=meta['records'][0]['samples'][0]
    assert sample['mask']['mismatch_vs_frozen_effective_pixels']==0
    assert sample['nonzero_masked_normalized_values_in_effective']==0


def test_wrong_mask_is_recorded_not_hidden_as_a_pass():
    e,mask,image=inputs();o=observer(Model(),e)
    o.img_propagation(image,None,torch.zeros_like(mask),'nearest')
    assert o.records[0]['samples'][0]['mask']['mismatch_vs_frozen_effective_pixels']==16


def test_unregistered_context_shape_fails():
    e,mask,image=inputs();o=observer(Model(),e)
    with pytest.raises(ValueError,match='NATIVE_SHAPE'):o.img_propagation(image[:,:2],None,mask[:,:2],'nearest')


def test_partial_observation_not_sealed():
    e,_,_=inputs();o=observer(Model(),e)
    with pytest.raises(ValueError,match='INCOMPLETE'):o.metadata()


def test_repeat_input_mismatch_is_not_silently_overwritten():
    e,_,_=inputs();o=observer(Model(),e);o.save('same_key',np.ones(4))
    with pytest.raises(ValueError,match='REPEATED_NATIVE'):o.save('same_key',np.zeros(4))


def test_full_registered_worker_trace_matches_original_three_calls(causal_fixture):
    import json
    from chaoyang.inpainting_candidate import v1_mco_worker
    f=causal_fixture
    f.packet['config'].update(mask_causal_pair=False,context_clean_pair=True,
        context_experiment='TRACE_UNCHANGED_ORIGINAL_PATH')
    f.packet['inputs']['frozen_repaired_masks']=f.packet['inputs']['frozen_base_masks']
    assert v1_mco_worker.execute_v1_mco_stage(f.packet)==0
    p=f.output/'sessions/synthetic_session/Clean'
    trace=json.loads((p/'PP_OBSERVATION.json').read_text())
    receipt=json.loads((p/'CLEAN_CONTEXT_PAIR_RECEIPT.json').read_text())
    assert trace['pixel_calls']==trace['forward_calls']==1
    assert not any(receipt['A_B_difference_pixels_per_frame'])
    assert receipt['A_repeat_reproducibility']['exact_equal']
