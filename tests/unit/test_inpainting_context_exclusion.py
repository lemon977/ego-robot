import numpy as np
import pytest
from chaoyang.inpainting_candidate.v1_context_exclusion import restore_fixed_write,execute

def fixture():
    raw=np.full((3,16,24,3),80,np.uint8);working=np.full_like(raw,180)
    effective=np.zeros(raw.shape[:3],bool);effective[:,3:12,3:12]=True
    protected=np.zeros_like(effective);protected[:,8:11,15:20]=True
    return raw,working,effective,protected

def test_broad_internal_output_cannot_delete_domain_outside_fixed_mask():
    raw,working,e,p=fixture();result,v=restore_fixed_write(raw,working,e,p)
    assert np.array_equal(result[~e],raw[~e]) and np.array_equal(result[e],working[e])
    assert np.array_equal(result[p],raw[p]) and not any(v['protected_changed_pixels_per_frame'])
    assert not any(v['outside_delete_changed_pixels_per_frame'])

def test_inputs_not_mutated():
    raw,working,e,p=fixture();before=[v.copy()for v in [raw,working,e,p]]
    restore_fixed_write(raw,working,e,p)
    assert all(np.array_equal(a,b)for a,b in zip([raw,working,e,p],before))

def test_domain_drift_fails_closed():
    raw,working,e,p=fixture();p[:,4,4]=True
    with pytest.raises(ValueError,match='MASK_DOMAIN'):restore_fixed_write(raw,working,e,p)

def test_model_call_budget_is_fixed_three_and_no_render():
    with pytest.raises(ValueError,match='STRICT_THREE'):
        execute({'mode':'Clean','config':{'context_exclusion_pair':True},'budgets':{'gpu_count':1,
            'model_calls_max':2,'render_calls_max':0,'training_steps_max':0},'sessions':[{}]},authority=lambda:pytest.fail('no imports'))
