import numpy as np
import pytest
from chaoyang.inpainting_candidate.v1_context_seam import reconstruct_fixed_boundary,require_candidate


def test_harmonic_offset_removes_known_boundary_step_without_changing_texture():
    y,x=np.mgrid[:17,:19];rgb=np.stack([60+x+y,90+x,80+y],axis=-1).astype(np.uint8)
    work=rgb+np.uint8(25);e=np.zeros((17,19),bool);e[3:-3,4:-4]=True;p=np.zeros_like(e)
    out,d=reconstruct_fixed_boundary(rgb,work,e,p)
    np.testing.assert_array_equal(out,rgb)
    assert d['status']=='SOLVED'and d['outside_E_changed_pixels']==0
    assert d['linear_residual_max_abs']<1e-8


def test_raw_human_pixels_inside_write_domain_do_not_enter_solution():
    raw=np.full((15,17,3),75,np.uint8);work=np.full_like(raw,100)
    e=np.zeros(raw.shape[:2],bool);e[2:-2,2:-2]=True;p=np.zeros_like(e)
    first,_=reconstruct_fixed_boundary(raw,work,e,p)
    raw[e]=[255,0,0]
    second,_=reconstruct_fixed_boundary(raw,work,e,p)
    np.testing.assert_array_equal(first,second)


def test_protected_hole_is_exact_raw_boundary():
    raw=np.full((15,17,3),80,np.uint8);work=np.full_like(raw,100)
    e=np.zeros(raw.shape[:2],bool);e[2:-2,2:-2]=True
    p=np.zeros_like(e);p[6:9,6:9]=True;e[p]=False;raw[p]=[190,130,50]
    out,d=reconstruct_fixed_boundary(raw,work,e,p)
    np.testing.assert_array_equal(out[p],raw[p]);np.testing.assert_array_equal(out[~e],raw[~e])
    assert d['protected_changed_pixels']==0


def test_touching_image_edge_is_supported_if_domain_has_raw_boundary():
    raw=np.full((15,17,3),80,np.uint8);work=np.full_like(raw,100)
    e=np.zeros(raw.shape[:2],bool);e[:,8:]=True
    out,d=reconstruct_fixed_boundary(raw,work,e,np.zeros_like(e))
    np.testing.assert_array_equal(out,raw);assert d['outside_E_changed_pixels']==0


def test_empty_and_unanchored_domains():
    raw=np.full((5,7,3),80,np.uint8);p=np.zeros((5,7),bool)
    out,d=reconstruct_fixed_boundary(raw,raw.copy(),p,p)
    np.testing.assert_array_equal(out,raw);assert d['status']=='EMPTY_WRITE_DOMAIN'
    with pytest.raises(ValueError,match='BOUNDARY_ABSENT'):
        reconstruct_fixed_boundary(raw,raw,np.ones_like(p),p)


def test_invalid_dtype_and_protected_conflict_rejected():
    raw=np.full((5,7,3),80,np.uint8);e=np.zeros((5,7),bool);e[2,2]=True
    with pytest.raises(ValueError,match='PROTECTED_WRITE_CONFLICT'):
        reconstruct_fixed_boundary(raw,raw,e,e)
    with pytest.raises(ValueError,match='RGB_BOOL_DOMAIN'):
        reconstruct_fixed_boundary(raw.astype(float),raw,e,np.zeros_like(e))


def test_consumer_requires_registered_full_native_zero_model_cpu_contract():
    packet={'mode':'Clean','config':{'cpu_context_seam':True,
        'algorithm':'POISSON_WORKING_GRADIENT_RAW_FIXED_BOUNDARY'},
        'budgets':{k:0 for k in ['gpu_count','model_calls_max','render_calls_max','training_steps_max']},
        'model_weights':[],'sessions':[{'session_key':'synthetic','native_frames':5}]}
    require_candidate(packet)
    packet['budgets']['gpu_count']=1
    with pytest.raises(ValueError,match='FULL_NATIVE_REQUIRED'):require_candidate(packet)
    packet['budgets']['gpu_count']=0;packet['sessions'][0]['output_frames']=[0]
    with pytest.raises(ValueError,match='FULL_NATIVE_REQUIRED'):require_candidate(packet)


def test_uncertain_raw_boundary_colour_cannot_diffuse_into_write_domain():
    raw=np.full((17,19,3),80,np.uint8);work=np.full_like(raw,80)
    e=np.zeros(raw.shape[:2],bool);e[3:14,4:15]=True;p=np.zeros_like(e)
    raw[8,15]=255;allowed=np.ones_like(e);allowed[8,15]=False
    classic,_=reconstruct_fixed_boundary(raw,work,e,p)
    veto,d=reconstruct_fixed_boundary(raw,work,e,p,raw_boundary_allowed=allowed)
    assert classic[8,14].mean()>80
    assert np.all(veto[e]==80)and np.array_equal(veto[~e],raw[~e])
    assert d['raw_boundary_veto_pairs']>0
    # A different unknown Raw colour is still not a gradient donor.
    raw[8,15]=[0,255,0]
    second,_=reconstruct_fixed_boundary(raw,work,e,p,raw_boundary_allowed=allowed)
    np.testing.assert_array_equal(veto[e],second[e])


def test_boundary_veto_preserves_allowed_background_correction():
    raw=np.full((17,19,3),80,np.uint8);work=np.full_like(raw,100)
    e=np.zeros(raw.shape[:2],bool);e[3:14,4:15]=True;p=np.zeros_like(e)
    allowed=np.ones_like(e)
    normal,_=reconstruct_fixed_boundary(raw,work,e,p)
    same,_=reconstruct_fixed_boundary(raw,work,e,p,raw_boundary_allowed=allowed)
    np.testing.assert_array_equal(normal,same)
    with pytest.raises(ValueError,match='BOUNDARY_SOURCE_BOOL_DOMAIN'):
        reconstruct_fixed_boundary(raw,work,e,p,raw_boundary_allowed=allowed.astype(int))
