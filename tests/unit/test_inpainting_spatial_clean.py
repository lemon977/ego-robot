import numpy as np
import pytest
from chaoyang.inpainting_candidate.v1_spatial_clean import spatial_clean,require_spatial,background_tone_clean,nonincreasing_brightness_guard


def test_spatial_fixed_domain_and_protected_bytes():
    raw=np.full((3,32,40,3),70,np.uint8);e=np.zeros(raw.shape[:3],bool);e[:,10:22,12:26]=True
    p=np.zeros_like(e);p[:,14:17,16:20]=True;e[p]=False
    raw[e]=255;raw[p]=[240,220,30]
    out,work=spatial_clean(raw,e,p)
    assert np.array_equal(work,e|p) and np.array_equal(out[p],raw[p])
    assert np.array_equal(out[~e],raw[~e]) and out[e].max()<100


def test_fill_frame_local_with_fixed_mask_only():
    raw=np.random.default_rng(0).integers(0,255,(4,24,32,3),dtype=np.uint8)
    e=np.zeros(raw.shape[:3],bool);e[:,6:18,8:20]=True;p=np.zeros_like(e)
    a,_=spatial_clean(raw,e,p);raw[2:]=255;b,_=spatial_clean(raw,e,p)
    assert np.array_equal(a[:2],b[:2])  # NOT a claim about the offline upstream Mask.


def test_no_whole_image_source_rejected():
    raw=np.zeros((2,8,8,3),np.uint8);e=np.ones(raw.shape[:3],bool)
    with pytest.raises(ValueError,match='NO_VISIBLE'):spatial_clean(raw,e,np.zeros_like(e))


def test_cpu_strict_contract_rejects_model_calls():
    p={'mode':'Clean','config':{'cpu_spatial_clean':True,'spatial_algorithm':'TELEA_FIXED3_PROTECTED_EXCLUDED'},
       'budgets':{k:0 for k in ['gpu_count','model_calls_max','render_calls_max','training_steps_max']},
       'sessions':[{'native_frames':3}],'model_weights':[]}
    require_spatial(p);p['budgets']['model_calls_max']=1
    with pytest.raises(ValueError,match='STRICT_ZERO_NEURAL'):require_spatial(p)


def test_tone_removes_synthetic_white_patch_without_touching_visible_pixels():
    raw=np.full((3,40,48,3),70,np.uint8);e=np.zeros(raw.shape[:3],bool);e[:,8:32,10:38]=True
    p=np.zeros_like(e);p[:,17:20,20:25]=True;e[p]=False
    raw[e]=245;raw[p]=[240,200,30]
    parent=raw.copy();parent[e]=200
    out,work,stats=background_tone_clean(raw,parent,e,p)
    assert np.array_equal(work,e|p) and np.array_equal(out[~e],raw[~e])
    import cv2
    full=np.stack([cv2.distanceTransform(v.astype(np.uint8),cv2.DIST_L2,5)>=4 for v in e]) & e
    assert np.array_equal(out[p],raw[p]) and np.all(out[full]<90)
    assert len(stats)==3


def test_tone_rejects_invalid_parent_pixels():
    raw=np.full((2,16,20,3),70,np.uint8);e=np.zeros(raw.shape[:3],bool);p=np.zeros_like(e)
    parent=raw.copy();parent[0,0,0]=100
    with pytest.raises(ValueError,match='PARENT_PIXEL_PRESERVATION'):background_tone_clean(raw,parent,e,p)


def test_scale_separated_band_records_exact_single_change():
    raw=np.full((2,48,64,3),70,np.uint8);e=np.zeros(raw.shape[:3],bool);e[:,10:38,12:52]=True
    p=np.zeros_like(e);parent=raw.copy();parent[e]=200
    raw[e]=240
    _,work,old=background_tone_clean(raw,parent,e,p)
    out,work2,new=background_tone_clean(raw,parent,e,p,detail_sigma=3.)
    assert np.array_equal(work,work2) and np.array_equal(out[~e],raw[~e])
    assert all(a['sigma_from_mask_max_distance']==b['sigma_from_mask_max_distance']
               and a['detail_MAD_RGB']==b['detail_MAD_RGB'] and b['detail_sigma']==3. for a,b in zip(old,new))


def test_guard_never_lifts_dark_structure_and_keeps_all_nonwrite_pixels():
    base=np.full((2,16,20,3),200,np.uint8);base[:,:8]=30
    proposal=np.full_like(base,100);e=np.ones(base.shape[:3],bool);p=np.zeros_like(e);p[:,:2]=True;e[p]=False
    out=nonincreasing_brightness_guard(base,proposal,e,p)
    assert np.array_equal(out[:,:8],base[:,:8]) and np.all(out[:,8:]==100)
    assert np.array_equal(out[~e],base[~e]) and np.all(out<=base)


def test_guard_preserves_channel_proportions_and_zero_pixels():
    base=np.zeros((1,4,5,3),np.uint8);base[0,2,2]=[240,120,60]
    proposal=base//2;e=np.ones(base.shape[:3],bool);p=np.zeros_like(e)
    out=nonincreasing_brightness_guard(base,proposal,e,p)
    assert np.array_equal(out,proposal)
