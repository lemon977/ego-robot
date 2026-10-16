import numpy as np
import pytest
from chaoyang.inpainting_candidate.v1_scene_donor import donor_fill,registrations


def synthetic():
    n,h,w=6,24,32
    rgb=np.full((n,h,w,3),73,np.uint8);rgb[:,:,:,1]=101
    e=np.zeros((n,h,w),bool);e[0,8:15,10:18]=True
    p=np.zeros_like(e);p[0,0:3,0:3]=True
    clean=rgb.copy();clean[e]=255
    H=np.tile(np.eye(3),(n,2,1,1))
    return rgb,clean,e,p,H


def test_real_source_addresses_and_fixed_pixels():
    raw,clean,e,p,H=synthetic()
    result,prov,count,stats=donor_fill(raw,clean,e,p,e|p,H)
    assert np.array_equal(result,raw)
    assert np.all(count[e]>=3)
    for frame,x,y in prov[e]:assert np.array_equal(raw[frame,y,x],[73,101,73])
    assert np.all(prov[~e]==-1)
    assert stats['registration_qualified'] is False and stats['training_eligible'] is False


def test_no_three_donors_no_fill():
    raw,clean,e,p,H=synthetic();H[3:]=np.nan
    result,prov,_,_=donor_fill(raw,clean,e,p,e|p,H)
    assert np.array_equal(result,clean) and np.all(prov==-1)


def test_disjoint_fits_disagree_no_fill():
    raw,clean,e,p,H=synthetic();H[1:,1,0,2]=2
    result,prov,_,_=donor_fill(raw,clean,e,p,e|p,H)
    assert np.array_equal(result,clean) and np.all(prov==-1)


def test_excluded_source_never_used():
    raw,clean,e,p,H=synthetic();exclude=e|p;exclude[1:]=True
    result,prov,_,_=donor_fill(raw,clean,e,p,exclude,H)
    assert np.array_equal(result,clean) and np.all(prov==-1)


def test_invalid_frame_registration_no_fake_identity():
    raw,clean,e,p,H=synthetic();H[0]=np.nan
    result,prov,_,_=donor_fill(raw,clean,e,p,e|p,H)
    assert np.array_equal(result,clean) and np.all(prov==-1)


def test_no_features_record_failure():
    raw,_,e,p,_=synthetic()
    H,rows,anchor=registrations(np.zeros_like(raw),e|p)
    assert rows[anchor]['status']=='ANCHOR_IDENTITY'
    assert all(r['status']=='NO_DESCRIPTORS' for i,r in enumerate(rows) if i!=anchor)
    assert np.isnan(H[1:]).all()


def test_violating_protected_domain_rejected():
    raw,clean,e,p,H=synthetic();p[e]=True
    with pytest.raises(ValueError,match='FROZEN_NATIVE_DOMAIN'):donor_fill(raw,clean,e,p,e|p,H)
