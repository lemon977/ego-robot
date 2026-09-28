import numpy as np
import pytest
from chaoyang.ops.run_four_stream_completion_clean_role_refine import refine_masks

def fields():
    return [np.zeros((5,7),dtype=bool) for _ in range(5)]

def test_only_hand_confirmed_unknown_is_added():
    w,p,u,h,v=fields()
    w[0,0]=True
    u[1,1]=h[1,1]=True
    u[1,2]=True
    h[2,2]=True
    e,n,remaining,conflict=refine_masks(w,p,u,h,v)
    assert e.sum()==1 and e[1,1]
    assert n.sum()==2 and n[0,0]
    assert remaining.sum()==1 and remaining[1,2]
    assert not n[2,2]
    assert not conflict.any()

def test_visible_card_and_hand_overlap_stays_unknown():
    w,p,u,h,v=fields()
    u[1,1]=h[1,1]=v[1,1]=True
    e,n,remaining,conflict=refine_masks(w,p,u,h,v)
    assert not e.any() and not n.any()
    assert remaining[1,1] and conflict[1,1]

def test_protected_card_kernel_never_added():
    w,p,u,h,v=fields()
    p[1,1]=u[1,1]=h[1,1]=True
    e,n,remaining,conflict=refine_masks(w,p,u,h,v)
    assert not e.any() and not n.any() and remaining[1,1]

def test_unknown_without_hand_does_not_become_write():
    w,p,u,h,v=fields();u[:]=True
    e,n,remaining,conflict=refine_masks(w,p,u,h,v)
    assert not n.any() and np.array_equal(remaining,u)

@pytest.mark.parametrize('which',['protect','unknown'])
def test_predecessor_write_conflicts_fail_closed(which):
    w,p,u,h,v=fields();w[1,1]=True
    (p if which=='protect' else u)[1,1]=True
    with pytest.raises(ValueError,match='PREDECESSOR_MASK_CONFLICT'):
        refine_masks(w,p,u,h,v)

def test_uint16_or_wrong_domain_not_silently_used_as_boolean():
    w,p,u,h,v=fields()
    with pytest.raises(ValueError,match='BOOLEAN_MATCHING_DOMAIN_REQUIRED'):
        refine_masks(w,p,u,h.astype(np.uint16),v)
    with pytest.raises(ValueError,match='BOOLEAN_MATCHING_DOMAIN_REQUIRED'):
        refine_masks(w,p,u,h[:2],v)
