import numpy as np
import pytest
from chaoyang.ops import run_four_stream_completion_clean_context as p
from chaoyang.ops import run_four_stream_completion_clean as c01

class Core:
    def __init__(self):self.inputs=None
    def run(self,outputs,inputs):
        self.inputs=inputs
        return [np.full((1,3,512,512),211,np.float32)]

def test_context_covers_original_hand_but_paste_scope_unchanged():
    prop=np.full((720,960,3),100,np.uint8);res=np.zeros((720,960),np.uint8);res[100:200,100:200]=255
    hand=np.zeros((960,1280),bool);hand[500:700,500:700]=True
    core=Core();wrapper=p.ContextSession(core,res,hand)
    internal=c01._apply_lama_residual(wrapper,"out",prop,res)
    assert np.array_equal(internal[res==0],prop[res==0])
    assert np.all(internal[res>0]==211)
    assert wrapper.probe["conditioning_added_from_hand_px"]>0
    assert wrapper.probe["unmasked_hand_support_px"]==0
    assert np.all(core.inputs["mask"][p.hand_context(hand)>0]==1)

def test_no_hand_support_reduces_exactly_to_prior_mask():
    res=np.zeros((720,960),np.uint8);res[20:100,20:100]=255
    w=p.ContextSession(Core(),res,np.zeros((960,1280),bool))
    assert np.array_equal(w.support,w.previous_support)

def test_empty_output_residual_never_calls_model_even_if_hand_support():
    prop=np.full((720,960,3),123,np.uint8);res=np.zeros((720,960),np.uint8)
    core=Core();w=p.ContextSession(core,res,np.ones((960,1280),bool))
    assert np.array_equal(c01._apply_lama_residual(w,"out",prop,res),prop)
    assert core.inputs is None and w.probe is None

def test_hand_bad_domain_rejected():
    with pytest.raises(ValueError,match="HAND_DOMAIN"):p.hand_context(np.zeros((960,1280),np.uint8))

def test_caller_mask_mismatch_rejected():
    w=p.ContextSession(Core(),np.zeros((720,960),np.uint8),np.zeros((960,1280),bool))
    with pytest.raises(RuntimeError,match="UNEXPECTED_CALLER_MASK"):
        w.run(["out"],{"mask":np.ones((1,1,512,512),np.float32),"image":np.zeros((1,3,512,512),np.float32)})

def test_visible_card_guard_changes_only_authorized14_and_fair_reference():
    raw=np.full((8,8,3),11,np.uint8);reference=np.full_like(raw,211)
    write=np.ones((8,8),bool);protect=np.zeros_like(write);card=np.zeros_like(write);card.flat[:14]=True
    w,p2,fair,metrics=p.visible_card_guard(raw,reference,write,protect,card)
    assert metrics["write_removed_px"]==14
    assert metrics["reference_restored_changed_px"]==14
    assert not (w&card).any() and np.array_equal(p2,card)
    assert np.array_equal(fair[card],raw[card]) and np.array_equal(fair[~card],reference[~card])
    assert np.all(reference==211) and np.all(write)

def test_visible_card_guard_rejects_invalid_domain():
    raw=np.zeros((8,8,3),np.uint8);m=np.zeros((8,8),bool)
    with pytest.raises(ValueError,match="GUARD_MASK_DOMAIN"):
        p.visible_card_guard(raw,raw,m,m,m.astype(np.uint8))
