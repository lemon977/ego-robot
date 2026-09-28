import cv2
import numpy as np
import pytest
from chaoyang.ops.run_four_stream_completion_clean_mask_support import support_mask, SupportSession
from chaoyang.ops.run_human_to_robot_shared_delivery_clean import _apply_lama_residual

def fixture():
    mask=np.zeros((720,960),dtype=np.uint8)
    mask[103:204,307:409]=255
    image=np.zeros((720,960,3),dtype=np.uint8)
    image[mask>0]=127
    return image,mask

def test_nearest_mask_negative_exposes_gray_pixels_as_known():
    image,mask=fixture()
    resized=cv2.resize(image,(512,512),interpolation=cv2.INTER_LINEAR)
    old=cv2.resize(mask,(512,512),interpolation=cv2.INTER_NEAREST)>0
    assert np.any((resized[:,:,0]>0)&~old)

def test_linear_support_covers_every_gray_contribution():
    image,mask=fixture()
    alpha=cv2.resize((mask>0).astype(np.float32),(512,512),interpolation=cv2.INTER_LINEAR)
    new=support_mask(mask)[0,0]>0
    resized=cv2.resize(image,(512,512),interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(new,alpha>0)
    assert not np.any((resized[:,:,0]>0)&~new)

def test_wrapper_keeps_image_tensor_unchanged_and_original_write_scope():
    image,mask=fixture()
    seen=[]
    class Session:
        def run(self,outputs,inputs):
            expected=cv2.resize(image,(512,512),interpolation=cv2.INTER_LINEAR).transpose(2,0,1)[None].astype(np.float32)/255
            assert np.array_equal(inputs["image"],expected)
            assert np.array_equal(inputs["mask"],support_mask(mask))
            seen.append(True)
            return [np.full((1,3,512,512),211,dtype=np.float32)]
    wrapper=SupportSession(Session(),mask)
    output=_apply_lama_residual(wrapper,"output",image,mask)
    assert seen==[True]
    assert np.array_equal(output[mask==0],image[mask==0])
    assert np.all(output[mask>0]==211)
    assert wrapper.probe["added_mask_px"]>0

def test_empty_mask_skips_model_with_null_probe():
    image,mask=fixture();mask[:]=0
    class Session:
        def run(self,*args):raise AssertionError("No inference")
    wrapper=SupportSession(Session(),mask)
    output=_apply_lama_residual(wrapper,"output",image,mask)
    assert wrapper.probe is None
    assert np.array_equal(output,image)

def test_wrong_mask_caller_rejected():
    image,mask=fixture()
    wrapper=SupportSession(None,mask)
    with pytest.raises(RuntimeError,match="UNEXPECTED_CALLER_MASK"):
        wrapper.run(["output"],{"mask":np.zeros((1,1,512,512),np.float32),"image":None})

def test_wrong_source_mask_domain_rejected():
    with pytest.raises(ValueError,match="MASK_DOMAIN"):
        support_mask(np.zeros((512,512)))
