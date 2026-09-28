import numpy as np
import pytest
from chaoyang.ops import run_four_stream_completion_clean_production as p

class Model:
    def __init__(self,invalid=False):self.calls=0;self.invalid=invalid
    def run(self,names,inputs):
        self.calls+=1
        assert inputs["image"].shape==(1,3,512,512)
        assert inputs["image"].min()>=0 and inputs["image"].max()<=1
        q=np.empty((1,3,512,512),np.float32);q[:,0]=211;q[:,1]=31;q[:,2]=71
        if self.invalid:q[0,0,0,0]=np.nan
        return [q]

def inputs():
    prop=np.full((720,960,3),100,np.uint8);res=np.zeros((720,960),np.uint8);res[100:500,100:700]=255
    raw=np.full((960,1280,3),17,np.uint8);write=np.zeros((960,1280),bool);write[80:800,80:1000]=True
    protect=np.zeros_like(write);protect[90:120,90:120]=True
    unknown=np.zeros_like(write);unknown[130:160,130:160]=True;write&=~(protect|unknown)
    return prop,res,raw,write,protect,unknown

def test_actual_production_caller_calls_same_fixed_adapter():
    w=p.consumer_wiring()
    assert w["same_helper_object"] and w["legacy_full_entry_not_executed"]

def test_production_consumer_color_211_and_raw_boundaries():
    m=Model();args=inputs();out,internal,labels,metrics=p.consume(m,"out",*args)
    assert m.calls==1 and internal[200,200].tolist()==[71,31,211]
    assert out[300,300].tolist()==[71,31,211]
    raw=args[2];write,protect,unknown=args[3:]
    assert np.array_equal(out[~write],raw[~write])
    assert np.array_equal(out[protect],raw[protect])
    assert np.array_equal(out[unknown],raw[unknown])
    assert all(metrics[k]==0 for k in ("outside_write_changed","protected_changed","unknown_changed"))
    assert set(np.unique(labels))=={0,1,2,3,4,5}
    assert sum(metrics["source_pixel_counts"].values())==960*1280

def test_empty_residual_does_not_call_model_or_claim_lama():
    args=list(inputs());args[1][:]=0;m=Model()
    out,internal,labels,metrics=p.consume(m,"out",*args)
    assert m.calls==0 and not np.isin(labels,[2,3]).any()
    assert np.array_equal(internal,args[0])
    assert metrics["raw_fallback_inside_write"] is False

def test_conflicting_write_rejected_before_inference():
    args=list(inputs());args[3][95,95]=True;m=Model()
    with pytest.raises(ValueError,match="WRITE_CONFLICT"):p.consume(m,"out",*args)
    assert m.calls==0

def test_invalid_model_fails_without_raw_success():
    with pytest.raises(RuntimeError,match="LAMA_OUTPUT_INVALID"):p.consume(Model(True),"out",*inputs())

def test_wrong_mask_domain_rejected():
    args=list(inputs());args[4]=args[4].astype(np.uint8)
    with pytest.raises(ValueError,match="MASK_DOMAIN"):p.consume(Model(),"out",*args)
