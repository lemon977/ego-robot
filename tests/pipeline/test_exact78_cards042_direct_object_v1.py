import numpy as np
from chaoyang.ops.run_exact78_cards042_direct_object_v1 import (
    associated_component, hint_points, rgb_components, select_rgb_component)


def test_hint_keyframe_exact():
    assert hint_points(125)==[(555.0,564.0),(640.0,485.0)]


def test_hint_interpolates_two_tracks():
    value=hint_points(135)
    assert 555 < value[0][0] < 626
    assert 485 < value[1][1] < 649


def test_rgb_component_prefers_prompt_hit():
    image=np.zeros((960,1280,3),np.uint8)
    image[500:560,600:680]=255
    rows=rgb_components(image,[(620,520),(640,530)])
    selected=select_rgb_component(rows)
    assert selected and selected['positive_hits']==2


def test_associated_component_discards_unprompted_blob():
    mask=np.zeros((100,100),bool);mask[10:20,10:20]=1;mask[50:70,50:70]=1
    selected,discarded,hits=associated_component(mask,[(55,55)])
    assert selected.sum()==400 and discarded==100 and hits==1
    assert type(discarded) is int and type(hits) is int
