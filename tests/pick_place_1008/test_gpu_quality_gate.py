import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest

from chaoyang.pick_place_1008 import pipeline, visual


def _fixture(tmp_path, monkeypatch, *, quality_problems=(), point_status="COMPLETE"):
    raw=np.arange(3*8*8*3,dtype=np.uint8).reshape(3,8,8,3)
    removal=np.zeros((3,8,8),bool);removal[:,2:5,2:5]=True
    protected=np.zeros_like(removal);protected[:,1,1]=True
    metadata={"quality_problems":copy.deepcopy(list(quality_problems))}
    if point_status is not None:metadata["point_tracking"]={"status":point_status}
    masks={"core":removal.copy(),"removal":removal,"protected_visible":protected,
           "sam_core":removal.copy(),"metadata":metadata}
    seeds={"objects":[{"obj_id":1}],"raw_frame_index":0}
    calls=[]
    def build(sam,frames,session,visible_seeds=None):
        calls.append(("build",visible_seeds))
        assert frames is raw
        return masks
    monkeypatch.setattr(visual,"build_visual_masks",build)
    def no_decode(*args,**kwargs):
        raise AssertionError("provided original RGB unexpectedly decoded again")
    monkeypatch.setattr(pipeline.source,"load_rgb_frames",no_decode)
    class Painter:
        def __init__(self,parent):
            calls.append(("painter_init",None))
            self.settings=dict(visual.SETTINGS)
            self.effective_masks=removal.copy()
            self.last_validation={"fixture":"CPU_NO_MODEL_CALLS"}
        def inpaint(self,frames,delete,protected_pixels):
            calls.append(("inpaint",None))
            assert frames is raw and delete is removal and protected_pixels is protected
            return frames.copy()
    monkeypatch.setattr(visual,"ProtectedProPainter",Painter)
    folder=tmp_path/"session";folder.mkdir()
    return SimpleNamespace(raw=raw,removal=removal,protected=protected,masks=masks,metadata=metadata,
                           seeds=seeds,calls=calls,folder=folder,
                           models={"sam":object(),"propainter":SimpleNamespace(settings=dict(visual.SETTINGS))},session=SimpleNamespace())


def _run(f,seeded=True):
    return pipeline.gpu_stage(f.models,f.folder,f.session,visible_seeds=f.seeds if seeded else None,raw_rgb=f.raw)


def _assert_failure_evidence_and_no_inpainting(f):
    assert not any(event[0]in("painter_init","inpaint")for event in f.calls)
    assert {p.name for p in f.folder.iterdir()}=={"raw_rgb.npz","masks.npz","MASK_QUALITY_FAILURE.json"}
    with np.load(f.folder/"raw_rgb.npz",allow_pickle=False)as data:
        assert np.array_equal(data["rgb"],f.raw)
    with np.load(f.folder/"masks.npz",allow_pickle=False)as data:
        unpacked=np.unpackbits(data["removal"],axis=1,count=64).reshape(f.removal.shape).astype(bool)
        assert np.array_equal(unpacked,f.removal)
    evidence=json.loads((f.folder/"MASK_QUALITY_FAILURE.json").read_text())
    assert evidence["status"]=="MASK_QUALITY_NOT_ADMITTED"
    assert evidence["inpainting_performed"]is False
    assert evidence["training_eligible"]is False and evidence["control_ground_truth"]is False
    return evidence


@pytest.mark.parametrize("seeded",[True,False])
def test_explicit_mask_quality_failure_stops_before_propainter_with_sealed_input_evidence(tmp_path,monkeypatch,seeded):
    problem={"code":"CUP_ENTIRE_SESSION_SUPPORT_MISSING","severity":"QUALITY_FAILURE","frames":[0,1,2]}
    f=_fixture(tmp_path,monkeypatch,quality_problems=[problem])
    original=copy.deepcopy(f.metadata)
    with pytest.raises(RuntimeError,match="MASK_QUALITY_NOT_ADMITTED"):_run(f,seeded)
    evidence=_assert_failure_evidence_and_no_inpainting(f)
    assert problem in evidence["quality_failures"]
    assert evidence["metadata"]==original and f.metadata==original


@pytest.mark.parametrize("status",["PARTIAL_SUPPORT","NO_SUPPORTED_TRACKS","TRACKING_ERROR","PROPAGATION_ABORTED",None])
def test_incomplete_or_missing_seeded_point_status_blocks_inpainting_without_other_failure(tmp_path,monkeypatch,status):
    f=_fixture(tmp_path,monkeypatch,point_status=status)
    with pytest.raises(RuntimeError,match="MASK_QUALITY_NOT_ADMITTED"):_run(f)
    evidence=_assert_failure_evidence_and_no_inpainting(f)
    assert any(problem["code"]=="POINT_TRACKING_NOT_COMPLETE"for problem in evidence["quality_failures"])
    assert f.calls==[("build",f.seeds)]


def test_point_runtime_failure_and_cup_failure_both_remain_in_failure_evidence(tmp_path,monkeypatch):
    problem={"code":"CUP_ENTIRE_SESSION_PROTECTED_SUPPORT_MISSING","severity":"QUALITY_FAILURE","frames":[0,1,2]}
    f=_fixture(tmp_path,monkeypatch,quality_problems=[problem],point_status="TRACKING_ERROR")
    with pytest.raises(RuntimeError,match="MASK_QUALITY_NOT_ADMITTED"):_run(f)
    evidence=_assert_failure_evidence_and_no_inpainting(f)
    assert problem in evidence["quality_failures"]
    assert any(p["code"]=="POINT_TRACKING_NOT_COMPLETE"for p in evidence["quality_failures"])
    assert evidence["metadata"]["point_tracking"]["status"]=="TRACKING_ERROR"


@pytest.mark.parametrize("review_problem",[None,{"code":"ONE_PIXEL_CUP_WEARER_CONFLICT","severity":"REVIEW_REQUIRED","frames":[0]}])
def test_complete_masks_can_enter_cpu_stub_and_review_only_flags_remain_honest(tmp_path,monkeypatch,review_problem):
    f=_fixture(tmp_path,monkeypatch,quality_problems=[]if review_problem is None else[review_problem])
    result=_run(f)
    assert [event[0]for event in f.calls]==["build","painter_init","inpaint"]
    assert result["status"]=="COMPLETE"
    assert result["metadata"]==f.metadata
    assert not(f.folder/"MASK_QUALITY_FAILURE.json").exists()
    with np.load(f.folder/"clean_rgb.npz",allow_pickle=False)as data:
        assert np.array_equal(data["rgb"],f.raw)
    assert result["protected_pixel_channel_difference_count"]==0
    assert result["outside_effective_difference_count"]==0


def test_legacy_diagnostic_path_does_not_require_point_metadata(tmp_path,monkeypatch):
    f=_fixture(tmp_path,monkeypatch,point_status=None)
    result=_run(f,seeded=False)
    assert result["status"]=="COMPLETE"
    assert f.calls[0]==("build",None)
