from types import SimpleNamespace
import copy
import numpy as np
import pytest

from chaoyang.pick_place_1008 import visual


def _fixture(monkeypatch, *, anchor=0):
    n, h, w = 4, 30, 40
    rgb = np.zeros((n, h, w, 3), np.uint8)
    session = SimpleNamespace(frame_count=n, width=w, height=h, task="pick_place",
                              motion_valid={"hand_display_valid": np.ones((2,n),bool)},
                              native_motion={"head_source": np.array(["RAW_POSE"]*n)})
    geometry = {"hands": np.full((2,n,21,2),[20.,20.]), "hands_inside":np.ones((2,n,21),bool),
                "forearm":np.full((2,n,6,2),[20.,20.]),
                "tracker":np.full((2,n,36,2),[20.,20.]),
                "forearm_radius":np.full((2,n),100.),"wrist_px_per_m":np.full((2,n),1000.)}
    monkeypatch.setattr(visual.parent_mask,"pico_geometry",lambda *args: geometry)
    def no_geometry_authority(*args,**kwargs):
        raise AssertionError("legacy hull/provider deletion path called")
    monkeypatch.setattr(visual.parent_mask,"build_masks",no_geometry_authority)
    monkeypatch.setattr(visual.parent_mask,"_providers",no_geometry_authority)
    monkeypatch.setattr(visual.parent_mask,"_hull_mask",no_geometry_authority)
    seeds=[]
    tracks={}
    objects={}
    masks={}
    for obj_id,side,role,roi,point in [
        (1,"left","glove",(slice(20,28),slice(2,10)),[5,24]),
        (2,"left","controller",(slice(11,15),slice(2,8)),[5,13]),
        (5,"right","glove",(slice(20,28),slice(30,38)),[34,24]),
        (6,"right","controller",(slice(11,15),slice(31,37)),[34,13])]:
        seed={"obj_id":obj_id,"side":side,"role":role,"frame_idx":anchor,
              "points_px":[point,[20,4]],"point_labels":[1,0],"visible_confirmed":True}
        seeds.append(seed)
        mask=np.zeros((n,h,w),bool);mask[:,roi[0],roi[1]]=True;masks[obj_id]=mask
        tracks[obj_id]={frame:np.packbits(mask[frame])for frame in range(n)}
        objects[str(obj_id)]={"role":role,"side":side,"anchor_mask_present":True,"status":"SUPPORTED",
                              "track_frame_indices":list(range(n))}
    point_meta={"status":"COMPLETE","objects":objects,"errors":[],"aborted":[]}
    calls=[]
    def track(sam,frames,passed_seeds,passed_anchor):
        calls.append(("points",passed_anchor))
        assert passed_seeds==seeds
        return tracks,point_meta
    monkeypatch.setattr(visual,"_point_tracks",track)
    cup=np.zeros((n,h,w),bool);cup[:,2:8,17:22]=True
    class Sam:
        aborted=[]
        def track_text(self,frames,prompt,passed_anchor):
            calls.append(("text",prompt,passed_anchor))
            assert prompt==visual.PROMPTS["object"]
            return {99:{frame:np.packbits(cup[frame])for frame in range(n)}}
    return SimpleNamespace(n=n,h=h,w=w,rgb=rgb,session=session,seeds=seeds,
                           tracks=tracks,point_meta=point_meta,masks=masks,cup=cup,
                           sam=Sam(),calls=calls,geometry=geometry)


def _run(fixture, packet=False):
    seeds={"raw_frame_index":fixture.seeds[0]["frame_idx"],"objects":fixture.seeds}if packet else fixture.seeds
    return visual.build_visual_masks(fixture.sam,fixture.rgb,fixture.session,visible_seeds=seeds)


@pytest.mark.parametrize("packet",[False,True])
def test_point_path_observed_segments_are_only_core_authority_not_any_geometry_hull(monkeypatch,packet):
    f=_fixture(monkeypatch)
    original=copy.deepcopy(f.seeds)
    out=_run(f,packet)
    union=np.logical_or.reduce(list(f.masks.values()))
    assert np.array_equal(out["core"],union)
    assert not out["core"][:,20,20].any() and not out["removal"][:,20,20].any()
    assert np.array_equal(out["sam_core"],f.masks[1]|f.masks[5])
    assert np.array_equal(out["equipment"],f.masks[2]|f.masks[6])
    assert f.calls==[("points",0),("text",visual.PROMPTS["object"],0)]
    assert f.seeds==original
    meta=out["metadata"]
    assert meta["geometry_diagnostics"]["geometry_deletion_authority"]is False
    assert meta["geometry_diagnostics"]["cross_side_registration"]is False
    assert meta["geometry_diagnostics"]["unsupported_hand_hull_deletion"]is False
    assert meta["foreground_ids_per_frame"]==[[1,2,5,6]]*f.n
    assert meta["cup_support"]["cup_observed_pixels_per_frame"]==[30]*f.n
    assert meta["quality_problems"]==[]


def test_nonzero_anchor_keeps_seed_id_and_does_not_use_geometry_to_pick_another_anchor(monkeypatch):
    f=_fixture(monkeypatch,anchor=2)
    out=_run(f,True)
    assert out["metadata"]["anchor_frame"]==2
    assert f.calls[0]==("points",2)and f.calls[1][-1]==0


def test_no_anchor_support_excludes_later_prediction_and_reports_required_role_failure(monkeypatch):
    f=_fixture(monkeypatch)
    f.point_meta["objects"]["1"]["anchor_mask_present"]=False
    f.point_meta["status"]="PARTIAL_SUPPORT"
    out=_run(f)
    assert not (out["core"]&f.masks[1]).any()
    codes=[row["code"]for row in out["metadata"]["quality_problems"]]
    assert "VISIBLE_ROLE_ANCHOR_SUPPORT_MISSING"in codes
    assert "REQUIRED_VISIBLE_GLOVE_OR_CONTROLLER_SUPPORT_MISSING"in codes
    assert "POINT_TRACK_SUPPORT_OR_RUNTIME_INCOMPLETE"in codes
    assert not out["metadata"]["foreground_role_support"]["1"]["anchor_mask_present"]


def test_empty_observed_frame_does_not_get_filled_by_a_projected_hand(monkeypatch):
    f=_fixture(monkeypatch)
    for obj_id in f.tracks:
        f.tracks[obj_id][2]=np.packbits(np.zeros((f.h,f.w),bool))
    out=_run(f)
    assert not out["core"][2].any()
    assert out["metadata"]["foreground_ids_per_frame"][2]==[]
    # Temporal bracket is explicit inference; it never becomes observed core.
    assert out["metadata"]["removal"]["bracket_added_pixels_per_frame"][2]>0


def test_missing_current_cup_support_is_not_carried_and_cannot_qualify_from_other_frames(monkeypatch):
    f=_fixture(monkeypatch)
    f.cup[2]=False
    out=_run(f)
    assert not out["cup_observed"][2].any()and not out["protected_visible"][2].any()
    rows=out["metadata"]["quality_problems"]
    empty=next(row for row in rows if row["code"]=="CUP_CURRENT_FRAME_SUPPORT_EMPTY_OR_OCCLUDED")
    assert empty["severity"]=="REVIEW_REQUIRED"and empty["frames"]==[2]
    assert not any(row["severity"]=="QUALITY_FAILURE"for row in rows)
    assert out["metadata"]["cup_support"]["whole_video_has_cup_support"]is True
    assert any(row["code"]=="CUP_CURRENT_FRAME_SUPPORT_SUDDEN_DROP"for row in rows)
    assert out["protected_visible"][0].any()


def test_foreground_cup_overlap_is_separate_from_unknown_order_and_flags_protection_shrink(monkeypatch):
    f=_fixture(monkeypatch)
    f.masks[5][1]|=f.cup[1]
    f.tracks[5][1]=np.packbits(f.masks[5][1])
    out=_run(f)
    assert np.array_equal(out["semantic_conflict"][1],f.cup[1])
    assert not out["protected_visible"][1].any()
    assert np.array_equal(out["unknown"],out["protected_visible"])
    codes=[row["code"]for row in out["metadata"]["quality_problems"]]
    assert "CUP_ALL_OBSERVED_SUPPORT_VETOED_BY_FOREGROUND"in codes
    assert "CUP_PROTECTED_SUPPORT_SHRUNK_BY_FOREGROUND"in codes
    assert "PAPER_CUP_WEARER_SEMANTIC_CONFLICT"in codes


def test_compatible_effective_mask_preserves_current_visible_cup_after_parent_dilation(monkeypatch):
    f=_fixture(monkeypatch)
    # Put current cup beside a tracked controller so internal dilation reaches it.
    f.cup[:]=False;f.cup[:,8:11,5:8]=True
    out=_run(f)
    effective,validation=visual.effective_clean_masks(out["removal"],out["protected_visible"])
    assert not(effective&out["protected_visible"]).any()
    assert any(validation["dilated_protected_overlap_per_frame"])
    assert not(effective[:,20,20]).any()


@pytest.mark.parametrize("mutation",["duplicate","outside","frame_alias_conflict","empty_fg","unknown_side","unknown_role"])
def test_invalid_seed_domains_fail_before_model_calls(monkeypatch,mutation):
    f=_fixture(monkeypatch)
    if mutation=="duplicate":f.seeds[1]["obj_id"]=f.seeds[0]["obj_id"]
    elif mutation=="outside":f.seeds[0]["points_px"][0]=[40,24]
    elif mutation=="frame_alias_conflict":f.seeds[1]["frame_index"]=1
    elif mutation=="empty_fg":f.seeds[0]["point_labels"]=[0,0]
    elif mutation=="unknown_side":f.seeds[0]["side"]="both"
    elif mutation=="unknown_role":f.seeds[0]["role"]="robot"
    with pytest.raises(ValueError):_run(f)
    assert f.calls==[]


def test_unknown_adapter_object_id_and_frame_are_rejected(monkeypatch):
    f=_fixture(monkeypatch);f.tracks[15]={0:np.packbits(f.masks[1][0])}
    with pytest.raises(ValueError,match="UNKNOWN_OBJECT_ID"):_run(f)
    f.tracks.pop(15);f.tracks[1][f.n]=np.packbits(f.masks[1][0])
    with pytest.raises(ValueError,match="TRACK_FRAME_DOMAIN"):_run(f)


def test_real_cpu_point_adapter_integrates_and_clears_state_before_cup_text(monkeypatch):
    from chaoyang.pick_place_1008 import sam_point_adapter
    f=_fixture(monkeypatch)
    events=[]
    class State(dict):
        def clear(self):
            events.append("point_state_clear")
            super().clear()
    class Model:
        def init_state(self,**kwargs):
            assert len(kwargs["resource_path"])==f.n
            self.ids=[];self.state=State(active=True);return self.state
        def _cache_frame_outputs(self,state,frame_idx,obj_id_to_mask,**kwargs):
            state.setdefault("cached_frame_outputs",{})[frame_idx]=obj_id_to_mask
        def output(self,frame):
            return {"out_obj_ids":np.asarray(self.ids,np.int64),
                    "out_binary_masks":np.stack([f.masks[obj][frame]for obj in self.ids])}
        def add_prompt(self,**kwargs):
            assert kwargs["text_str"]is None and kwargs["rel_coordinates"]is True
            assert kwargs["output_prob_thresh"]==.5
            assert kwargs["clear_old_points"]is True
            self.ids.append(kwargs["obj_id"]);events.append("add:"+str(kwargs["obj_id"]))
            return kwargs["frame_idx"],self.output(kwargs["frame_idx"])
        def propagate_in_video(self,**kwargs):
            for frame in range(f.n):
                yield frame,self.output(frame)
    f.sam.model=Model()
    parent_text=f.sam.track_text
    def checked_text(*args):
        assert events[-1]=="point_state_clear"
        events.append("cup_text")
        return parent_text(*args)
    f.sam.track_text=checked_text
    monkeypatch.setattr(visual,"_point_tracks",sam_point_adapter.track_visible_points)
    out=_run(f)
    assert out["metadata"]["point_tracking"]["status"]=="COMPLETE"
    assert out["metadata"]["point_tracking"]["one_video_state"]is True
    assert events==["add:1","add:2","add:5","add:6","point_state_clear","cup_text"]
    assert np.array_equal(out["core"],np.logical_or.reduce(list(f.masks.values())))
    assert out["metadata"]["foreground_ids_per_frame"]==[[1,2,5,6]]*f.n


@pytest.mark.parametrize("missing",["cup","protected"])
def test_entire_session_without_usable_cup_support_still_fails_closed(monkeypatch,missing):
    f=_fixture(monkeypatch)
    if missing=="cup":
        f.cup[:]=False
    else:
        for frame in range(f.n):
            f.masks[5][frame]|=f.cup[frame]
            f.tracks[5][frame]=np.packbits(f.masks[5][frame])
    out=_run(f)
    assert not out["protected_visible"].any()
    problems=out["metadata"]["quality_problems"]
    code=("CUP_ENTIRE_SESSION_SUPPORT_MISSING"if missing=="cup"
          else"CUP_ENTIRE_SESSION_PROTECTED_SUPPORT_MISSING")
    failure=next(row for row in problems if row["code"]==code)
    assert failure["severity"]=="QUALITY_FAILURE"
    assert failure["frames"]==list(range(f.n))
    assert out["metadata"]["cup_support"]["whole_video_has_protected_visible_support"]is False
    assert out["metadata"]["hidden_object_protection"]is False
    # No previous cup shape or projected hull creates protection for missing pixels.
    assert not out["unknown"].any()

def _reviewed_invisible_packet(f):
    validation = "NATIVE_QPC_AND_ORIGINAL_RGB_EXACT_VERIFIED"
    declaration = {
        "side": "left", "role": "controller",
        "visibility": "NOT_VISIBLE_IN_INSPECTED_NATIVE_FRAMES", "review": "AI_CONFIRMED_NOT_VISIBLE",
        "validation": validation, "global_all_frames_invisible": False,
        "unreviewed_visibility_status": "UNKNOWN", "later_visibility_review_needed": True,
        "inspected_frames": [{"raw_frame_index": frame, "qpc_ns": 1000 * (frame + 1),
                             "image": {"path": f"/raw/{frame}.png", "bytes": 12, "sha256": "a"*64}}
                            for frame in [0, 2, 3]],
    }
    f.seeds[:] = [seed for seed in f.seeds if seed["obj_id"] != 2]
    f.tracks.pop(2)
    f.point_meta["objects"].pop("2")
    return {"anchor_frame": 0, "objects": f.seeds, "seed_schema": "pick_place_visible_mask_seeds_v2",
            "seed_reference": {"path": "/validated/leaf.json", "bytes": 500, "sha256": "b"*64},
            "unseeded_roles": [declaration], "unseeded_roles_validation": validation}


def test_multi_anchor_keeps_each_object_actual_anchor_and_earliest_global_anchor(monkeypatch):
    f = _fixture(monkeypatch)
    f.seeds[0]["frame_idx"] = 2
    f.seeds[1]["frame_idx"] = 3
    out = _run(f, True)
    assert out["metadata"]["anchor_frame"] == 0
    assert out["metadata"]["foreground_role_support"]["1"]["anchor_raw_frame_index"] == 2
    assert out["metadata"]["foreground_role_support"]["2"]["anchor_raw_frame_index"] == 3
    assert np.array_equal(out["core"], np.logical_or.reduce(list(f.masks.values())))
    assert not out["metadata"]["geometry_diagnostics"]["geometry_deletion_authority"]


def test_reviewed_unseen_role_exempts_only_inspected_visibility_without_fabricated_pixels(monkeypatch):
    f = _fixture(monkeypatch)
    packet = _reviewed_invisible_packet(f)
    out = visual.build_visual_masks(f.sam, f.rgb, f.session, visible_seeds=packet)
    assert not out["core"][:, 12, 5].any()
    assert np.array_equal(out["core"], f.masks[1] | f.masks[5] | f.masks[6])
    meta = out["metadata"]
    assert meta["unseeded_role_declarations"] == packet["unseeded_roles"]
    assert meta["unreviewed_role_visibility_status"] == "UNKNOWN"
    assert meta["later_role_visibility_review_needed"] is True
    assert all(item["severity"] == "REVIEW_REQUIRED" for item in meta["quality_problems"])
    assert meta["visible_seed_evidence"]["seed_reference"] == packet["seed_reference"]
    assert meta["hidden_object_protection"] is False


@pytest.mark.parametrize("mutation", ["no_validation", "physical_truth", "global_claim", "unreviewed", "missing_mid"])
def test_unreviewed_or_false_unseen_role_cannot_exempt_quality_gate(monkeypatch, mutation):
    f = _fixture(monkeypatch)
    packet = _reviewed_invisible_packet(f)
    item = packet["unseeded_roles"][0]
    if mutation == "no_validation":
        packet.pop("unseeded_roles_validation")
    elif mutation == "physical_truth":
        item["physical_ground_truth"] = True
    elif mutation == "global_claim":
        item["global_all_frames_invisible"] = True
    elif mutation == "unreviewed":
        item["review"] = "PENDING"
    else:
        item["inspected_frames"] = item["inspected_frames"][:1] + item["inspected_frames"][-1:]
    with pytest.raises(ValueError, match="UNSEEDED"):
        visual.build_visual_masks(f.sam, f.rgb, f.session, visible_seeds=packet)
    assert f.calls == []


def test_actual_seed_anchor_failure_is_not_exempted_by_unseen_other_role(monkeypatch):
    f = _fixture(monkeypatch)
    packet = _reviewed_invisible_packet(f)
    f.point_meta["objects"]["5"]["anchor_mask_present"] = False
    out = visual.build_visual_masks(f.sam, f.rgb, f.session, visible_seeds=packet)
    failures = [item for item in out["metadata"]["quality_problems"] if item["severity"] == "QUALITY_FAILURE"]
    assert any(item.get("obj_id") == 5 for item in failures)
    assert any(item.get("side") == "right" and item.get("role") == "glove" for item in failures)

def test_cup_native_zero_anchor_is_independent_of_late_wearer_anchor(monkeypatch):
    f = _fixture(monkeypatch, anchor=3)
    out = _run(f, True)
    assert f.calls == [("points", 3), ("text", visual.PROMPTS["object"], 0)]
    assert out["metadata"]["anchor_frame"] == 3
    assert out["metadata"]["objects"]["primary_anchor_frame"] == 0
    assert out["metadata"]["objects"]["anchor_frame"] == 0
    assert len(out["metadata"]["objects"]["attempts"]) == 1


@pytest.mark.parametrize("retry_aborts", [False, True])
def test_empty_aborted_cup_native_zero_retries_mid_once_without_false_admission(monkeypatch, retry_aborts):
    f = _fixture(monkeypatch, anchor=3)
    anchors = []
    def text(frames, prompt, anchor):
        anchors.append(anchor)
        if len(anchors) == 1 or retry_aborts:
            f.sam.aborted = ["forward"]
            return {}
        f.sam.aborted = []
        return {99: {frame: np.packbits(f.cup[frame]) for frame in range(f.n)}}
    f.sam.track_text = text
    out = _run(f, True)
    assert anchors == [0, f.n // 2]
    attempts = out["metadata"]["objects"]["attempts"]
    assert [item["anchor_frame"] for item in attempts] == anchors
    assert attempts[0]["has_mask_proposal"] is False
    assert attempts[0]["propagation_aborted"] == ["forward"]
    failures = [item["code"] for item in out["metadata"]["quality_problems"] if item["severity"] == "QUALITY_FAILURE"]
    if retry_aborts:
        assert "CUP_TEXT_PROPAGATION_ABORTED" in failures
        assert "CUP_ENTIRE_SESSION_SUPPORT_MISSING" in failures
        assert not out["protected_visible"].any()
    else:
        assert attempts[1]["has_mask_proposal"] is True
        assert attempts[1]["propagation_aborted"] == []
        assert failures == []
        assert np.array_equal(out["protected_visible"], f.cup)


def test_empty_cup_without_abort_does_not_start_unrequested_retry(monkeypatch):
    f = _fixture(monkeypatch)
    anchors = []
    def text(frames, prompt, anchor):
        anchors.append(anchor)
        f.sam.aborted = []
        return {}
    f.sam.track_text = text
    out = _run(f)
    assert anchors == [0]
    assert any(item["code"] == "CUP_ENTIRE_SESSION_SUPPORT_MISSING"
               for item in out["metadata"]["quality_problems"])
