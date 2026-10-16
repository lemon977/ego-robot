import copy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from chaoyang.pick_place_1008 import audit, mask_seeds, sam_point_adapter as adapter, source


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "PROJECT_ROOT", tmp_path)
    rgb = np.zeros((4, 4, 8, 3), np.uint8)
    session = SimpleNamespace(calibration={"device_serial": "deviceA"},
                              qpc_ns=np.array([1000, 2000, 3000, 4000]))
    images = []
    for frame in range(4):
        rgb[frame] = [20 + frame, 70 + 2 * frame, 130 + 3 * frame]
        path = tmp_path / f"raw_{frame}.png"
        assert cv2.imwrite(str(path), rgb[frame, :, :, ::-1])
        images.append(source.ref(path))
    manifest = {"schema": "pick_place_calibration_source_images_v1", "status": "PASSED",
                "no_overlays": True, "qpc_modified": False, "lens_remap": False,
                "sessions": [{"session_id": "003", "device_serial": "deviceA", "images": [
                    {"session_id": "003", "device_serial": "deviceA", "raw_frame_index": f,
                     "qpc_ns": int(session.qpc_ns[f]), "image": images[f], "pixels_overlaid": False}
                    for f in range(4)]}]}
    objects = [{"obj_id": i + 1, "side": side, "role": role, "visible_confirmed": True,
                "frame_idx": 0, "points_px": [[1, 1], [0, 0]], "point_labels": [1, 0]}
               for i, (side, role) in enumerate([("left", "glove"), ("left", "controller"),
                                                ("right", "glove"), ("right", "controller")])]
    correction = {"session_id": "003", "device_serial": "deviceA", "frame_idx": 3,
                  "raw_frame_index": 3, "qpc_ns": 4000, "image": images[3],
                  "points_px": [[6, 2], [0, 0]], "point_labels": [1, 0],
                  "visible_confirmed": True, "seed_semantics": mask_seeds.CORRECTION_SEMANTICS}
    objects[0]["correction_points"] = [correction]
    record = {"schema": "pick_place_visible_mask_seeds_v3", "session_id": "003",
              "device_serial": "deviceA", "raw_frame_index": 0, "qpc_ns": 1000,
              "image": images[0], "visibility_review_status": "AI_CONFIRMED_VISIBLE_PIXELS",
              "physical_ground_truth": False, "training_eligible": False, "objects": objects}
    manifest_path, leaf, index = [tmp_path / name for name in ("SOURCE_IMAGES.json", "seed.json", "INDEX.json")]
    def save():
        manifest_path.write_text(json.dumps(manifest))
        record["source_images"] = source.ref(manifest_path)
        leaf.write_text(json.dumps(record))
        index.write_text(json.dumps({"schema": "pick_place_mask_seed_index_v1", "records": [
            {"session_id": "003", "seed": source.ref(leaf)}]}))
    save()
    return SimpleNamespace(rgb=rgb, session=session, images=images, manifest=manifest,
                           record=record, correction=correction, index=index, leaf=leaf,
                           manifest_path=manifest_path, save=save)


class RefinementModel:
    """Official-shaped same-ID conditioning with owned native frame cache."""
    def __init__(self, *, unsupported_anchor=False, missing_correction=False, wrong_frame=False):
        self.calls, self.events, self.cache_calls = [], [], []
        self.points = {}
        self.unsupported_anchor = unsupported_anchor
        self.missing_correction, self.wrong_frame = missing_correction, wrong_frame
        self.cleared = False

    def init_state(self, **kwargs):
        model = self
        class State(dict):
            def clear(self):
                model.cleared = True
                super().clear()
        self.state = State()
        self.count = len(kwargs["resource_path"])
        self.width, self.height = kwargs["resource_path"][0].size
        self.events.append("init")
        return self.state

    def _cache_frame_outputs(self, state, frame_idx, obj_id_to_mask):
        assert state is self.state and obj_id_to_mask == {}
        state.setdefault("cached_frame_outputs", {})[frame_idx] = {}
        self.cache_calls.append(frame_idx)

    def output(self, *, omit=None):
        ids = sorted(self.points)
        masks = np.zeros((len(ids), self.height, self.width), bool)
        for index, obj in enumerate(ids):
            if obj != omit:
                x, y = self.points[obj]
                masks[index, y, x] = True
        return {"out_obj_ids": np.asarray(ids, np.int64), "out_binary_masks": masks}

    def add_prompt(self, **kwargs):
        obj, frame = kwargs["obj_id"], kwargs["frame_idx"]
        self.events.append(("prompt", obj, frame))
        self.calls.append(kwargs)
        assert kwargs["inference_state"] is self.state and kwargs["text_str"] is None
        assert kwargs["clear_old_points"] is True and kwargs["rel_coordinates"] is True
        point = kwargs["points"][0].numpy() * [self.width, self.height]
        self.points[obj] = tuple(np.rint(point).astype(int))
        omit = obj if ((self.unsupported_anchor and frame == 0 and obj == 1)
                       or (self.missing_correction and frame == 3 and obj == 1)) else None
        result_frame = 2 if self.wrong_frame and frame == 3 else frame
        return result_frame, self.output(omit=omit)

    def propagate_in_video(self, **kwargs):
        self.events.append(("propagate", kwargs["reverse"]))
        anchor = kwargs["start_frame_idx"]
        reverse = kwargs["reverse"]
        maximum = kwargs["max_frame_num_to_track"]
        end = max(anchor - maximum, 0) if reverse else min(anchor + maximum, self.count - 1)
        order = range(anchor - 1, end - 1, -1) if reverse else range(anchor, end + 1)
        assert kwargs["max_frame_num_to_track"] == len(order)
        for frame in order:
            yield frame, self.output()


def test_loader_native_evidence_reaches_same_id_official_refinement_and_all_frames(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    original_record, original_rgb = copy.deepcopy(f.record), f.rgb.copy()
    loaded = mask_seeds.load_visible_seed(f.index, "003", f.session, f.rgb)
    correction = loaded["objects"][0]["correction_points"][0]
    assert correction["native_evidence_validation"] == adapter.CORRECTION_VALIDATION
    model = RefinementModel()
    tracks, metadata = adapter.track_visible_points(SimpleNamespace(model=model), f.rgb,
                                                     loaded["objects"], loaded["anchor_frame"])
    assert metadata["status"] == "COMPLETE"
    assert [(call["obj_id"], call["frame_idx"]) for call in model.calls] == [(1,0),(2,0),(3,0),(4,0),(1,3)]
    assert np.allclose(model.calls[-1]["points"].numpy(), [[6/8,2/4],[0,0]])
    assert model.calls[-1]["point_labels"].tolist() == [1,0]
    assert model.cache_calls == [0,1,2,3]
    assert metadata["objects"]["1"]["supported_correction_raw_frame_indices"] == [3]
    assert metadata["correction_count"] == 1
    assert all(set(values) == {0,1,2,3} for values in tracks.values())
    assert np.unpackbits(tracks[1][3],count=32).reshape(4,8)[2,6]
    assert model.cleared and f.record == original_record and np.array_equal(f.rgb, original_rgb)


@pytest.mark.parametrize("mutation,error", [
    ("qpc","QPC"), ("missing_qpc","QPC"), ("wrong_frame","QPC"),
    ("frame_aliases","FRAME_ALIASES"), ("wrong_device","IDENTITY"),
    ("wrong_session","IDENTITY"), ("wrong_role","IDENTITY"),
    ("not_visible","IDENTITY"), ("physical_truth","IDENTITY"),
    ("missing_image","IMAGE_REFERENCE"), ("other_native_png","RGB_PIXEL"),
    ("own_rgb_changed","RGB_PIXEL"), ("png_bytes_drift","REFERENCE_DRIFT"),
    ("source_manifest_drift","REFERENCE_DRIFT"), ("unsealed_manifest","SOURCE_IMAGES_DOMAIN"),
    ("manifest_other_session","SOURCE_SESSION_BINDING"),
    ("not_manifest_member","ROW_MEMBERSHIP"), ("duplicate_manifest_row","ROW_MEMBERSHIP"),
    ("negative_only","POINT_DOMAIN"), ("float_labels","POINT_DOMAIN"),
    ("outside","POINT_DOMAIN"), ("nan","POINT_DOMAIN"),
    ("duplicate_frame","DISTINCT_NATIVE"), ("anchor_frame","DISTINCT_NATIVE"),
    ("too_many_corrections","CORRECTION_COUNT"), ("legacy_v2","V3_REQUIRED"),
])
def test_correction_cannot_bypass_native_image_clock_identity_or_visibility(tmp_path, monkeypatch, mutation, error):
    f = fixture(tmp_path, monkeypatch)
    c = f.correction
    if mutation == "qpc": c["qpc_ns"] = 4001
    elif mutation == "missing_qpc": c.pop("qpc_ns")
    elif mutation == "wrong_frame": c["frame_idx"] = c["raw_frame_index"] = 4
    elif mutation == "frame_aliases": c["frame_idx"] = 2
    elif mutation == "wrong_device": c["device_serial"] = "other"
    elif mutation == "wrong_session": c["session_id"] = "100"
    elif mutation == "wrong_role": c["role"] = "controller"
    elif mutation == "not_visible": c["visible_confirmed"] = False
    elif mutation == "physical_truth": c["physical_ground_truth"] = True
    elif mutation == "missing_image": c.pop("image")
    elif mutation == "other_native_png": c["image"] = f.images[0]
    elif mutation == "own_rgb_changed": f.rgb[3,1,1,0] += 1
    elif mutation == "png_bytes_drift":
        path = Path(c["image"]["path"]); path.write_bytes(path.read_bytes()+b"drift")
    elif mutation == "unsealed_manifest": f.manifest["status"] = "RUNNING"
    elif mutation == "manifest_other_session": f.manifest["sessions"][0]["images"][3]["session_id"] = "100"
    elif mutation == "not_manifest_member": f.manifest["sessions"][0]["images"].pop()
    elif mutation == "duplicate_manifest_row": f.manifest["sessions"][0]["images"].append(copy.deepcopy(f.manifest["sessions"][0]["images"][3]))
    elif mutation == "negative_only": c["point_labels"] = [0,0]
    elif mutation == "float_labels": c["point_labels"] = [1.,0.]
    elif mutation == "outside": c["points_px"][0][0] = 8
    elif mutation == "nan": c["points_px"][0][1] = float("nan")
    elif mutation == "duplicate_frame": f.record["objects"][0]["correction_points"].append(copy.deepcopy(c))
    elif mutation == "anchor_frame": c.update(frame_idx=0,raw_frame_index=0)
    elif mutation == "too_many_corrections": f.record["objects"][0]["correction_points"] = [copy.deepcopy(c)]*17
    elif mutation == "legacy_v2": f.record["schema"] = "pick_place_visible_mask_seeds_v2"
    f.save()
    if mutation == "source_manifest_drift": f.manifest_path.write_text(f.manifest_path.read_text()+" ")
    with pytest.raises(Exception, match=error):
        mask_seeds.load_visible_seed(f.index,"003",f.session,f.rgb)


def test_same_pixels_from_other_session_png_still_require_exact_manifest_row(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    duplicate = tmp_path/"100_raw_0003.png"
    duplicate.write_bytes(Path(f.images[3]["path"]).read_bytes())
    f.correction["image"] = source.ref(duplicate)
    f.save()
    with pytest.raises(ValueError,match="ROW_MEMBERSHIP"):
        mask_seeds.load_visible_seed(f.index,"003",f.session,f.rgb)


@pytest.mark.parametrize("failure,message", [("missing","SUPPORT_REQUIRED"),("frame","FRAME_BINDING")])
def test_each_correction_must_return_same_target_id_nonempty_at_its_native_frame(tmp_path, monkeypatch, failure, message):
    f = fixture(tmp_path, monkeypatch)
    loaded = mask_seeds.load_visible_seed(f.index,"003",f.session,f.rgb)
    model = RefinementModel(missing_correction=failure=="missing",wrong_frame=failure=="frame")
    _, meta = adapter.track_visible_points(SimpleNamespace(model=model), f.rgb,loaded["objects"],0)
    assert meta["status"] == "TRACKING_ERROR"
    assert message in meta["errors"][0]["message"]
    assert meta["errors"][0]["phase"] == "add_correction:1:3"
    assert not any(event[0]=="propagate" for event in model.events if isinstance(event,tuple))
    assert model.cleared


def test_later_manual_correction_does_not_silently_replace_failed_original_anchor(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch); loaded=mask_seeds.load_visible_seed(f.index,"003",f.session,f.rgb)
    model=RefinementModel(unsupported_anchor=True)
    tracks,meta=adapter.track_visible_points(SimpleNamespace(model=model),f.rgb,loaded["objects"],0)
    assert meta["status"]=="PARTIAL_SUPPORT" and 1 not in tracks
    assert not meta["objects"]["1"]["anchor_mask_present"]
    assert meta["objects"]["1"]["supported_correction_raw_frame_indices"]==[3]


def test_correction_before_own_anchor_drives_minimum_native_propagation(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    obj=f.record["objects"][0]
    obj.update(frame_idx=3,raw_frame_index=3,qpc_ns=4000,image=f.images[3])
    f.correction.update(frame_idx=1,raw_frame_index=1,qpc_ns=2000,image=f.images[1])
    for other in f.record["objects"][1:]:other.update(frame_idx=2,qpc_ns=3000,image=f.images[2])
    f.save();loaded=mask_seeds.load_visible_seed(f.index,"003",f.session,f.rgb)
    assert loaded["anchor_frame"]==1
    model=RefinementModel();tracks,meta=adapter.track_visible_points(SimpleNamespace(model=model),f.rgb,loaded["objects"],2)
    assert meta["status"]=="COMPLETE" and meta["anchor_raw_frame_index"]==1
    assert [item["yielded_raw_frame_indices"] for item in meta["propagation"]]==[[1,2,3],[0]]
    assert all(set(values)=={0,1,2,3} for values in tracks.values())


def test_adapter_rejects_unverified_correction_before_model_state_allocation(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    model=RefinementModel()
    with pytest.raises(adapter.SamPointContractError,match="VERIFIED_NATIVE"):
        adapter.track_visible_points(SimpleNamespace(model=model),f.rgb,f.record["objects"],0)
    assert model.events==[]


def test_explicit_returned_empty_retracts_prior_corrected_mask_and_fails_final_support(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch); loaded=mask_seeds.load_visible_seed(f.index,"003",f.session,f.rgb)
    class Retracting(RefinementModel):
        def propagate_in_video(self,**kwargs):
            for frame,output in super().propagate_in_video(**kwargs):
                if frame==3:
                    index=output["out_obj_ids"].tolist().index(1)
                    output["out_binary_masks"][index]=False
                yield frame,output
    model=Retracting();tracks,meta=adapter.track_visible_points(SimpleNamespace(model=model),f.rgb,loaded["objects"],0)
    assert meta["status"]=="TRACKING_ERROR" and 3 not in tracks.get(1,{})
    assert meta["objects"]["1"]["initial_supported_correction_raw_frame_indices"]==[3]
    assert meta["objects"]["1"]["supported_correction_raw_frame_indices"]==[]
    assert meta["lost_required_native_support"]==[{"obj_id":1,"raw_frame_index":3,"kind":"CORRECTION"}]
    assert meta["explicit_empty_frame_retractions"][-1]["raw_frame_index"]==3
    assert meta["errors"][-1]["phase"]=="required_support_after_propagation" and model.cleared


def test_later_other_id_prompt_cannot_leave_old_empty_anchor_as_supported(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch); loaded=mask_seeds.load_visible_seed(f.index,"003",f.session,f.rgb)
    class RetractingAnchor(RefinementModel):
        def add_prompt(self,**kwargs):
            frame,output=super().add_prompt(**kwargs)
            if frame==0 and len(self.points)>1:
                output["out_binary_masks"][output["out_obj_ids"].tolist().index(1)]=False
            return frame,output
    model=RetractingAnchor();tracks,meta=adapter.track_visible_points(SimpleNamespace(model=model),f.rgb,loaded["objects"],0)
    assert meta["status"]=="TRACKING_ERROR" and 1 not in tracks
    assert meta["objects"]["1"]["initial_anchor_mask_present"] is True
    assert meta["objects"]["1"]["anchor_mask_present"] is False
    assert meta["errors"][-1]["phase"]=="required_support_after_prompts"
    assert not any(event[0]=="propagate" for event in model.events if isinstance(event,tuple))
    assert model.cleared


def test_absent_id_in_partial_output_does_not_retract_native_conditioned_mask(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch); loaded=mask_seeds.load_visible_seed(f.index,"003",f.session,f.rgb)
    class Partial(RefinementModel):
        def propagate_in_video(self,**kwargs):
            for frame,output in super().propagate_in_video(**kwargs):
                if frame==3:
                    keep=output["out_obj_ids"]!=1
                    output={"out_obj_ids":output["out_obj_ids"][keep],"out_binary_masks":output["out_binary_masks"][keep]}
                yield frame,output
    model=Partial();tracks,meta=adapter.track_visible_points(SimpleNamespace(model=model),f.rgb,loaded["objects"],0)
    assert meta["status"]=="COMPLETE" and 3 in tracks[1]
    assert meta["objects"]["1"]["supported_correction_raw_frame_indices"]==[3]
    assert meta["explicit_empty_frame_retractions"]==[]
