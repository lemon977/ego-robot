"""Synthetic Composite receipts/pixels only; no real001 or model execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.inpainting_candidate import composite as c
from chaoyang.inpainting_candidate.contracts import SCHEMA, canonical_sha, file_ref, verify_sealed_stage


def json_ref(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return file_ref(path)


@pytest.fixture
def case(tmp_path, monkeypatch):
    assert tmp_path.is_relative_to(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(c, "PROJECT_ROOT", tmp_path)
    for relative in c.CODE_CLOSURE:
        p = tmp_path / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# synthetic code\n")
    raw_video = tmp_path / "raw/vst.h264"
    raw_video.parent.mkdir()
    raw_video.write_bytes(b"synthetic raw video, not decoded")
    video_ref = file_ref(raw_video)
    count, shape = 2, (2, 2, 8)
    raw = np.full((*shape, 3), [9, 19, 29], np.uint8)
    mapping = [{"native_frame_index": i, "decoded_display_index": i, "model_input_frame": i,
        "output_frame": i, "qpc_declared_ns": 1_000_000_000 + i * 40_000_000,
        "physical_eye": "right", "physical_eye_verified": False, "sbs_index": 1,
        "acquisition_frame_id": "UNKNOWN", "timebase_status": c.TIMEBASE,
        "pixel_sha256": hashlib.sha256(raw[i].data).hexdigest()} for i in range(count)]
    source_dir = tmp_path / "Source"
    source_dir.mkdir()
    np.savez_compressed(source_dir / "RAW_RGB.npz", rgb=raw)
    source = {"schema": SCHEMA, "stage": "Source", "status": "SEALED_COMPLETE",
        "component_signature": "a" * 64, "native_frames": count, "session_id": "synthetic001",
        "outputs": [file_ref(source_dir / "RAW_RGB.npz"),
            json_ref(source_dir / "FRAME_MAP.json", mapping), json_ref(source_dir / "SOURCE_REPORT.json",
                {"actual_inputs": {"video": video_ref}})]}
    source_ref = json_ref(source_dir / "RECEIPT.json", source)
    masks = {key: np.zeros(shape, bool) for key in
             ("effective_removal", "protected_visible", "object_observed", "unknown", "human", "equipment")}
    masks["protected_visible"][:, 0, [2, 4, 5]] = True
    masks["object_observed"][:, 0, [2, 4, 5, 6]] = True
    masks["human"][:, 0, 4] = True
    masks["equipment"][:, 0, 5] = True
    masks["effective_removal"][:, 0, 6] = True
    masks["effective_removal"][:, 1, 0] = True
    clean = {"clean_rgb": raw.copy(), "effective_removal": masks["effective_removal"].copy(),
             "protected_visible": masks["protected_visible"].copy()}
    clean["clean_rgb"][:, 1, 0] = [70, 80, 90]
    arrays = {"rgba": np.zeros((*shape, 4), np.uint8),
        "depth_m": np.full(shape, np.inf, np.float32), "link_id": np.full(shape, -1, np.int32),
        "qpc_ns": np.asarray([r["qpc_declared_ns"] for r in mapping], np.int64)}
    arrays["rgba"][:, 0, :7] = [100, 110, 120, 255]
    arrays["depth_m"][:, 0, :7] = .75
    arrays["link_id"][:, 0, :7] = [0, 2000, 2001, 4000, 2001, 2001, 2001]
    ids = {"0": {"instance_id": "left_tianji", "part_id": "arm", "link_id": "Link1_L"},
        "2000": {"instance_id": "left_kaihand", "part_id": "palm", "link_id": "hand_l_base_link"},
        "2001": {"instance_id": "left_kaihand", "part_id": "finger", "link_id": "index_link"},
        "4000": {"instance_id": "left_flange", "part_id": "flange", "link_id": "flange_left"}}
    refs, stage_values = {}, {}
    frame_states = {name: [{"native_frame_index": i, "visual_quality": "NOT_EVALUATED",
        "unknown_pixels": 3, "known_issue": "HIGH" if i == 0 else "UNKNOWN"} for i in range(count)]
        for name in ("Mask", "Clean", "Render")}

    def save_stage(name):
        stage_dir = tmp_path / name
        stage_dir.mkdir(exist_ok=True)
        key, payload = {"Mask": ("MASK_ARRAYS.npz", masks), "Clean": ("CLEAN_ARRAYS.npz", clean),
                        "Render": ("ROBOT_BUFFERS.npz", arrays)}[name]
        np.savez_compressed(stage_dir / key, **payload)
        stage = stage_values.get(name, {"schema": SCHEMA, "stage": name, "status": "SEALED_COMPLETE",
            "component_signature": {"Mask": "b", "Clean": "d", "Render": "e"}[name] * 64,
            "visual_quality": "NOT_EVALUATED", "human_review": "PENDING"})
        stage["outputs"] = [file_ref(stage_dir / key), json_ref(stage_dir / "FRAME_MAP.json", mapping),
            json_ref(stage_dir / "FRAME_STATUS.json", frame_states[name])]
        if name in {"Mask", "Clean"}:
            stage.update(native_input_frames=count, native_output_frames=count)
            stage["actual_inputs"] = {"raw_source": video_ref,
                "frames_sha256": c._rgb_sha(raw), "frame_map_sha256": canonical_sha(mapping)}
            if name == "Clean":
                stage["actual_inputs"]["mask_receipt"] = refs["Mask"]
        else:
            stage.update(native_frames=count, source_receipt=source_ref, source_signature="a" * 64,
                session_id=source["session_id"], complete_robot=True)
            stage["outputs"].append(json_ref(stage_dir / "LABEL_IDENTITY.json", ids))
        stage_values[name] = stage
        refs[name] = json_ref(stage_dir / "RECEIPT.json", stage)
        return refs[name]

    for name in ("Mask", "Clean", "Render"):
        save_stage(name)
    review_ref_path = tmp_path / "review_evidence_only.txt"
    # Not JSON: the producer must only SHA-check this ref, never load annotations.
    review_ref_path.write_bytes(b"actually reviewed evidence, contains no production pixels")
    quality = {}
    for name in ("Mask", "Clean"):
        quality[name] = json_ref(tmp_path / (name + "_QUALITY.json"), {
            "schema": "inpainting-upstream-quality-status-v1", "stage": name,
            "upstream_receipt": refs[name], "visual_quality": "FAILED",
            "review_reference": file_ref(review_ref_path), "human_review": "PENDING", "hidden_regions": "UNKNOWN"})
    calls, state = [], {"decode_count": count, "gate_calls": 0, "revoke_at": None, "identity_at": None}

    def authority():
        state["gate_calls"] += 1
        return {"execution_allowed": state["gate_calls"] != state["revoke_at"],
            "algorithm_execution_allowed": True, "task_id": "synthetic_registered_composite",
            "run_signature": "f" * 64,
            "fencing_token": "drift" if state["gate_calls"] == state["identity_at"] else "fence"}

    def writer(path, frames, given_map):
        assert given_map == mapping
        calls.append((path.name, frames.copy()))
        path.write_bytes(b"synthetic video, no real codec")
        return {"video": file_ref(path), "decoded_frames": state["decode_count"]}

    monkeypatch.setattr(c, "_write_review_video", writer)
    args = dict(source_receipt=source_ref, source_signature="a" * 64,
        mask_receipt=refs["Mask"], mask_signature="b" * 64,
        clean_receipt=refs["Clean"], clean_signature="d" * 64,
        render_receipt=refs["Render"], render_signature="e" * 64,
        output_dir=tmp_path / "Composite", component_signature="c" * 64,
        repo_root=tmp_path, authorization=authority, upstream_quality_evidence=quality)
    return SimpleNamespace(args=args, raw=raw, mapping=mapping, masks=masks, clean=clean, arrays=arrays,
        ids=ids, refs=refs, stage_values=stage_values, save_stage=save_stage, calls=calls,
        state=state, frame_states=frame_states, video_ref=video_ref)


def test_native_composite_unknown_geometry_model_display_and_rgb(case):
    receipt = c.build_composite_stage(**case.args)
    assert receipt["stage"] == "Composite" and receipt["native_frames"] == 2
    assert receipt["mask_quality"] == receipt["clean_quality"] == "FAILED"
    assert receipt["qualified_trusted_occlusion_pixels"] == 0
    assert receipt["visual_quality"] == "NOT_REVIEWED" and receipt["human_review"] == "PENDING"
    assert receipt["control_ground_truth"] is False and receipt["training_eligible"] is False
    verify_sealed_stage(receipt, "c" * 64, output_root=case.args["output_dir"])
    with np.load(case.args["output_dir"] / "COMPOSITE_ARRAYS.npz", allow_pickle=False) as arrays:
        assert arrays["final_rgb"].shape == (2, 2, 8, 3)
        assert np.all(arrays["geometry_decision"] == 0)
        assert arrays["display_policy"][0, 0, 2] == 3
        assert arrays["display_policy"][0, 0, 4:7].tolist() == [4, 4, 4]
        assert arrays["final_rgb"][0, 0, 2].tolist() == [9, 19, 29]  # raw object support
        assert arrays["final_rgb"][0, 0, 4].tolist() == [100, 110, 120]  # same finger remains drawn
        assert arrays["final_rgb"][0, 1, 0].tolist() == [70, 80, 90]  # existing Clean, not redilate
        assert arrays["source_link_labels"][0, 0, 3] == 4000
        assert arrays["dense_robot_owner"][0, 0, 3] == 3
        assert arrays["dense_robot_owner"][0, 1, 0] == -1
    report = json.loads((case.args["output_dir"] / "COMPOSITE_REPORT.json").read_text())
    assert report["clean_score_improved_by_robot_coverage"] is False
    assert report["object_first_fallback_pixels"] == 2
    assert report["object_support_qualification"] == c.SUPPORT_QUALIFICATION
    assert all(x["qualified_trusted_occlusion_pixels"] == 0 for x in report["link_statistics"].values())
    states = json.loads((case.args["output_dir"] / "FRAME_STATUS.json").read_text())
    assert states[0]["upstream_status"]["Mask"]["known_issue"] == "HIGH"
    assert states[1]["upstream_status"]["Mask"]["known_issue"] == "UNKNOWN"
    assert states[0]["component_clean_quality"] == "FAILED"
    assert states[0]["hidden_regions"] == "UNKNOWN"
    assert [x[0] for x in case.calls] == ["FINAL_CANDIDATE.mp4", "RAW_MASK_CLEAN_ROBOT_REVIEW.mp4"]
    assert case.calls[1][1].shape == (2, 2, 32, 3)


def test_uint8_alpha_div255_not_bool_or_bgr(case):
    case.arrays["rgba"][:, 0, 0, 3] = 128
    case.args["render_receipt"] = case.save_stage("Render")
    c.build_composite_stage(**case.args)
    with np.load(case.args["output_dir"] / "COMPOSITE_ARRAYS.npz") as arrays:
        expected = np.rint(case.raw[0, 0, 0] * (1 - 128 / 255) + np.asarray([100, 110, 120]) * (128 / 255)).astype(np.uint8)
        assert np.array_equal(arrays["final_rgb"][0, 0, 0], expected)


def test_unknown_object_support_is_not_foreground(case):
    case.masks["unknown"][:, 0, 2] = True
    case.args["mask_receipt"] = case.save_stage("Mask")
    case.args["clean_receipt"] = case.save_stage("Clean")
    case.args["upstream_quality_evidence"] = None
    c.build_composite_stage(**case.args)
    with np.load(case.args["output_dir"] / "COMPOSITE_ARRAYS.npz") as arrays:
        assert not arrays["model_predicted_object_support"].any()
        assert arrays["display_policy"][0, 0, 2] == 4
        assert arrays["final_rgb"][0, 0, 2].tolist() == [100, 110, 120]


def test_no_quality_evidence_does_not_invent_failed_or_pass(case):
    case.args["upstream_quality_evidence"] = None
    receipt = c.build_composite_stage(**case.args)
    assert receipt["clean_quality"] == receipt["mask_quality"] == "NOT_EVALUATED"


@pytest.mark.parametrize("field", ["source_signature", "mask_signature", "clean_signature", "render_signature", "component_signature"])
def test_bad_signature_no_outputs(case, field):
    case.args[field] = "wrong"
    with pytest.raises(c.CompositeStageError, match="SIGNATURE_REQUIRED"):
        c.build_composite_stage(**case.args)
    assert not case.args["output_dir"].exists()


@pytest.mark.parametrize("at", [1, 2, 3, 4])
def test_gate_revocation_never_seals(case, at):
    case.state["revoke_at"] = at
    with pytest.raises(c.CompositeStageError, match="NOT_AUTHORIZED"):
        c.build_composite_stage(**case.args)
    assert not (case.args["output_dir"] / "RECEIPT.json").exists()


def test_writer_identity_drift_never_seals(case):
    case.state["identity_at"] = 2
    with pytest.raises(c.CompositeStageError, match="WRITER_IDENTITY_CHANGED"):
        c.build_composite_stage(**case.args)


def test_no_packet_authority_rejected(case):
    case.args["authorization"] = None
    with pytest.raises(c.CompositeStageError, match="AUTHORIZATION_REQUIRED"):
        c.build_composite_stage(**case.args)


@pytest.mark.parametrize("key", ["effective_removal", "protected_visible"])
def test_resigned_clean_EP_tamper_not_redilated_or_accepted(case, key):
    case.clean[key][0, 1, 7] = True
    case.args["clean_receipt"] = case.save_stage("Clean")
    with pytest.raises(c.CompositeStageError, match="ACTUAL_CLEAN_EP"):
        c.build_composite_stage(**case.args)


def test_clean_outside_E_pollution_rejected(case):
    case.clean["clean_rgb"][0, 1, 7] = [1, 2, 3]
    case.args["clean_receipt"] = case.save_stage("Clean")
    case.args["upstream_quality_evidence"] = None
    with pytest.raises(ValueError, match="ZERO_CHANGE_INVARIANT"):
        c.build_composite_stage(**case.args)


@pytest.mark.parametrize("stage", ["Mask", "Clean", "Render"])
def test_partial_not_consumable(case, stage):
    case.stage_values[stage]["status"] = "PARTIAL"
    case.args[stage.lower() + "_receipt"] = case.save_stage(stage)
    with pytest.raises(ValueError, match="PARTIAL_NOT_CONSUMABLE"):
        c.build_composite_stage(**case.args)


def test_resigned_native_map_drift_rejected(case):
    stage = case.stage_values["Render"]
    frame_ref = next(x for x in stage["outputs"] if Path(x["path"]).name == "FRAME_MAP.json")
    other = deepcopy(case.mapping)
    other[0]["physical_eye"] = "left"
    stage["outputs"] = [json_ref(Path(x["path"]), other) if x is frame_ref else x for x in stage["outputs"]]
    case.args["render_receipt"] = json_ref(Path(case.refs["Render"]["path"]), stage)
    with pytest.raises(c.CompositeStageError, match="NATIVE_MAP_CHANGED"):
        c.build_composite_stage(**case.args)


@pytest.mark.parametrize("error", ["depth", "background", "alpha", "label", "qpc"])
def test_render_actual_buffers_negative(case, error):
    if error == "depth": case.arrays["depth_m"][0, 0, 0] = np.inf
    elif error == "background": case.arrays["depth_m"][0, 1, 0] = .5
    elif error == "alpha": case.arrays["rgba"][0, 1, 0, 3] = 255
    elif error == "label": case.arrays["link_id"][0, 0, 0] = 12345
    else: case.arrays["qpc_ns"][1] += 1
    case.args["render_receipt"] = case.save_stage("Render")
    with pytest.raises(c.CompositeStageError):
        c.build_composite_stage(**case.args)


def test_model_raw_source_drift_is_not_other_session(case):
    case.stage_values["Mask"]["actual_inputs"]["raw_source"] = {"path": "other"}
    case.args["mask_receipt"] = json_ref(Path(case.refs["Mask"]["path"]), case.stage_values["Mask"])
    with pytest.raises(c.CompositeStageError, match="RAW_PIXEL_OR_TIME_BINDING"):
        c.build_composite_stage(**case.args)


def test_raw_frame_pixel_sha_checked(case):
    case.mapping[0]["pixel_sha256"] = "0" * 64
    root = Path(case.args["source_receipt"]["path"]).parent
    source = json.loads((root / "RECEIPT.json").read_text())
    source["outputs"] = [json_ref(root / "FRAME_MAP.json", case.mapping) if Path(x["path"]).name == "FRAME_MAP.json" else x for x in source["outputs"]]
    case.args["source_receipt"] = json_ref(root / "RECEIPT.json", source)
    case.stage_values["Render"]["source_receipt"] = case.args["source_receipt"]
    # save_stage reconstructs source from original ref; directly reseal this row.
    case.args["render_receipt"] = json_ref(Path(case.refs["Render"]["path"]), case.stage_values["Render"])
    with pytest.raises(c.CompositeStageError, match="FRAME_OR_PIXEL_SHA"):
        c.build_composite_stage(**case.args)


@pytest.mark.parametrize("field,value", [("visual_quality", "PASSED"), ("hidden_regions", "RESTORED"), ("stage", "Render")])
def test_quality_status_cannot_change_qualification(case, field, value):
    ref = case.args["upstream_quality_evidence"]["Clean"]
    status = json.loads(Path(ref["path"]).read_text())
    status[field] = value
    case.args["upstream_quality_evidence"]["Clean"] = json_ref(Path(ref["path"]), status)
    with pytest.raises(c.CompositeStageError, match="QUALITY_EVIDENCE_BINDING"):
        c.build_composite_stage(**case.args)


def test_actual_video_decode_count_checked(case):
    case.state["decode_count"] = 1
    with pytest.raises(c.CompositeStageError, match="FULL_NATIVE_VIDEO_DECODE"):
        c.build_composite_stage(**case.args)
    assert not (case.args["output_dir"] / "RECEIPT.json").exists()


def test_fresh_output_required(case):
    case.args["output_dir"].mkdir()
    with pytest.raises(c.CompositeStageError, match="FRESH_OUTPUT"):
        c.build_composite_stage(**case.args)


def test_dense_sparse_mapping_does_not_use_sparse_id_as_index():
    labels = np.asarray([[[4000, 2001, -1]]], np.int32)
    identities = {"2001": {"instance_id": "hand", "part_id": "finger", "link_id": "index"},
                  "4000": {"instance_id": "flange", "part_id": "flange", "link_id": "left"}}
    owner, entries, names = c._dense_owner(labels, identities)
    assert owner.tolist() == [[[1, 0, -1]]]
    assert [row["source_link_label"] for row in entries] == [2001, 4000]
    assert len(names) == 2
