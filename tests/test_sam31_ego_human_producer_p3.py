from __future__ import annotations

import inspect
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from pipeline.sam31_ego_human_producer_p3 import (
    ProducerP3ContractError,
    SamInstances,
    derive_two_limb_boxes,
    select_frame_candidate,
)
from pipeline.sam31_geometric_box_adapter_v1 import (
    Sam31GeometricBoxAdapterV1,
    Sam31GeometricBoxContractError,
)
from tools.run_sam31_ego_human_producer_p3 import (
    prepare_owned_run,
    prepare_session_tree,
    require_ordinary,
    verify_qa_gates,
)


HEIGHT = 100
WIDTH = 100


def _joints():
    left = np.zeros((21, 2), dtype=np.float64)
    right = np.zeros((21, 2), dtype=np.float64)
    left[0] = [10, 50]
    right[0] = [90, 70]
    for index in range(1, 21):
        left[index] = [14 + index, 42 + (index % 5) * 4]
        right[index] = [86 - index, 62 + (index % 5) * 4]
    return np.stack([left, right])


def _rectangle(x0, y0, x1, y1):
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _instances(masks, scores=None, ids=None):
    array = (
        np.stack(masks).astype(bool)
        if masks
        else np.zeros((0, HEIGHT, WIDTH), dtype=bool)
    )
    count = len(array)
    return SamInstances(
        masks=array,
        scores=np.asarray(scores if scores is not None else [0.9] * count),
        instance_ids=np.asarray(ids if ids is not None else list(range(count))),
    )


def _good_inputs(include_background=True):
    left = _rectangle(0, 38, 39, 65)
    right = _rectangle(61, 58, 100, 87)
    background = _rectangle(43, 4, 58, 70)
    masks = [left, right] + ([background] if include_background else [])
    box_instances = _instances(masks, ids=[10, 20, 99][: len(masks)])
    object_mask = _rectangle(44, 20, 58, 40)
    objects = _instances([object_mask], ids=[50])
    prompts = derive_two_limb_boxes(_joints(), WIDTH, HEIGHT)
    return box_instances, objects, prompts, object_mask


def test_box_formula_uses_authoritative_sides_and_image_boundaries():
    result = derive_two_limb_boxes(_joints(), WIDTH, HEIGHT)
    left, right = result.boxes_xywh
    assert left[0] == pytest.approx(0.0)
    assert right[0] + right[2] == pytest.approx(1.0)
    assert result.box_labels.tolist() == [1, 1]
    assert result.diagnostics["geometry_created_human_pixels"] == 0
    assert result.diagnostics["hard_crop_applied"] is False


def test_box_formula_is_resolution_normalized():
    first = derive_two_limb_boxes(_joints(), WIDTH, HEIGHT).boxes_xywh
    second = derive_two_limb_boxes(_joints() * 2, WIDTH * 2, HEIGHT * 2).boxes_xywh
    np.testing.assert_allclose(first, second, atol=0.011)


def test_complete_candidate_selects_two_ego_instances_and_rejects_background():
    box_instances, objects, prompts, cad = _good_inputs()
    result = select_frame_candidate(
        box_prompt_instances=box_instances,
        object_prompt_instances=objects,
        prompt_boxes=prompts,
        joints_by_side=_joints(),
        cad_object_mask=cad,
    )
    assert result.sufficient
    assert result.selected_ids == {"left": 10, "right": 20}
    background = next(
        item
        for item in result.diagnostics["box_instance_diagnostics"]
        if item["instance_id"] == 99
    )
    assert background["eligible_sides"] == []
    assert not np.logical_and(result.final_mask, box_instances.masks[2]).any()


def test_prompt_box_is_not_a_hard_crop_or_pixel_source():
    box_instances, objects, prompts, cad = _good_inputs(include_background=False)
    left = box_instances.masks[0].copy()
    left[96, 50] = True
    changed = SamInstances(
        masks=np.stack([left, box_instances.masks[1]]),
        scores=box_instances.scores,
        instance_ids=box_instances.instance_ids,
    )
    result = select_frame_candidate(
        box_prompt_instances=changed,
        object_prompt_instances=objects,
        prompt_boxes=prompts,
        joints_by_side=_joints(),
        cad_object_mask=cad,
    )
    assert result.final_mask[96, 50]
    assert result.diagnostics["pixel_lineage"]["hard_crop_applied"] is False
    assert result.diagnostics["pixel_lineage"]["prompt_box_created_human_pixels"] == 0


def test_empty_sam_output_holds_and_geometry_cannot_fill():
    _, objects, prompts, cad = _good_inputs()
    result = select_frame_candidate(
        box_prompt_instances=_instances([]),
        object_prompt_instances=objects,
        prompt_boxes=prompts,
        joints_by_side=_joints(),
        cad_object_mask=cad,
    )
    assert result.final_mask.sum() == 0
    assert not result.sufficient
    assert set(result.hold_reasons) >= {
        "missing_box_prompt_complete_limb_instance:left",
        "missing_box_prompt_complete_limb_instance:right",
    }


def test_ambiguous_instance_is_not_assigned_to_both_sides():
    box_instances, objects, prompts, cad = _good_inputs(include_background=False)
    ambiguous = np.logical_or(box_instances.masks[0], box_instances.masks[1])
    result = select_frame_candidate(
        box_prompt_instances=_instances([ambiguous], ids=[77]),
        object_prompt_instances=objects,
        prompt_boxes=prompts,
        joints_by_side=_joints(),
        cad_object_mask=cad,
    )
    assert not result.sufficient
    assert result.selected_ids == {}


def test_object6d_selected_instance_only_subtracts():
    box_instances, _, prompts, _ = _good_inputs(include_background=False)
    object_mask = _rectangle(30, 45, 48, 62)
    result = select_frame_candidate(
        box_prompt_instances=box_instances,
        object_prompt_instances=_instances([object_mask], ids=[50]),
        prompt_boxes=prompts,
        joints_by_side=_joints(),
        cad_object_mask=object_mask,
    )
    assert result.diagnostics["pre_object_overlap_pixels"] > 0
    assert result.diagnostics["post_object_overlap_pixels"] == 0
    assert not np.logical_and(result.final_mask, object_mask).any()
    all_sam = np.logical_or.reduce(box_instances.masks, axis=0)
    assert not np.logical_and(result.final_mask, ~all_sam).any()


def test_missing_visible_object_instance_holds_without_geometry_mask():
    box_instances, _, prompts, cad = _good_inputs(include_background=False)
    result = select_frame_candidate(
        box_prompt_instances=box_instances,
        object_prompt_instances=_instances([]),
        prompt_boxes=prompts,
        joints_by_side=_joints(),
        cad_object_mask=cad,
    )
    assert "missing_object6d_selected_visible_object_instance" in result.hold_reasons
    assert result.object_protect_mask.sum() == 0


def test_temporal_regression_rejects_without_warping_output():
    box_instances, objects, prompts, cad = _good_inputs(include_background=False)
    previous = {
        "left": np.flip(box_instances.masks[0], axis=1),
        "right": np.flip(box_instances.masks[1], axis=1),
    }
    result = select_frame_candidate(
        box_prompt_instances=box_instances,
        object_prompt_instances=objects,
        prompt_boxes=prompts,
        joints_by_side=_joints(),
        cad_object_mask=cad,
        previous_side_masks=previous,
    )
    assert any(reason.startswith("side_temporal_iou_below_limit") for reason in result.hold_reasons)
    assert result.diagnostics["pixel_lineage"]["flow_warp_created_output_pixels"] == 0


def test_selector_api_has_no_session_or_frame_exception_surface():
    parameters = inspect.signature(select_frame_candidate).parameters
    assert "session_id" not in parameters
    assert "frame_index" not in parameters


def test_invalid_prompt_box_area_fails_closed():
    joints = _joints().copy()
    joints[0, :, 0] = np.linspace(5, 95, 21)
    joints[0, :, 1] = np.linspace(5, 95, 21)
    with pytest.raises(ProducerP3ContractError, match="area bound"):
        derive_two_limb_boxes(joints, WIDTH, HEIGHT)


class Sam3MultiplexTrackingWithInteractivity:
    def __init__(self):
        self.init_calls = []
        self.prompt_calls = []

    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        async_loading_frames=False,
        use_torchcodec=False,
        use_cv2=False,
        input_is_mp4=False,
    ):
        self.init_calls.append((resource_path, offload_video_to_cpu, async_loading_frames))
        return {"resource_path": resource_path}

    def add_prompt(
        self,
        inference_state,
        frame_idx,
        text_str=None,
        clear_old_points=True,
        points=None,
        point_labels=None,
        boxes_xywh=None,
        box_labels=None,
        clear_old_boxes=True,
        output_prob_thresh=0.5,
        obj_id=None,
        rel_coordinates=True,
    ):
        self.prompt_calls.append(
            {
                "frame_idx": frame_idx,
                "text_str": text_str,
                "points": points,
                "boxes_xywh": boxes_xywh,
                "box_labels": box_labels,
                "output_prob_thresh": output_prob_thresh,
            }
        )
        return frame_idx, {"out_binary_masks": [True]}

    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        output_prob_thresh=0.5,
        compute_stability_score=False,
        is_instance_processing=False,
        is_last_batch=False,
    ):
        yield 0, {"out_binary_masks": [True]}


Sam3MultiplexTrackingWithInteractivity.__module__ = (
    "sam3.model.sam3_multiplex_tracking"
)


class Sam3MultiplexVideoPredictor:
    def __init__(self):
        self.model = Sam3MultiplexTrackingWithInteractivity()


Sam3MultiplexVideoPredictor.__module__ = (
    "sam3.model.sam3_multiplex_video_predictor"
)


def _start(adapter, **updates):
    request = {
        "type": "start_session",
        "resource_path": "/isolated/verified/raw",
        "session_id": "box-session",
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
        "async_loading_frames": False,
    }
    request.update(updates)
    return adapter.handle_request(request)


def _add(adapter, **updates):
    request = {
        "type": "add_geometric_box_prompt",
        "session_id": "box-session",
        "frame_index": 0,
        "boxes_xywh": [[0.0, 0.2, 0.4, 0.6], [0.6, 0.4, 0.4, 0.6]],
        "box_labels": [1, 1],
        "output_prob_thresh": 0.5,
    }
    request.update(updates)
    return adapter.handle_request(request)


def test_geometric_adapter_maps_exact_official_box_api_without_text_or_points():
    predictor = Sam3MultiplexVideoPredictor()
    adapter = Sam31GeometricBoxAdapterV1(predictor)
    _start(adapter)
    response = _add(adapter)
    assert response["prompt_type"] == "official_sam31_positive_boxes"
    call = predictor.model.prompt_calls[-1]
    assert call["text_str"] is None
    assert call["points"] is None
    assert call["box_labels"] == [1, 1]
    assert adapter.signature_evidence["signatures_match"] is True


@pytest.mark.parametrize(
    "change,pattern",
    [
        ({"box_labels": [1, 0]}, "labels"),
        ({"boxes_xywh": [[0, 0, 1, 1]]}, "exactly two"),
        ({"boxes_xywh": [[0, 0, 0.5, 0.5], [0.8, 0.8, 0.4, 0.4]]}, "inside"),
        ({"output_prob_thresh": 0.4}, "fixed at 0.5"),
        ({"unexpected": True}, "unknown keys"),
    ],
)
def test_geometric_adapter_rejects_prompt_contract_drift(change, pattern):
    adapter = Sam31GeometricBoxAdapterV1(Sam3MultiplexVideoPredictor())
    _start(adapter)
    with pytest.raises(Sam31GeometricBoxContractError, match=pattern):
        _add(adapter, **change)


def test_geometric_adapter_rejects_state_offload_and_duplicate_session():
    adapter = Sam31GeometricBoxAdapterV1(Sam3MultiplexVideoPredictor())
    with pytest.raises(Sam31GeometricBoxContractError, match="no pinned multiplex"):
        _start(adapter, offload_state_to_cpu=True)
    _start(adapter)
    with pytest.raises(Sam31GeometricBoxContractError, match="already exists"):
        _start(adapter)


def test_geometric_adapter_has_no_point_or_text_request_fallback():
    adapter = Sam31GeometricBoxAdapterV1(Sam3MultiplexVideoPredictor())
    _start(adapter)
    with pytest.raises(Sam31GeometricBoxContractError, match="unsupported"):
        adapter.handle_request(
            {"type": "add_prompt", "session_id": "box-session", "text": "a sleeve"}
        )


def test_owned_run_and_session_tree_are_exclusive(tmp_path):
    (tmp_path / "_run").mkdir()
    token = "a" * 64
    component, marker = prepare_owned_run(tmp_path, "p3_test_run", token)
    assert component.is_dir()
    assert marker.is_file() and not marker.is_symlink()
    with pytest.raises(FileExistsError):
        prepare_owned_run(tmp_path, "p3_test_run", token)
    prepare_session_tree(component, "grap_a_cap_004")
    with pytest.raises(FileExistsError):
        prepare_session_tree(component, "grap_a_cap_004")


@pytest.mark.parametrize(
    "run_id,token,pattern",
    [
        ("../escape", "a" * 64, "invalid run-id"),
        ("p3_good", "A" * 64, "owner-token"),
        ("p3_good", "abc", "owner-token"),
    ],
)
def test_owned_run_rejects_unresolved_identity(tmp_path, run_id, token, pattern):
    (tmp_path / "_run").mkdir()
    with pytest.raises(ValueError, match=pattern):
        prepare_owned_run(tmp_path, run_id, token)


def test_input_hash_and_symlink_drift_fail_closed(tmp_path):
    payload = tmp_path / "input.json"
    payload.write_bytes(b"frozen-input")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    assert require_ordinary(payload, digest, len(b"frozen-input")) == payload.resolve()
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        require_ordinary(payload, "0" * 64)
    link = tmp_path / "link.json"
    link.symlink_to(payload)
    with pytest.raises(RuntimeError, match="ordinary file"):
        require_ordinary(link, digest)


def test_qa_gate_rehashes_before_execution_and_detects_drift(tmp_path):
    audits = tmp_path / "audits"
    audits.mkdir()
    project = Path(__file__).resolve().parents[1]
    compat_name = "INDEPENDENT_QA_SAM31_COMPAT_ADAPTER.json"
    p2_name = "INDEPENDENT_QA_SAM31_PRODUCER_P2_DEV15.json"
    shutil.copy2(project / "audits" / compat_name, audits / compat_name)
    shutil.copy2(project / "audits" / p2_name, audits / p2_name)
    evidence = verify_qa_gates(tmp_path)
    assert evidence["compat_adapter"]["p0_findings"] == 0
    assert evidence["p2_independent_qa"]["p0_findings"] == 0
    with (audits / p2_name).open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(RuntimeError, match="byte-count mismatch"):
        verify_qa_gates(tmp_path)
