"""CPU-only binding and synthetic-layer tests; never invoke Robot raster."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from chaoyang.governance.common import REPO_ROOT, artifact_ref
from chaoyang.pipeline import exact78_clean_robot_diagnostic_cards042_v1 as candidate


TASKS = {
    "scene_result": "exact78_raw_to_robot_t2_cards042_frontend_v1_20260930",
    "action_result": "exact78_raw_to_robot_t2_0902_cards042_v1_20260930",
    "render_result": "exact78_robot_render_direction_cards042_v1_20260930",
}


def sealed_spec() -> dict:
    return {role: artifact_ref(REPO_ROOT / "_run/current" / task /
                               "attempts/attempt_0001/RESULT.json")
            for role, task in TASKS.items()}


@pytest.fixture(scope="module")
def binding() -> dict:
    return candidate.prepare_binding(sealed_spec())


def test_sealed_cards042_clip_index_binding_only(binding: dict) -> None:
    assert binding["frame_count"] == 171
    assert binding["fps_nominal"] == 30
    assert binding["time_mode"] == "CLIP_INDEX_ONLY_TIMEBASE_UNVERIFIED"
    assert binding["image_domain_claim"].endswith("PIXEL_EQUIVALENCE_NOT_PROVEN")
    assert binding["geometry_decision"] == "UNKNOWN"
    assert binding["display_policy"] == candidate.DISPLAY_POLICY
    assert len(binding["clean_rows"]) == len(binding["raw_rows"]) == 171


def test_reject_unfrozen_or_nonexistent_outer() -> None:
    spec = sealed_spec()
    spec["scene_result"] = {**spec["scene_result"], "sha256": "0" * 64}
    with pytest.raises(candidate.CompositeBindingError, match="UNFROZEN_OUTER_RESULT"):
        candidate.prepare_binding(spec)


def test_reject_frame_swap_or_timebase_upgrade() -> None:
    row = {"decode_display_index": 0, "model_input_index": 0,
           "model_input": {"sha256": "same"}, "capture_frame_id": None,
           "time_mode": "TIMEBASE_UNVERIFIED"}
    mask = {"decode_display_index": 0, "model_input": row["model_input"],
            "masks": {"human": "h"}}
    clean = {"decode_display_index": 0, "model_output_index": 0,
             "model_input": row["model_input"], "masks": mask["masks"]}
    provenance = {"decoded_display_index": 0, "model_input_index": 0,
                  "output_index": 0, "capture_frame_id": None,
                  "time_mode": "CLIP_FRAME_INDEX_ONLY"}
    jpeg = {"decoded_display_index": 0, "model_input_index": 0}
    candidate.validate_frame_rows([row], [mask], [clean], [provenance], [jpeg], 1)
    swapped = deepcopy(provenance)
    swapped["output_index"] = 1
    with pytest.raises(candidate.CompositeBindingError, match="FRAME_INDEX_MISMATCH"):
        candidate.validate_frame_rows([row], [mask], [clean], [swapped], [jpeg], 1)
    upgraded = deepcopy(row)
    upgraded["time_mode"] = "QPC_SYNC_VERIFIED"
    with pytest.raises(candidate.CompositeBindingError, match="TIMEBASE_CLAIM_DRIFT"):
        candidate.validate_frame_rows([upgraded], [mask], [clean], [provenance], [jpeg], 1)


def test_missing_layer_and_bad_depth_stop() -> None:
    clean = np.zeros((960, 1280, 3), np.uint8)
    with pytest.raises(candidate.CompositeBindingError, match="BLOCKED_RENDER_LAYER_ABSENT"):
        candidate.compose_frame(clean, None, None, None, frame_index=0)
    rgb = np.zeros((480, 640, 3), np.uint8)
    labels = np.full((480, 640), -1, np.int32)
    labels[0, 0] = 7
    depth = np.zeros((480, 640), np.float32)
    depth[0, 0] = np.nan
    with pytest.raises(candidate.CompositeBindingError, match="ROBOT_LAYER_DEPTH_INVALID"):
        candidate.compose_frame(clean, rgb, labels, {"depth_m": depth}, frame_index=0)
    with pytest.raises(candidate.CompositeBindingError, match="BLOCKED_RENDER_LAYER_ABSENT"):
        candidate.compose_frame(clean, rgb, labels, {}, frame_index=0)


def test_synthetic_clean_robot_overlay_is_diagnostic_only() -> None:
    clean = np.full((960, 1280, 3), 20, np.uint8)
    rgb = np.full((480, 640, 3), 250, np.uint8)
    labels = np.full((480, 640), -1, np.int32)
    labels[0, 0] = 3
    depth = np.full((480, 640), np.inf, np.float32)
    depth[0, 0] = 0.5
    panel, state = candidate.compose_frame(clean, rgb, labels, {"depth_m": depth},
                                           frame_index=85)
    assert panel.shape == (480, 640, 3)
    assert tuple(panel[0, 0]) == (250, 250, 250)
    assert tuple(panel[0, 1]) == (20, 20, 20)
    assert state["robot_visible_pixels"] == 1
    assert state["robot_link_labels_present"] == [3]
    assert state["geometry_decision"] == "UNKNOWN"
    assert state["display_policy"] == candidate.DISPLAY_POLICY
    assert state["pixel_source_enum"] == {
        "0": "SYNTHETIC_CLEAN_BACKGROUND", "1": "RENDERED_ROBOT"}
    assert not state["scene_depth_consumed"]
    assert not state["object_mask_consumed_for_occlusion"]
    assert not state["visual_qualified"]
    assert not state["product_qualified"]
    assert not state["training_eligible"]
    assert not state["control_ground_truth"]


def test_frame_adapter_uses_corrected_camera_and_full_robot(
        binding: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_raster(_raster, _assets, q_arm, q_hand, mounts, camera, k,
                    cache, flange, **kwargs):
        called.update({"camera": camera.copy(), "kwargs": kwargs,
                       "q_arm": q_arm.copy(), "q_hand": q_hand.copy(),
                       "mounts": mounts.copy(), "k": k.copy()})
        rgb = np.zeros((480, 640, 3), np.uint8)
        labels = np.full((480, 640), -1, np.int32)
        labels[0, 0] = 1
        depth = np.full((480, 640), np.inf, np.float32)
        depth[0, 0] = 0.3
        return rgb, labels, {"depth_m": depth}

    monkeypatch.setattr(candidate.direction.raster, "render_robot", fake_raster)
    _, row = candidate.render_frame(binding, 0, assets=object(), cache={},
                                    flange=np.empty((0, 3, 3)))
    assert np.array_equal(called["camera"], binding["states"]["T_camera_base"])
    assert called["kwargs"]["complete_robot"] is True
    assert called["kwargs"]["return_buffers"] is True
    assert row["raw_model_input"] == binding["raw_rows"][0]["model_input"]
    assert row["display_mask_refs"]["removal"] == binding["mask_rows"][0]["masks"]["removal"]
    assert row["clean"] == binding["clean_rows"][0]["clean"]
    assert row["q_state_ref"] == binding["robot_states_ref"]
    assert row["scene_decode_display_index"] == row["hawor_output_index"] == 0


def test_iterator_is_fixed_171_frames_and_has_no_writer(
        binding: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(candidate.direction, "load_pinned_robot_assets", lambda _root: object())
    monkeypatch.setattr(candidate.direction.raster, "naturalv2_local_triangles",
                        lambda _path: np.empty((0, 3, 3)))
    seen = []

    def fake_frame(_binding, index, *, assets, cache, flange):
        seen.append(index)
        return np.empty((0, 0, 3), np.uint8), {"output_index": index}

    monkeypatch.setattr(candidate, "render_frame", fake_frame)
    rows = [row for _, row in candidate.iter_frames(binding)]
    assert seen == list(range(171))
    assert [row["output_index"] for row in rows] == seen
