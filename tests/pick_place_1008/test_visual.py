from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.exact78_baseline.clean import ProPainter, SETTINGS
from chaoyang.pick_place_1008 import visual


def _masks(count=2, height=16, width=16):
    removal = np.zeros((count, height, width), bool)
    removal[:, 8, 5] = True
    protected = np.zeros_like(removal)
    protected[:, 8, 9] = True
    return removal, protected


def test_internal_dilation_cannot_reenter_protected_cup():
    removal, protected = _masks()
    effective, record = visual.effective_clean_masks(removal, protected)
    assert not (effective & protected).any()
    assert record["dilated_protected_overlap_per_frame"] == [1, 1]
    assert record["effective_protected_overlap_per_frame"] == [0, 0]
    assert effective[:, 8, 8].all()
    assert not removal[:, 8, 8].any()


def test_mask_domains_and_frozen_settings_are_checked():
    removal, protected = _masks()
    with pytest.raises(ValueError, match="MASK_DOMAIN_OR_TYPE"):
        visual.effective_clean_masks(removal, protected.astype(np.uint8))
    with pytest.raises(ValueError, match="FROZEN_SETTINGS_DRIFT"):
        visual.effective_clean_masks(removal, protected, {**SETTINGS, "mask_dilation": 3})


def _fake_parent():
    """Real CPU tensors with spies; no pretrained model or CUDA work."""
    torch = pytest.importorskip("torch")

    class Raft:
        def __call__(self, frames, iters):
            assert iters == 20
            b, t, _, h, w = frames.shape
            zero = torch.zeros((b, t - 1, 2, h, w), dtype=frames.dtype)
            return zero, zero.clone()

    class Flow:
        def __init__(self):
            self.masks = []

        def forward_bidirect_flow(self, part, masks):
            self.masks.append(masks.clone())
            return part, None

        def combine_flow(self, part, predicted, masks):
            self.masks.append(masks.clone())
            return predicted

    class Model:
        def __init__(self):
            self.propagation = []
            self.inference_masks = []

        def img_propagation(self, masked, flows, masks, mode):
            assert mode == "nearest"
            self.propagation.append((masked.clone(), masks.clone()))
            b, t, _, h, w = masked.shape
            return torch.ones((b * t, 3, h, w), dtype=masked.dtype), torch.zeros(
                (b * t, 1, h, w), dtype=masks.dtype)

        def __call__(self, images, flows, masks, updated_mask, count):
            self.inference_masks.append(masks.clone())
            return torch.ones((count, 3, *images.shape[-2:]), dtype=images.dtype)

    parent = object.__new__(ProPainter)
    parent.settings = dict(SETTINGS)
    parent.torch, parent.device = torch, torch.device("cpu")
    parent.raft, parent.flow, parent.model = Raft(), Flow(), Model()
    return parent


@pytest.mark.parametrize("count", [2, 83])
def test_protection_reaches_flow_propagation_and_model_input(count):
    parent = _fake_parent()
    removal, protected = _masks(count)
    frames = np.full((count, 16, 16, 3), 41, np.uint8)
    frames[:, 8, 9] = (230, 120, 17)
    original, initial_removal, initial_protected = frames.copy(), removal.copy(), protected.copy()
    cleaner = visual.ProtectedProPainter(parent)
    result = cleaner.inpaint(frames, removal, protected)
    assert np.array_equal(result[protected], original[protected])
    assert np.array_equal(result[~cleaner.effective_masks], original[~cleaner.effective_masks])
    assert np.array_equal(frames, original)
    assert np.array_equal(removal, initial_removal)
    assert np.array_equal(protected, initial_protected)
    assert parent.flow.masks and parent.model.propagation and parent.model.inference_masks
    for mask in parent.flow.masks + parent.model.inference_masks:
        assert (mask[:, :, :, 8, 9] == 0).all()
    expected = parent.torch.tensor((230, 120, 17)).float().div(127.5).sub(1).half()
    for masked, mask in parent.model.propagation:
        assert (mask[:, :, :, 8, 9] == 0).all()
        assert parent.torch.equal(masked[0, 0, :, 8, 9], expected)
    record = cleaner.last_validation
    assert record["model_protected_rgb_unchanged"] is True
    assert record["protected_changed_pixels_per_frame"] == [0] * count
    assert record["outside_delete_changed_pixels_per_frame"] == [0] * count
    assert len(record["output_frame_visits"]) == count


def test_empty_protection_matches_frozen_parent_path():
    frames = np.arange(2 * 16 * 16 * 3, dtype=np.uint8).reshape(2, 16, 16, 3)
    removal, protected = _masks()
    protected[:] = False
    first, second = _fake_parent(), _fake_parent()
    baseline = first.inpaint(frames, removal)
    candidate = visual.ProtectedProPainter(second).inpaint(frames, removal, protected)
    assert np.array_equal(candidate, baseline)


def test_frozen_model_settings_cannot_change_after_wrapper_creation():
    parent = _fake_parent()
    cleaner = visual.ProtectedProPainter(parent)
    parent.settings["ref_stride"] = 11
    removal, protected = _masks()
    with pytest.raises(ValueError, match="FROZEN_SETTINGS_DRIFT"):
        cleaner.inpaint(np.zeros((2, 16, 16, 3), np.uint8), removal, protected)


def test_candidate_compose_unknown_cup_object_first_keeps_unknown():
    shape = (24, 32)
    raw = np.full((*shape, 3), 37, np.uint8)
    robot = np.full_like(raw, 222)
    alpha = np.zeros(shape, bool)
    alpha[10:13, 10:13] = True
    protected = np.zeros(shape, bool)
    protected[11, 11] = True
    masks = {"core": np.zeros(shape, bool), "protected_visible": protected,
             "unknown": protected.copy()}
    result = visual.compose_frame(raw, raw.copy(), robot, alpha, masks, robot_observed=True)
    assert np.array_equal(result["image"][11, 11], raw[11, 11])
    assert np.array_equal(result["image"][10, 10], robot[10, 10])
    assert result["object_unknown_overlap"][11, 11]
    assert result["unknown_visible_object_first"][11, 11]
    assert not result["draw"][11, 11]
    assert result["physical_occlusion_authority"] is False
    assert result["physical_contact_ground_truth"] is False
    assert masks["unknown"][11, 11]


def test_unknown_without_visible_support_does_not_restore_hidden_cup():
    shape = (24, 32)
    raw = np.zeros((*shape, 3), np.uint8)
    robot = np.full_like(raw, 222)
    alpha, unknown = np.zeros(shape, bool), np.zeros(shape, bool)
    alpha[11, 11] = unknown[11, 11] = True
    masks = {"core": np.zeros(shape, bool), "protected_visible": np.zeros(shape, bool),
             "unknown": unknown}
    result = visual.compose_frame(raw, raw, robot, alpha, masks, robot_observed=True)
    assert np.array_equal(result["image"][11, 11], robot[11, 11])
    assert result["unknown_visible_object_first_pixels"] == 0


def test_compose_rejects_cup_drift_and_suppresses_unobserved_robot():
    shape = (24, 32)
    raw = np.zeros((*shape, 3), np.uint8)
    cup = np.zeros(shape, bool)
    cup[11, 11] = True
    masks = {"core": np.zeros(shape, bool), "protected_visible": cup, "unknown": cup}
    clean = raw.copy()
    clean[11, 11] = 1
    with pytest.raises(RuntimeError, match="CLEAN_PROTECTED_PIXELS_DRIFT"):
        visual.compose_frame(raw, clean, raw, cup, masks, robot_observed=True)
    result = visual.compose_frame(raw, raw, np.full_like(raw, 200), cup, masks, robot_observed=False)
    assert np.array_equal(result["image"], raw)
    assert result["unknown_visible_object_first_pixels"] == 0


def test_equipment_only_keeps_components_supported_by_wearer():
    shape = (24, 32)
    human = np.zeros((2, *shape), bool)
    human[:, 10, 10] = True
    equipment = np.zeros(shape, bool)
    equipment[10, 10:15] = True
    equipment[1, 1:4] = True  # Separate background component.
    tracks = {1: {frame: np.packbits(equipment) for frame in range(2)}}
    selected, rows = visual._owned_equipment(tracks, human, 0.6)
    assert rows[0]["kept"]
    assert selected[:, 10, 10:15].all()
    assert not selected[:, 1, 1:4].any()


def _fake_mask_run(monkeypatch, *, cup_visible=True, semantic_conflict=False):
    shape, count = (24, 32), 2
    core = np.ones((count, *shape), bool)  # Geometric fallback can cover a real cup.
    semantic = np.zeros_like(core)
    cup = np.zeros(shape, bool)
    if cup_visible:
        cup[10:14, 10:14] = True
    if semantic_conflict:
        semantic[:, 10, 10] = True
    info = {"provider": np.zeros((count, 2), np.uint8),
            "pico_shift_px": np.zeros((count, 2, 2))}
    monkeypatch.setattr(visual.parent_mask, "pico_geometry", lambda session, scale: {})
    monkeypatch.setattr(visual.parent_mask, "anchor_frame", lambda geometry, shape: 0)
    monkeypatch.setattr(visual, "_track_wearer_with_semantics",
                        lambda *args: (core.copy(), core.copy(), semantic.copy(), info))

    class Sam:
        aborted = []

        def track_text(self, frames, prompt, anchor):
            if prompt == visual.PROMPTS["object"] and cup_visible:
                return {1: {frame: np.packbits(cup) for frame in range(count)}}
            return {}

    session = SimpleNamespace(frame_count=count, width=32, height=24, task="pick_place",
                              native_motion={"head_source": np.full(count, "RAW_POSE")})
    result = visual.build_visual_masks(Sam(), np.zeros((count, *shape, 3), np.uint8), session)
    return result, cup


def test_geometric_hull_cannot_veto_current_cup_protection(monkeypatch):
    result, cup = _fake_mask_run(monkeypatch)
    assert result["core"][:, cup].all()
    assert result["protected_visible"][:, cup].all()
    assert not result["removal"][:, cup].any()
    assert np.array_equal(result["unknown"], result["protected_visible"])
    assert result["metadata"]["geometry_vetoes_cup_protection"] is False


def test_semantic_identity_conflict_is_separate_from_occlusion_unknown(monkeypatch):
    result, cup = _fake_mask_run(monkeypatch, semantic_conflict=True)
    assert result["semantic_conflict"][:, 10, 10].all()
    assert not result["protected_visible"][:, 10, 10].any()
    assert not result["unknown"][:, 10, 10].any()
    assert result["protected_visible"][:, 11, 11].all()
    assert any(row["code"] == "PAPER_CUP_WEARER_SEMANTIC_CONFLICT"
               for row in result["metadata"]["quality_problems"])


def test_missing_cup_support_is_quality_failure(monkeypatch):
    result, _ = _fake_mask_run(monkeypatch, cup_visible=False)
    assert not result["protected_visible"].any()
    assert any(row["code"] == "PAPER_CUP_VISIBLE_SUPPORT_MISSING"
               and row["severity"] == "QUALITY_FAILURE"
               for row in result["metadata"]["quality_problems"])


@pytest.mark.parametrize("fps", [25, 30])
def test_candidate_windows_use_qpc_and_missing_cup_stays_unknown(fps):
    count = fps * 2 + 1
    qpc = np.arange(count, dtype=np.int64) * (3_000_000 // fps)
    human = np.zeros((count, 480, 640), bool)
    cup = np.zeros_like(human)
    cup[:, 50, 100] = True
    human[fps // 2, 50, 88] = True  # 12 px at 640x480.
    cup[-1] = False
    result = visual.build_contact_candidates(qpc, 3_000_000, human, cup)
    trigger = fps // 2
    assert result["frames"][trigger]["candidate"]
    assert result["frames"][-1]["status"] == "UNKNOWN"
    assert result["frames"][-1]["physical_contact"] == "UNKNOWN"
    span = result["windows"][0]
    time = qpc[trigger] / 3_000_000
    assert span["start_time_s"] == pytest.approx(max(0, time - 0.5))
    assert span["end_time_s"] == pytest.approx(time + 0.5)
    assert result["physical_contact_ground_truth"] is False


def test_projected_fingertip_and_window_merge_have_no_physical_claim():
    qpc = np.arange(51, dtype=np.int64) * 40_000
    human = np.zeros((51, 24, 32), bool)
    cup = np.zeros_like(human)
    cup[:, 10, 10] = True
    points = np.full((2, 51, 21, 2), np.nan)
    points[0, 10, 4] = (10, 10)
    points[0, 24, 8] = (10, 10)
    result = visual.build_contact_candidates(qpc, 1_000_000, human, cup, hand_points_px=points)
    assert result["frames"][10]["reasons"] == ["PROJECTED_FINGERTIP_CUP_IMAGE_PROXIMITY"]
    assert len(result["windows"]) == 1
    assert result["windows"][0]["candidate_frames"] == [10, 24]
    points[0, 24, 8] = np.nan
    points[0, 40, 8] = (10, 10)
    result = visual.build_contact_candidates(qpc, 1_000_000, human, cup, hand_points_px=points)
    assert len(result["windows"]) == 2


def test_duplicate_qpc_and_wrong_resolution_are_rejected():
    human = np.zeros((2, 24, 32), bool)
    with pytest.raises(ValueError, match="NOT_STRICTLY_INCREASING"):
        visual.build_contact_candidates(np.array([10, 10]), 1000, human, human)
    wrong = np.zeros((2, 16, 16), bool)
    with pytest.raises(ValueError, match="ISOTROPIC"):
        visual.build_contact_candidates(np.array([10, 20]), 1000, wrong, wrong)


def test_unsigned_backward_qpc_and_invalid_projection_stay_unknown():
    human = np.zeros((2, 24, 32), bool)
    cup = human.copy()
    cup[:, 10, 10] = True
    with pytest.raises(ValueError, match="NOT_STRICTLY_INCREASING"):
        visual.build_contact_candidates(np.array([20, 10], np.uint64), 1000, human, cup)
    result = visual.build_contact_candidates(
        np.array([10, 20]), 1000, human, cup,
        hand_points_px=np.full((2, 2, 5, 2), np.nan))
    assert all(row["status"] == "UNKNOWN" and row["projected_hand_valid"] is False
               for row in result["frames"])


def _render_contract():
    from chaoyang.pick_place_1008 import render_stage

    count, shape = 2, (2, 480, 640)
    raw = np.full((*shape, 3), 21, np.uint8)
    masks = {name: np.zeros(shape, bool) for name in
             ("core", "removal", "protected_visible", "unknown", "sam_core", "equipment")}
    masks["raw_rgb"] = raw
    motion = {"q_arm": np.zeros((count, 2, 7)), "q_hand": np.zeros((count, 2, 22)),
              "T_world_base": np.eye(4),
              "qpc_velocity_violation": np.zeros((count, 2), bool),
              "qpc_acceleration_violation": np.zeros((count, 2), bool),
              "arm_position_mm": np.zeros((2, count)),
              "arm_rotation_deg": np.zeros((2, count))}
    session = SimpleNamespace(frame_count=count, qpc_ns=np.array([1_000_000_000, 1_040_000_000]),
                              c2w=np.repeat(np.eye(4)[None], count, axis=0))
    return render_stage, session, motion, raw, masks


def test_render_requires_lossless_raw_and_same_effective_mask_without_rendering():
    rd, session, motion, raw, masks = _render_contract()
    _, _, validation = rd._validate_inputs(session, motion, raw.copy(), masks)
    assert validation["clean_outside_changed_pixels_per_frame"] == [0, 0]
    missing = dict(masks)
    missing.pop("raw_rgb")
    with pytest.raises(ValueError, match="RAW_LOSSLESS_ARRAY_REQUIRED"):
        rd._validate_inputs(session, motion, raw, missing)
    changed = raw.copy()
    changed[0, 20, 20] = 22
    with pytest.raises(ValueError, match="CLEAN_PIXEL_PROTECTION_FAILED"):
        rd._validate_inputs(session, motion, changed, masks)
    masks["effective_removal"] = np.ones_like(masks["removal"])
    with pytest.raises(ValueError, match="EFFECTIVE_MASK_SIGNATURE_MISMATCH"):
        rd._validate_inputs(session, motion, raw, masks)


def test_render_metadata_unwraps_gpu_stage_wrapper_without_rendering():
    from chaoyang.pick_place_1008.render_stage import _mask_metadata

    problem = {"code": "PAPER_CUP_VISIBLE_SUPPORT_MISSING", "severity": "QUALITY_FAILURE"}
    assert _mask_metadata({"metadata": {"metadata": {"quality_problems": [problem]}}}) == {
        "quality_problems": [problem]}


def test_diagnostic_hand_geometry_cannot_support_sam_ownership():
    geometry = {
        "hands": np.ones((2, 2, 21, 2)),
        "forearm": np.ones((2, 2, 6, 2)),
        "forearm_radius": np.ones((2, 2)),
        "tracker": np.ones((2, 2, 36, 2)),
        "wrist_px_per_m": np.ones((2, 2)),
        "hands_inside": np.ones((2, 2, 21), bool),
    }
    session = SimpleNamespace(frame_count=2, motion_valid={
        "hand_display_valid": np.array([[True, False], [True, True]])},
        native_motion={"head_source": np.array(["RAW_POSE", "RAW_POSE"])})
    gated, invalid = visual._gate_diagnostic_geometry(geometry, session)
    assert invalid == [[0, 1]]
    assert np.isnan(gated["hands"][0, 1]).all()
    assert np.isnan(gated["tracker"][0, 1]).all()
    assert not gated["hands_inside"][0, 1].any()
    assert np.isfinite(gated["hands"][1]).all()
    assert geometry["hands_inside"].all()
    assert np.isfinite(geometry["hands"]).all()


def test_camera_diagnostic_and_missing_head_cannot_support_geometry():
    geometry = {"hands": np.ones((2, 4, 21, 2)),
                "hands_inside": np.ones((2, 4, 21), bool)}
    session = SimpleNamespace(frame_count=4, motion_valid={
        "hand_display_valid": np.ones((2, 4), bool)}, native_motion={
        "head_source": np.array(["RAW_POSE", "INTERP_SHORT", "EDGE_HOLD_SHORT",
                                 "DIAGNOSTIC_LONG_HOLD"])})
    gated, invalid = visual._gate_diagnostic_geometry(geometry, session)
    assert invalid == [[0, 3], [1, 3]]
    assert np.isfinite(gated["hands"][:, :3]).all()
    assert np.isnan(gated["hands"][:, 3]).all()
    session.native_motion = {}
    assert not visual.head_hint_valid(session).any()
    gated, invalid = visual._gate_diagnostic_geometry(geometry, session)
    assert len(invalid) == 8
    assert np.isnan(gated["hands"]).all()
    assert np.isfinite(geometry["hands"]).all()


def test_contact_and_abnormal_windows_share_qpc_rule():
    qpc = np.arange(51, dtype=np.int64) * 40_000_000
    candidate = visual.review_windows_from_qpc(qpc, 1e9, [10, 24])
    abnormal = visual.review_windows_from_qpc(qpc, 1e9, [24, 10, 10],
                                            trigger_key="abnormal_frames")
    assert len(candidate) == len(abnormal) == 1
    for name in ("start_time_s", "end_time_s", "start_frame", "end_frame_inclusive"):
        assert candidate[0][name] == abnormal[0][name]
    assert abnormal[0]["abnormal_frames"] == [10, 24]
    with pytest.raises(ValueError, match="TRIGGER_FRAME_DOMAIN"):
        visual.review_windows_from_qpc(qpc, 1e9, [True])


def test_real_upstream_refs_are_checked_without_copying(tmp_path):
    from chaoyang.pick_place_1008 import render_stage
    from chaoyang.pick_place_1008.source import ref

    old = tmp_path / "previous_attempt"
    old.mkdir()
    pins = {}
    for role in ("actions", "frame_mapping", "raw_rgb", "clean_rgb", "masks", "motion_info"):
        path = old / (role + ".bin")
        path.write_bytes(("immutable " + role).encode())
        pins[role] = ref(path)
    actual = render_stage._upstream_refs(tmp_path, {"upstream_refs": pins})
    assert actual == pins
    assert all("previous_attempt" in item["path"] for item in actual.values())
    assert actual["actions"] is not pins["actions"]
    (old / "actions.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="REFERENCE_DRIFT"):
        render_stage._upstream_refs(tmp_path, {"upstream_refs": pins})
    with pytest.raises(ValueError, match="REFERENCES_REQUIRED"):
        render_stage._upstream_refs(tmp_path, {})


def test_frame_states_serialize_null_errors_and_seal_real_anomalies():
    import json
    from chaoyang.pick_place_1008 import render_stage

    count = 3
    shape = (count, 24, 32)
    masks = {name: np.zeros(shape, bool) for name in
             ("protected_visible", "semantic_conflict", "sam_core", "equipment")}
    masks["sam_core"][0, 10, 10] = True
    masks["semantic_conflict"][1, 10, 10] = True
    motion = {
        "qpc_velocity_violation": np.array([[False, False], [True, False], [False, False]]),
        "qpc_acceleration_violation": np.array([[False, False], [False, False], [False, True]]),
        "arm_position_mm": np.array([[1., 21., np.nan], [2., 2., 2.]]),
        "arm_rotation_deg": np.array([[1., 2., np.inf], [1., 16., 1.]]),
    }
    session = SimpleNamespace(
        frame_count=count, qpc_ns=np.array([1_000_000_000, 1_040_000_000, 1_080_000_000]),
        hand_observed=np.array([[True, True, False], [True, False, True]]),
        hand_source=np.array([["RAW_COMPOSED", "RAW_COMPOSED", "DIAGNOSTIC_LONG_HOLD"],
                              ["RAW_COMPOSED", "INTERP_SHORT", "RAW_COMPOSED"]]),
        motion_valid={"hand_display_valid": np.array([[True, True, False], [True, True, True]]),
                      "controller_valid": np.ones((count, 2), bool),
                      "manus_valid": np.ones((count, 2), bool),
                      "head_valid": np.array([True, False, False])},
        native_motion={"head_source": np.array(["RAW_POSE", "INTERP_SHORT", "DIAGNOSTIC_LONG_HOLD"])},
        mapping={"video_qpc_ns": np.array([1_000_000_000, 1_040_000_000, 1_080_000_000])})
    validation = {"clean_protected_changed_pixels_per_frame": [0] * count,
                  "clean_outside_changed_pixels_per_frame": [0] * count}
    rows = {"final_protected_changed_pixels": [0] * count,
            "feather_restore_pixels": [0] * count, "robot_pixels": [0, 1, 1]}
    states = render_stage._frame_states(session, motion, masks, validation, rows,
                                      [{"candidate": False}] * count)
    assert states[1]["hands"]["left"]["qpc_velocity_violation"]
    assert states[1]["hands"]["left"]["position_violation"]
    assert states[1]["hands"]["right"]["rotation_violation"]
    assert states[1]["hands"]["right"]["tracking_gate_applicable"] is False
    invalid = states[2]["hands"]["left"]
    assert invalid["tracking_error_mm"] is None and invalid["rotation_deg"] is None
    assert invalid["position_violation"] is None and invalid["rotation_violation"] is None
    assert invalid["tracking_error_valid"] is False and invalid["diagnostic_display"] is True
    assert states[2]["hands"]["right"]["qpc_acceleration_violation"]
    abnormal = render_stage._abnormal_evidence(session.qpc_ns, states, masks, rows)
    assert abnormal["abnormal_raw_frame_indices"] == [0, 1, 2]
    assert abnormal["required_abnormal_windows"][0]["abnormal_frames"] == [0, 1, 2]
    assert "ROBOT_EMPTY_WHILE_WEARER_VISIBLE" in states[0]["abnormal_reasons"]
    assert "CUP_WEARER_SEMANTIC_CONFLICT" in states[1]["abnormal_reasons"]
    json.dumps(states, default=render_stage._json_default, allow_nan=False)


def test_render_rejects_wrong_qpc_flag_shape_before_models():
    rd, session, motion, raw, masks = _render_contract()
    motion["qpc_velocity_violation"] = np.zeros((2, 2, 7), bool)
    with pytest.raises(ValueError, match="qpc_velocity_violation"):
        rd._validate_inputs(session, motion, raw, masks)
