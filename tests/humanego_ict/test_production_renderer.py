"""Synthetic CPU contracts only: no real asset load, IK, model or NAS read."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib

import numpy as np
import pytest

from chaoyang.humanego_ict import production_renderer as module


def snapshot(frame=32, anchor=10_000_000_000):
    hand = np.zeros((21, 3))
    hand[0] = (0., 0., 1.)
    for start, x in ((1, -.05), (5, -.03), (9, 0.), (13, .03), (17, .05)):
        for offset in range(4):
            hand[start + offset] = (x, .03 + offset * .02, 1.)
    hands = np.stack((hand, hand + np.array([.2, 0., 0.])))
    return {"session_id": "synthetic_001", "frame_id": frame, "anchor_qpc_ns": anchor,
        "raw_rgb_frame_id": frame, "raw_rgb_array_sha256": hashlib.sha256(raw_image().tobytes()).hexdigest(),
        "timestamp_unit": "nanoseconds", "clock_domain": "SYNTHETIC_HOST_QPC_NS",
        "camera_pose_direction": "p_world = c2w @ p_camera",
        "hand_coordinate_frame": "WORLD_METRES",
        "head_source_qpc_ns": anchor - 2_000_000,
        "controller_source_qpc_ns": np.array([anchor - 2_000_000] * 2, np.int64),
        "manus_source_qpc_ns": np.array([anchor - 3_000_000, anchor - 4_000_000], np.int64),
        "head_valid": True, "controller_valid": np.ones(2, bool), "hand_valid": np.ones(2, bool),
        "source_signature": "1" * 64,
        "actual_input_refs": [{"path": "/path/to/chaoyang/synthetic_not_read.jsonl",
                               "sha256": "2" * 64, "bytes": 123}],
        "c2w": np.eye(4), "K": np.array([[575., 0., 319.5], [0., 575., 239.5], [0., 0., 1.]]),
        "gravity": np.array([0., 0., -1.]), "hands_world": hands}


class FakeBackend:
    synthetic_fixture = True

    def __init__(self):
        self.base_calls = 0
        self.solve_calls = 0
        self.previous = []
        self.render_calls = 0
        self.empty_alpha = False
        self.bad_layer = False
        self.bad_mask = False
        self.bad_arm = False

    def prepare_base(self, value):
        self.base_calls += 1
        base = np.eye(4)
        base[0, 3] = value["frame_id"] * .001
        return base, {"fixture": True, "first_frame_only": value["frame_id"]}

    def solve(self, value, base, previous_arm):
        self.solve_calls += 1
        self.previous.append(None if previous_arm is None else previous_arm.copy())
        q = np.full((2, 7), value["frame_id"] * .001)
        if self.bad_arm:
            q[0, 0] = np.nan
        return q, np.zeros((2, 22)), {"numeric_tracking_status": "FAILED",
                                    "position_mm": 90., "fixture": True}

    def current_mask(self, value, shape):
        mask = np.zeros(shape, bool)
        mask[90:100, 100:120] = True
        return mask.astype(np.uint8) if self.bad_mask else mask

    def render(self, arm, hand, camera_from_base, k, shape):
        self.render_calls += 1
        layer = np.full((*shape, 3), (20, 80, 220), dtype=np.uint8)
        alpha = np.zeros(shape, bool)
        if not self.empty_alpha:
            alpha[100:110, 110:130] = True
        return layer.astype(float) if self.bad_layer else layer, alpha


def raw_image():
    pixels = np.full((480, 640, 3), 100, dtype=np.uint8)
    pixels[94:97, 104:108] = (40, 50, 60)
    return pixels


def producer(backend=None):
    return module.CausalRobotizedProducer("/path/to/chaoyang",
        backend=FakeBackend() if backend is None else backend, allow_synthetic_fixture=True)


def test_anchor_pixels_and_degraded_qualification():
    image, value = raw_image(), snapshot()
    before = image.copy()
    result = producer().produce_anchor(image, value)
    assert result.rgb.shape == (480, 640, 3)
    assert result.rgb.dtype == np.uint8
    assert np.array_equal(image, before)
    assert np.array_equal(result.clean[~result.mask], before[~result.mask])
    assert result.rgb[104, 120].tolist() == [20, 80, 220]
    assert result.diagnostics["motion"]["numeric_tracking_status"] == "FAILED"
    assert result.diagnostics["geometry_decision"] == "UNKNOWN"
    assert result.diagnostics["complete_product_quality_pass"] is False
    assert result.diagnostics["control_ground_truth"] is False
    assert result.diagnostics["physical_deployment_authorized"] is False
    assert result.diagnostics["human_review_status"] == "NOT_REVIEWED"
    assert result.diagnostics["synthetic_fixture"] is True
    assert result.input_trace["donor_frame_ids"] == [32]
    assert len(result.input_trace["raw_rgb_array_sha256"]) == 64
    assert result.input_trace["producer_authorization"] is None


def test_first_base_frozen_and_past_only_warmstart():
    backend, value = FakeBackend(), snapshot()
    instance = producer(backend)
    first = instance.produce_anchor(raw_image(), value)
    second = instance.produce_anchor(raw_image(), snapshot(64, value["anchor_qpc_ns"] + 1_000_000_000))
    assert backend.base_calls == 1
    assert backend.previous[0] is None
    assert np.allclose(backend.previous[1], .032)
    assert second.input_trace["base_anchor"]["frame_id"] == 32
    assert first.input_trace["base_sha256"] == second.input_trace["base_sha256"]
    assert second.input_trace["donor_frame_ids"] == [64]


def test_sessions_are_independent():
    backend, instance = FakeBackend(), producer()
    instance = producer(backend)
    first = snapshot()
    second = snapshot(64)
    second["session_id"] = "synthetic_002"
    instance.produce_anchor(raw_image(), first)
    instance.produce_anchor(raw_image(), second)
    assert backend.base_calls == 2
    assert backend.previous == [None, None]


@pytest.mark.parametrize("key", ["head_source_qpc_ns", "controller_source_qpc_ns", "manus_source_qpc_ns"])
def test_future_sensor_rejected_before_compute(key):
    value, backend = snapshot(), FakeBackend()
    if key == "head_source_qpc_ns":
        value[key] = value["anchor_qpc_ns"] + 1
    else:
        value[key][1] = value["anchor_qpc_ns"] + 1
    with pytest.raises(module.ProductionRenderError, match="SOURCE_AFTER_ANCHOR"):
        producer(backend).produce_anchor(raw_image(), value)
    assert backend.base_calls == backend.solve_calls == backend.render_calls == 0


def test_source_age_gate_preserved():
    value = snapshot()
    value["head_source_qpc_ns"] = value["anchor_qpc_ns"] - 40_000_001
    with pytest.raises(module.ProductionRenderError, match="SOURCE_AGE_GATE_FAILED"):
        producer().produce_anchor(raw_image(), value)
    value["head_source_qpc_ns"] += 1
    module.validate_replay_snapshot(value)


@pytest.mark.parametrize("key,value,code", [
    ("frame_id", True, "INTEGER_REQUIRED"),
    ("frame_id", -1, "INTEGER_REQUIRED"),
    ("anchor_qpc_ns", 1.5, "INTEGER_REQUIRED"),
    ("anchor_qpc_ns", 2**63, "INTEGER_REQUIRED"),
    ("timestamp_unit", "ticks", "TIMESTAMP_UNIT_UNQUALIFIED"),
    ("clock_domain", "", "CLOCK_DOMAIN_REQUIRED"),
    ("camera_pose_direction", "camera_from_world", "CAMERA_DIRECTION_REQUIRED"),
    ("hand_coordinate_frame", "LOCAL", "HAND_FRAME_REQUIRED"),
    ("source_signature", "not-a-sha", "SOURCE_SHA_INVALID"),
    ("hand_valid", [True, False], "HAND_ANCHOR_INVALID"),
    ("controller_valid", [True, False], "CONTROLLER_HEAD_ANCHOR_INVALID"),
    ("head_valid", False, "CONTROLLER_HEAD_ANCHOR_INVALID"),
    ("head_valid", 1, "CONTROLLER_HEAD_ANCHOR_INVALID"),
    ("actual_input_refs", [], "ACTUAL_INPUT_REFS_REQUIRED"),
])
def test_bad_contract_fields_rejected(key, value, code):
    current = snapshot()
    current[key] = value
    with pytest.raises(module.ProductionRenderError, match=code):
        producer().produce_anchor(raw_image(), current)


@pytest.mark.parametrize("key,value,code", [
    ("c2w", np.zeros((4, 4)), "NONRIGID"),
    ("K", np.zeros((3, 3)), "INTRINSIC_INVALID"),
    ("gravity", np.zeros(3), "GRAVITY_INVALID"),
    ("hands_world", np.zeros((2, 21, 3)), "DEGENERATE_HAND_GEOMETRY"),
    ("hands_world", np.full((2, 21, 3), np.nan), "ARRAY_INVALID"),
])
def test_invalid_geometry_rejected(key, value, code):
    current = snapshot()
    current[key] = value
    with pytest.raises(module.ProductionRenderError, match=code):
        producer().produce_anchor(raw_image(), current)


@pytest.mark.parametrize("change,code", [
    ({"frame_id": 32, "anchor_qpc_ns": 11_000_000_000}, "NOT_STRICTLY_INCREASING"),
    ({"frame_id": 64, "anchor_qpc_ns": 10_000_000_000}, "NOT_STRICTLY_INCREASING"),
    ({"source_signature": "3" * 64}, "SOURCE_SIGNATURE_DRIFT"),
    ({"clock_domain": "OTHER_CLOCK"}, "SOURCE_SIGNATURE_DRIFT"),
    ({"K": np.array([[600., 0., 319.5], [0., 600., 239.5], [0., 0., 1.]])}, "FIXED_CAMERA_PRIOR_DRIFT"),
])
def test_session_drift_and_order_rejected(change, code):
    instance = producer()
    instance.produce_anchor(raw_image(), snapshot())
    value = snapshot(change.get("frame_id", 64), change.get("anchor_qpc_ns", 11_000_000_000))
    value.update(change)
    with pytest.raises(module.ProductionRenderError, match=code):
        instance.produce_anchor(raw_image(), value)


def test_session_override_cannot_change_identity():
    with pytest.raises(module.ProductionRenderError, match="SESSION_ID_MISMATCH"):
        producer().produce_anchor(raw_image(), snapshot(), session_id="not_this_session")


@pytest.mark.parametrize("flag,code", [
    ("bad_arm", "ARRAY_INVALID:q_arm"), ("bad_mask", "CURRENT_MASK_DOMAIN_INVALID"),
    ("bad_layer", "ROBOT_LAYER_DOMAIN_INVALID"), ("empty_alpha", "ROBOT_LAYER_ABSENT"),
])
def test_bad_compute_result_never_cached_or_admitted(flag, code):
    backend = FakeBackend()
    setattr(backend, flag, True)
    instance = producer(backend)
    with pytest.raises(module.ProductionRenderError, match=code):
        instance.produce_anchor(raw_image(), snapshot())
    assert instance._sessions == {}


def test_unfrozen_recipe_and_unqualified_backend_rejected():
    recipe = dict(module.DEFAULT_RECIPE, inpaint_radius_px=7)
    with pytest.raises(module.ProductionRenderError, match="UNFROZEN_RECIPE"):
        module.CausalRobotizedProducer("/mnt", recipe=recipe)
    with pytest.raises(module.ProductionRenderError, match="BACKEND_INJECTION_FIXTURE_ONLY"):
        module.CausalRobotizedProducer("/mnt", backend=FakeBackend())
    with pytest.raises(module.ProductionRenderError, match="AUTHORIZER_REQUIRED"):
        module.CausalRobotizedProducer("/mnt")


def grant():
    return {"execution_mode": "HUMANEGO_ICT_PRODUCTION_PREPARE_CPU",
        "cpu_renderer_allowed": True, "live_writer_verified": True,
        "task_id": "SYNTHETIC_AUTHORIZATION_NOT_REAL", "run_signature": "4" * 64,
        "deadline_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "writer": {"pid": 1, "proc_start_ticks": 1, "executor_epoch": 1, "fencing_token": "fixture"}}


def test_cpu_preparation_cannot_load_assets(monkeypatch):
    value = grant()
    value["execution_mode"] = "CPU_DEVELOPMENT_ONLY"
    calls = []
    monkeypatch.setattr(module, "_FrozenRobotBackend", lambda _repo: calls.append("FORBIDDEN"))
    instance = module.CausalRobotizedProducer("/path/to/chaoyang", runtime_authorizer=lambda: value)
    with pytest.raises(module.ProductionRenderError, match="AUTHORIZATION_REJECTED"):
        instance.produce_anchor(raw_image(), snapshot())
    assert calls == []


def test_authorizer_expiry_rechecked_before_solver(monkeypatch):
    value, backend, calls = grant(), FakeBackend(), []
    monkeypatch.setattr(module, "_FrozenRobotBackend", lambda _repo: backend)

    def authorize():
        calls.append(1)
        if len(calls) == 3:
            value["deadline_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        return value

    instance = module.CausalRobotizedProducer("/path/to/chaoyang", runtime_authorizer=authorize)
    with pytest.raises(module.ProductionRenderError, match="DEADLINE_EXPIRED"):
        instance.produce_anchor(raw_image(), snapshot())
    assert backend.base_calls == 1 and backend.solve_calls == 0
    assert instance._sessions == {}


def test_runtime_binding_cannot_hot_switch(monkeypatch):
    value, backend, calls = grant(), FakeBackend(), []
    monkeypatch.setattr(module, "_FrozenRobotBackend", lambda _repo: backend)

    def authorize():
        calls.append(1)
        if len(calls) == 3:
            value["writer"]["fencing_token"] = "other-writer"
        return deepcopy(value)

    instance = module.CausalRobotizedProducer("/path/to/chaoyang", runtime_authorizer=authorize)
    with pytest.raises(module.ProductionRenderError, match="AUTHORIZATION_SIGNATURE_DRIFT"):
        instance.produce_anchor(raw_image(), snapshot())
    assert backend.solve_calls == 0


def test_single_frame_adapter_contains_no_future_or_fps():
    value = module.validate_replay_snapshot(snapshot())
    one = module._single_frame_session(value)
    assert one.frame_count == 1 and not hasattr(one, "fps")
    assert one.hands.shape == (2, 1, 21, 3) and one.c2w.shape == (1, 4, 4)
    assert not one.body_observed.any()
    assert np.allclose(one.camera_points(one.hands[0, 0], 0), value["hands_world"][0])


def test_large_integer_qpc_is_not_rounded_through_float():
    anchor = 2**62 + 123
    current = snapshot(anchor=anchor)
    current["head_source_qpc_ns"] = anchor + 1
    with pytest.raises(module.ProductionRenderError, match="SOURCE_AFTER_ANCHOR"):
        module.validate_replay_snapshot(current)


def test_wrong_raw_shape_rejected_before_backend():
    backend = FakeBackend()
    with pytest.raises(module.ProductionRenderError, match="RAW_PIXEL_DOMAIN_INVALID"):
        producer(backend).produce_anchor(np.zeros((240, 320, 3), np.uint8), snapshot())
    assert backend.base_calls == 0


def test_wrong_anchor_pixels_rejected_before_backend():
    backend, image = FakeBackend(), raw_image()
    image[0, 0, 0] += 1
    with pytest.raises(module.ProductionRenderError, match="SELECTED_ANCHOR_SHA_MISMATCH"):
        producer(backend).produce_anchor(image, snapshot())
    assert backend.base_calls == backend.solve_calls == backend.render_calls == 0


def test_wrong_raw_frame_id_rejected_before_backend():
    backend, value = FakeBackend(), snapshot()
    value["raw_rgb_frame_id"] += 1
    with pytest.raises(module.ProductionRenderError, match="SELECTED_ANCHOR_ID_MISMATCH"):
        producer(backend).produce_anchor(raw_image(), value)
    assert backend.base_calls == backend.solve_calls == backend.render_calls == 0


def test_opencv_worker_threads_capped_at_two():
    producer()
    assert module.cv2.getNumThreads() == 2
