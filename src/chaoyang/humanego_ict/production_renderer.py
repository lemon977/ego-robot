"""Current/past-only Robotized training-anchor candidate, never the V9 video.

This module is deliberately not an operation or a publisher. The registered
consumer verifies original source references and writes/admits the returned
pixels. No historical Mask/Clean/q/base arrays, SAM, ProPainter, SDK or network
are read here. Full-resolution videos and physical robot control are out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import cv2
import numpy as np


SCHEMA = "causal-current-frame-robotized-anchor-v1"
DEFAULT_RECIPE = {
    "schema": SCHEMA,
    "width": 640,
    "height": 480,
    "inpaint_method": "OPENCV_TELEA_CURRENT_FRAME_ONLY",
    "inpaint_radius_px": 3,
    "mask_rule": "CURRENT_PROJECTED_HAND_FOREARM_TRACKER_EXISTING_CONSTANTS",
    "base_rule": "FIRST_LEGAL_ANCHOR_EXISTING_SEARCH_THEN_FREEZE",
    "arm_rule": "CURRENT_ANCHOR_PAST_WARMSTART_NO_SAVGOL",
    "hand_rule": "CURRENT_ANCHOR_EXISTING_SINGLE_FRAME_SOLVER",
    "display_policy": "ROBOT_OVER_SPATIAL_INPAINT_GEOMETRY_UNKNOWN",
    "complete_robot": True,
}
SOURCE_TIME_KEYS = ("head", "controller_left", "controller_right", "manus_left", "manus_right")
MAX_SOURCE_AGE_NS = 40_000_000  # Unchanged native source gate.
MAX_SIGNED_INT64 = int(np.iinfo(np.int64).max)


class ProductionRenderError(ValueError):
    """Explicit whole-anchor rejection; there is no neutral/future fallback."""


def _signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _sha(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ProductionRenderError(f"RGB_SOURCE_SHA_INVALID:{name}")
    return value


def _integer(value: Any, name: str) -> int:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not 0 <= int(value) <= MAX_SIGNED_INT64):
        raise ProductionRenderError(f"RGB_INTEGER_REQUIRED:{name}")
    return int(value)


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.shape != shape or array.dtype.kind not in "fi"
            or not np.isfinite(array).all()):
        raise ProductionRenderError(f"RGB_ARRAY_INVALID:{name}")
    return array.astype(np.float64, copy=True)


def _rigid(value: Any, name: str) -> np.ndarray:
    pose = _array(value, (4, 4), name)
    r = pose[:3, :3]
    if (not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-8)
            or not np.allclose(r.T @ r, np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(r), 1., atol=2e-5)):
        raise ProductionRenderError(f"RGB_NONRIGID:{name}")
    return pose


def validate_replay_snapshot(replay: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only an explicitly selected native past/current snapshot.

    The original-file SHA verification is performed by the registered source
    consumer, not replaced by this small structural/causal validation.
    """
    if not isinstance(replay, Mapping):
        raise ProductionRenderError("RGB_REPLAY_SNAPSHOT_REQUIRED")
    result = dict(replay)
    sid = replay.get("session_id")
    if not isinstance(sid, str) or not sid.strip():
        raise ProductionRenderError("RGB_SESSION_ID_REQUIRED")
    result["frame_id"] = _integer(replay.get("frame_id"), "frame_id")
    if _integer(replay.get("raw_rgb_frame_id"), "raw_rgb_frame_id") != result["frame_id"]:
        raise ProductionRenderError("RGB_RAW_SELECTED_ANCHOR_ID_MISMATCH")
    result["raw_rgb_array_sha256"] = _sha(replay.get("raw_rgb_array_sha256"), "raw_rgb_array_sha256")
    anchor = result["anchor_qpc_ns"] = _integer(replay.get("anchor_qpc_ns"), "anchor_qpc_ns")
    if replay.get("timestamp_unit") != "nanoseconds":
        raise ProductionRenderError("RGB_TIMESTAMP_UNIT_UNQUALIFIED")
    clock = replay.get("clock_domain")
    if not isinstance(clock, str) or not clock.strip():
        raise ProductionRenderError("RGB_CLOCK_DOMAIN_REQUIRED")
    if replay.get("camera_pose_direction") != "p_world = c2w @ p_camera":
        raise ProductionRenderError("RGB_CAMERA_DIRECTION_REQUIRED")
    if replay.get("hand_coordinate_frame") != "WORLD_METRES":
        raise ProductionRenderError("RGB_HAND_FRAME_REQUIRED")
    head_time = _integer(replay.get("head_source_qpc_ns"), "head_source_qpc_ns")
    selected_times = {"head": head_time}
    for input_key, output_keys in (
        ("controller_source_qpc_ns", ("controller_left", "controller_right")),
        ("manus_source_qpc_ns", ("manus_left", "manus_right")),
    ):
        times = np.asarray(replay.get(input_key))
        if times.shape != (2,) or times.dtype.kind not in "iu":
            raise ProductionRenderError(f"RGB_SENSOR_TIMES_REQUIRED:{input_key}")
        selected_times.update({key: _integer(value, key) for key, value in zip(output_keys, times, strict=True)})
    if any(qpc > anchor for qpc in selected_times.values()):
        raise ProductionRenderError("RGB_SENSOR_SOURCE_AFTER_ANCHOR")
    if any(anchor - qpc > MAX_SOURCE_AGE_NS for qpc in selected_times.values()):
        raise ProductionRenderError("RGB_SENSOR_SOURCE_AGE_GATE_FAILED")
    result["sensor_source_qpc_ns"] = selected_times
    result["source_signature"] = _sha(replay.get("source_signature"), "source_signature")
    refs = replay.get("actual_input_refs")
    if not isinstance(refs, (list, tuple)) or not refs:
        raise ProductionRenderError("RGB_ACTUAL_INPUT_REFS_REQUIRED")
    for index, ref in enumerate(refs):
        if (not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str)
                or not Path(ref["path"]).is_absolute()):
            raise ProductionRenderError("RGB_ACTUAL_INPUT_REF_INVALID")
        _integer(ref.get("bytes"), f"ref_bytes_{index}")
        _sha(ref.get("sha256"), f"ref_sha_{index}")
    result["actual_input_refs"] = [dict(ref) for ref in refs]
    result["c2w"] = _rigid(replay.get("c2w"), "c2w")
    k = result["K"] = _array(replay.get("K"), (3, 3), "K")
    if min(k[0, 0], k[1, 1]) <= 0 or not np.allclose(k[2], (0, 0, 1), atol=1e-8):
        raise ProductionRenderError("RGB_INTRINSIC_INVALID")
    gravity = result["gravity"] = _array(replay.get("gravity"), (3,), "gravity")
    if np.linalg.norm(gravity) < 1e-8:
        raise ProductionRenderError("RGB_GRAVITY_INVALID")
    hands = result["hands_world"] = _array(replay.get("hands_world"), (2, 21, 3), "hands_world")
    valid = np.asarray(replay.get("hand_valid"))
    if valid.shape != (2,) or valid.dtype != np.bool_ or not valid.all():
        raise ProductionRenderError("RGB_NATIVE_HAND_ANCHOR_INVALID")
    controller_valid = np.asarray(replay.get("controller_valid"))
    if (controller_valid.shape != (2,) or controller_valid.dtype != np.bool_
            or not controller_valid.all() or not isinstance(replay.get("head_valid"), (bool, np.bool_))
            or not replay["head_valid"]):
        raise ProductionRenderError("RGB_NATIVE_CONTROLLER_HEAD_ANCHOR_INVALID")
    for hand in hands:
        if (np.linalg.norm(hand[9] - hand[0]) < 1e-8
                or np.linalg.norm(np.cross(hand[5] - hand[0], hand[17] - hand[0])) < 1e-10):
            raise ProductionRenderError("RGB_DEGENERATE_HAND_GEOMETRY")
    return result


def _single_frame_session(snapshot: Mapping[str, Any]) -> Any:
    """No old PicoSession/source loader is invoked; all arrays have length one."""
    c2w = snapshot["c2w"]
    w2c = np.linalg.inv(c2w)
    k = snapshot["K"]
    session = SimpleNamespace(frame_count=1, width=640, height=480,
        K=k, c2w=c2w[None], gravity=snapshot["gravity"],
        hands=snapshot["hands_world"][:, None],
        hand_observed=np.ones((2, 1), dtype=bool),
        hand_source=np.full((2, 1), "CURRENT_PAST_NATIVE", dtype="U32"),
        body=np.zeros((1, 24, 3)), body_observed=np.zeros(1, dtype=bool))
    # There is no fps, interpolation, smoothing or synthesized source clock.
    session.camera_points = lambda points, _frame: points @ w2c[:3, :3].T + w2c[:3, 3]
    session.project = lambda points: (points @ k.T)[..., :2] / (points @ k.T)[..., 2:3]
    return session


class _FrozenRobotBackend:
    synthetic_fixture = False

    def __init__(self, repo_root: Path):
        # Lazy: CPU preparation imports do not load assets or invoke IK/render.
        from chaoyang.exact78_baseline import mask, retarget, render
        from chaoyang.exact78_baseline.robot_model import Robot
        self.mask_module, self.retarget = mask, retarget
        self.robot = Robot(repo_root)
        self.renderer = render.RobotRenderer(self.robot)

    def prepare_base(self, snapshot):
        session = _single_frame_session(snapshot)
        roots = self.retarget.hand_root_targets(session.hands, self.robot)
        base, report = self.retarget.place_base(self.robot, session.gravity,
            session.c2w, session.body, session.body_observed, roots)
        return base, report

    def solve(self, snapshot, base, previous_arm):
        session = _single_frame_session(snapshot)
        roots_world = self.retarget.hand_root_targets(session.hands, self.robot)
        roots_base = np.einsum("ij,snjk->snik", np.linalg.inv(base), roots_world)
        hints = self.retarget.elbow_hints(session.body, session.body_observed,
                                       base, self.robot, roots_base)
        arm, hand = np.zeros((2, 7)), np.zeros((2, 22))
        errors = []
        for side in range(2):
            if previous_arm is None:
                arm[side] = self.retarget.solve_arm_track(self.robot, side,
                    roots_base[side], hints[side])[0]
            else:
                arm[side] = self.retarget.solve_arm_frame(self.robot, side,
                    roots_base[side, 0], hints[side, 0], previous_arm[side],
                    previous_arm[side])
            hand[side] = self.retarget.solve_hand_track(self.robot, side,
                                                      session.hands[side])[0][0]
            p, r = self.retarget.arm_errors(self.robot, side, arm[side], roots_base[side, 0])
            errors.append({"position_mm": p, "rotation_deg": r,
                "numeric_tracking_status": "PASSED" if (
                    p <= self.retarget.TRACK_POSITION_MM and r <= self.retarget.TRACK_ROTATION_DEG
                ) else "FAILED"})
        return arm, hand, {"per_side_tracking": errors,
            "track_position_mm_gate": self.retarget.TRACK_POSITION_MM,
            "track_rotation_deg_gate": self.retarget.TRACK_ROTATION_DEG,
            "complete_path_speed_acceleration_qualification": "UNKNOWN_ANCHOR_ONLY_NOT_FULL_PATH"}

    def current_mask(self, snapshot, shape):
        geometry = self.mask_module.pico_geometry(_single_frame_session(snapshot), 1.)
        mask = np.zeros(shape, dtype=bool)
        for side in range(2):
            mask |= self.mask_module._hull_mask(geometry["hands"][side, 0], shape,
                                               self.mask_module.HAND_HULL_PAD_PX)
            mask |= self.mask_module._capsule_mask(geometry["forearm"][side, 0],
                geometry["forearm_radius"][side, 0], shape)
            mask |= self.mask_module._hull_mask(geometry["tracker"][side, 0], shape,
                                               self.mask_module.BAND_PAD_PX)
        return mask

    def render(self, arm, hand, camera_from_base, k, shape):
        bgr, alpha, _roots = self.renderer.render(arm, hand, camera_from_base, k,
            complete_robot=True, width=shape[1], height=shape[0])
        return bgr[..., ::-1].copy(), alpha


@dataclass(frozen=True)
class RobotizedAnchorResult:
    rgb: np.ndarray
    mask: np.ndarray
    clean: np.ndarray
    robot_alpha: np.ndarray
    diagnostics: dict[str, Any]
    input_trace: dict[str, Any]


class CausalRobotizedProducer:
    """Sequential per-session anchor producer with frozen first-anchor placement.

    A real backend needs a fresh registered-runtime authorizer. Synthetic
    injected backends are only for explicitly marked CPU tests and can never
    receive a non-fixture admission from this module.
    """
    def __init__(self, repo_root: Path, *, recipe: Mapping[str, Any] | None = None,
                 runtime_authorizer: Callable[[], Any] | None = None,
                 backend: Any = None, allow_synthetic_fixture: bool = False):
        cv2.setNumThreads(2)
        self.repo_root = Path(repo_root)
        self.recipe = dict(DEFAULT_RECIPE if recipe is None else recipe)
        if self.recipe != DEFAULT_RECIPE:
            raise ProductionRenderError("RGB_UNFROZEN_RECIPE")
        self.recipe_sha256 = _signature(self.recipe)
        if backend is not None and (not allow_synthetic_fixture
                                   or getattr(backend, "synthetic_fixture", False) is not True):
            raise ProductionRenderError("RGB_BACKEND_INJECTION_FIXTURE_ONLY")
        if backend is None and not callable(runtime_authorizer):
            raise ProductionRenderError("RGB_REGISTERED_RUNTIME_AUTHORIZER_REQUIRED")
        self._backend, self._authorize = backend, runtime_authorizer
        self._fixture = backend is not None
        self._sessions: dict[str, dict[str, Any]] = {}
        self._execution_binding: dict[str, Any] | None = None

    def _guard(self):
        if not self._fixture:
            grant = self._authorize()  # Rechecks the live writer using the real /proc view.
            if (not isinstance(grant, Mapping)
                    or grant.get("execution_mode") != "HUMANEGO_ICT_PRODUCTION_PREPARE_CPU"
                    or grant.get("cpu_renderer_allowed") is not True
                    or grant.get("live_writer_verified") is not True
                    or not isinstance(grant.get("task_id"), str) or not grant["task_id"]):
                raise ProductionRenderError("RGB_REGISTERED_RUNTIME_AUTHORIZATION_REJECTED")
            _sha(grant.get("run_signature"), "runtime_run_signature")
            writer = grant.get("writer")
            if not isinstance(writer, Mapping):
                raise ProductionRenderError("RGB_RUNTIME_WRITER_UNQUALIFIED")
            for key in ("pid", "proc_start_ticks", "executor_epoch"):
                if _integer(writer.get(key), key) <= 0:
                    raise ProductionRenderError("RGB_RUNTIME_WRITER_UNQUALIFIED")
            if not isinstance(writer.get("fencing_token"), str) or not writer["fencing_token"]:
                raise ProductionRenderError("RGB_RUNTIME_WRITER_UNQUALIFIED")
            try:
                deadline = datetime.fromisoformat(grant["deadline_at"].replace("Z", "+00:00"))
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                raise ProductionRenderError("RGB_RUNTIME_DEADLINE_UNQUALIFIED") from error
            if deadline.tzinfo is None or datetime.now(timezone.utc) >= deadline:
                raise ProductionRenderError("RGB_RUNTIME_DEADLINE_EXPIRED")
            binding = {"task_id": grant["task_id"], "run_signature": grant["run_signature"],
                "writer": dict(writer), "deadline_at": grant["deadline_at"]}
            if self._execution_binding is not None and self._execution_binding != binding:
                raise ProductionRenderError("RGB_RUNTIME_AUTHORIZATION_SIGNATURE_DRIFT")
            self._execution_binding = binding

    def produce_anchor(self, raw_rgb: np.ndarray, replay: Mapping[str, Any], *,
                       session_id: str | None = None) -> RobotizedAnchorResult:
        snapshot = validate_replay_snapshot(replay)
        sid = snapshot["session_id"]
        if session_id is not None and sid != session_id:
            raise ProductionRenderError("RGB_SESSION_ID_MISMATCH")
        raw = np.asarray(raw_rgb)
        shape = (self.recipe["height"], self.recipe["width"])
        if raw.shape != (*shape, 3) or raw.dtype != np.uint8:
            raise ProductionRenderError("RGB_RAW_PIXEL_DOMAIN_INVALID")
        raw = raw.copy()
        raw_digest = hashlib.sha256(raw.tobytes()).hexdigest()
        if raw_digest != snapshot["raw_rgb_array_sha256"]:
            raise ProductionRenderError("RGB_RAW_SELECTED_ANCHOR_SHA_MISMATCH")
        existing = self._sessions.get(sid)
        if existing is not None:
            if (snapshot["frame_id"] <= existing["frame_id"]
                    or snapshot["anchor_qpc_ns"] <= existing["anchor_qpc_ns"]):
                raise ProductionRenderError("RGB_ANCHORS_NOT_STRICTLY_INCREASING")
            if (snapshot["source_signature"] != existing["source_signature"]
                    or snapshot["clock_domain"] != existing["clock_domain"]):
                raise ProductionRenderError("RGB_SESSION_SOURCE_SIGNATURE_DRIFT")
            if (not np.array_equal(snapshot["K"], existing["K"])
                    or not np.array_equal(snapshot["gravity"], existing["gravity"])):
                raise ProductionRenderError("RGB_SESSION_FIXED_CAMERA_PRIOR_DRIFT")
        self._guard()
        if self._backend is None:
            self._backend = _FrozenRobotBackend(self.repo_root)
        self._guard()
        if existing is None:
            base, base_report = self._backend.prepare_base(snapshot)
            base = _rigid(base, "first_anchor_base")
            base_anchor = {"frame_id": snapshot["frame_id"],
                "anchor_qpc_ns": snapshot["anchor_qpc_ns"],
                "sensor_source_qpc_ns": snapshot["sensor_source_qpc_ns"],
                "actual_input_refs": snapshot["actual_input_refs"],
                "source_signature": snapshot["source_signature"]}
            previous = None
        else:
            base, base_report, base_anchor = (existing[key] for key in ("base", "base_report", "base_anchor"))
            previous = existing["q_arm"].copy()
        self._guard()
        arm, hand, motion_report = self._backend.solve(snapshot, base.copy(), previous)
        arm = _array(arm, (2, 7), "q_arm")
        hand = _array(hand, (2, 22), "q_hand")
        self._guard()
        removal = np.asarray(self._backend.current_mask(snapshot, shape))
        if removal.shape != shape or removal.dtype != np.bool_:
            raise ProductionRenderError("RGB_CURRENT_MASK_DOMAIN_INVALID")
        clean = cv2.inpaint(raw, removal.astype(np.uint8) * 255,
                           self.recipe["inpaint_radius_px"], cv2.INPAINT_TELEA)
        if not np.array_equal(clean[~removal], raw[~removal]):
            raise ProductionRenderError("RGB_INPAINT_CHANGED_OUTSIDE_CURRENT_MASK")
        self._guard()
        layer, alpha = self._backend.render(arm.copy(), hand.copy(),
            np.linalg.inv(snapshot["c2w"]) @ base, snapshot["K"].copy(), shape)
        layer, alpha = np.asarray(layer), np.asarray(alpha)
        if layer.shape != raw.shape or layer.dtype != np.uint8 or alpha.shape != shape or alpha.dtype != np.bool_:
            raise ProductionRenderError("RGB_ROBOT_LAYER_DOMAIN_INVALID")
        if not alpha.any():
            raise ProductionRenderError("RGB_ROBOT_LAYER_ABSENT")
        # Same existing 1px feather; no scene-depth/hidden-object truth claim.
        weight = cv2.GaussianBlur(alpha.astype(np.float32), (3, 3), .7)
        weight[alpha] = np.maximum(weight[alpha], .5)
        rgb = np.clip(clean.astype(np.float32) * (1 - weight[..., None])
                      + layer.astype(np.float32) * weight[..., None], 0, 255).astype(np.uint8)
        self._guard()
        trace = {"schema": SCHEMA, "session_id": sid, "source_frame_id": snapshot["frame_id"],
            "anchor_qpc_ns": snapshot["anchor_qpc_ns"], "timestamp_unit": "nanoseconds",
            "clock_domain": snapshot["clock_domain"],
            "sensor_source_qpc_ns": snapshot["sensor_source_qpc_ns"],
            "actual_input_refs": snapshot["actual_input_refs"],
            "source_signature": snapshot["source_signature"],
            "raw_rgb_array_sha256": raw_digest,
            "raw_rgb_array_domain": {"shape": list(raw.shape), "dtype": "uint8", "colour_order": "RGB"},
            "donor_frame_ids": [snapshot["frame_id"]],
            "recipe_sha256": self.recipe_sha256, "base_anchor": base_anchor,
            "base_sha256": _signature(base.tolist()), "synthetic_fixture": self._fixture,
            "producer_authorization": self._execution_binding,
            "motion_inputs": "CURRENT_NATIVE_PLUS_PRIOR_ACCEPTED_ANCHOR_SOLUTION",
            "mask_inputs": "CURRENT_NATIVE_PROJECTED_GEOMETRY_ONLY",
            "clean_inputs": "CURRENT_RAW_RGB_SPATIAL_ONLY_NO_TEMPORAL_DONORS"}
        diagnostics = {"render_execution_status": "COMPLETE", "visual_quality_status": "UNREVIEWED",
            "training_rgb_input_class": "CAUSAL_CANDIDATE_REQUIRES_CONSUMER_ADMISSION",
            "rgb_domain": "FRESH_CAUSAL_SPATIAL_INPAINT_ROBOT_ANCHOR_NOT_V9_VIDEO",
            "geometry_decision": "UNKNOWN", "display_policy": self.recipe["display_policy"],
            "protected_visible_object_qualification": "UNKNOWN_NOT_IDENTIFIED",
            "depth_and_hidden_object_pixels_are_truth": False,
            "clean_quality": "DEGRADED_SPATIAL_FILL_REVIEW_NOT_DONE",
            "forearm_and_tracker_support": "EXISTING_MODEL_PRIOR_NOT_MEASURED_GEOMETRY",
            "robot_visible_pixel_count": int(alpha.sum()), "removal_pixel_count": int(removal.sum()),
            "mask_outside_raw_unchanged": True, "motion": motion_report,
            "base": base_report, "q_arm": arm.tolist(), "q_hand": hand.tolist(),
            "T_world_base": base.tolist(), "synthetic_fixture": self._fixture,
            "control_ground_truth": False, "physical_deployment_authorized": False,
            "complete_product_quality_pass": False, "human_review_status": "NOT_REVIEWED"}
        self._sessions[sid] = {"base": base, "base_report": base_report,
            "base_anchor": base_anchor, "q_arm": arm.copy(), "frame_id": snapshot["frame_id"],
            "anchor_qpc_ns": snapshot["anchor_qpc_ns"], "clock_domain": snapshot["clock_domain"],
            "source_signature": snapshot["source_signature"], "K": snapshot["K"].copy(),
            "gravity": snapshot["gravity"].copy()}
        return RobotizedAnchorResult(rgb, removal.copy(), clean, alpha.copy(), diagnostics, trace)
