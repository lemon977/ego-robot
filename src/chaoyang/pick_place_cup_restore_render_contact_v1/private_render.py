"""One-pass QPC video delivery using the frozen robot assets and renderer.

Only registered production callers may invoke render_session. Importing this
module does not initialize a renderer, codec, model, GPU, or output directory.
"""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from chaoyang.exact78_baseline import render as rd
from chaoyang.exact78_baseline.pipeline import _robot
from chaoyang.exact78_baseline.robot_model import FINGERS
from chaoyang.pick_place_1008 import source, visual

WIDTH, HEIGHT = 640, 480


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2,
                  default=_json_default, allow_nan=False)
        stream.write("\n")


def _validate_inputs(session, motion: Mapping, clean_rgb: np.ndarray,
                     masks: Mapping) -> tuple[np.ndarray, np.ndarray, dict]:
    count, shape = session.frame_count, (session.frame_count, HEIGHT, WIDTH)
    if "raw_rgb" not in masks:
        raise ValueError("RAW_LOSSLESS_ARRAY_REQUIRED")
    raw = visual._rgb_frames(masks["raw_rgb"])
    clean = visual._rgb_frames(clean_rgb)
    if raw.shape != (*shape, 3) or clean.shape != raw.shape:
        raise ValueError("RENDER_RAW_CLEAN_DOMAIN")
    source.timeline(session.qpc_ns, "render")
    if len(session.qpc_ns) != count:
        raise ValueError("RENDER_QPC_FRAME_COUNT")
    for name, expected in (("q_arm", (count, 2, 7)), ("q_hand", (count, 2, 22)),
                           ("T_world_base", (4, 4))):
        value = np.asarray(motion[name])
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"RENDER_MOTION_DOMAIN:{name}")
    for name in ("qpc_velocity_violation", "qpc_acceleration_violation"):
        visual._bool_masks(motion[name], (count, 2), name)
    for name in ("arm_position_mm", "arm_rotation_deg"):
        if np.asarray(motion[name]).shape != (2, count):
            raise ValueError(f"RENDER_TRACKING_ERROR_DOMAIN:{name}")
    if session.c2w.shape != (count, 4, 4) or not np.isfinite(session.c2w).all():
        raise ValueError("RENDER_CAMERA_DOMAIN")
    for name in ("core", "removal", "protected_visible", "unknown", "sam_core", "equipment"):
        visual._bool_masks(masks[name], shape, name)
    effective, validation = visual.effective_clean_masks(masks["removal"], masks["protected_visible"])
    if "effective_removal" in masks and not np.array_equal(
            visual._bool_masks(masks["effective_removal"], shape, "effective_removal"), effective):
        raise ValueError("RENDER_EFFECTIVE_MASK_SIGNATURE_MISMATCH")
    changed = np.any(raw != clean, axis=-1)
    outside = np.count_nonzero(changed & ~effective, axis=(1, 2))
    protected = np.count_nonzero(changed & masks["protected_visible"], axis=(1, 2))
    if outside.any() or protected.any():
        raise ValueError("RENDER_CLEAN_PIXEL_PROTECTION_FAILED")
    validation.update({"clean_outside_changed_pixels_per_frame": outside.tolist(),
                       "clean_protected_changed_pixels_per_frame": protected.tolist()})
    return raw, clean, validation


class _QpcVideoEncoder:
    """Streaming twin of source.encode_qpc_frames, with identical codec/PTS rules."""

    def __init__(self, path: Path, qpc_ns: np.ndarray, width: int, height: int):
        import av

        if path.exists():
            raise source.SourceContractError("RAW_INPUT_VIDEO_EXISTS")
        self.path, self.width, self.height = path, width, height
        self.pts = source.pts_from_qpc(qpc_ns)
        self.tick, self.count, self.closed = Fraction(1, 1_000_000), 0, False
        self.av = av
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=Fraction(30, 1))
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt, self.stream.time_base = "yuv420p", self.tick
        self.stream.codec_context.time_base = self.tick
        self.stream.codec_context.thread_count = 2
        self.stream.options = {"crf": "12", "preset": "fast", "bf": "0"}

    def write(self, pixels: np.ndarray) -> None:
        if (self.closed or self.count >= len(self.pts) or
                pixels.shape != (self.height, self.width, 3) or pixels.dtype != np.uint8):
            raise source.SourceContractError("RAW_VIDEO_FRAME_DOMAIN_OR_COUNT")
        frame = self.av.VideoFrame.from_ndarray(pixels, format="bgr24")
        frame.pts, frame.time_base = int(self.pts[self.count]), self.tick
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.count += 1

    def abort(self) -> None:
        if not self.closed:
            self.container.close()
            self.closed = True

    def close(self) -> dict:
        if self.closed:
            raise RuntimeError("QPC_ENCODER_ALREADY_CLOSED")
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
            self.closed = True
        if self.count != len(self.pts):
            raise source.SourceContractError(f"RAW_VIDEO_QPC_FRAME_COUNT:{self.count}:{len(self.pts)}")
        recovered = []
        with self.av.open(str(self.path)) as container:
            container.streams.video[0].codec_context.thread_count = 2
            for frame in container.decode(video=0):
                if (frame.width, frame.height) != (self.width, self.height) or frame.pts is None:
                    raise source.SourceContractError("RAW_ENCODED_FRAME_DOMAIN_OR_PTS")
                recovered.append(float(frame.pts * frame.time_base))
            recovered_tick = float(container.streams.video[0].time_base)
        if len(recovered) != len(self.pts):
            raise source.SourceContractError("RAW_ENCODED_FRAME_COUNT_MISMATCH")
        error = np.abs(np.asarray(recovered) - self.pts / 1e6)
        if error.max(initial=0.) > max(float(self.tick), recovered_tick) + 1e-9:
            raise source.SourceContractError("RAW_ENCODED_PTS_DRIFT")
        return {"time_base": "1/1000000", "quantization_max_ns": 500,
                "recovered_time_base_seconds": recovered_tick,
                "pts_roundtrip_max_error_seconds": float(error.max(initial=0.)),
                "decoded_frames": len(recovered),
                "frame_order": "RAW_VIDEO_PICTURE_ORDER_WITH_QPC_PTS",
                "source_equivalent_encoder": "source.encode_qpc_frames",
                "codec": "libx264", "crf": "12", "preset": "fast", "bf": "0",
                "thread_count": 2}


def _display_valid(session) -> np.ndarray:
    value = np.asarray(session.motion_valid.get("hand_display_valid", session.hand_observed), bool)
    if value.shape != (2, session.frame_count):
        raise ValueError("RENDER_HAND_DISPLAY_VALID_DOMAIN")
    return value


def _project_robot_tips(robot, q_arm, q_hand, camera, K, valid) -> np.ndarray:
    """Use the same FK, q and camera as the raster; output (2,5,2)."""
    tips = np.full((2, 5, 2), np.nan)
    for side in range(2):
        if not valid[side]:
            continue
        root = camera @ robot.arm_frames(side, q_arm[side])["root"]
        contract = robot.hands[side]
        for index, finger in enumerate(FINGERS):
            local = robot.finger_points(side, finger, q_hand[side][contract.groups[finger]])[-1]
            point = root[:3, :3] @ local + root[:3, 3]
            if np.isfinite(point).all() and point[2] > 0.05:
                projected = K @ point
                tips[side, index] = projected[:2] / projected[2]
    return tips


def _frame_states(session, motion, masks, validation, render_rows, candidate_rows) -> list[dict]:
    count = session.frame_count
    pts = source.pts_from_qpc(session.qpc_ns)
    display = _display_valid(session)
    states = []
    hand_sources = np.asarray(motion.get("hand_source", session.hand_source))
    if hand_sources.shape != (2, count):
        raise ValueError("RENDER_HAND_SOURCE_DOMAIN")
    for frame in range(count):
        side_rows = {}
        for side, name in enumerate(("left", "right")):
            position = float(motion["arm_position_mm"][side, frame])
            rotation = float(motion["arm_rotation_deg"][side, frame])
            position_valid, rotation_valid = bool(np.isfinite(position)), bool(np.isfinite(rotation))
            controllers = session.motion_valid.get("controller_valid")
            manus = session.motion_valid.get("manus_valid")
            side_rows[name] = {
                "source": str(hand_sources[side, frame]),
                "raw_observed": bool(session.hand_observed[side, frame]),
                "display_valid": bool(display[side, frame]),
                "wrist_position_valid": bool(controllers[frame, side]) if controllers is not None else False,
                "wrist_orientation_valid": bool(controllers[frame, side]) if controllers is not None else False,
                "handshape_valid": bool(manus[frame, side]) if manus is not None else False,
                "wrist_authority": "CONTROLLER_PLUS_INSTALLATION_PRIOR",
                "diagnostic_display": not bool(display[side, frame]),
                "tracking_error_mm": position if position_valid else None,
                "rotation_deg": rotation if rotation_valid else None,
                "tracking_error_valid": position_valid and rotation_valid,
                "tracking_gate_applicable": bool(session.hand_observed[side, frame]),
                "position_violation": bool(position > 20.0) if position_valid else None,
                "rotation_violation": bool(rotation > 15.0) if rotation_valid else None,
                "qpc_velocity_violation": bool(motion["qpc_velocity_violation"][frame, side]),
                "qpc_acceleration_violation": bool(motion["qpc_acceleration_violation"][frame, side]),
                "control_ground_truth": False,
            }
        row = {
            "frame_index": frame, "raw_frame_index": frame,
            "raw_qpc_ns": int(session.qpc_ns[frame]),
            "pts_us": int(pts[frame]),
            "hands": side_rows,
            "camera_raw_valid": bool(session.motion_valid.get("head_valid", np.zeros(count, bool))[frame]),
            "camera_source": str(session.native_motion.get("head_source", np.full(count, "UNAVAILABLE"))[frame]),
            "model_cup_protected_pixels": int(masks["protected_visible"][frame].sum()),
            "model_cup_observed_pixels": int(masks["cup_observed"][frame].sum()),
            "legacy_unknown_pixels": int(masks["unknown"][frame].sum()),
            "legacy_held_pixels": int(masks["held"][frame].sum()),
            "cup_identity_conflict_pixels": int(masks.get("semantic_conflict", np.zeros((1, 1, 1), bool))[frame].sum())
                if "semantic_conflict" in masks else 0,
            "occlusion_order": "UNKNOWN",
            "visual_contact_candidate": bool(candidate_rows[frame]["candidate"]),
            "physical_contact": "UNKNOWN", "physical_contact_ground_truth": False,
            "clean_protected_changed_pixels": validation["clean_protected_changed_pixels_per_frame"][frame],
            "clean_outside_changed_pixels": validation["clean_outside_changed_pixels_per_frame"][frame],
            "final_protected_changed_pixels": render_rows["final_protected_changed_pixels"][frame],
            "model_cup_protection_feather_restore_pixels": render_rows["feather_restore_pixels"][frame],
            "training_eligible": False, "control_ground_truth": False,
        }
        row["mapping"] = {name: np.asarray(value)[frame] for name, value in session.mapping.items()
                          if np.asarray(value).ndim > 0 and len(value) == count}
        states.append(row)
    return states


def _upstream_refs(repo: Path, masks: Mapping) -> dict:
    refs = masks.get("upstream_refs")
    roles = {"actions", "frame_mapping", "raw_rgb", "clean_rgb", "masks", "motion_info"}
    if not isinstance(refs, dict) or not roles <= refs.keys():
        raise ValueError("RENDER_UPSTREAM_REFERENCES_REQUIRED")
    result = {}
    for role in sorted(roles):
        pin = refs[role]
        if not isinstance(pin, dict):
            raise ValueError(f"RENDER_UPSTREAM_REFERENCE_INVALID:{role}")
        path = Path(pin["path"]).resolve(strict=True)
        if not path.is_relative_to(repo):
            raise ValueError(f"RENDER_UPSTREAM_OUTSIDE_PROJECT:{role}")
        actual = source.ref(path)
        if actual != pin:
            raise ValueError(f"RENDER_UPSTREAM_REFERENCE_DRIFT:{role}")
        result[role] = dict(pin)
    return result


def _abnormal_evidence(qpc_ns, states: list[dict], masks: Mapping, render_rows: Mapping) -> dict:
    evidence = {}
    for frame, row in enumerate(states):
        reasons = []
        for side, value in row["hands"].items():
            if value["qpc_velocity_violation"]:
                reasons.append(f"QPC_VELOCITY_GT_3_6_RAD_S:{side}")
            if value["qpc_acceleration_violation"]:
                reasons.append(f"QPC_ACCELERATION_GT_54_RAD_S2:{side}")
            prefix = "OBSERVED" if value["tracking_gate_applicable"] else "DISPLAY_ONLY"
            if value["position_violation"]:
                reasons.append(f"{prefix}_POSITION_GT_20_MM:{side}")
            if value["rotation_violation"]:
                reasons.append(f"{prefix}_ROTATION_GT_15_DEG:{side}")
            if not value["tracking_error_valid"]:
                reasons.append(f"TRACKING_ERROR_INVALID:{side}")
            if value["diagnostic_display"]:
                reasons.append(f"DIAGNOSTIC_HAND_DISPLAY:{side}")
            elif not value["raw_observed"]:
                reasons.append(f"SHORT_FILLED_HAND_DISPLAY:{side}")
        if row["camera_source"] not in visual.HEAD_HINT_SOURCES:
            reasons.append("DIAGNOSTIC_OR_UNAVAILABLE_CAMERA")
        elif not row["camera_raw_valid"]:
            reasons.append("SHORT_FILLED_CAMERA_DISPLAY")
        if row["cup_identity_conflict_pixels"]:
            reasons.append("CUP_WEARER_SEMANTIC_CONFLICT")
        wearer_visible = bool((masks["sam_core"][frame] | masks["equipment"][frame]).any())
        if wearer_visible and not render_rows["robot_pixels"][frame]:
            reasons.append("ROBOT_EMPTY_WHILE_WEARER_VISIBLE")
        if reasons:
            evidence[str(frame)] = reasons
        row["abnormal_reasons"] = reasons
    indices = sorted(int(frame) for frame in evidence)
    windows = visual.review_windows_from_qpc(np.asarray(qpc_ns), 1e9, indices,
                                              trigger_key="abnormal_frames")
    return {"abnormal_raw_frame_indices": indices,
            "abnormal_frame_reasons": evidence, "required_abnormal_windows": windows}


def _mask_metadata(masks: Mapping) -> dict:
    value = masks.get("metadata", {})
    while isinstance(value, dict) and isinstance(value.get("metadata"), dict):
        value = value["metadata"]
    return value if isinstance(value, dict) else {}


def render_session(repo: Path, folder: Path, session: source.RawSession,
                   motion: Mapping[str, Any], clean_rgb: np.ndarray,
                   masks: Mapping[str, Any], *, tick=lambda: None, asset_guard=lambda robot: None) -> dict:
    """Produce complete final/review videos once; review/quality remain pending."""
    started = time.monotonic()
    repo, folder = Path(repo).resolve(strict=True), Path(folder).resolve()
    if folder == repo or not folder.is_relative_to(repo):
        raise ValueError("RENDER_OUTPUT_OUTSIDE_PROJECT")
    raw, clean, validation = _validate_inputs(session, motion, clean_rgb, masks)
    upstream_refs = _upstream_refs(repo, masks)
    folder.mkdir(parents=True, exist_ok=True)
    required_outputs = ("final.mp4", "review.mp4", "frame_states.jsonl", "contact_events.json", "SESSION.json")
    if any((folder / name).exists() for name in required_outputs):
        raise ValueError("RENDER_FRESH_OUTPUT_REQUIRED")
    count = session.frame_count
    actual_render_view_count = 0
    K = rd.scaled_intrinsics(session.K, WIDTH, HEIGHT, (session.width, session.height))
    T_base = np.asarray(motion["T_world_base"])
    robot = _robot(repo)
    asset_guard(robot)
    renderer = rd.RobotRenderer(robot)
    external = rd.external_camera(np.linalg.inv(session.c2w[0]) @ T_base)
    display_valid = _display_valid(session)
    camera_hint_valid = visual.head_hint_valid(session)
    hand_source = np.asarray(motion.get("hand_source", session.hand_source))
    tips = np.full((2, count, 5, 2), np.nan)
    rows = {"robot_pixels": [], "object_front_pixels": [], "feather_restore_pixels": [],
            "final_protected_changed_pixels": [], "removal_uncovered_pixels": []}
    final_encoder = review_encoder = None
    try:
        final_encoder = _QpcVideoEncoder(folder / "final.mp4", session.qpc_ns, WIDTH, HEIGHT)
        review_encoder = _QpcVideoEncoder(folder / "review.mp4", session.qpc_ns, WIDTH * 3, HEIGHT * 2)
        for frame in range(count):
            tick()
            raw_bgr, clean_bgr = raw[frame][..., ::-1], clean[frame][..., ::-1]
            camera = np.linalg.inv(session.c2w[frame]) @ T_base
            bgr, alpha, _ = renderer.render(motion["q_arm"][frame], motion["q_hand"][frame],
                                            camera, K, complete_robot=False)
            actual_render_view_count += 1
            one = {key: masks[key][frame] for key in ("core", "protected_visible", "unknown")}
            composed = visual.compose_frame(raw_bgr, clean_bgr, bgr, alpha, one,
                                            robot_observed=True, masks_available=True)
            final = rd.over(clean_bgr, bgr, composed["draw"])
            final[composed["object_front"]] = clean_bgr[composed["object_front"]]
            # Gaussian feather spreads beyond binary draw. Protect after feather.
            feather_support = rd.soft_alpha(composed["draw"]) > 0
            protect = masks["protected_visible"][frame]
            restore = protect & feather_support
            final[protect] = clean_bgr[protect]
            protected_changed = int(np.any(final != clean_bgr, axis=-1)[protect].sum())
            if protected_changed:
                raise RuntimeError("FINAL_CUP_PROTECTION_FAILED_AFTER_FEATHER")
            final_encoder.write(final)
            ebgr, ealpha, _ = renderer.render(motion["q_arm"][frame], motion["q_hand"][frame],
                                              external, rd.EXTERNAL_K, complete_robot=True)
            actual_render_view_count += 1
            skeleton = raw_bgr.copy()
            head_source = str(session.native_motion.get("head_source", np.full(count, "UNAVAILABLE"))[frame])
            valid = display_valid[:, frame] & camera_hint_valid[frame]
            tips[:, frame] = _project_robot_tips(robot, motion["q_arm"][frame],
                                                 motion["q_hand"][frame], camera, K, valid)
            for side, colour in ((0, (255, 128, 0)), (1, (0, 128, 255))):
                points = session.camera_points(session.hands[side, frame], frame)
                if valid[side] and np.isfinite(points).all() and (points[:, 2] > 0.05).all():
                    pixels = session.project(points) * (WIDTH / session.width)
                    rd.draw_hand(skeleton, pixels, colour, f"{'LR'[side]}:{hand_source[side, frame]}")
            overlay = raw_bgr.copy()
            removal = masks["removal"][frame]
            overlay[removal] = (0.4 * overlay[removal] + 0.6 * np.array((255, 0, 255))).astype(np.uint8)
            overlay[protect] = (0.4 * overlay[protect] + 0.6 * np.array((255, 160, 0))).astype(np.uint8)
            panels = (
                rd.title(skeleton, f"{session.session_id} {frame + 1}/{count} RAW PICO+MANUS"),
                rd.title(overlay, "HISTORICAL WEARER REQUEST / MODEL CUP PROTECTION"),
                rd.title(clean_bgr, "Q-FROZEN CUP-MASK RAW RESTORE CLEAN"),
                rd.title(rd.on_grey(bgr, alpha), "EGO ROBOT / SAME SAVED Q"),
                rd.title(rd.on_grey(ebgr, ealpha), "EXTERNAL COMPLETE ROBOT"),
                rd.title(final, "FINAL / VISUAL UNKNOWN CUP OBJECT FIRST"),
            )
            review = np.concatenate((np.concatenate(panels[:3], axis=1),
                                     np.concatenate(panels[3:], axis=1)), axis=0)
            review_encoder.write(review)
            rows["robot_pixels"].append(int(alpha.sum()))
            rows["object_front_pixels"].append(int(composed["object_front"].sum()))
            rows["feather_restore_pixels"].append(int(restore.sum()))
            rows["final_protected_changed_pixels"].append(protected_changed)
            rows["removal_uncovered_pixels"].append(int((removal & ~composed["draw"]).sum()))
        final_encoding = final_encoder.close()
        review_encoding = review_encoder.close()
    except BaseException:
        if final_encoder is not None:
            final_encoder.abort()
        if review_encoder is not None:
            review_encoder.abort()
        raise
    contact = visual.build_contact_candidates(session.qpc_ns, 1e9,
                masks["sam_core"] | masks["equipment"], masks["protected_visible"], hand_points_px=tips)
    contact["fingertip_source"] = "ROBOT_FK_SAME_SAVED_Q_CAMERA_AND_RENDER"
    contact["action_modified_by_contact"] = False
    from .render_adapter import annotate_contact_roles
    contact = annotate_contact_roles(contact, masks)
    _write_json(folder / "contact_events.json", contact)
    states = _frame_states(session, motion, masks, validation, rows, contact["frames"])
    abnormal = _abnormal_evidence(session.qpc_ns, states, masks, rows)
    with (folder / "frame_states.jsonl").open("x", encoding="utf-8") as stream:
        for row in states:
            stream.write(json.dumps(row, ensure_ascii=False, default=_json_default, allow_nan=False) + "\n")
    np.savez_compressed(folder / "render_validation.npz",
                       **{key: np.asarray(value) for key, value in rows.items()},
                       robot_projected_fingertips_px=tips.astype(np.float32))
    outputs = {name: source.ref(folder / name) for name in
               ("final.mp4", "review.mp4", "frame_states.jsonl", "contact_events.json", "render_validation.npz")}
    outputs.update(upstream_refs)
    required_samples = sorted({0, count - 1, *[int(round((count - 1) * fraction))
                                             for fraction in (.25, .5, .75)]})
    value = {
        "schema_version": "pick-place-cup-restore-fresh-render-contact-session-candidate-v1",
        "status": "EXECUTION_COMPLETE_REVIEW_PENDING", "execution_complete": True,
        "session_id": folder.name, "raw_session_id": session.session_id, "frame_count": count,
        "output_size": [WIDTH, HEIGHT], "review_size": [WIDTH * 3, HEIGHT * 2],
        "qpc_source": "RAW_VST_QPC_DECLARED_NS", "exposure_synchronized": False,
        "final_encoding": final_encoding, "review_encoding": review_encoding,
        "outputs": outputs, "source_input": source.ref(session.raw_dir / "SESSION_INPUT.json"),
        "camera": session.calibration, "clean_validation": validation, "render_validation": rows,
        "quality_problems": _mask_metadata(masks).get("quality_problems", []),
        "baseline_quality_problems_history_not_recomputed_under_newP": masks["baseline_quality_problems"],
        "mask_policy": "NEW_P_OLD_MODEL_CUP_NEW_E_OLD_E_MINUS_CUP_OLD8_UNCHANGED",
        "held_unknown_are_legacy_model_masks_not_physical_occlusion": True,
        "ai_review": {"status": "PENDING", "actually_checked_frames": [],
                      "required_fixed_frames": required_samples,
                      "required_candidate_windows": contact["windows"],
                      **abnormal,
                      "automatic_anomaly_review": "SEALED_ACTUAL_QPC_MOTION_SOURCE_AND_MASK_EVIDENCE"},
        "human_review": "PENDING", "visual_quality": "UNREVIEWED",
        "physical_contact_ground_truth": False, "control_ground_truth": False,
        "training_eligible": False, "contact_changes_q": False,
        "render_seconds": round(time.monotonic() - started, 3),
        "actual_render_view_count": actual_render_view_count,
        "claim_limit": "Complete offline visual candidate; execution is not visual acceptance or physical truth.",
    }
    _write_json(folder / "SESSION.json", value)
    return value
