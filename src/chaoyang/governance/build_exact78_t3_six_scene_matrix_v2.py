"""Read-only six-session T3 scene join with monotonic, fail-closed gates.

This is a candidate report generator, not a T3 sealer or a ``chaoyang run``
operation.  It binds only frozen T0 sessions and fixed sealed baseline T2
Action/Scene task IDs.  Active Clean successors are never selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chaoyang.governance import build_exact78_t3_six_matrix_v1 as action
from chaoyang.governance import build_exact78_t3_six_scene_matrix_v1 as scene_v1


ROOT = Path(__file__).resolve().parents[3]
PINNED_SCENE_V1_SHA256 = "a025056af5f463699dd4ff6e75d7213c5d31da7338481ce0293e83b77510b6ba"
NOT_EVALUATED = "NOT_EVALUATED"
NOT_QUALIFIED = "NOT_QUALIFIED"


def qualification(action_claim: Any, scene_claim: Any) -> str:
    """A negative claim vetoes; two positives still need independent review."""
    if action_claim is False or scene_claim is False:
        return NOT_QUALIFIED
    if action_claim is True and scene_claim is True:
        return "NOT_EVALUATED_INDEPENDENT_REVIEW_REQUIRED"
    return NOT_EVALUATED


def _safe_scene(value: dict[str, Any], task_id: str) -> dict[str, Any]:
    """Failed/absent Scene has a complete, non-qualifying report shape."""
    return {"status": value.get("status", NOT_EVALUATED),
            "task_id": value.get("task_id", task_id),
            "terminal_result": value.get("terminal_result"),
            "terminal_receipt": value.get("terminal_receipt"),
            "mask": value.get("mask", {"status": NOT_EVALUATED}),
            "clean": value.get("clean", {"status": NOT_EVALUATED}),
            "visual_qualified": value.get("visual_qualified"),
            "product_qualified": value.get("product_qualified"),
            "training_eligible": value.get("training_eligible"),
            "user_review": value.get("user_review", NOT_EVALUATED),
            "qpc_head_sync": value.get("qpc_head_sync", NOT_EVALUATED)}


def _seal_child(root: Path, attempt: Path, seals: dict[str, Any], name: str,
                session_id: str, frames: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    seal_ref = seals.get(name)
    seal_path, seal = action._verified(root, seal_ref, required_parent=attempt)
    child_ref = seal.get("result")
    child_path, child = action._verified(root, child_ref, required_parent=attempt)
    if (seal_path != attempt / "candidate" / f"SEAL_{name}.json"
            or child_path != attempt / "candidate" / f"{name}.json"
            or seal.get("session_id") != session_id or seal.get("stage") != name
            or seal.get("frame_count") != (0 if name == "MAP" else frames)
            or child.get("session_id") != session_id):
        raise ValueError(f"SCENE_{name}_BINDING_DRIFT")
    return seal_ref, child_ref, child


def _coverage(mask: dict[str, Any], clean: dict[str, Any], frames: int) -> dict[str, Any]:
    mask_rows, clean_rows = mask["rows"], clean["rows"]
    if (len(mask_rows) != frames or len(clean_rows) != frames
            or [row.get("decode_display_index") for row in mask_rows] != list(range(frames))
            or [row.get("decode_display_index") for row in clean_rows] != list(range(frames))):
        raise ValueError("SCENE_COVERAGE_FRAME_MAP_DRIFT")
    roles = ("human", "equipment", "object", "finger_sleeve", "yellow_cable")
    nonempty = {role: sum(row.get("instance_counts", {}).get(role, 0) > 0
                          for row in mask_rows) for role in roles}
    return {"mask_frames": frames, "clean_frames": frames,
            "sam_instance_nonempty_frames": nonempty,
            "removal_pixels_total": sum(int(row.get("removal_pixels", 0)) for row in mask_rows),
            "clean_synthetic_pixels_total": sum(int(row.get("synthetic_pixels", 0)) for row in clean_rows),
            "clean_real_donor_pixels": clean.get("real_donor_pixels", NOT_EVALUATED),
            "coverage_claim": "SAM_TEXT_CANDIDATES_NOT_VISIBLE_OBJECT_OR_DEVICE_TRUTH"}


def _scene_source_evidence(root: Path, session_id: str, frames: int, fps: int,
                           action_rgb: dict[str, Any], selected: dict[str, Any],
                           scene: dict[str, Any], action_row: dict[str, Any]
                           ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify T0/source references and explicitly refuse cross-stream sync claims."""
    task_id = scene["task_id"]
    attempt = root / "_run/current" / task_id / "attempts/attempt_0001"
    _, outer = action._verified(root, scene["terminal_result"], required_parent=attempt)
    spec_ref = outer.get("frontend_spec")
    spec_path, spec = action._verified(root, spec_ref, required_parent=attempt)
    if (spec_path != attempt / "FRONTEND_SPEC.json" or spec.get("session_id") != session_id
            or spec.get("frame_count") != frames or spec.get("source_fps") != fps):
        raise ValueError("SCENE_FRONTEND_SPEC_IDENTITY_DRIFT")
    family = spec["source_family"]
    expected = selected["files"]["vst.h264" if family == "VST_0915" else "stereo"]
    expected_camera = selected["files"]["camera_params.json" if family == "VST_0915" else "camera"]
    if spec.get("source_video") != expected or spec.get("camera_params") != expected_camera:
        raise ValueError("SCENE_SOURCE_NOT_T0_RAW_OR_CAMERA")
    seals = outer.get("stage_seals", {})
    map_seal, map_ref, map_result = _seal_child(root, attempt, seals, "MAP", session_id, frames)
    raw_seal, raw_ref, raw_result = _seal_child(root, attempt, seals, "RAW", session_id, frames)
    raw_rows = raw_result.get("rows")
    if (not isinstance(raw_rows, list) or len(raw_rows) != frames
            or [row.get("decode_display_index") for row in raw_rows] != list(range(frames))
            or [row.get("model_input_index") for row in raw_rows] != list(range(frames))
            or map_result.get("source_video") != expected
            or raw_result.get("source_video") != expected
            or raw_result.get("camera_params") != expected_camera
            or raw_result.get("source_fps") != fps):
        raise ValueError("SCENE_MAP_RAW_FRAME_OR_SOURCE_DRIFT")
    if action_row["execution"]["status"] == "PASSED":
        audit_ref = action_row["hawor"].get("input_audit")
        _, audit = action._verified(root, audit_ref,
            required_parent=root / "_run/current" / action_row["task_id"] / "attempts/attempt_0001")
        if audit.get("raw_source_ref") != action_rgb:
            raise ValueError("HAWOR_ACTION_SOURCE_NOT_T0")
        action_audit = audit_ref
    else:
        action_audit = None
    same_bytes = action_rgb == expected
    if family == "VST_0915":
        if not same_bytes or spec.get("source_index") != 1:
            raise ValueError("VST_ACTION_SCENE_RAW_OR_EYE_DRIFT")
        alignment = "SAME_RAW_VST_BYTES_INDEX_ONLY_TIMEBASE_UNVERIFIED"
        time_mode = "TIMEBASE_UNVERIFIED"
    else:
        if same_bytes or spec.get("source_index") != 0:
            raise ValueError("TRACKER_MONO_STEREO_SOURCE_EXPECTATION_DRIFT")
        alignment = "NOT_PROVEN_MONO_STEREO_FRAME_MAPPING"
        time_mode = "CLIP_INDEX_ONLY_HEAD_CLOCK_UNVERIFIED"
    counts = {"capture_frame_id_nonnull": sum(row.get("capture_frame_id") is not None for row in raw_rows),
              "qpc_nonnull": sum(row.get("qpc_ticks") is not None for row in raw_rows),
              "pts_nonnull": sum(row.get("pts") is not None for row in raw_rows),
              "model_input_rows": sum(isinstance(row.get("model_input"), dict) for row in raw_rows)}
    evidence = {"action_rgb_source": action_rgb, "action_hawor_input_audit": action_audit,
                "scene_raw_source": expected, "scene_camera_params": expected_camera,
                "scene_frontend_spec": spec_ref,
                "scene_map_seal": map_seal, "scene_map_result": map_ref,
                "scene_raw_seal": raw_seal, "scene_raw_result": raw_ref}
    mapping = {"status": alignment, "same_source_file_sha": same_bytes,
               "source_family": family, "scene_physical_source_index": spec["source_index"],
               "time_mode": time_mode, "scene_raw_frame_rows": frames,
               "scene_raw_support_counts": counts,
               "action_t0_capture_to_decode_alignment": action_row["capture_to_decode_alignment"],
               "claim_limit": "No mono↔stereo or QPC/Head synchronization inferred from equal indices"}
    return evidence, mapping


def build_matrix(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve(strict=True)
    old_ref = action._ref(Path(scene_v1.__file__).resolve(strict=True))
    if old_ref["sha256"] != PINNED_SCENE_V1_SHA256:
        raise ValueError("PINNED_SCENE_V1_BUILDER_SHA_DRIFT")
    prequal = action._ref(root / scene_v1.PREQUAL_REL)
    correction = action._ref(root / scene_v1.CORRECTION_REL)
    if prequal["sha256"] != scene_v1.PREQUAL_SHA or correction["sha256"] != scene_v1.CORRECTION_SHA:
        raise ValueError("GEOMETRY_PREQUAL_SHA_DRIFT")
    capability = action._json(root / scene_v1.CORRECTION_REL)
    base = action.build_matrix(root)
    if set(base["sessions"]) != set(scene_v1.FRONTENDS):
        raise ValueError("T3_SIX_SESSION_SET_DRIFT")
    state_path = root / scene_v1.STATE_REL
    state_ref = action._ref(state_path)
    tasks = action._json(state_path)["tasks"]
    task_by_id = {task["task_id"]: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("TASK_LEDGER_DUPLICATE_IDS")
    freeze = action._json(root / action.FREEZE_REL)["selected"]
    rows: dict[str, Any] = {}
    for sid, action_row in sorted(base["sessions"].items()):
        task_id, _ = scene_v1.FRONTENDS[sid]
        selected = freeze[sid]
        frames, fps = action_row["frames"], action_row["nominal_fps"]
        scene = _safe_scene(scene_v1._terminal_frontend(root, sid, task_by_id.get(task_id), frames), task_id)
        scene_evidence: dict[str, Any] = {}
        mapping: dict[str, Any] = {"status": NOT_EVALUATED, "time_mode": NOT_EVALUATED}
        coverage: dict[str, Any] = {"status": NOT_EVALUATED}
        clean_internal_videos: dict[str, Any] = {}
        if scene["status"] == "PASSED_EXECUTION_ONLY":
            scene_evidence, mapping = _scene_source_evidence(
                root, sid, frames, fps, action_row["raw_source"], selected, scene, action_row)
            _, mask = action._verified(root, scene["mask"]["result"])
            _, clean = action._verified(root, scene["clean"]["result"])
            coverage = _coverage(mask, clean, frames)
            if clean.get("source_fps") != fps:
                raise ValueError("SCENE_CLEAN_FPS_T0_DRIFT")
            clean_internal_videos = clean.get("internal_videos", {})
            if set(clean_internal_videos) != {"inpaint_out.mp4", "masked_in.mp4"}:
                raise ValueError("SCENE_CLEAN_INTERNAL_VIDEO_SET_DRIFT")
            for video in clean_internal_videos.values():
                if (video.get("decoded_frames") != frames or video.get("fps") != fps
                        or not isinstance(video.get("artifact"), dict)):
                    raise ValueError("SCENE_CLEAN_INTERNAL_VIDEO_FRAME_DRIFT")
        source = scene_v1.SOURCE[sid]
        calibration = capability.get("calibration_states", {}).get(sid)
        if not isinstance(calibration, dict):
            raise ValueError("T0_CORRECTED_CALIBRATION_MISSING")
        if source == "0901" and calibration.get("metric_geometry") != "BLOCKED_CALIBRATION_ABSENT":
            raise ValueError("T0_0901_CALIBRATION_CLAIM_DRIFT")
        depth_input, object_input = scene_v1.GEOMETRY_INPUT_PREQUAL[sid]
        if any(action_row["stage_qualification"].get(key) != NOT_EVALUATED
               for key in ("depth", "object6d", "occlusion")):
            raise ValueError("FRESH_GEOMETRY_REQUIRES_SEPARATE_PROOF")
        rows[sid] = {
            "session_id": sid, "source": source,
            "selection_role": action_row["selection_role"],
            "split_claim": action_row.get("split_claim"),
            "source_group": action_row["source_group"],
            "raw": {"frames": frames, "nominal_fps": fps,
                    "action_rgb_source": action_row["raw_source"],
                    "scene_raw_source": scene_evidence.get("scene_raw_source"),
                    "physical_eye": action_row["physical_eye"],
                    "t0_time_mode": action_row["time_mode"],
                    "capture_to_decode_alignment": action_row["capture_to_decode_alignment"],
                    "head_pose_clock_domain": action_row["head_pose_clock_domain"],
                    "corrected_metric_geometry": calibration["metric_geometry"]},
            "cross_lane_frame_mapping": mapping,
            "execution": {"action_status": action_row["execution"]["status"],
                          "action_complete": action_row["execution"].get("complete", False),
                          "robot_video_complete_sealed_claim": action_row["robot"].get("video_complete_sealed_claim", False),
                          "scene_frontend_status": scene["status"],
                          "scene_mask_sealed_frames": scene["mask"].get("frame_count", NOT_EVALUATED),
                          "scene_clean_sealed_frames": scene["clean"].get("frame_count", NOT_EVALUATED)},
            "hawor": action_row["hawor"], "robot": action_row["robot"],
            "mask": scene["mask"], "clean": scene["clean"],
            "coverage": coverage,
            "geometry_prequalification": {
                "depth": {"input_status": depth_input, "fresh_result": NOT_EVALUATED,
                          "field_qualification": NOT_EVALUATED},
                "object6d": {"input_status": object_input, "fresh_result": NOT_EVALUATED,
                             "field_qualification": NOT_EVALUATED},
                "trusted_occlusion": {"input_status": scene_v1.OCCLUSION_PREQUAL,
                                      "geometry_decision_coverage": NOT_EVALUATED,
                                      "display_fallback_coverage": NOT_EVALUATED,
                                      "field_qualification": NOT_EVALUATED}},
            "visual_qualification": qualification(action_row["visual_qualified"], scene["visual_qualified"]),
            "full_product_qualification": qualification(action_row["product_qualified"], scene["product_qualified"]),
            "training_label_qualification": qualification(action_row["training_eligible"], scene["training_eligible"]),
            "ai_review": NOT_EVALUATED,
            "human_review": ("PENDING" if action_row["user_review"] == "PENDING"
                             and scene["user_review"] == "PENDING" else NOT_EVALUATED),
            "evidence": {"action": action_row["evidence"],
                         "scene_terminal_result": scene["terminal_result"],
                         "scene_terminal_receipt": scene["terminal_receipt"],
                         "scene_mask_result": scene["mask"].get("result"),
                         "scene_clean_result": scene["clean"].get("result"),
                         "scene_clean_internal_videos": clean_internal_videos,
                         **scene_evidence},
        }
    if action._ref(state_path) != state_ref:
        raise ValueError("TASK_STATE_CHANGED_DURING_SUCCESSOR_JOIN")
    development = [sid for sid, row in rows.items() if row["selection_role"] == "development"]
    holdout = [sid for sid, row in rows.items() if row["selection_role"] == "holdout"]
    if len(development) != 4 or len(holdout) != 2:
        raise ValueError("T3_4DEV_2HOLDOUT_SPLIT_DRIFT")
    return {
        "schema_version": "exact78-t3-six-scene-joined-cpu-snapshot-v2",
        "status": "READ_ONLY_SUCCESSOR_NOT_T3_SEAL",
        "input_evidence": {"action_builder": action._ref(Path(action.__file__).resolve()),
                           "scene_v1_builder": old_ref,
                           "t0_freeze": base["inputs"]["t0_freeze"],
                           "t0_frame_time_map": base["inputs"]["t0_frame_time_map"],
                           "state_for_terminal_eligibility_only": state_ref,
                           "geometry_prequal_review": prequal,
                           "t0_corrected_capability": correction,
                           "generator_code": action._ref(Path(__file__).resolve())},
        "selection": {"development": development, "holdout": holdout,
                      "source_holdout_gap": {"source": "0902", "session_id": None,
                                             "status": "SPLIT_NOT_ESTABLISHED"}},
        "sessions": rows,
        "claim_limit": "Sealed execution only; mono/stereo frame equivalence and VST timebase unproven; no visual, geometry, product, training or user approval",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(build_matrix(args.repo_root), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
