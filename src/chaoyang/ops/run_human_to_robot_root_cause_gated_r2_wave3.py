#!/usr/bin/env python3
"""R2 wave 3: explicit Contact/R1 funnel and deterministic renderer isolation.

This wave does not manufacture Contact or Clean.  It records the first local
consumer blocker for every fixed session, then runs real PyBullet renders that
isolate q, camera and background changes on the frozen 007 motion.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from chaoyang.ops.run_human_to_robot_root_cause_gated_r2_wave0 import (
    ATTEMPT, REPO, TASK, ref, write_json,
)
from chaoyang.ops.run_human_to_robot_root_cause_gated_r2_wave2 import append_state
from chaoyang.pipeline.full_robot_review_v2 import MeshScene
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.array(archive[key], copy=True) for key in archive.files}


def contact_funnel() -> dict:
    out = ATTEMPT / "lanes/lane1_scene/contact_r1_funnel_wave3/RESULT.json"
    depth_root = REPO / "_run/current/0915_robot15h_foundationstereo_wave0_v1/attempts/attempt_0001/sessions"
    object_root = REPO / "_run/current/0915_robot15h_geometry_object6d_wave0_v1/attempts/attempt_0001/sessions"
    cases = [
        ("play_cards_0915_031", 149, "playing_cards", "recovered_031_wave0", [0, 102]),
        ("get_potato_chips_0915_007", 378, "potato_chips", "recovered_007_wave0", [378, 378]),
        ("get_potato_chips_0902_103", 284, "potato_chips", "recovered_0902_103_wave2", [284, 281]),
        ("play_cards_0902_042", 171, "playing_cards", "recovered_0902_042_wave2", [171, 171]),
    ]
    rows = []
    for session, frames, task, motion_folder, hand_counts in cases:
        depth_dir = depth_root / task / session
        depth_contract = depth_dir / "DEPTH_CONTRACT.json"
        object_result = object_root / task / session / "RESULT.json"
        stages = {
            "timeline_hand_valid": {"status": "PASS" if sum(hand_counts) else "FAIL", "side_frames": hand_counts},
            "depth_registration": {"status": "NOT_AVAILABLE", "frames": 0},
            "object_identity": {"status": "NOT_AVAILABLE"},
            "visible_surface": {"status": "NOT_AVAILABLE", "frames": 0},
            "residual_boundary": {"status": "NOT_EVALUATED"},
            "fixed_wrist_feasibility": {"status": "NOT_EVALUATED"},
            "r1_execution": {"status": "NOT_EXECUTED"},
            "hard_gate": {"status": "NOT_EVALUATED"},
        }
        evidence = {
            "motion": ref(ATTEMPT / "lanes/lane2_motion" / motion_folder / "ROBOT_R0_V1.npz")
        }
        first = None
        if depth_contract.is_file():
            depth = json.loads(depth_contract.read_text(encoding="utf-8"))
            evidence["depth_contract"] = ref(depth_contract)
            stages["depth_registration"] = {
                "status": "PASS_VISUAL_OBJECT6D_ONLY" if depth.get("consumption_authorized") else "FAIL",
                "frames": int(depth.get("frame_count", 0)),
                "strict_metric_contact_authorized": bool(depth.get("strict_metric_contact_authorized")),
                "coordinate_domain": depth.get("depth_reference"),
            }
        else:
            first = "BLOCKED_DEPTH_REGISTRATION_ABSENT"
        if object_result.is_file():
            obj = json.loads(object_result.read_text(encoding="utf-8"))
            evidence["object6d"] = ref(object_result)
            counts = obj.get("observable_patch_counts", {})
            stages["object_identity"] = {"status": "PASS_VISIBLE_SURFACE_ONLY", "instances": sorted(counts)}
            stages["visible_surface"] = {"status": "PASS_VISIBLE_SURFACE_ONLY", "frames_by_instance": counts}
        elif first is None:
            first = "BLOCKED_OBJECT6D_VISIBLE_PATCH_ABSENT"
        # Existing Depth contracts explicitly close strict metric Contact.  A
        # visual Object6D scope cannot be silently promoted by this successor.
        if first is None and not stages["depth_registration"].get("strict_metric_contact_authorized", False):
            first = "BLOCKED_STRICT_METRIC_CONTACT_AUTHORITY"
        if session == "play_cards_0915_031" and hand_counts[0] == 0:
            stages["timeline_hand_valid"]["left_role"] = "INCONCLUSIVE_OBSERVABILITY"
            first = "BLOCKED_LEFT_HAND_ROLE_FOR_BILATERAL_PRODUCT"
        rows.append({
            "session_id": session,
            "frame_count": frames,
            "stages": stages,
            "first_blocker": first,
            "r1_windows_executed": 0,
            "r1_adopted_windows": 0,
            "execution": "EXECUTED_FUNNEL_ONLY",
            "quality": "INCONCLUSIVE",
            "adoption": "NOT_ADOPTED",
            "evidence": evidence,
            "unaffected_work": ["Scene candidate diagnostics", "Motion/R0", "Sensor backend", "Local/HuRo numeric comparison"],
        })
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_CONTACT_R1_FUNNEL_V1",
        "task_id": TASK,
        "created_at": now(),
        "sessions": rows,
        "r1_windows_screened": 0,
        "r1_windows_executed": 0,
        "r1_windows_adopted": 0,
        "claim_limit": "Funnel evidence only; no Contact, force, ground-truth or Robot R1 claim.",
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    write_json(out, result)
    return result


def _capture(scene: MeshScene, view) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = scene.b
    values = b.getCameraImage(
        640, 480, view, scene.projection,
        renderer=b.ER_TINY_RENDERER, shadow=0, physicsClientId=scene.client,
    )
    rgb = np.asarray(values[2], np.uint8).reshape(480, 640, 4)[..., :3].copy()
    depth = np.asarray(values[3], np.float32).reshape(480, 640).copy()
    segmentation = np.asarray(values[4], np.int32).reshape(480, 640).copy()
    return rgb, depth, segmentation


def renderer_four_way() -> dict:
    output = ATTEMPT / "lanes/lane4_compare/renderer_four_way_wave3"
    if (output / "RESULT.json").exists():
        raise FileExistsError(f"IMMUTABLE_RESULT_EXISTS:{output}")
    output.mkdir(parents=True, exist_ok=True)
    source = REPO / "_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/lanes/ai4_huro/common_review_v2_run1/get_potato_chips_0915_007/MOTION_RESULT.npz"
    motion = load_npz(source)
    assets = load_pinned_robot_assets(REPO)
    valid = np.asarray(motion["wrist_valid"] & motion["finger_valid"], bool)
    active_side = int(np.argmax(valid.sum(axis=0)))
    usable = np.flatnonzero(valid[:, active_side])
    if len(usable) < 2:
        raise ValueError("NO_TWO_SAME_SIDE_VALID_RENDER_FRAMES")
    base = int(usable[len(usable) // 3])
    q0 = motion["q22"][base].copy()
    distances = np.linalg.norm(motion["q22"][usable, active_side] - q0[active_side][None], axis=1)
    alt = int(usable[int(np.argmax(distances))])
    if not float(distances.max()) > 1e-6:
        raise ValueError("NO_MEANINGFUL_Q_VARIATION")
    scene = MeshScene(assets, motion)
    try:
        scene.frame(base)
        fixed1 = _capture(scene, scene.main_view)
        scene.frame(base)
        fixed2 = _capture(scene, scene.main_view)
        original_q = motion["q22"][base].copy()
        motion["q22"][base] = motion["q22"][alt]
        scene.frame(base)
        q_only = _capture(scene, scene.main_view)
        motion["q22"][base] = original_q
        scene.frame(base)
        camera_view = scene.b.computeViewMatrixFromYawPitchRoll(
            scene.camera_center, scene.camera_distance, 47, -22, 0, 2
        )
        camera_only = _capture(scene, camera_view)
    finally:
        scene.close()
    fixed_rgb, fixed_depth, fixed_seg = fixed1
    robot_mask = fixed_seg >= 0
    bg_a = np.full_like(fixed_rgb, 35)
    bg_b = np.zeros_like(fixed_rgb)
    bg_b[..., 0] = 35; bg_b[..., 1] = 95; bg_b[..., 2] = 145
    composite_a = np.where(robot_mask[..., None], fixed_rgb, bg_a)
    composite_b = np.where(robot_mask[..., None], fixed_rgb, bg_b)
    background_change = np.any(composite_a != composite_b, axis=2)
    fixed_equal = all(np.array_equal(a, b) for a, b in zip(fixed1, fixed2, strict=True))
    q_changed = np.any(fixed_rgb != q_only[0], axis=2)
    camera_changed = np.any(fixed_rgb != camera_only[0], axis=2)
    if not fixed_equal or not q_changed.any() or not camera_changed.any():
        raise ValueError("RENDER_ISOLATION_TEST_FAILED")
    if np.any(background_change & robot_mask) or not np.any(background_change & ~robot_mask):
        raise ValueError("BACKGROUND_ISOLATION_TEST_FAILED")
    panels = [fixed_rgb, fixed2[0], q_only[0], camera_only[0], composite_a, composite_b]
    titles = ["fixed #1", "fixed #2", "q only", "camera only", "background A", "background B"]
    canvas = np.full((960, 1920, 3), 245, np.uint8)
    for index, (panel, title) in enumerate(zip(panels, titles, strict=True)):
        row, col = divmod(index, 3)
        x, y = col * 640, row * 480
        canvas[y:y + 480, x:x + 640] = panel
        cv2.putText(canvas, title, (x + 12, y + 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (10, 10, 10), 2)
    montage = output / "RENDERER_FOUR_WAY_DIAGNOSTIC.png"
    cv2.imwrite(str(montage), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    arrays = output / "RENDERER_BUFFERS.npz"
    np.savez_compressed(
        arrays, fixed_rgb=fixed_rgb, fixed_depth=fixed_depth, fixed_segmentation=fixed_seg,
        q_only_rgb=q_only[0], camera_only_rgb=camera_only[0], robot_mask=robot_mask,
        composite_background_a=composite_a, composite_background_b=composite_b,
    )
    result = {
        "schema_version": "HUMAN_TO_ROBOT_R2_RENDERER_FOUR_WAY_V1",
        "task_id": TASK,
        "created_at": now(),
        "session_id": "get_potato_chips_0915_007",
        "base_frame": base,
        "q_source_frame": alt,
        "active_physical_side": active_side,
        "inactive_side_policy": "NEUTRAL_GRAY_DISPLAY_ONLY_NOT_OBSERVED",
        "fixed_repeat_byte_exact": fixed_equal,
        "q_only_changed_pixels": int(q_changed.sum()),
        "camera_only_changed_pixels": int(camera_changed.sum()),
        "background_only_changed_pixels": int(background_change.sum()),
        "background_changed_robot_pixels": int((background_change & robot_mask).sum()),
        "robot_mask_pixels": int(robot_mask.sum()),
        "structure": "PASS",
        "quality": "DIAGNOSTIC_ONLY",
        "adoption": "NOT_ADOPTED",
        "source": ref(source),
        "outputs": {"buffers": ref(arrays), "montage": ref(montage)},
        "claim_limit": "Actual pinned-asset PyBullet isolation diagnostic; not final Clean composite, occlusion, or visual acceptance.",
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
    }
    write_json(output / "RESULT.json", result)
    return result


def update_motion_observability_state() -> None:
    state_path = ATTEMPT / "lanes/lane2_motion/STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    evidence = ATTEMPT / "lanes/lane2_motion/observability_031_wave1/RESULT.json"
    artifacts = state.setdefault("latest_artifacts", [])
    item = ref(evidence)
    if not any(row.get("path") == item["path"] for row in artifacts if isinstance(row, dict)):
        artifacts.append(item)
    blockers = state.setdefault("blockers", [])
    blocker = {
        "code": "031_LEFT_OBSERVABILITY_INCONCLUSIVE",
        "missing": "Independent side-resolved RGB evidence for the left hand",
        "consumer": "031 bilateral Motion/Product quality",
        "owner": "lane2_motion",
        "unblock_action": "Obtain independent side-resolved hand evidence; do not infer from evaluated HaWoR.",
        "affected": ["031 bilateral Motion quality", "031 bilateral product"],
        "unaffected": ["007 Motion/R0", "Scene diagnostics", "Sensor backend", "Local/HuRo numeric comparison"],
        "evidence": item,
    }
    blockers = [row for row in blockers if row.get("code") != blocker["code"]]
    blockers.append(blocker)
    state.update(blockers=blockers, current_action="031 left remains observability-inconclusive; 007 and 0902 Motion proceed independently.",
                 writer={"pid": os.getpid(), "proc_start_ticks": None}, updated_at=now())
    write_json(state_path, state)


def main() -> int:
    funnel = contact_funnel()
    render = renderer_four_way()
    update_motion_observability_state()
    funnel_path = ATTEMPT / "lanes/lane1_scene/contact_r1_funnel_wave3/RESULT.json"
    render_path = ATTEMPT / "lanes/lane4_compare/renderer_four_way_wave3/RESULT.json"
    append_state("lane1_scene", [funnel_path], "Contact/R1 funnel terminalized per session; no R1 window was authorized or executed.")
    append_state("lane4_compare", [render_path], "Actual pinned-asset q/camera/background isolation render passed; method adoption remains closed.")
    progress = {
        "schema_version": "HUMAN_TO_ROBOT_R2_PROGRESS_V1",
        "task_id": TASK,
        "stage": "WAVE3_CONTACT_FUNNEL_AND_RENDERER_ISOLATION",
        "observed_at": now(),
        "contact_r1": {
            "sessions": len(funnel["sessions"]),
            "executed_windows": funnel["r1_windows_executed"],
            "adopted_windows": funnel["r1_windows_adopted"],
            "first_blockers": {row["session_id"]: row["first_blocker"] for row in funnel["sessions"]},
        },
        "renderer": {key: render[key] for key in (
            "fixed_repeat_byte_exact", "q_only_changed_pixels", "camera_only_changed_pixels",
            "background_only_changed_pixels", "background_changed_robot_pixels",
        )},
        "clean_sessions": 0,
        "product_sessions": 0,
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
    }
    write_json(ATTEMPT / "PROGRESS_WAVE3.json", progress)
    print(json.dumps(progress, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
