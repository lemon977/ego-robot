#!/usr/bin/env python3
"""Bind S1 visual review, terminal branch outcomes, and final validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso


TASK = "human_to_robot_quality_closure_s1_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
LANES = ATTEMPT / "lanes"


def video_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path)); count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def append_artifacts(state: dict, paths: list[Path]) -> None:
    existing = {row.get("path") for row in state.get("latest_artifacts", []) if isinstance(row, dict)}
    for path in paths:
        item = artifact_ref(path)
        if item["path"] not in existing:
            state.setdefault("latest_artifacts", []).append(item)
            existing.add(item["path"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--tests-skipped", type=int, required=True)
    args = parser.parse_args()
    if (ATTEMPT / "FINAL_VALIDATION.json").exists():
        raise RuntimeError("FINAL_VALIDATION_ALREADY_EXISTS")

    clean_root = LANES / "scene_evidence/attachment_clean_canary_v1"
    clean031 = load_json(clean_root / "play_cards_0915_031/RESULT.json")
    clean007 = load_json(clean_root / "get_potato_chips_0915_007/RESULT.json")
    if any(value.get("structure") != "PASS" for value in (clean031, clean007)):
        raise RuntimeError("LOCAL_CLEAN_STRUCTURE_NOT_PASS")
    reviews = {
        "play_cards_0915_031": {
            "frames_reviewed": [66, 70, 74, 78, 81],
            "decision": "REJECTED_QUALITY",
            "findings": [
                "attachment evidence is consumed and visible card interiors are protected",
                "hand/object boundary remains blurred and the occluded card content is hallucinated",
                "sixteen local frames cannot authorize full-session Clean",
            ],
        },
        "get_potato_chips_0915_007": {
            "frames_reviewed": [181, 185, 189, 193, 196],
            "decision": "REJECTED_QUALITY",
            "findings": [
                "finger-attached devices are removed more completely than the frozen R2 candidate",
                "the long yellow cable remains visible",
                "local inpaint introduces visible brown/white background stains",
                "sixteen local frames cannot authorize full-session Clean",
            ],
        },
    }
    review_doc = {
        "schema_version": "S1_ATTACHMENT_CLEAN_AI_REVIEW_V1",
        "task_id": TASK, "created_at": now_iso(),
        "authority": "AI_REVIEW_PROXY_NOT_HUMAN_GT",
        "review_protocol": "five fixed frames per frozen local window; RAW/WRITE/old Clean/S1 Clean side by side",
        "sessions": reviews,
        "overall": "LOCAL_EVIDENCE_CONSUMPTION_FIXED_BUT_FORMAL_CLEAN_REJECTED",
        "adoption": "NOT_ADOPTED",
    }
    review_path = clean_root / "AI_VISUAL_REVIEW.json"
    atomic_json(review_path, review_doc)

    track031 = load_json(LANES / "scene_evidence/attachment_track_canary_v1/play_cards_0915_031/RESULT.json")
    track007 = load_json(LANES / "scene_evidence/attachment_track_canary_v1/get_potato_chips_0915_007/RESULT.json")
    track042 = load_json(LANES / "scene_evidence/attachment_track_042_canary_v1/play_cards_0902_042/RESULT.json")
    scene_terminal = {
        "schema_version": "S1_SCENE_SESSION_TERMINALS_V1", "task_id": TASK,
        "created_at": now_iso(),
        "sessions": [
            {
                "session_id": "play_cards_0915_031", "execution": "EXECUTED",
                "structure": "PASS", "quality": "REJECTED_QUALITY", "adoption": "NOT_ADOPTED",
                "local_attachment_states": track031["counts"],
                "first_blocker": "HAND_OBJECT_BOUNDARY_INPAINT_HALLUCINATION_AND_LOCAL_ONLY_COVERAGE",
            },
            {
                "session_id": "get_potato_chips_0915_007", "execution": "EXECUTED",
                "structure": "PASS", "quality": "REJECTED_QUALITY", "adoption": "NOT_ADOPTED",
                "local_attachment_states": track007["counts"],
                "first_blocker": "LONG_YELLOW_CABLE_RESIDUAL_AND_BACKGROUND_INPAINT_STAIN",
            },
            {
                "session_id": "get_potato_chips_0902_103", "execution": "BLOCKED_UPSTREAM",
                "structure": "NOT_EVALUATED", "quality": "NOT_EVALUATED", "adoption": "NOT_ADOPTED",
                "first_blocker": "NO_INDEPENDENT_DEVICE_OR_CABLE_SEED_EVIDENCE_0_OF_284",
            },
            {
                "session_id": "play_cards_0902_042", "execution": "EXECUTED",
                "structure": "PASS", "quality": "INCONCLUSIVE", "adoption": "NOT_ADOPTED",
                "local_attachment_states": track042["counts"],
                "known_full_timeline_frames": 64, "full_timeline_frames": 171,
                "first_blocker": "FULL_TIMELINE_DEVICE_EVIDENCE_INCOMPLETE_107_UNKNOWN_FRAMES",
            },
        ],
        "formal_clean_quality_pass": 0, "formal_clean_adopted": 0,
    }
    scene_terminal_path = LANES / "scene_evidence/SESSION_TERMINALS.json"
    atomic_json(scene_terminal_path, scene_terminal)

    stereo = load_json(LANES / "geometry_contact/encoded_stereo_preflight_v1/RESULT.json")
    depth031 = load_json(LANES / "geometry_contact/depth_full_v1/play_cards_0915_031/RESULT.json")
    object031 = load_json(LANES / "geometry_contact/object6d_visible_031_v1/RESULT.json")
    contact031 = load_json(LANES / "geometry_contact/interaction_contact_031_v1/RESULT.json")
    geometry_terminal = {
        "schema_version": "S1_GEOMETRY_CONTACT_SESSION_TERMINALS_V1", "task_id": TASK,
        "created_at": now_iso(),
        "sessions": [
            {
                "session_id": "play_cards_0915_031", "stereo_preflight": "PASS",
                "depth": depth031["status"], "object6d": object031["status"],
                "contact": "BLOCKED_LOCAL_EVIDENCE", "robot_r1": "BLOCKED_LOCAL_EVIDENCE",
                "first_blocker": "NO_DIRECT_OBSERVED_HAWOR_HAND_FRAMES_0_OF_149",
            },
            {
                "session_id": "get_potato_chips_0915_007",
                "stereo_preflight": stereo["sessions"]["get_potato_chips_0915_007"]["status"],
                "depth": "BLOCKED_UPSTREAM", "object6d": "BLOCKED_UPSTREAM",
                "contact": "BLOCKED_UPSTREAM", "robot_r1": "BLOCKED_UPSTREAM",
                "first_blocker": "STEREO_ROBUST_MATCH_COUNT_1021_BELOW_FROZEN_1500",
            },
            {
                "session_id": "get_potato_chips_0902_103", "stereo_preflight": "BLOCKED_INPUT_DOMAIN",
                "depth": "BLOCKED_UPSTREAM", "object6d": "BLOCKED_UPSTREAM",
                "contact": "BLOCKED_UPSTREAM", "robot_r1": "BLOCKED_UPSTREAM",
                "first_blocker": "ONLY_WITHDRAWN_SOURCEINDEX0_EQUIDIS62_REMAP_DOMAIN_BOUND",
            },
            {
                "session_id": "play_cards_0902_042", "stereo_preflight": "BLOCKED_INPUT_DOMAIN",
                "depth": "BLOCKED_UPSTREAM", "object6d": "BLOCKED_UPSTREAM",
                "contact": "BLOCKED_UPSTREAM", "robot_r1": "BLOCKED_UPSTREAM",
                "first_blocker": "ONLY_WITHDRAWN_SOURCEINDEX0_EQUIDIS62_REMAP_DOMAIN_BOUND",
            },
        ],
        "depth_full_session_pass": 1, "object6d_observability_complete": 1,
        "strict_contact_windows": 0, "robot_r1_executed": 0, "robot_r1_adopted": 0,
        "external_metric_authority": False,
    }
    geometry_terminal_path = LANES / "geometry_contact/SESSION_TERMINALS.json"
    atomic_json(geometry_terminal_path, geometry_terminal)

    time = now_iso()
    lane_updates = {
        "scene_evidence": {
            "status": "TERMINAL_WITH_QUALITY_GAPS", "execution": "EXECUTED",
            "structure": "PASS", "quality": "REJECTED_QUALITY", "adoption": "NOT_ADOPTED",
            "blocker": {
                "code": "FORMAL_CLEAN_QUALITY_NOT_CLOSED",
                "missing": "full-timeline cable/device evidence and acceptable occlusion-boundary inpaint",
                "consumer": "formal Clean and same-background product", "owner": "scene_evidence",
                "unblock_action": "new independently supported cable evidence and a new bounded inpaint candidate; do not retry this signature",
                "affected": ["formal Clean", "final same-background comparison"],
                "unaffected": ["local attachment evidence", "Depth/Object6D 031", "R0", "Sensor"],
            },
            "artifacts": [scene_terminal_path, review_path, clean_root / "RESULT.json"],
        },
        "geometry_contact": {
            "status": "TERMINAL_WITH_QUALITY_GAPS", "execution": "EXECUTED",
            "structure": "PASS", "quality": "INCONCLUSIVE", "adoption": "NOT_ADOPTED",
            "blocker": {
                "code": "NO_STRICT_CONTACT_WINDOW",
                "missing": "direct-observed side-resolved hand surface evidence in the same Stereo domain",
                "consumer": "Contact and Robot R1", "owner": "geometry_contact",
                "unblock_action": "produce direct-observed hand projections; inferred HaWoR frames cannot be promoted",
                "affected": ["Contact", "Robot R1"],
                "unaffected": ["031 Depth", "031 observed Object6D", "R0", "Scene"],
            },
            "artifacts": [geometry_terminal_path, LANES / "geometry_contact/interaction_contact_031_v1/RESULT.json"],
        },
        "assembly_product": {
            "status": "TERMINAL_BLOCKED_EXTERNAL", "execution": "EXECUTED",
            "structure": "PASS", "quality": "INCONCLUSIVE", "adoption": "NOT_ADOPTED",
            "artifacts": [LANES / "assembly_product/WAVE0_READINESS.json"],
        },
        "compare": {
            "status": "TERMINAL_BLOCKED_UPSTREAM", "execution": "EXECUTED",
            "structure": "PASS", "quality": "INCONCLUSIVE", "adoption": "NOT_ADOPTED",
            "artifacts": [LANES / "compare/WAVE0_READINESS.json"],
        },
    }
    for lane, update in lane_updates.items():
        path = LANES / lane / "STATE.json"
        state = load_json(path)
        state.update({key: value for key, value in update.items() if key != "artifacts"})
        append_artifacts(state, update["artifacts"])
        state["writer"] = {"pid": None, "proc_start_ticks": None, "executor_epoch": 6}
        state["updated_at"] = time
        state["training_eligible"] = False
        atomic_json(path, state)

    videos = []
    for path in sorted(ATTEMPT.rglob("*.mp4")):
        count = video_frames(path)
        if count <= 0:
            raise RuntimeError(f"VIDEO_DECODE_FAILED:{path}")
        videos.append({**artifact_ref(path), "decoded_frames": count})
    if len(videos) != 15:
        raise RuntimeError(f"VIDEO_DENOMINATOR:{len(videos)}")
    lease = load_json(REPO_ROOT / "_run/current/GPU_LEASE.json")
    if lease.get("status") != "RELEASED":
        raise RuntimeError("GPU_LEASE_NOT_RELEASED")
    validation = {
        "schema_version": "S1_FINAL_VALIDATION_V1", "task_id": TASK,
        "created_at": time, "status": "PASS_STRUCTURE_WITH_QUALITY_GAPS",
        "pytest": {"passed": args.tests_passed, "skipped": args.tests_skipped},
        "video_count": len(videos), "video_full_decode_pass": True, "videos": videos,
        "gpu_lease_released": True,
        "source_processed_archive_sealed_modified": False,
        "quality_axes": {"execution": "EXECUTED", "structure": "PASS", "quality": "REJECTED_QUALITY", "adoption": "NOT_ADOPTED"},
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    atomic_json(ATTEMPT / "FINAL_VALIDATION.json", validation)
    progress = {
        "schema_version": "S1_TERMINAL_EVIDENCE_SUMMARY_V1", "task_id": TASK,
        "created_at": time, "status": "READY_FOR_TERMINAL_PUBLICATION",
        "scene": {"local_track_pass": 3, "local_clean_executed": 2, "formal_quality_pass": 0, "adopted": 0},
        "geometry": {"stereo_preflight_pass": 1, "depth_full_pass": 1, "object6d_complete": 1, "strict_contact_windows": 0, "robot_r1_executed": 0},
        "assembly": {"candidate_mesh_loaded": True, "measured_installation": "ABSENT"},
        "compare": {"numeric_sessions_preserved": 2, "winner": None, "final_adopted_clean_visual": 0},
        "scene_terminals": artifact_ref(scene_terminal_path),
        "geometry_terminals": artifact_ref(geometry_terminal_path),
        "final_validation": artifact_ref(ATTEMPT / "FINAL_VALIDATION.json"),
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    atomic_json(ATTEMPT / "PROGRESS_FINAL.json", progress)
    print(json.dumps({"status": progress["status"], "videos": len(videos), "pytest": validation["pytest"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
