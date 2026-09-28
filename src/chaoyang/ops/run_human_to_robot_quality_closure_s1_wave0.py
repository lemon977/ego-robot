#!/usr/bin/env python3
"""Produce the first evidence/readiness wave for the R2 quality successor."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from chaoyang.governance.common import artifact_ref, atomic_json, load_json

REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_quality_closure_s1_20260923"
ATTEMPT = REPO / f"_run/current/{TASK}/attempts/attempt_0001"
R2 = REPO / "_run/current/human_to_robot_root_cause_gated_r2_20260923/attempts/attempt_0001"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_once(path: Path, value: dict) -> dict:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"IMMUTABLE_CONFLICT:{path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    return artifact_ref(path)


def main() -> int:
    scene_rows = []
    sources = {
        "031": ("play_cards_0915_031", 149),
        "007": ("get_potato_chips_0915_007", 378),
        "103": ("get_potato_chips_0902_103", 284),
        "042": ("play_cards_0902_042", 171),
    }
    for short, (session, frames) in sources.items():
        conservative_path = R2 / f"lanes/lane1_scene/conservative_repair_{short}_wave1/RESULT.json"
        conservative = load_json(conservative_path)
        seed_frames: list[int] = []
        diagnostic_ref = None
        diagnostic_path = R2 / f"lanes/lane1_scene/reference_diagnostic_{short}_wave0/RESULT.json"
        if diagnostic_path.is_file():
            diagnostic = load_json(diagnostic_path)
            seed_frames = [int(row["frame_id"]) for row in diagnostic["rows"] if int(row["device_px"]) > 0]
            diagnostic_ref = artifact_ref(diagnostic_path)
        device_frames = int(conservative["device_nonempty_frames"])
        if len(seed_frames) not in {0, device_frames}:
            raise RuntimeError(f"DEVICE_FRAME_COUNT_MISMATCH:{short}")
        scene_rows.append({
            "session_id": session,
            "frame_count": frames,
            "existing_device_evidence_frames": device_frames,
            "existing_device_evidence_fraction": device_frames / frames,
            "known_seed_frame_ids": seed_frames,
            "independent_full_timeline_device_evidence": False,
            "same_signature_scene_retry_allowed": False,
            "next_action": (
                "BUILD_AND_VALIDATE_LOCAL_ATTACHMENT_TRACK_FROM_EXISTING_SEEDS"
                if seed_frames else "NO_SEED_EVIDENCE_TERMINAL_UNLESS_NEW_INDEPENDENT_SOURCE_EXISTS"
            ),
            "conservative_candidate": artifact_ref(conservative_path),
            "reference_diagnostic": diagnostic_ref,
        })
    evidence = {
        "schema_version": "HUMAN_TO_ROBOT_QUALITY_CLOSURE_SCENE_EVIDENCE_READINESS_V1",
        "task_id": TASK,
        "created_at": now(),
        "status": "READY_FOR_ONE_NEW_SIGNATURE_CANDIDATE_ON_PRIMARY_SEED_WINDOWS",
        "sessions": scene_rows,
        "aggregate": {
            "sessions": 4,
            "frames": sum(row["frame_count"] for row in scene_rows),
            "existing_device_evidence_frames": sum(row["existing_device_evidence_frames"] for row in scene_rows),
            "full_timeline_independent_device_evidence_sessions": 0,
        },
        "quality_rule": "Seed masks are evidence inputs, not automatic full-timeline truth. Propagation must remain local, identity-bound and UNKNOWN outside validated support.",
        "training_eligible": False,
    }
    evidence_ref = write_once(ATTEMPT / "lanes/scene_evidence/WAVE0_READINESS.json", evidence)

    prior_funnel = R2 / "lanes/lane1_scene/contact_r1_funnel_wave9/RESULT.json"
    funnel = load_json(prior_funnel)
    geometry = {
        "schema_version": "HUMAN_TO_ROBOT_QUALITY_CLOSURE_GEOMETRY_READINESS_V1",
        "task_id": TASK,
        "created_at": now(),
        "status": "NO_CURRENT_TARGET_SESSION_METRIC_WINDOW",
        "sessions": [{
            "session_id": row["session_id"],
            "prior_first_blocker": row["corrected_first_r1_blocker"],
            "eligible_metric_contact_windows": 0,
            "same_session_depth_object6d_bound": False,
            "next_action": "PINNED_SAME_SESSION_DEPTH_PREFLIGHT_BEFORE_ANY_CONTACT",
        } for row in funnel["sessions"]],
        "excluded_cross_session_evidence": "play_cards_0915_001 Depth/Object6D cannot authorize 031/007/0902 sessions.",
        "prior_funnel": artifact_ref(prior_funnel),
        "external_metric_authority": False,
        "training_eligible": False,
    }
    geometry_ref = write_once(ATTEMPT / "lanes/geometry_contact/WAVE0_READINESS.json", geometry)

    assembly_source = R2 / "lanes/lane2_motion/assembly_scene_canary_wave11/RESULT.json"
    assembly = {
        "schema_version": "HUMAN_TO_ROBOT_QUALITY_CLOSURE_ASSEMBLY_READINESS_V1",
        "task_id": TASK,
        "created_at": now(),
        "status": "CANDIDATE_GEOMETRY_ONLY",
        "candidate_mesh_loaded": True,
        "measured_mount": "ABSENT",
        "robot_tcp": "ABSENT",
        "camera_world_to_base": "ABSENT",
        "source": artifact_ref(assembly_source),
        "next_action": "KEEP_COMPLETE_ASSEMBLY_ADOPTION_BLOCKED_UNTIL_MEASURED_EVIDENCE",
        "physical_deployable": False,
    }
    assembly_ref = write_once(ATTEMPT / "lanes/assembly_product/WAVE0_READINESS.json", assembly)

    compare = {
        "schema_version": "HUMAN_TO_ROBOT_QUALITY_CLOSURE_COMPARE_READINESS_V1",
        "task_id": TASK,
        "created_at": now(),
        "status": "NUMERIC_COMPLETE_VISUAL_WAITING_FOR_ADOPTED_CLEAN",
        "numeric_sessions": 2,
        "winner": None,
        "sources": [
            artifact_ref(R2 / "lanes/lane4_compare/same_rejected_background_031_wave10/RESULT.json"),
            artifact_ref(R2 / "lanes/lane4_compare/same_rejected_background_007_wave10/RESULT.json"),
        ],
        "next_action": "DO_NOT_RERUN_NUMERIC_COMPARISON; FINAL_VISUAL_ONLY_AFTER_ADOPTED_SAME_SESSION_CLEAN",
        "control_ground_truth": False,
    }
    compare_ref = write_once(ATTEMPT / "lanes/compare/WAVE0_READINESS.json", compare)

    lane_updates = {
        "scene_evidence": ("RUNNING_WITH_BOUNDED_CANDIDATE", "EXECUTED", "PASS", "INCONCLUSIVE", {
            "code": "FULL_TIMELINE_DEVICE_EVIDENCE_ABSENT",
            "missing": "independent full-timeline device/cable evidence and directly-visible object protection",
            "consumer": "formal Scene/Clean quality",
            "owner": "scene_evidence",
            "unblock_action": "validate one local attachment tracker candidate on frozen seed windows; preserve UNKNOWN elsewhere",
            "affected": ["formal Clean", "same-background adopted product"],
            "unaffected": ["same-session Depth preflight", "R0", "Sensor", "numeric comparison"],
        }, [evidence_ref]),
        "geometry_contact": ("READY_FOR_SAME_SESSION_PREFLIGHT", "EXECUTED", "PASS", "NOT_EVALUATED", {
            "code": "TARGET_SESSION_DEPTH_OBJECT6D_NOT_BOUND",
            "missing": "same-session registered Depth/Object6D finite visible patch",
            "consumer": "real occlusion, Contact and Robot R1",
            "owner": "geometry_contact",
            "unblock_action": "run encoded-domain same-session preflight before any metric inference",
            "affected": ["real occlusion", "Contact", "Robot R1"],
            "unaffected": ["Scene evidence", "R0", "Sensor", "numeric comparison"],
        }, [geometry_ref]),
        "assembly_product": ("BLOCKED_MEASURED_INSTALLATION", "EXECUTED", "PASS", "INCONCLUSIVE", {
            "code": "MEASURED_ASSEMBLY_TRANSFORMS_ABSENT",
            "missing": "measured mount, Robot TCP and camera/world-to-base",
            "consumer": "complete assembly adoption and physical deployment",
            "owner": "assembly_product",
            "unblock_action": "capture measured calibration; do not infer it from STEP",
            "affected": ["complete assembly adoption", "physical deployment"],
            "unaffected": ["offline candidate geometry", "Scene", "R0", "Sensor"],
        }, [assembly_ref]),
        "compare": ("WAITING_ADOPTED_CLEAN_FOR_FINAL_VISUAL", "EXECUTED", "PASS", "INCONCLUSIVE", {
            "code": "ADOPTED_SAME_SESSION_CLEAN_ABSENT",
            "missing": "quality-adopted same-session Clean",
            "consumer": "final same-background visual comparison only",
            "owner": "compare",
            "unblock_action": "consume adopted Clean if Scene quality passes; do not rerun numeric solve",
            "affected": ["final same-background visual comparison"],
            "unaffected": ["existing numeric comparison", "solver diagnostics"],
        }, [compare_ref]),
    }
    for lane, (status, execution, structure, quality, blocker, refs) in lane_updates.items():
        path = ATTEMPT / f"lanes/{lane}/STATE.json"
        value = load_json(path)
        value.update({
            "status": status,
            "execution": execution,
            "structure": structure,
            "quality": quality,
            "adoption": "NOT_ADOPTED",
            "blocker": blocker,
            "latest_artifacts": refs,
            "updated_at": now(),
            "writer": {"pid": None, "proc_start_ticks": None, "executor_epoch": 6},
        })
        atomic_json(path, value)

    progress = {
        "schema_version": "HUMAN_TO_ROBOT_QUALITY_CLOSURE_PROGRESS_V1",
        "task_id": TASK,
        "stage": "WAVE0_EVIDENCE_AND_GEOMETRY_READINESS",
        "created_at": now(),
        "scene": evidence_ref,
        "geometry_contact": geometry_ref,
        "assembly_product": assembly_ref,
        "compare": compare_ref,
        "new_model_invocations": 0,
        "same_signature_scene_retries": 0,
        "next_actions": [
            "local seed-bound attachment tracker canary on 031 and 007",
            "same-session encoded-domain Depth preflight beginning with 031 or 007",
        ],
    }
    out = write_once(ATTEMPT / "PROGRESS_WAVE0.json", progress)
    print(json.dumps({"status": "WAVE0_COMPLETE", "output": out}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
