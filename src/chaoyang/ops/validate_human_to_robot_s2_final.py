#!/usr/bin/env python3
"""Fail-closed final structural/media validation for Human-to-Robot S2."""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json


TASK = "human_to_robot_evidence_unlock_s2_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
PRODUCTS = {
    "get_potato_chips_0915_007": (ROOT / "lanes/motion_product/formal_product_007/attempt_0002/PRODUCT_RESULT.json", 378),
    "play_cards_0915_031": (ROOT / "lanes/motion_product/formal_product_031/attempt_0004/PRODUCT_RESULT.json", 149),
    "get_potato_chips_0902_103": (ROOT / "lanes/motion_product/formal_product_get_potato_chips_0902_103/attempt_0002/PRODUCT_RESULT.json", 284),
    "play_cards_0902_042": (ROOT / "lanes/motion_product/formal_product_play_cards_0902_042/attempt_0002/PRODUCT_RESULT.json", 171),
}
LANES = ("scene", "sensor", "motion_product", "compare")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _checked(item: dict) -> Path:
    path = Path(str(item["path"])).resolve(strict=True)
    actual = artifact_ref(path)
    # Artifact references may carry non-authority annotations such as
    # ``decoded_frames``.  Integrity is defined by the canonical three fields;
    # requiring whole-dict equality incorrectly rejects such annotated refs.
    if any(item.get(key) != actual[key] for key in ("path", "bytes", "sha256")):
        raise RuntimeError(f"ARTIFACT_REF_DRIFT:{path}")
    return path


def _decode(item: dict, expected: int, role: str, session_id: str) -> dict:
    path = _checked(item)
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames", "-of",
        "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    observed = process.stdout.strip()
    if process.returncode != 0 or observed != str(expected):
        raise RuntimeError(f"VIDEO_DECODE:{role}:{session_id}:{observed}:{expected}")
    return {"role": role, "session_id": session_id, "frames": expected, "video": item}


def _junit(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise RuntimeError(f"JUNIT_EMPTY:{path}")
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def _reload_npz(item: dict, label: str) -> dict:
    path = _checked(item)
    with np.load(path, allow_pickle=False) as archive:
        names = sorted(archive.files)
        if not names:
            raise RuntimeError(f"EMPTY_NPZ:{label}:{path}")
        shapes = {name: list(np.asarray(archive[name]).shape) for name in names}
    return {"label": label, "artifact": item, "arrays": len(names), "shapes": shapes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-early-close", action="store_true")
    args = parser.parse_args()
    output = ROOT / "FINAL_VALIDATION.json"
    if output.exists():
        raise RuntimeError(f"IMMUTABLE_FINAL_VALIDATION_EXISTS:{output}")
    state = load_json(REPO_ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    row = next(item for item in state["tasks"] if item.get("task_id") == TASK)
    elapsed = (datetime.now().astimezone() - datetime.fromisoformat(str(row["t0"]))).total_seconds()
    if elapsed < 10 * 3600 and not args.allow_early_close:
        raise RuntimeError(f"H10_FINAL_VALIDATION_NOT_REACHED:{elapsed:.1f}")
    h9 = ROOT / "checkpoints/H9_RESULT.json"
    if not h9.is_file():
        raise RuntimeError("H9_COMPONENT_FREEZE_MISSING")

    videos: list[dict] = []
    products: dict[str, dict] = {}
    for session_id, (path, expected) in PRODUCTS.items():
        result = load_json(path)
        if result.get("session_id") != session_id:
            raise RuntimeError(f"PRODUCT_SESSION:{session_id}")
        if result.get("execution") != "EXECUTED" or result.get("structure") != "PASS":
            raise RuntimeError(f"PRODUCT_NOT_STRUCTURAL:{session_id}")
        if result.get("quality") != "REJECTED_QUALITY" or result.get("adoption") != "CANDIDATE_ONLY":
            raise RuntimeError(f"PRODUCT_QUALITY_PROMOTION_UNSUPPORTED:{session_id}")
        if result.get("renderer_interface") != "RGB_ALPHA_OPTICAL_DEPTH_VALID_COMPONENT_ID_V1":
            raise RuntimeError(f"PRODUCT_RENDERER_INTERFACE:{session_id}")
        if result.get("compositor_interface") != "VISIBLE_SURFACE_OWNERSHIP_V1_NO_RGB_FALLBACK":
            raise RuntimeError(f"PRODUCT_COMPOSITOR_INTERFACE:{session_id}")
        if int(result.get("decoded_frames", -1)) != expected or int(result.get("expected_frames", -1)) != expected:
            raise RuntimeError(f"PRODUCT_FRAME_LEDGER:{session_id}")
        videos.append(_decode(result["product_video"], expected, "FORMAL_PRODUCT_CANDIDATE", session_id))
        products[session_id] = {
            "result": artifact_ref(path),
            "video": result["product_video"],
            "quality": result["quality"],
            "adoption": result["adoption"],
            "known_decision_coverage": result["known_decision_coverage"],
            "unknown_decision_ratio": result["unknown_decision_ratio"],
            "adapter_visible_frames": result["adapter_visible_frames"],
        }

    sensor = load_json(ROOT / "lanes/sensor/reuse_audit_v1/RESULT.json")
    if sensor.get("structure") != "PASS" or len(sensor.get("sessions", [])) != 3:
        raise RuntimeError("SENSOR_REUSE_NOT_CLOSED")
    npz: list[dict] = []
    for item in sensor["sessions"]:
        expected = int(item["expected_frames"])
        videos.append(_decode(item["video"], expected, "SENSOR_COMMON_BACKEND_REVIEW", item["session_id"]))
        underlying = load_json(_checked(item["result"]))
        npz.append(_reload_npz(underlying["hand_motion"], f"{item['session_id']}:HAND_MOTION"))
        npz.append(_reload_npz(underlying["robot_backend"], f"{item['session_id']}:ROBOT_BACKEND"))

    compare_results = {
        "get_potato_chips_0915_007": ROOT / "lanes/compare/adapter_refresh_007/attempt_0001/RESULT.json",
        "play_cards_0915_031": ROOT / "lanes/compare/adapter_refresh_031/attempt_0001/RESULT.json",
    }
    for session_id, path in compare_results.items():
        result = load_json(path)
        if result.get("structure") != "PASS" or result.get("winner") is not None:
            raise RuntimeError(f"COMPARE_SCOPE:{session_id}")
        if result.get("real_adapter_consumed_both_methods") is not True:
            raise RuntimeError(f"COMPARE_ADAPTER_NOT_CONSUMED:{session_id}")
        videos.append(_decode(result["video"], int(result["frame_count"]), "LOCAL_HURO_ADAPTER_DIAGNOSTIC", session_id))

    occlusion = load_json(ROOT / "lanes/scene/h3_occlusion_031/attempt_0003/RESULT.json")
    if occlusion.get("quality") != "INCONCLUSIVE" or occlusion.get("human_equipment_raw_depth_excluded") is not True:
        raise RuntimeError("OCCLUSION_SCOPE_PROMOTED_OR_INVALID")
    videos.append(_decode(occlusion["video"], 16, "H3_OCCLUSION_ABC_DIAGNOSTIC", "play_cards_0915_031"))
    attachment_path = REPO_ROOT / "_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/attachment_clean_canary_v1/get_potato_chips_0915_007/RESULT.json"
    attachment = load_json(attachment_path)
    if attachment.get("full_session_clean_authority") is not False:
        raise RuntimeError("ATTACHMENT_CANARY_SCOPE_PROMOTED")
    videos.append(_decode(attachment["review_video"], 16, "BOUNDED_ATTACHMENT_CLEAN_DIAGNOSTIC", "get_potato_chips_0915_007"))

    if len(videos) != 11:
        raise RuntimeError(f"EXPECTED_ELEVEN_CANONICAL_VIDEOS:{len(videos)}")
    resume_path = ROOT / "formal_entry_resume/attempt_0002/RESULT.json"
    resume = load_json(resume_path)
    if resume.get("status") != "PASS" or resume.get("reused") != 4 or resume.get("mutated") != 0:
        raise RuntimeError("FORMAL_ENTRY_RESUME_NOT_PASS")

    project = _junit(ROOT / "PYTEST_PRE_H3_PROJECT_LOCAL.xml")
    collision = _junit(ROOT / "PYTEST_COLLISION_PRE_H3_FIXED.xml")
    closure = _junit(ROOT / "PYTEST_S2_CLOSURE_CURRENT.xml")
    if project != {"tests": 1490, "failures": 0, "errors": 0, "skipped": 1}:
        raise RuntimeError(f"PROJECT_LOCAL_TESTS:{project}")
    if collision != {"tests": 6, "failures": 0, "errors": 0, "skipped": 0}:
        raise RuntimeError(f"COLLISION_TESTS:{collision}")
    if closure != {"tests": 41, "failures": 0, "errors": 0, "skipped": 0}:
        raise RuntimeError(f"S2_CLOSURE_TESTS:{closure}")

    lane_states = {}
    for lane in LANES:
        path = ROOT / f"lanes/{lane}/STATE.json"
        value = load_json(path)
        if value.get("checkpoint") != "H9":
            raise RuntimeError(f"LANE_NOT_H9_FROZEN:{lane}")
        # STATE.json is intentionally updated again during terminalization.  Keep
        # the H9 semantic snapshot here rather than a soon-to-be-stale artifact
        # reference; the terminal result binds the final STATE.json bytes.
        lane_states[lane] = {
            key: value.get(key)
            for key in (
                "status",
                "execution",
                "structure",
                "quality",
                "adoption",
                "checkpoint",
                "updated_at",
            )
        }

    result = {
        "schema_version": "HUMAN_TO_ROBOT_S2_FINAL_VALIDATION_V1",
        "task_id": TASK,
        "created_at": _now(),
        "elapsed_seconds": elapsed,
        "nominal_time_gate_seconds": 10 * 3600,
        "early_close_authorized": bool(args.allow_early_close and elapsed < 10 * 3600),
        "status": "PASS_STRUCTURE_WITH_QUALITY_GAPS",
        "products": products,
        "formal_entry_resume": artifact_ref(resume_path),
        "videos": videos,
        "video_count": len(videos),
        "video_full_decode_pass": True,
        "npz_reload": npz,
        "npz_reload_pass": True,
        "checkpoints": {
            name: artifact_ref(ROOT / f"checkpoints/{name}_RESULT.json")
            for name in ("H3", "H6", "H9")
        },
        "lane_states": lane_states,
        "regression": {
            "project_local_main": project,
            "collision_dedicated_fixture": collision,
            "s2_current_closure": closure,
            "hardlink_transaction_tests": "14_NOT_EVALUATED_CPFS_PROJECT_LOCAL_TMPDIR",
            "full_suite_claim": False,
        },
        "quality_summary": {
            "formal_product_candidates": 4,
            "formal_product_quality_pass": 0,
            "formal_product_adopted": 0,
            "sensor_sessions_reverified": 3,
            "local_huro_winner": None,
            "robot_r1_screened": 0,
            "robot_r1_executed": 0,
            "robot_r1_adopted": 0,
        },
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
        "claim_limit": (
            "Structural/media/reload closure with explicit quality gaps. Four structural products, "
            "zero quality passes, zero adoptions, no completed Contact/Robot R1 screening or execution, "
            "and no full-suite filesystem claim."
        ),
    }
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "videos": len(videos), "products": "4/4", "quality_pass": "0/4", "output": artifact_ref(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
