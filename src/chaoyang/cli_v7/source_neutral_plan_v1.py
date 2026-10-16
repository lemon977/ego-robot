"""Source-neutral, check-only plan for the existing tracker and VST RGB domains.

This code creates no Task Packet and never invokes a stage executor. A new
finite code-acceptance packet must precede any executable successor.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import re
from typing import Any

from chaoyang.cli.exact78_first_submit_bridge_candidate import _admit_known_domain
from chaoyang.governance import execute_exact78_first_submit_cpu_candidate_v1 as cpu
from chaoyang.governance.common import REPO_ROOT, TASK_STATE_PATH, load_json
from chaoyang.governance.exact78_cold_start_guard_v1 import (
    DATA_ROOT, ColdStartInputError, file_ref, inspect_session,
)


SCHEMA = "exact78-generic-submit-check-manifest-v1"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
ACTIVE = frozenset({"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"})
ROLES = {
    "TRACKER_CLIP_RGB": frozenset({"video", "source_stereo_video", "camera_params",
                                   "slam", "clip_manifest"}),
    "0915_RAW_VST": frozenset({"raw/vst.h264", "raw/vst.qpc.ts.jsonl",
                               "raw/vst.ts.jsonl", "raw/camera_params.json",
                               "raw/camera_params.meta.json"}),
}
FORBIDDEN = frozenset({"preprocess", "all_data", "trackingdata", "processed",
                       "clean", "robot", "pico", "manus", "tactile", "hand", "controller"})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _source_root(session: Path, allowed_root: Path) -> Path:
    if session.is_symlink():
        raise ColdStartInputError("SYMLINK_SESSION_FORBIDDEN")
    path = session.resolve(strict=True)
    root = allowed_root.resolve(strict=True)
    if not path.is_dir() or not path.is_relative_to(root):
        raise ColdStartInputError("SOURCE_OUTSIDE_ALLOWED_ROOT")
    if any(part.lower() in FORBIDDEN for part in path.relative_to(root).parts):
        raise ColdStartInputError("DERIVED_OR_HAND_MOTION_SESSION_FORBIDDEN")
    relative = path.relative_to(root).parts
    if len(relative) < 2:
        raise ColdStartInputError("SOURCE_DOMAIN_UNVERIFIED")
    if not (relative[0] in {"chips_cards_tracker_0901", "chips_cards_tracker_0902",
                            "chips_cards_hands_0915"}):
        raise ColdStartInputError("SOURCE_DOMAIN_UNVERIFIED")
    return path


def _source_evidence(session: Path, allowed_root: Path) -> dict[str, Any]:
    path = _source_root(session, allowed_root)
    # This legacy precheck examines only the five consumed tracker roles, before
    # inspect_session hashes them; historical manifest sidecars are never inputs.
    cpu._precheck_declared_paths(path, allowed_root=allowed_root)
    source = inspect_session(path, allowed_root=allowed_root)
    kind = source["kind"]
    if kind not in ROLES or set(source["files"]) != ROLES[kind]:
        raise ColdStartInputError("SOURCE_ROLE_SET_UNSUPPORTED")
    if (kind == "TRACKER_CLIP_RGB") != path.relative_to(allowed_root.resolve()).parts[0].startswith("chips_cards_tracker_"):
        raise ColdStartInputError("SOURCE_DOMAIN_ROLE_CONFLICT")
    domain = _admit_known_domain(source)
    for role, ref in source["files"].items():
        raw = Path(ref["path"])
        if (not raw.is_relative_to(path)
                or any(part.lower() in FORBIDDEN for part in raw.relative_to(path).parts)
                or file_ref(raw) != ref):
            raise ColdStartInputError("SOURCE_ROLE_SHA_OR_SCOPE_DRIFT:" + role)
    return {"source": source, "domain": domain}


def _writer_state() -> str:
    state, index = load_json(TASK_STATE_PATH), load_json(INDEX)
    active_rows = [row for row in index.get("task_packets", [])
                   if row.get("execution_allowed") is True]
    active_tasks = [row for row in state.get("tasks", []) if row.get("status") in ACTIVE]
    if len(active_rows) > 1 or len(active_tasks) > 1:
        raise ColdStartInputError("DOUBLE_WRITER_OR_MULTIPLE_ACTIVE_PACKETS")
    if bool(active_rows) != bool(active_tasks) or bool(state.get("next_task")) != bool(active_rows):
        raise ColdStartInputError("ACTIVE_WRITER_LEDGER_INDEX_DRIFT")
    if active_rows and active_rows[0].get("task_id") != active_tasks[0].get("task_id"):
        raise ColdStartInputError("ACTIVE_WRITER_LEDGER_INDEX_DRIFT")
    return "WAIT_ACTIVE_WRITER" if active_rows else "READY_FOR_FUTURE_CODE_ACCEPTANCE_ONLY"


def plan_one(session: Path, *, expected_sha: str | None = None,
             allowed_root: Path = DATA_ROOT) -> dict[str, Any]:
    evidence = _source_evidence(session, allowed_root)
    source, domain = evidence["source"], evidence["domain"]
    identity = {"schema_version": "exact78-source-neutral-check-plan-v1",
                "source_kind": source["kind"], "session_root": source["session_root"],
                "session_id": source["session_id"], "source_files": source["files"],
                "physical_eye": domain["physical_eye"], "crop_xywh": domain["crop_xywh"],
                "time_mode": domain["time_mode"],
                "capture_to_decode_alignment": domain["capture_to_decode_alignment"],
                "frame_count_declared": domain["declared_frame_count"],
                "fps_nominal": domain["fps_nominal"],
                "calibration_claim": domain["calibration_claim"]}
    signature = _digest(identity)
    if expected_sha is not None and (not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
                                     or not hmac.compare_digest(signature, expected_sha)):
        raise ColdStartInputError("EXPECTED_INPUT_SHA_DRIFT")
    return {"status": _writer_state(), "source": identity,
            "submission_input_sha256": signature,
            "candidate_stages": ["CPU_ADMISSION", "HAWOR_V4", "ROBOT_V1"],
            "stage_packet_construction": "NOT_IMPLEMENTED",
            "decode_verified": False, "independent_source_group": "UNVERIFIED",
            "world_pose_authority": False, "metric_authority": False,
            "control_ground_truth": False, "visual_qualified": False,
            "training_eligible": False, "registration_started": False,
            "model_calls": 0}


def plan_manifest(path: Path, *, expected_sha: str,
                  allowed_root: Path = DATA_ROOT) -> dict[str, Any]:
    if path.is_symlink():
        raise ColdStartInputError("SYMLINK_MANIFEST_FORBIDDEN")
    manifest = path.resolve(strict=True)
    if not manifest.is_file() or not manifest.is_relative_to(REPO_ROOT.resolve()):
        raise ColdStartInputError("MANIFEST_OUTSIDE_PROJECT")
    ref = file_ref(manifest)
    if (not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or not hmac.compare_digest(ref["sha256"], expected_sha)):
        raise ColdStartInputError("EXPECTED_MANIFEST_SHA_DRIFT")
    value = load_json(manifest)
    rows = value.get("sessions") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or set(value) != {"schema_version", "sessions"}
            or value["schema_version"] != SCHEMA or not isinstance(rows, list)
            or not rows or len(rows) > 32):
        raise ColdStartInputError("MANIFEST_SCHEMA_INVALID")
    if any(not isinstance(row, dict) or set(row) != {"path", "expected_input_sha256"}
           or not isinstance(row["path"], str) or not isinstance(row["expected_input_sha256"], str)
           for row in rows):
        raise ColdStartInputError("MANIFEST_ROW_INVALID")
    if len({row["path"] for row in rows}) != len(rows):
        raise ColdStartInputError("MANIFEST_DUPLICATE_SESSION")
    decisions = [plan_one(Path(row["path"]), expected_sha=row["expected_input_sha256"],
                          allowed_root=allowed_root) for row in rows]
    if len({row["source"]["session_root"] for row in decisions}) != len(decisions):
        raise ColdStartInputError("MANIFEST_ALIAS_DUPLICATE_SESSION")
    return {"status": "CHECK_ONLY_MATRIX", "manifest": ref, "sessions": decisions,
            "batch_execution_authorized": False, "registration_started": False,
            "model_calls": 0}


__all__ = ["plan_one", "plan_manifest"]
