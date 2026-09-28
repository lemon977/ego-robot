"""Purge only the frozen, unconsumed historical Stereo frame arrays."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, load_json, validate_artifact_ref
from chaoyang.ops.run_human_to_robot_product_first_cable import ROOT, TASK
from chaoyang.ops.run_human_to_robot_product_first_cleanup_batch1 import (
    _tree, isolate_verified, rollback_verified,
)
from chaoyang.ops.run_human_to_robot_product_first_cleanup_batch2_preflight import (
    OLD, OUT, _busy, _decode, _external_refs,
)

SELECTION = OUT / "SELECTED_TARGETS.json"
RECEIPT = OUT / "DELETE_RECEIPT.json"
BINDING = OLD / "W1_INPUT_BINDING.json"
SIGNATURE = OLD / "RUN_SIGNATURE.json"


def _event(state: str, row: dict, **extra: object) -> None:
    with (OUT / "DELETE_EVENTS.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"state": state, "session_id": row["session_id"],
                                 "source": row["source"], **extra}, ensure_ascii=False,
                                sort_keys=True) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def _current_consumers() -> dict:
    manifest = load_json(ROOT / "_run/current/human_to_robot_baseline_v1_convergence_20260923/attempts/attempt_0001/delivery/DELIVERY_MANIFEST.json")
    if len(manifest["slots"]) != 15:
        raise RuntimeError("CURRENT_VIDEO_SLOT_COUNT")
    checked = 0
    for row in manifest["slots"]:
        if validate_artifact_ref(row["video"]):
            raise RuntimeError(f"CURRENT_VIDEO_REF_INVALID:{row['slot_id']}")
        if _decode(Path(row["video"]["path"])) != int(row["expected_frames"]):
            raise RuntimeError(f"CURRENT_VIDEO_DECODE_INVALID:{row['slot_id']}")
        checked += 1
    for relative in (
        "lanes/scene/cable_007/clean_window_v1/RESULT.json",
        "lanes/scene/object_patch_031/window_v1/RESULT.json",
        "lanes/motion_product/robot_correspondence_031/window_v1/RESULT.json",
        "lanes/motion_product/formal_007_current/attempt_0001/PRODUCT_RESULT.json",
        "lanes/motion_product/formal_007_current/RESUME_RECEIPT.json",
    ):
        if not (ROOT / f"_run/current/{TASK}/attempts/attempt_0001" / relative).is_file():
            raise RuntimeError(f"CURRENT_CONSUMER_MISSING:{relative}")
    validation = subprocess.run(["/usr/local/bin/python", "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src"),
                       "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(ROOT / ".cache/tmp"),
                       "XDG_CACHE_HOME": str(ROOT / ".cache/xdg")},
        capture_output=True, text=True, check=False)
    if validation.returncode or '"status": "PASS"' not in validation.stdout:
        raise RuntimeError(f"GOVERNANCE_AFTER_ISOLATION:{validation.stdout[-900:]}")
    return {"fifteen_video_slots": checked, "governance": "PASS",
            "current_007_031_consumers": "PRESENT"}


def _reproducer_inputs(sessions: set[str]) -> dict:
    """Ensure a future authorized replay still has its actual immutable inputs."""
    binding = load_json(BINDING)
    signature = load_json(SIGNATURE)
    if binding.get("task_id") != "0915_robot15h_foundationstereo_waves_v1":
        raise RuntimeError("HISTORICAL_BINDING_CHANGED")
    rows = {row["session_id"]: row for row in binding["sessions"]}
    if not sessions.issubset(rows):
        raise RuntimeError("HISTORICAL_INPUT_NOT_BOUND")
    for session in sessions:
        for field in ("source_stereo", "camera_params"):
            if validate_artifact_ref(rows[session][field]):
                raise RuntimeError(f"HISTORICAL_REPRO_INPUT_MISSING:{session}:{field}")
    for field in ("base_w0_algorithm_runner", "frozen_canary_implementation", "gpu_launcher", "runner", "checkpoint"):
        if validate_artifact_ref(signature["runtime"][field]):
            raise RuntimeError(f"HISTORICAL_REPRO_CODE_OR_WEIGHT_CHANGED:{field}")
    return {"bound_sessions": sorted(sessions), "raw_and_camera_sha_verified": True,
            "runner_and_weight_sha_verified": True,
            "warning": "Raw inputs and producer assets survive, but the original W1 runner's historical release deadline now prevents direct replay; a separately authorized replay route would be needed and byte-identical output is not guaranteed."}


def main() -> int:
    index = load_json(ROOT / "tasks/current/INDEX.json")
    if len(index.get("task_packets", [])) != 1 or index["task_packets"][0].get("task_id") != TASK:
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)
    if (OUT / "HISTORICAL_PAYLOAD_TRIM_NOTICE.json").exists():
        raise RuntimeError("PREEXISTING_TRIM_NOTICE")
    selection = load_json(SELECTION)
    if selection.get("status") != "FROZEN_CANDIDATES_NOT_DELETED" or len(selection.get("targets", [])) != 2:
        raise RuntimeError("BATCH2_NOT_FROZEN")
    packet = load_json(ROOT / f"tasks/current/{TASK}/TASK_PACKET.json")
    conditional = sum(packet["target_sessions"].values(), [])
    for row in selection["targets"]:
        source = Path(row["source"])
        session = row["session_id"]
        if session in conditional or source.parent != OLD / f"sessions/potato_chips/{session}":
            raise RuntimeError(f"TARGET_SCOPE_CHANGED:{session}")
        if source.is_symlink() or not source.is_dir() or _tree(source) != row["identity"]:
            raise RuntimeError(f"TARGET_IDENTITY_CHANGED:{session}")
        if _external_refs(session) or _busy(source):
            raise RuntimeError(f"NEW_CONSUMER_OR_PROCESS:{session}")
        if validate_artifact_ref(row["historical_result"]) or validate_artifact_ref(row["historical_depth_summary"]):
            raise RuntimeError(f"HISTORICAL_RECEIPT_CHANGED:{session}")
        if validate_artifact_ref(row["retained_review"]) or _decode(Path(row["retained_review"]["path"])) != row["frame_count"]:
            raise RuntimeError(f"OLD_REVIEW_CHANGED:{session}")
        for ref in row["retained_samples"].values():
            if validate_artifact_ref(ref):
                raise RuntimeError(f"RETAINED_SAMPLE_CHANGED:{session}")
            with np.load(ref["path"], allow_pickle=False) as archive:
                if not archive.files:
                    raise RuntimeError(f"RETAINED_SAMPLE_EMPTY:{session}")
    if not Path(selection["retained_environment"]).is_file() or not Path(selection["retained_weight"]).is_file():
        raise RuntimeError("REPRODUCER_ENV_OR_WEIGHT_LOST")
    reproducer = _reproducer_inputs({row["session_id"] for row in selection["targets"]})
    isolated = []
    for row in selection["targets"]:
        _event("SELECTED", row, identity=row["identity"])
    try:
        for row in selection["targets"]:
            destination = OUT / f"isolated_{row['session_id']}"
            isolate_verified(Path(row["source"]), destination, row["identity"])
            isolated.append((row, destination))
            _event("ISOLATED", row, isolation=str(destination))
        checks = _current_consumers()
        for row, _ in isolated:
            if validate_artifact_ref(row["historical_result"]) or validate_artifact_ref(row["historical_depth_summary"]):
                raise RuntimeError(f"HISTORICAL_RESULT_MUTATED:{row['session_id']}")
            if _external_refs(row["session_id"]):
                raise RuntimeError(f"NEW_REFERENCE_AFTER_ISOLATION:{row['session_id']}")
            _event("CONSUMER_CHECKED", row, checks="PASS")
        for row, destination in isolated:
            if Path(row["source"]).exists() or _tree(destination) != row["identity"]:
                raise RuntimeError(f"PURGE_IDENTITY_OR_NEW_CONTENT:{row['session_id']}")
    except Exception:
        for row, destination in reversed(isolated):
            try:
                rollback_verified(Path(row["source"]), destination, row["identity"])
                _event("ROLLED_BACK", row)
            except RuntimeError:
                _event("ROLLBACK_BLOCKED_NEW_CONTENT_OR_IDENTITY", row)
        raise
    for row, destination in isolated:
        shutil.rmtree(destination)
        _event("PURGED", row, logical_deleted_bytes=row["identity"]["logical_bytes"])
    notice = {"schema_version": "PRODUCT_FIRST_HISTORICAL_STEREO_TRIM_NOTICE_V1",
              "task_id": TASK, "scope": [row["session_id"] for row in selection["targets"]],
              "historical_receipts_modified": False,
              "now_unavailable": "full per-frame historical Stereo NPZ arrays for 056/068",
              "still_available": "old RESULT/DEPTH_SUMMARY, all original review videos, three NPZ samples per session, raw inputs, pinned producer code, environment and model weight",
              "new_authority": "METADATA_AND_SPARSE_FORENSIC_ONLY",
              "external_metric_authority": False,
              "claim_limit": "Old consumption_authorized flags are historical; after payload trim these two full sessions cannot be directly consumed as Depth inputs. The expired W1 release clock blocks direct replay with the old CLI."}
    notice_path = OUT / "HISTORICAL_PAYLOAD_TRIM_NOTICE.json"
    if notice_path.exists():
        raise FileExistsError(notice_path)
    notice_path.write_text(json.dumps(notice, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {"schema_version": "PRODUCT_FIRST_CLEANUP_BATCH2_RECEIPT_V1",
              "task_id": TASK, "execution": "ACTUALLY_PURGED", "selected": artifact_ref(SELECTION),
              "trim_notice": artifact_ref(notice_path), "consumer_checks": checks,
              "reproducer_inputs": reproducer,
              "purged": [{"session_id": row["session_id"], "source": row["source"],
                          "identity": row["identity"]} for row in selection["targets"]],
              "logical_deleted_bytes": selection["logical_bytes_candidate"],
              "allocated_deleted_bytes_estimate": selection["allocated_bytes_candidate"],
              "physical_reclaimed_bytes": "UNKNOWN_CPFS_SHARED_ALLOCATION",
              "isolated_remaining_bytes": 0, "historical_receipts_modified": False,
              "claim_limit": "Only two exact, unconsumed historical derived frame trees purged; current 15 slots, source data, 031 Depth, environment and weights remain. Historical full depth consumption revoked by appended notice."}
    RECEIPT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ACTUALLY_PURGED", "count": len(result["purged"]),
                      "logical_deleted_bytes": result["logical_deleted_bytes"],
                      "receipt": str(RECEIPT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
