"""Read-only three-anchor CPU preflight for one VST finger-sleeve text change.

This is a Task Packet draft, not a model runner or registration path.  It
consumes only sealed Mask13 files and deliberately leaves the yellow-cable
failure unresolved.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

from chaoyang.governance.common import REPO_ROOT, artifact_ref, validate_artifact_ref


TASK_ID = "exact78_vst_finger_sleeve_prompt14_cards001_0915_001_v1_20261001"
OPERATION = "run_exact78_vst_finger_sleeve_prompt14_candidate_v1"
ROUTE_KEY = "exact78_vst_finger_sleeve_prompt14_candidate_v1"
SOURCE_SESSION = "play_cards_0915_001"
FRAMES = (0, 75, 149)
ROLE = "finger_sleeve"
OLD_PROMPT = "a finger sleeve"
NEW_PROMPT = "a white finger-worn sensor sleeve"
MASK13_RESULT_SHA = "217a363afecb9793eec0b60770f1d88245656d00984c088f0b17a0f4b5c9f080"
MASK13_ANCHOR_SHA = "fe1cfe09e1199c11350e37bff5c2f226c1f0c69ec152aa5214f3579482006a96"
MASK13_PACKET_SHA = "bfd27f8b67b199aeb1f9de20738253e929dcd7c2c6ada181f36aa7aa9329c8f7"
CHECKPOINT_SHA = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
VENDOR_MANIFEST_SHA = "1b42f75001be6aa1da3e64e20dd3589b4cf8bdebe84fecd8114675976a74f8ac"
SEALED = (REPO_ROOT / "_run/current/exact78_equipment_prompt_mask13_four_dev_v1_20261001"
          "/attempts/attempt_0001")
PACKET = REPO_ROOT / "tasks/current/exact78_equipment_prompt_mask13_four_dev_v1_20261001/TASK_PACKET.json"
ATTEMPT = REPO_ROOT / "_run/current" / TASK_ID / "attempts/attempt_0001"


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("PROMPT14_JSON_OBJECT_REQUIRED")
    return value


def _exact_ref(path: Path, expected_sha: str) -> dict[str, Any]:
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise RuntimeError("PROMPT14_SYMLINK_OR_PATH_DRIFT:" + str(path))
    ref = artifact_ref(path)
    if ref["sha256"] != expected_sha:
        raise RuntimeError("PROMPT14_SHA_DRIFT:" + str(path))
    return ref


def _role_prompts(row: dict[str, Any], old: dict[str, Any]) -> dict[str, str]:
    """Prove exactly one of the six sealed Mask13 role texts changes."""
    other = old["other_role_prompts"]
    expected = {
        "human": old["new_human_prompt"], "object": other["poker_object"],
        "equipment_original": other["vst_equipment"],
        "equipment_added": old["equipment_prompt_added"],
        "finger_sleeve": other["vst_finger_sleeve"],
        "yellow_cable": other["vst_yellow_cable"],
    }
    if row.get("prompts") != expected or expected[ROLE] != OLD_PROMPT:
        raise RuntimeError("PROMPT14_SEALED_ROLE_DRIFT")
    changed = dict(expected, finger_sleeve=NEW_PROMPT)
    if {role for role in expected if changed[role] != expected[role]} != {ROLE}:
        raise RuntimeError("PROMPT14_SINGLE_ROLE_DRIFT")
    return changed


def frozen_inputs() -> dict[str, Any]:
    outer_path = SEALED / "RESULT.json"
    anchor_path = SEALED / "candidate/MASK_ANCHOR.json"
    outer_ref = _exact_ref(outer_path, MASK13_RESULT_SHA)
    anchor_ref = _exact_ref(anchor_path, MASK13_ANCHOR_SHA)
    packet_ref = _exact_ref(PACKET, MASK13_PACKET_SHA)
    outer, anchor, packet = _object(outer_path), _object(anchor_path), _object(PACKET)
    if (outer.get("status") != "PASSED" or outer.get("execution_complete") is not True
            or outer.get("mask_anchor") != anchor_ref
            or outer.get("mask_quality_gate") != "REJECTED"
            or outer.get("clean_called") is not False
            or anchor.get("schema_version") != "exact78-equipment-prompt-mask13-v1"
            or anchor.get("model_calls") != 54
            or packet.get("task_id") != "exact78_equipment_prompt_mask13_four_dev_v1_20261001"):
        raise RuntimeError("PROMPT14_MASK13_SEAL_DRIFT")
    config = packet["config"]
    if (config.get("output_prob_thresh") != 0.5
            or config.get("instance_score_gate") != 0.5
            or config.get("other_role_prompts", {}).get("vst_finger_sleeve") != OLD_PROMPT
            or config.get("other_role_prompts", {}).get("vst_yellow_cable") != "a yellow cable"
            or config.get("clean_calls_authorized") != 0):
        raise RuntimeError("PROMPT14_THRESHOLD_OR_BASE_CONFIG_DRIFT")
    checkpoint = packet["inputs"]["sam_checkpoint"]
    vendor = packet["inputs"]["sam_vendor_manifest"]
    if (checkpoint.get("sha256") != CHECKPOINT_SHA
            or vendor.get("sha256") != VENDOR_MANIFEST_SHA
            or validate_artifact_ref(checkpoint) or validate_artifact_ref(vendor)):
        raise RuntimeError("PROMPT14_MODEL_OR_VENDOR_DRIFT")
    source = [item for item in packet["inputs"]["sessions"]
              if item.get("session_id") == SOURCE_SESSION]
    if (len(source) != 1 or source[0].get("source_family") != "VST_0915"
            or [item["frame"] for item in source[0]["anchors"]] != list(FRAMES)):
        raise RuntimeError("PROMPT14_SOURCE_OR_FRAMES_DRIFT")
    rows = [row for row in anchor["rows"] if row.get("session_id") == SOURCE_SESSION]
    if len(rows) != 3 or [row["frame"] for row in rows] != list(FRAMES):
        raise RuntimeError("PROMPT14_ANCHOR_GRID_DRIFT")
    pinned_rows: list[dict[str, Any]] = []
    for row, source_anchor in zip(rows, source[0]["anchors"], strict=True):
        frame = row["frame"]
        _role_prompts(row, config)
        if (row.get("source_family") != "VST_0915"
                or row.get("raw_image") != source_anchor["raw_image"]
                or any(row.get("instance_counts", {}).get(name) != 0 for name in
                       ("finger_sleeve", "yellow_cable", "equipment_original", "equipment_added"))):
            raise RuntimeError("PROMPT14_BASELINE_COUNT_OR_RAW_DRIFT")
        refs = {"raw_image": row["raw_image"], "sam_input": row["sam_input"]}
        refs.update({role: row["masks"][role] for role in
                     ("human", "object", "equipment", "finger_sleeve", "yellow_cable")})
        for name, ref in refs.items():
            path = Path(ref["path"])
            if (not path.is_relative_to(REPO_ROOT) or path.is_symlink()
                    or path.resolve(strict=True) != path
                    or (name == "sam_input" and path != SEALED / "candidate/sam_inputs"
                        / SOURCE_SESSION / f"{frame:06d}/000000.jpg")
                    or (name in {"human", "object", "equipment", "finger_sleeve", "yellow_cable"}
                        and path != SEALED / "candidate/masks" / SOURCE_SESSION / name
                        / f"{frame:06d}.png")
                    or validate_artifact_ref(ref)):
                raise RuntimeError("PROMPT14_ANCHOR_REF_DRIFT:" + name + ":" + str(frame))
        pinned_rows.append({"frame": frame, "refs": refs,
                            "old_prompts": row["prompts"],
                            "candidate_prompts": _role_prompts(row, config)})
    return {"sealed_mask13_result": outer_ref, "sealed_mask13_anchor": anchor_ref,
            "sealed_mask13_packet": packet_ref, "sam_checkpoint": checkpoint,
            "sam_vendor_manifest": vendor, "session_id": SOURCE_SESSION,
            "anchors": pinned_rows}


def packet_draft(inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """Finite reviewable draft; no writer token, code closure or execution authority."""
    frozen = frozen_inputs() if inputs is None else inputs
    now = datetime.now().astimezone()
    return {
        "schema_version": "exact78-vst-finger-sleeve-prompt14-task-packet-draft-v1",
        "status": "DRAFT_NOT_REGISTERED", "task_id": TASK_ID,
        "phase": "EXACT78_VST_FINGER_SLEEVE_PROMPT14_DEV_CANDIDATE",
        "execution_mode": "EXACT78_CANDIDATE", "candidate_operation": OPERATION,
        "route_key": ROUTE_KEY, "source_session": SOURCE_SESSION,
        "anchor_frames": list(FRAMES), "inputs": frozen,
        "single_change": {"role": ROLE, "from": OLD_PROMPT, "to": NEW_PROMPT},
        "frozen": {"sam_checkpoint_sha256": CHECKPOINT_SHA,
                   "vendor_manifest_sha256": VENDOR_MANIFEST_SHA,
                   "output_prob_thresh": 0.5, "instance_score_gate": 0.5,
                   "jpg_human_object_equipment_from_sealed_mask13": True,
                   "yellow_cable_prompt_and_mask_unchanged": True},
        "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 1,
                    "model_calls_max": 3, "render_calls": 0, "clean_calls": 0,
                    "training_steps": 0, "wall_seconds_max": 900,
                    "session_attempts_max": 1},
        "deadline_rule": "publisher sets fresh deadline no later than 900 seconds after registration",
        "example_deadline_at": (now + timedelta(seconds=900)).isoformat(),
        "write_set": [str(ATTEMPT.relative_to(REPO_ROOT)),
                      f"tasks/current/{TASK_ID}/TASK_PACKET.json",
                      f"tasks/receipts/{TASK_ID.upper()}.json",
                      "tasks/current/INDEX.json", "docs/governance",
                      "_run/current/GPU_LEASE.json", "_run/current/GPU_LEASE.lock"],
        "required_outputs": ["candidate/FINGER_SLEEVE_ANCHORS.json",
                             "candidate/RESULT.json", "GPU_RECEIPT.json",
                             "RESULT.json", "terminal_receipt"],
        "stop_conditions": ["SOURCE_OR_CODE_SHA_DRIFT", "DOUBLE_WRITER",
                            "GPU_LEASE_UNAVAILABLE", "THREE_MODEL_CALL_LIMIT",
                            "UNEXPECTED_PROMPT_OR_THRESHOLD", "DEADLINE",
                            "ONE_ATTEMPT_TERMINAL"],
        "quality_hold": "YELLOW_CABLE_ZERO_DETECTION_REMAINS; CLEAN_NOT_ADMITTED",
        "claim_limit": "Three development finger-sleeve masks only; no Clean, visual, Robot, geometry, product or training pass",
        "publication_blockers": ["GPU_OPERATION_AND_FINITE_EXECUTOR_NOT_IMPLEMENTED",
                                 "CLI_AND_ALGORITHM_ROUTE_NOT_STAGED",
                                 "FRESH_CAS_TASK_PACKET_AND_WRITER_REQUIRED"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    assert args.check_only
    draft = packet_draft()
    print(json.dumps({"status": draft["status"], "task_id": TASK_ID,
                      "source_session": SOURCE_SESSION, "anchor_frames": list(FRAMES),
                      "single_change": draft["single_change"],
                      "budgets": draft["budgets"],
                      "yellow_cable": "UNRESOLVED_QUALITY_HOLD",
                      "publication_blockers": draft["publication_blockers"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
