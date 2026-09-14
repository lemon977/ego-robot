#!/usr/bin/env python3
"""Publish one exact78 Robot FAILED_QUALITY_C terminal from bounded numeric reviews.

This publisher is deliberately one-way and no-clobber.  It never converts a
HOLD review into a pose/action authority.  It only seals an explicit V5.2
terminal after the frozen Clean result is usable, one or both Robot numeric
gates remain failed, and the complete review media is exact and decodable.
The caller must state whether one forward round or both forward and
bidirectional rounds were consumed; the receipt never invents an extra round.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


REPO = Path("/mnt/workspace/code/chaoyang")
COHORT = REPO / "tasks/control/runs/20260908_two_task_e2e_baseline_v1/EXACT78_BATCH_MANIFEST.json"
WAVE0 = REPO / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json"
FROZEN_COHORT_SHA = "b6dcfec1fb41b9ffb0b9b19cc62d029f5cedf57f41552a611272cc9e5c6e43a8"
FROZEN_WAVE0_SHA = "10ee9e3668aa4928f0087c961dbb5d909202af576f859adf7dbb02604f5b3bd1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"ordinary file required: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def exact_reference(reference: dict[str, Any], *, base: Path = REPO) -> Path:
    path = Path(reference["path"])
    if not path.is_absolute():
        path = base / path
    if ref(path) != {"path": str(path.resolve()), "bytes": int(reference["bytes"]), "sha256": reference["sha256"]}:
        raise RuntimeError(f"reference mismatch: {path}")
    return path.resolve()


def video_frames(path: Path) -> int:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,nb_frames", "-of", "json", str(path),
    ]
    stream = json.loads(subprocess.check_output(command, text=True))["streams"][0]
    return int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)


def atomic_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"no-clobber terminal exists: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def identity(data: dict[str, Any], task: str, session: str, frame_count: int, label: str) -> None:
    actual_task = data.get("task")
    actual_session = data.get("session", data.get("session_id"))
    if (actual_task, actual_session, int(data.get("frame_count", -1))) != (task, session, frame_count):
        raise RuntimeError(f"{label} identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chips", "poker"), required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--clean-result", type=Path, required=True)
    parser.add_argument("--hawor-result", type=Path, required=True)
    parser.add_argument("--arm-result", type=Path, required=True)
    parser.add_argument("--arm-states", type=Path, required=True)
    parser.add_argument("--hand-result", type=Path, required=True)
    parser.add_argument("--hand-states", type=Path, required=True)
    parser.add_argument("--review-result", type=Path, required=True)
    parser.add_argument("--method-rounds-consumed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if sha256(COHORT) != FROZEN_COHORT_SHA or sha256(WAVE0) != FROZEN_WAVE0_SHA:
        raise RuntimeError("frozen cohort/Wave0 SHA mismatch")
    cohort = load(COHORT)
    rows = [row for row in cohort["sessions"] if row["session_id"] == args.session and row["task"] == args.task]
    if len(rows) != 1:
        raise RuntimeError("session is not one unique frozen cohort row")
    row = rows[0]
    frame_count = int(row["frame_count"])
    if args.session not in {item["session_id"] for item in load(WAVE0)["sessions"]}:
        raise RuntimeError("quality terminal publisher requires a Wave0 session")

    clean = load(args.clean_result)
    identity(clean, args.task, args.session, frame_count, "Clean")
    if clean.get("grade") != "B" or clean.get("downstream_authorized") is not True:
        raise RuntimeError("Clean is not Grade B downstream-authorized")
    if any(value != "PASS" for value in clean.get("hard_gates", {}).values()):
        raise RuntimeError("Clean hard gate failed")

    hawor = load(args.hawor_result)
    identity(hawor, args.task, args.session, frame_count, "HaWoR temporal")
    if hawor.get("status") != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
        raise RuntimeError("HaWoR temporal result is not numeric PASS")

    arm = load(args.arm_result)
    hand = load(args.hand_result)
    review = load(args.review_result)
    identity(arm, args.task, args.session, frame_count, "arm")
    identity(hand, args.task, args.session, frame_count, "hand")
    identity(review, args.task, args.session, frame_count, "review")
    allowed_numeric_statuses = {"PASS_NUMERIC_CANARY_NO_AUTHORITY", "HOLD_NUMERIC_CANARY"}
    if arm.get("status") not in allowed_numeric_statuses or hand.get("status") not in allowed_numeric_statuses:
        raise RuntimeError("unexpected arm/hand numeric status")
    arm_failed = arm.get("status") == "HOLD_NUMERIC_CANARY"
    hand_failed = hand.get("status") == "HOLD_NUMERIC_CANARY"
    if not (arm_failed or hand_failed):
        raise RuntimeError("FAILED_QUALITY_C requires at least one exhausted numeric HOLD")
    if review.get("authority") is not False or review.get("action_sidecar_published") is not False:
        raise RuntimeError("review claim boundary violation")
    if bool(review.get("numeric", {}).get("arm_pass")) == arm_failed:
        raise RuntimeError("review arm pass flag contradicts numeric result")
    if bool(review.get("numeric", {}).get("hand_pass")) == hand_failed:
        raise RuntimeError("review hand pass flag contradicts numeric result")

    # The full-session review is the binding receipt for the exact state/result
    # files rendered into its pixels.  Reject callers that substitute another
    # numeric attempt while reusing an unrelated review video.
    review_lineage = review.get("lineage")
    if not isinstance(review_lineage, dict):
        raise RuntimeError("review does not expose pinned lineage")
    lineage_args = {
        "hawor_result": args.hawor_result,
        "arm_result": args.arm_result,
        "arm_states": args.arm_states,
        "hand_result": args.hand_result,
        "hand_states": args.hand_states,
    }
    for key, supplied in lineage_args.items():
        if key not in review_lineage:
            raise RuntimeError(f"review lineage missing {key}")
        pinned = exact_reference(review_lineage[key])
        supplied_ref = ref(supplied)
        if pinned != supplied.resolve() or sha256(pinned) != supplied_ref["sha256"]:
            raise RuntimeError(f"review lineage does not match supplied {key}")
    video = exact_reference(review["outputs"]["video"])
    manifest = exact_reference(review["outputs"]["frame_manifest"])
    if int(review["outputs"]["video"]["decoded_frames"]) != frame_count or video_frames(video) != frame_count:
        raise RuntimeError("full review video frame count mismatch")
    if int(review["outputs"]["frame_manifest"]["rows"]) != frame_count:
        raise RuntimeError("review manifest frame count mismatch")

    evidence = {
        "clean_result": ref(args.clean_result),
        "hawor_result": ref(args.hawor_result),
        "arm_result": ref(args.arm_result),
        "arm_states": ref(args.arm_states),
        "hand_result": ref(args.hand_result),
        "hand_states": ref(args.hand_states),
        "review_result": ref(args.review_result),
        "review_video": ref(video),
        "review_manifest": ref(manifest),
    }
    round_suffix = "FORWARD_ONLY" if args.method_rounds_consumed == 1 else "FORWARD_AND_BIDIRECTIONAL"
    reason_codes = []
    if arm_failed:
        reason_codes.append(f"ARM_NUMERIC_GATE_FAILED_AFTER_{round_suffix}")
    if hand_failed:
        reason_codes.append(f"HAND_NUMERIC_GATE_FAILED_AFTER_{round_suffix}")
    result = {
        "schema_version": "exact78-v52-robot-terminal-v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "FAILED_QUALITY_C",
        "terminal": True,
        "terminal_mode": "FAILED_QUALITY_C",
        "task": args.task,
        "session": args.session,
        "frame_count": frame_count,
        "clean_join_ready": True,
        "robot_visual_usable": False,
        "downstream_authorized": False,
        "authority": False,
        "contact_state": "UNKNOWN",
        "metric_object_geometry": False,
        "contact_frame_valid": False,
        "control_ground_truth": False,
        "action_sidecar_published": False,
        "method_rounds_consumed": args.method_rounds_consumed,
        "review_lineage_verified": True,
        "reason_codes": reason_codes,
        "numeric": {"arm": arm.get("metrics"), "hand": hand.get("metrics")},
        "evidence": evidence,
        "frozen_sources": {"cohort": ref(COHORT), "wave0": ref(WAVE0)},
        "claim_limit": "Explicit Robot visual quality-C terminal only; no Robot pose/action/contact/training or physical deployment authority.",
    }
    output = args.output_root.resolve() / args.task / args.session / "RESULT.json"
    atomic_new(output, result)
    print(json.dumps({"status": result["status"], "terminal_mode": result["terminal_mode"], "result": ref(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
