#!/usr/bin/env python3
"""
Adopt one already-rendered, numerically gated Robot review into the exact78 V5.2
POSE_ONLY_VISUAL_ROBOT contract without upgrading any physical/contact claim.

This tool is deliberately narrow.  It verifies every pinned predecessor byte and
SHA, verifies that the frozen Wave0 Clean result is Grade B and downstream-ready,
then materializes one immutable visual trajectory sidecar plus copied review
media.  It never emits an action sidecar and it never fabricates contact,
object geometry, adapter CAD, TCP, mount, or world-to-base calibration truth.

Example:
  python3 -m tools.adopt_exact78_pose_only_visual_robot_v52 \
    --session play_cards_0903_203 \
    --legacy-result tasks/control/runs/20260911_six_session_robot_motion_transfer_successor_v2/full_review/play_cards_0903_203/RESULT.json \
    --output-root tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/pose_only_visual_robot_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path("/mnt/workspace/code/chaoyang")
DEFAULT_SELECTION = REPO / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json"
SCHEMA = "exact78-pose-only-visual-robot-v52-v1"
SIDECAR_SCHEMA = "exact78-visual-robot-trajectory-sidecar-v52-v1"
UNKNOWN_CONTACT_CODE = np.uint8(0)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ref(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    p = path.resolve()
    display = str(p.relative_to(relative_to.resolve())) if relative_to is not None else str(p)
    return {"path": display, "bytes": p.stat().st_size, "sha256": sha256(p)}


def resolve_repo(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def require_ref(entry: dict[str, Any], label: str) -> Path:
    for key in ("path", "bytes", "sha256"):
        if key not in entry:
            raise RuntimeError(f"{label}: missing {key}")
    p = resolve_repo(str(entry["path"]))
    if not p.is_file() or p.is_symlink():
        raise RuntimeError(f"{label}: regular non-symlink file required: {p}")
    if p.stat().st_size != int(entry["bytes"]):
        raise RuntimeError(f"{label}: byte mismatch: {p}")
    actual = sha256(p)
    if actual != entry["sha256"]:
        raise RuntimeError(f"{label}: SHA mismatch: {p}")
    return p


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def ffprobe_video(path: Path) -> dict[str, int | str]:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(path),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    stream = data["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": str(stream["avg_frame_rate"]),
        "decoded_frames": int(stream["nb_read_frames"]),
    }


def selection_entry(selection: dict[str, Any], session: str) -> dict[str, Any]:
    rows = [row for row in selection.get("sessions", []) if row.get("session_id") == session]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one frozen Wave0 row for {session}; got {len(rows)}")
    return rows[0]


def validate_clean(row: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    pin = row.get("existing_clean")
    if not isinstance(pin, dict):
        # New successor results use the planned output location.
        p = Path(row["planned_outputs"]["propainter_root"]) / "RESULT.json"
        if not p.is_file():
            raise RuntimeError("frozen row has neither existing_clean nor a published planned Clean result")
        pin = ref(p)
    clean_path = require_ref(pin, "frozen Wave0 Clean result")
    clean = load_json(clean_path)
    if clean.get("grade") != "B" or not clean.get("downstream_authorized", False):
        raise RuntimeError(f"Clean is not Grade-B downstream-ready: {clean_path}")
    if clean.get("session") != row["session_id"]:
        raise RuntimeError("Clean session mismatch")
    if int(clean.get("frame_count", -1)) != int(row["frame_count"]):
        raise RuntimeError("Clean frame-count mismatch")
    return clean_path, clean


def validate_hawor_chain(row: dict[str, Any], legacy: dict[str, Any]) -> dict[str, Any]:
    frozen_result_path = require_ref(row["upstream"]["hawor"], "frozen HaWoR result")
    frozen = load_json(frozen_result_path)
    frozen_npz = require_ref(frozen["outputs"]["npz"], "frozen HaWoR NPZ")

    temporal_result_path = require_ref(legacy["lineage"]["hawor_result"], "temporal HaWoR result")
    temporal = load_json(temporal_result_path)
    temporal_inputs = temporal.get("lineage") or temporal.get("inputs")
    if not isinstance(temporal_inputs, dict) or "bounded_npz" not in temporal_inputs:
        raise RuntimeError("temporal HaWoR result does not expose a pinned bounded_npz predecessor")
    temporal_input = require_ref(temporal_inputs["bounded_npz"], "temporal HaWoR bounded input")
    if temporal_input.resolve() != frozen_npz.resolve() or sha256(temporal_input) != sha256(frozen_npz):
        raise RuntimeError("temporal HaWoR is not derived from the frozen Wave0 HaWoR NPZ")
    temporal_output = require_ref(legacy["lineage"]["hawor_npz"], "temporal HaWoR output")
    if temporal.get("status") != "PASS_NUMERIC_NEEDS_HUMAN_REVIEW":
        raise RuntimeError("temporal HaWoR predecessor did not pass its numeric gate")
    return {
        "frozen_result": ref(frozen_result_path),
        "frozen_npz": ref(frozen_npz),
        "temporal_result": ref(temporal_result_path),
        "temporal_output": ref(temporal_output),
    }


def numeric_gate(legacy: dict[str, Any]) -> None:
    if legacy.get("status") != "PASS_NUMERIC_RENDER_READY_FOR_HUMAN_REVIEW":
        raise RuntimeError("legacy Robot review did not pass its numeric/render gate")
    numeric = legacy.get("numeric", {})
    for part in ("arm", "hand"):
        if int(numeric.get(part, {}).get("failed_rows", -1)) != 0:
            raise RuntimeError(f"legacy {part} contains failed rows")
    arm = numeric["arm"]
    hand = numeric["hand"]
    limits = [
        (arm["position_mm_max"], 10.0, "arm position"),
        (arm["rotation_deg_max"], 5.0, "arm rotation"),
        (arm["velocity_rad_per_frame_max_contiguous"], 0.120000001, "arm velocity"),
        (arm["acceleration_rad_per_frame2_max_contiguous"], 0.060000001, "arm acceleration"),
        (hand["velocity_rad_per_frame_max_contiguous"], 0.080000001, "hand velocity"),
        (hand["acceleration_rad_per_frame2_max_contiguous"], 0.060000001, "hand acceleration"),
    ]
    for value, maximum, label in limits:
        if float(value) > maximum:
            raise RuntimeError(f"{label} gate failed: {value} > {maximum}")


def write_sidecar(path: Path, arm_path: Path, hand_path: Path, frame_count: int) -> dict[str, Any]:
    arm = np.load(arm_path, allow_pickle=False)
    hand = np.load(hand_path, allow_pickle=False)
    required_arm = {
        "q_arm", "T_world_base", "T_tool_hand_root", "T_target_hand_root_world",
        "T_actual_hand_root_world", "valid_side_frame", "source_frames",
    }
    required_hand = {"q_hand", "q_hand_raw", "valid_side_frame", "source_frames"}
    if not required_arm.issubset(arm.files) or not required_hand.issubset(hand.files):
        raise RuntimeError("Robot state predecessor is missing required arrays")
    arm_frames = np.asarray(arm["source_frames"], dtype=np.int64)
    hand_frames = np.asarray(hand["source_frames"], dtype=np.int64)
    expected = np.arange(frame_count, dtype=np.int64)
    if not np.array_equal(arm_frames, expected) or not np.array_equal(hand_frames, expected):
        raise RuntimeError("Robot predecessor does not cover exact contiguous source frames")
    av = np.asarray(arm["valid_side_frame"], dtype=bool).T
    hv = np.asarray(hand["valid_side_frame"], dtype=bool).T
    if av.shape != (frame_count, 2) or hv.shape != (frame_count, 2):
        raise RuntimeError("unexpected validity shape")
    valid = av & hv
    # NaN is required on UNKNOWN rows and forbidden on valid rows.
    for name, array in (("q_arm", arm["q_arm"]), ("q_hand", hand["q_hand"])):
        finite = np.isfinite(array).all(axis=-1)
        if np.any(valid & ~finite):
            raise RuntimeError(f"{name} has non-finite values on valid rows")
        if np.any(~valid & finite):
            raise RuntimeError(f"{name} silently fills UNKNOWN rows")

    kwargs: dict[str, Any] = {
        "schema_version_utf8": np.frombuffer(SIDECAR_SCHEMA.encode(), dtype=np.uint8),
        "source_frame_ids": expected,
        "side_order_utf8": np.frombuffer(b"left,right", dtype=np.uint8),
        "q_arm_rad": np.asarray(arm["q_arm"], dtype=np.float64),
        "q_hand_rad": np.asarray(hand["q_hand"], dtype=np.float64),
        "q_hand_raw_rad": np.asarray(hand["q_hand_raw"], dtype=np.float64),
        "T_world_base": np.asarray(arm["T_world_base"], dtype=np.float64),
        "T_tool_hand_root": np.asarray(arm["T_tool_hand_root"], dtype=np.float64),
        "T_target_hand_root_world": np.asarray(arm["T_target_hand_root_world"], dtype=np.float64),
        "T_actual_hand_root_world": np.asarray(arm["T_actual_hand_root_world"], dtype=np.float64),
        "valid_side_frame": valid,
        "contact_state": np.full((frame_count, 2), UNKNOWN_CONTACT_CODE, dtype=np.uint8),
        "contact_frame_valid": np.zeros((frame_count, 2), dtype=bool),
        "metric_object_geometry": np.asarray(False, dtype=bool),
        "control_ground_truth": np.asarray(False, dtype=bool),
    }
    if "accepted_offset_camera" in arm.files:
        kwargs["accepted_offset_camera"] = np.asarray(arm["accepted_offset_camera"], dtype=np.float64)
    np.savez_compressed(path, **kwargs)
    return {
        "valid_side_rows": int(valid.sum()),
        "unknown_side_rows": int((~valid).sum()),
        "all_contact_unknown": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--legacy-result", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    selection_path = args.selection.resolve()
    legacy_path = args.legacy_result.resolve()
    selection = load_json(selection_path)
    if sha256(selection_path) != "10ee9e3668aa4928f0087c961dbb5d909202af576f859adf7dbb02604f5b3bd1":
        raise RuntimeError("Wave0 selection SHA is not the frozen V5.2 authority")
    row = selection_entry(selection, args.session)
    legacy = load_json(legacy_path)
    if legacy.get("session") != args.session or legacy.get("task") != row["task"]:
        raise RuntimeError("legacy result identity mismatch")
    if int(legacy.get("frame_count", -1)) != int(row["frame_count"]):
        raise RuntimeError("legacy frame count differs from frozen cohort")
    numeric_gate(legacy)
    clean_path, _ = validate_clean(row)
    hawor_chain = validate_hawor_chain(row, legacy)

    arm_path = require_ref(legacy["lineage"]["arm_states"], "arm states")
    hand_path = require_ref(legacy["lineage"]["hand_states"], "hand states")
    require_ref(legacy["lineage"]["arm_result"], "arm result")
    require_ref(legacy["lineage"]["hand_result"], "hand result")
    source_video = require_ref(legacy["lineage"]["source_video"], "source video")
    review_video = require_ref(legacy["outputs"]["video"], "legacy review video")
    frame_manifest = require_ref(legacy["outputs"]["frame_manifest"], "legacy frame manifest")
    probe = ffprobe_video(review_video)
    frame_count = int(row["frame_count"])
    if probe["decoded_frames"] != frame_count:
        raise RuntimeError("review video decoded-frame mismatch")
    manifest_rows = frame_manifest.read_text().splitlines()
    if len(manifest_rows) != frame_count:
        raise RuntimeError("frame manifest row-count mismatch")

    output_root = args.output_root.resolve()
    final = output_root / args.session
    if final.exists():
        raise RuntimeError(f"immutable output already exists: {final}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.session}.staging.", dir=output_root))
    try:
        sidecar = staging / "VISUAL_ROBOT_TRAJECTORY_SIDECAR.npz"
        coverage = write_sidecar(sidecar, arm_path, hand_path, frame_count)
        copied_video = staging / f"{args.session}_POSE_ONLY_VISUAL_ROBOT_FULLSESSION.mp4"
        copied_manifest = staging / "FRAME_MANIFEST.jsonl"
        shutil.copy2(review_video, copied_video)
        shutil.copy2(frame_manifest, copied_manifest)

        script_path = Path(__file__).resolve()
        input_manifest = {
            "selection": ref(selection_path),
            "frozen_clean_result": ref(clean_path),
            "hawor_chain": hawor_chain,
            "legacy_robot_result": ref(legacy_path),
            "legacy_arm_states": ref(arm_path),
            "legacy_hand_states": ref(hand_path),
            "source_video": ref(source_video),
        }
        code_closure = {"adopter": ref(script_path)}
        run_signature_payload = {
            "input_manifest_sha": hashlib.sha256(canonical_json(input_manifest)).hexdigest(),
            "code_closure_sha": hashlib.sha256(canonical_json(code_closure)).hexdigest(),
            "config_sha": "ABSENT_ADOPTION_ONLY",
            "model_weights_sha": "ABSENT_NO_MODEL_EXECUTION",
            "calibration_sha_or_absent": "ABSENT_PHYSICAL_CALIBRATION_BLOCKED_EXTERNAL",
            "schema_version": SCHEMA,
        }
        run_signature = hashlib.sha256(canonical_json(run_signature_payload)).hexdigest()
        lineage = {
            "schema_version": "exact78-pose-only-visual-robot-lineage-v1",
            "session": args.session,
            "task": row["task"],
            "input_manifest": input_manifest,
            "code_closure": code_closure,
            "run_signature_payload": run_signature_payload,
            "run_signature": run_signature,
        }
        (staging / "LINEAGE.json").write_bytes(canonical_json(lineage))

        result = {
            "schema_version": SCHEMA,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "POSE_ONLY_VISUAL_ROBOT_REVIEW_READY",
            "terminal_mode": "POSE_ONLY_VISUAL_ROBOT",
            "authority": False,
            "session": args.session,
            "task": row["task"],
            "frame_count": frame_count,
            "source_frames": {"first": 0, "last": frame_count - 1, "exact_contiguous": True},
            "clean_join_ready": True,
            "world_first_camera_motion_removed": True,
            "fixed_task_level_placement": True,
            "base_motion_during_session": False,
            "contact_state": "UNKNOWN",
            "metric_object_geometry": False,
            "contact_frame_valid": False,
            "control_ground_truth": False,
            "action_sidecar_published": False,
            "physical_deployment": {
                "status": "BLOCKED_EXTERNAL",
                "missing": [
                    "REAL_KAIHAND_ADAPTER_CAD", "REAL_TCP_CALIBRATION",
                    "REAL_MOUNT_CALIBRATION", "REAL_WORLD_TO_ROBOT_BASE_CALIBRATION",
                ],
            },
            "numeric": legacy["numeric"],
            "coverage": coverage,
            "review_video_probe": probe,
            "outputs": {
                "trajectory_sidecar": ref(sidecar, relative_to=staging),
                "review_video": ref(copied_video, relative_to=staging),
                "frame_manifest": ref(copied_manifest, relative_to=staging),
                "lineage": ref(staging / "LINEAGE.json", relative_to=staging),
            },
            "run_signature": run_signature,
            "input_signature": run_signature_payload["input_manifest_sha"],
            "prepare_signature": "NOT_APPLICABLE_IMMUTABLE_ADOPTION",
            "producer_signature": run_signature,
            "publish_signature": run_signature,
            "claim_limit": (
                "Numeric pose-only visual Robot trajectory and full-session review media only; "
                "not contact truth, not metric object geometry, not Robot action/control ground truth, "
                "and not physically deployable without the named external calibration/CAD evidence."
            ),
        }
        (staging / "RESULT.json").write_bytes(canonical_json(result))
        checks = []
        for p in sorted(staging.iterdir()):
            if p.is_file():
                checks.append(f"{sha256(p)}  {p.name}")
        (staging / "SHA256SUMS").write_text("\n".join(checks) + "\n")
        os.replace(staging, final)
        print(json.dumps({"status": "PASS", "output": str(final), "run_signature": run_signature}, indent=2))
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
