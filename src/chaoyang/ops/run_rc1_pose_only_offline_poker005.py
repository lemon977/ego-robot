#!/usr/bin/env python3
"""One bounded, CPU-only Poker005 Robot review; never a causal/authority artifact."""
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from chaoyang.ops import run_rc1_pose_only_offline_single as common


ROOT = Path(__file__).resolve().parents[3]
SESSION = "play_cards_0901_005"
TASK_ID = "rc1_poker005_pose_only_offline_canary"
FRAMES = 520
common.SESSION = SESSION  # Heartbeat in the shared runner records the correct identity.


def checked_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_full_video(path: Path) -> dict:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames,avg_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    if int(stream["nb_read_frames"]) != FRAMES:
        raise RuntimeError(f"frame count {stream['nb_read_frames']} != {FRAMES}")
    decoded = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
         "-f", "null", "-"], capture_output=True, text=True, timeout=300,
    )
    if decoded.returncode:
        raise RuntimeError(f"full decode failed: {decoded.stderr[-400:]}")
    return common.ref(path) | {
        "frames": FRAMES, "fps": stream["avg_frame_rate"], "decode": "PASS_FFMPEG_XERROR"
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet_path = args.packet.resolve(strict=True)
    packet = checked_json(packet_path)
    if (packet.get("task_id"), packet.get("session_id"), packet.get("mode")) != (
        TASK_ID, SESSION, "OFFLINE_VISUAL"
    ):
        raise RuntimeError("packet identity/mode mismatch")
    attempt = args.output.resolve()
    if not attempt.is_relative_to(Path(packet["write_root"]).resolve()) or attempt.exists():
        raise RuntimeError("output must be a fresh attempt below the declared write root")
    for label, pin in packet["inputs"].items():
        common.verify_pin(pin, label)
    for label, pin in packet["code"].items():
        common.verify_pin(pin, f"code:{label}")
    selection = checked_json(Path(packet["inputs"]["selection"]["path"]))
    selected = [row for row in selection["rows"] if row.get("session_id") == SESSION]
    matrix = checked_json(Path(packet["inputs"]["matrix"]["path"]))
    upstream = [row for row in matrix["rows"] if row.get("session_id") == SESSION]
    if len(selected) != 1 or len(upstream) != 1 or selected[0].get("task") != "poker":
        raise RuntimeError("session not unique in frozen selection/matrix")
    if upstream[0].get("frame_count") != FRAMES or not upstream[0].get("three_upstream_ab"):
        raise RuntimeError("frame count or triple A/B preflight failed")
    if packet["config"]["collision_frames"] != FRAMES:
        raise RuntimeError("packet frame configuration mismatch")
    signature_payload = {
        "schema": packet["schema_version"],
        "inputs": {key: pin["sha256"] for key, pin in packet["inputs"].items()},
        "code": {key: pin["sha256"] for key, pin in packet["code"].items()},
        "config": packet["config"],
        "model_weights": "PINNED_URDF_ASSET_PIN",
        "calibration": "ABSENT_POSE_ONLY",
    }
    run_signature = hashlib.sha256(json.dumps(
        signature_payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    attempt.mkdir(parents=True)
    common.atomic_json(attempt / "RUN_START.json", {
        "task_id": TASK_ID, "session_id": SESSION, "created_at": common.now(),
        "packet": common.ref(packet_path), "run_signature": run_signature,
        "input_mode": "OFFLINE_VISUAL", "gpu_required": False,
        "central_gpu_lease_at_start": checked_json(ROOT / "_run/current/GPU_LEASE.json"),
        "control_ground_truth": False, "physical_deployment_authorized": False,
    })
    heartbeat = common.Heartbeat(attempt / "HEARTBEAT.json", packet["fencing_token"])
    heartbeat.start()
    deadline = time.monotonic() + int(packet["budget"]["wall_cap_s"])
    receipts = []
    terminal, error, video = "FAILED_RUNTIME_FINAL", None, None
    eligible_ids: list[int] = []
    unknown_ids: list[int] = []
    try:
        inp = {key: pin["path"] for key, pin in packet["inputs"].items()}
        config = packet["config"]
        arm_dir, hand_dir, audit_dir, review_dir = (
            attempt / name for name in ("arm", "hand", "collision", "review")
        )
        arm_result, hand_result = arm_dir / "RESULT.json", hand_dir / "RESULT.json"
        arm_states, hand_states = arm_dir / "ARM_CANARY_STATES.npz", hand_dir / "HAND_STATES.npz"
        receipts.append(common.phase("arm", [sys.executable, "-m", "chaoyang.ops.run_robot_motion_transfer_arm_canary_v3",
            "--task", "poker", "--session", SESSION, "--hawor", inp["hawor_npz"],
            "--hawor-result", inp["hawor_result"], "--accepted-states", inp["template_states"],
            "--accepted-hawor", inp["template_hawor"], "--task-base-backoff-m",
            str(config["task_base_backoff_m"]), "--output", str(arm_result)],
            attempt, heartbeat, phase_cap_s=2400, deadline=deadline, allow_quality_hold=True))
        receipts.append(common.phase("hand", [sys.executable, "-m", "chaoyang.ops.run_robot_hand_fullsession_v2",
            "--task", "poker", "--session", SESSION, "--hawor", inp["hawor_npz"],
            "--hawor-result", inp["hawor_result"], "--accepted-states", inp["template_states"],
            "--output", str(hand_result)],
            attempt, heartbeat, phase_cap_s=2100, deadline=deadline, allow_quality_hold=True))
        with np.load(arm_states) as arm, np.load(hand_states) as hand:
            av, hv = arm["valid_side_frame"], hand["valid_side_frame"]
            if av.shape != (2, FRAMES) or hv.shape != (2, FRAMES):
                raise RuntimeError("arm/hand validity shapes are not [2,520]")
            both = np.all(av.astype(bool) & hv.astype(bool), axis=0)
            eligible_ids = np.flatnonzero(both).astype(int).tolist()
            unknown_ids = np.flatnonzero(~both).astype(int).tolist()
        common.atomic_json(attempt / "FRAME_VALIDITY.json", {
            "session_id": SESSION, "frame_count": FRAMES,
            "bilateral_valid_count": len(eligible_ids), "bilateral_valid_frame_ids": eligible_ids,
            "unknown_frame_ids": unknown_ids,
            "claim_limit": "Unknown frames do not receive digital-collision PASS by interpolation.",
        })
        if not eligible_ids:
            raise RuntimeError("no bilateral-valid frames available for digital collision audit")
        receipts.append(common.phase("collision", [sys.executable, "-m", "chaoyang.ops.audit_robot_geometry_self_collision_v71",
            "--session-id", SESSION, "--arm-states", str(arm_states),
            "--hand-states", str(hand_states), "--frame-count", str(len(eligible_ids)),
            "--output-dir", str(audit_dir)], attempt, heartbeat, phase_cap_s=600,
            deadline=deadline, allow_quality_hold=True))
        receipts.append(common.phase("render", [sys.executable, "-m", "chaoyang.ops.render_robot_motion_transfer_fullsession_v2",
            "--task", "poker", "--session", SESSION, "--hawor", inp["hawor_npz"],
            "--hawor-result", inp["hawor_result"], "--arm-states", str(arm_states),
            "--arm-result", str(arm_result), "--hand-states", str(hand_states),
            "--hand-result", str(hand_result), "--fixed-placement-label", config["placement_label"],
            "--allow-hand-hold-review", "--allow-arm-hold-review", "--output-dir", str(review_dir)],
            attempt, heartbeat, phase_cap_s=1200, deadline=deadline))
        video = decode_full_video(review_dir / f"{SESSION}_ROBOT_WORLD_FIRST_GAIN1_FULLSESSION.mp4")
        arm = checked_json(arm_result)
        hand = checked_json(hand_result)
        collision = checked_json(audit_dir / "RESULT.json")
        hard_pass = (not unknown_ids and arm["status"].startswith("PASS")
                     and hand["status"].startswith("PASS")
                     and collision["status"].startswith("PASS"))
        terminal = "PASSED_DIAGNOSTIC" if hard_pass else "FAILED_QUALITY_C_DIAGNOSTIC_VIDEO"
    except Exception as exc:  # Always seal this one algorithm attempt.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        heartbeat.phase = "TERMINAL"
        heartbeat.stop()
        result = {
            "schema_version": "rc1-poker-pose-only-offline-canary-v1", "task_id": TASK_ID,
            "session_id": SESSION, "task": "poker", "created_at": common.now(),
            "terminal_status": terminal, "error": error, "run_signature": run_signature,
            "packet": common.ref(packet_path), "phase_receipts": receipts, "full_video": video,
            "bilateral_valid_count": len(eligible_ids), "unknown_frame_ids": unknown_ids,
            "arm_result": common.ref(attempt / "arm/RESULT.json") if (attempt / "arm/RESULT.json").is_file() else None,
            "hand_result": common.ref(attempt / "hand/RESULT.json") if (attempt / "hand/RESULT.json").is_file() else None,
            "collision_result": common.ref(attempt / "collision/RESULT.json") if (attempt / "collision/RESULT.json").is_file() else None,
            "frame_validity": common.ref(attempt / "FRAME_VALIDITY.json") if (attempt / "FRAME_VALIDITY.json").is_file() else None,
            "input_mode": "OFFLINE_VISUAL", "training_eligible": False,
            "control_ground_truth": False, "physical_deployment_authorized": False,
            "authority_promoted": False,
            "claim_limit": "Full-session pose-only development video only. Unknown frames cannot be called collision-safe; not causal, contact, control or physical authority.",
        }
        common.atomic_json(attempt / "RESULT.json", result)
        common.atomic_json(attempt / "RESULT_SUMMARY.json", {
            "task_id": TASK_ID, "status": terminal, "video": video, "error": error,
            "result": common.ref(attempt / "RESULT.json"),
        })
    print(json.dumps({"status": terminal, "video": video, "error": error,
                      "result": str(attempt / "RESULT.json")}, ensure_ascii=False))
    return 0 if terminal == "PASSED_DIAGNOSTIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
