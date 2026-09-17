#!/usr/bin/env python3
"""Resume one failed pose-only canary from fully sealed arm/hand phase outputs.

The predecessor failed before collision computation because the audit CLI was
asked for 645 bilateral-valid frames although two frames have UNKNOWN right
hand.  This successor verifies exact phase closure and audits all 643 valid
frames without filling the two UNKNOWN frames or changing solver states.
"""
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
from datetime import datetime
from pathlib import Path

import numpy as np

from chaoyang.ops.run_rc1_pose_only_offline_single import Heartbeat, atomic_json, phase, ref, verify_pin


SESSION = "play_cards_0901_001"
EXPECTED_FRAMES = 645
EXPECTED_UNKNOWN = [393, 394]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def valid_states(arm_path: Path, hand_path: Path) -> tuple[int, list[int]]:
    with np.load(arm_path, allow_pickle=False) as source:
        arm = {key: np.asarray(source[key]) for key in source.files}
    with np.load(hand_path, allow_pickle=False) as source:
        hand = {key: np.asarray(source[key]) for key in source.files}
    if arm["q_arm"].shape != (EXPECTED_FRAMES, 2, 7) or hand["q_hand"].shape != (EXPECTED_FRAMES, 2, 22):
        raise RuntimeError("arm/hand shape mismatch")
    arm_valid = np.asarray(arm["valid_side_frame"], dtype=bool)
    hand_valid = np.asarray(hand["valid_side_frame"], dtype=bool)
    if arm_valid.shape != (2, EXPECTED_FRAMES) or not np.array_equal(arm_valid, hand_valid):
        raise RuntimeError("arm/hand valid identity mismatch")
    if not np.array_equal(arm["source_frames"], np.arange(EXPECTED_FRAMES)) or not np.array_equal(arm["source_frames"], hand["source_frames"]):
        raise RuntimeError("source frame identity mismatch")
    if not np.isfinite(arm["q_arm"][arm_valid.T]).all() or not np.isfinite(hand["q_hand"][hand_valid.T]).all():
        raise RuntimeError("non-finite valid Robot state")
    if not np.isnan(arm["q_arm"][~arm_valid.T]).all() or not np.isnan(hand["q_hand"][~hand_valid.T]).all():
        raise RuntimeError("UNKNOWN Robot state was filled")
    unknown = np.flatnonzero(~np.all(arm_valid, axis=0)).tolist()
    if unknown != EXPECTED_UNKNOWN:
        raise RuntimeError(f"unexpected UNKNOWN frame ids: {unknown}")
    return int(np.sum(np.all(arm_valid, axis=0))), unknown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet_path = args.packet.resolve(strict=True)
    packet = json.loads(packet_path.read_text())
    if packet.get("task_id") != "rc1_poker001_pose_only_offline_audit_resume" or packet.get("session_id") != SESSION:
        raise RuntimeError("successor packet identity mismatch")
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(Path(packet["write_root"]).resolve()):
        raise RuntimeError("fresh attempt under declared write_root required")
    sources = {name: verify_pin(pin, name) for name, pin in packet["reuse_refs"].items()}
    for name, pin in packet["code"].items():
        verify_pin(pin, f"code:{name}")
    source_start = json.loads(sources["source_start"].read_text())
    source_result = json.loads(sources["source_result"].read_text())
    source_packet = json.loads(sources["source_packet"].read_text())
    if source_result.get("terminal_status") != "FAILED_RUNTIME_FINAL" or "collision: exit=1" not in str(source_result.get("error")):
        raise RuntimeError("predecessor runtime classification mismatch")
    if source_start["run_signature"] != source_result["run_signature"] or source_start["packet"]["sha256"] != packet["reuse_refs"]["source_packet"]["sha256"]:
        raise RuntimeError("predecessor run signature/packet mismatch")
    if source_packet["session_id"] != SESSION:
        raise RuntimeError("predecessor packet session mismatch")
    arm_result, hand_result = (json.loads(sources[name].read_text()) for name in ("arm_result", "hand_result"))
    if any(value.get("session") != SESSION or value.get("frame_count") != EXPECTED_FRAMES for value in (arm_result, hand_result)):
        raise RuntimeError("phase identity/frame count mismatch")
    if arm_result["output_states"]["sha256"] != packet["reuse_refs"]["arm_states"]["sha256"] or hand_result["output_states"]["sha256"] != packet["reuse_refs"]["hand_states"]["sha256"]:
        raise RuntimeError("phase RESULT does not bind state SHA")
    if arm_result["status"] != "HOLD_NUMERIC_CANARY" or hand_result["status"] != "PASS_NUMERIC_CANARY_NO_AUTHORITY":
        raise RuntimeError("phase status mismatch")
    bilateral_count, unknown_frames = valid_states(sources["arm_states"], sources["hand_states"])
    if bilateral_count != 643:
        raise RuntimeError("expected 643 bilateral-valid frames")
    # The only changed config is the collision CLI frame count; all upstream
    # input, asset, solver and placement signatures remain the predecessor's.
    successor_signature = hashlib.sha256(json.dumps({
        "parent_run_signature": source_start["run_signature"],
        "audit_frame_count": bilateral_count,
        "audit_code_sha": packet["code"]["collision"]["sha256"],
        "render_code_sha": packet["code"]["render"]["sha256"],
        "schema": "rc1-pose-only-offline-audit-resume-v1",
    }, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True)
    atomic_json(output / "REUSE_PROOF.json", {
        "schema_version": "rc1-pose-only-exact-reuse-proof-v1",
        "session_id": SESSION, "created_at": now(),
        "source_run_signature": source_start["run_signature"],
        "successor_signature": successor_signature,
        "predecessor_terminal": ref(sources["source_result"]),
        "source_phase_results": {name: ref(sources[name]) for name in ("arm_result", "arm_states", "hand_result", "hand_states")},
        "same_solver_states": True,
        "only_changed_argument": {"collision_frame_count": {"from": 645, "to": 643}},
        "bilateral_valid_frames": bilateral_count,
        "unknown_not_audited_frame_ids": unknown_frames,
        "claim_limit": "Exactly reuses fully sealed arm/hand outputs; no partial states, state edits, or quality promotion.",
    })
    heartbeat = Heartbeat(output / "HEARTBEAT.json", packet["fencing_token"])
    heartbeat.start()
    deadline = time.monotonic() + int(packet["budget"]["wall_cap_s"])
    rows = []
    error = None
    terminal = "FAILED_RUNTIME_FINAL"
    video = None
    collision_result = None
    try:
        audit_dir = output / "collision"
        rows.append(phase("collision", [sys.executable, "-m", "chaoyang.ops.audit_robot_geometry_self_collision_v71",
            "--session-id", SESSION, "--arm-states", str(sources["arm_states"]),
            "--hand-states", str(sources["hand_states"]), "--frame-count", str(bilateral_count),
            "--output-dir", str(audit_dir)], output, heartbeat, phase_cap_s=600,
            deadline=deadline, allow_quality_hold=True))
        collision = json.loads((audit_dir / "RESULT.json").read_text())
        collision_result = ref(audit_dir / "RESULT.json")
        if collision["selected_frames"] != [index for index in range(EXPECTED_FRAMES) if index not in unknown_frames]:
            raise RuntimeError("collision did not audit every bilateral-valid frame")
        if not all(collision["gates"].values()):
            # A collision C is still evidence, but a severe self-penetration
            # must not be rendered as if structurally safe.
            if float(collision["max_penetration_m"]) > 0.003:
                terminal = "FAILED_QUALITY_C_STRUCTURAL_NO_VIDEO"
                raise RuntimeError(f"severe digital self-penetration {collision['max_penetration_m']} m")
        render_dir = output / "review"
        config = source_packet["config"]
        rows.append(phase("render", [sys.executable, "-m", "chaoyang.ops.render_robot_motion_transfer_fullsession_v2",
            "--task", "poker", "--session", SESSION, "--hawor", source_packet["inputs"]["hawor_npz"]["path"],
            "--hawor-result", source_packet["inputs"]["hawor_result"]["path"],
            "--arm-states", str(sources["arm_states"]), "--arm-result", str(sources["arm_result"]),
            "--hand-states", str(sources["hand_states"]), "--hand-result", str(sources["hand_result"]),
            "--fixed-placement-label", config["placement_label"],
            "--allow-hand-hold-review", "--allow-arm-hold-review", "--output-dir", str(render_dir)],
            output, heartbeat, phase_cap_s=1200, deadline=deadline))
        target = render_dir / f"{SESSION}_ROBOT_WORLD_FIRST_GAIN1_FULLSESSION.mp4"
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames,avg_frame_rate", "-of", "json", str(target)],
            capture_output=True, text=True, timeout=120, check=True)
        stream = json.loads(probe.stdout)["streams"][0]
        if int(stream["nb_read_frames"]) != EXPECTED_FRAMES or stream["avg_frame_rate"] != "30/1":
            raise RuntimeError("full video frame/fps mismatch")
        decode = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(target),
            "-f", "null", "-"], capture_output=True, text=True, timeout=300)
        if decode.returncode != 0:
            raise RuntimeError(f"full video decode failed: {decode.stderr[-300:]}")
        video = ref(target) | {"frames": EXPECTED_FRAMES, "fps": "30/1", "decode": "PASS_FFMPEG_XERROR"}
        terminal = "FAILED_QUALITY_C_DIAGNOSTIC_VIDEO"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        heartbeat.phase = "TERMINAL"
        heartbeat.stop()
        atomic_json(output / "RESULT.json", {
            "schema_version": "rc1-pose-only-offline-audit-resume-result-v1",
            "session_id": SESSION, "task_id": packet["task_id"], "created_at": now(),
            "terminal_status": terminal, "error": error, "successor_signature": successor_signature,
            "reuse_proof": ref(output / "REUSE_PROOF.json"),
            "collision_result": collision_result, "full_video": video,
            "arm_result": ref(sources["arm_result"]), "hand_result": ref(sources["hand_result"]),
            "bilateral_valid_frames": bilateral_count, "unknown_not_audited_frame_ids": unknown_frames,
            "input_mode": "OFFLINE_VISUAL", "training_eligible": False,
            "control_ground_truth": False, "physical_deployment_authorized": False,
            "authority_promoted": False,
            "phase_receipts": rows,
            "claim_limit": "C-grade pose-only diagnostic only; 643/645 bilateral-valid frames collision-audited, 393-394 UNKNOWN, no causal or Robot authority.",
        })
        atomic_json(output / "RESULT_SUMMARY.json", {
            "session_id": SESSION, "status": terminal, "error": error,
            "video": video, "result": ref(output / "RESULT.json"),
        })
    print(json.dumps({"status": terminal, "error": error, "video": video,
                      "result": str(output / "RESULT.json")}, ensure_ascii=False))
    return 0 if terminal == "FAILED_QUALITY_C_DIAGNOSTIC_VIDEO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
