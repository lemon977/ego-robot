#!/usr/bin/env python3
"""One-session, Clean-independent, CPU-only pose-only Robot review runner.

The input packet is immutable and pins every source/code/asset.  This runner
does not turn an offline trajectory into causal training or Robot authority.
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
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "rc1-pose-only-offline-generic-packet-v1"
REQUIRED_INPUTS = (
    "selection", "matrix", "hawor_result", "hawor_npz", "role_result",
    "object_result", "raw_video", "template_states", "template_hawor",
    "robot_asset_pin",
)
REQUIRED_CODE = (
    "runner", "run_robot_motion_transfer_arm_canary_v3.py",
    "run_robot_hand_fullsession_v2.py", "audit_robot_geometry_self_collision_v71.py",
    "render_robot_motion_transfer_fullsession_v2.py",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify_ref(pin: dict[str, Any], label: str) -> Path:
    path = Path(pin["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label}: absolute regular file required: {path}")
    actual = ref(path)
    if (actual["bytes"], actual["sha256"]) != (pin["bytes"], pin["sha256"]):
        raise RuntimeError(f"{label}: bytes/SHA mismatch: {path}")
    return path


def atomic_json(path: Path, data: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def producer_signature(packet: dict[str, Any]) -> str:
    payload = {
        "schema_version": SCHEMA,
        "task": packet["task"],
        "session_id": packet["session_id"],
        "frame_count": packet["frame_count"],
        "inputs": {key: value["sha256"] for key, value in sorted(packet["inputs"].items())},
        "code": {key: value["sha256"] for key, value in sorted(packet["code"].items())},
        "config": packet["config"],
        "initialization_provenance": packet.get("initialization_provenance"),
        "model_weights": packet["model_weights"],
        "calibration": packet["calibration"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_probe(path: Path) -> dict[str, Any]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=180, check=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("source video must have one selected video stream")
    stream = streams[0]
    return {"frames": int(stream["nb_read_frames"]), "fps": stream["avg_frame_rate"],
            "width": int(stream["width"]), "height": int(stream["height"])}


def preflight(packet_path: Path) -> dict[str, Any]:
    packet_path = packet_path.resolve(strict=True)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("schema_version") != SCHEMA or packet.get("mode") != "OFFLINE_VISUAL":
        raise RuntimeError("packet schema/mode mismatch")
    session, task, frames = packet.get("session_id"), packet.get("task"), packet.get("frame_count")
    if (not isinstance(session, str) or not session.startswith(("play_cards_", "get_potato_chips_"))
            or task not in ("poker", "chips") or not isinstance(frames, int) or frames <= 0):
        raise RuntimeError("invalid full session identity/task/frame count")
    if (task == "poker") != session.startswith("play_cards_"):
        raise RuntimeError("session prefix/task mismatch")
    if set(REQUIRED_INPUTS) - packet.get("inputs", {}).keys():
        raise RuntimeError("missing required input pins")
    if set(REQUIRED_CODE) - packet.get("code", {}).keys():
        raise RuntimeError("missing required code pins")
    for name, pin in packet["inputs"].items():
        verify_ref(pin, f"input:{name}")
    for name, pin in packet["code"].items():
        verify_ref(pin, f"code:{name}")
    if Path(packet["code"]["runner"]["path"]).resolve() != Path(__file__).resolve():
        raise RuntimeError("packet runner pin does not bind this implementation")
    selection = json.loads(Path(packet["inputs"]["selection"]["path"]).read_text())
    selected = [row for row in selection["rows"] if row.get("session_id") == session]
    matrix = json.loads(Path(packet["inputs"]["matrix"]["path"]).read_text())
    upstream = [row for row in matrix["rows"] if row.get("session_id") == session]
    if len(selected) != 1 or len(upstream) != 1 or selected[0].get("task") != task:
        raise RuntimeError("session not unique in frozen selection/matrix")
    if upstream[0].get("frame_count") != frames or not upstream[0].get("three_upstream_ab"):
        raise RuntimeError("frame count or triple A/B preflight failed")
    if not all(upstream[0][stage].get("grade") in ("A", "B") for stage in ("hawor", "role_mask", "object_mask")):
        raise RuntimeError("upstream grade not A/B")
    revision_mode = packet.get("hawor_input_revision", "PINNED_CURRENT")
    if revision_mode not in ("PINNED_CURRENT", "DEVELOPMENT_SUCCESSOR"):
        raise RuntimeError("unknown HaWoR input revision mode")
    for stage, label in (("role_mask", "role_result"), ("object_mask", "object_result")):
        if upstream[0][stage]["result"]["sha256"] != packet["inputs"][label]["sha256"]:
            raise RuntimeError(f"{stage}: packet does not match frozen matrix")
    if revision_mode == "PINNED_CURRENT":
        if upstream[0]["hawor"]["result"]["sha256"] != packet["inputs"]["hawor_result"]["sha256"]:
            raise RuntimeError("HaWoR packet does not match frozen matrix")
    else:
        parent = packet["inputs"].get("hawor_parent_result")
        if parent is None or parent["sha256"] != upstream[0]["hawor"]["result"]["sha256"]:
            raise RuntimeError("development HaWoR must pin the frozen parent result")
    hawor = json.loads(Path(packet["inputs"]["hawor_result"]["path"]).read_text())
    hawor_npz_ref = hawor.get("outputs", {}).get("npz", hawor.get("npz", {}))
    if hawor_npz_ref.get("sha256") != packet["inputs"]["hawor_npz"]["sha256"]:
        raise RuntimeError("HaWoR result/NPZ mismatch")
    if revision_mode == "DEVELOPMENT_SUCCESSOR":
        recovery = hawor.get("recovery_receipt", {})
        if (hawor.get("authority") is not False or hawor.get("historical_sha_equal") is not False
                or recovery.get("sha256") != packet["inputs"].get("hawor_recovery_receipt", {}).get("sha256")):
            raise RuntimeError("development HaWoR successor provenance is incomplete")
    source = hawor.get("inputs", {}).get("source_video", {})
    if source.get("sha256") != packet["inputs"]["raw_video"]["sha256"]:
        raise RuntimeError("HaWoR/raw video mismatch")
    source_info = source_probe(Path(packet["inputs"]["raw_video"]["path"]))
    if source_info["frames"] != frames:
        raise RuntimeError(f"source video has {source_info['frames']} frames, expected {frames}")
    config, budget = packet["config"], packet["budget"]
    if (config.get("collision_frames") != frames or config.get("render_fps") != "source_fps"
            or not isinstance(config.get("task_base_backoff_m"), (int, float))
            or not isinstance(config.get("placement_label"), str)):
        raise RuntimeError("config frame/fps/placement contract mismatch")
    if (budget.get("runtime_attempts") != 1 or budget.get("gpu_bytes") != 0
            or not 0 < budget.get("wall_cap_s", 0) <= 5400):
        raise RuntimeError("budget contract mismatch")
    if packet.get("model_weights") != "PINNED_URDF_ASSET_PIN" or packet.get("calibration") != "ABSENT_POSE_ONLY":
        raise RuntimeError("pose-only model/calibration semantics mismatch")
    provenance = packet.get("initialization_provenance", {})
    if provenance.get("mode") != "INDEPENDENT_TASK_PRESET_NEUTRAL_OR_SAME_SESSION_PREFIX":
        raise RuntimeError(
            "CROSS_SESSION_TEMPLATE_BLOCKED: v3 arm initializer imports accepted_states/"
            "accepted_hawor q0, T_camera_base, offset and rotation from another session; "
            f"sources={provenance.get('template_source_session_ids', {})}"
        )
    # Even a packet claiming a safe mode must not bypass the implementation gate:
    # the v3 command below has no neutral, target-session-only initializer.
    raise RuntimeError("INDEPENDENT_PLACEMENT_RUNNER_NOT_IMPLEMENTED: do not execute v3 template initializer")
    return {"status": "PASS_INPUT_PREFLIGHT", "task_id": packet["task_id"], "session_id": session,
            "frame_count": frames, "source_video": source_info,
            "producer_signature": producer_signature(packet),
            "input_mode": "OFFLINE_VISUAL", "training_eligible": False,
            "control_ground_truth": False, "physical_deployment_authorized": False}


class Heartbeat:
    def __init__(self, path: Path, session: str, token: str):
        self.path, self.session, self.token = path, session, token
        self.phase = "STARTING"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self) -> None:
        startticks = Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]
        while not self.stop_event.is_set():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            with temp.open("w", encoding="utf-8") as stream:
                json.dump({"session_id": self.session, "pid": os.getpid(),
                           "process_startticks": startticks, "fencing_token": self.token,
                           "phase": self.phase, "heartbeat_at": now()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
            self.stop_event.wait(30)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)


def run_phase(name: str, command: list[str], attempt: Path, heartbeat: Heartbeat,
              *, cap_s: int, deadline: float, allow_quality_hold: bool = False) -> dict[str, Any]:
    heartbeat.phase = name
    remaining = int(deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError("global wall cap exhausted")
    log = attempt / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True,
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"})
        try:
            exit_code = child.wait(timeout=min(cap_s, remaining))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            raise TimeoutError(f"{name}: time cap exhausted; see {log}")
    receipt = {"phase": name, "command": command, "exit_code": exit_code,
               "wall_seconds": round(time.monotonic() - started, 3), "log": ref(log)}
    if exit_code not in ({0, 2} if allow_quality_hold else {0}):
        raise RuntimeError(f"{name}: exit={exit_code}; see {log}")
    return receipt


def full_video(path: Path, expected_frames: int, expected_fps: str) -> dict[str, Any]:
    info = source_probe(path)
    if info["frames"] != expected_frames or info["fps"] != expected_fps:
        raise RuntimeError(f"full video frames/fps mismatch: {info}")
    decoded = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(path),
                              "-f", "null", "-"], capture_output=True, text=True, timeout=300)
    if decoded.returncode:
        raise RuntimeError(f"full video decode failed: {decoded.stderr[-400:]}")
    return ref(path) | {"frames": expected_frames, "fps": info["fps"], "decode": "PASS_FFMPEG_XERROR"}


def require_exact_audited_frame_ids(eligible: list[int], selected_frames: list[int]) -> None:
    """Do not mistake a requested count for coverage of nonconsecutive IDs."""
    if selected_frames != eligible:
        raise RuntimeError("collision auditor did not cover exact nonconsecutive eligible IDs")


def audit_numeric_states(arm_path: Path, hand_path: Path, frames: int) -> dict[str, Any]:
    """Check finite/proper transforms and the pinned URDF joint limits."""
    from chaoyang.ops import run_robot_hand_fullsession_v2 as hand_program
    from chaoyang.ops import run_robot_motion_transfer_arm_canary_v2 as arm_program

    with np.load(arm_path, allow_pickle=False) as source:
        arm = {key: np.asarray(source[key]) for key in source.files}
    with np.load(hand_path, allow_pickle=False) as source:
        hand = {key: np.asarray(source[key]) for key in source.files}
    qa = np.asarray(arm["q_arm"], dtype=np.float64)
    qh = np.asarray(hand["q_hand"], dtype=np.float64)
    valid = np.asarray(arm["valid_side_frame"], dtype=bool)
    if qa.shape != (frames, 2, 7) or qh.shape != (frames, 2, 22) or valid.shape != (2, frames):
        raise RuntimeError("arm/hand q or validity shape mismatch")
    if not np.array_equal(valid, hand["valid_side_frame"]):
        raise RuntimeError("arm/hand validity mismatch")
    if not np.array_equal(arm["source_frames"], np.arange(frames)) or not np.array_equal(
        arm["source_frames"], hand["source_frames"]
    ):
        raise RuntimeError("arm/hand source frame identity mismatch")
    mask = valid.T
    if not np.isfinite(qa[mask]).all() or not np.isfinite(qh[mask]).all():
        raise RuntimeError("valid arm/hand q contains non-finite values")
    if not np.isnan(qa[~mask]).all() or not np.isnan(qh[~mask]).all():
        raise RuntimeError("unknown arm/hand q must remain NaN")
    roots = np.asarray(arm["T_actual_hand_root_world"], dtype=np.float64)
    if roots.shape != (frames, 2, 4, 4):
        raise RuntimeError("actual hand-root transform shape mismatch")
    selected = roots[mask]
    if not np.isfinite(selected).all() or not np.allclose(selected[:, 3], [0, 0, 0, 1], atol=1e-8):
        raise RuntimeError("valid hand-root transform is non-finite/non-homogeneous")
    rotations = selected[:, :3, :3]
    if not np.allclose(rotations @ np.transpose(rotations, (0, 2, 1)), np.eye(3), atol=1e-4):
        raise RuntimeError("hand-root rotation not orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-4):
        raise RuntimeError("hand-root rotation not proper")
    assets = arm_program.old.load_pinned_robot_assets(ROOT)
    arm_lower, arm_upper = arm_program.taskfit.arm_limits(assets)
    contracts = hand_program.handfit.model_contract(hand_program.old.official,
                                                     hand_program.shared.wrist_adapter, assets)
    for side in range(2):
        if not np.all((qa[:, side][valid[side]] >= arm_lower[side] - 1e-7)
                      & (qa[:, side][valid[side]] <= arm_upper[side] + 1e-7)):
            raise RuntimeError(f"side {side}: URDF arm limit violation")
        lower, upper = contracts[side]["lower"], contracts[side]["upper"]
        if not np.all((qh[:, side][valid[side]] >= lower - 1e-7)
                      & (qh[:, side][valid[side]] <= upper + 1e-7)):
            raise RuntimeError(f"side {side}: URDF hand limit violation")
    return {"status": "PASS_STRUCTURAL_NUMERIC", "frames": frames,
            "valid_side_rows": int(valid.sum()), "bilateral_valid_frames": int(np.all(valid, axis=0).sum()),
            "invalid_side_rows": int(valid.size - valid.sum()),
            "claim_limit": "Digital finite/SE3/URDF limit consistency only; not physical precision."}


def classify_digital_geometry(*, frames: int, bilateral_valid_count: int,
                              arm_result: dict[str, Any], hand_result: dict[str, Any],
                              collision_result: dict[str, Any]) -> dict[str, Any]:
    """Keep hard numeric/digital gates distinct from KaiHand pose similarity."""
    arm_gates = arm_result.get("gates", {})
    hand_gates = hand_result.get("gates", {})
    arm_hard = all(arm_gates.get(key) is True for key in (
        "base_fixed", "motion_gain_exact_one", "missing_frames_unknown_not_filled",
        "per_side_anchor_is_observed",
    ))
    hand_hard = all(hand_gates.get(key) is True for key in (
        "missing_unknown_not_filled", "thumb_independent_q0_to_q5", "four_finger_chain_semantics",
    ))
    collision_hard = collision_result.get("status", "").startswith("PASS")
    hard_valid_rows = arm_hard and hand_hard and collision_hard and bilateral_valid_count > 0
    return {
        "terminal_status": "PASSED_DIAGNOSTIC" if hard_valid_rows else "FAILED_QUALITY_C_DIAGNOSTIC_VIDEO",
        "hard_geometry_valid_rows_pass": hard_valid_rows,
        "unknown_frames": frames - bilateral_valid_count,
        "soft_human_pose_pass": hand_gates.get("anatomy_all_observed"),
        "soft_arm_pose_branch_pass": arm_gates.get("all_observed_rows_pose_branch"),
        "soft_temporal_pass": all(hand_gates.get(key) is True for key in ("velocity", "acceleration")),
        "arm_ik_target_diagnostic": arm_gates.get("all_observed_rows_pose_branch"),
        "digital_collision_pass": collision_hard,
        "claim_limit": "PASS applies to observed valid rows only; UNKNOWN frames are not collision-certified and no causal/physical authority is granted.",
    }


def publish_watermark(source: Path, target: Path, *, quality_c: bool, unknown_frames: list[int]) -> None:
    if target.exists():
        raise FileExistsError(target)
    label = "FAILED_QUALITY_C | OFFLINE_VISUAL | NOT TRAINING" if quality_c else "DEVELOPMENT | OFFLINE_VISUAL | NOT TRAINING"
    color = "red" if quality_c else "black"
    filter_text = (f"drawbox=x=0:y=0:w=iw:h=56:color={color}@0.9:t=fill,"
                   f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
                   f"text='{label}':fontcolor=white:fontsize=24:x=20:y=13")
    if unknown_frames:
        terms = "+".join(f"eq(n\\,{frame})" for frame in unknown_frames)
        filter_text += (",drawbox=x=0:y=0:w=iw:h=ih:color=yellow@0.8:t=12:"
                        f"enable='{terms}',drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
                        f"text='UNKNOWN ROBOT STATE':fontcolor=yellow:fontsize=24:x=20:y=65:enable='{terms}'")
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror",
                    "-i", str(source), "-vf", filter_text, "-an", "-c:v", "libx264", "-preset", "fast",
                    "-crf", "19", "-pix_fmt", "yuv420p", str(target)], check=True, timeout=600)


def execute(packet_path: Path, output: Path) -> int:
    checked = preflight(packet_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    session, task, frames = packet["session_id"], packet["task"], packet["frame_count"]
    output = output.resolve()
    write_root = Path(packet["write_root"]).resolve()
    if not output.is_relative_to(write_root) or output.exists():
        raise RuntimeError("fresh output under packet write_root required")
    governance = subprocess.run([sys.executable, "-m", "chaoyang.governance.validate_governance_state"],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
    if governance.returncode or json.loads(governance.stdout).get("status") != "PASS":
        raise RuntimeError("current governance not PASS")
    output.mkdir(parents=True)
    atomic_json(output / "RUN_START.json", checked | {"packet": ref(packet_path), "created_at": now(),
                                                "gpu_required": False, "authority_promoted": False})
    heartbeat = Heartbeat(output / "HEARTBEAT.json", session, packet["fencing_token"])
    heartbeat.start()
    deadline = time.monotonic() + packet["budget"]["wall_cap_s"]
    receipts: list[dict[str, Any]] = []
    terminal, error, video = "FAILED_RUNTIME_FINAL", None, None
    eligible: list[int] = []
    try:
        inp = {key: pin["path"] for key, pin in packet["inputs"].items()}
        config, budget = packet["config"], packet["budget"]
        arm, hand, collision, review = (output / name for name in ("arm", "hand", "collision", "review"))
        arm_result, hand_result = arm / "RESULT.json", hand / "RESULT.json"
        arm_states, hand_states = arm / "ARM_CANARY_STATES.npz", hand / "HAND_STATES.npz"
        receipts.append(run_phase("arm", [sys.executable, "-m", "chaoyang.ops.run_robot_motion_transfer_arm_canary_v3",
            "--task", task, "--session", session, "--hawor", inp["hawor_npz"],
            "--hawor-result", inp["hawor_result"], "--accepted-states", inp["template_states"],
            "--accepted-hawor", inp["template_hawor"], "--task-base-backoff-m",
            str(config["task_base_backoff_m"]), "--output", str(arm_result)], output, heartbeat,
            cap_s=budget["arm_cap_s"], deadline=deadline, allow_quality_hold=True))
        receipts.append(run_phase("hand", [sys.executable, "-m", "chaoyang.ops.run_robot_hand_fullsession_v2",
            "--task", task, "--session", session, "--hawor", inp["hawor_npz"],
            "--hawor-result", inp["hawor_result"], "--accepted-states", inp["template_states"],
            "--output", str(hand_result)], output, heartbeat,
            cap_s=budget["hand_cap_s"], deadline=deadline, allow_quality_hold=True))
        numeric = audit_numeric_states(arm_states, hand_states, frames)
        atomic_json(output / "NUMERIC_AUDIT.json", numeric)
        with np.load(arm_states, allow_pickle=False) as a, np.load(hand_states, allow_pickle=False) as h:
            arm_valid = np.asarray(a["valid_side_frame"], dtype=bool)
            hand_valid = np.asarray(h["valid_side_frame"], dtype=bool)
            if arm_valid.shape != (2, frames) or not np.array_equal(arm_valid, hand_valid):
                raise RuntimeError("arm/hand validity shape or identity mismatch")
            if not np.array_equal(a["source_frames"], h["source_frames"]):
                raise RuntimeError("arm/hand source frame identity mismatch")
            eligible = np.flatnonzero(np.all(arm_valid, axis=0)).astype(int).tolist()
        atomic_json(output / "FRAME_VALIDITY.json", {"session_id": session, "frame_count": frames,
                    "bilateral_valid_frame_ids": eligible,
                    "unknown_frame_ids": sorted(set(range(frames)) - set(eligible)),
                    "claim_limit": "Only explicitly listed frame IDs can receive digital collision audit."})
        if eligible:
            receipts.append(run_phase("collision", [sys.executable, "-m", "chaoyang.ops.audit_robot_geometry_self_collision_v71",
                "--session-id", session, "--arm-states", str(arm_states), "--hand-states", str(hand_states),
                "--frame-count", str(len(eligible)), "--output-dir", str(collision)], output, heartbeat,
                cap_s=budget["collision_cap_s"], deadline=deadline, allow_quality_hold=True))
            audited = json.loads((collision / "RESULT.json").read_text())
            require_exact_audited_frame_ids(eligible, audited.get("selected_frames", []))
        receipts.append(run_phase("render", [sys.executable, "-m", "chaoyang.ops.render_robot_motion_transfer_fullsession_v2",
            "--task", task, "--session", session, "--hawor", inp["hawor_npz"],
            "--hawor-result", inp["hawor_result"], "--arm-states", str(arm_states),
            "--arm-result", str(arm_result), "--hand-states", str(hand_states),
            "--hand-result", str(hand_result), "--fixed-placement-label", config["placement_label"],
            "--allow-hand-hold-review", "--allow-arm-hold-review", "--output-dir", str(review)],
            output, heartbeat, cap_s=budget["render_cap_s"], deadline=deadline))
        source_video = review / f"{session}_ROBOT_WORLD_FIRST_GAIN1_FULLSESSION.mp4"
        full_video(source_video, frames, checked["source_video"]["fps"])
        arm_data = json.loads(arm_result.read_text())
        hand_data = json.loads(hand_result.read_text())
        collision_data = json.loads((collision / "RESULT.json").read_text()) if eligible else {}
        gate_decision = classify_digital_geometry(frames=frames, bilateral_valid_count=len(eligible),
                                                  arm_result=arm_data, hand_result=hand_data,
                                                  collision_result=collision_data)
        atomic_json(output / "HARD_SOFT_DECISION.json", gate_decision)
        terminal = gate_decision["terminal_status"]
        hard_pass = gate_decision["hard_geometry_valid_rows_pass"]
        marked = review / f"{session}_ROBOT_OFFLINE_VISUAL_WATERMARKED_FULLSESSION.mp4"
        publish_watermark(source_video, marked, quality_c=not hard_pass,
                          unknown_frames=sorted(set(range(frames)) - set(eligible)))
        video = full_video(marked, frames, checked["source_video"]["fps"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        heartbeat.phase = "TERMINAL"
        heartbeat.stop()
        result = {"schema_version": "rc1-pose-only-offline-generic-result-v1",
                  "task_id": packet["task_id"], "session_id": session, "task": task,
                  "terminal_status": terminal, "error": error, "created_at": now(),
                  "producer_signature": checked["producer_signature"], "packet": ref(packet_path),
                  "phase_receipts": receipts, "bilateral_valid_frame_ids": eligible,
                  "numeric_audit": ref(output / "NUMERIC_AUDIT.json") if (output / "NUMERIC_AUDIT.json").is_file() else None,
                  "hard_soft_decision": ref(output / "HARD_SOFT_DECISION.json") if (output / "HARD_SOFT_DECISION.json").is_file() else None,
                  "full_video": video, "input_mode": "OFFLINE_VISUAL", "training_eligible": False,
                  "control_ground_truth": False, "physical_deployment_authorized": False,
                  "authority_promoted": False,
                  "claim_limit": "One full-session offline diagnostic. Hard geometry only when all structural and digital collision gates pass; no causal training, metric contact, control or deployment authority."}
        atomic_json(output / "RESULT.json", result)
        atomic_json(output / "RESULT_SUMMARY.json", {"session_id": session, "status": terminal,
                    "video": video, "error": error, "result": ref(output / "RESULT.json")})
    print(json.dumps({"status": terminal, "video": video, "error": error}, ensure_ascii=False))
    return 0 if terminal == "PASSED_DIAGNOSTIC" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(preflight(args.packet), ensure_ascii=False))
        return 0
    if args.output is None:
        parser.error("--output is required unless --preflight-only")
    return execute(args.packet, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
