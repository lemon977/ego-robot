#!/usr/bin/env python3
"""Bounded, immutable pose-only Robot visual canary for one frozen RC1 session.

This is an offline diagnostic, not an RC1 causal input or Robot authority.  It
uses the existing motion-transfer arm, hand, collision and rendering programs
without weakening their numeric gates.  A quality-HOLD may still yield a
watermarked full-session review video, but never a hard-geometry PASS.
"""
from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[3]
SESSION = "play_cards_0901_001"
TASK = "poker"
EXPECTED_FRAMES = 645


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def verify_pin(pin: dict[str, Any], label: str) -> Path:
    path = Path(pin["path"]).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label}: regular non-symlink file required: {path}")
    actual = ref(path)
    if actual["bytes"] != pin["bytes"] or actual["sha256"] != pin["sha256"]:
        raise RuntimeError(f"{label}: bytes/SHA mismatch: {path}")
    return path


class Heartbeat:
    def __init__(self, path: Path, token: str):
        self.path, self.token = path, token
        self.phase = "STARTING"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        startticks = Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]
        while not self._stop.is_set():
            atomic_json(self.path, {
                "schema_version": "rc1-pose-only-heartbeat-v1",
                "session_id": SESSION,
                "pid": os.getpid(),
                "process_startticks": startticks,
                "fencing_token": self.token,
                "phase": self.phase,
                "heartbeat_at": now(),
                "authority": False,
            }, replace=True)
            self._stop.wait(30)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def phase(name: str, command: list[str], attempt: Path, heartbeat: Heartbeat,
          *, phase_cap_s: int, deadline: float, allow_quality_hold: bool = False) -> dict[str, Any]:
    heartbeat.phase = name
    log = attempt / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("90-minute global wall cap reached")
    cap = max(1, min(phase_cap_s, int(remaining)))
    started = time.monotonic()
    with log.open("x", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"})
        try:
            code = process.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            raise TimeoutError(f"{name}: phase cap {cap}s exhausted; see {log}")
    row = {"phase": name, "command": command, "exit_code": code,
           "wall_seconds": round(time.monotonic() - started, 3), "log": ref(log)}
    if code not in ({0, 2} if allow_quality_hold else {0}):
        raise RuntimeError(f"{name}: exit={code}; see {log}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    packet_path = args.packet.resolve(strict=True)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if packet.get("session_id") != SESSION or packet.get("task_id") != "rc1_poker001_pose_only_offline_canary":
        raise RuntimeError("packet identity mismatch")
    attempt = args.output.resolve()
    declared_root = Path(packet["write_root"]).resolve()
    if not attempt.is_relative_to(declared_root) or attempt.exists():
        raise RuntimeError("fresh attempt under packet write_root required")
    for label, pin in packet["inputs"].items():
        verify_pin(pin, label)
    for label, pin in packet["code"].items():
        verify_pin(pin, f"code:{label}")
    selection = json.loads(Path(packet["inputs"]["selection"]["path"]).read_text())
    matches = [row for row in selection["rows"] if row.get("session_id") == SESSION]
    if len(matches) != 1 or matches[0].get("task") != TASK:
        raise RuntimeError("session not uniquely in frozen Robot30 selection")
    matrix = json.loads(Path(packet["inputs"]["matrix"]["path"]).read_text())
    matches = [row for row in matrix["rows"] if row.get("session_id") == SESSION]
    if len(matches) != 1 or not matches[0].get("three_upstream_ab") or matches[0].get("frame_count") != EXPECTED_FRAMES:
        raise RuntimeError("triple A/B or frame-count contract failed")
    signature_payload = {
        "schema": "rc1-pose-only-offline-single-v1",
        "inputs": {key: value["sha256"] for key, value in packet["inputs"].items()},
        "code": {key: value["sha256"] for key, value in packet["code"].items()},
        "config": packet["config"],
        "weights": "PINNED_URDF_ASSET_PIN",
        "calibration": "ABSENT_POSE_ONLY",
    }
    run_signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    attempt.mkdir(parents=True)
    atomic_json(attempt / "RUN_START.json", {
        "task_id": packet["task_id"], "session_id": SESSION, "created_at": now(),
        "packet": ref(packet_path), "run_signature": run_signature,
        "input_mode": "OFFLINE_VISUAL", "gpu_required": False,
        "central_gpu_lease_at_start": json.loads((ROOT / "_run/current/GPU_LEASE.json").read_text()),
        "control_ground_truth": False, "physical_deployment_authorized": False,
    })
    heartbeat = Heartbeat(attempt / "HEARTBEAT.json", packet["fencing_token"])
    heartbeat.start()
    deadline = time.monotonic() + int(packet["budget"]["wall_cap_s"])
    rows: list[dict[str, Any]] = []
    terminal = "FAILED_RUNTIME_FINAL"
    error: str | None = None
    video: dict[str, Any] | None = None
    try:
        inp = {key: str(Path(pin["path"])) for key, pin in packet["inputs"].items()}
        arm_dir, hand_dir, audit_dir, render_dir = (attempt / name for name in ("arm", "hand", "collision", "review"))
        arm_result, hand_result = arm_dir / "RESULT.json", hand_dir / "RESULT.json"
        arm_states, hand_states = arm_dir / "ARM_CANARY_STATES.npz", hand_dir / "HAND_STATES.npz"
        config = packet["config"]
        rows.append(phase("arm", [sys.executable, "-m", "chaoyang.ops.run_robot_motion_transfer_arm_canary_v3",
            "--task", TASK, "--session", SESSION, "--hawor", inp["hawor_npz"],
            "--hawor-result", inp["hawor_result"], "--accepted-states", inp["template_states"],
            "--accepted-hawor", inp["template_hawor"], "--task-base-backoff-m", str(config["task_base_backoff_m"]),
            "--output", str(arm_result)], attempt, heartbeat, phase_cap_s=2400, deadline=deadline, allow_quality_hold=True))
        rows.append(phase("hand", [sys.executable, "-m", "chaoyang.ops.run_robot_hand_fullsession_v2",
            "--task", TASK, "--session", SESSION, "--hawor", inp["hawor_npz"],
            "--hawor-result", inp["hawor_result"], "--accepted-states", inp["template_states"],
            "--output", str(hand_result)], attempt, heartbeat, phase_cap_s=2100, deadline=deadline, allow_quality_hold=True))
        rows.append(phase("collision", [sys.executable, "-m", "chaoyang.ops.audit_robot_geometry_self_collision_v71",
            "--session-id", SESSION, "--arm-states", str(arm_states), "--hand-states", str(hand_states),
            "--frame-count", str(EXPECTED_FRAMES), "--output-dir", str(audit_dir)],
            attempt, heartbeat, phase_cap_s=600, deadline=deadline, allow_quality_hold=True))
        rows.append(phase("render", [sys.executable, "-m", "chaoyang.ops.render_robot_motion_transfer_fullsession_v2",
            "--task", TASK, "--session", SESSION, "--hawor", inp["hawor_npz"],
            "--hawor-result", inp["hawor_result"], "--arm-states", str(arm_states),
            "--arm-result", str(arm_result), "--hand-states", str(hand_states),
            "--hand-result", str(hand_result), "--fixed-placement-label", config["placement_label"],
            "--allow-hand-hold-review", "--allow-arm-hold-review", "--output-dir", str(render_dir)],
            attempt, heartbeat, phase_cap_s=1200, deadline=deadline))
        target = render_dir / f"{SESSION}_ROBOT_WORLD_FIRST_GAIN1_FULLSESSION.mp4"
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=nb_read_frames,avg_frame_rate", "-of", "json", str(target)],
            capture_output=True, text=True, timeout=120, check=True)
        stream = json.loads(probe.stdout)["streams"][0]
        if int(stream["nb_read_frames"]) != EXPECTED_FRAMES:
            raise RuntimeError(f"full video frame count {stream['nb_read_frames']} != {EXPECTED_FRAMES}")
        decoded = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(target),
            "-f", "null", "-"], capture_output=True, text=True, timeout=300)
        if decoded.returncode != 0:
            raise RuntimeError(f"ffmpeg full decode failed: {decoded.stderr[-400:]}")
        video = ref(target) | {"frames": EXPECTED_FRAMES, "fps": stream["avg_frame_rate"], "decode": "PASS_FFMPEG_XERROR"}
        arm = json.loads(arm_result.read_text())
        hand = json.loads(hand_result.read_text())
        collision = json.loads((audit_dir / "RESULT.json").read_text())
        terminal = "PASSED_DIAGNOSTIC" if (arm["status"].startswith("PASS") and hand["status"].startswith("PASS")
                   and collision["status"].startswith("PASS")) else "FAILED_QUALITY_C_DIAGNOSTIC_VIDEO"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        heartbeat.phase = "TERMINAL"
        heartbeat.stop()
        atomic_json(attempt / "RESULT.json", {
            "schema_version": "rc1-poker-pose-only-offline-canary-v1",
            "task_id": packet["task_id"], "session_id": SESSION, "task": TASK,
            "created_at": now(), "terminal_status": terminal, "error": error,
            "run_signature": run_signature, "packet": ref(packet_path),
            "phase_receipts": rows, "full_video": video,
            "arm_result": ref(attempt / "arm/RESULT.json") if (attempt / "arm/RESULT.json").is_file() else None,
            "hand_result": ref(attempt / "hand/RESULT.json") if (attempt / "hand/RESULT.json").is_file() else None,
            "collision_result": ref(attempt / "collision/RESULT.json") if (attempt / "collision/RESULT.json").is_file() else None,
            "input_mode": "OFFLINE_VISUAL", "training_eligible": False,
            "control_ground_truth": False, "physical_deployment_authorized": False,
            "authority_promoted": False,
            "claim_limit": "One full-session pose-only development video; hard geometry only if full collision and numeric gates pass. Not causal, contact, control or physical authority.",
        })
        atomic_json(attempt / "RESULT_SUMMARY.json", {
            "task_id": packet["task_id"], "status": terminal,
            "video": video, "error": error, "result": ref(attempt / "RESULT.json"),
        })
    print(json.dumps({"status": terminal, "video": video, "error": error,
                      "result": str(attempt / "RESULT.json")}, ensure_ascii=False))
    return 0 if terminal == "PASSED_DIAGNOSTIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
