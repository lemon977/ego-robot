#!/usr/bin/env python3
"""One bounded, CPU-only Chips182 Robot review using a fresh HaWoR revision.

The missing historical temporal NPZ cannot be recreated byte-identically.  This
runner pins the newly computed NPZ and recomputes arm/hand states; it never
combines those inputs with historical Robot states or publishes authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SESSION = "get_potato_chips_0903_182"
RUN = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_multiline_optimization_v1/lane_e_robot30_successor/chips182"
ATTEMPT = RUN / "attempts/attempt_0002"
PACKET = RUN / "TASK_PACKET_ATTEMPT_0002.json"
FRAMES = 351


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def file_ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path: Path, value: dict, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def pins() -> dict:
    base = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_multiline_optimization_v1/lane_e_robot30"
    previous = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260911_six_session_robot_motion_transfer_successor_v2"
    template = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260908_two_task_e2e_baseline_v1"
    recovered_npz = base / "attempts/attempt_0005/hawor_temporal_recovery/HAWOR_TEMPORAL_SO3_JERK_SUCCESSOR.npz"
    same_session_npz = RUN / "inputs" / SESSION / "HAWOR_TEMPORAL_SO3_JERK_SUCCESSOR.npz"
    if not same_session_npz.exists():
        same_session_npz.parent.mkdir(parents=True, exist_ok=True)
        with recovered_npz.open("rb") as source, same_session_npz.open("xb") as target:
            shutil.copyfileobj(source, target, 8 << 20)
            target.flush()
            os.fsync(target.fileno())
    if file_ref(same_session_npz)["sha256"] != file_ref(recovered_npz)["sha256"]:
        raise RuntimeError("same-session recovery copy SHA mismatch")
    items = {
        "hawor_npz": same_session_npz,
        "hawor_recovery": base / "attempts/attempt_0005/hawor_temporal_recovery/RESULT.json",
        "historical_robot_result": previous / f"full_review/{SESSION}/RESULT.json",
        "selection": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_rc1_final_v1/robot30_selection/attempts/attempt_0002/ROBOT30_SELECTION.json",
        "raw_video": Path("/mnt/data/egodata/datasets/ego/chips_cards_tracker_0903/potato_chips") / SESSION / f"CameraRecord_{SESSION}.mp4",
        "template_states": template / "robot_mount_proxy_baseline_v1/task_action_canary_v1/get_potato_chips_0902_034_same_side_world_temporal48_v3/WORLD_FIRST_SAME_SIDE_48FRAME_STATES.npz",
        "template_hawor": template / "hawor_fresh_bounded_v2_v1/get_potato_chips_0902_034/HAWOR_BOUNDED_PARAMETER_SUCCESSOR.npz",
    }
    return {key: file_ref(path) for key, path in items.items()}


def prepare() -> None:
    inputs = pins()
    recovery = json.loads(Path(inputs["hawor_recovery"]["path"]).read_text())
    if recovery["status"] != "BLOCKED_REFERENCE_PROOF" or recovery["actual_npz"]["sha256"] != inputs["hawor_npz"]["sha256"]:
        raise RuntimeError("fresh HaWoR provenance does not match recovery receipt")
    if recovery["inputs"]["raw_video"]["sha256"] != inputs["raw_video"]["sha256"]:
        raise RuntimeError("raw video SHA mismatch")
    historical = json.loads(Path(inputs["historical_robot_result"]["path"]).read_text())
    if historical["frame_count"] != FRAMES or historical["session"] != SESSION:
        raise RuntimeError("historical session/frame identity mismatch")
    selection = json.loads(Path(inputs["selection"]["path"]).read_text())
    rows = [row for row in selection["rows"] if row.get("session_id") == SESSION]
    if len(rows) != 1 or rows[0].get("task") != "chips":
        raise RuntimeError("not unique in frozen Robot30 selection")
    code_paths = [
        Path(__file__), ROOT / "src/chaoyang/ops/run_robot_motion_transfer_arm_canary_v3.py",
        ROOT / "src/chaoyang/ops/run_robot_motion_transfer_arm_canary_v2.py",
        ROOT / "src/chaoyang/ops/run_robot_hand_fullsession_v2.py",
        ROOT / "src/chaoyang/ops/audit_robot_geometry_self_collision_v71.py",
        ROOT / "src/chaoyang/ops/render_robot_motion_transfer_fullsession_v2.py",
    ]
    assets = [
        ROOT / "assets/robot/tianji/marvin_description/urdf/marvin_CCS_m6.urdf",
        ROOT / "assets/robot/kaihand/packages/KaiBot-Dexhand shell-URDF-L-260624(1620)/urdf/KaiBot-Dexhand shell-URDF-L-260624(1620).urdf",
        ROOT / "assets/robot/kaihand/packages/KaiBot-Dexhand shell-URDF-R-260424(1430)/urdf/KaiBot-Dexhand shell-URDF-R-260424(1430).urdf",
    ]
    code = {path.name: file_ref(path) for path in code_paths}
    asset_refs = {path.name: file_ref(path) for path in assets}
    config = {"task": "chips", "session_id": SESSION, "frames": FRAMES,
              "base_backoff_m": 0.2, "placement_label": "fresh offline revision; fixed backoff=0.200m",
              "input_mode": "OFFLINE_VISUAL", "collision_sample_frames": FRAMES,
              "wall_cap_seconds": 3600, "runtime_attempts": 1}
    packet = {
        "schema_version": "rc1-chips182-fresh-offline-robot-packet-v1",
        "task_id": "rc1_chips182_fresh_offline_robot_canary",
        "objective": "Recompute Robot states and full review from fresh pinned HaWoR; never adopt historical states",
        "non_goals": ["historical SHA recovery", "authority", "causal training", "physical control"],
        "session_id": SESSION, "inputs": inputs, "code": code, "assets": asset_refs,
        "config": config, "write_set": [str(ATTEMPT)],
        "stop_conditions": ["PASSED_DIAGNOSTIC", "FAILED_QUALITY_C_DIAGNOSTIC_VIDEO", "FAILED_RUNTIME_FINAL", "BLOCKED_PREREQ"],
        "claim_limit": "One OFFLINE_VISUAL full-session review only; strict numerical and digital collision audit retained.",
    }
    packet["recovery_source_npz"] = file_ref(Path(recovery["actual_npz"]["path"]))
    packet["prior_failed_attempt"] = file_ref(RUN / "attempts/attempt_0001/RESULT.json")
    write_json(PACKET, packet)
    print(json.dumps({"packet": file_ref(PACKET), "status": "READY_CPU_ONLY"}))


class Heartbeat:
    def __init__(self, path: Path, token: str):
        self.path, self.token = path, token
        self.phase = "STARTING"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self) -> None:
        ticks = Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]
        while not self.stop_event.is_set():
            write_json(self.path, {"task_id": "rc1_chips182_fresh_offline_robot_canary",
                                   "pid": os.getpid(), "startticks": ticks, "fencing_token": self.token,
                                   "phase": self.phase, "heartbeat_at": stamp()}, replace=True)
            self.stop_event.wait(30)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3)


def execute_phase(name: str, command: list[str], hb: Heartbeat, deadline: float, cap: int) -> dict:
    hb.phase = name
    log = ATTEMPT / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    seconds = min(cap, max(1, int(deadline - started)))
    with log.open("x", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        try:
            code = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            raise TimeoutError(f"{name} timed out after {seconds}s")
    row = {"phase": name, "exit_code": code, "wall_s": round(time.monotonic() - started, 3),
           "command": command, "log": file_ref(log)}
    if code not in {0, 2}:
        raise RuntimeError(f"{name} exit={code}; see {log}")
    return row


def run() -> int:
    packet_path = PACKET
    packet = json.loads(packet_path.read_text())
    if packet.get("session_id") != SESSION or ATTEMPT.exists():
        raise RuntimeError("packet identity mismatch or attempt already exists")
    for group in ("inputs", "code", "assets"):
        for key, expected in packet[group].items():
            if file_ref(Path(expected["path"])) != expected:
                raise RuntimeError(f"pinned closure changed: {group}/{key}")
    signature = hashlib.sha256(json.dumps({k: packet[k] for k in ("inputs", "code", "assets", "config")},
                                         sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    ATTEMPT.mkdir(parents=True)
    start = {"task_id": packet["task_id"], "session_id": SESSION, "started_at": stamp(),
             "run_signature": signature, "packet": file_ref(packet_path), "input_mode": "OFFLINE_VISUAL",
             "control_ground_truth": False, "physical_deployment_authorized": False}
    write_json(ATTEMPT / "RUN_START.json", start)
    recovery = packet["inputs"]["hawor_recovery"]
    raw = packet["inputs"]["raw_video"]
    hawor_result_path = ATTEMPT / f"{SESSION}_FRESH_HAWOR_REVISION_RESULT.json"
    write_json(hawor_result_path, {
        "schema_version": "rc1-fresh-hawor-temporal-revision-for-robot-v1", "session_id": SESSION,
        "status": "PASS_DEVELOPMENT_NUMERIC_NOT_HISTORICAL_SHA", "inputs": {"source_video": raw},
        "npz": packet["inputs"]["hawor_npz"], "recovery_receipt": recovery,
        "historical_sha_equal": False, "authority": False, "input_mode": "OFFLINE_VISUAL",
        "claim_limit": "Fresh numerically gated HaWoR temporal input, not byte-identical historical artifact or causal truth.",
    })
    token = f"chips182-{os.getpid()}-{int(time.time())}"
    hb = Heartbeat(ATTEMPT / "HEARTBEAT.json", token)
    hb.start()
    deadline = time.monotonic() + int(packet["config"]["wall_cap_seconds"])
    inp = {key: row["path"] for key, row in packet["inputs"].items()}
    phases: list[dict] = []
    video = None
    terminal = "FAILED_RUNTIME_FINAL"
    error = None
    arm_dir, hand_dir, audit_dir, review_dir = (ATTEMPT / name for name in ("arm", "hand", "collision", "review"))
    arm_result, hand_result = arm_dir / "RESULT.json", hand_dir / "RESULT.json"
    arm_states, hand_states = arm_dir / "ARM_CANARY_STATES.npz", hand_dir / "HAND_STATES.npz"
    try:
        py = sys.executable
        phases.append(execute_phase("arm", [py, "-m", "chaoyang.ops.run_robot_motion_transfer_arm_canary_v3",
            "--task", "chips", "--session", SESSION, "--hawor", inp["hawor_npz"],
            "--hawor-result", str(hawor_result_path), "--accepted-states", inp["template_states"],
            "--accepted-hawor", inp["template_hawor"], "--task-base-backoff-m", "0.2",
            "--output", str(arm_result)], hb, deadline, 1800))
        phases.append(execute_phase("hand", [py, "-m", "chaoyang.ops.run_robot_hand_fullsession_v2",
            "--task", "chips", "--session", SESSION, "--hawor", inp["hawor_npz"],
            "--hawor-result", str(hawor_result_path), "--accepted-states", inp["template_states"],
            "--output", str(hand_result)], hb, deadline, 1200))
        phases.append(execute_phase("collision", [py, "-m", "chaoyang.ops.audit_robot_geometry_self_collision_v71",
            "--session-id", SESSION, "--arm-states", str(arm_states), "--hand-states", str(hand_states),
            "--frame-count", str(FRAMES), "--output-dir", str(audit_dir)], hb, deadline, 600))
        phases.append(execute_phase("render", [py, "-m", "chaoyang.ops.render_robot_motion_transfer_fullsession_v2",
            "--task", "chips", "--session", SESSION, "--hawor", inp["hawor_npz"],
            "--hawor-result", str(hawor_result_path), "--arm-states", str(arm_states),
            "--arm-result", str(arm_result), "--hand-states", str(hand_states),
            "--hand-result", str(hand_result), "--fixed-placement-label", packet["config"]["placement_label"],
            "--allow-arm-hold-review", "--allow-hand-hold-review", "--output-dir", str(review_dir)], hb, deadline, 900))
        target = review_dir / f"{SESSION}_ROBOT_WORLD_FIRST_GAIN1_FULLSESSION.mp4"
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                                "-show_entries", "stream=nb_read_frames,avg_frame_rate", "-of", "json", str(target)],
                               capture_output=True, text=True, timeout=120, check=True)
        stream = json.loads(probe.stdout)["streams"][0]
        if int(stream["nb_read_frames"]) != FRAMES:
            raise RuntimeError(f"video frames {stream['nb_read_frames']} != {FRAMES}")
        decoded = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(target),
                                  "-f", "null", "-"], capture_output=True, text=True, timeout=300)
        if decoded.returncode != 0:
            raise RuntimeError(f"full video decode failed: {decoded.stderr[-400:]}")
        video = file_ref(target) | {"frames": FRAMES, "fps": stream["avg_frame_rate"], "decode": "PASS_FFMPEG_XERROR"}
        arm_status = json.loads(arm_result.read_text())["status"]
        hand_status = json.loads(hand_result.read_text())["status"]
        collision_status = json.loads((audit_dir / "RESULT.json").read_text())["status"]
        terminal = ("PASSED_DIAGNOSTIC" if all(x.startswith("PASS") for x in (arm_status, hand_status, collision_status))
                    else "FAILED_QUALITY_C_DIAGNOSTIC_VIDEO")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        hb.phase = "TERMINAL"
        hb.stop()
        write_json(ATTEMPT / "RESULT.json", {
            "schema_version": "rc1-chips182-fresh-offline-robot-result-v1", "task_id": packet["task_id"],
            "session_id": SESSION, "terminal_status": terminal, "error": error,
            "run_signature": signature, "packet": file_ref(packet_path), "phase_receipts": phases,
            "full_video": video, "arm_result": file_ref(arm_result) if arm_result.exists() else None,
            "hand_result": file_ref(hand_result) if hand_result.exists() else None,
            "collision_result": file_ref(audit_dir / "RESULT.json") if (audit_dir / "RESULT.json").exists() else None,
            "input_mode": "OFFLINE_VISUAL", "training_eligible": False, "authority": False,
            "control_ground_truth": False, "physical_deployment_authorized": False,
            "claim_limit": "Single full-session offline digital Robot diagnostic; fresh HaWoR revision does not inherit historical authority.",
        })
        write_json(ATTEMPT / "RESULT_SUMMARY.json", {"status": terminal, "video": video,
                                                      "error": error, "result": file_ref(ATTEMPT / "RESULT.json")})
    print(json.dumps({"status": terminal, "video": video, "error": error}, ensure_ascii=False))
    return 0 if terminal == "PASSED_DIAGNOSTIC" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
        return 0
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
