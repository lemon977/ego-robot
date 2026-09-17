#!/usr/bin/env python3
"""Run bounded H3/H4 sensor-domain canaries after their H0 prerequisite closes.

This worker intentionally does not expand 106,868 frames blindly.  It records
one same-session Stereo canary and one 12-frame sensor-mask canary, estimates
the full-batch budget, and closes each task with a new immutable receipt.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/causal_training_R7_2/AUTOMATION_STATE.json"
BASE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/sensor"
OUTPUT = BASE / "automatic_canaries_R7_1"
STATUS_RECEIPT = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"
SESSION = Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_0910/cleaned/playing_cards/play_cards_0910_053")
H4_CONFIG = BASE / "h4_mask/canary_configs/play_cards_0910_053.json"
TOTAL_FRAMES = 106868
TERMINAL_UPSTREAM = {"TERMINAL", "BLOCKED_RESOURCE", "BLOCKED_PREREQ", "FAILED_RUNTIME_FINAL", "FAILED_QUALITY_C"}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(command: list[str], log: Path, timeout: int) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            handle.write(f"\nAUTOMATION_TIMEOUT_SECONDS={timeout}\n")
            return 124


def update_task(task_id: str, status: str, phase: str, result: Path) -> None:
    for _ in range(12):
        revision = int(load(STATUS_RECEIPT)["governance_revision"])
        command = [
            sys.executable, "-m", "chaoyang.governance.update_task_state",
            "--task-id", task_id, "--status", status, "--phase", phase,
            "--clear-runtime", "--result", str(result.resolve()),
            "--message", "V7.1 sensor canary terminal; canary is not full-batch authority",
            "--expected-revision", str(revision),
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if completed.returncode == 0:
            return
        conflict_text = (completed.stdout + completed.stderr).lower()
        if not any(
            marker in conflict_text
            for marker in ("revision conflict", "cas revision mismatch")
        ):
            raise RuntimeError((completed.stdout + completed.stderr)[-3000:])
        time.sleep(1)
    raise RuntimeError(f"governance CAS retry exhausted: {task_id}")


def terminal_payload(stage: str, canary: Path, gpu_receipt: Path) -> tuple[str, dict[str, Any]]:
    gpu = load(gpu_receipt) if gpu_receipt.is_file() else {}
    result = load(canary) if canary.is_file() else {}
    if result:
        if stage == "h3":
            automatic = result.get("status") == "PASS_DEVELOPMENT_CANARY"
            wall = float(result.get("single_frame_wall_seconds_including_calibration_and_model_load") or 0)
            sample_frames = 1
        else:
            automatic = result.get("automatic_gate_pass") is True
            wall = float(result.get("resource", {}).get("wall_seconds") or 0)
            sample_frames = int(result.get("frame_count") or 12)
        projected = wall / max(sample_frames, 1) * TOTAL_FRAMES / 3600.0
        status = "BLOCKED_RESOURCE" if automatic else "FAILED_QUALITY_C"
        reason = "FULL_BATCH_EXCEEDS_12_GPU_HOUR_BUDGET" if projected > 12 else "FULL_BATCH_REQUIRES_NEW_EXPLICIT_TASK_PACKET"
        if not automatic:
            reason = "DEVELOPMENT_CANARY_FAILED_QUALITY_GATE"
    elif gpu.get("status") == "BLOCKED_RESOURCE":
        status, reason, wall, projected, sample_frames = "BLOCKED_RESOURCE", "GPU_WAIT_BUDGET_EXHAUSTED", 0.0, None, 0
    else:
        status, reason, wall, projected, sample_frames = "FAILED_RUNTIME_FINAL", "CANARY_DID_NOT_PUBLISH_RESULT", 0.0, None, 0
    payload = {
        "schema_version": "sensor-stage-canary-terminal-v71-v1",
        "status": status,
        "stage": stage,
        "artifact_revision": "R7_1",
        "validity": "VALID_FOR_PINNED_REVISION",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reason": reason,
        "canary_automatic_gate_pass": bool(result) and automatic,
        "sample_frames": sample_frames,
        "sample_wall_seconds": wall,
        "cohort_total_frames": TOTAL_FRAMES,
        "conservative_projected_gpu_hours": projected,
        "batch_gpu_budget_hours": 12,
        "canary_result": ref(canary) if canary.is_file() else None,
        "gpu_receipt": ref(gpu_receipt) if gpu_receipt.is_file() else None,
        "claim_limit": "Development canary and bounded batch decision only; no full H3/H4 authority, external metric truth or physical deployment authority.",
    }
    return status, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-state", type=Path, default=UPSTREAM)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=1209600)
    parser.add_argument("--h3-attempt-id", default="attempt_0003")
    parser.add_argument("--h4-attempt-id", default="attempt_0002")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-visual-aux-wait",
        action="store_true",
        help="Start the independent H3/H4 canaries immediately after external H0 validation.",
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    state_path = output / "AUTOMATION_STATE.json"
    state = {
        "schema_version": "post-visual-aux-sensor-canaries-v71-v1",
        "status": (
            "DRY_RUN_READY"
            if args.dry_run
            else "READY_H0_INDEPENDENT_START"
            if args.skip_visual_aux_wait
            else "WAITING_VISUAL_AUX_TERMINAL"
        ),
        "pid": os.getpid(), "upstream_state": str(args.upstream_state.resolve()),
        "claim_limit": "Two bounded development canaries; no sensor batch authority.",
    }
    atomic_json(state_path, state)
    if args.dry_run:
        if not SESSION.is_dir() or not H4_CONFIG.is_file():
            raise RuntimeError("sensor canary inputs missing")
        print(json.dumps(state, ensure_ascii=False))
        return 0
    if not args.skip_visual_aux_wait:
        started = time.monotonic()
        while True:
            upstream = load(args.upstream_state) if args.upstream_state.is_file() else {}
            if upstream.get("status") in TERMINAL_UPSTREAM:
                break
            if time.monotonic() - started > args.max_wait_seconds:
                state.update(status="BLOCKED_RESOURCE", reason="VISUAL_AUX_WAIT_BUDGET_EXHAUSTED")
                atomic_json(state_path, state)
                return 3
            state.update(updated_at=datetime.now().astimezone().isoformat(timespec="seconds"), upstream_status=upstream.get("status", "MISSING"))
            atomic_json(state_path, state)
            time.sleep(max(5, args.poll_seconds))
    else:
        state.update(
            updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            dependency_mode="H0_ONLY_V71_DAG",
            upstream_status="NOT_REQUIRED",
        )
        atomic_json(state_path, state)

    h3_root = BASE / "h3_stereo/attempts" / args.h3_attempt_id
    h3_gpu = output / "H3_GPU_RECEIPT.json"
    if not h3_root.exists() and not h3_gpu.exists():
        state.update(status="RUNNING_H3_CANARY", updated_at=datetime.now().astimezone().isoformat(timespec="seconds"))
        atomic_json(state_path, state)
        run([
            sys.executable, "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
            "--task-id", "sensor_h3_stereo_v1", "--attempt-id", args.h3_attempt_id,
            "--priority", "CANARY", "--wait-seconds", "1800", "--wall-seconds", "3600",
            "--receipt", str(h3_gpu), "--claim-limit", "One-frame same-session optical-Z development canary only.",
            "--", sys.executable, "src/chaoyang/ops/run_sensor_h3_foundationstereo_canary_v71.py",
            "--session-root", str(SESSION), "--output-root", str(h3_root), "--frame", "90",
            "--cohort-total-frames", str(TOTAL_FRAMES),
        ], output / "H3.log", 5500)
    h3_status, h3_terminal = terminal_payload("h3", h3_root / "RESULT.json", h3_gpu)
    h3_terminal_path = h3_root / "TERMINAL_RESULT.json"
    if not h3_terminal_path.exists():
        atomic_json(h3_terminal_path, h3_terminal)
    update_task("sensor_h3_stereo_v1", h3_status, "sensor_h3_canary_terminal", h3_terminal_path)

    h4_root = BASE / "h4_mask/attempts" / args.h4_attempt_id
    h4_gpu = output / "H4_GPU_RECEIPT.json"
    if not h4_root.exists() and not h4_gpu.exists():
        state.update(status="RUNNING_H4_CANARY", updated_at=datetime.now().astimezone().isoformat(timespec="seconds"))
        atomic_json(state_path, state)
        run([
            sys.executable, "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
            "--task-id", "sensor_h4_mask_v1", "--attempt-id", args.h4_attempt_id,
            "--priority", "CANARY", "--wait-seconds", "1800", "--wall-seconds", "7200",
            "--receipt", str(h4_gpu), "--claim-limit", "Twelve-frame glove/Controller/card development canary only.",
            "--", sys.executable, "src/chaoyang/ops/run_chips001_pico_mask_temporal.py",
            "--output-root", str(h4_root), "--lease-holder", "sensor_h4_mask_v1",
            "--task-config", str(H4_CONFIG),
        ], output / "H4.log", 9100)
    h4_status, h4_terminal = terminal_payload("h4", h4_root / "RESULT.json", h4_gpu)
    h4_terminal_path = h4_root / "TERMINAL_RESULT.json"
    if not h4_terminal_path.exists():
        atomic_json(h4_terminal_path, h4_terminal)
    update_task("sensor_h4_mask_v1", h4_status, "sensor_h4_canary_terminal", h4_terminal_path)

    subprocess.run([sys.executable, "src/chaoyang/ops/build_sensor_pipeline_terminal_matrix_v71.py"], cwd=ROOT, check=True)
    final = {
        "schema_version": "post-visual-aux-sensor-canaries-result-v71-v1",
        "status": "TERMINAL",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "h3_terminal": ref(h3_terminal_path), "h4_terminal": ref(h4_terminal_path),
        "claim_limit": state["claim_limit"],
    }
    final_path = output / "RESULT.json"
    atomic_json(final_path, final)
    state.update(status="TERMINAL", result=ref(final_path), updated_at=final["generated_at"])
    atomic_json(state_path, state)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
