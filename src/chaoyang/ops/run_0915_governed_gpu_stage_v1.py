#!/usr/bin/env python3
"""Run one registered 0915 GPU stage through the central TTL lease."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid

from chaoyang.governance.campaign_0915_task_specs_v1 import build_packet


ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json"
INDEX = ROOT / "tasks/current/INDEX.json"
INPUT_ATTEMPT = ROOT / "_run/current/0915_input_prepare_cad_v1/attempts/attempt_0001"
HAWOR_ATTEMPT = ROOT / "_run/current/0915_hawor_full_v1/attempts/attempt_0001"

GPU_TASKS = {
    "0915_hawor_full_v1": {
        "phase": "0915_HAWOR_PHYSICAL_LEFT_FULL",
        "worker_output": "hawor",
        "command": [
            str(ROOT / "src/chaoyang/ops/hawor_python.sh"),
            str(ROOT / "src/chaoyang/ops/run_0915_hawor_persistent_worker_v1.py"),
            "--prepared-manifest",
            str(INPUT_ATTEMPT / "prepared_physical_left/BATCH_RESULT.json"),
            "--output-root", "{worker_output}",
        ],
    },
    "0915_foundationstereo_full_v1": {
        "phase": "0915_FOUNDATIONSTEREO_PHYSICAL_LEFT_FULL",
        "worker_output": "depth",
        "command": [
            str(ROOT / "src/chaoyang/ops/foundationstereo_gpu_python.sh"),
            str(ROOT / "src/chaoyang/ops/run_0915_foundationstereo_persistent_worker_v2.py"),
            "--prepared-manifest",
            str(INPUT_ATTEMPT / "prepared_physical_left/BATCH_RESULT.json"),
            "--output-root", "{worker_output}",
        ],
    },
    "0915_sam31_mask_full_v1": {
        "phase": "0915_SAM31_ONLY_MASK_FULL",
        "worker_output": "sam31",
        "command": [
            sys.executable,
            str(ROOT / "src/chaoyang/ops/run_0915_sam31_persistent_masks_v2.py"),
            "--prepared-manifest",
            str(INPUT_ATTEMPT / "prepared_physical_left/BATCH_RESULT.json"),
            "--hawor-result", str(HAWOR_ATTEMPT / "hawor/BATCH_RESULT.json"),
            "--output-root", "{worker_output}",
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_route(task_id: str) -> dict[str, Any]:
    expected = build_packet(task_id)
    packet_path = ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    packet = load(packet_path)
    if packet != expected:
        raise RuntimeError("materialized task packet differs from frozen campaign spec")
    if len(packet.get("weights", [])) != 1:
        raise RuntimeError("GPU task must bind exactly one logical weight")
    if task_id == "0915_sam31_mask_full_v1":
        encoded = json.dumps(packet, sort_keys=True).lower()
        if "sam2" in encoded or "cutie" in encoded:
            raise RuntimeError("SAM3.1 task packet contains an alternate model")
    state = load(STATE)
    if state.get("next_task", {}).get("task_id") != task_id:
        raise RuntimeError("task is not current next_task")
    task = next((row for row in state.get("tasks", []) if row.get("task_id") == task_id), None)
    if task is None or task.get("status") not in {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}:
        raise RuntimeError("task ledger state is not executable")
    index = load(INDEX)
    route = next((row for row in index.get("task_packets", []) if row.get("task_id") == task_id), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("current index does not authorize task")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("task packet SHA differs from current index")
    return packet


def heartbeat(task_id: str, phase: str, pid: int) -> None:
    completed = subprocess.run([
        sys.executable, "-m", "chaoyang.governance.heartbeat_task",
        "--task-id", task_id, "--pid", str(pid), "--status", "RUNNING",
        "--phase", phase,
    ], cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])


def build_worker_command(task_id: str, worker_output: Path) -> list[str]:
    return [
        str(worker_output) if value == "{worker_output}" else value
        for value in GPU_TASKS[task_id]["command"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, choices=tuple(GPU_TASKS))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=61_440)
    parser.add_argument("--gpu-wait-seconds", type=int, default=21_600)
    parser.add_argument("--wall-seconds", type=int, default=259_200)
    parser.add_argument("--heartbeat-seconds", type=int, default=45)
    args = parser.parse_args()
    packet = validate_route(args.task_id)
    spec = GPU_TASKS[args.task_id]
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh immutable attempt required: {output}")
    output.mkdir(parents=True)
    worker_output = output / spec["worker_output"]
    receipt = output / "GPU_COMMAND_RECEIPT.json"
    worker_command = build_worker_command(args.task_id, worker_output)
    lease_command = [
        sys.executable,
        str(ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"),
        "--task-id", args.task_id,
        "--attempt-id", output.name,
        "--priority", "FULL_BATCH",
        "--gpu-id", str(args.gpu_id),
        "--min-free-mib", str(args.min_free_mib),
        "--wait-seconds", str(args.gpu_wait_seconds),
        "--wall-seconds", str(args.wall_seconds),
        "--receipt", str(receipt),
        "--claim-limit", packet["claim_limit"],
        "--", *worker_command,
    ]
    atomic_json(output / "COMMAND.json", {
        "schema_version": "0915-governed-gpu-command-v1",
        "task_id": args.task_id,
        "weights": packet["weights"],
        "worker_command": worker_command,
        "lease_command": lease_command,
    })
    heartbeat(args.task_id, spec["phase"], os.getpid())
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    started = time.time()
    with (output / "GPU_WRAPPER.log").open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            lease_command, cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, text=True,
        )
        while process.poll() is None:
            time.sleep(max(10, args.heartbeat_seconds))
            heartbeat(args.task_id, spec["phase"], os.getpid())
            state_path = worker_output / "STATE.json"
            state = load(state_path) if state_path.is_file() else {}
            atomic_json(output / "PROGRESS.json", {
                "schema_version": "0915-governed-gpu-progress-v1",
                "task_id": args.task_id,
                "updated_unix": time.time(),
                "wrapper_pid": process.pid,
                "wrapper_returncode": process.poll(),
                "worker": {key: state.get(key) for key in (
                    "state", "completed", "session_count", "passed",
                    "quality_c", "rejected_quality", "blocked_upstream",
                    "failed_runtime", "model_load_count",
                )},
            })
    gpu_receipt = load(receipt) if receipt.is_file() else {}
    batch_path = worker_output / "BATCH_RESULT.json"
    batch = load(batch_path) if batch_path.is_file() else {}
    passed = bool(
        process.returncode == 0
        and gpu_receipt.get("status") == "PASSED"
        and batch.get("status") == "COMPLETED_ALL_TERMINAL"
        and batch.get("session_count") == 220
        and batch.get("failed_runtime") == 0
        and (args.task_id != "0915_sam31_mask_full_v1"
             or batch.get("model_identity") == "SAM3.1_ONLY_USER_LOCKED")
    )
    status = "PASSED" if passed else (
        "BLOCKED_RESOURCE" if gpu_receipt.get("status") == "BLOCKED_RESOURCE"
        else "FAILED_RUNTIME_FINAL"
    )
    result = {
        "schema_version": "0915-governed-gpu-stage-result-v1",
        "task_id": args.task_id,
        "status": status,
        "weights": packet["weights"],
        "mask_model_policy": "SAM3.1_ONLY_USER_LOCKED",
        "gpu_wrapper_status": gpu_receipt.get("status"),
        "gpu_wrapper_returncode": process.returncode,
        "batch_result": ref(batch_path) if batch_path.is_file() else None,
        "counts": {key: batch.get(key) for key in (
            "session_count", "frame_count", "passed", "quality_c",
            "rejected_quality", "blocked_upstream", "failed_runtime",
        )},
        "wall_seconds": time.time() - started,
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-governed-gpu-stage-receipt-v1",
        "task_id": args.task_id,
        "status": status,
        "result": ref(output / "RESULT.json"),
        "gpu_command_receipt": ref(receipt) if receipt.is_file() else None,
    })
    print(json.dumps({"task_id": args.task_id, "status": status, "counts": result["counts"]}, sort_keys=True))
    return 0 if passed else (3 if status == "BLOCKED_RESOURCE" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
