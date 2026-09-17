#!/usr/bin/env python3
"""Weightless 0915 tactile-v2 audit, physical-left preparation and CAD audit."""

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


ROOT = Path(__file__).resolve().parents[3]
TASK_IDS = ("0915_input_prepare_cad_v1", "0915_input_prepare_cad_v2")
STATE = ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json"
INDEX = ROOT / "tasks/current/INDEX.json"
DATASET = Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915")
STEP = ROOT / "assets/robot/hardware_handoff/kaihand_flange_adapter_v1/received_design/KAI_HAND固定件.STEP"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_route(task_id: str) -> None:
    packet_path = ROOT / "tasks/current" / task_id / "TASK_PACKET.json"
    packet = load(packet_path)
    state = load(STATE)
    index = load(INDEX)
    if packet.get("weights") != "ABSENT":
        raise RuntimeError("input preparation/CAD task must have weights=ABSENT")
    if state.get("next_task", {}).get("task_id") != task_id:
        raise RuntimeError("task is not current next_task")
    task = next((row for row in state.get("tasks", [])
                 if row.get("task_id") == task_id), None)
    if task is None or task.get("status") not in {"PENDING", "CLAIMED", "RUNNING"}:
        raise RuntimeError("task state is not executable")
    route = next((row for row in index.get("task_packets", [])
                  if row.get("task_id") == task_id), None)
    if route is None or route.get("execution_allowed") is not True:
        raise RuntimeError("task index does not authorize execution")
    if route.get("packet_sha256") != sha256(packet_path):
        raise RuntimeError("task packet SHA mismatch")


def heartbeat(task_id: str, pid: int) -> None:
    completed = subprocess.run([
        sys.executable, "-m", "chaoyang.governance.heartbeat_task",
        "--task-id", task_id, "--pid", str(pid), "--status", "RUNNING",
        "--phase", "0915_INPUT_AUDIT_PREPARE_AND_CAD",
    ], cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError((completed.stderr or completed.stdout)[-4000:])


def run(command: list[str], log: Path, *, task_id: str,
        heartbeat_seconds: int) -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with log.open("w", encoding="utf-8", buffering=1) as stream:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, text=True,
        )
        while process.poll() is None:
            time.sleep(max(10, heartbeat_seconds))
            heartbeat(task_id, os.getpid())
        return int(process.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", choices=TASK_IDS,
                        default="0915_input_prepare_cad_v1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--heartbeat-seconds", type=int, default=45)
    args = parser.parse_args()
    task_id = args.task_id
    validate_route(task_id)
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh immutable attempt required: {output}")
    output.mkdir(parents=True)
    started = time.time()
    heartbeat(task_id, os.getpid())

    audit = output / "SELF_CONTAINMENT_V2.json"
    prepared = output / "prepared_physical_left"
    cad = output / "kaihand_adapter_step_audit"
    commands = [
        ("audit", [
            sys.executable, "-m",
            "chaoyang.ops.audit_0915_processed_self_containment_v2",
            "--dataset-root", str(DATASET), "--output", str(audit),
        ]),
        ("prepare", [
            sys.executable, "-m",
            "chaoyang.ops.prepare_0915_physical_left_batch_v1",
            "--dataset-root", str(DATASET), "--output-root", str(prepared),
        ]),
        ("cad", [
            sys.executable, "-m", "chaoyang.ops.audit_kaihand_adapter_step_v1",
            "--step", str(STEP), "--output-root", str(cad),
        ]),
    ]
    returncodes: dict[str, int] = {}
    for name, command in commands:
        returncodes[name] = run(
            command, output / f"{name.upper()}.log",
            task_id=task_id,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        atomic_json(output / "PROGRESS.json", {
            "schema_version": "0915-input-prepare-cad-progress-v1",
            "task_id": task_id, "updated_unix": time.time(),
            "returncodes": returncodes,
        })
        if returncodes[name] != 0:
            break
    audit_value = load(audit) if audit.is_file() else None
    prepared_result = prepared / "BATCH_RESULT.json"
    prepared_value = load(prepared_result) if prepared_result.is_file() else None
    cad_result = cad / "RESULT.json"
    cad_value = load(cad_result) if cad_result.is_file() else None
    passed = bool(
        returncodes == {"audit": 0, "prepare": 0, "cad": 0}
        and audit_value and audit_value.get("status") == "PASS"
        and prepared_value and prepared_value.get("status") == "PASS"
        and cad_value and cad_value.get("status") == "PRESENT_CANDIDATE_GEOMETRY"
    )
    result = {
        "schema_version": "0915-input-prepare-cad-result-v1",
        "task_id": task_id,
        "status": "PASSED" if passed else "FAILED_RUNTIME_FINAL",
        "returncodes": returncodes,
        "self_containment": ref(audit) if audit.is_file() else None,
        "prepared_manifest": ref(prepared_result) if prepared_result.is_file() else None,
        "cad_result": ref(cad_result) if cad_result.is_file() else None,
        "counts": {
            "sessions": (prepared_value or {}).get("session_count"),
            "frames": (prepared_value or {}).get("frame_count"),
            "audit_failed": (audit_value or {}).get("failed"),
        },
        "weights": "ABSENT",
        "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
        "started_unix": started, "finished_unix": time.time(),
        "wall_seconds": time.time() - started,
        "claim_limit": (
            "Processed-only audit, physical-left visual preparation and STEP "
            "candidate geometry only; no model inference, measured installation "
            "transform, TCP, camera/base calibration or deployment authority."
        ),
    }
    atomic_json(output / "RESULT.json", result)
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-input-prepare-cad-run-receipt-v1",
        "task_id": task_id, "status": result["status"],
        "result": ref(output / "RESULT.json"),
    })
    print(json.dumps({"status": result["status"],
                      "counts": result["counts"]}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
