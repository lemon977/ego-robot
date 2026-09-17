#!/usr/bin/env python3
"""Execute the weightless 0915 self-audit and 0916 cleaning task."""

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


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_0916_input_audit_clean_v1"
PACKET = ROOT / "tasks/current" / TASK_ID / "TASK_PACKET.json"
RECEIPT = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"
STATE = ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json"
INDEX = ROOT / "tasks/current/INDEX.json"
DATA_0915 = Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_hands__0915")
SOURCE_0916 = Path("/mnt/data/egodata/datasets/ego/chips_cards_handle_highview_0916")
TARGET_0916 = Path("/mnt/data/egodata/datasets/ego/processed/chips_cards_handle_highview_0916")


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


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_start_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[19])


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_route() -> tuple[dict[str, Any], dict[str, Any]]:
    packet = load(PACKET)
    state = load(STATE)
    index = load(INDEX)
    if state.get("next_task", {}).get("task_id") != TASK_ID:
        raise RuntimeError("task is not the current next_task")
    row = next((item for item in state.get("tasks", [])
                if item.get("task_id") == TASK_ID), None)
    if row is None or row.get("status") not in {"PENDING", "CLAIMED", "RUNNING"}:
        raise RuntimeError("task state is not executable")
    entry = next((item for item in index.get("task_packets", [])
                  if item.get("task_id") == TASK_ID), None)
    if entry is None or entry.get("execution_allowed") is not True:
        raise RuntimeError("current index does not authorize execution")
    if entry.get("packet_sha256") != sha256(PACKET):
        raise RuntimeError("task packet SHA mismatch")
    if packet.get("weights") != "ABSENT":
        raise RuntimeError("weightless task unexpectedly binds a model")
    return packet, row


def heartbeat(pid: int) -> None:
    command = [
        sys.executable, "-m", "chaoyang.governance.heartbeat_task",
        "--task-id", TASK_ID, "--pid", str(pid), "--status", "RUNNING",
        "--phase", "PROCESSED_INPUT_AUDIT_AND_0916_CLEANING",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True,
                               text=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            "governance heartbeat failed: "
            + (completed.stderr or completed.stdout)[-4000:]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--executor-epoch", type=int, required=True)
    parser.add_argument("--fencing-token", required=True)
    parser.add_argument("--heartbeat-seconds", type=int, default=45)
    args = parser.parse_args()
    if args.executor_epoch < 1 or len(args.fencing_token) < 16:
        parser.error("positive executor epoch and fencing token >=16 chars required")
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh immutable attempt required: {output}")
    packet, _task_row = validate_route()
    output.mkdir(parents=True)

    started = time.time()
    signature = {
        "schema_version": "0915-0916-input-run-signature-v1",
        "task_id": TASK_ID,
        "input_manifest": {
            "0915_processed": str(DATA_0915),
            "0916_source": str(SOURCE_0916),
            "0916_target": str(TARGET_0916),
        },
        "code": {
            "guardian": ref(Path(__file__)),
            "audit": ref(ROOT / "src/chaoyang/ops/audit_0915_processed_self_containment_v1.py"),
            "cleaner": ref(ROOT / "src/chaoyang/ops/batch_clean_handle_content_v4.py"),
            "converter": ref(ROOT / "src/chaoyang/ops/convert_handle_egodex_v3.py"),
        },
        "config": {"clean_workers": 2, "processed_only": True},
        "weights": "ABSENT",
        "calibration": "PER_SESSION_CAMERA_PARAMS_READ_ONLY",
        "schemas": ["processed-session-self-containment-v1",
                    "handle-content-gate-dataset-result-v4"],
    }
    signature_sha = hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    atomic_json(output / "RUN_SIGNATURE.json", {**signature,
                "run_signature_sha256": signature_sha})
    atomic_json(output / "CLAIM.json", {
        "schema_version": "0915-0916-input-executor-claim-v1",
        "task_id": TASK_ID,
        "pid": os.getpid(),
        "proc_start_ticks": process_start_ticks(os.getpid()),
        "executor_epoch": args.executor_epoch,
        "fencing_token_sha256": hashlib.sha256(
            args.fencing_token.encode("utf-8")).hexdigest(),
        "run_signature_sha256": signature_sha,
        "task_packet": ref(PACKET),
        "started_unix": started,
    })
    heartbeat(os.getpid())

    self_receipt = output / "SELF_CONTAINMENT.json"
    audit_command = [
        sys.executable, "-m", "chaoyang.cli", "run",
        "audit_0915_processed_self_containment_v1",
        "--dataset-root", str(DATA_0915), "--output", str(self_receipt),
    ]
    clean_command = [
        sys.executable, "-m", "chaoyang.cli", "run",
        "batch_clean_handle_content_v4",
        "--task-source", f"playing_cards={SOURCE_0916 / 'cards_130_0916'}",
        "--task-source", f"potato_chips={SOURCE_0916 / 'chips_110_0916'}",
        "--date-tag", "0916", "--dataset-id", "chips_cards_handle_highview_0916_v4",
        "--target-root", str(TARGET_0916), "--workers", "2",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    processes: dict[str, subprocess.Popen[str]] = {}
    logs: dict[str, Any] = {}
    try:
        for name, command in (("audit", audit_command), ("clean", clean_command)):
            log_path = output / f"{name.upper()}.log"
            log = log_path.open("w", encoding="utf-8", buffering=1)
            logs[name] = log
            processes[name] = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, text=True,
            )
        while any(process.poll() is None for process in processes.values()):
            time.sleep(max(10, args.heartbeat_seconds))
            heartbeat(os.getpid())
            progress = {
                "schema_version": "0915-0916-input-progress-v1",
                "task_id": TASK_ID,
                "updated_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "processes": {
                    name: {"pid": process.pid, "returncode": process.poll()}
                    for name, process in processes.items()
                },
            }
            state_path = TARGET_0916 / "STATE.json"
            if state_path.is_file():
                state = load(state_path)
                progress["cleaning"] = {
                    key: state.get(key) for key in (
                        "state", "session_count", "completed", "cleaned",
                        "rejected", "failed",
                    )
                }
            atomic_json(output / "PROGRESS.json", progress)
    finally:
        for log in logs.values():
            log.close()

    returncodes = {name: process.returncode for name, process in processes.items()}
    dataset_result = TARGET_0916 / "DATASET_RESULT.json"
    audit_value = load(self_receipt) if self_receipt.is_file() else None
    clean_value = load(dataset_result) if dataset_result.is_file() else None
    passed = (
        returncodes == {"audit": 0, "clean": 0}
        and isinstance(audit_value, dict) and audit_value.get("status") == "PASS"
        and isinstance(clean_value, dict) and clean_value.get("state") == "COMMITTED"
        and clean_value.get("session_count") == 240
        and clean_value.get("failed") == 0
    )
    result = {
        "schema_version": "0915-0916-input-audit-clean-result-v1",
        "task_id": TASK_ID,
        "status": "PASSED" if passed else "FAILED_RUNTIME_FINAL",
        "run_signature_sha256": signature_sha,
        "returncodes": returncodes,
        "self_containment": ref(self_receipt) if self_receipt.is_file() else None,
        "dataset_result": ref(dataset_result) if dataset_result.is_file() else None,
        "counts": {
            "0915_sessions": (audit_value or {}).get("session_count"),
            "0915_frames": (audit_value or {}).get("frame_count"),
            "0915_failed": (audit_value or {}).get("failed"),
            "0916_sessions": (clean_value or {}).get("session_count"),
            "0916_cleaned": (clean_value or {}).get("cleaned"),
            "0916_rejected": (clean_value or {}).get("rejected"),
            "0916_failed": (clean_value or {}).get("failed"),
        },
        "pico26": "PRESENT_PRESERVED_NOT_CONSUMED",
        "weights": "ABSENT",
        "started_unix": started,
        "finished_unix": time.time(),
        "wall_seconds": time.time() - started,
        "claim_limit": "Processed self-containment and 0916 cleaning only; no downstream model authority.",
    }
    result_path = output / "RESULT.json"
    atomic_json(result_path, result)
    atomic_json(output / "METRICS.json", {
        "status": result["status"], **result["counts"],
        "wall_seconds": result["wall_seconds"],
    })
    artifacts = [
        output / name for name in (
            "RUN_SIGNATURE.json", "CLAIM.json", "SELF_CONTAINMENT.json",
            "AUDIT.log", "CLEAN.log", "RESULT.json", "METRICS.json",
        ) if (output / name).is_file()
    ]
    atomic_json(output / "ARTIFACT_MANIFEST.json", {
        "schema_version": "0915-0916-input-artifact-manifest-v1",
        "artifacts": [ref(path) for path in artifacts],
        "external_dataset_result": ref(dataset_result) if dataset_result.is_file() else None,
    })
    atomic_json(output / "RUN_RECEIPT.json", {
        "schema_version": "0915-0916-input-run-receipt-v1",
        "task_id": TASK_ID,
        "status": result["status"],
        "result": ref(result_path),
        "artifact_manifest": ref(output / "ARTIFACT_MANIFEST.json"),
        "run_signature_sha256": signature_sha,
        "fencing_token_sha256": hashlib.sha256(
            args.fencing_token.encode("utf-8")).hexdigest(),
    })
    print(json.dumps({
        "status": result["status"], "result": str(result_path),
        "counts": result["counts"],
    }, ensure_ascii=False), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
