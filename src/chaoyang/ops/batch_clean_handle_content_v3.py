#!/usr/bin/env python3
"""Audit and publish handle datasets with tactile/content admission gates."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any
import uuid


TASK_PREFIX = {"potato_chips": "get_potato_chips", "playing_cards": "play_cards"}
ADMITTED = {
    "SESSION_CONTENT_ADMISSIBLE",
    "SESSION_CONTENT_ADMISSIBLE_WITH_DECLARED_MANUS_ABSENCE",
}


class BatchError(RuntimeError):
    pass


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def process_start_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[19])


def acquire_lock(target: Path) -> tuple[int, dict[str, Any]]:
    path = target / ".batch_content_gate_v3.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.lseek(descriptor, 0, os.SEEK_SET)
        owner = os.read(descriptor, 8192).decode(errors="replace")
        os.close(descriptor)
        raise BatchError(f"target has a live owner: {owner}") from exc
    receipt = {
        "schema_version": "handle-content-gate-lock-v3",
        "pid": os.getpid(), "start_ticks": process_start_ticks(os.getpid()),
        "acquired_unix": time.time(), "target": str(target),
    }
    payload = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    os.ftruncate(descriptor, 0)
    os.write(descriptor, payload)
    os.fsync(descriptor)
    return descriptor, receipt


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise BatchError("--task-source must be TASK=/absolute/path")
    task, text = value.split("=", 1)
    if task not in TASK_PREFIX:
        raise BatchError(f"unsupported task {task!r}")
    path = Path(text).resolve(strict=True)
    return task, path


def enumerate_items(values: list[str], date_tag: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    sources: dict[str, Path] = {}
    for value in values:
        task, path = parse_source(value)
        if task in sources:
            raise BatchError(f"duplicate task source {task}")
        sources[task] = path
    items: list[dict[str, str]] = []
    for task, root in sorted(sources.items()):
        sessions = sorted(path for path in root.iterdir()
                          if path.is_dir() and len(path.name) == 3 and path.name.isdigit())
        if not sessions:
            raise BatchError(f"no NNN session directories below {root}")
        for source in sessions:
            items.append({"task": task, "session_id": source.name,
                          "source": str(source), "date_tag": date_tag})
    return items, {task: str(path) for task, path in sources.items()}


def audit_one(item: dict[str, str], converter: Path) -> dict[str, Any]:
    descriptor, temporary_name = tempfile.mkstemp(prefix="handle-v3-audit-", suffix=".json")
    os.close(descriptor)
    temporary = Path(temporary_name)
    target_placeholder = Path(tempfile.gettempdir()) / (
        f"handle-v3-never-publish-{os.getpid()}-{uuid.uuid4().hex}"
    )
    command = [
        sys.executable, str(converter), "--source", item["source"],
        "--target", str(target_placeholder),
        "--session-id", f"audit_{item['session_id']}",
        "--selection-convention", "legacy_tracker_left",
        "--image-domain-mode", "passthrough_scaled_source_domain",
        "--audit-only", "--report", str(temporary),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise BatchError((completed.stderr or completed.stdout)[-12000:])
        audit = json.loads(temporary.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)
    tactile = audit.get("tactile_quality")
    if tactile is None:
        tactile = (audit.get("content_admission") or {}).get("tactile_quality")
    content_status = (audit.get("content_admission") or {}).get("status")
    if audit.get("status") == "REJECTED_TACTILE_QUALITY":
        classification, reason = "REJECTED", "TACTILE_QUALITY"
    elif not isinstance(tactile, dict) or tactile.get("status") != "PASSED_TACTILE_QUALITY":
        classification, reason = "REJECTED", "TACTILE_QUALITY_RECEIPT_MISSING"
    elif content_status not in ADMITTED:
        classification, reason = "REJECTED", "VISUAL_OR_TRACKING_CONTENT"
    else:
        classification, reason = "CLEANED", "ADMITTED"
    return {
        "task": item["task"], "session_id": item["session_id"],
        "source": item["source"], "classification": classification,
        "reason": reason, "content_status": content_status,
        "tactile_status": tactile.get("status") if isinstance(tactile, dict) else None,
        "audit": audit,
    }


def audit_all(items: list[dict[str, str]], converter: Path,
              sources: dict[str, str], date_tag: str,
              dataset_id: str, report: Path) -> dict[str, Any]:
    started = time.time()
    results: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        try:
            result = audit_one(item, converter)
        except Exception as exc:  # noqa: BLE001
            result = {
                "task": item["task"], "session_id": item["session_id"],
                "source": item["source"], "classification": "REJECTED",
                "reason": "AUDIT_FAILURE", "error": repr(exc),
                "traceback": traceback.format_exc()[-12000:],
            }
        results.append(result)
        if index == 1 or index % 10 == 0 or index == len(items):
            print(json.dumps({"phase": "AUDIT", "completed": index,
                              "total": len(items),
                              "admitted": sum(row["classification"] == "CLEANED"
                                              for row in results),
                              "rejected": sum(row["classification"] == "REJECTED"
                                              for row in results)}, sort_keys=True),
                  flush=True)
    value = {
        "schema_version": "handle-content-gate-audit-v3",
        "state": "AUDITED", "dataset_id": dataset_id, "date_tag": date_tag,
        "sources": sources, "converter": str(converter),
        "policy": {
            "tactile": "tactile-content-quality-policy-v1",
            "content_admitted": sorted(ADMITTED),
            "single_inactive_tactile_side": "WARNING_NOT_REJECTION",
            "missing_manus": "ABSENT_NOT_CAPTURED_NOT_REJECTION",
        },
        "session_count": len(results),
        "admitted": sum(row["classification"] == "CLEANED" for row in results),
        "rejected": sum(row["classification"] == "REJECTED" for row in results),
        "started_unix": started, "finished_unix": time.time(),
        "results": results,
    }
    atomic_json(report, value)
    return value


def target_for(root: Path, item: dict[str, str]) -> Path:
    return (root / "cleaned" / item["task"] /
            f"{TASK_PREFIX[item['task']]}_{item['date_tag']}_{item['session_id']}")


def rejection_for(root: Path, item: dict[str, str]) -> Path:
    return root / "rejected" / item["task"] / item["session_id"]


def validate_existing(target: Path, source: Path) -> dict[str, Any] | None:
    path = target / "CONVERSION_RESULT.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "PASS_FORMAT_COMPATIBLE", "source": str(source),
        "target": str(target), "content_status": result.get("content_status"),
    }
    if (result.get("status") != expected["status"] or
            result.get("source") != expected["source"] or
            result.get("target") != expected["target"] or
            result.get("content_status") not in ADMITTED):
        return None
    for path in (target / "clip_manifest.json",
                 target / "preprocess/pico_humanego_manifest.json",
                 target / f"CameraRecord_{target.name}.mp4"):
        if not path.is_file() or path.stat().st_size == 0:
            return None
    return result


def convert_one(item: dict[str, str], converter: Path,
                root: Path) -> dict[str, Any]:
    source = Path(item["source"])
    target = target_for(root, item)
    rejected = rejection_for(root, item)
    existing = validate_existing(target, source) if target.exists() else None
    if existing is not None:
        return {"task": item["task"], "session_id": item["session_id"],
                "source": str(source), "target": str(target),
                "classification": "CLEANED", "status": "RESUMED",
                "content_status": existing.get("content_status"),
                "frames": existing.get("validation", {}).get("frame_count")}
    rejection_receipt = rejected / "RESULT.json"
    if rejection_receipt.is_file():
        previous = json.loads(rejection_receipt.read_text(encoding="utf-8"))
        if previous.get("source") == str(source):
            return {"task": item["task"], "session_id": item["session_id"],
                    "source": str(source), "target": str(rejected),
                    "classification": "REJECTED", "status": "RESUMED",
                    "reason": previous.get("reason")}
    if target.exists():
        raise BatchError(f"target exists without a valid v3 receipt: {target}")
    if rejected.exists():
        raise BatchError(f"rejection exists without a valid v3 receipt: {rejected}")

    audit = audit_one(item, converter)
    if audit["classification"] == "REJECTED":
        receipt = {
            "schema_version": "handle-content-gate-rejection-v3",
            "status": "REJECTED_BY_CONTENT_GATE",
            "task": item["task"], "session_id": item["session_id"],
            "source": str(source), "reason": audit["reason"],
            "content_status": audit.get("content_status"),
            "tactile_status": audit.get("tactile_status"),
            "audit": audit["audit"], "committed_unix": time.time(),
        }
        atomic_json(rejection_receipt, receipt)
        return {"task": item["task"], "session_id": item["session_id"],
                "source": str(source), "target": str(rejected),
                "classification": "REJECTED", "status": "COMMITTED",
                "reason": audit["reason"],
                "content_status": audit.get("content_status")}

    command = [
        sys.executable, str(converter), "--source", str(source),
        "--target", str(target), "--session-id", target.name,
        "--selection-convention", "legacy_tracker_left",
        "--image-domain-mode", "passthrough_scaled_source_domain",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise BatchError((completed.stderr or completed.stdout)[-12000:])
    result = validate_existing(target, source)
    if result is None:
        raise BatchError("converter returned success without a valid atomic v3 target")
    result["content_gate_audit"] = audit
    result["batch_policy"] = "handle-content-gate-batch-v3"
    atomic_json(target / "CONVERSION_RESULT.json", result)
    return {"task": item["task"], "session_id": item["session_id"],
            "source": str(source), "target": str(target),
            "classification": "CLEANED", "status": "COMMITTED",
            "content_status": result.get("content_status"),
            "frames": result.get("validation", {}).get("frame_count")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "convert"), required=True)
    parser.add_argument("--task-source", action="append", required=True)
    parser.add_argument("--date-tag", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--audit-report", type=Path)
    args = parser.parse_args()
    converter = Path(__file__).resolve().parent / "convert_handle_egodex_v3.py"
    items, sources = enumerate_items(args.task_source, args.date_tag)
    if args.mode == "audit":
        if args.audit_report is None:
            raise BatchError("--audit-report is required in audit mode")
        summary = audit_all(items, converter, sources, args.date_tag,
                            args.dataset_id, args.audit_report.absolute())
        print(json.dumps({key: summary[key] for key in
                          ("state", "session_count", "admitted", "rejected")},
                         sort_keys=True))
        return 0

    if args.target_root is None:
        raise BatchError("--target-root is required in convert mode")
    root = args.target_root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    lock_descriptor, lock_receipt = acquire_lock(root)
    terminal = root / "DATASET_RESULT.json"
    if terminal.is_file():
        prior = json.loads(terminal.read_text(encoding="utf-8"))
        if prior.get("state") == "COMMITTED":
            raise BatchError(f"immutable completed target: {root}")
    (root / "cleaned").mkdir(exist_ok=True)
    (root / "rejected").mkdir(exist_ok=True)
    started = time.time()
    state: dict[str, Any] = {
        "schema_version": "handle-content-gate-batch-state-v3",
        "state": "RUNNING", "dataset_id": args.dataset_id,
        "date_tag": args.date_tag, "sources": sources,
        "session_count": len(items), "completed": 0, "cleaned": 0,
        "rejected": 0, "failed": 0, "converter": str(converter),
        "execution_policy": "SERIAL_CPU_LOW_PRIORITY_EXTERNAL_LAUNCH",
        "lock": lock_receipt, "started_unix": started, "results": [],
    }
    atomic_json(root / "QUALITY_POLICY.json", {
        "schema_version": "handle-content-quality-policy-v3",
        "tactile": "tactile-content-quality-policy-v1",
        "admitted_content_statuses": sorted(ADMITTED),
        "single_inactive_tactile_side": "WARNING_NOT_REJECTION",
        "manus_absence": "DECLARED_ABSENT_NOT_FABRICATED_NOT_REJECTION",
        "source_immutability": "READ_ONLY_SNAPSHOT_VALIDATED",
    })
    atomic_json(root / "STATE.json", state)
    results: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        try:
            result = convert_one(item, converter, root)
        except Exception as exc:  # noqa: BLE001
            result = {
                "task": item["task"], "session_id": item["session_id"],
                "source": item["source"], "classification": "FAILED",
                "status": "FAILED", "error": repr(exc),
                "traceback": traceback.format_exc()[-12000:],
            }
        results.append(result)
        state.update({
            "completed": index,
            "cleaned": sum(row["classification"] == "CLEANED" for row in results),
            "rejected": sum(row["classification"] == "REJECTED" for row in results),
            "failed": sum(row["classification"] == "FAILED" for row in results),
            "updated_unix": time.time(), "results": results,
        })
        atomic_json(root / "STATE.json", state)
        print(json.dumps({"phase": "CONVERT", "completed": index,
                          "total": len(items), "cleaned": state["cleaned"],
                          "rejected": state["rejected"], "failed": state["failed"],
                          "last": f"{item['task']}/{item['session_id']}"},
                         sort_keys=True), flush=True)
    final_state = "COMMITTED" if state["failed"] == 0 else "COMPLETED_WITH_FAILURES"
    result = {**state, "schema_version": "handle-content-gate-dataset-result-v3",
              "state": final_state, "finished_unix": time.time()}
    atomic_json(terminal, result)
    state.update({"state": final_state, "finished_unix": result["finished_unix"]})
    atomic_json(root / "STATE.json", state)
    os.close(lock_descriptor)
    print(json.dumps({"state": final_state, "session_count": len(items),
                      "cleaned": state["cleaned"], "rejected": state["rejected"],
                      "failed": state["failed"], "target": str(root)}, sort_keys=True))
    return 0 if state["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
