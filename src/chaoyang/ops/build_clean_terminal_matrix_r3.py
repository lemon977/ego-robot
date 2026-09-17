#!/usr/bin/env python3
"""Build an immutable, normalized R3 Wave0 Clean terminal matrix.

This is deliberately a read-only audit of the pinned selection and an existing
terminal ledger.  It never updates governance and never rewrites a Clean final.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SELECTION = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json"
DEFAULT_LEDGER = REPO_ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_exact78_clean_runtime_recovery_v53/EXACT78_WAVE0_CLEAN_COMBINED_TERMINAL_MATRIX_V53.json"
DEFAULT_STATUS_MIN = REPO_ROOT / "docs/governance/CURRENT_PROJECT_STATUS_MIN.json"
TERMINAL_STATUSES = {
    "PASSED",
    "FAILED_QUALITY_C",
    "FAILED_RUNTIME_FINAL",
    "BLOCKED_PREREQ",
    "BLOCKED_RESOURCE",
    "BLOCKED_EXTERNAL",
    "BLOCKED_REFERENCE_PROOF",
    "UNKNOWN_VERIFICATION_REQUIRED",
    "CANCELLED",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def validate_ref(reference: dict[str, Any], *, label: str) -> Path:
    required = {"path", "bytes", "sha256"}
    if not required.issubset(reference):
        raise ValueError(f"{label}: incomplete artifact reference")
    path = Path(reference["path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label}: missing absolute artifact: {path}")
    actual_bytes = path.stat().st_size
    actual_sha = sha256_file(path)
    if actual_bytes != reference["bytes"] or actual_sha != reference["sha256"]:
        raise ValueError(f"{label}: bytes/SHA mismatch: {path}")
    return path


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_status(status: str) -> str:
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"unsupported/non-terminal Clean status: {status}")
    return status


def result_supports_join(result: dict[str, Any]) -> bool:
    hard_gates = result.get("hard_gates")
    hard_pass = isinstance(hard_gates, dict) and hard_gates and all(value == "PASS" for value in hard_gates.values())
    return (
        result.get("grade") in {"A", "B"}
        and result.get("downstream_authorized") is True
        and hard_pass
        and str(result.get("status", "")).startswith("PASS_")
    )


def build_rows(selection: dict[str, Any], ledger: dict[str, Any]) -> list[dict[str, Any]]:
    selected = selection.get("sessions", [])
    terminals = ledger.get("sessions", [])
    if len(selected) != 58 or len(terminals) != 58:
        raise ValueError(f"Wave0 must have exactly 58 rows: selection={len(selected)}, ledger={len(terminals)}")

    selected_by_id = {row["session_id"]: row for row in selected}
    terminal_by_id = {row["session"]: row for row in terminals}
    if len(selected_by_id) != 58 or len(terminal_by_id) != 58:
        raise ValueError("duplicate session identity in selection or terminal ledger")
    if selected_by_id.keys() != terminal_by_id.keys():
        missing = sorted(selected_by_id.keys() - terminal_by_id.keys())
        extra = sorted(terminal_by_id.keys() - selected_by_id.keys())
        raise ValueError(f"selection/ledger identity mismatch: missing={missing}, extra={extra}")

    rows: list[dict[str, Any]] = []
    for selected_row in sorted(selected, key=lambda row: (int(row["position"]), row["session_id"])):
        session_id = selected_row["session_id"]
        terminal = terminal_by_id[session_id]
        if terminal.get("task") != selected_row.get("task") or int(terminal.get("position")) != int(selected_row["position"]):
            raise ValueError(f"{session_id}: task/position identity mismatch")
        status = normalize_status(str(terminal["status"]))
        result_path = validate_ref(terminal["terminal_result"], label=f"{session_id}.terminal_result")
        result = load_json(result_path)
        if result.get("session") not in {None, session_id}:
            raise ValueError(f"{session_id}: terminal RESULT session mismatch")

        join_ready = status == "PASSED" and result_supports_join(result)
        if bool(terminal.get("clean_join_ready")) != join_ready:
            raise ValueError(f"{session_id}: ledger join_ready contradicts RESULT evidence")
        rows.append(
            {
                "position": int(selected_row["position"]),
                "task": selected_row["task"],
                "session_id": session_id,
                "frame_count": int(selected_row["frame_count"]),
                "artifact_revision": "R7_0",
                "validity": "VALID_FOR_PINNED_REVISION",
                "clean_status": status,
                "passed": status == "PASSED",
                "failed_quality_c": status == "FAILED_QUALITY_C",
                "failed_runtime_final": status == "FAILED_RUNTIME_FINAL",
                "blocked": status.startswith("BLOCKED_") or status == "UNKNOWN_VERIFICATION_REQUIRED",
                "clean_join_ready": join_ready,
                "terminal_result": artifact_ref(result_path),
                "result_contract": {
                    "schema_version": result.get("schema_version"),
                    "producer_status": result.get("status"),
                    "grade": result.get("grade"),
                    "downstream_authorized": result.get("downstream_authorized"),
                    "hard_gates_all_pass": result_supports_join(result),
                    "claim_limit": result.get("claim_limit"),
                },
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--source-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--status-min", type=Path, default=DEFAULT_STATUS_MIN)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"immutable attempt path already exists: {output}")
    output.mkdir(parents=True)

    generated_at = now_iso()
    selection = load_json(args.selection.resolve())
    ledger = load_json(args.source_ledger.resolve())
    status_min = load_json(args.status_min.resolve())
    rows = build_rows(selection, ledger)
    counts = Counter(row["clean_status"] for row in rows)
    task_counts = Counter(row["task"] for row in rows)
    join_ready = sum(bool(row["clean_join_ready"]) for row in rows)
    blocked = sum(bool(row["blocked"]) for row in rows)

    task_packet = {
        "schema_version": "clean-terminal-matrix-r3-task-packet-v1",
        "task_id": "CLEAN-MATRIX-R3",
        "attempt_id": output.name,
        "execution_mode": "CPU_READ_ONLY_AUDIT_PLUS_IMMUTABLE_OUTPUT",
        "objective": "Normalize and close the pinned 58-row Wave0 Clean terminal matrix.",
        "read_set": [str(args.status_min.resolve()), str(args.selection.resolve()), str(args.source_ledger.resolve())],
        "write_set": [str(output)],
        "authority_update_allowed": False,
        "robot_read_write_set_touched": False,
        "claim_limit": "Structural/provenance/decode Clean eligibility only; no semantic, contact-preservation, physical-background, Robot, or geometry authority.",
    }
    write_json(output / "TASK_PACKET.json", task_packet)
    write_json(
        output / "FACT_SNAPSHOT.json",
        {
            "schema_version": "clean-terminal-matrix-r3-fact-snapshot-v1",
            "captured_at": generated_at,
            "governance_revision_observed": status_min.get("governance_revision"),
            "governance_generation_id_observed": status_min.get("generation_id"),
            "governance_freshness_observed": status_min.get("freshness"),
            "wave_counts_observed": status_min.get("waves"),
            "selection": artifact_ref(args.selection.resolve()),
            "source_terminal_ledger": artifact_ref(args.source_ledger.resolve()),
            "claim_limit": "Point-in-time input snapshot; it does not update or replace current governance.",
        },
    )

    matrix = {
        "schema_version": "clean-terminal-matrix-r3-v1",
        "task_id": "CLEAN-MATRIX-R3",
        "attempt_id": output.name,
        "generated_at": generated_at,
        "status": "PASS_58_UNIQUE_TERMINALS",
        "artifact_revision": "R7_0",
        "validity": "VALID_FOR_PINNED_REVISION",
        "selection": artifact_ref(args.selection.resolve()),
        "source_terminal_ledger": artifact_ref(args.source_ledger.resolve()),
        "counts": {
            "selected": 58,
            "passed": counts["PASSED"],
            "failed_quality_c": counts["FAILED_QUALITY_C"],
            "failed_runtime_final": counts["FAILED_RUNTIME_FINAL"],
            "blocked": blocked,
            "clean_join_ready": join_ready,
            "chips": task_counts["chips"],
            "poker": task_counts["poker"],
        },
        "sessions": rows,
        "authority_promoted": False,
        "claim_limit": "58-row R7_0 terminal accounting and structural/provenance/decode Clean eligibility only. Synthetic pixels are not physical background truth; contact/object preservation and semantic correctness require separate gates.",
    }
    matrix_path = output / "CLEAN_TERMINAL_MATRIX.json"
    write_json(matrix_path, matrix)

    csv_path = output / "CLEAN_TERMINAL_MATRIX.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "position", "task", "session_id", "frame_count", "artifact_revision", "validity",
            "clean_status", "passed", "failed_quality_c", "failed_runtime_final", "blocked",
            "clean_join_ready", "result_path", "result_bytes", "result_sha256",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            result_ref = row["terminal_result"]
            writer.writerow({
                **{key: row[key] for key in fieldnames if key in row},
                "result_path": result_ref["path"],
                "result_bytes": result_ref["bytes"],
                "result_sha256": result_ref["sha256"],
            })

    metrics = {
        "schema_version": "clean-terminal-matrix-r3-metrics-v1",
        "task_id": "CLEAN-MATRIX-R3",
        "generated_at": generated_at,
        "rows": len(rows),
        "unique_sessions": len({row["session_id"] for row in rows}),
        "counts": matrix["counts"],
        "terminal_result_refs_rehashed": len(rows),
        "terminal_result_ref_mismatches": 0,
        "selection_ledger_identity_match": True,
        "authority_promoted": False,
        "claim_limit": matrix["claim_limit"],
    }
    write_json(output / "METRICS.json", metrics)
    (output / "DECISION.md").write_text(
        "# CLEAN-MATRIX-R3 决定\n\n"
        "状态：`PASSED`。冻结 Wave0 的 58 条会话均有唯一 Clean 终态；本次逐条重新校验了终态 RESULT 的绝对路径、bytes 与 SHA256。\n\n"
        f"- PASSED：{counts['PASSED']}\n"
        f"- FAILED_QUALITY_C：{counts['FAILED_QUALITY_C']}\n"
        f"- FAILED_RUNTIME_FINAL：{counts['FAILED_RUNTIME_FINAL']}\n"
        f"- BLOCKED：{blocked}\n"
        f"- clean_join_ready：{join_ready}\n\n"
        "边界：这里的通过只表示现有 Clean 基线的结构、来源、帧序、解码与既有 Grade-B 门闭合；不证明接触区物体保留正确、语义正确，也不证明合成背景是真实物理背景。未晋升 current authority。\n",
        encoding="utf-8",
    )
    write_json(
        output / "NEXT_ACTION.json",
        {
            "schema_version": "clean-terminal-matrix-r3-next-action-v1",
            "task_id": "CLEAN-MATRIX-R3",
            "status": "PASSED",
            "next_task_id": "MASK-MATRICES-R3",
            "prerequisites": ["single governance aggregator publication", "current governance remains PASS/FRESH"],
            "stop_condition": "Do not promote semantic/contact preservation without separate Clean successor evidence.",
        },
    )

    manifest_inputs = [
        output / "TASK_PACKET.json",
        output / "FACT_SNAPSHOT.json",
        matrix_path,
        csv_path,
        output / "METRICS.json",
        output / "DECISION.md",
        output / "NEXT_ACTION.json",
    ]
    write_json(
        output / "ARTIFACT_MANIFEST.json",
        {
            "schema_version": "clean-terminal-matrix-r3-artifact-manifest-v1",
            "task_id": "CLEAN-MATRIX-R3",
            "attempt_id": output.name,
            "artifacts": [artifact_ref(path) for path in manifest_inputs],
            "authority_promoted": False,
            "claim_limit": matrix["claim_limit"],
        },
    )
    result = {
        "schema_version": "clean-terminal-matrix-r3-result-v1",
        "task_id": "CLEAN-MATRIX-R3",
        "attempt_id": output.name,
        "terminal_status": "PASSED",
        "generated_at": generated_at,
        "counts": matrix["counts"],
        "matrix_json": artifact_ref(matrix_path),
        "matrix_csv": artifact_ref(csv_path),
        "metrics": artifact_ref(output / "METRICS.json"),
        "decision": artifact_ref(output / "DECISION.md"),
        "artifact_manifest": artifact_ref(output / "ARTIFACT_MANIFEST.json"),
        "authority_promoted": False,
        "current_governance_updated": False,
        "claim_limit": matrix["claim_limit"],
    }
    write_json(output / "RESULT.json", result)
    write_json(
        output / "RUN_RECEIPT.json",
        {
            "schema_version": "clean-terminal-matrix-r3-run-receipt-v1",
            "task_id": "CLEAN-MATRIX-R3",
            "attempt_id": output.name,
            "terminal_status": "PASSED",
            "generated_at": now_iso(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "gpu_used": False,
            "execution_mode": "CPU_READ_ONLY_AUDIT_PLUS_IMMUTABLE_OUTPUT",
            "result": artifact_ref(output / "RESULT.json"),
            "artifact_manifest": artifact_ref(output / "ARTIFACT_MANIFEST.json"),
            "current_governance_updated": False,
            "authority_promoted": False,
            "claim_limit": matrix["claim_limit"],
        },
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
