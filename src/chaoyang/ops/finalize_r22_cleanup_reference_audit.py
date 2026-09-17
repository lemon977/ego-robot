#!/usr/bin/env python3
"""Refine attempt-4 cleanup references without traversing or deleting assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ATTEMPT4 = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h/lanes/cleanup_current_only_r72_dryrun/attempts/attempt_0004_bounded_audit"
TASK_PACKET = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h/task_packets/cleanup_current_only_r72_dryrun/TASK_PACKET.json"
RUN_START = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h/g0/attempts/attempt_0001/RUN_START_SNAPSHOT.json"
V6_FINAL = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_cleanup_v6/FINAL_RESULT.json"
CURRENT_RECEIPT = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"
EXTERNAL_PROTECTED = (Path("/mnt/data/egodata"), Path("/nas"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def relative_relation(candidate: Path, target: Path) -> str | None:
    try:
        target.relative_to(candidate)
        return "TARGET_INSIDE_CANDIDATE"
    except ValueError:
        pass
    try:
        candidate.relative_to(target)
        return "CANDIDATE_INSIDE_BROAD_TARGET"
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    old_candidates = load(ATTEMPT4 / "CLEANUP_CANDIDATES.json")
    old_graph = load(ATTEMPT4 / "CLEANUP_REFERENCE_GRAPH.json")
    targets = [(Path(path), sources) for path, sources in old_graph["source_by_target"].items()]
    rows = []
    blocker_counts: Counter[str] = Counter()
    for index, old in enumerate(old_candidates["rows"]):
        candidate = Path(old["path"])
        resolved = candidate.resolve(strict=False)
        if any(relative_relation(resolved, protected.resolve()) is not None for protected in EXTERNAL_PROTECTED):
            raise RuntimeError("attempt-4 candidate unexpectedly overlaps an external protected root")
        strong: list[dict[str, Any]] = []
        broad: list[dict[str, Any]] = []
        for target, sources in targets:
            relation = relative_relation(resolved, target.resolve(strict=False))
            record = {"target": str(target), "sources": sources}
            if relation == "TARGET_INSIDE_CANDIDATE":
                strong.append(record)
            elif relation == "CANDIDATE_INSIDE_BROAD_TARGET":
                broad.append(record)
        blockers = ["EXACT_RECURSIVE_BYTES_NOT_MEASURED"]
        if strong:
            blockers.append("STRONG_CURRENT_OR_ONE_HOP_DESCENDANT_REFERENCE")
        if broad:
            blockers.append("BROAD_ANCESTOR_NAVIGATION_REFERENCE_REQUIRES_REVIEW")
        if old["git_matches"]:
            blockers.append("GIT_DIRTY_OR_UNTRACKED_CLASSIFICATION_REQUIRED")
        if old["process_references"]:
            blockers.append("ATTEMPT4_ACTIVE_PID_FD_OR_CWD_SNAPSHOT")
        if old["protected_prefix"]:
            blockers.append("HARD_PROTECTED_PREFIX")
        blocker_counts.update(blockers)
        try:
            value = candidate.lstat()
            root_stat = {
                "exists": True, "root_lstat_bytes_only": value.st_size,
                "device": value.st_dev, "inode": value.st_ino,
                "mtime_ns": value.st_mtime_ns,
            }
        except FileNotFoundError:
            root_stat = {"exists": False, "root_lstat_bytes_only": 0}
            blockers.append("CANDIDATE_ROOT_CHANGED_OR_MISSING")
        rows.append({
            "candidate_index": index,
            "path": str(candidate),
            "candidate_class": "RUN_ROOT" if "/tasks/control/runs/" in str(candidate) else "RUNTIME_ROOT",
            "root_stat_non_recursive": root_stat,
            "exact_recursive_bytes": None,
            "strong_descendant_references": strong,
            "broad_ancestor_navigation_references": broad,
            "git_matches_from_attempt4": old["git_matches"],
            "process_references_from_attempt4": old["process_references"],
            "hard_protected_prefix": old["protected_prefix"],
            "decision": "BLOCKED_REFERENCE_PROOF",
            "blockers": blockers,
            "delete_authorized": False,
        })

    strong_count = sum(bool(row["strong_descendant_references"]) for row in rows)
    broad_only = sum(
        not row["strong_descendant_references"] and bool(row["broad_ancestor_navigation_references"])
        for row in rows
    )
    reference_audit = {
        "schema_version": "chaoyang-cleanup-reference-audit-r22-v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "BLOCKED_REFERENCE_PROOF",
        "scope": "REPOSITORY_LOCAL_RUN_AND_RUNTIME_ROOTS_ONLY",
        "candidate_count": len(rows),
        "run_root_count": sum(row["candidate_class"] == "RUN_ROOT" for row in rows),
        "runtime_root_count": sum(row["candidate_class"] == "RUNTIME_ROOT" for row in rows),
        "candidate_roots_present": sum(row["root_stat_non_recursive"]["exists"] for row in rows),
        "strong_current_or_one_hop_reference_count": strong_count,
        "broad_navigation_only_count": broad_only,
        "exact_recursive_bytes_measured_count": 0,
        "exact_recursive_bytes_total": None,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "reference_semantics": {
            "strong": "A current/one-hop target equals or lies below the candidate root.",
            "broad": "The candidate lies below a broad navigation root such as the repository; this is not by itself proof that the candidate is an authority artifact, but still requires explicit review.",
        },
        "attempt4_process_snapshot_is_current": False,
        "external_protected": [
            {"path": "/mnt/data/egodata", "traversed": False, "touched": False},
            {"path": "/nas", "traversed": False, "touched": False},
        ],
        "historical_context": {
            "v6_final_result": ref(V6_FINAL),
            "claim_limit": "V6 deletion totals describe completed historical cleanup, not the current 101 roots and not current reclaimable bytes.",
        },
        "source_inputs": {
            "task_packet": ref(TASK_PACKET),
            "run_start_snapshot": ref(RUN_START),
            "attempt4_candidates": ref(ATTEMPT4 / "CLEANUP_CANDIDATES.json"),
            "attempt4_reference_graph": ref(ATTEMPT4 / "CLEANUP_REFERENCE_GRAPH.json"),
            "attempt4_pre_delete_manifest": ref(ATTEMPT4 / "PRE_DELETE_MANIFEST.json"),
            "current_receipt_identity_at_finalize": ref(CURRENT_RECEIPT),
        },
        "delete_executed": False,
        "delete_authorized": False,
    }
    pre_delete = {
        "schema_version": "chaoyang-pre-delete-manifest-r22-reference-audit-v1",
        "generated_at": reference_audit["generated_at"],
        "status": "BLOCKED_REFERENCE_PROOF",
        "candidates": rows,
        "candidate_count": len(rows),
        "exact_recursive_bytes_total": None,
        "delete_executed": False,
        "delete_authorized": False,
        "explicitly_out_of_scope": [
            "/mnt/data/egodata", "/nas", "docs", "tools", "tests", "contracts",
            "permanent deletion of any old run or runtime root",
        ],
    }
    write_json(output / "REFERENCE_AUDIT.json", reference_audit)
    write_json(output / "PRE_DELETE_MANIFEST.json", pre_delete)
    commands = (
        "# Read-only, resumable byte proof for exactly one candidate.\n"
        "# Repeat the same command until status=BYTE_SCAN_COMPLETE_NOT_DELETE_AUTHORITY,\n"
        "# then review references/capsules before moving to the next candidate index.\n"
        "PYTHONPATH=. python src/chaoyang/ops/scan_cleanup_candidate_shard_r22.py \\\n+  --pre-delete-manifest archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h/lanes/cleanup_current_only_r72_dryrun/attempts/attempt_0005_reference_audit/PRE_DELETE_MANIFEST.json \\\n+  --candidate-index 0 \\\n+  --state archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h/lanes/cleanup_current_only_r72_dryrun/sharded_byte_scan/candidate_000.state.json \\\n+  --max-entries 20000 --max-wall-seconds 120\n"
    )
    # Keep shell continuation lines free of patch-style prefixes.
    commands = commands.replace("\n+", "\n").replace(
        "attempt_0005_reference_audit", output.name
    )
    write_text(output / "SHARDED_SCAN_COMMANDS.sh", commands)
    metrics = {
        "schema_version": "chaoyang-cleanup-reference-audit-r22-metrics-v1",
        "status": "BLOCKED_REFERENCE_PROOF",
        "candidate_roots": len(rows),
        "strong_references": strong_count,
        "broad_navigation_only": broad_only,
        "recursive_bytes_known": 0,
        "deleted_bytes": 0,
        "external_paths_touched": 0,
    }
    write_json(output / "METRICS.json", metrics)
    result = {
        "schema_version": "chaoyang-cleanup-reference-audit-r22-result-v1",
        "task_id": "cleanup_current_only_r72_reference_audit",
        "attempt_id": output.name,
        "status": "BLOCKED_REFERENCE_PROOF",
        "reference_audit": ref(output / "REFERENCE_AUDIT.json"),
        "pre_delete_manifest": ref(output / "PRE_DELETE_MANIFEST.json"),
        "exact_recursive_bytes": None,
        "old_assets_deleted": 0,
        "external_paths_touched": 0,
        "claim_limit": "Read-only reference refinement. No current authority, old run, document, tool, test, contract, /mnt/data/egodata or /nas asset was modified or deleted.",
    }
    write_json(output / "RESULT.json", result)
    write_json(output / "RUN_RECEIPT.json", {
        "schema_version": "chaoyang-cleanup-reference-audit-r22-run-receipt-v1",
        "task_id": result["task_id"], "attempt_id": result["attempt_id"],
        "status": result["status"], "generated_at": reference_audit["generated_at"],
        "host": socket.gethostname(), "read_only_audit": True,
        "delete_executed": False, "current_ledger_updated": False,
    })
    write_json(output / "NEXT_ACTION.json", {
        "schema_version": "chaoyang-cleanup-reference-audit-r22-next-v1",
        "status": "BLOCKED_REFERENCE_PROOF",
        "next": "Run bounded sharded byte scans per candidate; then refresh PID/FD/CWD, Git, current receipt and evidence-capsule proofs before any separate commit-delete task.",
        "command_file": ref(output / "SHARDED_SCAN_COMMANDS.sh"),
        "permanent_delete_authorized": False,
    })
    write_text(output / "DECISION.md", (
        "# R2.2 Cleanup 只读引用审计\n\n"
        f"101 个本地候选全部保持 `BLOCKED_REFERENCE_PROOF`：{strong_count} 个存在强 current/一跳后代引用，"
        f"{broad_only} 个目前只有宽泛祖先导航引用，但全部缺精确递归字节证明。\n\n"
        "本轮没有递归扫描大树、没有永久删除、没有触碰 `/mnt/data/egodata` 或 `/nas`，也没有修改 current ledger。"
        "后续必须按 `SHARDED_SCAN_COMMANDS.sh` 分片补字节证据，并重新核验进程、Git、current receipt 和证据胶囊。\n"
    ))
    artifacts = [ref(output / name) for name in (
        "RESULT.json", "METRICS.json", "RUN_RECEIPT.json", "NEXT_ACTION.json",
        "DECISION.md", "REFERENCE_AUDIT.json", "PRE_DELETE_MANIFEST.json",
        "SHARDED_SCAN_COMMANDS.sh",
    )]
    write_json(output / "ARTIFACT_MANIFEST.json", {
        "schema_version": "chaoyang-cleanup-reference-audit-r22-artifact-manifest-v1",
        "task_id": result["task_id"], "status": result["status"],
        "artifacts": artifacts, "delete_executed": False,
    })
    print(json.dumps({
        "status": result["status"], "candidate_roots": len(rows),
        "strong_references": strong_count, "broad_navigation_only": broad_only,
        "result": ref(output / "RESULT.json"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
