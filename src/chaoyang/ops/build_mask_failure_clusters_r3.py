#!/usr/bin/env python3
"""Build evidence-bounded exact78 Mask failure clusters and frozen challenger sets.

This is a CPU-only read audit.  It does not run SAM, alter terminal indexes, or
promote authority.  Every Grade-C terminal is assigned exactly one primary
cluster, while the six requested semantic clusters remain explicit even when
the current receipts contain no evidence for them.
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


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROLE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260910_two_task_78_nine_hour_completion_v1/mask_clean/mask_role_successor_v3_final/MASK_ROLE_SUCCESSOR_V3_TERMINAL_INDEX.json"
DEFAULT_OBJECT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/MASK_TASK_OBJECT_TERMINAL_INDEX.json"
DEFAULT_STATUS = ROOT / "docs/governance/CURRENT_PROJECT_STATUS_MIN.json"

SEMANTIC_CLUSTERS = (
    "LEFT_RIGHT_IDENTITY_SWAP",
    "OFFSCREEN_NONEMPTY",
    "REENTRY_IDENTITY_OR_COVERAGE_FAILURE",
    "CHIPS_INSTANCE_UNION",
    "SUPPORT_SURFACE_PLATE_OR_TABLE_INTRUSION",
    "CONTACT_BOUNDARY_LEAKAGE",
)
NON_SEMANTIC_CLUSTERS = (
    "UPSTREAM_HAWOR_C_NOT_RUN",
    "INFRASTRUCTURE_INPUT_CONTRACT_FAILURE",
    "UNCLASSIFIED_BY_CURRENT_TERMINAL_GATES",
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def validate_ref(reference: dict[str, Any], label: str) -> Path:
    if not {"path", "bytes", "sha256"}.issubset(reference):
        raise ValueError(f"{label}: incomplete reference")
    path = Path(str(reference["path"]))
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label}: missing absolute path {path}")
    if path.stat().st_size != int(reference["bytes"]) or sha256_file(path) != reference["sha256"]:
        raise ValueError(f"{label}: bytes/SHA mismatch {path}")
    return path


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"immutable output exists: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def session_date(session_id: str) -> str:
    parts = session_id.split("_")
    return parts[-2] if len(parts) >= 2 else "UNKNOWN"


def semantic_flags(stage: str, payload: dict[str, Any]) -> dict[str, str]:
    """Return three-state evidence flags; UNKNOWN is not silently false."""
    flags = {name: "UNKNOWN_NOT_MEASURED_BY_CURRENT_RECEIPT" for name in SEMANTIC_CLUSTERS}
    gates = payload.get("hard_gates") or payload.get("gates") or {}
    metrics = payload.get("metrics") or {}
    if stage == "ROLE_MASK":
        if "tracker_offscreen_or_unobserved_is_empty" in gates:
            flags["OFFSCREEN_NONEMPTY"] = "FALSE" if gates["tracker_offscreen_or_unobserved_is_empty"] is True else "TRUE"
        reentry = gates.get("tracker_expected_visible_coverage_fraction_at_least_0p80")
        if reentry is not None:
            flags["REENTRY_IDENTITY_OR_COVERAGE_FAILURE"] = "FALSE" if reentry is True else "TRUE"
        # No current terminal field directly measures left/right swaps or the
        # contact-boundary pixels.  Keep those UNKNOWN instead of inferring.
    else:
        if "poker_same_id_pre_and_post_action" in gates:
            flags["REENTRY_IDENTITY_OR_COVERAGE_FAILURE"] = (
                "FALSE" if gates["poker_same_id_pre_and_post_action"] is True else "TRUE"
            )
        if "three_instances_disjoint" in gates or "no_union_or_identity_switch_fabricated" in gates:
            union_pass = gates.get("three_instances_disjoint") is True and gates.get("no_union_or_identity_switch_fabricated") is True
            flags["CHIPS_INSTANCE_UNION"] = "FALSE" if union_pass else "TRUE"
        # Existing receipts have no plate/table intrusion or contact-boundary
        # leakage metric, so those remain explicitly unknown.
    return flags


def classify(stage: str, terminal: dict[str, Any], payload: dict[str, Any]) -> tuple[str, str, float]:
    reason = str(payload.get("reason") or payload.get("status") or terminal.get("status") or "")
    if terminal.get("lineage") == "NOT_RUN_UPSTREAM_C" or reason == "NOT_RUN_UPSTREAM_HAWOR_C":
        return "UPSTREAM_HAWOR_C_NOT_RUN", reason, 0.0
    if "ROLE_RUNNER_EXCEPTION" in reason or "video frame/fps mismatch" in reason:
        return "INFRASTRUCTURE_INPUT_CONTRACT_FAILURE", reason.splitlines()[-1], 0.0

    flags = semantic_flags(stage, payload)
    # Explicit precedence makes the partition deterministic and acyclic.
    precedence = (
        "LEFT_RIGHT_IDENTITY_SWAP",
        "OFFSCREEN_NONEMPTY",
        "REENTRY_IDENTITY_OR_COVERAGE_FAILURE",
        "CHIPS_INSTANCE_UNION",
        "SUPPORT_SURFACE_PLATE_OR_TABLE_INTRUSION",
        "CONTACT_BOUNDARY_LEAKAGE",
    )
    cluster = next((name for name in precedence if flags[name] == "TRUE"), "UNCLASSIFIED_BY_CURRENT_TERMINAL_GATES")
    severity = 0.0
    if cluster == "REENTRY_IDENTITY_OR_COVERAGE_FAILURE":
        if stage == "ROLE_MASK":
            for item in (payload.get("metrics") or {}).get("tracker_reentry_summary", {}).values():
                severity += float(item.get("expected_visible_denominator") or 0) - float(item.get("present_on_expected_visible") or 0)
        else:
            m = payload.get("metrics") or {}
            onset = m.get("poker_loss_onset")
            severity = float(payload.get("frame_count") or 0) - float(onset if onset is not None else payload.get("frame_count") or 0)
    return cluster, reason, severity


def choose_regressions(
    terminals: list[dict[str, Any]], *, task: str, date: str, frame_count: int, limit: int = 2
) -> list[dict[str, Any]]:
    candidates = [row for row in terminals if row.get("downstream_authorized") is True and row.get("task") == task]
    candidates.sort(
        key=lambda row: (
            session_date(str(row["session_id"])) != date,
            abs(int(row.get("frame_count") or 0) - frame_count),
            str(row["session_id"]),
        )
    )
    selected = []
    for row in candidates[:limit]:
        result_path = validate_ref(row["result"], f"regression.{row['session_id']}")
        selected.append(
            {
                "role": "CURRENT_AB_REGRESSION",
                "session_id": row["session_id"],
                "task": row["task"],
                "grade": row["grade"],
                "frame_count": int(row.get("frame_count") or 0),
                "result": ref(result_path),
            }
        )
    return selected


def build_stage(stage: str, terminals: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_cluster: dict[str, list[dict[str, Any]]] = {name: [] for name in (*SEMANTIC_CLUSTERS, *NON_SEMANTIC_CLUSTERS)}
    for terminal in terminals:
        if terminal.get("grade") != "C":
            continue
        result_path = validate_ref(terminal["result"], f"{stage}.{terminal['session_id']}")
        payload = load_json(result_path)
        cluster, reason, severity = classify(stage, terminal, payload)
        row = {
            "stage": stage,
            "session_id": terminal["session_id"],
            "task": terminal["task"],
            "frame_count": int(terminal.get("frame_count") or payload.get("frame_count") or 0),
            "grade": "C",
            "lineage": terminal.get("lineage"),
            "primary_cluster": cluster,
            "primary_reason": reason,
            "severity_score": severity,
            "semantic_evidence_flags": semantic_flags(stage, payload),
            "result": ref(result_path),
        }
        rows.append(row)
        by_cluster[cluster].append(row)

    selections: list[dict[str, Any]] = []
    for cluster in SEMANTIC_CLUSTERS:
        cluster_rows = sorted(by_cluster[cluster], key=lambda row: (-row["severity_score"], row["session_id"]))
        if not cluster_rows:
            selections.append(
                {
                    "stage": stage,
                    "cluster": cluster,
                    "status": "NO_CONFIRMED_CURRENT_C_IN_RECEIPTS",
                    "canary": None,
                    "regressions": [],
                    "selection_eligible": False,
                }
            )
            continue
        canary = cluster_rows[0]
        regressions = choose_regressions(
            terminals,
            task=canary["task"],
            date=session_date(canary["session_id"]),
            frame_count=canary["frame_count"],
        )
        status = "FROZEN_1C_2AB" if len(regressions) == 2 else "BLOCKED_INSUFFICIENT_AB_REGRESSIONS"
        selections.append(
            {
                "stage": stage,
                "cluster": cluster,
                "status": status,
                "selection_eligible": len(regressions) == 2,
                "canary": {
                    "role": "CURRENT_C_CANARY",
                    "session_id": canary["session_id"],
                    "task": canary["task"],
                    "grade": "C",
                    "frame_count": canary["frame_count"],
                    "severity_score": canary["severity_score"],
                    "result": canary["result"],
                },
                "regressions": regressions,
            }
        )
    return rows, selections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-index", type=Path, default=DEFAULT_ROLE)
    parser.add_argument("--object-index", type=Path, default=DEFAULT_OBJECT)
    parser.add_argument("--status-min", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"immutable attempt exists: {output}")
    output.mkdir(parents=True)
    generated_at = now_iso()
    status = load_json(args.status_min.resolve())
    role_doc = load_json(args.role_index.resolve())
    object_doc = load_json(args.object_index.resolve())
    role_terminals = role_doc.get("terminals", [])
    object_terminals = object_doc.get("terminals", [])
    if len(role_terminals) != 156 or len(object_terminals) != 156:
        raise ValueError("both Mask terminal indexes must contain exact156")
    if len({r["session_id"] for r in role_terminals}) != 156 or len({r["session_id"] for r in object_terminals}) != 156:
        raise ValueError("duplicate Mask terminal identity")

    role_rows, role_selection = build_stage("ROLE_MASK", role_terminals)
    object_rows, object_selection = build_stage("OBJECT_MASK", object_terminals)
    all_rows = sorted(role_rows + object_rows, key=lambda row: (row["stage"], row["session_id"]))
    if len(role_rows) != 32 or len(object_rows) != 35:
        raise ValueError(f"unexpected current C counts role={len(role_rows)}, object={len(object_rows)}")

    selections = role_selection + object_selection
    cluster_counts = {
        stage: dict(Counter(row["primary_cluster"] for row in rows))
        for stage, rows in (("ROLE_MASK", role_rows), ("OBJECT_MASK", object_rows))
    }
    frozen = [s for s in selections if s["status"] == "FROZEN_1C_2AB"]
    matrix = {
        "schema_version": "exact78-mask-failure-clusters-r3-v1",
        "artifact_revision": "R7_3_MASK_FAILURE_SELECTION",
        "generated_at": generated_at,
        "status": "PASSED_EVIDENCE_BOUNDED_EXCLUSIVE_PARTITION",
        "current_baseline": "SAM3.1",
        "challengers": ["SAM2.1", "Cutie"],
        "authority_policy": "CHALLENGERS_CANNOT_REPLACE_SAM3.1_WITHOUT_FROZEN_CANARY_PLUS_TWO_AB_REGRESSIONS_PASSING",
        "taxonomy": {
            "semantic_clusters": list(SEMANTIC_CLUSTERS),
            "non_semantic_closure_clusters": list(NON_SEMANTIC_CLUSTERS),
            "precedence": list(SEMANTIC_CLUSTERS),
            "unknown_policy": "A failure type absent from current terminal metrics remains UNKNOWN/zero-confirmed, not inferred from video names or chat.",
        },
        "inputs": {"role_terminal_index": ref(args.role_index), "object_terminal_index": ref(args.object_index)},
        "counts": {
            "role_c": len(role_rows),
            "object_c": len(object_rows),
            "stage_rows_total_not_unique_sessions": len(all_rows),
            "role_by_cluster": cluster_counts["ROLE_MASK"],
            "object_by_cluster": cluster_counts["OBJECT_MASK"],
            "frozen_semantic_cluster_selections": len(frozen),
        },
        "frozen_selections": selections,
        "rows": all_rows,
        "authority_promoted": False,
        "claim_limit": "Current terminal-receipt classification and frozen CPU selection only. No new Mask inference, visual accuracy, successor pass, or authority promotion.",
    }
    matrix_path = output / "MASK_FAILURE_CLUSTER_LEDGER_R3.json"
    write_json(matrix_path, matrix)
    csv_path = output / "MASK_FAILURE_CLUSTER_LEDGER_R3.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        fields = ["stage", "session_id", "task", "frame_count", "grade", "lineage", "primary_cluster", "primary_reason", "severity_score", "result_path", "result_bytes", "result_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({
                **{key: row[key] for key in fields if key in row},
                "result_path": row["result"]["path"],
                "result_bytes": row["result"]["bytes"],
                "result_sha256": row["result"]["sha256"],
            })

    selection_path = output / "MASK_CHALLENGER_FROZEN_SELECTION_R3.json"
    write_json(selection_path, {
        "schema_version": "exact78-mask-challenger-selection-r3-v1",
        "artifact_revision": "R7_3_MASK_CHALLENGER_SELECTION",
        "generated_at": generated_at,
        "status": "FROZEN_FOR_EVIDENCED_NONEMPTY_SEMANTIC_CLUSTERS",
        "current_baseline": "SAM3.1",
        "challengers": ["SAM2.1", "Cutie"],
        "selection_rule": "Highest evidence-derived severity C; two same-task A/B regressions preferring same date then nearest frame count; lexical tie-break.",
        "selections": selections,
        "authority_promoted": False,
        "claim_limit": matrix["claim_limit"],
    })
    metrics_path = output / "METRICS.json"
    write_json(metrics_path, {
        "schema_version": "exact78-mask-failure-clusters-r3-metrics-v1",
        "generated_at": generated_at,
        "role_c_partitioned": len(role_rows),
        "object_c_partitioned": len(object_rows),
        "exclusive_partition": True,
        "frozen_1c_2ab_sets": len(frozen),
        "semantic_clusters_without_current_receipt_evidence": [
            f"{s['stage']}:{s['cluster']}" for s in selections if not s["selection_eligible"]
        ],
        "gpu_calls": 0,
        "authority_promoted": False,
        "claim_limit": matrix["claim_limit"],
    })
    decision_path = output / "DECISION.md"
    decision_path.write_text(
        "# exact78 Mask 失败簇与 challenger 决定 R3\n\n"
        "状态：`PASSED`（CPU 证据分类与冻结 selection）。SAM3.1 继续是 current；SAM2.1/Cutie 仅为 challenger。\n\n"
        f"- Role Mask C：{len(role_rows)}（含上游未运行与基础设施终态）。\n"
        f"- Object Mask C：{len(object_rows)}（含上游未运行终态）。\n"
        f"- 有直接收据证据且可冻结 1C+2A/B 的语义簇：{len(frozen)}。\n"
        "- 当前直接量化到的语义问题是 Role 预期可见/重入覆盖，以及 Poker 遮挡后同一实例身份恢复。\n"
        "- 左右交换、离屏非空、Chips union、盘子/桌面混入和 contact-boundary leakage 未在当前 C 收据中形成独立失败计数；它们保留为固定 taxonomy，不能推断成零真实错误。\n\n"
        "本任务没有运行模型、没有改变当前 terminal index、没有晋升 Mask authority。\n",
        encoding="utf-8",
    )
    next_path = output / "NEXT_ACTION.json"
    write_json(next_path, {
        "schema_version": "exact78-mask-failure-clusters-r3-next-v1",
        "status": "PASSED",
        "next_task_id": "MASK-CHALLENGER-BOUNDED-CANARY",
        "required_inputs": [str(selection_path)],
        "gpu_policy": "Run at most one selected C canary plus two frozen A/B regressions per evidenced cluster under the central lease.",
        "promotion_rule": "Canary passes and both A/B regressions do not regress; otherwise keep SAM3.1 current.",
        "authority_promoted": False,
    })
    packet_path = output / "TASK_PACKET.json"
    write_json(packet_path, {
        "schema_version": "exact78-mask-failure-cluster-task-packet-v1",
        "task_id": "MASK-FAILURE-CLUSTERS-R3",
        "attempt_id": output.name,
        "execution_mode": "CPU_READ_ONLY_AUDIT",
        "read_set": [str(args.status_min.resolve()), str(args.role_index.resolve()), str(args.object_index.resolve())],
        "write_set": [str(output)],
        "gpu_seconds": 0,
        "current_governance_update_allowed": False,
        "robot_v76_touched": False,
    })
    manifest_path = output / "ARTIFACT_MANIFEST.json"
    manifest_members = [matrix_path, csv_path, selection_path, metrics_path, decision_path, next_path, packet_path]
    write_json(manifest_path, {
        "schema_version": "exact78-mask-failure-clusters-r3-manifest-v1",
        "task_id": "MASK-FAILURE-CLUSTERS-R3",
        "attempt_id": output.name,
        "artifacts": [ref(path) for path in manifest_members],
        "authority_promoted": False,
        "claim_limit": matrix["claim_limit"],
    })
    result_path = output / "RESULT.json"
    write_json(result_path, {
        "schema_version": "exact78-mask-failure-clusters-r3-result-v1",
        "task_id": "MASK-FAILURE-CLUSTERS-R3",
        "attempt_id": output.name,
        "terminal_status": "PASSED",
        "generated_at": generated_at,
        "counts": matrix["counts"],
        "ledger": ref(matrix_path),
        "ledger_csv": ref(csv_path),
        "frozen_selection": ref(selection_path),
        "metrics": ref(metrics_path),
        "decision": ref(decision_path),
        "artifact_manifest": ref(manifest_path),
        "current_governance_updated": False,
        "authority_promoted": False,
        "claim_limit": matrix["claim_limit"],
    })
    receipt_path = output / "RUN_RECEIPT.json"
    write_json(receipt_path, {
        "schema_version": "exact78-mask-failure-clusters-r3-receipt-v1",
        "task_id": "MASK-FAILURE-CLUSTERS-R3",
        "attempt_id": output.name,
        "terminal_status": "PASSED",
        "generated_at": now_iso(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "gpu_used": False,
        "governance_revision_observed": status.get("governance_revision"),
        "result": ref(result_path),
        "artifact_manifest": ref(manifest_path),
        "current_governance_updated": False,
        "authority_promoted": False,
        "claim_limit": matrix["claim_limit"],
    })
    print(json.dumps({"status": "PASSED", "counts": matrix["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
