#!/usr/bin/env python3
from __future__ import annotations

"""Build the 156-row first-blocker conversion ledger from current terminal indexes."""

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260908_two_task_e2e_baseline_v1/EXACT78_BATCH_MANIFEST.json"
HAWOR = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/HAWOR_TERMINAL_INDEX.json"
ROLE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260910_two_task_78_nine_hour_completion_v1/mask_clean/mask_role_successor_v3_final/MASK_ROLE_SUCCESSOR_V3_TERMINAL_INDEX.json"
OBJECT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/MASK_TASK_OBJECT_TERMINAL_INDEX.json"
JOIN = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260910_two_task_78_nine_hour_completion_v1/mask_clean/mask_role_successor_v3_final/MASK_JOIN_READY_STATE_V3.json"
OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/conversion/revisions/R7_1"
REPORT = ROOT / "docs/research/current/reports/CONVERSION_CAUSE_REPORT_V2_ZH.md"
ARCHIVED_TASKS = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks"


def archived_path(value: str) -> Path:
    prefix = str(ROOT / "tasks") + "/"
    return ARCHIVED_TASKS / value[len(prefix):] if value.startswith(prefix) else Path(value)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def indexed(path: Path) -> dict[str, dict[str, Any]]:
    rows = load(path)["terminals"]
    result = {str(row["session_id"]): row for row in rows}
    if len(rows) != 156 or len(result) != 156:
        raise RuntimeError(f"terminal index is not exact156: {path}")
    return result


def result_payload(row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result")
    if not isinstance(result, dict):
        return {}
    path = archived_path(str(result.get("path", "")))
    return load(path) if path.is_file() else {}


def role_reason(row: dict[str, Any]) -> tuple[str, str, str]:
    payload = result_payload(row)
    reason = str(payload.get("reason") or "")
    if "ROLE_RUNNER_EXCEPTION" in reason or "video frame/fps mismatch" in reason:
        return "INFRASTRUCTURE_FAILURE", "ROLE_INPUT_VIDEO_FRAME_FPS_MISMATCH", reason.splitlines()[-1]
    gates = payload.get("hard_gates", {})
    failed = sorted(name for name, value in gates.items() if value is False)
    detail = "|".join(failed) if failed else str(payload.get("status") or row.get("status"))
    return "ALGORITHM_QUALITY", "ROLE_MASK_QUALITY_GATE", detail


def hawor_reason(row: dict[str, Any]) -> tuple[str, str, str]:
    payload = result_payload(row)
    gates = payload.get("hard_gates") or payload.get("gates") or {}
    failed = sorted(name for name, value in gates.items() if value is False)
    detail = "|".join(failed) or str(payload.get("reason") or payload.get("status") or row.get("status"))
    return "ALGORITHM_QUALITY", "HAWOR_NUMERIC_OR_TEMPORAL_GATE", detail


def object_reason(row: dict[str, Any]) -> tuple[str, str, str]:
    payload = result_payload(row)
    metrics = payload.get("metrics", {})
    if metrics.get("initial_identity_frame") is None:
        return "ALGORITHM_QUALITY", "OBJECT_IDENTITY_NO_INITIAL_ANCHOR", "initial_identity_frame=None"
    return (
        "ALGORITHM_QUALITY",
        "OBJECT_IDENTITY_LOSS_AFTER_ANCHOR",
        f"initial={metrics.get('initial_identity_frame')};loss={metrics.get('poker_loss_onset')};post_observed={metrics.get('poker_post_observed')}",
    )


def build() -> dict[str, Any]:
    raw = load(RAW)
    hawor = indexed(HAWOR)
    role = indexed(ROLE)
    object_mask = indexed(OBJECT)
    join = load(JOIN)
    calibrated = set(join["calibrated_ready_sessions"])
    triple = set(join["grade_ab_sessions"])
    rows: list[dict[str, Any]] = []
    for source in raw["sessions"]:
        session = source["session_id"]
        h, r, o = hawor[session], role[session], object_mask[session]
        h_pass = bool(h.get("downstream_authorized"))
        r_pass = bool(r.get("downstream_authorized"))
        o_pass = bool(o.get("downstream_authorized"))
        if not h_pass:
            blocker = "HAWOR_C"
            blocker_class, reason_code, detail = hawor_reason(h)
            reason = f"{reason_code}:{detail}"
            evidence = h["result"]
            tier = "TIER_B_BLOCKED"
            route = "HAWOR_BOUNDED_SUCCESSOR"
        elif not r_pass:
            blocker = "ROLE_MASK_C"
            blocker_class, reason_code, detail = role_reason(r)
            reason = f"{reason_code}:{detail}"
            evidence = r["result"]
            tier = "TIER_B_BLOCKED"
            route = "ROLE_MASK_FOUR_ROLE_SUCCESSOR"
        elif not o_pass:
            blocker = "OBJECT_MASK_C"
            blocker_class, reason_code, detail = object_reason(o)
            reason = f"{reason_code}:{detail}"
            evidence = o["result"]
            tier = "TIER_B_BLOCKED"
            route = "OBJECT_IDENTITY_CAUSAL_REENTRY_SUCCESSOR"
        elif session not in calibrated:
            blocker = "CALIBRATION_MISSING"
            blocker_class = "MISSING_EVIDENCE"
            reason = "NO_VERIFIED_SAME_DEVICE_SAME_CONTRACT_METRIC_CALIBRATION"
            evidence = JOIN.as_posix()
            tier = "TIER_V_VISUAL"
            route = "VERIFY_HISTORICAL_CALIBRATION_OR_KEEP_VISUAL_TIER"
        else:
            blocker = "METRIC_GEOMETRY_READY"
            blocker_class = "NONE"
            reason = "THREE_INPUT_LANES_AB_AND_METRIC_CALIBRATION"
            evidence = JOIN.as_posix()
            tier = "TIER_M_METRIC"
            route = "DEPTH_OBJECT6D_CLEAN_ROBOT"
        rows.append({
            "position": source["position"], "session_id": session, "task": source["task"].upper(),
            "date": source["date"], "split": source["split"], "frame_count": source["frame_count"],
            "hawor_grade": h["grade"], "role_mask_grade": r["grade"], "object_mask_grade": o["grade"],
            "triple_ab": session in triple, "metric_calibration": session in calibrated,
            "first_blocker": blocker, "blocker_class": blocker_class, "reason": reason,
            "downstream_tier": tier, "successor_route": route, "evidence": evidence,
        })
    if len(rows) != 156 or len({row["session_id"] for row in rows}) != 156:
        raise RuntimeError("ledger lost exact78 identity closure")
    expected = {"HAWOR_C": 12, "ROLE_MASK_C": 20, "OBJECT_MASK_C": 23, "CALIBRATION_MISSING": 43, "METRIC_GEOMETRY_READY": 58}
    first_counts = {key: sum(row["first_blocker"] == key for row in rows) for key in expected}
    if first_counts != expected:
        raise RuntimeError(f"first blocker partition mismatch: {first_counts}")
    class_counts = {key: sum(row["blocker_class"] == key for row in rows) for key in ("ALGORITHM_QUALITY", "INFRASTRUCTURE_FAILURE", "MISSING_EVIDENCE", "NONE")}
    task_counts = {
        task: {key: sum(row["task"] == task and row["first_blocker"] == key for row in rows) for key in expected}
        for task in ("CHIPS", "POKER")
    }
    rates = {
        "hawor_given_raw": {"numerator": 144, "denominator": 156, "rate": 144 / 156},
        "role_given_hawor": {"numerator": 124, "denominator": 144, "rate": 124 / 144},
        "object_given_hawor_role": {"numerator": 101, "denominator": 124, "rate": 101 / 124},
        "metric_given_triple_ab": {"numerator": 58, "denominator": 101, "rate": 58 / 101},
        "metric_given_raw": {"numerator": 58, "denominator": 156, "rate": 58 / 156},
    }
    return {
        "schema_version": "conversion-cause-ledger-v2", "artifact_revision": "R7_1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASSED_EXACT156_FIRST_BLOCKER_PARTITION", "immutable_inputs": [ref(p) for p in (RAW, HAWOR, ROLE, OBJECT, JOIN)],
        "counts": {"sessions": 156, "first_blocker": first_counts, "blocker_class": class_counts, "by_task": task_counts},
        "conditional_conversion_rates": rates, "rows": rows,
        "claim_limit": "First-blocker engineering attribution from current terminal indexes. Grade C is not automatically a visual-model failure; calibration absence is missing evidence, not algorithm failure.",
    }


def report(value: dict[str, Any]) -> str:
    c = value["counts"]
    rates = value["conditional_conversion_rates"]
    lines = [
        "# exact78 转换率首阻塞归因 V2", "",
        "> 本页由 156 行机器账本生成；C 不自动等于模型视觉失败。", "",
        "## 首个互斥阻塞", "", "| 首阻塞 | 数量 |", "|---|---:|",
    ]
    for key, count in c["first_blocker"].items():
        lines.append(f"| `{key}` | {count} |")
    lines += ["", "## 条件转换率", "", "| 条件 | 通过/分母 | 转换率 |", "|---|---:|---:|"]
    for key, item in rates.items():
        lines.append(f"| `{key}` | {item['numerator']}/{item['denominator']} | {item['rate']:.1%} |")
    lines += [
        "", "## 原因性质", "", "| 性质 | 数量 |", "|---|---:|",
    ]
    for key, count in c["blocker_class"].items():
        lines.append(f"| `{key}` | {count} |")
    lines += [
        "", "## 当前直接结论", "",
        "- 58/156 是达到公制几何层的总转换率，不是 Robot 转换率。",
        "- 缺标定的 43 条已经通过三路视觉输入，只能进入 Visual Tier；它们不是算法失败。",
        "- Role Mask 首阻塞 20 条中包含运行时/输入合同问题，必须与真正的 mask 质量失败分开整改。",
        "- Object Mask 的 23 条首阻塞均为 Poker 身份锚点缺失或锚点后快速丢失，适合独立的因果重入 successor。",
        "- 所有明细、证据路径和 successor route 见同目录 JSON/CSV。", "",
    ]
    return "\n".join(lines)


def main() -> None:
    value = build()
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "CONVERSION_CAUSE_LEDGER_V2.json"
    csv_path = OUT / "CONVERSION_CAUSE_LEDGER_V2.csv"
    json_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = ["position", "session_id", "task", "date", "split", "frame_count", "hawor_grade", "role_mask_grade", "object_mask_grade", "triple_ab", "metric_calibration", "first_blocker", "blocker_class", "reason", "downstream_tier", "successor_route"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in value["rows"]:
            writer.writerow({key: row[key] for key in fields})
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report(value), encoding="utf-8")
    result = {
        "schema_version": "conversion-cause-ledger-result-v2", "status": "PASSED",
        "artifact_revision": "R7_1",
        "supersedes_task_result_sha256": "f93637d6485f15a3f2b7d68355aa8472cdc28cfcc7beb7b05e573da883a87490",
        "publication_note": "Published to an immutable revision after detecting that the first invocation reused the legacy task result path; governance is updated only to this revisioned receipt.",
        "outputs": [ref(json_path), ref(csv_path), ref(REPORT)], "counts": value["counts"],
        "claim_limit": value["claim_limit"],
    }
    (OUT / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
