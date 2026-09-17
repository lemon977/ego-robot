#!/usr/bin/env python3
"""Build an evidence-bound exact78 conversion funnel and optimization report."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
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


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def percentage(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--clean-audit", type=Path, required=True)
    parser.add_argument("--visual-aux-preflight", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    matrix_path = args.matrix.resolve(strict=True)
    clean_audit_path = args.clean_audit.resolve(strict=True)
    visual_path = args.visual_aux_preflight.resolve(strict=True)
    matrix = load(matrix_path)
    rows = matrix["rows"]
    if len(rows) != 156 or len({row["session_id"] for row in rows}) != 156:
        raise RuntimeError("exact78 matrix must contain 156 unique sessions")
    clean_audit = load(clean_audit_path)
    visual = load(visual_path)

    first_blocker = Counter()
    for row in rows:
        if row["hawor"]["grade"] not in {"A", "B"}:
            first_blocker["HAWOR_C"] += 1
        elif row["role_mask"]["grade"] not in {"A", "B"}:
            first_blocker["ROLE_MASK_C_AFTER_HAWOR_AB"] += 1
        elif row["object_mask"]["grade"] not in {"A", "B"}:
            first_blocker["OBJECT_MASK_C_AFTER_HAWOR_ROLE_AB"] += 1
        elif not row["metric_ready_wave0"]:
            first_blocker["CALIBRATION_MISSING_AFTER_TRIPLE_AB"] += 1
        else:
            first_blocker["METRIC_READY"] += 1
    expected = {
        "HAWOR_C": 12,
        "ROLE_MASK_C_AFTER_HAWOR_AB": 20,
        "OBJECT_MASK_C_AFTER_HAWOR_ROLE_AB": 23,
        "CALIBRATION_MISSING_AFTER_TRIPLE_AB": 43,
        "METRIC_READY": 58,
    }
    if dict(first_blocker) != expected:
        raise RuntimeError(f"unexpected upstream decomposition: {first_blocker}")

    per_task = {}
    for task in ("chips", "poker"):
        selected = [row for row in rows if row["task"] == task]
        per_task[task] = {
            "raw": len(selected),
            "hawor_ab": sum(row["hawor"]["grade"] in {"A", "B"} for row in selected),
            "role_ab": sum(row["role_mask"]["grade"] in {"A", "B"} for row in selected),
            "object_ab": sum(row["object_mask"]["grade"] in {"A", "B"} for row in selected),
            "triple_ab": sum(row["three_upstream_ab"] for row in selected),
            "metric_ready": sum(row["metric_ready_wave0"] for row in selected),
        }

    clean_counts = clean_audit["counts"]
    if clean_counts["selected"] != 58 or clean_counts["passed"] != 39:
        raise RuntimeError("unexpected predecessor Clean audit counts")
    clean_matrix = load(Path(clean_audit["matrix"]["path"]))
    import_failures = 0
    for row in clean_matrix["sessions"]:
        if row["status"] != "FAILED_RUNTIME_FINAL":
            continue
        terminal = load(Path(row["terminal_result"]["path"]))
        latest_attempt = load(Path(terminal["attempts"][-1]["path"]))
        log_path = Path(latest_attempt["error"].split("log=", 1)[1].split("; command=", 1)[0])
        if "cannot import name 'prepare_exact78_clean_expanded_role_v3'" in log_path.read_text(
            errors="replace"
        ):
            import_failures += 1
    if import_failures != 19:
        raise RuntimeError("Clean runtime-root-cause closure failed")

    robot_states = Counter(row["robot_current_state"] for row in rows)
    robot_failure_combinations = Counter(
        tuple(row["reason_codes"])
        for row in rows
        if row["robot_current_state"] == "FAILED_QUALITY_C"
    )
    robot_terminal_count = (
        robot_states["FAILED_QUALITY_C"]
        + robot_states["POSE_ONLY_VISUAL_ROBOT_REVIEW_READY"]
    )
    robot_candidate_rate_on_terminal = percentage(
        robot_states["POSE_ONLY_VISUAL_ROBOT_REVIEW_READY"], robot_terminal_count
    )
    funnel = [
        {"stage": "Raw", "count": 156, "previous": 156},
        {"stage": "HaWoR A/B", "count": 144, "previous": 156},
        {"stage": "HaWoR + Role Mask A/B", "count": 124, "previous": 144},
        {"stage": "三路共同 A/B", "count": 101, "previous": 124},
        {"stage": "metric geometry ready", "count": 58, "previous": 101},
        {"stage": "Depth B", "count": 58, "previous": 58},
        {"stage": "Object6D B", "count": 58, "previous": 58},
        {"stage": "Clean PASSED（successor前）", "count": 39, "previous": 58},
        {
            "stage": "Robot visual candidates（非authority）",
            "count": robot_states["POSE_ONLY_VISUAL_ROBOT_REVIEW_READY"],
            "previous": 39,
        },
        {"stage": "Robot authority", "count": 0, "previous": 156},
        {"stage": "Visual Aux checkpoints", "count": 0, "previous": 4},
    ]
    for item in funnel:
        item["retention_percent"] = percentage(item["count"], item["previous"])

    result = {
        "schema_version": "exact78-conversion-funnel-v1",
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS_EVIDENCE_BOUND_ANALYSIS",
        "sources": {
            "cross_stage_matrix": ref(matrix_path),
            "clean_audit": ref(clean_audit_path),
            "visual_aux_preflight": ref(visual_path),
        },
        "funnel": funnel,
        "per_task": per_task,
        "exclusive_first_blocker": dict(first_blocker),
        "clean": {
            "passed": clean_counts["passed"],
            "runtime_failed": clean_counts["failed_runtime_final"],
            "quality_failed": clean_counts["failed_quality_c"],
            "cleanup_induced_missing_import": import_failures,
            "interpretation": "19 rows are infrastructure-invalidated, not manifests/legacy_data/model quality failures",
        },
        "robot_draft": {
            "snapshot_created_at": matrix["created_at"],
            "states": dict(robot_states),
            "failure_combinations": {" + ".join(key): value for key, value in robot_failure_combinations.items()},
            "authority": 0,
            "terminal_rows": robot_terminal_count,
            "candidate_rate_on_terminal_percent": robot_candidate_rate_on_terminal,
        },
        "recoverable_capacity": {
            "clean_runtime_successor": {
                "currently_invalidated": 19,
                "maximum_clean_gain_if_all_pass": 19,
                "does_not_require_algorithm_threshold_change": True,
            },
            "visual_metric_decoupling": {
                "triple_mask_ab": 101,
                "metric_ready": 58,
                "calibration_missing_visual_candidates": 43,
                "policy": "Allow Visual Clean and pose-only non-contact use; keep metric contact blocked.",
            },
            "upstream_quality_clusters": {
                "hawor": 12,
                "role_mask_after_hawor": 20,
                "object_mask_after_hawor_role": 23,
                "policy": "Bounded canary plus frozen A/B regression; never loosen gates globally.",
            },
        },
        "visual_aux": {
            "checkpoints": 0,
            "chips_wave0_train_validation_sessions": [
                visual["tasks"]["chips"]["wave0_sessions"]["train"],
                visual["tasks"]["chips"]["wave0_sessions"]["validation"],
            ],
            "poker_wave0_train_validation_sessions": [
                visual["tasks"]["poker"]["wave0_sessions"]["train"],
                visual["tasks"]["poker"]["wave0_sessions"]["validation"],
            ],
            "poker_visual_wave1_required": visual["tasks"]["poker"]["selected_visual_wave1"],
            "required_minimum": {"train_sessions": 16, "validation_sessions": 3, "train_windows": 256, "validation_windows": 48},
        },
        "claim_limit": "Current conversion diagnosis only. Robot candidates are not authority; checkpoints remain zero until real paired visual bundles pass their contract.",
    }

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / "CONVERSION_FUNNEL.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (output / "CONVERSION_FUNNEL.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "count", "previous", "retention_percent"])
        writer.writeheader()
        writer.writerows(funnel)

    lines = [
        "# exact78 转换漏斗、原因与优化顺序（证据生成）",
        "",
        f"> 生成时间：{result['created_at']}。Robot 数量采用矩阵快照 {matrix['created_at']}；实时运行状态只读事实账本。",
        "",
        "## 当前漏斗",
        "",
        "| 阶段 | 数量 | 相对上一阶段留存率 |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {item['stage']} | {item['count']} | {item['retention_percent']:.2f}% |" for item in funnel
    )
    lines.extend(
        [
            "",
            "## 为什么会从 100 条变成很少的 Robot",
            "",
            "按互斥的第一个阻塞点，156 条被精确分成：12 条 HaWoR C、20 条 Role Mask C、23 条 Object Mask C、43 条三路已过但缺同会话标定、58 条 metric-ready。损失是多道硬门串联，不是单一 Robot 模型造成。",
            "",
            "Wave0 Clean 的 19 条失败全部是清理误删当前 import 后，在 CPU prepare 启动前失败；Clean 本身没有质量 C。这 19 条必须由 fresh successor 重跑，不能计作数据质量差。",
            "",
            f"Robot 当前矩阵有 {robot_states['FAILED_QUALITY_C']} 条质量 C、{robot_states['POSE_ONLY_VISUAL_ROBOT_REVIEW_READY']} 条视觉候选、{robot_states['READY_FOR_ROBOT_CURRENT_DRAFT']} 条待运行；已封账 Robot 行中的视觉候选率为 {robot_candidate_rate_on_terminal:.2f}%。这些仍不是 authority，正式 Robot authority 为 0。",
            "",
            "换算到每100条 Raw，当前上游结构大约留下92条 HaWoR、79条 HaWoR+Role、65条三路A/B、37条 metric-ready。旧 Clean 的基础设施误删又把可运行 Clean 临时压到25条；这不是模型转换率。按当前已封账 Robot 行估算，最终肉眼可看候选只约占 Raw 的个位数百分比，因此必须分别修基础设施、上游分流和 Robot 可达/手形，而不是把所有损失归咎于一个模型。",
            "",
            "## 优化顺序",
            "",
            "1. P0：完成 19 条 Clean runtime successor，恢复 metric-ready 58 条的真实 Clean 分母。",
            "2. P0：Robot arm 与 hand 分簇处理。Arm 先做可达工作空间归一化、per-side first-observed anchor 与确定性 base placement；Hand 单独做逐指尺度、关节限位和时序门，不以放宽阈值换通过率。",
            "3. P1：Role Mask 20 条换用 HaWoR 几何实例提示和离屏重捕获，并用旧 A/B regression 防退化。",
            "4. P1：Poker Object Mask 23 条使用 action-conditioned identity 和遮挡 UNKNOWN；禁止用同类最近物体或 union 冒充目标牌。",
            "5. P1：对 43 条 calibration-missing 只接受同设备、同 session 身份可验证标定；同时允许其进入 Visual Clean / pose-only 路径，contact 帧置 invalid，避免视觉训练被公制门不必要截断。",
            "6. P2：遮挡 compositor 必须在冻结 goldset 上同时报告 known accuracy 与 coverage；没有人工/外部真值时只能保留 development evidence。",
            "",
            "## 如何提高转换率而不破坏通用性",
            "",
            "- 先回收假损失：19 条 Clean 是基础设施 import 错误，修复后最多直接恢复19条，不改任何质量阈值。",
            "- 把视觉路径与公制路径分层：43 条缺标定会话可进入 Visual Clean / pose-only 非接触训练，但继续禁止进入 metric contact。这样避免不相关的标定门截断视觉监督。",
            "- 对 Role/Object Mask 按失败簇修复，不按会话手工打补丁；每轮必须用旧 A/B regression 证明通用性。",
            "- Robot 将 arm reachability、hand retarget、contact topology 分开记账；只有失败的子问题进入 successor，防止整条链反复重跑。",
            "- 用分层终态矩阵报告 METRIC_CONTACT、POSE_ONLY_VISUAL、C/BLOCKED；提高可用覆盖率不能靠把 UNKNOWN 或质量 C 改名为 PASS。",
            "",
            "## 四个 checkpoint 的约束路径",
            "",
            "四支均为 Visual Aux：Chips/Poker × Human Raw/Robotized，只监督 future-2D，不使用数字 q 作为真实动作。Chips Wave0 理论上有 27 train/4 validation；Poker 只有 10/1，因此必须确定性补齐冻结的 Visual Wave1 6 train+2 validation，再构建完全同窗的 Raw/Robotized bundle。之后顺序执行四个真实 epoch-0，最后串行训练并输出 checkpoint、loss、ADE/FDE/PCK。",
            "",
            "当前 checkpoint 数为 0；在 paired window 与 training_valid_mask 未闭合前不得启动优化器或伪造训练完成。",
            "",
            "## 结论边界",
            "",
            "以上是当前内部 authority 与数字模型的转换分析。它不证明 Stereo 毫米级真实精度，不把 visual_robot_trajectory_sidecar 当真实控制动作，也不把 Robot 候选视频晋升为部署 authority。",
        ]
    )
    atomic_text(output / "REPORT_ZH.md", "\n".join(lines) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output), "counts": first_blocker}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
