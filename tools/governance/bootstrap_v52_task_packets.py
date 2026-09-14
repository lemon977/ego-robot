from __future__ import annotations

"""Create the immutable exact78 V5.2 task packets and bounded context cards."""

import json
import os
from pathlib import Path
from typing import Any

from tools.governance.common import atomic_write, canonical_bytes, now_iso, sha256_bytes
from tools.governance.v52_contracts import validate_task_packet


ROOT = Path(__file__).resolve().parents[2]
PACKET_ROOT = ROOT / "tasks/control/runs/20260913_exact78_v52/task_packets"
BASE_READS = [
    "docs/governance/CURRENT_STATUS_RECEIPT.json",
    "docs/governance/CURRENT_PROJECT_STATUS_MIN.json",
    "docs/governance/PLAN_REVISION.json",
]


TASKS: list[dict[str, Any]] = [
    {
        "task_id": "exact78_v52_stage0",
        "objective": "完成治理恢复后的Task Packet、executor fencing、统一run signature与故障注入验收。",
        "reads": BASE_READS + ["AGENTS.md", "tools/governance/v52_contracts.py"],
        "writes": ["docs/governance/", "tasks/control/runs/20260913_exact78_v52/"],
        "prerequisites": ["governance_recovery_v5"],
        "gates": ["receipt_fresh", "no_ghost_pid", "fault_injection_pass"],
        "budgets": {"cpu_seconds": 3600, "gpu_seconds": 0, "wall_seconds": 7200},
        "attempt_max": 3,
        "stop": "任一revision/SHA冲突、重复executor被接受或不同SHA final可覆盖。",
        "output": ["STAGE0_RESULT.json", "FAULT_INJECTION_RESULT.json"],
        "claim": "治理合同与测试权限；不产生Clean、Robot、接触或训练authority。",
    },
    {
        "task_id": "exact78_v52_lane_a_clean",
        "objective": "完成冻结Wave0 58条Clean唯一终态和clean_join_ready矩阵。",
        "reads": BASE_READS + [
            "tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json",
            "tools/preflight_exact78_clean_wave_v3.py",
            "tools/run_exact78_clean_wave_guardian_v3.py",
        ],
        "writes": ["tasks/control/runs/20260913_exact78_v52/lane_a_clean/"],
        "prerequisites": ["exact78_v52_stage0"],
        "gates": ["wave0_sha_exact", "preflight_ready", "gpu_gate_3x10s", "immutable_final"],
        "budgets": {"cpu_seconds": 86400, "gpu_seconds": 32400, "wall_seconds": 129600},
        "attempt_max": 3,
        "stop": "GPU等待超过30分钟、准备签名冲突、对象保护或媒体硬门失败。",
        "output": ["EXACT78_WAVE0_CLEAN_TERMINAL_MATRIX.json", "RESULT.json"],
        "claim": "仅Clean视觉authority；合成像素不是物理背景真值。",
    },
    {
        "task_id": "exact78_v52_lane_b_upstream_c",
        "objective": "按四簇、两轮successor预算有界修复上游C并只追加Wave1/2 delta。",
        "reads": BASE_READS + [
            "tasks/control/runs/20260913_exact78_v3_lane_c_successors_v1/REMEDIATION_SELECTION.json",
            "tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE1_DELTA.json",
        ],
        "writes": ["tasks/control/runs/20260913_exact78_v52/lane_b_upstream_c/"],
        "prerequisites": ["exact78_v52_stage0"],
        "gates": ["one_failed_canary", "two_frozen_regressions", "same_device_calibration_only"],
        "budgets": {"cpu_seconds": 57600, "gpu_seconds": 28800, "wall_seconds": 57600},
        "attempt_max": 2,
        "stop": "每簇4小时工程墙钟或2 GPU小时或两轮successor耗尽。",
        "output": ["FAILURE_CLUSTER_TERMINALS.json", "WAVE_DELTA_RECEIPT.json"],
        "claim": "只修复已冻结失败簇；禁止改变Wave0或跨session借标定。",
    },
    {
        "task_id": "exact78_v52_lane_c_contact_robot",
        "objective": "通过确定性Contact fixtures和goldset后，生成156行Robot唯一终态矩阵。",
        "reads": BASE_READS + [
            "docs/pipeline/MASQUERADE_CONTACT_OCCLUSION_LEARNINGS_ZH.md",
            "tasks/control/runs/20260913_exact78_v3_wave_clean_v1/EXACT78_WAVE0_SELECTION.json",
        ],
        "writes": ["tasks/control/runs/20260913_exact78_v52/lane_c_contact_robot/"],
        "prerequisites": ["exact78_v52_stage0", "exact78_v52_lane_a_clean"],
        "gates": ["contact_fixtures_all_pass", "goldset_thresholds", "4_to_24_to_full_promotion"],
        "budgets": {"cpu_seconds": 86400, "gpu_seconds": 43200, "wall_seconds": 172800},
        "attempt_max": 2,
        "stop": "fixture失败、goldset不达门、单帧30秒预算或两轮方法耗尽。",
        "output": ["CONTACT_OCCLUSION_GOLDSET_V1_RESULT.json", "EXACT78_ROBOT_TERMINAL_MATRIX.json"],
        "claim": "数字视觉Robot轨迹；无CAD/TCP/mount/world-base标定时物理部署BLOCKED_EXTERNAL。",
    },
    {
        "task_id": "exact78_v52_lane_d_visual_aux",
        "objective": "通过固定split eligibility后训练并封存四支Visual Aux checkpoint及曲线和视频。",
        "reads": BASE_READS + [
            "tasks/control/runs/20260908_two_task_e2e_baseline_v1/training_exact78_visual_ab_prepare_v1/FOUR_CHECKPOINT_CURRENT_READINESS.json",
            "HumanEgo/tools/train_embodiment.py",
        ],
        "writes": ["tasks/control/runs/20260913_exact78_v52/lane_d_visual_aux/"],
        "prerequisites": ["exact78_v52_stage0", "exact78_v52_lane_c_contact_robot"],
        "gates": ["eligibility_minimums", "raw_robot_window_identity", "four_real_epoch0"],
        "budgets": {"cpu_seconds": 86400, "gpu_seconds": 172800, "wall_seconds": 259200},
        "attempt_max": 3,
        "stop": "数据门不满足或每支12 GPU小时/180 epoch/早停终态。",
        "output": ["VISUAL_AUX_CHECKPOINT_INDEX.json", "RESULT.json"],
        "claim": "Visual auxiliary prediction only; control_ground_truth=false。",
    },
    {
        "task_id": "exact78_v52_lane_e_baseline_docs",
        "objective": "建立唯一当前基线注册表并将current、history和obsolete文档分层。",
        "reads": BASE_READS + ["README.md", "docs/pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md"],
        "writes": ["docs/", "README.md", "tasks/control/runs/20260913_exact78_v52/lane_e_baseline_docs/"],
        "prerequisites": ["exact78_v52_stage0"],
        "gates": ["one_current_baseline_registry", "no_manual_current_counts", "bounded_navigation"],
        "budgets": {"cpu_seconds": 14400, "gpu_seconds": 0, "wall_seconds": 28800},
        "attempt_max": 3,
        "stop": "任一current链接指向过时authority或基线缺code/weights/schema边界。",
        "output": ["CURRENT_BASELINE_REGISTRY.json", "DOCUMENT_CLASSIFICATION.json"],
        "claim": "文档和基线索引；不提升任何算法质量等级。",
    },
    {
        "task_id": "exact78_v52_lane_f_cleanup",
        "objective": "生成保护快照/引用图，删除立即安全缓存并将其余候选可恢复隔离。",
        "reads": BASE_READS + ["README.md", ".gitignore"],
        "writes": ["archive/", "tasks/control/runs/20260913_exact78_v52/lane_f_cleanup/"],
        "prerequisites": ["exact78_v52_stage0"],
        "gates": ["reference_graph", "zero_fd_cwd", "git_authority_unchanged"],
        "budgets": {"cpu_seconds": 21600, "gpu_seconds": 0, "wall_seconds": 43200},
        "attempt_max": 3,
        "stop": "路径所有权不明、current/repro引用存在、活跃FD/CWD或跨盘非原子隔离。",
        "output": ["CLEANUP_PROTECTION_SNAPSHOT.json", "CLEANUP_FINAL_REPORT_ZH.md"],
        "claim": "仅明确项目缓存可永久删除；quarantine必须等7天并人工commit-delete。",
    },
]


def write_once(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"immutable task-packet artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    created_at = "2026-09-13T23:00:00+08:00"
    index = []
    for definition in TASKS:
        packet = {
            "schema_version": "exact78-task-packet-v1",
            "plan_revision": "exact78-v5.2",
            "task_id": definition["task_id"],
            "objective": definition["objective"],
            "read_set": definition["reads"],
            "write_set": definition["writes"],
            "prerequisites": definition["prerequisites"],
            "gates": definition["gates"],
            "budgets": definition["budgets"],
            "attempt_max": definition["attempt_max"],
            "stop_condition": definition["stop"],
            "output_contract": definition["output"],
            "claim_limit": definition["claim"],
            "executor_epoch": 1,
            "fencing": {
                "pid_startticks_required": True,
                "immutable_final": True,
                "partial_attempt_is_never_successor_input": True,
            },
            "initial_search_result_limit": 20,
            "initial_log_line_limit": 80,
            "created_at": created_at,
        }
        errors = validate_task_packet(packet)
        if errors:
            raise RuntimeError(f"invalid packet {definition['task_id']}: {errors}")
        task_root = PACKET_ROOT / definition["task_id"]
        packet_bytes = json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
        write_once(task_root / "TASK_PACKET.json", packet_bytes)
        card = (
            f"# {definition['task_id']}\n\n"
            f"目标：{definition['objective']}\n\n"
            f"停止条件：{definition['stop']}\n\n"
            f"声明边界：{definition['claim']}\n\n"
            "执行前先校验 current receipt；只读 TASK_PACKET 的 read_set。\n"
        ).encode()
        write_once(task_root / "CONTEXT_CARD.md", card)
        summary = {
            "schema_version": "exact78-task-result-summary-v1",
            "task_id": definition["task_id"], "status": "PENDING",
            "attempts_consumed": 0, "result": None,
            "claim_limit": "Placeholder status only; not a task terminal.",
        }
        write_once(task_root / "RESULT_SUMMARY.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n")
        patch_receipt = {
            "schema_version": "exact78-patch-receipt-v1",
            "task_id": definition["task_id"], "status": "NO_PATCH_YET",
            "changed_paths": [], "code_closure_sha256": None,
        }
        write_once(task_root / "PATCH_RECEIPT.json", json.dumps(patch_receipt, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n")
        index.append({
            "task_id": definition["task_id"],
            "packet_path": str((task_root / "TASK_PACKET.json").relative_to(ROOT)),
            "packet_sha256": sha256_bytes(packet_bytes),
        })
    receipt = {
        "schema_version": "exact78-task-packet-bootstrap-v1",
        "created_at": created_at, "status": "PASS", "plan_revision": "exact78-v5.2",
        "task_packets": index,
    }
    receipt_path = PACKET_ROOT.parent / "TASK_PACKET_INDEX.json"
    payload = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    write_once(receipt_path, payload)
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
