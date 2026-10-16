#!/usr/bin/env python3
"""Publish one truthful Exact78 route without running or deleting algorithms."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from typing import Any

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    atomic_json,
    atomic_write,
    canonical_bytes,
    load_json,
    now_iso,
    process_identity,
    publish_bundle,
    validate_artifact_ref,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_baseline_route_consolidation_v2_20260930"
ROOT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_BASELINE_ROUTE_CONSOLIDATION_V2.json"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
ROUTE_ZH = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_ZH.md"
POINTER = REPO_ROOT / "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json"
LOCK = REPO_ROOT / "docs/current/EXACT78_CARDS042_VISUAL_BRANCH_LOCK_V2.json"
BASELINE_ZH = REPO_ROOT / "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_ZH.md"
ENTRY_ZH = REPO_ROOT / "docs/current/AI_WORK_ENTRY_ZH.md"
EXACT78_ZH = REPO_ROOT / "docs/current/EXACT78.md"
README_ZH = REPO_ROOT / "docs/current/README_ZH.md"
PLAN = REPO_ROOT / "docs/plans/EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929/00_EXECUTION.md"
GENERATOR = REPO_ROOT / "src/chaoyang/governance/current_r3_contracts.py"

PARENT_ROBOT = (
    REPO_ROOT
    / "_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json"
)
PARENT_ROBOT_SHA = "8338f2246e476e6afef2995eaa18423a4aafe0625ae829bd5315f809120a78e1"

STAGE_TASKS = {
    "role_mask": "exact78_cards042_role_mask_rebuild_v1_20260929",
    "object_mask": "exact78_cards042_direct_object_rebuild_v2_20260929",
    "clean": "exact78_cards042_fresh_clean_v1_20260929",
    "robot": "exact78_cards042_fresh_robot_v1_20260929",
}

P1_TASKS = {
    "retry6_eight_session": "exact78_general_p1_action_canary_v1_retry6_20260930",
    "retry8_three_session": "exact78_general_p1_action_canary_v1_retry8_20260930",
    "retry11_cards042": "exact78_general_p1_action_canary_v1_retry11_20260930",
    "fast_retry1_cards042": "exact78_general_p1_fast_correction_v1_retry1_20260930",
}

ROBOT100_GUARDS = [
    REPO_ROOT / "docs/current/ROBOT100_BATCH_ZH.md",
    REPO_ROOT / "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json",
    REPO_ROOT / "src/chaoyang/ops/run_robot100_world_first_v10.py",
    REPO_ROOT / "src/chaoyang/ops/run_robot100_constraint_review_v16_retry1.py",
]


def index_copy() -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in load_json(INDEX).items()
        if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}
    }


def write_text(path: Path, value: str) -> None:
    atomic_write(path, value.rstrip().encode("utf-8") + b"\n")


def refs_in(value: Any, location: str = "$") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= set(value):
            found.append((location, value))
        else:
            for key, child in value.items():
                found.extend(refs_in(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(refs_in(child, f"{location}[{index}]"))
    return found


def audit_refs(path: Path) -> dict[str, Any]:
    value = load_json(path)
    refs = refs_in(value)
    rows = []
    for location, reference in refs:
        errors = validate_artifact_ref(reference)
        rows.append(
            {
                "location": location,
                "path": reference["path"],
                "expected_bytes": reference["bytes"],
                "expected_sha256": reference["sha256"],
                "status": "MATCH" if not errors else "DRIFT_OR_MISSING",
                "errors": errors,
            }
        )
    return {
        "source": artifact_ref(path),
        "references": len(rows),
        "matched": sum(row["status"] == "MATCH" for row in rows),
        "failed": sum(row["status"] != "MATCH" for row in rows),
        "failures": [row for row in rows if row["status"] != "MATCH"],
    }


def p1_summary(path: Path) -> dict[str, Any]:
    result = load_json(path)
    sessions = result.get("sessions", [])
    accelerations = [
        row.get("reach_metrics", {}).get("acceleration_rad_per_frame2_max_contiguous")
        for row in sessions
        if row.get("reach_metrics", {}).get("acceleration_rad_per_frame2_max_contiguous") is not None
    ]
    metrics = result.get("metrics", {}).get("arm", {})
    if metrics.get("acceleration_rad_per_frame2_max_contiguous") is not None:
        accelerations.append(metrics["acceleration_rad_per_frame2_max_contiguous"])
    return {
        "result": artifact_ref(path),
        "status": result.get("status"),
        "execution_complete": result.get("execution_complete"),
        "visual_qualified": result.get("visual_qualified"),
        "quality_passed": result.get("quality_passed"),
        "training_eligible": result.get("training_eligible"),
        "four_metrics": result.get("four_metrics"),
        "acceleration_rad_per_frame2_max": max(accelerations) if accelerations else None,
        "failure_clusters": result.get("failure_clusters"),
    }


def update_exact78_documents(audit: dict[str, Any]) -> None:
    pointer = load_json(POINTER)
    if pointer.get("baseline_id") != "EXACT78_CARDS042_VISUAL_BASELINE_V2":
        raise RuntimeError("V2_POINTER_ID_DRIFT")
    if pointer["stages"]["robot"]["result_sha256"] != PARENT_ROBOT_SHA:
        raise RuntimeError("V2_PARENT_SHA_DRIFT")
    pointer["status"] = "CURRENT_VISUAL_REFERENCE_PARENT_REPLAY_CLOSURE_INCOMPLETE"
    pointer["execution_route"] = "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
    pointer["replay_contract"] = {
        "selected_artifacts_sha_intact": True,
        "strict_task_packet_dependency_closure": False,
        "current_one_click_operation": None,
        "general_raw_to_robot_qualified": False,
        "reason": "Four historical stage task packets contain five SHA drifts and four stale dependency paths; retained selected artifacts remain intact.",
    }
    pointer["acceptance"]["strict_replay_closure_passed"] = False
    pointer["acceptance"]["general_raw_to_robot_qualified"] = False
    pointer["claim_limit"] = (
        "Accepted single-session Cards042 visual-reference parent with intact selected artifacts. "
        "It is not a strict replay-closed or general raw-to-Robot executor; Clean remains synthetic, "
        "contact is 2-D visual only, motion is selected frozen state, and no training/control/deployment authority is granted."
    )
    atomic_json(POINTER, pointer)

    lock = load_json(LOCK)
    lock["status"] = "CURRENT_SINGLE_SESSION_VISUAL_REFERENCE_LOCK"
    lock["execution_route"] = "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
    lock["strict_task_packet_dependency_closure"] = False
    lock["general_raw_to_robot_qualified"] = False
    lock["current_one_click_operation"] = None
    atomic_json(LOCK, lock)

    p1 = audit["general_route_evidence"]
    route = {
        "schema_version": "exact78-execution-route-v1",
        "route_id": "EXACT78_EXECUTION_ROUTE_V1",
        "status": "CURRENT_NO_QUALIFIED_GENERAL_EXECUTOR",
        "recorded_at": now_iso(),
        "visual_reference_parent": {
            "baseline_id": "EXACT78_CARDS042_VISUAL_BASELINE_V2",
            "scope": "play_cards_0902_042_only",
            "pointer": artifact_ref(POINTER),
            "robot_result": artifact_ref(PARENT_ROBOT),
            "selected_artifacts_sha_intact": True,
            "user_visual_acceptance": "ACCEPTED_AS_CURRENT_BASELINE_WITH_KNOWN_CLEAN_AND_OCCLUSION_LIMITS",
            "strict_replay_closure": "FAILED",
            "generalization_claim": False,
        },
        "general_raw_to_robot": {
            "target": "arbitrary Exact78 raw session to first-person Robot plus independent whole-Robot diagnostic",
            "status": "P1_NOT_QUALIFIED",
            "current_runnable_operation": None,
            "why_null": [
                "No P1 candidate simultaneously passed execution, numeric action gates, visual review and quality.",
                "Historical operations are evidence producers, not a selected production entrypoint.",
                "V2 uses selected frozen motion and therefore cannot prove fresh Raw-to-Robot replay."
            ],
            "p0_source_inventory": p1["p0_source_inventory"],
            "p1_candidates": p1["p1_candidates"],
        },
        "routing_rule": {
            "read_first": [
                "AGENTS.md",
                "docs/current/EXACT78_EXECUTION_ROUTE_V1.json",
                "docs/plans/EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929/00_EXECUTION.md"
            ],
            "required_before_execution": [
                "validate-governance PASS",
                "new immutable task id and attempt",
                "execution_allowed=true task packet",
                "SHA-pinned inputs and runtime closure",
                "unchanged numeric thresholds and models"
            ],
            "forbidden_selection_methods": [
                "latest filename",
                "highest retry number",
                "playable MP4",
                "historical command",
                "Cards042 fitted template"
            ],
            "next_finite_task": "P1 successor: freeze one corrected generic runner, test Cards042 plus Chips103 plus at least one native-25fps session, and publish the four Section-1 metrics before expansion.",
        },
        "quality_boundary": {
            "execution_complete": "separate field",
            "visual_qualified": False,
            "quality_passed": False,
            "training_eligible": False,
            "physical_occlusion_validated": False,
            "control_or_deployment_authorized": False,
        },
        "robot100_0916": "INDEPENDENT_BRANCH_NOT_CONSUMED_OR_MODIFIED",
        "deletion_authorized": False,
        "claim_limit": "Truthful routing only. This document selects a visual parent and explicitly declines to select a general executor."
    }
    atomic_json(ROUTE, route)

    write_text(
        ROUTE_ZH,
        f"""# Exact78 当前唯一执行路由

## 结论

当前有一个**视觉父版本**，但还没有一个**质量通过的任意会话 Raw→Robot 一键入口**。

- 视觉父版本：`EXACT78_CARDS042_VISUAL_BASELINE_V2`
- 父 Robot RESULT SHA：`{PARENT_ROBOT_SHA}`
- 通用入口状态：`P1_NOT_QUALIFIED`
- 当前可运行通用 operation：`null`

这里的 `null` 是主动停止门。不得从 retry 编号、文件修改时间、可播放视频或旧命令自行挑选“最新版”。

## 为什么 V2 仍然保留

V2 的 171 帧视觉结果和所选视频 SHA 完整，用户已确认它可以作为当前基线；它继续是 Clean 和遮挡优化的唯一视觉父结果。但它使用冻结迁移的 canonical motion，不是 fresh HaWoR 的通用重跑证据。四个历史阶段 Task Packet 合计有 **{audit['stage_packet_closure']['failed']}** 个引用失配，因此不能再称为严格依赖闭包通过的“一键可重跑链”。

## 为什么通用入口还不能发布

- P0 的 156 条输入都可作为源输入解释；其中 113 条是 30 FPS、43 条是原生 25 FPS。原 P0 的 43 条 invalid 是固定 30 FPS 分类错误，不是源视频损坏。
- retry6 完成 8/8 个终态，但只有 6 个动作完成，跨会话 reach 为 67.51%。
- retry8 的 3 条 reach 为 99.18%，但颜色、法兰、手心方向和时间跳变的人工视觉检查不通过。
- retry11 修正了安装方向、橙色法兰和手心方向，reach 为 93.90%，但最大连续加速度为 0.10018，高于 0.06 门限，仍有跳变。
- fast retry1 最大连续加速度约 0.06，但 reach 只有 84.07%。

所以这些都只能作为 P1 失败/开发证据，不能成为生产入口。

## 下一有限任务

新建 P1 successor，只冻结一个通用 runner，至少固定验证 Cards042、Chips103 和一条原生 25 FPS 会话；保持现有数值门限和模型不变，逐帧输出动作来源、逐会话 base、第一人称和独立整机诊断，并在扩展前发布第 1 节四项指标。

## 强制边界

“执行完成”“视觉合格”“质量通过”“标签可训练”必须分开。本路由不授权训练、删除、物理遮挡、控制或上机，也不读取或修改 0916 Robot100 支线。
""",
    )

    write_text(
        BASELINE_ZH,
        f"""# Exact78 Cards042 当前视觉父版本 V2

## 定位

`EXACT78_CARDS042_VISUAL_BASELINE_V2` 是用户认可的单会话视觉父结果，不是已经证明可用于任意会话的一键 Raw→Robot 执行器。通用执行状态只看[当前执行路由](EXACT78_EXECUTION_ROUTE_ZH.md)。

- [171 帧四栏审阅视频](../../_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/cards042_fresh_clean_robot_review_4panel.mp4)，SHA-256 `afacfd4c73d6a234d49bc4134c76be17297c78b201ac806334ae98435ca5141b`
- [第一人称 Clean＋Robot](../../_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/cards042_fresh_clean_robot.mp4)，SHA-256 `a55536ea516549559163c56b35674dab64cc6710820f17ca9e7da66102f2024d`
- [独立整机运动](../../_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/cards042_selected_robot_external.mp4)，SHA-256 `469515341dffb3d508938dfc9d2698f952733f3d44382fc9e2c76f874eee672f`
- Robot RESULT SHA-256：`{PARENT_ROBOT_SHA}`

## 固定内容

| 阶段 | V2 事实 | 当前边界 |
|---|---|---|
| Raw | `play_cards_0902_042`，171 帧 | 单会话 |
| Role Mask | rebuild v1，Grade B | 历史包有合同 SHA 漂移 |
| Object Mask | direct rebuild v2，Grade B | 历史包有脚本 SHA 漂移 |
| Clean | fresh clean v1，固定 ProPainter | 可见残留，单项质量未通过 |
| Robot motion | 用户选择的冻结 canonical HAWOR/ARM/HAND | 不是 fresh HaWoR |
| Robot raster/compositor | 当前 z-buffer 与二维所有权合成 | 不是场景深度或物理接触 |

所选成片和依赖副本仍完整；但四个历史阶段 Task Packet 总计出现 {audit['stage_packet_closure']['failed']} 个失配引用，因此严格重放闭包未通过。后续任务必须以本结果为视觉比较父版本，并重新建立自己的 SHA 完整运行闭包，不能直接复活旧 task。

## 验收边界

| 项目 | 结论 |
|---|---|
| 完整渲染 | 通过，171 帧 |
| 用户视觉父版本 | 通过，可继续调整 |
| strict task replay | 未通过 |
| Clean 单独质量 | 未通过 |
| 物理遮挡/接触 | 未验证 |
| 任意会话 Raw→Robot | 未通过 |
| 标签训练资格 | 未通过 |
| 控制与上机 | 未授权 |

V2 不消费场景 Depth、Object6D、物理 Contact 或 0916 Controller＋MANUS。任何 0902 合成 Robot 视频都不能称为真实机器人动作数据。
""",
    )

    write_text(
        ENTRY_ZH,
        """# AI 工作入口

先读 `AGENTS.md` 并运行 `PYTHONPATH=src python -m chaoyang.cli validate-governance`。

## 分支路由

- **0902 Exact78／Inpainting：**先读[Exact78 当前执行路由](EXACT78_EXECUTION_ROUTE_ZH.md)和[机器路由](EXACT78_EXECUTION_ROUTE_V1.json)。Cards042 V2 是视觉父结果，不是通用一键执行器。
- **0916 PICO＋MANUS：**只从[0916 当前说明](ROBOT100_BATCH_ZH.md)和[机器指针](ROBOT100_HARDWARE_BASELINE_V16.json)开始；不要把它与 0902 Exact78 混用。

## Exact78 当前事实

- 视觉父版本：`EXACT78_CARDS042_VISUAL_BASELINE_V2`
- Robot RESULT SHA：`8338f2246e476e6afef2995eaa18423a4aafe0625ae829bd5315f809120a78e1`
- 任意会话 Raw→Robot：`P1_NOT_QUALIFIED`
- 当前通用 runnable operation：`null`

必须区分视觉父结果和执行入口。V2 所选视频完整，但旧阶段 Task Packet 的严格依赖闭包不完整；P1 的 retry6、retry8、retry11 和 fast retry1 都没有同时通过数值与人工视觉门。

只有新的 Task Packet 明确 `execution_allowed=true` 时才能执行。不得按目录时间、最高 retry、可播放 MP4 或历史命令选择基线；不得读取 Cards042 拟合模板作为通用会话参数；不得触碰另一支线产物。
""",
    )

    write_text(
        EXACT78_ZH,
        """# Exact78 当前入口

唯一入口是[中文执行路由](EXACT78_EXECUTION_ROUTE_ZH.md)和[机器执行路由](EXACT78_EXECUTION_ROUTE_V1.json)。

Cards042 的[视觉父版本 V2](EXACT78_CARDS042_VISUAL_BASELINE_ZH.md)仍作为画面比较与后续 Clean／遮挡调整的父结果，但不是严格重放闭包通过的通用执行器。当前任意会话 Raw→Robot 状态是 `P1_NOT_QUALIFIED`，机器路由中的 runnable operation 为 `null`。

任何新执行必须新建不可变 task/attempt、锁定输入和代码 SHA、保持既有模型与数值门限，并先在 Cards042、Chips103 和原生 25 FPS 会话上通过四项指标。旧计划、V3、虚拟相机、失败 retry、历史 partial 和 0916 Robot100 均不能自动成为 Exact78 successor。
""",
    )

    write_text(
        README_ZH,
        """# Chaoyang 当前执行入口

## 0902 Exact78

- [当前执行路由](EXACT78_EXECUTION_ROUTE_ZH.md)
- [机器执行路由](EXACT78_EXECUTION_ROUTE_V1.json)
- [Cards042 V2 视觉父版本](EXACT78_CARDS042_VISUAL_BASELINE_ZH.md)
- [直接观看 V2 四栏视频](../../_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/cards042_fresh_clean_robot_review_4panel.mp4)

V2 是用户认可的单会话视觉父结果；任意新会话 Raw→Robot 尚未质量通过，当前没有可选的通用 operation。不要从 retry、时间戳或可播放视频推断入口。

## 0916 PICO＋MANUS

见[独立支线说明](ROBOT100_BATCH_ZH.md)与[机器指针](ROBOT100_HARDWARE_BASELINE_V16.json)。它不属于 0902 Exact78，不得互相继承输入、动作或质量结论。

“执行完成”“视觉合格”“质量通过”“标签可训练”始终分开；任何运行仍需当前 Task Packet 的 `execution_allowed=true`。
""",
    )


def main() -> int:
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = index_copy()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    if ROOT.exists() or PACKET.exists() or RECEIPT.exists() or ROUTE.exists():
        raise RuntimeError("FRESH_TASK_REQUIRED")
    if artifact_ref(PARENT_ROBOT)["sha256"] != PARENT_ROBOT_SHA:
        raise RuntimeError("PARENT_ROBOT_RESULT_SHA_DRIFT")
    for path in (PLAN, GENERATOR, POINTER, LOCK, BASELINE_ZH, ENTRY_ZH, EXACT78_ZH, README_ZH):
        path.resolve(strict=True)

    guards_before = {str(path): artifact_ref(path) for path in ROBOT100_GUARDS if path.is_file()}
    root_relative = ROOT.relative_to(REPO_ROOT)
    ROOT.mkdir(parents=True)
    writer = {
        "pid": os.getpid(),
        "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
        "executor_epoch": 1,
        "fencing_token": secrets.token_hex(24),
        "scope": TASK,
    }
    inputs = {
        "plan": artifact_ref(PLAN),
        "visual_pointer_before": artifact_ref(POINTER),
        "visual_lock_before": artifact_ref(LOCK),
        "parent_robot_result": artifact_ref(PARENT_ROBOT),
        "generator": artifact_ref(GENERATOR),
        "p1_results": {
            name: artifact_ref(REPO_ROOT / "_run/current" / task / "attempts/attempt_0001/RESULT.json")
            for name, task in P1_TASKS.items()
        },
    }
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "decision": "TRUTHFUL_ROUTE_ONLY"})).hexdigest()
    packet = {
        "schema_version": "chaoyang-r22-task-packet-v1",
        "task_id": TASK,
        "objective": "Separate the accepted Cards042 visual parent from the not-yet-qualified general Exact78 executor and publish one truthful route",
        "phase": "EXACT78_BASELINE_ROUTE_CONSOLIDATION",
        "plan_revision": "EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929",
        "execution_revision": "EXACT78_BASELINE_ROUTE_CONSOLIDATION_V2",
        "deadline_at": (datetime.now().astimezone() + timedelta(minutes=30)).isoformat(),
        "read_set": [
            "AGENTS.md",
            "docs/plans/EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929/00_EXECUTION.md",
            "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json",
            "docs/current/EXACT78_CARDS042_VISUAL_BRANCH_LOCK_V2.json",
        ],
        "write_set": [
            str(root_relative),
            str(PACKET.relative_to(REPO_ROOT)),
            str(RECEIPT.relative_to(REPO_ROOT)),
            "docs/current/EXACT78_EXECUTION_ROUTE_V1.json",
            "docs/current/EXACT78_EXECUTION_ROUTE_ZH.md",
            "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json",
            "docs/current/EXACT78_CARDS042_VISUAL_BRANCH_LOCK_V2.json",
            "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_ZH.md",
            "docs/current/AI_WORK_ENTRY_ZH.md",
            "docs/current/EXACT78.md",
            "docs/current/README_ZH.md",
            "docs/governance",
            "tasks/current/INDEX.json",
        ],
        "prerequisites": [
            "GOVERNANCE_PASS",
            "NO_ACTIVE_TASK",
            "PARENT_ROBOT_RESULT_SHA_PINNED",
            "NO_MODEL_OR_THRESHOLD_CHANGE",
            "NO_DELETION",
            "ROBOT100_READ_ONLY",
        ],
        "required_outputs": [
            "stage_packet_closure_audit",
            "P1_candidate_metric_summary",
            "machine_execution_route",
            "Chinese_execution_handoff",
            "explicit_execution_visual_quality_training_boundaries",
        ],
        "budgets": {
            "workers_max": 1,
            "cpu_threads_total": 1,
            "gpu_count": 0,
            "training_steps": 0,
            "wall_seconds_max": 1800,
            "session_attempts_max": 1,
        },
        "stop_conditions": [
            "parent_sha_drift",
            "active_writer_conflict",
            "Robot100_guard_drift",
            "model_or_threshold_change",
            "deletion_attempt",
            "one_attempt_terminal",
        ],
        "claim_limit": "Governance and documentation convergence only; no algorithm run, training, deletion, threshold/model change or Robot100 mutation.",
        "writer": writer,
        "inputs": inputs,
        "runtime_code": [artifact_ref(Path(__file__)), artifact_ref(GENERATOR)],
        "run_signature": signature,
        "commands": ["python -m chaoyang.governance.consolidate_exact78_baseline_route_v2"],
    }
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    atomic_json(ROOT / "WRITER.json", writer)
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append(
        {
            "task_id": TASK,
            "phase": packet["phase"],
            "attempt": 1,
            "status": "RUNNING",
            "session": "Exact78_route",
            "updated_at": now,
            "heartbeat_at": now,
            "pid": writer["pid"],
            "proc_start_ticks": writer["proc_start_ticks"],
            "gpu_id": None,
            "task_packet": artifact_ref(PACKET),
            "run_signature": signature,
            "quality": "PENDING",
            "adoption": "PENDING",
            "review": "MACHINE_AUDIT",
        }
    )
    state["next_task"] = {
        "task_id": TASK,
        "session": "Exact78_route",
        "prerequisites": packet["prerequisites"],
        "expected_resource": "CPU_ONE_THREAD",
        "stop_condition": "one terminal route publication",
    }
    index["task_packets"].append(
        {
            "task_id": TASK,
            "packet_path": str(PACKET.relative_to(REPO_ROOT)),
            "packet_sha256": artifact_ref(PACKET)["sha256"],
            "execution_class": "CURRENT_LEDGER_ROUTABLE",
            "execution_allowed": True,
        }
    )
    index.update(
        packet_revision="EXACT78_BASELINE_ROUTE_CONSOLIDATION_V2_RUNNING",
        plan_revision=packet["plan_revision"],
        execution_revision=packet["execution_revision"],
        status="PASS",
        supersedes_index=artifact_ref(INDEX),
    )
    publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type="EXACT78_BASELINE_ROUTE_CONSOLIDATION_V2_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=index,
    )

    stage_audits = {}
    for name, task in STAGE_TASKS.items():
        stage_audits[name] = audit_refs(REPO_ROOT / "tasks/current" / task / "TASK_PACKET.json")
    stage_failed = sum(value["failed"] for value in stage_audits.values())
    stage_total = sum(value["references"] for value in stage_audits.values())
    stage_matched = sum(value["matched"] for value in stage_audits.values())

    p1 = {
        name: p1_summary(REPO_ROOT / "_run/current" / task / "attempts/attempt_0001/RESULT.json")
        for name, task in P1_TASKS.items()
    }
    p0_interpretation = load_json(
        REPO_ROOT / "_run/current" / P1_TASKS["retry6_eight_session"] / "attempts/attempt_0001/RESULT.json"
    )["p0_interpretation"]
    audit = {
        "schema_version": "exact78-baseline-route-audit-v2",
        "task_id": TASK,
        "visual_parent": {
            "robot_result": artifact_ref(PARENT_ROBOT),
            "expected_sha256": PARENT_ROBOT_SHA,
            "selected_artifacts": {
                key: {**reference, "status": "MATCH" if not validate_artifact_ref(reference) else "DRIFT"}
                for key, reference in load_json(POINTER)["selected_artifacts"].items()
            },
        },
        "stage_packet_closure": {
            "references": stage_total,
            "matched": stage_matched,
            "failed": stage_failed,
            "strict_replay_closure_passed": stage_failed == 0,
            "stages": stage_audits,
        },
        "general_route_evidence": {
            "p0_source_inventory": p0_interpretation,
            "p1_candidates": p1,
            "selected_general_executor": None,
        },
        "decisions": {
            "visual_parent_retained": True,
            "visual_parent_claimed_strictly_replayable": False,
            "general_executor_selected": False,
            "algorithm_executed": False,
            "files_deleted": False,
            "threshold_or_model_changed": False,
            "training_eligible": False,
        },
    }
    atomic_json(ROOT / "AUDIT.json", audit)
    update_exact78_documents(audit)

    issues = []
    for stage, stage_audit in stage_audits.items():
        for failure in stage_audit["failures"]:
            issues.append({"category": "STRICT_REPLAY_CLOSURE", "stage": stage, **failure})
    issues.extend(
        [
            {"category": "GENERAL_P1", "candidate": "retry6", "issue": "reach 0.675105 < 0.90 and only 6/8 action-complete"},
            {"category": "GENERAL_P1", "candidate": "retry8", "issue": "numeric reach passed but human visual review rejected color/flange/palm/jumps"},
            {"category": "GENERAL_P1", "candidate": "retry11", "issue": "acceleration 0.100183 > 0.06 with visible temporal jump"},
            {"category": "GENERAL_P1", "candidate": "fast_retry1", "issue": "reach 0.840678 < 0.90"},
        ]
    )
    atomic_json(ROOT / "ISSUES.json", {"task_id": TASK, "issues": issues})
    write_text(
        ROOT / "REPORT_ZH.md",
        f"""# Exact78 基线路由收敛结果

- V2 所选视觉产物：完整，继续作为 Cards042 唯一视觉父结果。
- 历史阶段 Task Packet：{stage_matched}/{stage_total} 引用匹配，{stage_failed} 个失配；严格重放闭包未通过。
- P0 源输入：156/156 可解释，其中 113 条 30 FPS、43 条 25 FPS。
- P1 通用执行器：未选中；retry6、retry8、retry11、fast retry1 均未同时通过数值与视觉门。
- 当前通用 runnable operation：`null`。
- 算法运行：否；删除：否；模型/门限修改：否；0916 修改：否。

完整机器证据见 `AUDIT.json` 与 `ISSUES.json`。本任务只完成路由和文档收敛，不把视觉父结果错误提升为通用执行基线。
""",
    )

    guards_after = {str(path): artifact_ref(path) for path in ROBOT100_GUARDS if path.is_file()}
    if guards_before != guards_after:
        raise RuntimeError("ROBOT100_GUARD_CHANGED")
    atomic_json(ROOT / "ROBOT100_GUARD.json", {"unchanged": True, "files": guards_after})
    result = {
        "schema_version": "exact78-baseline-route-consolidation-result-v2",
        "task_id": TASK,
        "status": "PASSED_TRUTHFUL_ROUTE_PUBLISHED",
        "execution_complete": True,
        "visual_parent_retained": True,
        "visual_parent_selected_artifacts_intact": True,
        "strict_replay_closure_passed": False,
        "general_raw_to_robot_qualified": False,
        "current_general_operation": None,
        "visual_qualified": False,
        "quality_passed": False,
        "training_eligible": False,
        "audit": artifact_ref(ROOT / "AUDIT.json"),
        "issues": artifact_ref(ROOT / "ISSUES.json"),
        "report": artifact_ref(ROOT / "REPORT_ZH.md"),
        "route": artifact_ref(ROUTE),
        "route_zh": artifact_ref(ROUTE_ZH),
        "visual_pointer": artifact_ref(POINTER),
        "visual_handoff": artifact_ref(BASELINE_ZH),
        "robot100_unchanged": True,
        "deletion_authorized": False,
        "claim_limit": packet["claim_limit"],
    }
    atomic_json(ROOT / "RESULT.json", result)
    atomic_json(
        RECEIPT,
        {
            "schema_version": "exact78-baseline-route-consolidation-receipt-v2",
            "task_id": TASK,
            "status": "PASSED",
            "result": artifact_ref(ROOT / "RESULT.json"),
            "route": artifact_ref(ROUTE),
            "visual_parent_robot_result_sha256": PARENT_ROBOT_SHA,
            "general_executor_selected": False,
            "deletion_authorized": False,
            "claim_limit": packet["claim_limit"],
        },
    )

    authority = copy.deepcopy(load_json(AUTHORITY_PATH))
    authority["claims"] = [
        claim
        for claim in authority.get("claims", [])
        if claim.get("scope") != "exact78/play_cards_0902_042/current_visual_baseline_v2"
    ]
    authority.setdefault("claims", []).append(
        {
            "claim": "Exact78 Cards042 V2 remains the accepted visual parent, while no general Raw-to-Robot executor is currently qualified.",
            "claim_limit": packet["claim_limit"],
            "scope": "exact78/current_visual_parent_and_general_route",
            "status": "DEVELOPMENT_EVIDENCE",
            "evidence": [artifact_ref(RECEIPT), artifact_ref(ROUTE), artifact_ref(PARENT_ROBOT)],
        }
    )
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(
        status="PASSED",
        pid=None,
        proc_start_ticks=None,
        updated_at=now_iso(),
        heartbeat_at=now_iso(),
        result=artifact_ref(ROOT / "RESULT.json"),
        terminal_receipt=artifact_ref(RECEIPT),
        quality="ROUTE_TRUTHFUL_GENERAL_NOT_QUALIFIED",
        adoption="CURRENT_EXACT78_ROUTE_V1",
        review="MACHINE_AUDITED",
    )
    state["next_task"] = None
    index = index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(
        packet_revision="EXACT78_BASELINE_ROUTE_CONSOLIDATION_V2_TERMINAL",
        plan_revision=packet["plan_revision"],
        execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS",
        supersedes_index=artifact_ref(INDEX),
    )
    published = publish_bundle(
        authority,
        state,
        event_type="EXACT78_BASELINE_ROUTE_CONSOLIDATION_V2_COMPLETE",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"],
        generator_path=Path(__file__),
        task_packet_index_path=INDEX,
        task_packet_index_value=index,
    )
    validation = subprocess.run(
        [sys.executable, "-B", "-m", "chaoyang.cli", "validate-governance"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )
    atomic_json(
        ROOT / "POST_VALIDATION.json",
        {"returncode": validation.returncode, "stdout": validation.stdout, "stderr": validation.stderr},
    )
    if validation.returncode:
        raise RuntimeError("POST_PUBLISH_VALIDATION_FAILED:" + validation.stdout[-3000:] + validation.stderr[-3000:])
    print(
        json.dumps(
            {
                "status": "PASSED",
                "governance_revision": published["governance_revision"],
                "result": artifact_ref(ROOT / "RESULT.json"),
                "route": artifact_ref(ROUTE),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
