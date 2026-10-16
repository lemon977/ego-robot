"""Bounded T0 correction of immutable calibration and command metadata."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets

from chaoyang.governance.common import (
    AUTHORITY_PATH, RECEIPT_PATH, REPO_ROOT, TASK_STATE_PATH,
    artifact_ref, atomic_json, atomic_write, canonical_bytes, load_json,
    now_iso, process_identity, publish_bundle, validate_artifact_ref,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


TASK = "exact78_raw_to_robot_t0_correction_v1_20260930"
SCRIPT = Path(__file__)
OUT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_RAW_TO_ROBOT_T0_CORRECTION_V1.json"
INDEX = REPO_ROOT / "tasks/current/INDEX.json"
PLAN = REPO_ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/00_EXECUTION.md"
CONTINUE = PLAN.parent / "01_CONTINUE_ZH.md"
UPSTREAM = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/attempts/attempt_0001"
UPSTREAM_PACKET = REPO_ROOT / "tasks/current/exact78_raw_to_robot_t0_input_freeze_retry1_20260930/TASK_PACKET.json"
UPSTREAM_RUNNER = REPO_ROOT / "src/chaoyang/governance/execute_exact78_raw_t0_retry1.py"
UPSTREAM_HELPER = REPO_ROOT / "src/chaoyang/governance/execute_exact78_raw_t0_v1.py"
FIRST_RESULT = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_input_freeze_v1_20260930/attempts/attempt_0001/RESULT.json"
FIRST_RECOVERY = FIRST_RESULT.parent / "TERMINALIZATION_RECOVERY.json"
ROUTE = REPO_ROOT / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json"
PARENT = REPO_ROOT / "_run/current/exact78_cards042_fresh_robot_v1_20260929/attempts/attempt_0001/robot/RESULT.json"
PARENT_SHA = "8338f2246e476e6afef2995eaa18423a4aafe0625ae829bd5315f809120a78e1"


def calibration(path: Path) -> dict:
    data = load_json(path)
    top = data.get("extrinsics")
    top_pair = isinstance(top, dict) and bool(top.get("left")) and bool(top.get("right"))
    nested_pair = all(bool(data.get(side, {}).get("extrinsic")) for side in ("left", "right"))
    meta = path.with_name("camera_params.meta.json")
    if top_pair and nested_pair:
        raise RuntimeError(f"AMBIGUOUS_EXTRINSIC_LAYOUT:{path}")
    if nested_pair and meta.is_file() and not load_json(meta).get("extrinsics_included"):
        raise RuntimeError(f"CALIBRATION_META_CONFLICT:{path}")
    location = "top_level_extrinsics_pair" if top_pair else "per_eye_extrinsic_pair" if nested_pair else "ABSENT"
    return {"camera_params": artifact_ref(path),
            "camera_params_meta": artifact_ref(meta) if meta.is_file() else "ABSENT",
            "extrinsic_field_location": location,
            "extrinsics": "SOURCE_PRESENT" if location != "ABSENT" else "ABSENT",
            "metric_geometry": "UNVERIFIED_SOURCE_CALIBRATION_ONLY" if location != "ABSENT"
                               else "BLOCKED_CALIBRATION_ABSENT",
            "visual_robot": "NOT_BLOCKED_BY_STEREO_EXTRINSICS"}


def prepare() -> tuple[dict, dict]:
    refs = {"upstream_input_freeze": artifact_ref(UPSTREAM / "INPUT_FREEZE.json"),
            "upstream_frame_time_map": artifact_ref(UPSTREAM / "FRAME_TIME_MAP.json"),
            "upstream_result": artifact_ref(UPSTREAM / "RESULT.json"),
            "upstream_packet": artifact_ref(UPSTREAM_PACKET),
            "upstream_runner_code": artifact_ref(UPSTREAM_RUNNER),
            "upstream_helper_code": artifact_ref(UPSTREAM_HELPER),
            "first_failed_result": artifact_ref(FIRST_RESULT),
            "first_terminal_recovery": artifact_ref(FIRST_RECOVERY),
            "plan": artifact_ref(PLAN), "route": artifact_ref(ROUTE),
            "visual_parent": artifact_ref(PARENT)}
    if refs["visual_parent"]["sha256"] != PARENT_SHA:
        raise RuntimeError("VISUAL_PARENT_SHA_DRIFT")
    if load_json(ROUTE)["general_raw_to_robot"]["current_runnable_operation"] is not None:
        raise RuntimeError("GENERAL_ROUTE_CONFLICT")
    upstream_packet = load_json(UPSTREAM_PACKET)
    if upstream_packet["commands"] != ["PYTHONPATH=src python -B -m chaoyang.governance.execute_exact78_raw_t0_v1"]:
        raise RuntimeError("UPSTREAM_COMMAND_DEFECT_NOT_REPRODUCED")
    if load_json(UPSTREAM / "RESULT.json")["status"] != "PASSED":
        raise RuntimeError("UPSTREAM_NOT_SEALED")
    if load_json(FIRST_RECOVERY)["status"] != "FAILED_RUNTIME_FINAL":
        raise RuntimeError("FIRST_ATTEMPT_NOT_TERMINAL")
    frozen = load_json(UPSTREAM / "INPUT_FREEZE.json")
    timeline = load_json(UPSTREAM / "FRAME_TIME_MAP.json")
    selected = frozen["selected"]
    expected = {"get_potato_chips_0901_013", "play_cards_0901_001",
                "get_potato_chips_0902_103", "play_cards_0902_042",
                "play_cards_0915_001", "play_cards_0915_002"}
    if set(selected) != expected or set(timeline["sessions"]) != expected:
        raise RuntimeError("FROZEN_SESSION_SET_DRIFT")
    if frozen["cohorts"]["0902"]["holdout_status"] != "SPLIT_NOT_ESTABLISHED":
        raise RuntimeError("0902_SPLIT_PROOF_DRIFT")
    if len(frozen["cohorts"]["0901"]["distinct_group_keys"]) != 2 or len(frozen["cohorts"]["0902"]["distinct_group_keys"]) != 1:
        raise RuntimeError("TRACKER_SOURCE_GROUP_COUNT_DRIFT")
    if any(".pico_editor_trash" in r["raw_path"] for cohort in frozen["cohorts"].values()
           for r in cohort["rows"]):
        raise RuntimeError("NONFORMAL_SOURCE_INCLUDED")
    states = {}
    for name, row in selected.items():
        camera_ref = row["files"].get("camera", row["files"].get("camera_params.json"))
        if not isinstance(camera_ref, dict) or validate_artifact_ref(camera_ref):
            raise RuntimeError(f"CAMERA_REF_DRIFT:{name}")
        states[name] = calibration(Path(camera_ref["path"]))
    if (states["play_cards_0902_042"]["extrinsic_field_location"] != "top_level_extrinsics_pair"
            or states["get_potato_chips_0902_103"]["extrinsic_field_location"] != "top_level_extrinsics_pair"
            or states["get_potato_chips_0901_013"]["extrinsic_field_location"] != "ABSENT"
            or states["play_cards_0901_001"]["extrinsic_field_location"] != "ABSENT"
            or states["play_cards_0915_001"]["extrinsic_field_location"] != "per_eye_extrinsic_pair"
            or states["play_cards_0915_002"]["extrinsic_field_location"] != "per_eye_extrinsic_pair"):
        raise RuntimeError("CALIBRATION_STRUCTURE_ASSERTION_FAILED")
    for name in ("play_cards_0915_001", "play_cards_0915_002"):
        row = timeline["sessions"][name]
        count = row["decoded_frames"]
        if count != row["raw_qpc_count"] or count != row["raw_capture_timestamp_count"]:
            raise RuntimeError(f"RAW_FRAME_TIMESTAMP_COUNT_CONFLICT:{name}")
        if row["time_mode"] != "TIMEBASE_UNVERIFIED" or row["capture_to_decode_alignment"] != "INDEX_ALIGNMENT_UNVERIFIED":
            raise RuntimeError(f"TIMEBASE_CLAIM_CONFLICT:{name}")
    cards002 = timeline["sessions"]["play_cards_0915_002"]["frame_count_provenance"]
    if (cards002["decoded_raw"], cards002["qpc_raw_lines"], cards002["timestamp_raw_lines"],
            cards002["legacy_processed_group_manifest"]) != (131, 131, 131, 130):
        raise RuntimeError("0915_CARDS002_COUNT_EVIDENCE_DRIFT")
    refs["calibration_files"] = {name: state["camera_params"] for name, state in states.items()}
    refs["calibration_meta_files"] = {name: state["camera_params_meta"] for name, state in states.items()}
    return refs, {"selected": selected, "timeline": timeline, "calibration_states": states,
                  "campaign_deadline_at": upstream_packet["campaign_deadline_at"]}


def packet_value(inputs: dict, observed: dict) -> dict:
    now = datetime.now().astimezone()
    campaign_deadline = datetime.fromisoformat(observed["campaign_deadline_at"])
    if campaign_deadline <= now:
        raise RuntimeError("CAMPAIGN_DEADLINE_EXPIRED")
    code = [artifact_ref(SCRIPT)]
    signature = hashlib.sha256(canonical_bytes({"inputs": inputs, "runtime_code": code,
        "schema": "exact78-raw-t0-correction-v1", "config": "read-only-correction"})).hexdigest()
    writer = {"pid": os.getpid(), "proc_start_ticks": process_identity(os.getpid())["start_ticks"],
              "executor_epoch": 1, "fencing_token": secrets.token_hex(24), "scope": TASK}
    return {"schema_version": "chaoyang-r22-task-packet-v1", "task_id": TASK,
        "objective": "Bind sealed T0 raw/frame evidence to corrected per-session calibration and actual runtime code/command closure",
        "phase": "EXACT78_RAW_TO_ROBOT_T0_CORRECTION", "plan_revision": "EXACT78_RAW_TO_ROBOT_V1_20260930",
        "execution_revision": "EXACT78_RAW_T0_CORRECTION_V1", "registered_at": now.isoformat(),
        "deadline_at": min(now + timedelta(hours=1), campaign_deadline).isoformat(),
        "campaign_deadline_at": campaign_deadline.isoformat(),
        "read_set": ["AGENTS.md", str(PLAN.relative_to(REPO_ROOT)), str((UPSTREAM / "RESULT.json").relative_to(REPO_ROOT)),
                     str((UPSTREAM / "INPUT_FREEZE.json").relative_to(REPO_ROOT)),
                     str((UPSTREAM / "FRAME_TIME_MAP.json").relative_to(REPO_ROOT)),
                     str(ROUTE.relative_to(REPO_ROOT)), str(PARENT.relative_to(REPO_ROOT)),
                     str(SCRIPT.relative_to(REPO_ROOT))],
        "write_set": [str(OUT.relative_to(REPO_ROOT)), str(PACKET.relative_to(REPO_ROOT)),
                      str(RECEIPT.relative_to(REPO_ROOT)), str(CONTINUE.relative_to(REPO_ROOT)),
                      "tasks/current/INDEX.json", "docs/governance"],
        "prerequisites": ["GOVERNANCE_PASS", "NO_ACTIVE_TASK", "UPSTREAM_SHA_PINNED",
                          "NO_MODEL_RUN", "NO_DELETION", "0916_READ_ONLY"],
        "required_outputs": ["CORRECTED_CAPABILITY.json", "01_CONTINUE_ZH.md", "RESULT.json"],
        "budgets": {"workers_max": 1, "cpu_threads_total": 1, "gpu_count": 0,
                    "training_steps": 0, "wall_seconds_max": 3600, "session_attempts_max": 1},
        "stop_conditions": ["SHA_DRIFT", "DOUBLE_WRITER", "DEADLINE", "ONE_ATTEMPT_TERMINAL"],
        "claim_limit": "T0 correction plus sealed upstream evidence only; no algorithm execution or product qualification",
        "writer": writer, "inputs": inputs, "runtime_code": code, "run_signature": signature,
        "commands": ["PYTHONPATH=src python -B -m chaoyang.governance.execute_exact78_raw_t0_retry2"]}


def index_copy() -> dict:
    return {key: copy.deepcopy(value) for key, value in load_json(INDEX).items()
            if key not in {"governance_revision", "generation_id", "generated_at", "supersedes_index"}}


def register(packet: dict) -> None:
    if any(path.exists() for path in (OUT, PACKET, RECEIPT, CONTINUE)):
        raise RuntimeError("FRESH_CORRECTION_REQUIRED")
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    index = index_copy()
    if state.get("next_task") or any(row.get("execution_allowed") for row in index["task_packets"]):
        raise RuntimeError("ACTIVE_TASK_CONFLICT")
    OUT.mkdir(parents=True)
    atomic_json(OUT / "WRITER.json", packet["writer"])
    PACKET.parent.mkdir(parents=True)
    atomic_json(PACKET, packet)
    now = now_iso()
    state["tasks"].append({"task_id": TASK, "phase": packet["phase"], "attempt": 1,
        "status": "RUNNING", "session": "Exact78_three_source", "updated_at": now,
        "heartbeat_at": now, "pid": packet["writer"]["pid"],
        "proc_start_ticks": packet["writer"]["proc_start_ticks"], "gpu_id": None,
        "task_packet": artifact_ref(PACKET), "run_signature": packet["run_signature"],
        "quality": "PENDING", "adoption": "NOT_ADOPTED", "review": "MACHINE_AUDIT"})
    for row in state["tasks"]:
        if row["task_id"] == "exact78_raw_to_robot_t0_input_freeze_retry1_20260930":
            row["quality"] = "NOT_ADOPTED_WRONG_CALIBRATION_AND_COMMAND_CLOSURE"
            row["adoption"] = "NOT_ADOPTED"
    state["next_task"] = {"task_id": TASK, "session": "Exact78_three_source",
        "prerequisites": packet["prerequisites"], "expected_resource": "CPU_ONE_THREAD",
        "stop_condition": "one immutable correction result or one-hour deadline"}
    index["task_packets"].append({"task_id": TASK, "packet_path": str(PACKET.relative_to(REPO_ROOT)),
        "packet_sha256": artifact_ref(PACKET)["sha256"],
        "execution_class": "CURRENT_LEDGER_ROUTABLE", "execution_allowed": True})
    index.update(packet_revision="EXACT78_RAW_T0_CORRECTION_V1_RUNNING",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS", supersedes_index=artifact_ref(INDEX))
    publish_bundle(load_json(AUTHORITY_PATH), state, event_type="EXACT78_RAW_T0_CORRECTION_REGISTERED",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=SCRIPT,
        task_packet_index_path=INDEX, task_packet_index_value=index)


def fence(packet: dict) -> None:
    writer = packet["writer"]
    identity = process_identity(os.getpid())
    if writer["pid"] != os.getpid() or writer["proc_start_ticks"] != identity["start_ticks"]:
        raise RuntimeError("WRITER_IDENTITY_DRIFT")
    if load_json(OUT / "WRITER.json") != writer:
        raise RuntimeError("WRITER_TOKEN_DRIFT")
    state = load_json(TASK_STATE_PATH)
    active = [row for row in state["tasks"] if row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE")]
    if len(active) != 1 or active[0]["task_id"] != TASK or active[0]["run_signature"] != packet["run_signature"]:
        raise RuntimeError("DOUBLE_WRITER_OR_SIGNATURE_DRIFT")


def finish(packet: dict, status: str, outputs: dict[str, Path], reason: str | None = None) -> dict:
    fence(packet)
    if status not in ("PASSED", "FAILED_RUNTIME_FINAL"):
        raise RuntimeError("INVALID_TERMINAL_STATUS")
    result_path = OUT / "RESULT.json"
    if result_path.exists() or RECEIPT.exists():
        raise RuntimeError("IMMUTABLE_RESULT_ALREADY_EXISTS")
    result = {"schema_version": "exact78-raw-t0-correction-result-v1", "task_id": TASK,
        "status": status, "execution_complete": status == "PASSED",
        "campaign_deadline_at": packet["campaign_deadline_at"],
        "upstream_input_freeze": packet["inputs"]["upstream_input_freeze"],
        "upstream_frame_time_map": packet["inputs"]["upstream_frame_time_map"],
        "outputs": {name: artifact_ref(path) for name, path in outputs.items()},
        "reason": reason, "models_run": False, "training_run": False,
        "source_mutated": False, "files_deleted": False, "robot100_modified": False,
        "general_operation": None, "visual_qualified": False, "product_qualified": False,
        "training_eligible": False,
        "next_action": "T1 register candidate CLI/recipe; consume upstream freeze and frame map only with this correction receipt",
        "claim_limit": packet["claim_limit"]}
    atomic_json(result_path, result)
    atomic_json(RECEIPT, {"schema_version": "exact78-raw-t0-correction-receipt-v1",
        "task_id": TASK, "status": status, "result": artifact_ref(result_path),
        "deletion_authorized": False, "claim_limit": packet["claim_limit"]})
    state = copy.deepcopy(load_json(TASK_STATE_PATH))
    row = next(item for item in state["tasks"] if item["task_id"] == TASK)
    row.update(status=status, pid=None, proc_start_ticks=None, updated_at=now_iso(), heartbeat_at=now_iso(),
        result=artifact_ref(result_path), terminal_receipt=artifact_ref(RECEIPT),
        quality="CORRECTED_INPUT_EVIDENCE_ONLY" if status == "PASSED" else "FAILED_RUNTIME",
        adoption="NOT_ADOPTED", review="MACHINE_AUDITED")
    state["next_task"] = None
    index = index_copy()
    for item in index["task_packets"]:
        if item["task_id"] == TASK:
            item["execution_allowed"] = False
    index.update(packet_revision="EXACT78_RAW_T0_CORRECTION_V1_TERMINAL",
        plan_revision=packet["plan_revision"], execution_revision=packet["execution_revision"],
        status="PASS_NO_ACTIVE_TASKS", supersedes_index=artifact_ref(INDEX))
    published = publish_bundle(load_json(AUTHORITY_PATH), state,
        event_type=f"EXACT78_RAW_T0_CORRECTION_{status}",
        expected_revision=load_json(RECEIPT_PATH)["governance_revision"], generator_path=SCRIPT,
        task_packet_index_path=INDEX, task_packet_index_value=index)
    return {"status": status, "governance_revision": published["governance_revision"],
            "result": artifact_ref(result_path)}


def execute(packet: dict, observed: dict) -> dict:
    fence(packet)
    selected = observed["selected"]
    counts = {"development": sum(row["role"] == "development" for row in selected.values()),
              "independent_holdout": sum(row["role"] == "holdout" for row in selected.values()),
              "selected_total": len(selected), "0902_independent_holdout": 0}
    capability = {"schema_version": "exact78-raw-t0-corrected-capability-v1",
        "status": "INPUT_EVIDENCE_FROZEN", "upstream_input_freeze": packet["inputs"]["upstream_input_freeze"],
        "upstream_frame_time_map": packet["inputs"]["upstream_frame_time_map"],
        "upstream_command_closure": "DEFECTIVE_DO_NOT_RUN_UPSTREAM_PACKET_COMMAND",
        "actual_upstream_runner": "PYTHONPATH=src python -B -m chaoyang.governance.execute_exact78_raw_t0_retry1",
        "upstream_runner_code": packet["inputs"]["upstream_runner_code"],
        "upstream_helper_code": packet["inputs"]["upstream_helper_code"],
        "correction_command": packet["commands"][0], "correction_runtime_code": packet["runtime_code"],
        "source_counts": counts, "source_split": {"0901": "FROZEN_INDEPENDENT_GROUP",
            "0902": "SPLIT_NOT_ESTABLISHED", "0915": "FROZEN_INDEPENDENT_GROUP"},
        "calibration_states": observed["calibration_states"],
        "0915_cards002_frame_count": {"raw_decoded": 131, "raw_qpc": 131,
            "raw_timestamp": 131, "legacy_group_manifest": 130,
            "finding": "RAW_INTERNAL_COUNTS_AGREE_LEGACY_DERIVED_COUNT_DIFFERS"},
        "0915_time_mode": "TIMEBASE_UNVERIFIED", "0915_capture_alignment": "INDEX_ALIGNMENT_UNVERIFIED",
        "model_closure": "PARTIAL_DEPTH_MODEL_NOT_SELECTED",
        "visual_robot": "NOT_RUN", "clean": "NOT_RUN", "depth": "NOT_RUN",
        "object6d": "NOT_RUN", "occlusion": "NOT_RUN", "product": "NOT_RUN",
        "training": "NOT_ELIGIBLE",
        "claim_limit": packet["claim_limit"]}
    atomic_json(OUT / "CORRECTED_CAPABILITY.json", capability)
    text = f"""# Exact78 Raw→Robot 续接（T0）

- 治理 T0 correction 已封存；真正采样为 4 条开发 + 2 条跨组留出，共 6 条。0902 正式会话只有 1 个 merged 采集组，独立留出 `SPLIT_NOT_ESTABLISHED`；不得写 7/7。第七条同组片只可作执行诊断。
- 0901 开发 Chips013 与留出 Cards001 属 2 个原始 merged 组；两条都缺双目外参，公制几何消费者 `BLOCKED_CALIBRATION_ABSENT`，视觉 Robot 不受此局部阻断。
- 0902 Cards042/Chips103 同一 merged 组；相机参数顶层 `extrinsics.left/right` 存在，公制质量仍未验证。0915 两条的 `left/right.extrinsic` 存在，同样未证明编码域几何质量。
- 0915 Cards002 原始 H.264 解码 131、原始 QPC 131、原始时间行 131；旧来源组 manifest 写 130。原始三者自洽，旧派生数只供对照。QPC 单位、频率、采样位置、头显位姿时钟域及采集 ID 到解码序号的对应仍未证，标 `TIMEBASE_UNVERIFIED` / `INDEX_ALIGNMENT_UNVERIFIED`。
- 原 T0 retry1 的 Task Packet 命令与实际 wrapper 不一致，且把 0902 外参误记 ABSENT。其 INPUT_FREEZE/FRAME_TIME_MAP 仅与本 correction RESULT 一起使用，原 RESULT 单独 `NOT_ADOPTED_WRONG_CALIBRATION_AND_COMMAND_CLOSURE`。
- H24 硬截止：`{packet['campaign_deadline_at']}`，从首个 T0 安全登记起算。当前 Robot/Clean/Depth/Object6D/遮挡/完整产品均未运行；通用入口仍为 `null`。
- 下一动作：T1 登记独立有限 Task Packet，实现 `chaoyang submit exact78-raw-to-robot` 候选入口、实际输入 SHA 与禁读负测，绑定本 correction 及上游两份 SHA；T2 只对预冻结样本按依赖流水运行，T3 按真实 6 条与缺失的 0902 留出发布 H24 矩阵。
- 清理 lane：C0 独立登记引用/PID/FD/CWD/Git/许可证证明，C1 仅在精确目标与证明齐全时删除；目前未删任何旧物。

权威证据：`{(OUT / 'RESULT.json').relative_to(REPO_ROOT)}`、`{(OUT / 'CORRECTED_CAPABILITY.json').relative_to(REPO_ROOT)}`、`{(UPSTREAM / 'INPUT_FREEZE.json').relative_to(REPO_ROOT)}`、`{(UPSTREAM / 'FRAME_TIME_MAP.json').relative_to(REPO_ROOT)}`。
"""
    atomic_write(CONTINUE, text.encode("utf-8"))
    atomic_json(OUT / "RESUME_SUMMARY.json", {"schema_version": "exact78-t0-resume-summary-v1",
        "next_action": "T1 candidate entrypoint finite packet", "campaign_deadline_at": packet["campaign_deadline_at"],
        "source_counts": counts, "blockers": ["0902_SPLIT_NOT_ESTABLISHED", "0915_TIMEBASE_UNVERIFIED"]})
    return finish(packet, "PASSED", {"corrected_capability": OUT / "CORRECTED_CAPABILITY.json",
        "continue_zh": CONTINUE, "resume_summary": OUT / "RESUME_SUMMARY.json"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    inputs, observed = prepare()
    packet = packet_value(inputs, observed)
    errors = _validate_packet(packet)
    if errors:
        raise RuntimeError("INVALID_PACKET:" + ";".join(errors))
    if args.check_only:
        print(json.dumps({"status": "DRY_RUN_PASS", "task_id": TASK,
            "selected": list(observed["selected"]),
            "calibration": {name: row["extrinsic_field_location"] for name, row in observed["calibration_states"].items()},
            "commands": packet["commands"], "runtime_code": packet["runtime_code"],
            "campaign_deadline_at": packet["campaign_deadline_at"],
            "inputs_sha": hashlib.sha256(canonical_bytes(inputs)).hexdigest()}, ensure_ascii=False))
        return 0
    register(packet)
    try:
        print(json.dumps(execute(packet, observed), ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        outputs = {name: path for name, path in (("corrected_capability", OUT / "CORRECTED_CAPABILITY.json"),
            ("continue_zh", CONTINUE), ("resume_summary", OUT / "RESUME_SUMMARY.json")) if path.exists()}
        print(json.dumps(finish(packet, "FAILED_RUNTIME_FINAL", outputs,
            f"{type(error).__name__}:{error}"), ensure_ascii=False), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
