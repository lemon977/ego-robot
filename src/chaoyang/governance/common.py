from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import fcntl
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[3]
GOVERNANCE_ROOT = REPO_ROOT / "docs" / "governance"
SCHEMA_ROOT = GOVERNANCE_ROOT / "schemas"
AUTHORITY_PATH = GOVERNANCE_ROOT / "CURRENT_AUTHORITY_INDEX.json"
TASK_STATE_PATH = GOVERNANCE_ROOT / "LONG_HORIZON_TASK_STATE.json"
STATUS_PATH = GOVERNANCE_ROOT / "CURRENT_PROJECT_STATUS_ZH.md"
RECEIPT_PATH = GOVERNANCE_ROOT / "CURRENT_STATUS_RECEIPT.json"
MIN_STATUS_PATH = GOVERNANCE_ROOT / "CURRENT_PROJECT_STATUS_MIN.json"
TASK_QUEUE_PATH = GOVERNANCE_ROOT / "TASK_QUEUE.json"
CHANGELOG_PATH = GOVERNANCE_ROOT / "STATE_CHANGELOG.jsonl"
LOCK_PATH = REPO_ROOT / "_run" / "current" / "locks" / "governance.lock"
BASELINE_REGISTRY_PATH = GOVERNANCE_ROOT / "CURRENT_BASELINE_REGISTRY_V2.json"
STAGE_BASELINES_PATH = GOVERNANCE_ROOT / "CURRENT_STAGE_BASELINES_ZH.md"
FILE_LAYOUT_PATH = GOVERNANCE_ROOT / "CURRENT_FILE_LAYOUT.json"
REGRESSION_MANIFEST_PATH = GOVERNANCE_ROOT / "CURRENT_REGRESSION_MANIFEST.json"
PLAN_REVISION_PATH = GOVERNANCE_ROOT / "PLAN_REVISION.json"
V71_TASK_PACKET_INDEX_PATH = GOVERNANCE_ROOT / "CURRENT_V71_TASK_PACKET_INDEX.json"
DOC_AUTHORITY_MAP_PATH = GOVERNANCE_ROOT / "DOC_AUTHORITY_MAP.json"
ALGORITHM_CONTRACT_PATH = GOVERNANCE_ROOT / "ALGORITHM_CONTRACT.json"
PLAN_MIGRATION_RECEIPT_PATH = GOVERNANCE_ROOT / "PLAN_MIGRATION_RECEIPT.json"
RC1_STATUS_MIN_PATH = GOVERNANCE_ROOT / "CURRENT_RC1_STATUS_MIN.json"
RC1_PLAN_PATH = GOVERNANCE_ROOT / "CHAOYANG_RC1_FINAL_DELIVERY_PLAN_ZH.md"
RC1_RELEASE_SPEC_PATH = GOVERNANCE_ROOT / "RC1_RELEASE_SPEC.json"
RC1_CONTRACT_PATH = GOVERNANCE_ROOT / "VISUAL_AUX_RC1_CONTRACT.json"
RC1_TASK_DAG_PATH = GOVERNANCE_ROOT / "RC1_TASK_DAG.json"

TASK_STATUSES = {
    "PENDING",
    "WAIT_GPU_RESOURCE",
    "CLAIMED",
    "RUNNING",
    "PASSED",
    "REJECTED_QUALITY",
    "FAILED_QUALITY_C",
    "FAILED_RUNTIME_RETRYABLE",
    "FAILED_RUNTIME_FINAL",
    "BLOCKED_PREREQ",
    "BLOCKED_RESOURCE",
    "BLOCKED_EXTERNAL",
    "BLOCKED_REFERENCE_PROOF",
    "UNKNOWN_VERIFICATION_REQUIRED",
    "CANCELLED",
}
TERMINAL_TASK_STATUSES = {
    "PASSED",
    "REJECTED_QUALITY",
    "FAILED_QUALITY_C",
    "FAILED_RUNTIME_FINAL",
    "BLOCKED_PREREQ",
    "BLOCKED_RESOURCE",
    "BLOCKED_EXTERNAL",
    "BLOCKED_REFERENCE_PROOF",
    "UNKNOWN_VERIFICATION_REQUIRED",
    "CANCELLED",
}
CLAIM_STATUSES = {
    "SUPPORTED_EXTERNAL_TRUTH",
    "SUPPORTED_INTERNAL_CONSISTENCY",
    "DEVELOPMENT_EVIDENCE",
    "HYPOTHESIS_ONLY",
    "UNSUPPORTED",
    "WITHDRAWN",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_ref(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)}


def validate_artifact_ref(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    path = Path(str(value.get("path", "")))
    if not path.is_absolute():
        return [f"artifact path is not absolute: {path}"]
    if not path.is_file():
        return [f"artifact missing: {path}"]
    actual_bytes = path.stat().st_size
    actual_sha = sha256_file(path)
    if value.get("bytes") != actual_bytes:
        errors.append(f"bytes mismatch for {path}: expected={value.get('bytes')} actual={actual_bytes}")
    if value.get("sha256") != actual_sha:
        errors.append(f"sha256 mismatch for {path}: expected={value.get('sha256')} actual={actual_sha}")
    return errors


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_schema(name: str, value: Mapping[str, Any]) -> None:
    def expand(node: Any) -> Any:
        if isinstance(node, dict):
            reference = node.get("$ref")
            if isinstance(reference, str) and reference.endswith(".schema.json") and "://" not in reference:
                return expand(json.loads((SCHEMA_ROOT / reference).read_text(encoding="utf-8")))
            return {key: expand(item) for key, item in node.items()}
        if isinstance(node, list):
            return [expand(item) for item in node]
        return node

    schema = expand(json.loads((SCHEMA_ROOT / name).read_text(encoding="utf-8")))
    jsonschema.Draft202012Validator(schema).validate(value)


def atomic_write(path: Path, data: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Mapping[str, Any], mode: int = 0o444) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n", mode)


@contextmanager
def governance_lock() -> Iterator[None]:
    GOVERNANCE_ROOT.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def git_metadata() -> dict[str, str]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"

    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current") or "DETACHED"}


def process_identity(pid: int) -> dict[str, Any]:
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.is_file():
        return {"pid": pid, "alive": False, "start_ticks": None}
    fields = stat_path.read_text(encoding="utf-8").split()
    return {"pid": pid, "alive": True, "start_ticks": int(fields[21])}


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def freshness(task_state: Mapping[str, Any], at: datetime | None = None) -> dict[str, Any]:
    current = at or datetime.now().astimezone()
    active = [
        task for task in task_state.get("tasks", [])
        if task.get("status") in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
    ]
    if not active:
        return {"status": "FRESH", "age_seconds": 0, "reason": "no_active_tasks"}
    newest = max(parse_time(str(task.get("heartbeat_at") or task.get("updated_at"))) for task in active)
    age = max(0, int((current - newest).total_seconds()))
    suspected = []
    for task in active:
        pid = task.get("pid")
        expected_ticks = task.get("proc_start_ticks")
        if pid is None:
            continue
        identity = process_identity(int(pid))
        if not identity["alive"] or (expected_ticks is not None and identity["start_ticks"] != expected_ticks):
            suspected.append(task.get("task_id"))
    if suspected or age > 90:
        return {"status": "SUSPECTED_DEAD_WORKER", "age_seconds": age, "tasks": suspected}
    if age <= 300:
        status = "FRESH"
    elif age <= 900:
        status = "DELAYED"
    else:
        status = "STALE"
    return {"status": status, "age_seconds": age, "reason": "active_task_heartbeat"}


def render_status(authority: Mapping[str, Any], task_state: Mapping[str, Any]) -> str:
    fresh = freshness(task_state)
    waves = authority.get("waves", {})
    lines = [
        "# 当前项目实时事实页",
        "",
        "> 本文档由机器状态自动生成。关键计数不得手工修改。",
        "",
        "## A. 快照身份",
        "",
        f"- 状态生成时间：`{authority['generated_at']}`",
        f"- governance revision：`{authority['governance_revision']}`",
        f"- generation id：`{authority['generation_id']}`",
        f"- freshness：`{fresh['status']}`（age={fresh['age_seconds']}s）",
        f"- generator code SHA：`{authority['generator_code_sha']}`",
        f"- repository：`{authority['repository']['commit']}` / `{authority['repository']['branch']}`",
        f"- host：`{task_state.get('host', 'UNKNOWN')}`",
        f"- data root：`{authority.get('data_root', 'UNKNOWN')}`",
        "",
        "## B. exact78 固定分母",
        "",
        f"- Raw：`{waves.get('raw_total', 0)}`",
        f"- Wave0 metric-ready：`{waves.get('wave0_metric_ready', 0)}`",
        f"- Wave0 已有 Clean：`{waves.get('wave0_clean_passed', 0)}`",
        f"- Wave0 待 Clean：`{waves.get('wave0_clean_pending', 0)}`",
        f"- Wave0 Clean 质量 C：`{waves.get('wave0_clean_failed_quality_c', 0)}`",
        f"- Wave0 Clean 运行失败终态：`{waves.get('wave0_clean_failed_runtime_final', 0)}`",
        f"- Wave0 Clean 前置阻塞：`{waves.get('wave0_clean_blocked_prereq', 0)}`",
        f"- Wave1 新增：`{waves.get('wave1_delta', 0)}`",
        f"- Wave2 新增：`{waves.get('wave2_delta', 0)}`",
        f"- 整个 exact78 缺标定：`{waves.get('raw_calibration_missing', 0)}/156`",
        f"- 三路 A/B 中缺标定：`{waves.get('join_calibration_missing', 0)}/101`",
        "",
        "## C. 各阶段实时矩阵",
        "",
        "| 阶段 | 总数 | PASSED | C | RUNNING | BLOCKED | 当前 authority |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for stage in authority.get("stages", []):
        lines.append(
            "| {stage} | {total} | {passed} | {grade_c} | {running} | {blocked} | `{scope}` |".format(
                stage=stage["stage"], total=stage["total"], passed=stage["passed"],
                grade_c=stage["grade_c"], running=stage["running"], blocked=stage["blocked"],
                scope=stage["authority_scope"],
            )
        )
    lines += [
        "",
        "> Clean 的 PASSED 数量只表示当前冻结合同下的结构、来源和解码终态；不表示接触边界、隐藏物体外观或 donor 语义已经通过。Robotized/Visual Aux 还必须单独通过 contact-preservation、pixel-source legality 和 Occlusion 门。",
    ]
    active = [
        task for task in task_state.get("tasks", [])
        if task.get("status") in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
    ]
    lines += ["", "## D. 当前运行任务", ""]
    if not active:
        lines.append("当前无活跃任务。")
    else:
        lines += [
            "| task_id | session | phase | attempt | PID | GPU | heartbeat | 状态 |",
            "|---|---|---|---:|---:|---:|---|---|",
        ]
        for task in active:
            lines.append(
                f"| `{task['task_id']}` | `{task.get('session') or '-'}` | `{task['phase']}` | "
                f"{task['attempt']} | {task.get('pid') or '-'} | {task.get('gpu_id') if task.get('gpu_id') is not None else '-'} | "
                f"`{task.get('heartbeat_at') or '-'}` | `{task['status']}` |"
            )
    lines += ["", "## E. 当前阻塞", "", "| 阻塞项 | 状态 | 影响范围 | 解除条件 |", "|---|---|---|---|"]
    blockers = task_state.get("blockers", [])
    if blockers:
        for item in blockers:
            lines.append(f"| {item['name']} | `{item['status']}` | {item['scope']} | {item['resolution']} |")
    else:
        lines.append("| 无 | - | - | - |")
    lines += ["", "## F. 当前可支持的结论", "", "### 可以支持", ""]
    supported = [claim for claim in authority.get("claims", []) if claim["status"].startswith("SUPPORTED_")]
    lines += [f"- {claim['claim']}（`{claim['status']}`；边界：{claim['claim_limit']}）" for claim in supported] or ["- 当前无已发布支持结论。"]
    lines += ["", "### 不能支持或仅为假设", ""]
    limited = [claim for claim in authority.get("claims", []) if not claim["status"].startswith("SUPPORTED_")]
    lines += [f"- {claim['claim']}（`{claim['status']}`；边界：{claim['claim_limit']}）" for claim in limited] or ["- 无。"]
    lines += ["", "## G. 最新状态变化", ""]
    events = task_state.get("recent_events", [])[-20:]
    for category in ("PASSED", "FAILED_QUALITY_C", "FAILED_RUNTIME", "BLOCKED"):
        lines += [f"### {category}", ""]
        selected = [event for event in reversed(events) if str(event.get("status", "")).startswith(category)][:5]
        lines += [
            f"- `{event.get('task_id')}` / `{event.get('session') or '-'}` / `{event.get('status')}` / "
            f"{event.get('result', {}).get('path', 'no-result')} / `{event.get('created_at')}`"
            for event in selected
        ] or ["- 无。"]
        lines.append("")
    next_task = task_state.get("next_task")
    lines += ["## H. 下一任务", ""]
    if next_task:
        lines += [
            f"- next_task_id：`{next_task['task_id']}`",
            f"- next_session：`{next_task.get('session') or '-'}`",
            f"- prerequisites：`{', '.join(next_task.get('prerequisites', [])) or '-'}`",
            f"- expected_resource：`{next_task.get('expected_resource', '-')}`",
            f"- stop_condition：`{next_task.get('stop_condition', '-')}`",
        ]
    else:
        lines.append("状态机当前未选择下一任务。")
    lines += [
        "",
        "## 固定读取协议",
        "",
        "回答进度、精度或 authority 前，必须先校验 `CURRENT_STATUS_RECEIPT.json` 绑定的文件SHA；"
        "若状态为 `STALE`、`SUSPECTED_DEAD_WORKER` 或 `STATUS_CONFLICT`，停止推断并先做恢复审计。",
        "",
    ]
    return "\n".join(lines)


def render_min_status(authority: Mapping[str, Any], task_state: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded context card consumed by agents and status tooling.

    This is deliberately derived from the two authority ledgers on every publish;
    it is an interface cache, never an independent source of truth.
    """
    active = [
        task for task in task_state.get("tasks", [])
        if task.get("status") in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
    ]
    return {
        "schema_version": "chaoyang-current-project-status-min-v1",
        "governance_revision": authority["governance_revision"],
        "generation_id": authority["generation_id"],
        "generated_at": authority["generated_at"],
        "freshness": freshness(task_state),
        "cohort": authority.get("cohort"),
        "waves": authority.get("waves", {}),
        "stage_claim_limits": {
            "clean_passed": "STRUCTURAL_PROVENANCE_DECODE_ONLY_NOT_CONTACT_OR_SEMANTIC_CORRECTNESS",
            "robotized_training": "REQUIRES_SEPARATE_CONTACT_PRESERVATION_PIXEL_SOURCE_AND_OCCLUSION_GATES",
        },
        "active_tasks": [
            {
                key: task.get(key)
                for key in (
                    "task_id", "phase", "status", "session", "attempt", "pid",
                    "proc_start_ticks", "gpu_id", "heartbeat_at",
                )
            }
            for task in active
        ],
        "next_task": task_state.get("next_task"),
        "authority_sources": {
            "receipt": str(RECEIPT_PATH),
            "authority": str(AUTHORITY_PATH),
            "task_state": str(TASK_STATE_PATH),
            "human_status": str(STATUS_PATH),
            "doc_authority_map": str(AUTHORITY_PATH.parent / "DOC_AUTHORITY_MAP.json"),
            "algorithm_contract": str(AUTHORITY_PATH.parent / "ALGORITHM_CONTRACT.json"),
        },
        "claim_limit": "Derived bounded context only; validate CURRENT_STATUS_RECEIPT.json before use.",
    }


def render_rc1_status_min(
    authority: Mapping[str, Any], task_state: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the bounded RC1-only context card from the current task ledger."""

    rc1_tasks = [
        task for task in task_state.get("tasks", [])
        if str(task.get("task_id", "")).startswith("rc1_")
    ]
    counts: dict[str, int] = {}
    for task in rc1_tasks:
        status = str(task.get("status"))
        counts[status] = counts.get(status, 0) + 1
    active = [
        str(task.get("task_id"))
        for task in rc1_tasks
        if task.get("status") in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
    ]
    next_task = task_state.get("next_task")
    next_id = str(next_task.get("task_id")) if isinstance(next_task, Mapping) else None
    if next_id is not None and not next_id.startswith("rc1_"):
        next_id = None
    return {
        "schema_version": "chaoyang-current-rc1-status-min-v1",
        "governance_revision": authority["governance_revision"],
        "generation_id": authority["generation_id"],
        "generated_at": authority["generated_at"],
        "freshness": freshness(task_state),
        "release_id": "chaoyang-visual-aux-rc1-final",
        "cohort": "exact78-chips78-poker78-frozen",
        "task_counts": counts,
        "active_task_ids": active,
        "next_task_id": next_id,
        "pair_status": task_state.get("rc1_pair_status", {
            "chips": "NOT_EVALUATED",
            "poker": "NOT_EVALUATED",
        }),
        "release_flags": task_state.get("rc1_release_flags", {
            "PIPELINE_OPERATIONALLY_CLOSED": False,
            "DATA_MINIMUM_MET_CHIPS": False,
            "DATA_MINIMUM_MET_POKER": False,
            "FOUR_CHECKPOINTS_TRAINED": False,
            "RATE_FINALIZED": False,
            "RC1_RELEASE_STATUS": "NOT_STARTED",
        }),
        "budget": task_state.get("rc1_budget", {
            "engineering_hours_limit": 24,
            "production_gpu_hours_limit": 96,
            "checkpoint_gpu_hours_limit": 48,
        }),
        "sources": {
            "receipt": str(RECEIPT_PATH),
            "plan": str(RC1_PLAN_PATH),
            "release_spec": str(RC1_RELEASE_SPEC_PATH),
            "contract": str(RC1_CONTRACT_PATH),
            "task_dag": str(RC1_TASK_DAG_PATH),
        },
        "claim_limit": "RC1 bounded context only; validate the current receipt before use and never fill missing facts from chat history.",
    }


def render_task_queue(task_state: Mapping[str, Any]) -> dict[str, Any]:
    """Render a machine-readable queue without inventing scheduling authority."""
    order = {
        "RUNNING": 0,
        "CLAIMED": 1,
        "WAIT_GPU_RESOURCE": 2,
        "PENDING": 3,
        "BLOCKED_PREREQ": 4,
        "BLOCKED_RESOURCE": 5,
        "BLOCKED_EXTERNAL": 6,
        "FAILED_QUALITY_C": 7,
        "REJECTED_QUALITY": 8,
        "FAILED_RUNTIME_FINAL": 9,
        "PASSED": 10,
        "CANCELLED": 11,
    }
    tasks = sorted(
        task_state.get("tasks", []),
        key=lambda item: (order.get(str(item.get("status")), 99), str(item.get("task_id"))),
    )
    return {
        "schema_version": "chaoyang-task-queue-v1",
        "governance_revision": task_state["governance_revision"],
        "generation_id": task_state["generation_id"],
        "generated_at": task_state["generated_at"],
        "tasks": [
            {
                key: item.get(key)
                for key in (
                    "task_id", "phase", "status", "session", "attempt",
                    "gpu_id", "updated_at", "heartbeat_at",
                )
            }
            for item in tasks
        ],
        "next_task": task_state.get("next_task"),
        "claim_limit": "Queue projection only; task packets and current receipt are authoritative.",
    }


def validate_authority(authority: Mapping[str, Any], verify_evidence: bool = True) -> list[str]:
    errors: list[str] = []
    validate_schema("authority_index.schema.json", authority)
    for claim in authority.get("claims", []):
        if claim["status"] not in CLAIM_STATUSES:
            errors.append(f"invalid claim status: {claim['status']}")
    if verify_evidence:
        for stage in authority.get("stages", []):
            for ref in stage.get("evidence", []):
                errors.extend(validate_artifact_ref(ref))
        for claim in authority.get("claims", []):
            for ref in claim.get("evidence", []):
                errors.extend(validate_artifact_ref(ref))
    return errors


def validate_task_state(task_state: Mapping[str, Any]) -> list[str]:
    tasks = list(task_state.get("tasks", []))
    errors = [
        f"invalid task status: {task['status']}"
        for task in tasks
        if task["status"] not in TASK_STATUSES
    ]
    if errors:
        return errors
    validate_schema("task_state.schema.json", task_state)
    task_by_id = {str(task.get("task_id")): task for task in tasks}
    next_task = task_state.get("next_task")
    if next_task is not None:
        next_task_id = str(next_task.get("task_id", ""))
        selected = task_by_id.get(next_task_id)
        if selected is None:
            errors.append(f"next_task does not exist in task state: {next_task_id}")
        elif selected.get("status") in TERMINAL_TASK_STATUSES:
            errors.append(
                f"next_task points to terminal task: {next_task_id}={selected.get('status')}"
            )
    return errors


def append_changelog(event: Mapping[str, Any]) -> None:
    previous_sha = None
    if CHANGELOG_PATH.is_file():
        lines = [line for line in CHANGELOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            previous_sha = json.loads(lines[-1]).get("event_sha256")
    payload = dict(event)
    payload["previous_event_sha256"] = previous_sha
    payload["event_sha256"] = sha256_bytes(canonical_bytes(payload))
    descriptor = os.open(CHANGELOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        with os.fdopen(descriptor, "ab", closefd=False) as handle:
            handle.write(canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def publish_bundle(
    authority: dict[str, Any],
    task_state: dict[str, Any],
    *,
    event_type: str,
    expected_revision: int | None,
    generator_path: Path,
    append_event: bool = True,
    plan_revision_value: Mapping[str, Any] | None = None,
    plan_migration_value: Mapping[str, Any] | None = None,
    task_packet_index_path: Path | None = None,
    task_packet_index_value: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    with governance_lock():
        doc_authority_map_path = AUTHORITY_PATH.parent / "DOC_AUTHORITY_MAP.json"
        algorithm_contract_path = AUTHORITY_PATH.parent / "ALGORITHM_CONTRACT.json"
        plan_migration_receipt_path = AUTHORITY_PATH.parent / "PLAN_MIGRATION_RECEIPT.json"
        task_packet_pointer_path = AUTHORITY_PATH.parent / "CURRENT_V71_TASK_PACKET_INDEX.json"
        current_revision = 0
        if RECEIPT_PATH.is_file():
            current_revision = int(load_json(RECEIPT_PATH)["governance_revision"])
        if expected_revision is not None and expected_revision != current_revision:
            raise RuntimeError(f"CAS revision mismatch: expected={expected_revision} current={current_revision}")
        revision = current_revision + 1
        generation_id = f"gov-{revision:06d}-{uuid.uuid4().hex[:12]}"
        created = now_iso()
        generator_sha = sha256_file(generator_path)
        repository = git_metadata()
        for value in (authority, task_state):
            value["governance_revision"] = revision
            value["generation_id"] = generation_id
            value["generated_at"] = created
            value["generator_code_sha"] = generator_sha
        authority["repository"] = repository
        task_state["repository"] = repository
        task_state["host"] = socket.gethostname()
        if plan_revision_value is not None:
            plan_revision = dict(plan_revision_value)
            plan_revision.update(
                governance_revision=revision,
                generation_id=generation_id,
                generated_at=created,
            )
            atomic_json(PLAN_REVISION_PATH, plan_revision)
        if plan_migration_value is not None:
            plan_migration = dict(plan_migration_value)
            plan_migration.update(
                effective_governance_revision=revision,
                generation_id=generation_id,
                generated_at=created,
            )
            validate_schema("plan_migration_receipt.schema.json", plan_migration)
            atomic_json(plan_migration_receipt_path, plan_migration)
        if (task_packet_index_path is None) != (task_packet_index_value is None):
            raise RuntimeError("task packet index path and value must be supplied together")
        if task_packet_index_path is not None and task_packet_index_value is not None:
            packet_index = dict(task_packet_index_value)
            packet_index.update(
                governance_revision=revision,
                generation_id=generation_id,
                generated_at=created,
            )
            atomic_json(task_packet_index_path, packet_index)
            index_ref = artifact_ref(task_packet_index_path)
            pointer = {
                "schema_version": "chaoyang-v71-task-packet-pointer-v2",
                "packet_revision": packet_index["packet_revision"],
                "plan_revision": packet_index["plan_revision"],
                "execution_revision": packet_index.get("execution_revision", "R3"),
                "governance_revision": revision,
                "generation_id": generation_id,
                "generated_at": created,
                "index_path": str(task_packet_index_path.resolve().relative_to(REPO_ROOT)),
                "index_bytes": index_ref["bytes"],
                "index_sha256": index_ref["sha256"],
            }
            atomic_json(task_packet_pointer_path, pointer)
        elif task_packet_pointer_path.is_file():
            pointer = load_json(task_packet_pointer_path)
            pointer.update(
                governance_revision=revision,
                generation_id=generation_id,
                generated_at=created,
            )
            atomic_json(task_packet_pointer_path, pointer)
        errors = validate_authority(authority) + validate_task_state(task_state)
        if errors:
            raise RuntimeError("governance validation failed:\n" + "\n".join(errors))
        from chaoyang.governance.current_baseline_v2 import build_layout, build_registry, build_regression_manifest, render_stage_doc

        baseline_registry = build_registry(authority, task_state)
        file_layout = build_layout(authority, task_state)
        regression_manifest = build_regression_manifest(authority)
        stage_baselines = render_stage_doc(baseline_registry).encode("utf-8")
        markdown = render_status(authority, task_state).encode("utf-8")
        min_status = render_min_status(authority, task_state)
        task_queue = render_task_queue(task_state)
        atomic_json(AUTHORITY_PATH, authority)
        atomic_json(TASK_STATE_PATH, task_state)
        from chaoyang.governance.build_algorithm_baseline_v2_status import publish_navigation_if_v2
        publish_navigation_if_v2(REPO_ROOT, task_state, authority["generated_at"])
        # V2 shallow navigation is a bounded snapshot, not a quality promotion.
        atomic_write(STATUS_PATH, markdown)
        atomic_json(MIN_STATUS_PATH, min_status)
        if RC1_PLAN_PATH.is_file() and RC1_RELEASE_SPEC_PATH.is_file() and RC1_CONTRACT_PATH.is_file():
            atomic_json(RC1_STATUS_MIN_PATH, render_rc1_status_min(authority, task_state))
        atomic_json(TASK_QUEUE_PATH, task_queue)
        atomic_json(BASELINE_REGISTRY_PATH, baseline_registry)
        atomic_write(STAGE_BASELINES_PATH, stage_baselines)
        atomic_json(FILE_LAYOUT_PATH, file_layout)
        atomic_json(REGRESSION_MANIFEST_PATH, regression_manifest)
        from chaoyang.governance.current_r3_contracts import (
            build_algorithm_contract,
            build_doc_authority_map,
            validate_doc_authority_map,
        )

        algorithm_contract = build_algorithm_contract(baseline_registry, authority)
        validate_schema("algorithm_contract.schema.json", algorithm_contract)
        atomic_json(algorithm_contract_path, algorithm_contract)
        doc_authority_map = build_doc_authority_map(authority, AUTHORITY_PATH.parent, REPO_ROOT)
        validate_schema("doc_authority_map.schema.json", doc_authority_map)
        doc_errors = validate_doc_authority_map(doc_authority_map)
        if doc_errors:
            raise RuntimeError("document authority validation failed:\n" + "\n".join(doc_errors))
        atomic_json(doc_authority_map_path, doc_authority_map)
        receipt_files = {
            "authority_index": artifact_ref(AUTHORITY_PATH),
            "task_state": artifact_ref(TASK_STATE_PATH),
            "project_status": artifact_ref(STATUS_PATH),
            "project_status_min": artifact_ref(MIN_STATUS_PATH),
            "task_queue": artifact_ref(TASK_QUEUE_PATH),
            "baseline_registry_v2": artifact_ref(BASELINE_REGISTRY_PATH),
            "stage_baselines": artifact_ref(STAGE_BASELINES_PATH),
            "file_layout": artifact_ref(FILE_LAYOUT_PATH),
            "regression_manifest": artifact_ref(REGRESSION_MANIFEST_PATH),
            "doc_authority_map": artifact_ref(doc_authority_map_path),
            "algorithm_contract": artifact_ref(algorithm_contract_path),
        }
        if PLAN_REVISION_PATH.is_file():
            receipt_files["plan_revision"] = artifact_ref(PLAN_REVISION_PATH)
        if task_packet_pointer_path.is_file():
            receipt_files["task_packet_index_v71"] = artifact_ref(task_packet_pointer_path)
            pointer = load_json(task_packet_pointer_path)
            pointed_index = Path(str(pointer["index_path"]))
            if not pointed_index.is_absolute():
                pointed_index = REPO_ROOT / pointed_index
            pointed_ref = artifact_ref(pointed_index)
            if pointer.get("index_bytes", pointed_ref["bytes"]) != pointed_ref["bytes"]:
                raise RuntimeError("current task packet index pointer bytes mismatch")
            if pointer.get("index_sha256") != pointed_ref["sha256"]:
                raise RuntimeError("current task packet index pointer SHA mismatch")
            receipt_files["task_packet_index_current"] = pointed_ref
        if plan_migration_receipt_path.is_file():
            receipt_files["plan_migration_receipt"] = artifact_ref(plan_migration_receipt_path)
        for key, path in (
            ("rc1_plan", RC1_PLAN_PATH),
            ("rc1_release_spec", RC1_RELEASE_SPEC_PATH),
            ("visual_aux_rc1_contract", RC1_CONTRACT_PATH),
            ("rc1_task_dag", RC1_TASK_DAG_PATH),
            ("rc1_status_min", RC1_STATUS_MIN_PATH),
        ):
            if path.is_file():
                receipt_files[key] = artifact_ref(path)
        receipt = {
            "schema_version": "chaoyang-current-status-receipt-v1",
            "governance_revision": revision,
            "generation_id": generation_id,
            "generated_at": created,
            "freshness": freshness(task_state),
            "files": receipt_files,
        }
        validate_schema("current_status_receipt.schema.json", receipt)
        atomic_json(RECEIPT_PATH, receipt)
        if append_event:
            append_changelog({
                "schema_version": "chaoyang-state-change-event-v1",
                "created_at": created,
                "event_type": event_type,
                "governance_revision": revision,
                "generation_id": generation_id,
                "receipt_sha256": sha256_file(RECEIPT_PATH),
            })
        return receipt
