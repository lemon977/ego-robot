from __future__ import annotations

import hashlib
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
PATH = TOOLS / "autonomous_20h_supervisor.py"
SPEC = importlib.util.spec_from_file_location("autonomous_20h_supervisor_tested", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def task(channel: str = "GPU", task_id: str | None = None) -> dict:
    return {
        "task_id": task_id or ("GPU-X" if channel == "GPU" else "CPU-X"),
        "channel": channel,
        "priority": 1,
        "budget_minutes": 10,
        "backup": False,
        "dependencies": [],
    }


def base_row(
    channel: str, task_id: str, *, dependencies: list[str] | None = None
) -> dict:
    return {
        "schema_version": "autonomous-20h-queue-v1",
        "queued_at": "2026-08-29T12:00:00+08:00",
        "task_id": task_id,
        "channel": channel,
        "priority": 1,
        "budget_minutes": 10,
        "backup": False,
        "dependencies": dependencies or [],
        "command_manifest": None,
        "command_manifest_bytes": None,
        "command_manifest_sha256": None,
    }


def binding_row(channel: str, task_id: str) -> dict:
    return {
        "schema_version": "autonomous-20h-queue-binding-v1",
        "queued_at": "2026-08-29T12:00:00+08:00",
        "task_id": task_id,
        "channel": channel,
        "command_manifest": str(MODULE.COMMAND_ROOT / f"{task_id}.json"),
        "command_manifest_bytes": 1,
        "command_manifest_sha256": "0" * 64,
    }


def event_row(event: str, task_id: str = "CPU-X", **fields) -> dict:
    return {
        "schema_version": "autonomous-20h-event-v1",
        "timestamp": "2026-08-29T12:00:00+08:00",
        "event": event,
        "task_id": task_id,
        "channel": task_id.split("-", 1)[0],
        **fields,
    }


def use_synthetic_event_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MODULE, "_verify_frozen_legacy_event_prefix", lambda _rows, _payload: None
    )


def set_event_rows(monkeypatch: pytest.MonkeyPatch, rows: list[dict]) -> None:
    monkeypatch.setattr(
        MODULE, "_jsonl_snapshot", lambda _path: (rows, b"synthetic-event-bytes")
    )


def valid_manifest(channel: str = "GPU") -> dict:
    allowed = str(ROOT / "archive/legacy_runs/unclassified/AUTONOMOUS_20H_20260829/checkpoints/ref.json")
    evidence_ref = {"path": allowed, "bytes": 1, "sha256": "0" * 64}
    manifest = {
        "schema_version": "autonomous-20h-command-v2",
        "task_id": "GPU-X" if channel == "GPU" else "CPU-X",
        "channel": channel,
        "command": [sys.executable, "-c", "pass"],
        "cwd": str(ROOT),
        "env": {"CUDA_VISIBLE_DEVICES": ""} if channel == "CPU" else {},
        "completion_mode": "EVIDENCE",
        "success_evidence": [evidence_ref],
        "classification": "B_TOOL",
        "a_class_exit_codes": [42],
    }
    if channel == "CPU":
        spec = {
            "refs": [
                {
                    **evidence_ref,
                    "json_assertions": [
                        {"pointer": "/schema_version", "equals": "fixture-v1"}
                    ],
                }
            ]
        }
        manifest["command"] = [
            "/usr/bin/python3",
            "-I",
            "-B",
            "-c",
            MODULE.EXACT_REFERENCE_VERIFIER_CODE,
            json.dumps(spec, sort_keys=True, separators=(",", ":")),
        ]
        manifest["completion_mode"] = "EVIDENCE"
        manifest["success_evidence"] = [evidence_ref]
        manifest["admission"] = {
            "authorization_tier": "T0_CONTROL_PLANE_CPU_ONLY",
            "execution_scope": "CONTROL_PLANE_METADATA_ONLY",
            "gpu_execution_admitted": False,
            "pixel_or_selector_execution_admission": 0,
            "a_class_holds_released": False,
            "allowed_input_paths": [allowed],
            "release_artifact": None,
        }
    else:
        manifest["admission"] = {
            "authorization_tier": "OWNER_AUTHORIZED_EXECUTION",
            "execution_scope": "AUTHORIZED_EXECUTION",
            "gpu_execution_admitted": True,
            "pixel_or_selector_execution_admission": 1,
            "a_class_holds_released": True,
            "allowed_input_paths": [allowed],
            "release_artifact": {
                "path": str(
                    ROOT / "archive/legacy_runs/unclassified/AUTONOMOUS_20H_20260829/checkpoints/release.json"
                ),
                "bytes": 1,
                "sha256": "0" * 64,
            },
        }
    return manifest


def test_authorization_and_shared_io_pins_are_live() -> None:
    pins = MODULE._verify_bootstrap_pins()
    assert pins["authorization"]["sha256"] == MODULE.AUTHORIZATION_SHA256
    assert pins["shared_io"]["sha256"] == MODULE.SHARED_IO_SHA256


def test_command_manifest_exact_schema() -> None:
    value = valid_manifest()
    assert MODULE._validate_command_manifest(value, task()) == value
    value["attacker"] = True
    with pytest.raises(MODULE.SupervisorError):
        MODULE._validate_command_manifest(value, task())


@pytest.mark.parametrize(
    "key,value",
    [
        ("cwd", "/tmp"),
        ("command", []),
        ("env", {"HOME": "/tmp"}),
        ("completion_mode", "PASS_PREFIX"),
        ("classification", "UNKNOWN"),
        ("a_class_exit_codes", [True]),
    ],
)
def test_command_manifest_rejects_fail_open_values(key: str, value: object) -> None:
    manifest = valid_manifest()
    manifest[key] = value
    with pytest.raises(MODULE.SupervisorError):
        MODULE._validate_command_manifest(manifest, task())


def test_evidence_completion_rehashes_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    evidence = tmp_path / "result"
    evidence.write_bytes(b"ok")
    manifest = valid_manifest()
    manifest["completion_mode"] = "EVIDENCE"
    manifest["success_evidence"] = [
        {
            "path": str(evidence),
            "bytes": 2,
            "sha256": hashlib.sha256(b"ok").hexdigest(),
        }
    ]
    assert MODULE._evidence_complete(manifest) is True
    evidence.write_bytes(b"drift")
    assert MODULE._evidence_complete(manifest) is False


def test_initial_queues_have_unique_ids_and_dependencies() -> None:
    ids = [row[0] for row in MODULE.INITIAL_GPU] + [
        row[0] for row in MODULE.INITIAL_CPU
    ]
    assert len(ids) == len(set(ids)) == 21
    known = set(ids)
    for row in MODULE.INITIAL_GPU:
        assert set(row[4]) <= known
    for row in MODULE.INITIAL_CPU:
        assert set(row[3]) <= known


def test_backup_is_not_ready_before_ten_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = task()
    queued.update(
        {
            "task_id": "GPU-BACKUP",
            "backup": True,
            "command_manifest": "x",
            "command_manifest_sha256": "0" * 64,
            "command_manifest_bytes": 1,
        }
    )
    monkeypatch.setattr(MODULE, "_manifest_for", lambda _task: valid_manifest())
    assert MODULE._ready_tasks({"GPU-BACKUP": queued}, {}, {}, "GPU", 599, {}) == []
    assert len(MODULE._ready_tasks({"GPU-BACKUP": queued}, {}, {}, "GPU", 600, {})) == 1


def test_b_retry_wait_is_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    queued = task("CPU")
    queued.update(
        {
            "command_manifest": "x",
            "command_manifest_sha256": "0" * 64,
            "command_manifest_bytes": 1,
        }
    )
    monkeypatch.setattr(MODULE, "_manifest_for", lambda _task: valid_manifest("CPU"))
    assert (
        MODULE._ready_tasks(
            {"CPU-X": queued}, {"CPU-X": "B_RETRY_WAIT"}, {}, "CPU", 0, {}
        )
        == []
    )


@pytest.mark.parametrize("state", ["RUNNING", "LAUNCHING"])
def test_untracked_running_or_launching_state_is_never_restarted(
    state: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = task("CPU")
    queued.update(binding_row("CPU", "CPU-X"))
    monkeypatch.setattr(
        MODULE, "_manifest_for", lambda _task: pytest.fail("must not reload or start")
    )
    assert (
        MODULE._ready_tasks({"CPU-X": queued}, {"CPU-X": state}, {}, "CPU", 0, {}) == []
    )


def test_reopen_resets_attempts_after_b_tool_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_synthetic_event_prefix(monkeypatch)
    events = tmp_path / "EVENTS.jsonl"
    rows = [
        event_row(
            "CARD_STARTED",
            pid=1,
            attempt=1,
            budget_minutes=1,
            command_manifest_sha256="0" * 64,
            log_path=str(tmp_path / "1"),
        ),
        event_row(
            "CARD_B_RETRY_QUEUED",
            completed_attempt=1,
            retry_not_before_epoch=0,
        ),
        event_row(
            "CARD_STARTED",
            pid=2,
            attempt=2,
            budget_minutes=1,
            command_manifest_sha256="0" * 64,
            log_path=str(tmp_path / "2"),
        ),
        event_row(
            "CARD_B_RETRY_QUEUED",
            completed_attempt=2,
            retry_not_before_epoch=0,
        ),
        event_row(
            "CARD_STARTED",
            pid=3,
            attempt=3,
            budget_minutes=1,
            command_manifest_sha256="0" * 64,
            log_path=str(tmp_path / "3"),
        ),
        event_row("CARD_DEGRADED_A_RETRY_EXHAUSTED", attempts=3, returncode=1),
        event_row(
            "CARD_B_REPAIR_REOPENED",
            previous_attempts=3,
            reset_attempts_to=0,
            reason="fixed B tool",
        ),
    ]
    events.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(MODULE, "EVENTS", events)
    monkeypatch.setattr(MODULE, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "LOG_ROOT", tmp_path)
    status, attempts, _ = MODULE._event_state()
    assert status["CPU-X"] == "PENDING"
    assert attempts["CPU-X"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(extra=True),
        lambda row: row.pop("backup"),
        lambda row: row.update(priority=True),
        lambda row: row.update(budget_minutes=0),
        lambda row: row.update(backup=0),
        lambda row: row.update(dependencies=["CPU-Y", "CPU-Y"]),
        lambda row: row.update(dependencies=["CPU-X"]),
        lambda row: row.update(command_manifest="/tmp/x"),
    ],
)
def test_validate_base_row_is_exact_and_typed(mutation) -> None:
    row = base_row("CPU", "CPU-X")
    mutation(row)
    with pytest.raises(MODULE.ControlPlaneBError):
        MODULE._validate_base_row(row, expected_channel="CPU")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(extra=True),
        lambda row: row.pop("command_manifest_bytes"),
        lambda row: row.update(channel="GPU"),
        lambda row: row.update(command_manifest="relative.json"),
        lambda row: row.update(command_manifest="/tmp/x.json"),
        lambda row: row.update(command_manifest_bytes=True),
        lambda row: row.update(command_manifest_sha256="A" * 64),
    ],
)
def test_validate_binding_row_is_exact_and_typed(mutation) -> None:
    row = binding_row("CPU", "CPU-X")
    mutation(row)
    with pytest.raises(MODULE.ControlPlaneBError):
        MODULE._validate_binding_row(row, expected_channel="CPU")


def test_queue_graph_rejects_duplicates_unknown_dependencies_and_cycles() -> None:
    cpu = base_row("CPU", "CPU-X")
    with pytest.raises(MODULE.ControlPlaneBError, match="duplicate queue base"):
        MODULE._merge_queue_rows([("CPU", cpu), ("CPU", dict(cpu))])
    unknown = base_row("CPU", "CPU-X", dependencies=["GPU-Y"])
    with pytest.raises(MODULE.ControlPlaneBError, match="unknown dependencies"):
        MODULE._merge_queue_rows([("CPU", unknown)])
    gpu = base_row("GPU", "GPU-Y", dependencies=["CPU-X"])
    cpu = base_row("CPU", "CPU-X", dependencies=["GPU-Y"])
    with pytest.raises(MODULE.ControlPlaneBError, match="cycle"):
        MODULE._merge_queue_rows([("GPU", gpu), ("CPU", cpu)])


def test_duplicate_binding_is_rejected_even_when_identical() -> None:
    base = base_row("CPU", "CPU-X")
    binding = binding_row("CPU", "CPU-X")
    with pytest.raises(MODULE.ControlPlaneBError, match="duplicate immutable"):
        MODULE._merge_queue_rows(
            [("CPU", base), ("CPU", binding), ("CPU", dict(binding))]
        )


def test_enqueue_validates_full_graph_before_one_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = [("CPU", base_row("CPU", "CPU-X"))]
    appended: list[dict] = []
    monkeypatch.setattr(MODULE, "initialize", lambda: None)
    monkeypatch.setattr(MODULE, "_queue_row_pairs", lambda: current)
    monkeypatch.setattr(MODULE, "_append_queue", lambda row: appended.append(dict(row)))
    monkeypatch.setattr(
        MODULE, "_advisory_lock", lambda *_args, **_kwargs: nullcontext()
    )
    MODULE.enqueue_card(
        "CPU-Y",
        "CPU",
        2,
        5,
        dependencies=["CPU-X"],
    )
    assert len(appended) == 1
    assert set(appended[0]) == MODULE.QUEUE_BASE_KEYS
    with pytest.raises(MODULE.ControlPlaneBError):
        MODULE.enqueue_card("CPU-X", "CPU", 2, 5)
    assert len(appended) == 1


def test_current_live_queue_prefix_remains_loadable_under_successor_schema() -> None:
    tasks = MODULE.load_tasks()
    assert len(tasks) == 21
    assert tasks["CPU-2-D4-CONTACT-COUNTERFACTUAL"]["command_manifest_sha256"] == (
        "eb4721e283831be641dabf60df4d486d7d05ef2eb0a41ac0f9d6d2611b385090"
    )


def test_current_live_event_prefix_replays_under_successor_schema() -> None:
    tasks = MODULE.load_tasks()
    statuses, attempts, rows = MODULE._event_state(tasks)
    assert len(rows) == 32
    assert sum(status == "COMPLETED" for status in statuses.values()) == 5
    assert sum(status == "DEGRADED_A" for status in statuses.values()) == 5
    assert attempts["CPU-4-TOOL-CONSISTENCY"] == 1


def test_frozen_legacy_prefix_rejects_same_length_wrong_raw_bytes() -> None:
    with pytest.raises(MODULE.ControlPlaneBError, match="bytes/SHA-256 mismatch"):
        MODULE._verify_frozen_legacy_event_prefix(
            [{} for _ in range(MODULE.LEGACY_EVENT_PREFIX_ROWS)],
            b"x" * MODULE.LEGACY_EVENT_PREFIX_BYTES,
        )


def test_event_replay_rejects_wrong_schema_illegal_completion_and_unknown_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_synthetic_event_prefix(monkeypatch)
    queued = {"CPU-X": task("CPU")}
    wrong_schema = event_row("CARD_COMPLETED", pid=1, attempt=1, elapsed_seconds=1.0)
    wrong_schema["schema_version"] = "attacker-v1"
    set_event_rows(monkeypatch, [wrong_schema])
    with pytest.raises(MODULE.ControlPlaneBError, match="schema version"):
        MODULE._event_state(queued)
    illegal = event_row("CARD_COMPLETED", pid=1, attempt=1, elapsed_seconds=1.0)
    set_event_rows(monkeypatch, [illegal])
    with pytest.raises(MODULE.ControlPlaneBError, match="illegal event transition"):
        MODULE._event_state(queued)
    unknown = event_row(
        "CARD_STARTED",
        "CPU-Y",
        pid=1,
        attempt=1,
        budget_minutes=1,
        command_manifest_sha256="0" * 64,
        log_path="/tmp/x",
    )
    set_event_rows(monkeypatch, [unknown])
    with pytest.raises(MODULE.ControlPlaneBError, match="unknown task"):
        MODULE._event_state(queued)
    wrong_type = event_row(
        "CARD_STARTED",
        pid=True,
        attempt=1,
        budget_minutes=1,
        command_manifest_sha256="0" * 64,
        log_path=str(MODULE.LOG_ROOT / "CPU-X.test.log"),
    )
    set_event_rows(monkeypatch, [wrong_type])
    with pytest.raises(
        MODULE.ControlPlaneBError, match="pid must be a positive integer"
    ):
        MODULE._event_state(queued)


def test_event_replay_binds_completion_to_started_pid_and_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_synthetic_event_prefix(monkeypatch)
    queued_task = task("CPU")
    queued_task.update(binding_row("CPU", "CPU-X"))
    queued = {"CPU-X": queued_task}
    started = event_row(
        "CARD_STARTED",
        pid=101,
        attempt=1,
        budget_minutes=1,
        command_manifest_sha256="0" * 64,
        log_path=str(MODULE.LOG_ROOT / "CPU-X.test.log"),
    )
    wrong_pid = event_row("CARD_COMPLETED", pid=202, attempt=1, elapsed_seconds=1.0)
    set_event_rows(monkeypatch, [started, wrong_pid])
    with pytest.raises(MODULE.ControlPlaneBError, match="completion identity"):
        MODULE._event_state(queued)

    wrong_attempt = event_row(
        "CARD_B_RETRY_QUEUED",
        completed_attempt=2,
        retry_not_before_epoch=0,
    )
    set_event_rows(monkeypatch, [started, wrong_attempt])
    with pytest.raises(MODULE.ControlPlaneBError, match="B retry identity"):
        MODULE._event_state(queued)


def test_event_replay_rejects_v2_manifest_sha_different_from_queue_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_synthetic_event_prefix(monkeypatch)
    queued_task = task("CPU")
    queued_task.update(binding_row("CPU", "CPU-X"))
    queued_task["command_manifest_sha256"] = "1" * 64
    nonce = "2" * 48
    rows = [
        event_row(
            "CARD_LAUNCH_INTENT",
            attempt=1,
            launch_nonce=nonce,
            budget_minutes=10,
            command_manifest_sha256="0" * 64,
            log_path=str(MODULE.LOG_ROOT / "CPU-X.test.log"),
        ),
        event_row(
            "CARD_STARTED",
            pid=101,
            pgid=101,
            attempt=1,
            launch_nonce=nonce,
            budget_minutes=10,
            command_manifest_sha256="0" * 64,
            log_path=str(MODULE.LOG_ROOT / "CPU-X.test.log"),
        ),
        event_row(
            "CARD_PAYLOAD_RELEASED",
            pid=101,
            pgid=101,
            attempt=1,
            launch_nonce=nonce,
            command_manifest_sha256="0" * 64,
        ),
        event_row("CARD_COMPLETED", pid=101, attempt=1, elapsed_seconds=1.0),
    ]
    set_event_rows(monkeypatch, rows)
    with pytest.raises(MODULE.ManifestEvidenceError, match="queue binding"):
        MODULE._event_state({"CPU-X": queued_task})


def v2_process_rows(completion: dict | None = None) -> list[dict]:
    nonce = "4" * 48
    manifest_sha256 = "0" * 64
    rows = [
        event_row(
            "CARD_LAUNCH_INTENT",
            attempt=1,
            launch_nonce=nonce,
            budget_minutes=10,
            command_manifest_sha256=manifest_sha256,
            log_path=str(MODULE.LOG_ROOT / "CPU-X.test.log"),
        ),
        event_row(
            "CARD_STARTED",
            pid=101,
            pgid=101,
            attempt=1,
            launch_nonce=nonce,
            budget_minutes=10,
            command_manifest_sha256=manifest_sha256,
            log_path=str(MODULE.LOG_ROOT / "CPU-X.test.log"),
        ),
        event_row(
            "CARD_PAYLOAD_RELEASED",
            pid=101,
            pgid=101,
            attempt=1,
            launch_nonce=nonce,
            command_manifest_sha256=manifest_sha256,
        ),
    ]
    if completion is None:
        completion = event_row(
            "CARD_COMPLETED",
            pid=101,
            pgid=101,
            attempt=1,
            launch_nonce=nonce,
            command_manifest_sha256=manifest_sha256,
            elapsed_seconds=1.0,
        )
    return [*rows, completion]


@pytest.mark.parametrize("mode", ["missing", "wrong"])
@pytest.mark.parametrize(
    "field,wrong_value",
    [
        ("command_manifest_sha256", "1" * 64),
        ("launch_nonce", "5" * 48),
        ("pgid", 202),
        ("pid", 202),
        ("attempt", 2),
    ],
)
def test_v2_completion_identity_tampering_fails_closed_before_dependency_unlock(
    field: str,
    wrong_value: object,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_synthetic_event_prefix(monkeypatch)
    parent = task("CPU", "CPU-X")
    parent.update(binding_row("CPU", "CPU-X"))
    dependent = task("CPU", "CPU-Y")
    dependent.update(binding_row("CPU", "CPU-Y"))
    dependent["dependencies"] = ["CPU-X"]
    queued = {"CPU-X": parent, "CPU-Y": dependent}
    completion = v2_process_rows()[-1]
    if mode == "missing":
        completion.pop(field)
    else:
        completion[field] = wrong_value
    set_event_rows(monkeypatch, v2_process_rows(completion))

    def replay_then_find_ready() -> list:
        statuses, attempts, _ = MODULE._event_state(queued)
        return MODULE._ready_tasks(queued, statuses, {}, "CPU", 0, attempts)

    monkeypatch.setattr(
        MODULE,
        "_manifest_for",
        lambda _task: pytest.fail("tampered completion must not unlock dependency"),
    )
    with pytest.raises(MODULE.SupervisorError):
        replay_then_find_ready()


def test_v2_completion_with_exact_process_identity_unlocks_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_synthetic_event_prefix(monkeypatch)
    parent = task("CPU", "CPU-X")
    parent.update(binding_row("CPU", "CPU-X"))
    dependent = task("CPU", "CPU-Y")
    dependent.update(binding_row("CPU", "CPU-Y"))
    dependent["dependencies"] = ["CPU-X"]
    queued = {"CPU-X": parent, "CPU-Y": dependent}
    set_event_rows(monkeypatch, v2_process_rows())
    statuses, attempts, _ = MODULE._event_state(queued)
    assert statuses["CPU-X"] == "COMPLETED"
    monkeypatch.setattr(MODULE, "_manifest_for", lambda _task: {"fixture": True})
    ready = MODULE._ready_tasks(queued, statuses, {}, "CPU", 0, attempts)
    assert [candidate[0]["task_id"] for candidate in ready] == ["CPU-Y"]


def test_legacy_started_is_confined_to_exact_frozen_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_rows, frozen_payload = MODULE._jsonl_snapshot(MODULE.EVENTS)
    assert len(frozen_rows) == MODULE.LEGACY_EVENT_PREFIX_ROWS
    queued = MODULE.load_tasks()
    synthetic = task("CPU")
    synthetic.update(binding_row("CPU", "CPU-X"))
    queued["CPU-X"] = synthetic
    rows = [
        *frozen_rows,
        event_row(
            "CARD_STARTED",
            pid=101,
            attempt=1,
            budget_minutes=10,
            command_manifest_sha256="0" * 64,
            log_path=str(MODULE.LOG_ROOT / "CPU-X.test.log"),
        ),
        event_row("CARD_COMPLETED", pid=101, attempt=1, elapsed_seconds=1.0),
    ]
    appended_payload = frozen_payload + b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        for row in rows[MODULE.LEGACY_EVENT_PREFIX_ROWS :]
    )
    monkeypatch.setattr(
        MODULE, "_jsonl_snapshot", lambda _path: (rows, appended_payload)
    )
    with pytest.raises(MODULE.ControlPlaneBError, match="after frozen prefix"):
        MODULE._event_state(queued)


def test_launch_intent_then_b_retry_replays_without_poisoning_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_synthetic_event_prefix(monkeypatch)
    queued_task = task("CPU")
    queued_task.update(binding_row("CPU", "CPU-X"))
    nonce = "3" * 48
    rows = [
        event_row(
            "CARD_LAUNCH_INTENT",
            attempt=1,
            launch_nonce=nonce,
            budget_minutes=10,
            command_manifest_sha256="0" * 64,
            log_path=str(MODULE.LOG_ROOT / "CPU-X.test.log"),
        ),
        event_row(
            "CARD_B_RETRY_QUEUED",
            completed_attempt=1,
            retry_not_before_epoch=0,
            launch_nonce=nonce,
            command_manifest_sha256="0" * 64,
            error="Popen failed",
        ),
    ]
    set_event_rows(monkeypatch, rows)
    statuses, attempts, _ = MODULE._event_state({"CPU-X": queued_task})
    assert statuses == {"CPU-X": "PENDING"}
    assert attempts == {"CPU-X": 1}


@pytest.mark.parametrize(
    "terminal",
    [
        event_row(
            "COMMAND_MANIFEST_REJECTED_A",
            command_manifest_sha256="0" * 64,
            error="SHA mismatch",
        ),
        event_row("CARD_DEGRADED_A", returncode=42, attempt=1),
    ],
)
def test_explicit_a_terminal_cannot_be_reopened(
    terminal: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_synthetic_event_prefix(monkeypatch)
    rows = []
    if terminal["event"] == "CARD_DEGRADED_A":
        rows.append(
            event_row(
                "CARD_STARTED",
                pid=101,
                attempt=1,
                budget_minutes=10,
                command_manifest_sha256="0" * 64,
                log_path=str(MODULE.LOG_ROOT / "CPU-X.test.log"),
            )
        )
    rows.extend(
        [
            terminal,
            event_row(
                "CARD_B_REPAIR_REOPENED",
                previous_attempts=1 if rows else 0,
                reset_attempts_to=0,
                reason="not actually B",
            ),
        ]
    )
    set_event_rows(monkeypatch, rows)
    with pytest.raises(MODULE.ControlPlaneBError, match="not eligible"):
        MODULE._event_state()


def test_reopen_command_rejects_explicit_a_terminal_without_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = {"CPU-X": task("CPU")}
    rows = [
        event_row(
            "COMMAND_MANIFEST_REJECTED_A",
            command_manifest_sha256="0" * 64,
            error="SHA mismatch",
        )
    ]
    appended: list[str] = []
    monkeypatch.setattr(MODULE, "initialize", lambda: None)
    monkeypatch.setattr(MODULE, "load_tasks", lambda: queued)
    monkeypatch.setattr(
        MODULE, "_event_state", lambda *_args: ({"CPU-X": "DEGRADED_A"}, {}, rows)
    )
    monkeypatch.setattr(
        MODULE, "_advisory_lock", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(
        MODULE, "_append_event", lambda event, **_fields: appended.append(event)
    )
    with pytest.raises(MODULE.SupervisorError, match="not eligible"):
        MODULE.reopen_card("CPU-X", "pretend B repair")
    assert appended == []


def test_jsonl_snapshot_retries_concurrent_growth(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "queue.jsonl"
    path.write_bytes(b"placeholder\n")
    calls = 0

    def fake_read(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise MODULE.strict_io.StrictIOError(
                "file size changed during same-fd read: test"
            )
        return SimpleNamespace(payload=b'{"ok":true}\n')

    monkeypatch.setattr(MODULE.strict_io, "read_bytes_nofollow", fake_read)
    monkeypatch.setattr(
        MODULE, "_advisory_lock", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)
    assert MODULE._jsonl(path) == [{"ok": True}]
    assert calls == 2


def test_jsonl_snapshot_retries_only_trailing_partial(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "queue.jsonl"
    path.write_bytes(b"placeholder\n")
    payloads = iter((b'{"ok":true}\n{"task"', b'{"ok":true}\n{"task":1}\n'))
    monkeypatch.setattr(
        MODULE.strict_io,
        "read_bytes_nofollow",
        lambda *_args, **_kwargs: SimpleNamespace(payload=next(payloads)),
    )
    monkeypatch.setattr(
        MODULE, "_advisory_lock", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)
    assert MODULE._jsonl(path) == [{"ok": True}, {"task": 1}]


def test_jsonl_durable_malformed_is_fail_closed_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "queue.jsonl"
    path.write_bytes(b"placeholder\n")
    calls = 0

    def fake_read(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(payload=b'{"broken":}\n')

    monkeypatch.setattr(MODULE.strict_io, "read_bytes_nofollow", fake_read)
    monkeypatch.setattr(
        MODULE, "_advisory_lock", lambda *_args, **_kwargs: nullcontext()
    )
    with pytest.raises(MODULE.ControlPlaneBError, match="invalid durable"):
        MODULE._jsonl(path)
    assert calls == 1


def test_jsonl_transient_retry_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "queue.jsonl"
    path.write_bytes(b"placeholder\n")
    calls = 0

    def fake_read(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise MODULE.strict_io.StrictIOError(
            "file size changed during same-fd read: test"
        )

    monkeypatch.setattr(MODULE.strict_io, "read_bytes_nofollow", fake_read)
    monkeypatch.setattr(
        MODULE, "_advisory_lock", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(MODULE.time, "sleep", lambda _seconds: None)
    with pytest.raises(MODULE.ControlPlaneBError, match="remained incomplete"):
        MODULE._jsonl(path)
    assert calls == MODULE.JSONL_SNAPSHOT_ATTEMPTS


def test_concurrent_append_and_read_yield_only_complete_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "queue.jsonl"
    path.write_bytes(b"")
    monkeypatch.setattr(MODULE, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "CONTROL_PLANE_IO_LOCK", tmp_path / "io.lock")
    snapshots: list[list[dict]] = []
    start = threading.Barrier(2)

    def writer() -> None:
        start.wait()
        for number in range(40):
            MODULE._append_jsonl(path, {"number": number})

    thread = threading.Thread(target=writer)
    thread.start()
    start.wait()
    while thread.is_alive():
        snapshots.append(MODULE._jsonl(path))
    thread.join(timeout=5)
    final = MODULE._jsonl(path)
    assert [row["number"] for row in final] == list(range(40))
    assert all(snapshot == final[: len(snapshot)] for snapshot in snapshots)


def test_control_plane_retry_exhaustion_is_terminal_and_not_flooded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    clock = iter((0.0, 61.0, 122.0, 183.0, 244.0))
    MODULE._CONTROL_PLANE_FAULTS.clear()
    monkeypatch.setattr(MODULE.time, "time", lambda: next(clock))
    monkeypatch.setattr(
        MODULE, "_append_event", lambda event, **_fields: events.append(event)
    )
    error = MODULE.ControlPlaneBError("durable defect")
    for _ in range(5):
        MODULE._record_control_plane_fault("QUEUES", error)
    assert events == [
        "CONTROL_PLANE_B_RETRY_QUEUED",
        "CONTROL_PLANE_B_RETRY_QUEUED",
        "CONTROL_PLANE_DEGRADED_A_RETRY_EXHAUSTED",
    ]
    MODULE._CONTROL_PLANE_FAULTS.clear()


@pytest.mark.parametrize("state", ["RUNNING", "COMPLETED", "SUSPENDED", "DEGRADED_A"])
def test_bind_rejects_bound_running_or_terminal_without_append(
    state: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = task("CPU")
    monkeypatch.setattr(MODULE, "initialize", lambda: None)
    monkeypatch.setattr(
        MODULE, "_advisory_lock", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(MODULE, "load_tasks", lambda: {"CPU-X": queued})
    monkeypatch.setattr(
        MODULE, "_event_state", lambda *_args: ({"CPU-X": state}, {}, [])
    )
    monkeypatch.setattr(
        MODULE, "_append_queue", lambda _row: pytest.fail("must not append")
    )
    with pytest.raises(MODULE.ControlPlaneBError, match="running or terminal"):
        MODULE.bind_manifest("CPU-X", Path("/does/not/matter"))


def test_bind_rejects_existing_binding_before_manifest_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = task("CPU")
    queued.update(binding_row("CPU", "CPU-X"))
    monkeypatch.setattr(MODULE, "initialize", lambda: None)
    monkeypatch.setattr(
        MODULE, "_advisory_lock", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(MODULE, "load_tasks", lambda: {"CPU-X": queued})
    monkeypatch.setattr(
        MODULE.strict_io,
        "read_bytes_nofollow",
        lambda *_args, **_kwargs: pytest.fail("must not read a replacement manifest"),
    )
    with pytest.raises(MODULE.ControlPlaneBError, match="already has an immutable"):
        MODULE.bind_manifest("CPU-X", Path("/does/not/matter"))


def test_recovery_rejects_started_manifest_sha_before_pid_or_manifest_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = task("CPU")
    queued.update(binding_row("CPU", "CPU-X"))
    row = {
        "event": "CARD_STARTED",
        "task_id": "CPU-X",
        "command_manifest_sha256": "1" * 64,
    }
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        MODULE,
        "_event_state",
        lambda *_args: ({"CPU-X": "RUNNING"}, {"CPU-X": 1}, [row]),
    )
    monkeypatch.setattr(
        MODULE,
        "_manifest_for",
        lambda *_args, **_kwargs: pytest.fail("must not read manifest"),
    )
    monkeypatch.setattr(
        MODULE, "_pid_alive", lambda _pid: pytest.fail("must not inspect PID")
    )
    monkeypatch.setattr(
        MODULE, "_append_event", lambda event, **fields: events.append((event, fields))
    )
    assert MODULE._recover_running_cards({"CPU-X": queued}) == {}
    assert [event for event, _ in events] == ["CARD_LINEAGE_REJECTED_A"]
    assert "does not match" in events[0][1]["error"]
    assert events[0][1]["command_manifest_sha256"] == "0" * 64
    assert events[0][1]["observed_command_manifest_identity"] == repr("1" * 64)


def test_recovered_dead_pid_never_completes_from_preexisting_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = task("CPU")
    queued.update(binding_row("CPU", "CPU-X"))
    nonce = "n" * 48
    started = event_row(
        "CARD_STARTED",
        pid=999999,
        pgid=999999,
        attempt=1,
        launch_nonce=nonce,
        budget_minutes=10,
        command_manifest_sha256="0" * 64,
        log_path="/tmp/x",
    )
    released = event_row(
        "CARD_PAYLOAD_RELEASED",
        pid=999999,
        pgid=999999,
        attempt=1,
        launch_nonce=nonce,
        command_manifest_sha256="0" * 64,
    )
    snapshot = ({"CPU-X": "RUNNING"}, {"CPU-X": 1}, [started, released])
    manifest = valid_manifest("CPU")
    events: list[str] = []
    monkeypatch.setattr(MODULE, "_manifest_for", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(MODULE, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        MODULE,
        "_evidence_complete",
        lambda _manifest: pytest.fail("must not infer completion"),
    )
    monkeypatch.setattr(
        MODULE, "_append_event", lambda event, **_fields: events.append(event)
    )
    assert MODULE._recover_running_cards({"CPU-X": queued}, snapshot) == {}
    assert events == ["CARD_B_RETRY_QUEUED"]


def test_activation_preflight_rejects_any_live_predecessor_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = {"CPU-X": task("CPU")}
    started = event_row(
        "CARD_STARTED",
        pid=123,
        attempt=1,
        budget_minutes=10,
        command_manifest_sha256="0" * 64,
        log_path="/tmp/x",
    )
    snapshot = ({"CPU-X": "RUNNING"}, {"CPU-X": 1}, [started])
    MODULE._CONTROL_PLANE_FAULTS.clear()
    monkeypatch.setattr(MODULE, "_pid_alive", lambda pid: pid == 123)
    monkeypatch.setattr(MODULE, "_manifest_for", lambda *_args, **_kwargs: None)
    with pytest.raises(
        MODULE.ControlPlaneBError,
        match="cannot (safely attach|reconstruct predecessor lineage)",
    ):
        MODULE._activation_preflight(queued, snapshot)


def test_activation_preflight_allows_identity_bound_v2_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued_task = task("CPU")
    queued_task.update(binding_row("CPU", "CPU-X"))
    queued = {"CPU-X": queued_task}
    nonce = "1" * 48
    started = event_row(
        "CARD_STARTED",
        pid=123,
        pgid=123,
        attempt=1,
        launch_nonce=nonce,
        budget_minutes=10,
        command_manifest_sha256="0" * 64,
        log_path=str(MODULE.LOG_ROOT / "CPU-X.test.log"),
    )
    released = event_row(
        "CARD_PAYLOAD_RELEASED",
        pid=123,
        pgid=123,
        attempt=1,
        launch_nonce=nonce,
        command_manifest_sha256="0" * 64,
    )
    snapshot = ({"CPU-X": "RUNNING"}, {"CPU-X": 1}, [started, released])
    MODULE._CONTROL_PLANE_FAULTS.clear()
    monkeypatch.setattr(MODULE, "_pid_alive", lambda pid: pid == 123)
    monkeypatch.setattr(
        MODULE, "_manifest_for", lambda *_args, **_kwargs: valid_manifest("CPU")
    )
    monkeypatch.setattr(
        MODULE, "_process_identity_matches", lambda *_args, **_kwargs: True
    )
    MODULE._activation_preflight(queued, snapshot)


def test_startup_event_fault_is_fail_closed_before_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = {"CPU-X": task("CPU")}
    MODULE._CONTROL_PLANE_FAULTS.clear()
    monkeypatch.setattr(
        MODULE,
        "_event_state",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MODULE.ControlPlaneBError("transient event snapshot")
        ),
    )
    snapshot = MODULE._event_state_fail_closed(queued)
    assert snapshot[0] == {"CPU-X": "SUSPENDED"}
    with pytest.raises(MODULE.ControlPlaneBError, match="control-plane read faults"):
        MODULE._activation_preflight(queued, snapshot)
    MODULE._CONTROL_PLANE_FAULTS.clear()


def test_a_evidence_classification_does_not_turn_generic_crash_into_a() -> None:
    manifest = valid_manifest("CPU")
    manifest["classification"] = "A_EVIDENCE"
    assert MODULE._is_a_class_returncode(manifest, 1) is False
    assert MODULE._is_a_class_returncode(manifest, 42) is True


def test_event_state_treats_manifest_a_rejection_as_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    use_synthetic_event_prefix(monkeypatch)
    set_event_rows(
        monkeypatch,
        [
            event_row(
                "COMMAND_MANIFEST_REJECTED_A",
                command_manifest_sha256="0" * 64,
                error="SHA identity mismatch",
            )
        ],
    )
    statuses, _, _ = MODULE._event_state()
    assert statuses["CPU-X"] == "DEGRADED_A"


def test_manifest_tool_rejection_backs_off_and_stops_after_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = task("CPU")
    queued.update(binding_row("CPU", "CPU-X"))
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        MODULE,
        "_manifest_for",
        lambda _task: (_ for _ in ()).throw(MODULE.ManifestToolError("bad schema")),
    )
    monkeypatch.setattr(
        MODULE, "_append_event", lambda event, **fields: events.append((event, fields))
    )
    assert MODULE._ready_tasks({"CPU-X": queued}, {}, {}, "CPU", 0, {}) == []
    assert [event for event, _ in events] == ["COMMAND_MANIFEST_B_RETRY_QUEUED"]
    assert (
        MODULE._ready_tasks(
            {"CPU-X": queued}, {"CPU-X": "B_RETRY_WAIT"}, {}, "CPU", 0, {"CPU-X": 1}
        )
        == []
    )
    assert len(events) == 1
    MODULE._ready_tasks(
        {"CPU-X": queued}, {"CPU-X": "PENDING"}, {}, "CPU", 0, {"CPU-X": 1}
    )
    MODULE._ready_tasks(
        {"CPU-X": queued}, {"CPU-X": "PENDING"}, {}, "CPU", 0, {"CPU-X": 2}
    )
    assert [event for event, _ in events] == [
        "COMMAND_MANIFEST_B_RETRY_QUEUED",
        "COMMAND_MANIFEST_B_RETRY_QUEUED",
        "COMMAND_MANIFEST_DEGRADED_A_RETRY_EXHAUSTED",
    ]
    MODULE._ready_tasks(
        {"CPU-X": queued}, {"CPU-X": "DEGRADED_A"}, {}, "CPU", 0, {"CPU-X": 3}
    )
    assert len(events) == 3


def test_manifest_exact_byte_mismatch_is_single_a_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = task("CPU")
    queued.update(binding_row("CPU", "CPU-X"))
    events: list[str] = []
    monkeypatch.setattr(
        MODULE,
        "_manifest_for",
        lambda _task: (_ for _ in ()).throw(MODULE.ManifestEvidenceError("SHA drift")),
    )
    monkeypatch.setattr(
        MODULE, "_append_event", lambda event, **_fields: events.append(event)
    )
    MODULE._ready_tasks({"CPU-X": queued}, {}, {}, "CPU", 0, {})
    MODULE._ready_tasks({"CPU-X": queued}, {"CPU-X": "DEGRADED_A"}, {}, "CPU", 0, {})
    assert events == ["COMMAND_MANIFEST_REJECTED_A"]


def test_legacy_manifest_is_recovery_only() -> None:
    manifest = valid_manifest("CPU")
    manifest.pop("admission")
    manifest["schema_version"] = "autonomous-20h-command-v1"
    with pytest.raises(MODULE.ManifestToolError, match="recovery-only"):
        MODULE._validate_command_manifest(manifest, task("CPU"))
    assert (
        MODULE._validate_command_manifest(
            manifest,
            task("CPU"),
            allow_legacy_recovery=True,
        )
        == manifest
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda admission, manifest: admission.update(
            pixel_or_selector_execution_admission=False
        ),
        lambda admission, manifest: admission.update(gpu_execution_admitted=True),
        lambda admission, manifest: admission.update(a_class_holds_released=True),
        lambda admission, manifest: admission.update(release_artifact={}),
        lambda admission, manifest: manifest["env"].update(CUDA_VISIBLE_DEVICES="0"),
        lambda admission, manifest: admission.update(allowed_input_paths=["relative"]),
        lambda admission, manifest: admission.update(
            allowed_input_paths=[str(ROOT / "datasets/grap_a_cap_025/x")]
        ),
    ],
)
def test_metadata_cpu_manifest_requires_explicit_zero_admission(mutation) -> None:
    manifest = valid_manifest("CPU")
    mutation(manifest["admission"], manifest)
    with pytest.raises(MODULE.ManifestToolError):
        MODULE._validate_command_manifest(manifest, task("CPU"))


def test_metadata_cpu_rejects_arbitrary_inline_python() -> None:
    manifest = valid_manifest("CPU")
    manifest["command"] = ["/usr/bin/python3", "-I", "-B", "-c", "pass", "{}"]
    with pytest.raises(MODULE.ManifestToolError, match="fixed exact-reference"):
        MODULE._validate_command_manifest(manifest, task("CPU"))


@pytest.mark.parametrize(
    "variable,value",
    [
        ("LD_PRELOAD", "/tmp/attacker.so"),
        ("LD_AUDIT", "/tmp/audit.so"),
        ("PYTHONPATH", "/tmp/attacker"),
        ("PYTHONHOME", "/tmp/python"),
    ],
)
def test_metadata_cpu_rejects_environment_injection(variable: str, value: str) -> None:
    manifest = valid_manifest("CPU")
    manifest["env"][variable] = value
    with pytest.raises(MODULE.ManifestToolError, match="admission contract"):
        MODULE._validate_command_manifest(manifest, task("CPU"))


@pytest.mark.parametrize(
    "filename",
    [
        "labels.json",
        "blind.json",
        "processed.json",
        "CLEAN_candidate.json",
        "025.json",
        "foo-025.json",
        "foo_025.json",
    ],
)
def test_metadata_cpu_rejects_forbidden_direct_json_families(filename: str) -> None:
    manifest = valid_manifest("CPU")
    forbidden = str(ROOT / "archive/legacy_runs/unclassified/AUTONOMOUS_20H_20260829/checkpoints" / filename)
    ref = {"path": forbidden, "bytes": 1, "sha256": "0" * 64}
    spec = {
        "refs": [
            {
                **ref,
                "json_assertions": [
                    {"pointer": "/schema_version", "equals": "fixture-v1"}
                ],
            }
        ]
    }
    manifest["admission"]["allowed_input_paths"] = [forbidden]
    manifest["success_evidence"] = [ref]
    manifest["command"][-1] = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    with pytest.raises(MODULE.ManifestToolError, match="forbidden"):
        MODULE._validate_command_manifest(manifest, task("CPU"))


@pytest.mark.parametrize(
    "path_suffix",
    ["nested/../025.json", "./foo-025.json", "archive/audits/../foo_025.json"],
)
def test_metadata_cpu_rejects_noncanonical_equivalents_of_025_paths(
    path_suffix: str,
) -> None:
    manifest = valid_manifest("CPU")
    path = str(ROOT / "archive/legacy_runs/unclassified/AUTONOMOUS_20H_20260829/checkpoints" / path_suffix)
    manifest["admission"]["allowed_input_paths"] = [path]
    with pytest.raises(MODULE.ManifestToolError):
        MODULE._validate_command_manifest(manifest, task("CPU"))


def test_fixed_reference_verifier_checks_bytes_sha_and_json_assertions(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    payload = b'{"admission":{"gpu_queueable_sessions":0}}\n'
    evidence.write_bytes(payload)
    ref = {
        "path": str(evidence),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "json_assertions": [
            {"pointer": "/admission/gpu_queueable_sessions", "equals": 0},
        ],
    }
    command = [
        "/usr/bin/python3",
        "-I",
        "-B",
        "-c",
        MODULE.EXACT_REFERENCE_VERIFIER_CODE,
        json.dumps({"refs": [ref]}, sort_keys=True, separators=(",", ":")),
    ]
    assert subprocess.run(command, check=False).returncode == 0
    ref["json_assertions"][0]["equals"] = 1
    command[-1] = json.dumps({"refs": [ref]}, sort_keys=True, separators=(",", ":"))
    assert subprocess.run(command, check=False).returncode == 42


def test_gpu_manifest_requires_separate_owner_release_contract() -> None:
    manifest = valid_manifest("GPU")
    manifest["admission"]["a_class_holds_released"] = False
    with pytest.raises(MODULE.ManifestToolError, match="explicit owner release"):
        MODULE._validate_command_manifest(manifest, task("GPU"))


def test_release_artifact_is_rehashed_and_authorizes_exact_task(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    release_path = tmp_path / "release.json"
    release = {
        "schema_version": "autonomous-20h-execution-release-v1",
        "task_id": "GPU-X",
        "channel": "GPU",
        "authorization_sha256": MODULE.AUTHORIZATION_SHA256,
        "release_authority": "OWNER_EXPLICIT_RELEASE",
        "released_at": "2026-08-29T12:00:00+08:00",
        "a_class_holds_released": True,
        "gpu_execution_admitted": True,
        "pixel_or_selector_execution_admission": 1,
    }
    payload = (json.dumps(release, sort_keys=True) + "\n").encode()
    release_path.write_bytes(payload)
    manifest = valid_manifest("GPU")
    manifest["admission"]["release_artifact"] = {
        "path": str(release_path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    MODULE._verify_release_artifact(manifest)
    release_path.write_bytes(b"drift")
    with pytest.raises(MODULE.ManifestEvidenceError, match="bytes/SHA"):
        MODULE._verify_release_artifact(manifest)


def test_start_card_intent_precedes_spawn_and_barrier_releases_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir()
    monkeypatch.setattr(MODULE, "LOG_ROOT", log_root)
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        MODULE, "_append_event", lambda event, **fields: events.append((event, fields))
    )
    queued = task("CPU")
    queued.update({"command_manifest_sha256": "0" * 64})
    manifest = {"command": [sys.executable, "-c", "pass"], "env": {}}
    card = MODULE._start_card(queued, manifest, 1)
    assert card is not None and card.process is not None
    assert card.process.wait(timeout=5) == 0
    card.log_handle.close()
    assert [event for event, _ in events] == [
        "CARD_LAUNCH_INTENT",
        "CARD_STARTED",
        "CARD_PAYLOAD_RELEASED",
    ]
    assert (
        events[0][1]["launch_nonce"]
        == events[1][1]["launch_nonce"]
        == card.launch_nonce
    )
    assert events[1][1]["pid"] == events[1][1]["pgid"]
    assert events[2][1]["pgid"] == events[1][1]["pgid"]


@pytest.mark.parametrize("fail_after_event", ["CARD_STARTED", "CARD_PAYLOAD_RELEASED"])
def test_barrier_crash_windows_never_execute_payload(
    fail_after_event: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir()
    marker = tmp_path / "payload-ran"
    monkeypatch.setattr(MODULE, "LOG_ROOT", log_root)
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    events: list[str] = []
    failed = False

    def append(event: str, **_fields) -> None:
        nonlocal failed
        events.append(event)
        if event == fail_after_event and not failed:
            failed = True
            raise MODULE.ControlPlaneBError("simulated supervisor crash boundary")

    monkeypatch.setattr(MODULE, "_append_event", append)
    queued = task("CPU")
    queued.update({"command_manifest_sha256": "0" * 64})
    manifest = {
        "command": [
            sys.executable,
            "-c",
            f"from pathlib import Path;Path({str(marker)!r}).write_text('ran')",
        ],
        "env": {},
    }
    assert MODULE._start_card(queued, manifest, 1) is None
    assert marker.exists() is False
    assert "CARD_B_RETRY_QUEUED" in events


def test_started_append_failure_records_intent_only_retry_that_replays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir()
    monkeypatch.setattr(MODULE, "LOG_ROOT", log_root)
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    durable_rows: list[dict] = []

    def append(event: str, **fields) -> None:
        if event == "CARD_STARTED":
            raise MODULE.ControlPlaneBError("simulated pre-persistence failure")
        durable_rows.append(event_row(event, **fields))

    monkeypatch.setattr(MODULE, "_append_event", append)
    queued = task("CPU")
    queued.update({"command_manifest_sha256": "0" * 64})
    manifest = {"command": [sys.executable, "-c", "pass"], "env": {}}
    assert MODULE._start_card(queued, manifest, 1) is None
    assert [row["event"] for row in durable_rows] == [
        "CARD_LAUNCH_INTENT",
        "CARD_B_RETRY_QUEUED",
    ]
    retry = durable_rows[-1]
    assert "pid" not in retry and "pgid" not in retry

    use_synthetic_event_prefix(monkeypatch)
    set_event_rows(monkeypatch, durable_rows)
    statuses, attempts, _ = MODULE._event_state({"CPU-X": queued})
    assert statuses["CPU-X"] == "B_RETRY_WAIT"
    assert attempts["CPU-X"] == 1


def test_popen_failure_terminalizes_launch_intent_as_b(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir()
    monkeypatch.setattr(MODULE, "LOG_ROOT", log_root)
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        MODULE, "_append_event", lambda event, **fields: events.append((event, fields))
    )
    monkeypatch.setattr(
        MODULE.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("boom")),
    )
    queued = task("CPU")
    queued.update({"command_manifest_sha256": "0" * 64})
    manifest = {"command": [sys.executable, "-c", "pass"], "env": {}}
    assert MODULE._start_card(queued, manifest, 1) is None
    assert [event for event, _ in events] == [
        "CARD_LAUNCH_INTENT",
        "CARD_B_RETRY_QUEUED",
    ]


def test_unresolved_launch_intent_is_reconciled_once_without_pid_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = task("CPU")
    queued.update(binding_row("CPU", "CPU-X"))
    intent = event_row(
        "CARD_LAUNCH_INTENT",
        attempt=1,
        launch_nonce="6" * 48,
        budget_minutes=10,
        command_manifest_sha256="0" * 64,
        log_path=str(MODULE.LOG_ROOT / "CPU-X.test.log"),
    )
    events: list[str] = []
    monkeypatch.setattr(
        MODULE,
        "_event_state",
        lambda *_args: ({"CPU-X": "LAUNCHING"}, {"CPU-X": 1}, [intent]),
    )
    monkeypatch.setattr(
        MODULE, "_append_event", lambda event, **_fields: events.append(event)
    )
    monkeypatch.setattr(
        MODULE, "_pid_alive", lambda _pid: pytest.fail("must not inspect any PID")
    )
    assert MODULE._recover_running_cards({"CPU-X": queued}) == {}
    assert events == ["CARD_B_RETRY_QUEUED"]


def test_cpu_start_forces_cuda_hidden(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir()
    monkeypatch.setattr(MODULE, "LOG_ROOT", log_root)
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "_append_event", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    queued = task("CPU")
    queued.update({"command_manifest_sha256": "0" * 64})
    manifest = {
        "command": [
            sys.executable,
            "-c",
            "import os;print(repr(os.environ.get('CUDA_VISIBLE_DEVICES')))",
        ],
        "env": {},
    }
    card = MODULE._start_card(queued, manifest, 1)
    assert card is not None and card.process is not None
    assert card.process.wait(timeout=5) == 0
    card.log_handle.close()
    log_path = next(log_root.iterdir())
    assert log_path.read_text(encoding="utf-8").strip() == "''"


def test_metadata_cpu_start_uses_clear_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir()
    monkeypatch.setattr(MODULE, "LOG_ROOT", log_root)
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(MODULE, "_append_event", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("LD_PRELOAD", "/tmp/attacker.so")
    monkeypatch.setenv("PYTHONPATH", "/tmp/attacker")
    queued = task("CPU")
    queued.update({"command_manifest_sha256": "0" * 64})
    manifest = {
        "command": [
            sys.executable,
            "-c",
            (
                "import os;print(repr((os.environ.get('LD_PRELOAD'),"
                "os.environ.get('PYTHONPATH'),os.environ.get('CUDA_VISIBLE_DEVICES'))))"
            ),
        ],
        "env": {"CUDA_VISIBLE_DEVICES": ""},
        "admission": {"execution_scope": "CONTROL_PLANE_METADATA_ONLY"},
    }
    card = MODULE._start_card(queued, manifest, 1)
    assert card is not None and card.process is not None
    assert card.process.wait(timeout=5) == 0
    card.log_handle.close()
    log_path = next(log_root.iterdir())
    assert log_path.read_text(encoding="utf-8").strip() == "(None, None, '')"
