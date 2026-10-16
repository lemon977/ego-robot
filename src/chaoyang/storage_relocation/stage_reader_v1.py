"""Read-only original stage proofs with separate logical/storage identities.

This is a storage adapter, not a producer, repair, or relaxed reuse gate.  The
checks below mirror ``pick_place_1008.stage_reuse._validate``.  Receipt paths,
producer ownership, binding hashes and returned references stay logical; only
the reader's verified file access may use a frozen NAS payload mapping.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from chaoyang.pick_place_1008.stage_reuse import (
    SCHEMA,
    TERMINAL_STATUSES,
    _canonical_hash,
    _deny,
    _plain,
)


class StageReader(Protocol):
    """The bounded, read-only subset of Original199Reader used here."""

    def reference_for(self, logical_path: str | Path) -> dict: ...

    def verify_ref(self, pin: dict) -> dict: ...

    def read_json(self, pin: dict): ...

    def physical_path(self, pin: dict) -> Path: ...


def _read(reader: StageReader, logical_path: Path) -> dict:
    value = reader.read_json(reader.reference_for(logical_path))
    if not isinstance(value, dict):
        _deny("JSON_OBJECT_REQUIRED")
    return value


def _logical_plain(path: Path, repo: Path) -> Path:
    """Keep project/anti-link checks without requiring a removed payload."""
    absolute = path.absolute()
    if not absolute.is_relative_to(repo):
        _deny("PATH_OUTSIDE_PROJECT:" + str(path))
    for item in (absolute, *absolute.parents):
        if item == repo:
            break
        if item.is_symlink():
            _deny("SYMLINK_PATH:" + str(item))
    # Existing metadata is still checked by the original _plain gate; a missing
    # payload is admissible only if the reader verifies its exact frozen map.
    return absolute


def _pin(reader: StageReader, pin: dict, expected_path: Path | None = None,
         repo: Path | None = None):
    if not isinstance(pin, dict) or set(pin) != {"path", "bytes", "sha256"}:
        _deny("ARTIFACT_REFERENCE_SHAPE")
    if expected_path is not None:
        if pin["path"] != str(expected_path):
            _deny("ARTIFACT_PATH_MISMATCH:" + str(pin.get("path")))
        if repo is not None:
            _logical_plain(expected_path, repo)
    verified = reader.verify_ref(pin)
    if verified != pin:
        _deny("ARTIFACT_REFERENCE_IDENTITY_CHANGED:" + str(pin.get("path")))


def validate_sealed_stage(reader: StageReader, repo: Path, root: Path,
                          sid: str, stage: str, expected: dict,
                          required_files: tuple[str, ...]):
    """Apply the original sealed-stage contract without rewriting any pin.

    Missing stage JSON and a valid different binding retain the original cache
    miss ``(None, None)`` semantics.  Malformed, unsealed, moved-identity or
    changed-byte evidence is rejected rather than treated as a cache miss.
    All small producer/ledger/receipt JSON stays at its original project path.
    """
    repo, root = Path(repo), Path(root)
    folder = root / "sessions" / sid
    path = folder / ("STAGE_" + stage + ".json")
    if not path.exists() and not path.is_symlink():
        return None, None
    _plain(path, repo)
    receipt = _read(reader, path)
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "COMPLETE":
        _deny("IDENTITY_SCHEMA_OR_INCOMPLETE_STAGE")
    binding = receipt.get("binding")
    if not isinstance(binding, dict):
        _deny("BINDING_MISSING")
    declared = receipt.get("binding_sha256")
    if _canonical_hash(binding) != declared:
        _deny("BINDING_DIGEST_TAMPERED")
    if (binding.get("stage") != stage or binding.get("session_id") != sid
            or type(binding.get("raw_frame_count")) is not int
            or binding["raw_frame_count"] <= 0):
        _deny("STAGE_SESSION_OR_FRAME_IDENTITY_MISMATCH")
    if _canonical_hash(expected) != declared:
        return None, None
    files = receipt.get("files")
    if not isinstance(files, dict) or set(files) != set(required_files):
        _deny("MANDATORY_OUTPUT_SET_MISMATCH")
    for name in required_files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            _deny("MANDATORY_OUTPUT_NAME_ESCAPE")
        _pin(reader, files[name], folder / relative, repo)
    code = binding.get("code_and_dependencies")
    if not isinstance(code, list) or not code:
        _deny("CODE_DEPENDENCY_CLOSURE_MISSING")
    for pin in code:
        _pin(reader, pin)
    raw = binding.get("raw_source")
    raw = list(raw.values()) if isinstance(raw, dict) else raw
    if not isinstance(raw, list) or not raw:
        _deny("RAW_SOURCE_CLOSURE_MISSING")
    for pin in raw:
        _pin(reader, pin)
    _pin(reader, binding["camera_config"])
    upstream = binding.get("upstream")
    if not isinstance(upstream, dict):
        _deny("UPSTREAM_CLOSURE_MISSING")
    for pin in upstream.values():
        _pin(reader, pin)
    producer = receipt.get("producer")
    if not isinstance(producer, dict):
        _deny("PRODUCER_MISSING")
    task_id = producer.get("task_id")
    if (not isinstance(task_id, str) or not task_id.startswith("exact78_pick_place_")
            or Path(task_id).name != task_id or "/" in task_id or "\\" in task_id):
        _deny("PRODUCER_TASK_IDENTITY")
    packet_path = repo / "tasks/current" / task_id / "TASK_PACKET.json"
    _pin(reader, producer["task_packet"], packet_path, repo)
    packet = _read(reader, packet_path)
    if (packet.get("task_id") != task_id or packet.get("run_signature") != producer.get("run_signature")
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"):
        _deny("PRODUCER_PACKET_IDENTITY_OR_SIGNATURE")
    result_path = repo / "_run/current" / task_id / "attempts/attempt_0001/RESULT.json"
    if producer.get("result_path") != str(result_path):
        _deny("PRODUCER_TERMINAL_PATH_MISMATCH")
    if root != result_path.parent / "candidate" or folder != result_path.parent / "candidate/sessions" / sid:
        _deny("STAGE_NOT_OWNED_BY_PRODUCER")
    ledger = _read(reader, repo / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    rows = [row for row in ledger.get("tasks", []) if row.get("task_id") == task_id]
    if len(rows) != 1 or rows[0].get("status") not in TERMINAL_STATUSES:
        _deny("PRODUCER_NOT_SEALED_TERMINAL")
    row = rows[0]
    if row.get("pid") is not None or row.get("proc_start_ticks") is not None:
        _deny("PRODUCER_TERMINAL_WRITER_PRESENT")
    _pin(reader, row["result"], result_path, repo)
    terminal = _read(reader, result_path)
    if (terminal.get("status") != row["status"] or terminal.get("task_id") != task_id
            or terminal.get("run_signature") != producer.get("run_signature")
            or terminal.get("task_packet") != producer["task_packet"]):
        _deny("PRODUCER_TERMINAL_IDENTITY_OR_SIGNATURE")
    index_path = result_path.parent / "candidate/INDEX.json"
    index_pin = terminal.get("required_outputs", {}).get("candidate/INDEX.json")
    if not isinstance(index_pin, dict):
        _deny("TERMINAL_INDEX_REFERENCE_MISSING")
    _pin(reader, index_pin, index_path, repo)
    index = _read(reader, index_path)
    rows = [row for row in index.get("sessions", []) if row.get("session_id") == sid]
    if len(rows) != 1:
        _deny("PRODUCER_INDEX_SESSION_IDENTITY")
    row = rows[0]
    direct = row.get("stage_receipts", {}).get(stage)
    nested = row.get("stages", {}).get(stage, {}).get("receipt")
    if direct is not None and nested is not None and direct != nested:
        _deny("PRODUCER_INDEX_STAGE_REFERENCE_CONFLICT")
    bound = direct if direct is not None else nested
    current = reader.reference_for(path)
    if bound != current:
        _deny("PRODUCER_INDEX_DOES_NOT_BIND_STAGE")
    _pin(reader, bound, path, repo)
    return folder, current


def validate_session_actions(reader: StageReader, actions_ref: dict,
                             frame_count: int) -> list[str]:
    """Run the unchanged action-array numerical gate on verified input bytes."""
    from chaoyang.pick_place_1008.audit import validate_action_arrays

    return validate_action_arrays(reader.physical_path(actions_ref), frame_count)
