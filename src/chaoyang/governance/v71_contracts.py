from __future__ import annotations

"""Dependency-light V7.1 revision and finite-terminal contracts.

The helpers in this module do not read or update a current ledger.  Workers may
use them to construct and publish immutable attempt artifacts; a separate
aggregator remains responsible for publishing current state.
"""

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARTIFACT_REVISION_VALIDITIES = frozenset(
    {"VALID_FOR_PINNED_REVISION", "STALE_FOR_LATEST_REVISION"}
)
TERMINAL_STATUSES = frozenset(
    {
        "PASSED",
        "FAILED_QUALITY_C",
        "FAILED_RUNTIME_FINAL",
        "BLOCKED_PREREQ",
        "BLOCKED_RESOURCE",
        "BLOCKED_EXTERNAL",
        "BLOCKED_REFERENCE_PROOF",
        "UNKNOWN_VERIFICATION_REQUIRED",
        "CANCELLED",
    }
)
NON_TERMINAL_STATUSES = frozenset(
    {"PENDING", "WAIT_GPU_RESOURCE", "CLAIMED", "RUNNING", "FAILED_RUNTIME_RETRYABLE"}
)
_REVISION_PREFIX = "R"


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_revision(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith(_REVISION_PREFIX):
        return False
    major, separator, minor = value[1:].partition("_")
    if not separator or not major.isdigit():
        return False
    base_minor = minor.split("_", 1)[0].split("-", 1)[0].split(".", 1)[0]
    return bool(base_minor) and base_minor.isdigit()


def validate_artifact_revision(value: Mapping[str, Any]) -> list[str]:
    """Validate V7.1 immutable revision metadata without touching its inputs."""

    required = {
        "artifact_id",
        "artifact_revision",
        "input_artifact_ids",
        "input_artifact_revisions",
        "input_manifest_sha",
        "producer_signature",
        "supersedes_artifact_id",
        "validity",
    }
    errors: list[str] = []
    missing = sorted(required - set(value))
    if missing:
        errors.append(f"missing artifact revision fields: {missing}")

    if not isinstance(value.get("artifact_id"), str) or not value.get("artifact_id"):
        errors.append("artifact_id must be a non-empty string")
    if not _is_revision(value.get("artifact_revision")):
        errors.append("artifact_revision must be an R<major>_<minor> revision")

    input_ids = value.get("input_artifact_ids")
    input_revisions = value.get("input_artifact_revisions")
    if not isinstance(input_ids, list) or any(not isinstance(item, str) or not item for item in input_ids):
        errors.append("input_artifact_ids must be a list of non-empty strings")
    if not isinstance(input_revisions, list) or any(not _is_revision(item) for item in input_revisions):
        errors.append("input_artifact_revisions must be a list of revisions")
    if isinstance(input_ids, list) and len(set(input_ids)) != len(input_ids):
        errors.append("input_artifact_ids must be unique")
    if isinstance(input_ids, list) and isinstance(input_revisions, list) and len(input_ids) != len(input_revisions):
        errors.append("input artifact ids and revisions must have equal lengths")

    if not _is_sha256(value.get("input_manifest_sha")):
        errors.append("input_manifest_sha must be a lowercase SHA-256")
    if not isinstance(value.get("producer_signature"), str) or not value.get("producer_signature"):
        errors.append("producer_signature must be a non-empty string")
    supersedes = value.get("supersedes_artifact_id")
    if supersedes is not None and (not isinstance(supersedes, str) or not supersedes):
        errors.append("supersedes_artifact_id must be null or a non-empty string")
    if supersedes == value.get("artifact_id"):
        errors.append("an artifact cannot supersede itself")
    if value.get("validity") not in ARTIFACT_REVISION_VALIDITIES:
        errors.append(f"validity must be one of {sorted(ARTIFACT_REVISION_VALIDITIES)}")
    return errors


def build_artifact_revision(
    *,
    artifact_id: str,
    artifact_revision: str,
    input_artifacts: Iterable[tuple[str, str]],
    input_manifest_sha: str,
    producer_signature: str,
    supersedes_artifact_id: str | None = None,
    validity: str = "VALID_FOR_PINNED_REVISION",
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated artifact envelope with positional input id/revision pins."""

    pins = list(input_artifacts)
    result: dict[str, Any] = {
        "artifact_id": artifact_id,
        "artifact_revision": artifact_revision,
        "input_artifact_ids": [item[0] for item in pins],
        "input_artifact_revisions": [item[1] for item in pins],
        "input_manifest_sha": input_manifest_sha,
        "producer_signature": producer_signature,
        "supersedes_artifact_id": supersedes_artifact_id,
        "validity": validity,
    }
    if payload is not None:
        result["payload"] = dict(payload)
    errors = validate_artifact_revision(result)
    if errors:
        raise ValueError("; ".join(errors))
    return result


def stale_for_latest_revision(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stale projection; never mutate or overwrite the pinned artifact."""

    errors = validate_artifact_revision(value)
    if errors:
        raise ValueError("; ".join(errors))
    result = dict(value)
    result["validity"] = "STALE_FOR_LATEST_REVISION"
    return result


def publish_new_revision(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish once and reject *every* pre-existing destination.

    Unlike the legacy idempotent final writer, this function refuses an
    existing path even when the bytes are identical.  A prepared temporary is
    hard-linked into place, so readers never observe a partial revision.
    """

    errors = validate_artifact_revision(value)
    if errors:
        raise ValueError("invalid artifact revision: " + "; ".join(errors))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, destination)  # raises FileExistsError, including identical content
        directory_fd = os.open(destination.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(destination.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "artifact_id": value["artifact_id"],
        "artifact_revision": value["artifact_revision"],
    }


def validate_task_packet_revision_contract(packet: Mapping[str, Any]) -> list[str]:
    """Validate an optional V7.1 packet policy while accepting legacy packets.

    The policy-only shape emitted by the V7.1 packet generator is sufficient.
    Claim-time tooling may additionally pin concrete input ids/revisions and an
    output revision in the same object.
    """

    contract = packet.get("artifact_revision_contract")
    legacy_alias = packet.get("revision_contract")
    if contract is None and legacy_alias is None:
        return []
    if contract is not None and legacy_alias is not None:
        return ["task packet must not define both artifact_revision_contract and revision_contract"]
    contract = contract if contract is not None else legacy_alias
    if not isinstance(contract, Mapping):
        return ["artifact_revision_contract must be an object"]

    errors: list[str] = []
    if contract.get("required_revision_status") != "VALID_FOR_PINNED_REVISION":
        errors.append("artifact_revision_contract.required_revision_status must be VALID_FOR_PINNED_REVISION")
    if contract.get("forbid_in_place_overwrite") is not True:
        errors.append("artifact_revision_contract.forbid_in_place_overwrite must be true")

    input_ids = contract.get("input_artifact_ids")
    input_revisions = contract.get("input_artifact_revisions")
    concrete_present = input_ids is not None or input_revisions is not None
    if concrete_present:
        if not isinstance(input_ids, list) or any(not isinstance(item, str) or not item for item in input_ids):
            errors.append("artifact_revision_contract.input_artifact_ids must be non-empty strings")
        if not isinstance(input_revisions, list) or any(not _is_revision(item) for item in input_revisions):
            errors.append("artifact_revision_contract.input_artifact_revisions must be revisions")
        if isinstance(input_ids, list) and isinstance(input_revisions, list) and len(input_ids) != len(input_revisions):
            errors.append("artifact_revision_contract input ids and revisions must have equal lengths")
        if not _is_sha256(contract.get("input_manifest_sha")):
            errors.append("artifact_revision_contract.input_manifest_sha must be a lowercase SHA-256")
    elif "input_manifest_sha" in contract and not _is_sha256(contract.get("input_manifest_sha")):
        errors.append("artifact_revision_contract.input_manifest_sha must be a lowercase SHA-256")

    if "output_artifact_revision" in contract and not _is_revision(contract.get("output_artifact_revision")):
        errors.append("artifact_revision_contract.output_artifact_revision must be an R<major>_<minor> revision")
    return errors


def summarize_terminal_closure(
    rows: Sequence[Mapping[str, Any]],
    expected_task_ids: Sequence[str] | None = None,
    *,
    task_id_field: str = "task_id",
    status_field: str = "status",
) -> dict[str, Any]:
    """Derive finite stage closure and authority from unique task terminals.

    Negative terminals count toward completion, but only an all-PASSED closed
    denominator authorizes downstream work.
    """

    expected = list(expected_task_ids) if expected_task_ids is not None else [
        str(row.get(task_id_field, "")) for row in rows
    ]
    expected_counter = Counter(expected)
    invalid_expected = sorted(task_id for task_id in expected if not task_id)
    duplicate_expected = sorted(task_id for task_id, count in expected_counter.items() if count > 1)
    expected_set = set(expected)

    row_ids = [str(row.get(task_id_field, "")) for row in rows]
    invalid_row_ids = sorted(task_id for task_id in row_ids if not task_id)
    row_counter = Counter(row_ids)
    duplicate_rows = sorted(task_id for task_id, count in row_counter.items() if count > 1)
    missing = sorted(expected_set - set(row_ids))
    unexpected = sorted(set(row_ids) - expected_set)
    non_terminal: list[str] = []
    invalid: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()
    for row, task_id in zip(rows, row_ids):
        status = str(row.get(status_field, ""))
        if task_id in expected_set:
            status_counts[status] += 1
        if status in NON_TERMINAL_STATUSES:
            non_terminal.append(task_id)
        elif status not in TERMINAL_STATUSES:
            invalid.append({"task_id": task_id, "status": status})

    stage_terminal_complete = not any(
        (
            invalid_expected,
            invalid_row_ids,
            duplicate_expected,
            duplicate_rows,
            missing,
            unexpected,
            non_terminal,
            invalid,
        )
    ) and len(rows) == len(expected)
    downstream_authorized = bool(expected) and stage_terminal_complete and status_counts == Counter(
        {"PASSED": len(expected)}
    )
    return {
        "stage_terminal_complete": stage_terminal_complete,
        "downstream_authorized": downstream_authorized,
        "denominator": len(expected),
        "terminal_count": sum(status_counts.get(status, 0) for status in TERMINAL_STATUSES),
        "status_counts": {status: status_counts.get(status, 0) for status in sorted(TERMINAL_STATUSES)},
        "missing_task_ids": missing,
        "unexpected_task_ids": unexpected,
        "duplicate_task_ids": sorted(set(duplicate_expected + duplicate_rows)),
        "invalid_task_ids": sorted(set(invalid_expected + invalid_row_ids)),
        "non_terminal_task_ids": sorted(non_terminal),
        "invalid_status_rows": invalid,
    }
