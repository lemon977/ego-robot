from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

# Support the documented absolute-script entry from any working directory.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chaoyang.governance.common import (
    ALGORITHM_CONTRACT_PATH,
    AUTHORITY_PATH,
    BASELINE_REGISTRY_PATH,
    CHANGELOG_PATH,
    DOC_AUTHORITY_MAP_PATH,
    MIN_STATUS_PATH,
    PLAN_MIGRATION_RECEIPT_PATH,
    RECEIPT_PATH,
    REGRESSION_MANIFEST_PATH,
    STATUS_PATH,
    TASK_QUEUE_PATH,
    TASK_STATE_PATH,
    REPO_ROOT,
    freshness,
    load_json,
    sha256_bytes,
    sha256_file,
    validate_artifact_ref,
    validate_authority,
    validate_schema,
    validate_task_state,
)
from chaoyang.governance.current_r3_contracts import validate_doc_authority_map
from chaoyang.governance.rc1_t5_consistency import validate_t5_current_flags


ARTIFACT_REF_KEYS = {"path", "bytes", "sha256"}
MAX_TRANSITIVE_SHA_BYTES = 64 * 1024 * 1024
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def iter_artifact_refs(value: object, trail: tuple[str, ...] = ()):
    """Yield nested artifact refs without treating their scalar fields as children."""
    if isinstance(value, dict):
        if ARTIFACT_REF_KEYS.issubset(value):
            yield ".".join(trail) or "<root>", value
            return
        for key, child in value.items():
            yield from iter_artifact_refs(child, trail + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_artifact_refs(child, trail + (str(index),))


def validate_transitive_artifact_refs(
    value: object,
    *,
    label: str,
    max_sha_bytes: int = MAX_TRANSITIVE_SHA_BYTES,
) -> tuple[list[str], dict[str, int]]:
    """Validate nested refs while avoiding multi-GB model rehash on every status check."""
    errors: list[str] = []
    seen: set[tuple[object, object, object]] = set()
    stats = {"unique_refs": 0, "sha_verified": 0, "large_sha_skipped": 0}
    for location, reference in iter_artifact_refs(value):
        identity = (reference.get("path"), reference.get("bytes"), reference.get("sha256"))
        if identity in seen:
            continue
        seen.add(identity)
        stats["unique_refs"] += 1
        path = Path(str(reference.get("path", "")))
        if not path.is_absolute():
            errors.append(f"{label}.{location}: artifact path is not absolute: {path}")
            continue
        if not path.is_file():
            errors.append(f"{label}.{location}: artifact missing: {path}")
            continue
        actual_bytes = path.stat().st_size
        if reference.get("bytes") != actual_bytes:
            errors.append(
                f"{label}.{location}: bytes mismatch for {path}: "
                f"expected={reference.get('bytes')} actual={actual_bytes}"
            )
        declared_bytes = reference.get("bytes")
        if isinstance(declared_bytes, int) and declared_bytes > max_sha_bytes:
            stats["large_sha_skipped"] += 1
            continue
        stats["sha_verified"] += 1
        actual_sha = sha256_file(path)
        if reference.get("sha256") != actual_sha:
            errors.append(
                f"{label}.{location}: sha256 mismatch for {path}: "
                f"expected={reference.get('sha256')} actual={actual_sha}"
            )
    return errors, stats


def validate_current_markdown_links(doc_authority_map: dict[str, object]) -> tuple[list[str], int]:
    """Treat broken local links as blockers only for receipt-registered CURRENT Markdown."""
    errors: list[str] = []
    checked = 0
    for document in doc_authority_map.get("documents", []):
        if not isinstance(document, dict) or document.get("status") != "CURRENT":
            continue
        source = Path(str(document.get("path", "")))
        if source.suffix.lower() != ".md" or not source.is_file():
            continue
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            for raw_target in MARKDOWN_LINK_RE.findall(line):
                target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
                if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                target = unquote(target.split("#", 1)[0])
                if not target:
                    continue
                checked += 1
                candidate = Path(target)
                if not candidate.is_absolute():
                    candidate = source.parent / candidate
                if not candidate.exists():
                    errors.append(
                        f"CURRENT document has broken local link: {source}:{line_number} -> {target}"
                    )
    return errors, checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-stale", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    receipt = load_json(RECEIPT_PATH)
    authority = load_json(AUTHORITY_PATH)
    task_state = load_json(TASK_STATE_PATH)
    for item in receipt["files"].values():
        errors.extend(validate_artifact_ref(item))
    for value, name in ((authority, "authority"), (task_state, "task_state")):
        if value["governance_revision"] != receipt["governance_revision"]:
            errors.append(f"{name} revision does not match receipt")
        if value["generation_id"] != receipt["generation_id"]:
            errors.append(f"{name} generation_id does not match receipt")
    errors.extend(validate_authority(authority))
    errors.extend(validate_task_state(task_state))
    errors.extend(validate_t5_current_flags(task_state))
    doc_authority_map = load_json(DOC_AUTHORITY_MAP_PATH)
    algorithm_contract = load_json(ALGORITHM_CONTRACT_PATH)
    baseline_registry = load_json(BASELINE_REGISTRY_PATH)
    regression_manifest = load_json(REGRESSION_MANIFEST_PATH)
    validate_schema("doc_authority_map.schema.json", doc_authority_map)
    validate_schema("algorithm_contract.schema.json", algorithm_contract)
    errors.extend(validate_doc_authority_map(doc_authority_map))
    transitive_stats: dict[str, dict[str, int]] = {}
    for value, label in (
        (algorithm_contract, "algorithm_contract"),
        (baseline_registry, "baseline_registry_v2"),
        (regression_manifest, "regression_manifest"),
    ):
        ref_errors, ref_stats = validate_transitive_artifact_refs(value, label=label)
        errors.extend(ref_errors)
        transitive_stats[label] = ref_stats
    link_errors, current_links_checked = validate_current_markdown_links(doc_authority_map)
    errors.extend(link_errors)
    for value, name in (
        (doc_authority_map, "doc_authority_map"),
        (algorithm_contract, "algorithm_contract"),
    ):
        if value.get("governance_revision") != receipt["governance_revision"]:
            errors.append(f"{name} revision does not match receipt")
        if value.get("generation_id") != receipt["generation_id"]:
            errors.append(f"{name} generation_id does not match receipt")
    if "plan_migration_receipt" in receipt.get("files", {}):
        migration = load_json(PLAN_MIGRATION_RECEIPT_PATH)
        validate_schema("plan_migration_receipt.schema.json", migration)
    pointer = load_json(REPO_ROOT / "docs/governance/CURRENT_V71_TASK_PACKET_INDEX.json")
    if pointer.get("governance_revision") != receipt["governance_revision"]:
        errors.append("task packet pointer revision does not match receipt")
    if pointer.get("generation_id") != receipt["generation_id"]:
        errors.append("task packet pointer generation_id does not match receipt")
    packet_index_path = Path(str(pointer.get("index_path", "")))
    if not packet_index_path.is_absolute():
        packet_index_path = REPO_ROOT / packet_index_path
    if not packet_index_path.is_file():
        errors.append(f"current task packet index missing: {packet_index_path}")
        packet_index = {"task_packets": []}
    else:
        packet_index = load_json(packet_index_path)
        index_ref = validate_artifact_ref({
            "path": str(packet_index_path.resolve()),
            "bytes": pointer.get("index_bytes", packet_index_path.stat().st_size),
            "sha256": pointer.get("index_sha256"),
        })
        errors.extend(index_ref)
        receipt_index = receipt.get("files", {}).get("task_packet_index_current")
        if receipt_index is None:
            errors.append("current receipt does not bind pointed task packet index")
        elif receipt_index.get("sha256") != pointer.get("index_sha256"):
            errors.append("receipt task packet index SHA does not match pointer")
    packet_rows = packet_index.get("task_packets", [])
    packet_ids = [str(item.get("task_id")) for item in packet_rows]
    if len(packet_ids) != len(set(packet_ids)):
        errors.append("current task packet index contains duplicate task ids")
    packet_entries = {str(item.get("task_id")): item for item in packet_rows}
    for task_id, entry in packet_entries.items():
        packet_path = Path(str(entry.get("packet_path", "")))
        if not packet_path.is_absolute():
            packet_path = REPO_ROOT / packet_path
        if not packet_path.is_file():
            errors.append(f"indexed task packet missing: {task_id} -> {packet_path}")
            continue
        actual_sha = sha256_file(packet_path)
        if entry.get("packet_sha256") != actual_sha:
            errors.append(f"indexed task packet SHA mismatch: {task_id}")
        for key in ("development_result", "development_receipt"):
            reference = entry.get(key)
            if isinstance(reference, dict):
                errors.extend(validate_artifact_ref(reference))
        execution_class = entry.get("execution_class")
        execution_allowed = entry.get("execution_allowed")
        if execution_class not in {
            "HISTORICAL_SUPERSEDED_READ_SET",
            "TERMINAL_LEDGER_EVIDENCE",
            "CATALOG_REFERENCE",
            "CURRENT_LEDGER_ROUTABLE",
        }:
            errors.append(f"indexed task lacks valid execution classification: {task_id}")
        if not isinstance(execution_allowed, bool):
            errors.append(f"indexed task lacks boolean execution_allowed: {task_id}")
    next_task = task_state.get("next_task")
    if next_task is not None:
        next_task_id = str(next_task.get("task_id", ""))
        if not any(str(item.get("task_id")) == next_task_id for item in task_state.get("tasks", [])):
            errors.append(f"next_task missing from task state: {next_task_id}")
        if next_task_id not in packet_entries:
            errors.append(f"next_task missing from current packet index: {next_task_id}")
    document_status = {
        str(Path(item["path"]).resolve()): item["status"]
        for item in doc_authority_map.get("documents", [])
    }
    task_status = {
        str(item.get("task_id")): str(item.get("status"))
        for item in task_state.get("tasks", [])
    }
    routable_statuses = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
    for task_id, entry in packet_entries.items():
        packet_path = Path(str(entry.get("packet_path", "")))
        if not packet_path.is_absolute():
            packet_path = REPO_ROOT / packet_path
        if not packet_path.is_file():
            continue
        packet = load_json(packet_path)
        superseded_paths: list[str] = []
        for raw_path in packet.get("read_set", []):
            candidate = Path(str(raw_path))
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            if document_status.get(str(candidate.resolve())) == "SUPERSEDED":
                superseded_paths.append(str(candidate.resolve()))
        if superseded_paths:
            if entry.get("execution_class") != "HISTORICAL_SUPERSEDED_READ_SET":
                errors.append(f"superseded read-set task is not historical-only: {task_id}")
            if entry.get("execution_allowed") is not False:
                errors.append(f"superseded read-set task is executable: {task_id}")
        if entry.get("execution_allowed") is True:
            if task_status.get(task_id) not in routable_statuses:
                errors.append(f"task marked executable without routable ledger state: {task_id}")
            if task_id != str((next_task or {}).get("task_id", "")):
                errors.append(f"task marked executable but is not current next_task: {task_id}")
    for task in task_state.get("tasks", []):
        if task.get("plan_execution_revision") != "R3":
            continue
        packet_ref = task.get("task_packet")
        if not isinstance(packet_ref, dict):
            continue
        packet_path = Path(str(packet_ref.get("path") or packet_ref.get("packet_path", "")))
        if not packet_path.is_absolute():
            packet_path = REPO_ROOT / packet_path
        if not packet_path.is_file():
            errors.append(f"R3 task packet missing: {packet_path}")
            continue
        packet = load_json(packet_path)
        indexed = packet_entries.get(str(task.get("task_id")))
        if indexed is None:
            errors.append(f"R3 task missing from current packet index: {task.get('task_id')}")
        else:
            actual_ref = {
                "path": str(packet_path.resolve()),
                "bytes": packet_path.stat().st_size,
                "sha256": sha256_file(packet_path),
            }
            if indexed.get("packet_sha256") != actual_ref["sha256"]:
                errors.append(f"R3 task packet SHA differs from current index: {task.get('task_id')}")
            if packet_ref.get("sha256") not in {None, actual_ref["sha256"]}:
                errors.append(f"R3 task row packet SHA mismatch: {task.get('task_id')}")
            if packet_ref.get("bytes") not in {None, actual_ref["bytes"]}:
                errors.append(f"R3 task row packet bytes mismatch: {task.get('task_id')}")
        for raw_path in packet.get("read_set", []):
            candidate = Path(str(raw_path))
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            if document_status.get(str(candidate.resolve())) == "SUPERSEDED":
                errors.append(f"R3 task references SUPERSEDED document: {task.get('task_id')} -> {candidate}")
    if not STATUS_PATH.read_text(encoding="utf-8").startswith("# 当前项目实时事实页"):
        errors.append("project status is not generated content")
    for projection_path, label in (
        (MIN_STATUS_PATH, "project_status_min"),
        (TASK_QUEUE_PATH, "task_queue"),
    ):
        projection = load_json(projection_path)
        if projection.get("governance_revision") != receipt["governance_revision"]:
            errors.append(f"{label} revision does not match receipt")
        if projection.get("generation_id") != receipt["generation_id"]:
            errors.append(f"{label} generation_id does not match receipt")
    previous = None
    if CHANGELOG_PATH.is_file():
        for index, line in enumerate(CHANGELOG_PATH.read_text(encoding="utf-8").splitlines(), 1):
            event = json.loads(line)
            claimed = event.pop("event_sha256")
            if event.get("previous_event_sha256") != previous:
                errors.append(f"changelog chain mismatch at line {index}")
            actual = sha256_bytes((json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())
            if actual != claimed:
                errors.append(f"changelog event sha mismatch at line {index}")
            previous = claimed
    fresh = freshness(task_state)
    if fresh["status"] in {"STALE", "SUSPECTED_DEAD_WORKER"} and not args.allow_stale:
        errors.append(f"freshness gate failed: {fresh}")
    output = {
        "status": "PASS" if not errors else "STATUS_CONFLICT",
        "errors": errors,
        "freshness": fresh,
        "freshness_scope": "active_task_heartbeat_only",
        "current_markdown_local_links_checked": current_links_checked,
        "transitive_refs": transitive_stats,
        "revision": receipt["governance_revision"],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
