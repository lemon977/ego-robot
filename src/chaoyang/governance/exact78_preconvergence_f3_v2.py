"""Conservative, read-only Exact78 archive eligibility inventory.

This module does not move, delete, register, or terminalize anything.  In
particular, an archive candidate is never promoted merely because a string
search did not find a reference: historical references and license rights
must be proven independently before a cleanup task can be authorized.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any, Mapping


CURRENT_REFERENCE_PATHS = (
    "docs/governance/CURRENT_STATUS_RECEIPT.json",
    "docs/governance/DOC_AUTHORITY_MAP.json",
    "docs/governance/ALGORITHM_CONTRACT.json",
    "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json",
    "docs/governance/CURRENT_REGRESSION_MANIFEST.json",
    "docs/governance/LONG_HORIZON_TASK_STATE.json",
    "docs/governance/TASK_QUEUE.json",
    "docs/current/EXACT78_EXECUTION_ROUTE_V1.json",
    "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json",
    "docs/current/EXACT78_ACTIVE_LINEAGE_V1.json",
    "tasks/current/INDEX.json",
)
SELECTED_REFERENCE_PATHS = frozenset({
    "docs/current/EXACT78_EXECUTION_ROUTE_V1.json",
    "docs/current/EXACT78_CARDS042_VISUAL_BASELINE_V2.json",
    "docs/current/EXACT78_ACTIVE_LINEAGE_V1.json",
})
PARENT_TASK = "exact78_cards042_fresh_robot_v1_20260929"


def _relative(repo: Path, path: Path) -> str:
    return str(path.relative_to(repo))


def _walk_apparent_bytes(directory: Path) -> tuple[int, int, list[str]]:
    """Count apparent bytes without following links or leaving the task root."""
    total, files, errors = 0, 0, []

    def onerror(error: OSError) -> None:
        errors.append(f"{error.filename}:{error.strerror}")

    for base, dirs, names in os.walk(directory, followlinks=False, onerror=onerror):
        for name in [".", *dirs, *names]:
            item = Path(base) if name == "." else Path(base) / name
            try:
                total += item.lstat().st_size
                if name in names:
                    files += 1
            except OSError as error:
                errors.append(f"{item}:{error.strerror}")
    return total, files, errors


def _read_references(repo: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    current: dict[str, str] = {}
    historical: dict[str, str] = {}
    errors: list[str] = []
    current_paths = [repo / value for value in CURRENT_REFERENCE_PATHS]
    historical_paths = [
        *sorted((repo / "tasks/current").glob("*/TASK_PACKET.json")),
        *sorted((repo / "tasks/receipts").glob("*.json")),
    ]
    for destination, paths in ((current, current_paths), (historical, historical_paths)):
        for path in paths:
            try:
                destination[_relative(repo, path)] = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                errors.append(f"{path}:{type(error).__name__}:{error}")
    return current, historical, errors


def _matches_reference(repo: Path, task: Path, documents: Mapping[str, str]) -> list[str]:
    absolute = str(task)
    relative = _relative(repo, task)
    # The task ID also catches structured references that store only an ID.
    return [name for name, content in documents.items()
            if absolute in content or relative in content or task.name in content]


def _process_holders(task_roots: list[Path]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    holders: dict[str, list[dict[str, Any]]] = {root.name: [] for root in task_roots}
    errors: list[str] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        pid = int(proc.name)
        seen: list[tuple[str, str]] = []
        try:
            seen.append(("cwd", os.readlink(proc / "cwd")))
        except (FileNotFoundError, ProcessLookupError):
            continue  # The process exited while /proc was being walked.
        except OSError as error:
            errors.append(f"pid={pid}:cwd:{type(error).__name__}")
        try:
            fd_entries = list((proc / "fd").iterdir())
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as error:
            errors.append(f"pid={pid}:fd-list:{type(error).__name__}")
            fd_entries = []
        for fd in fd_entries:
            try:
                seen.append((f"fd/{fd.name}", os.readlink(fd)))
            except (FileNotFoundError, ProcessLookupError):
                continue  # A single closed FD must not hide the other FDs.
            except OSError as error:
                errors.append(f"pid={pid}:fd/{fd.name}:{type(error).__name__}")
        for root in task_roots:
            prefix = str(root) + os.sep
            matched = [{"kind": kind, "path": value} for kind, value in seen
                       if value == str(root) or value.startswith(prefix)]
            if matched:
                holders[root.name].append({"pid": pid, "handles": matched[:25],
                                           "handle_count": len(matched)})
    return holders, errors


def _git_state(repo: Path) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=repo, capture_output=True, check=False,
    )
    if status.returncode:
        return {"status": "UNKNOWN", "error": status.stderr.decode("utf-8", "replace")}
    # For -z porcelain v1, a rename or copy has a second NUL-terminated path.
    records = status.stdout.decode("utf-8", "replace").split("\0")
    entries: list[str] = []
    index = 0
    while index < len(records) and records[index]:
        record = records[index]
        entries.append(record[3:] if len(record) >= 3 else record)
        if len(record) >= 2 and ("R" in record[:2] or "C" in record[:2]):
            index += 1
            if index < len(records):
                entries.append(records[index])
        index += 1
    ignore = subprocess.run(
        ["git", "check-ignore", "-v", "--", "_run/current/exact78_probe/RESULT.json"],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    return {
        "status": "OK", "dirty_untracked_paths": entries,
        "run_tree_ignored_by_git": ignore.returncode == 0,
        "ignore_rule": ignore.stdout.strip() if ignore.returncode == 0 else None,
        "limitation": "git status cannot enumerate ignored _run artifacts",
    }


def build_archive_eligibility(repo_root: Path, inventory: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a JSON-serializable audit; no result authorizes file movement.

    ``inventory`` may be the frozen F0 INVENTORY.json. It is used only to
    report which task directories already had a frozen RESULT reference.
    """
    repo = repo_root.resolve(strict=True)
    task_roots = sorted(path for path in (repo / "_run/current").glob("exact78_*")
                        if path.is_dir() and not path.is_symlink())
    current, historical, ref_errors = _read_references(repo)
    holders, proc_errors = _process_holders(task_roots)
    git = _git_state(repo)
    frozen_results = {
        Path(ref["path"]).parts[Path(ref["path"]).parts.index("current") + 1]
        for ref in (inventory or {}).get("historical_evidence", {}).get("result_files", [])
        if "current" in Path(ref["path"]).parts
    }
    license_files = sorted(_relative(repo, path) for path in repo.glob("LICENSE*") if path.is_file())
    rows: list[dict[str, Any]] = []
    for task in task_roots:
        apparent_bytes, file_count, size_errors = _walk_apparent_bytes(task)
        current_refs = _matches_reference(repo, task, current)
        selected_refs = [name for name in current_refs if name in SELECTED_REFERENCE_PATHS]
        historical_refs = _matches_reference(repo, task, historical)
        dirty = [path for path in git.get("dirty_untracked_paths", [])
                 if path == _relative(repo, task) or path.startswith(_relative(repo, task) + "/")]
        if task.name == PARENT_TASK or selected_refs:
            category = "MUST_KEEP"
            reason = "selected_visual_parent_or_explicit_selected_lineage_reference"
        else:
            category = "BLOCKED_REFERENCE_PROOF"
            reason = "historical_references_or_license_or_scan_proof_incomplete"
        rows.append({
            "task_directory": _relative(repo, task), "task_id": task.name,
            "category": category, "reason": reason,
            "apparent_bytes": apparent_bytes, "file_count": file_count,
            "has_frozen_result": task.name in frozen_results,
            "selected_refs": selected_refs, "current_refs": current_refs,
            "historical_refs": historical_refs,
            "open_processes": holders[task.name], "git_dirty_untracked_paths": dirty,
            "size_scan_errors": size_errors,
            "license_status": "UNKNOWN_NO_PER_ARTIFACT_RIGHTS_PROOF",
        })
    counts = {label: sum(row["category"] == label for row in rows)
              for label in ("MUST_KEEP", "ARCHIVE_CANDIDATE", "BLOCKED_REFERENCE_PROOF")}
    return {
        "schema_version": "exact78-archive-eligibility-audit-v2",
        "scope": "0902 Exact78 _run/current/exact78_* task directories; no move or delete",
        "rows": rows, "counts": counts,
        "task_directory_count": len(rows),
        "total_apparent_bytes": sum(row["apparent_bytes"] for row in rows),
        "bytes_by_category": {label: sum(row["apparent_bytes"] for row in rows
                                         if row["category"] == label) for label in counts},
        "reference_files_scanned": {"current": sorted(current), "historical": sorted(historical)},
        "reference_scan_errors": ref_errors, "process_scan_errors": proc_errors,
        "git": git, "root_license_files": license_files,
        "license_assessment": "UNKNOWN; root or vendor licenses do not prove rights for each artifact",
        "size_assessment": "Apparent bytes include directory metadata and may double-count hard links; not reclaimable bytes",
        "files_moved": False, "files_deleted": False,
        "cleanup_authorized": False,
    }
