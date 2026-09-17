#!/usr/bin/env python3
"""Build the R2.2 old-asset cleanup dry-run; never deletes old runs or documents."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.governance.common import (  # noqa: E402
    RECEIPT_PATH,
    artifact_ref,
    atomic_json,
    atomic_write,
    load_json,
    now_iso,
    validate_artifact_ref,
)


CURRENT_FILES = (
    ROOT / "docs/governance/CURRENT_AUTHORITY_INDEX.json",
    ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json",
    ROOT / "docs/governance/DOC_AUTHORITY_MAP.json",
    ROOT / "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json",
    ROOT / "docs/governance/CURRENT_REGRESSION_MANIFEST.json",
    ROOT / "docs/governance/CURRENT_V71_TASK_PACKET_INDEX.json",
)
RUNS = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs"
RUNTIME = ROOT / "_run"
HARD_PROTECTED = (
    ROOT / "docs/current/visuals",
    ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_chaoyang_r22_4h",
    ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1",
    ROOT / "third_party",
    ROOT / "assets/models",
)
EXTERNAL_PROTECTED = (Path("/mnt/data/egodata"), Path("/nas/chenxianchi"))
CONTROL_RUNTIME_NAMES = {
    ".gitkeep",
    "README.md",
    "GPU_LEASE.json",
    "GPU_LEASE.lock",
    ".gpu_lease.dirlock",
}
PATH_PATTERN = re.compile(r"/(?:mnt|nas|cpfs_infra)/[^\s\"'<>]+")


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _contained(path: Path, prefix: Path) -> bool:
    try:
        path.resolve().relative_to(prefix.resolve())
        return True
    except (ValueError, OSError):
        return False


def _references() -> tuple[set[Path], dict[str, list[str]]]:
    direct: set[Path] = set()
    source_by_target: dict[str, list[str]] = defaultdict(list)
    receipt = load_json(RECEIPT_PATH)
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError("current receipt conflict: " + "; ".join(errors))
        direct.add(Path(reference["path"]).resolve())
    queue = [path for path in CURRENT_FILES if path.is_file()] + sorted(direct)
    seen: set[Path] = set()
    for source in queue:
        source = source.resolve()
        if source in seen or not source.is_file():
            continue
        seen.add(source)
        try:
            if source.suffix == ".json":
                strings = list(_walk_strings(load_json(source)))
            else:
                strings = PATH_PATTERN.findall(source.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError):
            continue
        for raw in strings:
            if not raw.startswith("/"):
                continue
            target = Path(raw.rstrip(".,;:)]}"))
            try:
                resolved = target.resolve()
            except OSError:
                continue
            direct.add(resolved)
            source_by_target[str(resolved)].append(str(source))
    return direct, {key: sorted(set(value)) for key, value in source_by_target.items()}


def _tree_stats(path: Path) -> dict[str, int]:
    if path.is_symlink() or path.is_file():
        try:
            return {"bytes": path.lstat().st_size, "files": 1, "directories": 0}
        except FileNotFoundError:
            return {"bytes": 0, "files": 0, "directories": 0}
    # GNU du performs this large-tree traversal in native code and is far
    # cheaper than repeating Python stat calls over old frame directories.
    result = subprocess.run(
        ["du", "-sb", "--", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"bytes": 0, "files": -1, "directories": -1}
    return {
        "bytes": int(result.stdout.split()[0]),
        "files": -1,
        "directories": -1,
    }


def _git_porcelain() -> tuple[str, list[str]]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line]
    return result.stdout, lines


def _git_touches(candidate: Path, lines: list[str]) -> list[str]:
    rel = str(candidate.resolve().relative_to(ROOT))
    matches = []
    for line in lines:
        raw = line[3:]
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        if raw == rel or raw.startswith(rel + "/") or rel.startswith(raw.rstrip("/") + "/"):
            matches.append(line)
    return matches[:50]


def _all_proc_refs(candidates: list[Path]) -> dict[str, list[dict[str, Any]]]:
    matches: dict[str, list[dict[str, Any]]] = {str(path): [] for path in candidates}
    resolved_candidates = [(path, path.resolve()) for path in candidates]
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        for kind, link in (("cwd", entry / "cwd"), ("exe", entry / "exe")):
            try:
                target = link.resolve(strict=True)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            for original, candidate in resolved_candidates:
                if _contained(target, candidate):
                    matches[str(original)].append({"pid": pid, "kind": kind, "path": str(target)})
        fd_root = entry / "fd"
        try:
            fds = list(fd_root.iterdir())
        except (FileNotFoundError, PermissionError, OSError):
            continue
        for fd in fds:
            try:
                target = fd.resolve(strict=True)
            except (FileNotFoundError, PermissionError, OSError):
                continue
            for original, candidate in resolved_candidates:
                if _contained(target, candidate):
                    matches[str(original)].append({"pid": pid, "kind": "fd", "fd": fd.name, "path": str(target)})
                    break
    return matches


def _all_tree_stats(paths: list[Path]) -> dict[str, dict[str, int]]:
    result = subprocess.run(
        ["du", "-sb", "--", *[str(path) for path in paths]],
        check=False,
        capture_output=True,
        text=True,
    )
    values = {
        str(path): {"bytes": 0, "files": -1, "directories": -1}
        for path in paths
    }
    for line in result.stdout.splitlines():
        if "\t" not in line:
            continue
        size, raw = line.split("\t", 1)
        values[raw] = {"bytes": int(size), "files": -1, "directories": -1}
    return values


def _bounded_unknown_stats(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """List every root without an unbounded recursive network-filesystem walk."""
    return {
        str(path): {
            "bytes": 0,
            "files": -1,
            "directories": -1,
            "measurement_status": "DEFERRED_UNBOUNDED_NETWORK_IO",
        }
        for path in paths
    }


def _direct_sources(candidate: Path, refs: set[Path], by_target: dict[str, list[str]]) -> list[str]:
    sources: set[str] = set()
    for target in refs:
        if _contained(target, candidate) or _contained(candidate, target):
            sources.update(by_target.get(str(target), []))
            if not by_target.get(str(target)):
                sources.add("CURRENT_RECEIPT_OR_MACHINE_LEDGER")
    return sorted(sources)


def _row(
    path: Path,
    refs: set[Path],
    by_target: dict[str, list[str]],
    git_lines: list[str],
    stats: dict[str, int],
    process: list[dict[str, Any]],
) -> dict[str, Any]:
    current_sources = _direct_sources(path, refs, by_target)
    git = _git_touches(path, git_lines)
    protected_prefix = next((str(prefix) for prefix in HARD_PROTECTED if _contained(path, prefix) or _contained(prefix, path)), None)
    blockers = []
    if current_sources:
        blockers.append("CURRENT_OR_ONE_HOP_REFERENCE")
    if git:
        blockers.append("GIT_DIRTY_OR_UNTRACKED_CLASSIFICATION_REQUIRED")
    if process:
        blockers.append("ACTIVE_PID_FD_OR_CWD")
    if protected_prefix:
        blockers.append("HARD_PROTECTED_PREFIX")
    if stats.get("measurement_status") != "MEASURED":
        blockers.append("EXACT_RECURSIVE_BYTES_NOT_MEASURED")
    return {
        "path": str(path),
        "stats": stats,
        "current_or_one_hop_sources": current_sources,
        "git_matches": git,
        "process_references": process,
        "protected_prefix": protected_prefix,
        "decision": "BLOCKED_REFERENCE_PROOF" if blockers else "READY_FOR_EVIDENCE_CAPSULE_REVIEW",
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--bounded-no-size-scan",
        action="store_true",
        help="Enumerate every candidate root but defer recursive byte measurement when network I/O is unbounded.",
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise RuntimeError(f"fresh immutable output required: {output}")
    output.mkdir(parents=True)
    refs, by_target = _references()
    git_text, git_lines = _git_porcelain()
    atomic_write(output / "GIT_CLASSIFICATION_SNAPSHOT.txt", git_text.encode())

    run_paths = [path for path in sorted(RUNS.iterdir()) if path.is_dir()]
    runtime_paths = [
        path
        for path in sorted(RUNTIME.iterdir())
        if path.name not in CONTROL_RUNTIME_NAMES and not path.name.endswith(".lock")
    ]
    paths = run_paths + runtime_paths
    stats_by_path = (
        _bounded_unknown_stats(paths)
        if args.bounded_no_size_scan
        else _all_tree_stats(paths)
    )
    if not args.bounded_no_size_scan:
        for value in stats_by_path.values():
            value["measurement_status"] = "MEASURED" if value["bytes"] > 0 else "MEASUREMENT_FAILED"
    proc_by_path = _all_proc_refs(paths)
    run_rows = [
        _row(path, refs, by_target, git_lines, stats_by_path[str(path)], proc_by_path[str(path)])
        for path in run_paths
    ]
    runtime_rows = [
        _row(path, refs, by_target, git_lines, stats_by_path[str(path)], proc_by_path[str(path)])
        for path in runtime_paths
    ]
    all_rows = run_rows + runtime_rows
    ready = [row for row in all_rows if row["decision"] == "READY_FOR_EVIDENCE_CAPSULE_REVIEW"]
    blocked = [row for row in all_rows if row["decision"] == "BLOCKED_REFERENCE_PROOF"]
    graph_path = output / "CLEANUP_REFERENCE_GRAPH.json"
    candidates_path = output / "CLEANUP_CANDIDATES.json"
    predelete_path = output / "PRE_DELETE_MANIFEST.json"
    _write = lambda p, v: atomic_json(p, v)
    _write(
        graph_path,
        {
            "schema_version": "chaoyang-cleanup-reference-graph-r72-v1",
            "generated_at": now_iso(),
            "current_reference_count": len(refs),
            "source_by_target": by_target,
            "hard_protected": [str(path) for path in HARD_PROTECTED],
            "external_protected": [str(path) for path in EXTERNAL_PROTECTED],
            "git_entry_count": len(git_lines),
        },
    )
    _write(
        candidates_path,
        {
            "schema_version": "chaoyang-cleanup-candidates-r72-v1",
            "generated_at": now_iso(),
            "status": "DRY_RUN_COMPLETE_WITH_SIZE_SCAN_DEFERRED" if args.bounded_no_size_scan else "DRY_RUN_COMPLETE",
            "rows": all_rows,
            "counts": {"total": len(all_rows), "ready_for_capsule_review": len(ready), "blocked_reference_proof": len(blocked)},
            "bytes": {"total": sum(row["stats"]["bytes"] for row in all_rows), "ready_for_capsule_review": sum(row["stats"]["bytes"] for row in ready)},
        },
    )
    _write(
        predelete_path,
        {
            "schema_version": "chaoyang-pre-delete-manifest-r72-v1",
            "generated_at": now_iso(),
            "status": "DRY_RUN_ONLY_NO_OLD_ASSET_DELETE_AUTHORIZED",
            "ready_for_evidence_capsule_review": ready,
            "blocked": blocked,
            "commit_prerequisites": ["v77_adopt_receipt_published", "r22_tasks_terminal", "git_classified", "doc_navigation_regression_passed", "evidence_capsules_verified"],
            "delete_executed": False,
        },
    )
    result_path = output / "RESULT.json"
    _write(
        result_path,
        {
            "schema_version": "chaoyang-cleanup-r72-dryrun-result-v1",
            "task_id": "cleanup_current_only_r72_dryrun",
            "status": "PASSED",
            "reference_graph": artifact_ref(graph_path),
            "candidates": artifact_ref(candidates_path),
            "pre_delete_manifest": artifact_ref(predelete_path),
            "old_assets_deleted": 0,
            "external_paths_touched": 0,
            "recursive_size_scan": "DEFERRED_UNBOUNDED_NETWORK_IO" if args.bounded_no_size_scan else "MEASURED",
            "claim_limit": "Dry-run classification only; no old run, history, tool, test or contract was deleted.",
        },
    )
    _write(output / "METRICS.json", {"status": "PASSED", "candidate_roots": len(all_rows), "ready_for_capsule_review": len(ready), "blocked_reference_proof": len(blocked), "ready_bytes": sum(row["stats"]["bytes"] for row in ready), "deleted_bytes": 0})
    atomic_write(output / "DECISION.md", b"# Cleanup R7.2 dry-run\n\nOld assets were classified only. Permanent deletion is deferred until the R2.2 and Git/document gates close.\n")
    _write(output / "NEXT_ACTION.json", {"task_id": "cleanup_current_only_r72_commit", "status": "BLOCKED_PREREQ", "prerequisites": ["v77_adopt_receipt_published", "r22_tasks_terminal", "git_classified", "doc_navigation_regression_passed", "evidence_capsules_verified"]})
    manifest_path = output / "ARTIFACT_MANIFEST.json"
    _write(manifest_path, {"schema_version": "chaoyang-cleanup-r72-artifact-manifest-v1", "artifacts": [artifact_ref(path) for path in (graph_path, candidates_path, predelete_path, result_path, output / "METRICS.json", output / "DECISION.md", output / "NEXT_ACTION.json", output / "GIT_CLASSIFICATION_SNAPSHOT.txt")]})
    _write(output / "RUN_RECEIPT.json", {"schema_version": "chaoyang-cleanup-r72-run-receipt-v1", "task_id": "cleanup_current_only_r72_dryrun", "status": "PASSED", "result": artifact_ref(result_path), "manifest": artifact_ref(manifest_path)})
    print(json.dumps({"status": "PASSED", "roots": len(all_rows), "ready": len(ready), "ready_bytes": sum(row["stats"]["bytes"] for row in ready), "deleted": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
