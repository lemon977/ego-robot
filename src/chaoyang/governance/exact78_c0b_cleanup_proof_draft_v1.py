"""Unregistered C0b read-only proof prototype; never deletes or publishes.

The only CLI mode is --check-only. A future separately reviewed Task Packet may
call build_proof and seal its result. Incomplete enumeration, opaque files,
mutable targets, current references and legacy algorithm code fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "docs/plans/EXACT78_RAW_TO_ROBOT_V1_20260930/C0B_CANDIDATES_V1.json"
TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".md", ".txt", ".toml", ".yaml",
                           ".yml", ".py", ".sh", ".csv", ".ini", ".cfg"})
CONTENT_ROOTS = ("archive", "configs", "contracts", "docs", "governance", "manifests",
                 "review", "_run", "scripts", "src", "tasks", "tests", ".cache")
TRUSTED_NONCONSUMER_ROOTS = (".git",)
INTENTIONALLY_OMITTED = ("assets", "vendor", ".tmp", ".pytest_cache", " src",
                         "_run/current/environments", "_run/caches")
MAX_ENTRIES = 50000
MAX_TEXT_BYTES = 256 << 20
MAX_SINGLE_TEXT_BYTES = 16 << 20
MAX_ENUM_SECONDS = 40.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved)}


def inventory() -> list[dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "exact78-c0b-project-local-candidates-v1"
            or manifest.get("project_root") != str(ROOT)
            or manifest.get("status") != "DRAFT_NOT_REGISTERED_NOT_DELETION_AUTHORITY"):
        raise RuntimeError("C0B_MANIFEST_CONTRACT_DRIFT")
    rows, seen = [], set()
    for item in manifest["targets"]:
        relative = item["path"]
        path = Path(relative)
        if (path.is_absolute() or ".." in path.parts or relative in seen
                or any(part in {"0916", "robot100"} for part in path.parts)):
            raise RuntimeError(f"C0B_UNSAFE_OR_DUPLICATE_TARGET:{relative}")
        seen.add(relative)
        absolute = ROOT / path
        if absolute.is_symlink() or not absolute.is_file() or absolute.resolve() != absolute:
            raise RuntimeError(f"C0B_TARGET_NOT_EXACT_REGULAR_FILE:{relative}")
        if item["kind"] == "legacy_algorithm_code_retention_probe":
            if relative not in {
                "src/chaoyang/ops/run_exact78_hawor_raw_v1.py",
                "src/chaoyang/ops/run_exact78_hawor_raw_v2.py",
            }:
                raise RuntimeError(f"C0B_CODE_TARGET_NOT_ALLOWLISTED:{relative}")
        elif item["kind"] == "generated_cache":
            if relative != ".cache/s2/matplotlib/fontlist-v390.json":
                raise RuntimeError(f"C0B_CACHE_TARGET_NOT_ALLOWLISTED:{relative}")
        elif item["kind"] == "old_diagnostic_media":
            if (not relative.startswith("_run/maintenance/")
                    or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}):
                raise RuntimeError(f"C0B_MEDIA_TARGET_NOT_ALLOWLISTED:{relative}")
        else:
            raise RuntimeError(f"C0B_UNKNOWN_TARGET_KIND:{relative}")
        rows.append({"path": relative, "kind": item["kind"], "file": ref(absolute)})
    if not rows or len(rows) > 100:
        raise RuntimeError("C0B_CANDIDATE_COUNT_OUT_OF_BOUNDS")
    return rows


def reference_universe(*, max_entries: int = MAX_ENTRIES,
                       max_seconds: float = MAX_ENUM_SECONDS) -> dict[str, Any]:
    """Bounded whole-project enumeration; any gap invalidates C1 proof."""
    started = time.monotonic()
    files, omissions, opaque = [], [], []
    entries = 0
    expected = set(CONTENT_ROOTS) | set(TRUSTED_NONCONSUMER_ROOTS) | set(INTENTIONALLY_OMITTED)
    for child in ROOT.iterdir():
        if child.is_dir() and child.name not in expected:
            omissions.append(f"UNCLASSIFIED_ROOT_DIR:{child.name}")
        elif child.is_file() and child.suffix.lower() in TEXT_SUFFIXES:
            files.append(child)
        elif child.is_file():
            opaque.append(str(child.relative_to(ROOT)))
    stack = [ROOT / name for name in CONTENT_ROOTS if (ROOT / name).is_dir()]
    while stack:
        if entries >= max_entries or time.monotonic() - started > max_seconds:
            omissions.append("ENUMERATION_BUDGET_EXCEEDED")
            break
        directory = stack.pop()
        rel_dir = str(directory.relative_to(ROOT))
        if any(rel_dir == excluded or rel_dir.startswith(excluded + "/")
               for excluded in INTENTIONALLY_OMITTED):
            omissions.append(f"INTENTIONALLY_OMITTED:{rel_dir}")
            continue
        try:
            with os.scandir(directory) as scan:
                children = list(scan)
        except OSError as error:
            omissions.append(f"UNREADABLE_DIRECTORY:{rel_dir}:{type(error).__name__}")
            continue
        for entry in children:
            entries += 1
            path = Path(entry.path)
            relative = str(path.relative_to(ROOT))
            if entry.is_symlink():
                omissions.append(f"SYMLINK_UNRESOLVED:{relative}")
            elif entry.is_dir(follow_symlinks=False):
                stack.append(path)
            elif entry.is_file(follow_symlinks=False):
                if path.suffix.lower() in TEXT_SUFFIXES:
                    files.append(path)
                else:
                    opaque.append(relative)
            else:
                omissions.append(f"SPECIAL_FILE:{relative}")
            if entries >= max_entries:
                break
    for excluded in INTENTIONALLY_OMITTED:
        if (ROOT / excluded).exists() and not any(
                entry.startswith(f"INTENTIONALLY_OMITTED:{excluded}") for entry in omissions):
            omissions.append(f"INTENTIONALLY_OMITTED:{excluded}")
    files = sorted(set(files))
    return {"files": files, "omissions": sorted(set(omissions)),
            "opaque_files": sorted(set(opaque)), "entries_seen": entries,
            "enumeration_seconds": round(time.monotonic() - started, 3),
            "complete": not omissions and not opaque}


def scan_references(candidates: list[dict[str, Any]], universe: dict[str, Any]) -> dict[str, Any]:
    """Scan exact text bytes; skip only the inventory itself and target self-bytes."""
    candidate_paths = {row["path"] for row in candidates}
    exclusions = candidate_paths | {str(MANIFEST.relative_to(ROOT))}
    tokens = {row["path"]: tuple(value.encode("utf-8") for value in (
        row["path"], str(ROOT / row["path"]), Path(row["path"]).name,
        row["file"]["sha256"],
    )) for row in candidates}
    hits = {row["path"]: [] for row in candidates}
    scanned, bytes_read, errors = 0, 0, []
    for path in universe["files"]:
        relative = str(path.relative_to(ROOT))
        if relative in exclusions:
            continue
        try:
            size = path.stat().st_size
            if size > MAX_SINGLE_TEXT_BYTES or bytes_read + size > MAX_TEXT_BYTES:
                errors.append(f"TEXT_SCAN_BUDGET_EXCEEDED:{relative}")
                break
            data = path.read_bytes()
            if len(data) != size:
                errors.append(f"CONCURRENT_TEXT_SIZE_CHANGE:{relative}")
            bytes_read += len(data)
            scanned += 1
            for target, values in tokens.items():
                if any(value in data for value in values):
                    hits[target].append(relative)
        except OSError as error:
            errors.append(f"TEXT_READ_ERROR:{relative}:{type(error).__name__}")
    return {"hits": hits, "errors": errors, "scanned_text_files": scanned,
            "scanned_text_bytes": bytes_read,
            "complete": universe["complete"] and not errors}


def build_proof() -> dict[str, Any]:
    """For future governed C0b only; safe to call but never authorizes deletion."""
    from chaoyang.governance.execute_exact78_c0_cleanup_proof_v1 import git_state, process_holders

    candidates = inventory()
    universe = reference_universe()
    references = scan_references(candidates, universe)
    processes = process_holders(candidates)
    git = git_state(candidates)
    rows = []
    for candidate in candidates:
        relative = candidate["path"]
        hits = references["hits"][relative]
        current_hits = [path for path in hits if path.startswith(("docs/current/", "docs/governance/"))
                        or path == "tasks/current/INDEX.json"]
        historical_hits = [path for path in hits if path not in current_hits]
        holders = processes["holders"][relative]
        try:
            target_stable = ref(ROOT / relative) == candidate["file"]
        except OSError:
            target_stable = False
        blockers = []
        if current_hits:
            blockers.append("CURRENT_REFERENCE")
        if historical_hits:
            blockers.append("HISTORICAL_REFERENCE")
        if holders or processes["errors"]:
            blockers.append("LIVE_HOLDER_OR_PROCESS_SCAN_INCOMPLETE")
        if not references["complete"]:
            blockers.append("REFERENCE_PROOF_INCOMPLETE")
        if not target_stable:
            blockers.append("TARGET_SHA_DRIFT")
        if git.get("status") != "OK" or git.get("rows", {}).get(relative, {}).get("dirty_or_untracked"):
            blockers.append("GIT_DIRTY_OR_UNPROVEN")
        if candidate["kind"] == "legacy_algorithm_code_retention_probe":
            blockers.append("ALGORITHM_DEPENDENCY_CLOSURE_AND_REGRESSION_NOT_PROVEN")
        if candidate["kind"] == "old_diagnostic_media":
            blockers.append("FAILURE_EVIDENCE_RETENTION_AND_LICENSE_NOT_PROVEN")
        if candidate["kind"] == "generated_cache":
            blockers.append("CACHE_PROVENANCE_NOT_YET_RECONFIRMED")
        rows.append({**candidate, "current_refs": current_hits,
                     "historical_refs": historical_hits, "live_holders": holders,
                     "target_sha_stable": target_stable, "git": git.get("rows", {}).get(relative),
                     "proof_status": "MUST_KEEP" if current_hits or holders else "BLOCKED_REFERENCE_PROOF",
                     "blockers": sorted(set(blockers)), "c1_delete_authorized": False})
    return {"schema_version": "exact78-c0b-readonly-proof-draft-v1",
            "status": "AUDIT_ONLY_NOT_REGISTERED", "candidate_manifest": ref(MANIFEST),
            "target_count": len(rows), "target_bytes": sum(row["file"]["bytes"] for row in rows),
            "reference_universe": {key: value for key, value in universe.items() if key != "files"},
            "reference_scan": {key: value for key, value in references.items() if key != "hits"},
            "process_scan_errors": processes["errors"], "git_status": git["status"],
            "rows": rows, "deleted_count": 0, "c1_delete_authorized": False,
            "claim_limit": "Draft evidence only. No deletion until separate C1 packet, complete references, code closure and retention proof."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", required=True)
    parser.parse_args()
    rows = inventory()
    print(json.dumps({"status": "CHECK_ONLY_NO_SCAN_NO_WRITES",
                      "candidate_manifest": ref(MANIFEST),
                      "target_count": len(rows),
                      "target_bytes": sum(row["file"]["bytes"] for row in rows),
                      "kinds": {kind: sum(row["kind"] == kind for row in rows)
                                for kind in sorted({row["kind"] for row in rows})},
                      "reference_roots": CONTENT_ROOTS,
                      "known_omissions": INTENTIONALLY_OMITTED,
                      "c1_delete_authorized": False}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
