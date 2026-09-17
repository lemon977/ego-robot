#!/usr/bin/env python3
"""Build a deterministic content-hashed baseline inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

HISTORY_PREFIXES = (
    "_run/current/", "archive/baseline-20260917-0aa69e9/content/history/data/experiments/", "archive/baseline-20260917-0aa69e9/content/history/data/processed/", "archive/baseline-20260917-0aa69e9/content/history/docs/history/",
    "archive/baseline-20260917-0aa69e9/content/history/docs/organization/", "archive/baseline-20260917-0aa69e9/content/history/docs/collaboration/",
    "archive/baseline-20260917-0aa69e9/content/history/HumanEgo/artifacts/newtask_robot_bundles/",
    "archive/baseline-20260917-0aa69e9/content/history/HumanEgo/artifacts/newtask_robot_runs/", "archive/baseline-20260917-0aa69e9/content/history/HumanEgo/outputs/", "archive/baseline-20260917-0aa69e9/content/history/HumanEgo/_run/",
)
CACHE_PARTS = {"__pycache__", "archive/baseline-20260917-0aa69e9/content/regenerable/cache/.pytest_cache", "archive/baseline-20260917-0aa69e9/content/regenerable/cache/.ruff_cache", ".mypy_cache"}


def classify(path: str) -> str:
    parts = set(Path(path).parts)
    if parts & CACHE_PARTS or path.endswith((".pyc", ".pyo", ".log", ".lock", ".tmp")):
        return "REGENERABLE_CACHE"
    if path.startswith(HISTORY_PREFIXES):
        return "ARCHIVE_HISTORY"
    if path.startswith(("src/chaoyang/pipeline/", "src/chaoyang/human_ego/", "src/chaoyang/ops/", "configs/systems/")):
        return "MIGRATE_ACTIVE"
    if path.startswith(("docs/", "archive/baseline-20260917-0aa69e9/content/history/tasks/", "contracts/", "assets/")):
        return "REVIEW_REFERENCE_GRAPH"
    return "KEEP_OR_REVIEW"


def digest_file(path: Path) -> tuple[str, bool]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb", buffering=1024 * 1024) as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return digest.hexdigest(), stable


def merkle_root(leaves: list[bytes]) -> str:
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = leaves
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def iter_paths(root: Path):
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(root)
        skip_names = set(CACHE_PARTS)
        if relative_dir == Path("."):
            skip_names.update({".git", "archive"})
        if relative_dir == Path("assets"):
            skip_names.add("environments")
        directories[:] = sorted(name for name in directories if name not in skip_names)
        for name in sorted(files):
            yield current_path / name
        for name in sorted(directories):
            candidate = current_path / name
            if candidate.is_symlink():
                yield candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    leaves: list[bytes] = []
    counts: dict[str, int] = {}
    total_bytes = 0
    unstable = 0
    started = time.time()
    with partial.open("w", encoding="utf-8") as handle:
        for index, path in enumerate(iter_paths(root), start=1):
            relative = path.relative_to(root).as_posix()
            try:
                info = path.lstat()
                row: dict[str, object] = {
                    "path": relative,
                    "mode": stat.S_IMODE(info.st_mode),
                    "mtime_ns": info.st_mtime_ns,
                    "classification": classify(relative),
                }
                if path.is_symlink():
                    row.update(type="symlink", target=os.readlink(path), size=info.st_size)
                elif path.is_file():
                    sha256, stable = digest_file(path)
                    row.update(type="file", size=info.st_size, sha256=sha256, stable=stable)
                    total_bytes += info.st_size
                    unstable += int(not stable)
                else:
                    continue
            except (FileNotFoundError, PermissionError, OSError) as exc:
                row = {
                    "path": relative,
                    "classification": classify(relative),
                    "type": "error",
                    "error": type(exc).__name__,
                    "stable": False,
                }
                unstable += 1
            encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write(encoded + "\n")
            leaves.append(hashlib.sha256(encoded.encode("utf-8")).digest())
            key = str(row["classification"])
            counts[key] = counts.get(key, 0) + 1
            if index % 5000 == 0:
                print(f"inventory: {index} entries", file=sys.stderr, flush=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, output)
    summary = {
        "schema_version": 1,
        "root": str(root),
        "entry_count": len(leaves),
        "total_file_bytes": total_bytes,
        "unstable_entries": unstable,
        "classification_counts": dict(sorted(counts.items())),
        "merkle_root_sha256": merkle_root(leaves),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    summary_path = output.parent / "MERKLE_ROOT.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if unstable == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
