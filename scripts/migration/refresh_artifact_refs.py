#!/usr/bin/env python3
"""Refresh repository-local {path, bytes, sha256} artifact references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCAN_ROOTS = (
    "docs/governance", "docs/current", "configs/systems", "contracts",
    "tasks/current", "tasks/receipts", "manifests",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_files(root: Path):
    for relative_root in SCAN_ROOTS:
        start = root / relative_root
        if not start.exists():
            continue
        yield from sorted(start.rglob("*.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    cache: dict[tuple[str, int, int], str] = {}
    changed_files: set[str] = set()
    unresolved: set[tuple[str, str]] = set()

    def local_candidate(raw: str) -> Path | None:
        path = Path(raw)
        if path.is_absolute():
            try:
                path.relative_to(root)
            except ValueError:
                return None
            return path
        return (root / path).resolve()

    def refresh(node: Any, source: Path) -> bool:
        changed = False
        if isinstance(node, dict):
            is_ref = (
                isinstance(node.get("path"), str)
                and isinstance(node.get("bytes"), int)
                and isinstance(node.get("sha256"), str)
                and len(node.get("sha256", "")) == 64
            )
            if is_ref:
                raw = str(node.get("path", ""))
                candidate = local_candidate(raw)
                if candidate is not None:
                    if candidate.is_file():
                        stat = candidate.stat()
                        key = (str(candidate), stat.st_size, stat.st_mtime_ns)
                        digest = cache.get(key)
                        if digest is None:
                            digest = sha256_file(candidate)
                            cache[key] = digest
                        expected = {
                            "path": str(candidate),
                            "bytes": stat.st_size,
                            "sha256": digest,
                        }
                        for name, value in expected.items():
                            if node.get(name) != value:
                                node[name] = value
                                changed = True
                    else:
                        unresolved.add((str(source.relative_to(root)), raw))
            for value in node.values():
                changed = refresh(value, source) or changed
        elif isinstance(node, list):
            for value in node:
                changed = refresh(value, source) or changed
        return changed

    for pass_number in range(1, args.passes + 1):
        pass_changes = 0
        unresolved.clear()
        for path in json_files(root):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not refresh(value, path):
                continue
            mode = path.stat().st_mode
            temporary = path.with_name(f".{path.name}.refs.tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            os.chmod(temporary, mode)
            os.replace(temporary, path)
            changed_files.add(str(path.relative_to(root)))
            pass_changes += 1
        print(json.dumps({"pass": pass_number, "changed_files": pass_changes, "unresolved": len(unresolved)}))
        if pass_changes == 0:
            break

    result = {
        "schema_version": 1,
        "changed_files": sorted(changed_files),
        "changed_count": len(changed_files),
        "unresolved": [
            {"source": source, "path": path}
            for source, path in sorted(unresolved)
        ],
    }
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"changed_count": len(changed_files), "unresolved": len(unresolved)}))
    return 0 if not unresolved else 2


if __name__ == "__main__":
    raise SystemExit(main())
