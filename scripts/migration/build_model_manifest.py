#!/usr/bin/env python3
"""Build a deterministic SHA-256 inventory for local model payloads."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

EXCLUDE_NAMES = {"MANIFEST.json", "README.md"}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("assets/models"))
    parser.add_argument("--output", type=Path, default=Path("assets/models/MANIFEST.json"))
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    rows = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.name in EXCLUDE_NAMES or path.is_dir():
            continue
        relative = path.relative_to(root.parent.parent).as_posix()
        if path.is_symlink():
            rows.append({"path": relative, "type": "symlink", "target": os.readlink(path)})
            continue
        size = path.stat().st_size
        rows.append({"path": relative, "type": "file", "bytes": size, "sha256": sha256_file(path)})
        total += size
    value = {
        "schema_version": "chaoyang-model-assets-v2",
        "payload_policy": "Local model files are ignored by Git; every retained payload is SHA-256 bound here.",
        "files": rows,
        "summary": {"entries": len(rows), "files": sum(r["type"] == "file" for r in rows), "bytes": total},
    }
    output = args.output.resolve()
    temporary = output.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(value["summary"], sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
