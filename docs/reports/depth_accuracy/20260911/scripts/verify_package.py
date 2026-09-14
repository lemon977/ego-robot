#!/usr/bin/env python3
"""Build once or verify the self-contained depth accuracy convenience package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PACKAGE_MANIFEST.json"
SUMS = ROOT / "PACKAGE_SHA256SUMS.txt"
EXCLUDED = {MANIFEST.resolve(), SUMS.resolve()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_files() -> list[dict[str, object]]:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.resolve() in EXCLUDED:
            continue
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def write_atomic(path: Path, content: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def build(*, overwrite: bool) -> None:
    files = current_files()
    payload = {
        "schema_version": "depth-accuracy-convenience-package-v1",
        "status": "COPIED_SELF_CONTAINED_INDEX",
        "copied_not_moved": True,
        "source_project": "/mnt/workspace/code/chaoyang",
        "original_final_receipt": (
            "/mnt/workspace/code/chaoyang/tasks/control/runs/"
            "20260911_depth_accuracy_spatial_diagnostic_v1/FINAL_RECEIPT.json"
        ),
        "authority": False,
        "claim_limit": (
            "Convenience copy only; original task receipts remain authoritative. "
            "No external ground truth or millimeter Robot-contact claim."
        ),
        "counts": {
            "files": len(files),
            "bytes": sum(int(row["bytes"]) for row in files),
        },
        "files": files,
    }
    sums = "".join(
        f"{row['sha256']}  {row['path']}\n" for row in files
    )
    write_atomic(SUMS, sums.encode("utf-8"), overwrite=overwrite)
    write_atomic(
        MANIFEST,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        overwrite=overwrite,
    )
    print(json.dumps({"status": "BUILT", **payload["counts"]}, ensure_ascii=False))


def verify() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = payload["files"]
    actual = current_files()
    if actual != expected:
        expected_map = {row["path"]: row for row in expected}
        actual_map = {row["path"]: row for row in actual}
        missing = sorted(set(expected_map) - set(actual_map))
        unexpected = sorted(set(actual_map) - set(expected_map))
        changed = sorted(
            path
            for path in set(expected_map) & set(actual_map)
            if expected_map[path] != actual_map[path]
        )
        raise RuntimeError(
            f"package verification failed: missing={missing}, "
            f"unexpected={unexpected}, changed={changed}"
        )
    expected_sums = "".join(
        f"{row['sha256']}  {row['path']}\n" for row in expected
    )
    if SUMS.read_text(encoding="utf-8") != expected_sums:
        raise RuntimeError("PACKAGE_SHA256SUMS.txt differs from manifest")
    print(
        json.dumps(
            {
                "status": "PASS",
                "files": len(actual),
                "bytes": sum(int(row["bytes"]) for row in actual),
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.build and args.refresh:
        raise RuntimeError("choose only one of --build or --refresh")
    if args.build:
        build(overwrite=False)
    elif args.refresh:
        build(overwrite=True)
    else:
        verify()


if __name__ == "__main__":
    main()
