#!/usr/bin/env python3
"""Verify an archive inventory and perform a staged, non-destructive restore drill."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=1024 * 1024) as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merkle(leaves: list[bytes]) -> str:
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


def load_purge_receipt(root: Path, summary: dict[str, Any]) -> dict[str, Any] | None:
    path = root / "PURGE_RECEIPT.json"
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "chaoyang-archive-purge-receipt-v1":
        raise RuntimeError("unsupported purge receipt schema")
    if receipt.get("status") != "COMPLETE":
        raise RuntimeError("purge receipt is not COMPLETE")
    if receipt.get("archive_merkle_root_before_purge") != summary["merkle_root_sha256"]:
        raise RuntimeError("purge receipt is bound to a different archive Merkle root")
    prefixes = [row.get("prefix") for row in receipt.get("purged_prefixes", [])]
    if not prefixes or any(not isinstance(value, str) for value in prefixes):
        raise RuntimeError("purge receipt has no valid prefixes")
    for index, left in enumerate(sorted(prefixes)):
        for right in sorted(prefixes)[index + 1:]:
            if right.startswith(left):
                raise RuntimeError(f"overlapping purge receipt prefixes: {left}, {right}")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    root = args.archive.resolve(strict=True)
    summary = json.loads((root / "MERKLE_ROOT.json").read_text(encoding="utf-8"))
    purge_receipt = load_purge_receipt(root, summary)
    purge_expected = {
        row["prefix"]: row for row in (purge_receipt or {}).get("purged_prefixes", [])
    }
    purge_observed = {
        prefix: {
            "entries": 0,
            "files": 0,
            "symlinks": 0,
            "bytes": 0,
            "digest": hashlib.sha256(),
        }
        for prefix in purge_expected
    }

    rows: list[dict[str, Any]] = []
    leaves: list[bytes] = []
    missing: list[str] = []
    mismatched: list[str] = []
    purge_still_present: list[str] = []
    for raw in (root / "INVENTORY.jsonl").read_text(encoding="utf-8").splitlines():
        leaves.append(hashlib.sha256(raw.encode()).digest())
        row = json.loads(raw)
        rows.append(row)
        purge_matches = [prefix for prefix in purge_expected if row["path"].startswith(prefix)]
        if len(purge_matches) > 1:
            raise RuntimeError(f"overlapping purge prefixes for {row['path']}")
        path = root / row["path"]
        if purge_matches:
            group = purge_observed[purge_matches[0]]
            group["entries"] += 1
            group["files"] += int(row["type"] == "file")
            group["symlinks"] += int(row["type"] == "symlink")
            group["bytes"] += int(row["bytes"])
            group["digest"].update(raw.encode("utf-8") + b"\n")
            if path.exists() or path.is_symlink():
                purge_still_present.append(row["path"])
            continue
        if not (path.exists() or path.is_symlink()):
            missing.append(row["path"])
            continue
        info = path.lstat()
        if info.st_size != row["bytes"]:
            mismatched.append(row["path"])
        if row["type"] == "symlink" and os.readlink(path) != row["target"]:
            mismatched.append(row["path"])

    purge_errors: list[dict[str, object]] = []
    for prefix, expected in purge_expected.items():
        observed = purge_observed[prefix]
        actual = {
            "entries": observed["entries"],
            "files": observed["files"],
            "symlinks": observed["symlinks"],
            "bytes": observed["bytes"],
            "inventory_rows_sha256": observed["digest"].hexdigest(),
        }
        wanted = {name: expected.get(name) for name in actual}
        if actual != wanted:
            purge_errors.append({"prefix": prefix, "expected": wanted, "observed": actual})
    if purge_receipt:
        totals = purge_receipt.get("totals", {})
        calculated = {
            name: sum(int(row.get(name, 0)) for row in purge_expected.values())
            for name in ("entries", "files", "symlinks", "bytes")
        }
        if totals != calculated:
            purge_errors.append({"totals_expected": totals, "totals_observed": calculated})

    observed_root = merkle(leaves)
    if (
        observed_root != summary["merkle_root_sha256"]
        or len(rows) != summary["entries"]
        or missing
        or mismatched
        or purge_still_present
        or purge_errors
        or summary["unstable_entries"]
    ):
        raise RuntimeError({
            "merkle": observed_root,
            "missing": missing[:100],
            "mismatched": mismatched[:100],
            "purge_still_present": purge_still_present[:100],
            "purge_errors": purge_errors,
        })

    by_path = {row["path"]: row for row in rows}
    purged_prefixes = tuple(purge_expected)
    has_git_bundle = "git/repository.bundle" in by_path
    if has_git_bundle:
        samples = ["git/repository.bundle", "git/worktree.patch", "MOVES.tsv"]
    else:
        samples = []
        if "MOVES.tsv" in by_path:
            samples.append("MOVES.tsv")
        samples.extend(
            row["path"] for row in rows
            if row["type"] == "file"
            and row["path"].startswith("content/")
            and not row["path"].startswith(purged_prefixes)
            and row["path"] not in samples
        )
        samples = samples[:3]
    if not samples:
        raise RuntimeError("archive has no restorable file samples")

    staging = root / ".staging"
    staging.mkdir(exist_ok=True)
    drill = Path(tempfile.mkdtemp(prefix="restore-drill-", dir=staging))
    restored = []
    for relative in samples:
        row = by_path[relative]
        source = root / relative
        target = drill / "samples" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        observed = sha(target)
        if observed != row["sha256"]:
            raise RuntimeError(f"sample SHA mismatch: {relative}")
        restored.append({"path": relative, "bytes": row["bytes"], "sha256": observed})

    bundle_verify = "NOT_APPLICABLE"
    patch_checks = []
    if has_git_bundle:
        bundle = root / "git/repository.bundle"
        verify = subprocess.run(
            ["git", "bundle", "verify", str(bundle)],
            cwd=root, text=True, capture_output=True, check=False,
        )
        if verify.returncode:
            raise RuntimeError(verify.stderr)
        clone = drill / "bundle-clone"
        subprocess.run(
            ["git", "clone", "--no-checkout", str(bundle), str(clone)],
            cwd=root, check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", "pre-clean-20260917-0aa69e9"],
            cwd=clone, check=True, capture_output=True, text=True,
        )
        bundle_verify = "PASS"
        for relative in ("git/worktree.patch", "git/index.patch"):
            if relative not in by_path:
                continue
            patch = root / relative
            if patch.stat().st_size == 0:
                patch_checks.append({"path": relative, "status": "SKIPPED_EMPTY"})
                continue
            result = subprocess.run(
                ["git", "apply", "--check", str(patch)],
                cwd=clone, text=True, capture_output=True, check=False,
            )
            patch_checks.append({
                "path": relative,
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "stderr": result.stderr[-2000:],
            })
            if result.returncode:
                raise RuntimeError(patch_checks[-1])

    purge_summary = None
    if purge_receipt:
        purge_summary = {
            "status": "VERIFIED",
            "prefixes": sorted(purge_expected),
            **purge_receipt["totals"],
        }
    result = {
        "schema_version": 2,
        "status": "PASS",
        "inventory_entries": len(rows),
        "merkle_root_sha256": observed_root,
        "all_inventory_entries_accounted_for": True,
        "all_paths_and_sizes_verified": purge_receipt is None,
        "all_present_paths_and_sizes_verified": True,
        "receipted_purge": purge_summary,
        "sample_restores": restored,
        "bundle_verify": bundle_verify,
        "patch_checks": patch_checks,
        "staging_path": str(drill),
    }
    (root / "RESTORE_DRILL.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
