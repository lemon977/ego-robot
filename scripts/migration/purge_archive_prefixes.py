#!/usr/bin/env python3
"""Delete selected regenerable archive prefixes with a verifiable receipt."""

from __future__ import annotations

import argparse
import errno
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

RECEIPT_NAME = "PURGE_RECEIPT.json"
PARTIAL_NAME = "PURGE_RECEIPT.json.partial"
ALLOWED_PREFIX = "content/regenerable/"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalize_prefix(raw: str) -> str:
    value = raw.replace("\\", "/").strip("/") + "/"
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive prefix: {raw}")
    if not value.startswith(ALLOWED_PREFIX) or value == ALLOWED_PREFIX:
        raise ValueError(f"purge prefix must be below {ALLOWED_PREFIX}: {raw}")
    return value


def process_references(targets: list[Path]) -> list[dict[str, object]]:
    prefixes = tuple(str(path) for path in targets)
    hits: list[dict[str, object]] = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            command = Path(f"/proc/{name}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            command = ""
        slots = [("cwd", f"/proc/{name}/cwd"), ("exe", f"/proc/{name}/exe")]
        try:
            slots.extend((f"fd:{fd}", f"/proc/{name}/fd/{fd}") for fd in os.listdir(f"/proc/{name}/fd"))
        except OSError:
            pass
        for slot, candidate in slots:
            try:
                value = os.readlink(candidate)
            except OSError:
                continue
            if not value.startswith(prefixes):
                continue
            is_directory = Path(candidate).is_dir()
            readonly_observer = (
                slot.startswith("fd:")
                and is_directory
                and ".vscode-server" in command
                and "--type=fileWatcher" in command
            )
            hits.append({
                "pid": int(name),
                "slot": slot,
                "path": value,
                "is_directory": is_directory,
                "command": command,
                "classification": "READONLY_IDE_OBSERVER" if readonly_observer else "BLOCKING",
                "blocking": not readonly_observer,
            })
    return hits


def exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def quarantine_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.chaoyang-purge")


def retry_operation(label: str, action, still_needed, attempts: int = 240) -> None:
    for attempt in range(1, attempts + 1):
        try:
            action()
            return
        except FileNotFoundError:
            return
        except OSError as error:
            if error.errno not in {errno.EBUSY, errno.ENOTEMPTY}:
                raise
            if not still_needed():
                return
            print(json.dumps({
                "status": "RETRYING",
                "operation": label,
                "attempt": attempt,
                "errno": error.errno,
                "error": str(error),
            }), flush=True)
            time.sleep(min(5.0, 0.25 * attempt))
    raise RuntimeError(f"retry limit exceeded: {label}")


def purge_target(target: Path, prefix: str) -> None:
    quarantine = quarantine_path(target)
    if exists(target):
        if exists(quarantine):
            raise RuntimeError(f"both target and quarantine exist: {target}")
        retry_operation(
            f"rename:{prefix}",
            lambda: os.replace(target, quarantine),
            lambda: exists(target),
        )
        print(json.dumps({"status": "QUARANTINED", "prefix": prefix}), flush=True)
    if not exists(quarantine):
        print(json.dumps({"status": "ALREADY_ABSENT", "prefix": prefix}), flush=True)
        return
    retry_operation(
        f"delete:{prefix}",
        lambda: quarantine.unlink() if quarantine.is_symlink() or quarantine.is_file() else shutil.rmtree(quarantine),
        lambda: exists(quarantine),
    )
    print(json.dumps({"status": "DELETED", "prefix": prefix}), flush=True)


def inventory_groups(root: Path, prefixes: list[str]) -> list[dict[str, object]]:
    groups = {
        prefix: {"prefix": prefix, "entries": 0, "files": 0, "symlinks": 0, "bytes": 0, "raw": []}
        for prefix in prefixes
    }
    with (root / "INVENTORY.jsonl").open(encoding="utf-8") as handle:
        for raw in handle:
            encoded = raw.rstrip("\n")
            row = json.loads(encoded)
            matches = [prefix for prefix in prefixes if row["path"].startswith(prefix)]
            if not matches:
                continue
            if len(matches) != 1:
                raise RuntimeError(f"overlapping purge prefixes for {row['path']}")
            group = groups[matches[0]]
            group["entries"] += 1
            group["files"] += int(row["type"] == "file")
            group["symlinks"] += int(row["type"] == "symlink")
            group["bytes"] += int(row["bytes"])
            group["raw"].append(encoded)
    result = []
    for prefix in prefixes:
        group = groups[prefix]
        if not group["entries"]:
            raise RuntimeError(f"prefix has no inventory entries: {prefix}")
        digest = hashlib.sha256()
        for raw in group.pop("raw"):
            digest.update(raw.encode("utf-8") + b"\n")
        group["inventory_rows_sha256"] = digest.hexdigest()
        result.append(group)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prefix", action="append", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--expected-merkle")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = args.archive.resolve(strict=True)
    receipt_path = root / RECEIPT_NAME
    partial_path = root / PARTIAL_NAME
    if receipt_path.exists():
        raise RuntimeError("archive already has a completed purge receipt")
    if partial_path.exists() and not args.resume:
        raise RuntimeError("partial purge receipt exists; use --resume after inspection")
    if args.resume and not partial_path.exists():
        raise RuntimeError("--resume requested but no partial purge receipt exists")

    prefixes = sorted({normalize_prefix(value) for value in args.prefix})
    for index, left in enumerate(prefixes):
        for right in prefixes[index + 1:]:
            if right.startswith(left):
                raise ValueError(f"overlapping purge prefixes: {left}, {right}")

    summary = json.loads((root / "MERKLE_ROOT.json").read_text(encoding="utf-8"))
    merkle = summary["merkle_root_sha256"]
    if args.expected_merkle and args.expected_merkle != merkle:
        raise RuntimeError(f"archive Merkle drift: {merkle}")
    groups = inventory_groups(root, prefixes)
    targets = [(root / prefix.rstrip("/")).resolve(strict=False) for prefix in prefixes]
    for target in targets:
        target.relative_to(root)
    reference_targets = [
        candidate
        for target in targets
        for candidate in (target, quarantine_path(target))
        if exists(candidate)
    ]
    references = process_references(reference_targets)
    blocking_references = [row for row in references if row["blocking"]]
    fresh_receipt: dict[str, Any] = {
        "schema_version": "chaoyang-archive-purge-receipt-v1",
        "status": "PLANNED",
        "archive_root": str(root),
        "archive_merkle_root_before_purge": merkle,
        "reason": args.reason,
        "created_at": now(),
        "process_references_before_purge": references,
        "blocking_process_references_before_purge": blocking_references,
        "purged_prefixes": groups,
        "totals": {
            "entries": sum(int(group["entries"]) for group in groups),
            "files": sum(int(group["files"]) for group in groups),
            "symlinks": sum(int(group["symlinks"]) for group in groups),
            "bytes": sum(int(group["bytes"]) for group in groups),
        },
    }
    if args.resume:
        receipt = json.loads(partial_path.read_text(encoding="utf-8"))
        bindings = (
            receipt.get("schema_version") == fresh_receipt["schema_version"]
            and receipt.get("status") == "PLANNED"
            and receipt.get("archive_root") == fresh_receipt["archive_root"]
            and receipt.get("archive_merkle_root_before_purge") == merkle
            and receipt.get("reason") == args.reason
            and receipt.get("purged_prefixes") == groups
            and receipt.get("totals") == fresh_receipt["totals"]
        )
        if not bindings:
            raise RuntimeError("partial purge receipt does not match requested resume")
        receipt.setdefault("resume_attempts", []).append({
            "resumed_at": now(),
            "process_references": references,
            "blocking_process_references": blocking_references,
        })
    else:
        receipt = fresh_receipt

    if not args.execute:
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    if blocking_references:
        raise RuntimeError({"active_process_references": blocking_references})

    if not args.resume:
        atomic_json(partial_path, receipt)
    else:
        atomic_json(partial_path, receipt)
    for prefix, target in zip(prefixes, targets, strict=True):
        print(json.dumps({"status": "DELETING", "prefix": prefix}), flush=True)
        purge_target(target, prefix)
    remaining = [
        prefix for prefix, target in zip(prefixes, targets, strict=True)
        if exists(target) or exists(quarantine_path(target))
    ]
    if remaining:
        raise RuntimeError({"purge_targets_still_exist": remaining})
    receipt["status"] = "COMPLETE"
    receipt["completed_at"] = now()
    atomic_json(receipt_path, receipt)
    partial_path.unlink()
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
