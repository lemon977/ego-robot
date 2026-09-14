from __future__ import annotations

"""Transactionally move the current depth report package under docs/reports.

The package bytes are preserved exactly.  Historical immutable receipts inside
the package are not rewritten; PATH_REDIRECTS.json records their relocation.
Current governance authority paths are changed with a CAS publish before the
old root is deleted.
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    TASK_STATE_PATH,
    load_json,
    publish_bundle,
)


ROOT = Path("/mnt/workspace/code/chaoyang")
OLD = ROOT / "DEPTH_ACCURACY_PACKAGE_20260911"
NEW = ROOT / "docs/reports/depth_accuracy/20260911"
RUN = ROOT / "tasks/control/runs/20260914_chaoyang_cleanup_v6"
REDIRECTS = ROOT / "docs/governance/PATH_REDIRECTS.json"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def manifest(root: Path) -> dict[str, Any]:
    rows = []
    merkle = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and not item.is_symlink()):
        row = {
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        rows.append(row)
        merkle.update(
            f"{row['relative_path']}\0{row['bytes']}\0{row['sha256']}\n".encode()
        )
    return {
        "root": str(root),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "merkle_sha256": merkle.hexdigest(),
        "files": rows,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o444)
    os.replace(temporary, path)


def replace_prefix(value: Any, old: str, new: str) -> Any:
    if isinstance(value, dict):
        return {key: replace_prefix(child, old, new) for key, child in value.items()}
    if isinstance(value, list):
        return [replace_prefix(child, old, new) for child in value]
    if isinstance(value, str) and (value == old or value.startswith(old + "/")):
        return new + value[len(old):]
    return value


def process_refs(root: Path) -> list[dict[str, Any]]:
    prefix = str(root)
    rows = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        found = []
        for name in ("cwd", "exe"):
            try:
                value = os.readlink(process / name)
            except OSError:
                continue
            if value == prefix or value.startswith(prefix + "/"):
                found.append({"source": name, "value": value})
        try:
            descriptors = list((process / "fd").iterdir())
        except OSError:
            descriptors = []
        for descriptor in descriptors:
            try:
                value = os.readlink(descriptor)
            except OSError:
                continue
            if value == prefix or value.startswith(prefix + "/"):
                found.append({"source": f"fd:{descriptor.name}", "value": value})
        if found:
            rows.append({"pid": int(process.name), "references": found})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    if not OLD.is_dir():
        if NEW.is_dir():
            print(json.dumps({"status": "ALREADY_MIGRATED", "path": str(NEW)}))
            return 0
        raise SystemExit("neither source nor destination exists")
    if process_refs(OLD):
        raise SystemExit("source has active FD/CWD references")
    before = manifest(OLD)
    preflight = {
        "schema_version": "chaoyang-depth-package-migration-v6-preflight-v1",
        "created_at": now(),
        "host": socket.gethostname(),
        "status": "READY_COMMIT" if not NEW.exists() else "DESTINATION_EXISTS_VERIFY_REQUIRED",
        "source": before,
        "destination": str(NEW),
        "active_references": [],
        "preservation": "File bytes and relative paths remain exact; immutable historical internal paths resolve through PATH_REDIRECTS.json.",
    }
    write_json(RUN / "receipts/DEPTH_PACKAGE_MIGRATION_PRE_DELETE.json", preflight)
    if not args.commit:
        print(json.dumps({"status": "DRY_RUN", "files": before["file_count"], "bytes": before["total_bytes"]}))
        return 0

    if NEW.exists():
        after = manifest(NEW)
        if before["merkle_sha256"] != after["merkle_sha256"]:
            raise SystemExit("destination exists with a conflicting Merkle hash")
    else:
        temporary = NEW.with_name(f".{NEW.name}.{os.getpid()}.tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(OLD, temporary, symlinks=True)
        after = manifest(temporary)
        if before["merkle_sha256"] != after["merkle_sha256"]:
            shutil.rmtree(temporary)
            raise SystemExit("copy verification failed")
        os.replace(temporary, NEW)
        after = manifest(NEW)

    old_text, new_text = str(OLD), str(NEW)
    published = None
    for _ in range(30):
        receipt = load_json(RECEIPT_PATH)
        authority = replace_prefix(load_json(AUTHORITY_PATH), old_text, new_text)
        state = replace_prefix(load_json(TASK_STATE_PATH), old_text, new_text)
        try:
            published = publish_bundle(
                authority,
                state,
                event_type="DEPTH_PACKAGE_PATH_MIGRATED_V6",
                expected_revision=int(receipt["governance_revision"]),
                generator_path=Path(__file__),
            )
            break
        except RuntimeError as error:
            if "CAS revision mismatch" not in str(error):
                raise
    if published is None:
        raise SystemExit("governance CAS remained busy; copied package retained, old source not deleted")
    serialized = json.dumps(load_json(AUTHORITY_PATH), ensure_ascii=False)
    if old_text in serialized or new_text not in serialized:
        raise SystemExit("current authority path replacement did not close")

    redirects = {
        "schema_version": "chaoyang-path-redirects-v1",
        "generated_at": now(),
        "redirects": [{
            "old_prefix": old_text,
            "new_prefix": new_text,
            "reason": "V6 root-directory convergence; content is byte-identical.",
            "content_merkle_sha256": before["merkle_sha256"],
        }],
    }
    write_json(REDIRECTS, redirects)
    if process_refs(OLD):
        raise SystemExit("source acquired an active FD/CWD after authority migration; old source retained")
    shutil.rmtree(OLD)
    result = {
        "schema_version": "chaoyang-depth-package-migration-v6-result-v1",
        "completed_at": now(),
        "status": "PASS_MOVED_AND_OLD_DELETED",
        "source_absent": not OLD.exists(),
        "source": before,
        "destination": after,
        "path_redirects": {
            "path": str(REDIRECTS),
            "bytes": REDIRECTS.stat().st_size,
            "sha256": digest(REDIRECTS),
        },
        "published_governance_revision": published["governance_revision"],
    }
    write_json(RUN / "receipts/DEPTH_PACKAGE_MIGRATION_RESULT.json", result)
    print(json.dumps({"status": result["status"], "files": after["file_count"], "bytes": after["total_bytes"], "revision": published["governance_revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
