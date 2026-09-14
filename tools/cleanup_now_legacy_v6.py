"""Create a bounded NOW evidence capsule and delete only after target-local gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.cleanup_current_only_v6 import build_reference_graph, protection_snapshot


ROOT = Path("/mnt/workspace/code/chaoyang")
SOURCE = ROOT / "NOW"
CAPSULE = ROOT / "docs/history/2026-09-14/now_v6_capsule"
RUN = ROOT / "tasks/control/runs/20260914_chaoyang_cleanup_v6"
TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".txt", ".sh", ".yaml", ".yml", ".toml", ".py"}
REPRESENTATIVE = Path("clean/eval_delivery/medoid_v7_extrapolated/robot/CHIPS002_FOURLANE_FINAL.mp4")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o444)
    os.replace(temporary, path)


def inventory() -> tuple[list[dict[str, Any]], str]:
    rows = []
    merkle = hashlib.sha256()
    for path in sorted(SOURCE.rglob("*")):
        relative = str(path.relative_to(SOURCE))
        if path.is_symlink():
            row = {"relative_path": relative, "kind": "symlink", "target": os.readlink(path)}
        elif path.is_file():
            row = {"relative_path": relative, "kind": "file", "bytes": path.stat().st_size, "sha256": sha256(path)}
        else:
            continue
        rows.append(row)
        merkle.update((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode())
    return rows, merkle.hexdigest()


def process_refs() -> list[dict[str, Any]]:
    prefix = str(SOURCE)
    rows = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        found = []
        for name in ("cwd", "exe"):
            try:
                value = os.readlink(proc / name)
            except OSError:
                continue
            if value == prefix or value.startswith(prefix + "/"):
                found.append({"source": name, "value": value})
        try:
            descriptors = (proc / "fd").iterdir()
            for descriptor in descriptors:
                try:
                    value = os.readlink(descriptor)
                except OSError:
                    continue
                if value == prefix or value.startswith(prefix + "/"):
                    found.append({"source": f"fd:{descriptor.name}", "value": value})
        except OSError:
            pass
        if found:
            rows.append({"pid": int(proc.name), "references": found})
    return rows


def prepare() -> dict[str, Any]:
    if not SOURCE.is_dir():
        return {"status": "ALREADY_ABSENT", "source": str(SOURCE)}
    rows, merkle = inventory()
    copied = []
    for row in rows:
        if row["kind"] != "file":
            continue
        relative = Path(row["relative_path"])
        should_copy = relative.suffix.lower() in TEXT_SUFFIXES or relative == REPRESENTATIVE
        if not should_copy:
            continue
        source = SOURCE / relative
        target = CAPSULE / "retained" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != row["sha256"]:
                raise RuntimeError(f"capsule target conflict: {target}")
        else:
            shutil.copy2(source, target)
        copied.append({"relative_path": str(relative), "bytes": target.stat().st_size, "sha256": sha256(target)})
    value = {
        "schema_version": "chaoyang-now-legacy-capsule-v1",
        "created_at": now(),
        "status": "PREPARED_NOT_DELETED",
        "source_root": str(SOURCE),
        "source_merkle_sha256": merkle,
        "source_entries": rows,
        "retained_files": copied,
        "retention_rule": "Keep current-independent text/config/source plus one representative Clean/Robot review; all other old media/logs are indexed by bytes/SHA or symlink target only.",
        "claim_limit": "Historical navigation capsule; not current authority and not a substitute for raw data.",
    }
    atomic_json(CAPSULE / "LEGACY_EVIDENCE_CAPSULE.json", value)
    atomic_json(RUN / "LEGACY_EVIDENCE_CAPSULE_NOW.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-delete", action="store_true")
    args = parser.parse_args()
    value = prepare()
    if value.get("status") == "ALREADY_ABSENT":
        print(json.dumps(value))
        return 0
    if not args.commit_delete:
        print(json.dumps({"status": value["status"], "entries": len(value["source_entries"]), "retained": len(value["retained_files"])}))
        return 0
    refs = process_refs()
    if refs:
        raise SystemExit(f"NOW has active FD/CWD: {json.dumps(refs[:10], ensure_ascii=False)}")
    _, protected = protection_snapshot()
    protected_now = []
    source_lexical = Path(os.path.normpath(str(SOURCE)))
    for path in protected:
        try:
            # Protection paths can point at CPFS/OSSFS artifacts.  Resolving
            # every one performs remote metadata I/O and previously made this
            # target-local gate appear to wait for Clean.  Lexical containment
            # is sufficient because protection_snapshot already normalizes
            # absolute paths and NOW itself is not a symlink.
            candidate = Path(os.path.normpath(str(path)))
            candidate.relative_to(source_lexical)
            # The current cleanup controller necessarily names its own target.
            # That exact root is not evidence authority; real subpath
            # references remain protected and the current-code graph below is
            # the independent check for application dependencies.
            if candidate == source_lexical:
                continue
            protected_now.append(str(path))
        except ValueError:
            continue
    if protected_now:
        raise SystemExit(f"NOW has current transitive evidence references: {protected_now[:20]}")
    graph = build_reference_graph(protected)
    code_refs = []
    target_cleanup_controllers = {
        Path(__file__).resolve(),
        (ROOT / "tools/cleanup_current_only_v6.py").resolve(),
    }
    for path in graph["current_code_transitive"]:
        candidate = Path(path).resolve()
        if candidate in target_cleanup_controllers:
            continue
        text = candidate.read_text(encoding="utf-8", errors="replace")
        if str(SOURCE) in text or "NOW/" in text:
            code_refs.append(str(candidate))
    if code_refs:
        raise SystemExit(f"NOW is referenced by current code: {code_refs}")
    latest_rows, latest_merkle = inventory()
    if latest_merkle != value["source_merkle_sha256"]:
        raise SystemExit("NOW changed during capsule preparation")
    pre_delete = {
        "schema_version": "chaoyang-cleanup-v6-pre-delete-v1",
        "created_at": now(), "status": "READY_COMMIT_DELETE",
        "target": str(SOURCE), "entry_count": len(latest_rows),
        "merkle_sha256": latest_merkle, "active_references": refs,
        "capsule": str(CAPSULE / "LEGACY_EVIDENCE_CAPSULE.json"),
    }
    atomic_json(RUN / "receipts/PRE_DELETE_NOW_MANIFEST.json", pre_delete)
    shutil.rmtree(SOURCE)
    result = {
        "schema_version": "chaoyang-cleanup-v6-deletion-receipt-v1",
        "completed_at": now(), "status": "PASS_PERMANENT_DELETE",
        "target": str(SOURCE), "target_absent": not SOURCE.exists(),
        "entry_count": len(latest_rows), "source_merkle_sha256": latest_merkle,
        "capsule": str(CAPSULE / "LEGACY_EVIDENCE_CAPSULE.json"),
    }
    atomic_json(RUN / "receipts/DELETION_NOW_RECEIPT.json", result)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
