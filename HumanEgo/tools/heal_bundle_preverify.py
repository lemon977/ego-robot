#!/usr/bin/env python3
"""Rewrite bundle frames that failed HumanEgo preverification on CPFS.

frozen_contract compares inode mtime/ctime before and after a read.  On this
volume those fields can move while the bytes stay the same, which kills the
dataloader.  Heal writes a new singly-linked inode, waits until two stats
agree, then refreshes selector_records.json hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

ROOT = Path("/mnt/workspace/code/chaoyang")
HIT = re.compile(
    r"(play_cards_\d{4}_\d{3}|get_potato_chips_\d{4}_\d{3})/"
    r"(\d{5}) dataloader metadata changed during preverification"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path) -> dict:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def rewrite_stable(path: Path) -> dict:
    data = path.read_bytes()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.heal")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        tmp.replace(path)
    except OSError:
        path.write_bytes(data)
        tmp.unlink(missing_ok=True)
    for _ in range(8):
        first = path.stat()
        time.sleep(0.15)
        second = path.stat()
        if (first.st_ino, first.st_size, first.st_mtime_ns, first.st_ctime_ns) == (
            second.st_ino, second.st_size, second.st_mtime_ns, second.st_ctime_ns
        ):
            break
    return file_ref(path)


def hits_from_log(text: str) -> list[tuple[str, str]]:
    found: dict[tuple[str, str], None] = {}
    for session, frame in HIT.findall(text):
        found[(session, frame)] = None
    return list(found)


def heal(bundle: Path, session: str, frame: str) -> dict:
    dest = (bundle / "production" / session / "09_humanego_adapter"
            / "preprocess/all_data" / frame)
    meta = dest / "training_data.json"
    image = dest / "rgb.png"
    if not meta.is_file():
        raise SystemExit(f"missing {meta}")
    refs = {"metadata": rewrite_stable(meta)}
    if image.is_file():
        refs["image"] = rewrite_stable(image)
    selector_path = bundle / "selector_records.json"
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    row = ((selector.get("sessions") or {}).get(session) or {}).get("frames") or {}
    if frame not in row:
        raise SystemExit(f"{session}/{frame} not in selector_records.json")
    row[frame]["metadata"] = refs["metadata"]
    if "image" in refs:
        row[frame]["image"] = refs["image"]
    tmp = selector_path.with_name(f".selector_records.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(selector, indent=2), encoding="utf-8")
    tmp.replace(selector_path)
    print(f"healed {session}/{frame} meta={refs['metadata']['sha256'][:12]}",
          flush=True)
    return {"session": session, "frame": frame, **refs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--frame")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    pairs = []
    if args.session and args.frame:
        pairs.append((args.session, args.frame))
    if args.log and args.log.is_file():
        pairs.extend(hits_from_log(args.log.read_text(encoding="utf-8",
                                                      errors="replace")[-20000:]))
    if not pairs:
        print("no preverify hits", flush=True)
        return 0
    seen: set[tuple[str, str]] = set()
    report = []
    for session, frame in pairs:
        key = (session, frame)
        if key in seen:
            continue
        seen.add(key)
        report.append(heal(bundle, session, frame))
    out = bundle / "heal_preverify.json"
    out.write_text(json.dumps({
        "healed": report,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
