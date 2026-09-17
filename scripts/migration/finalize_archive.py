#!/usr/bin/env python3
"""Generate redirect, inventory and Merkle proofs for an archive baseline."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time

GENERATED = {"INVENTORY.jsonl", "INVENTORY.jsonl.partial", "MERKLE_ROOT.json", "RESTORE_DRILL.json"}

def digest_file(path: Path) -> tuple[str, bool]:
    before = path.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 * 1024 * 1024) as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat(follow_symlinks=False)
    return digest.hexdigest(), (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)

def merkle_root(leaves: list[bytes]) -> str:
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = leaves
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
    return level[0].hex()

def write_redirects(root: Path) -> int:
    mapping = {}
    for line in (root / "MOVES.tsv").read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        source, target, reason = parts
        mapping[source] = {"original_path": source, "current_path": target, "reason": reason}
    value = {"schema_version": 1, "redirects": [mapping[key] for key in sorted(mapping)]}
    (root / "PATH_REDIRECTS.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return len(mapping)

def iter_entries(root: Path):
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directory = Path(current)
        rel_dir = directory.relative_to(root)
        if rel_dir == Path("."):
            directories[:] = sorted(name for name in directories if name != ".staging")
        else:
            directories[:] = sorted(directories)
        for name in sorted(files):
            if rel_dir == Path(".") and name in GENERATED:
                continue
            yield directory / name
        for name in sorted(directories):
            candidate = directory / name
            if candidate.is_symlink():
                yield candidate

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    root = args.archive.resolve(strict=True)
    redirects = write_redirects(root)
    partial = root / "INVENTORY.jsonl.partial"
    output = root / "INVENTORY.jsonl"
    leaves=[]; total=0; files=0; links=0; unstable=0; started=time.time()
    with partial.open("w", encoding="utf-8") as handle:
        for index, path in enumerate(iter_entries(root), 1):
            relative=path.relative_to(root).as_posix(); info=path.lstat()
            row={"path":relative,"mode":stat.S_IMODE(info.st_mode),"mtime_ns":info.st_mtime_ns}
            if path.is_symlink():
                row.update(type="symlink",target=os.readlink(path),bytes=info.st_size); links += 1
            elif path.is_file():
                sha,stable=digest_file(path); row.update(type="file",bytes=info.st_size,sha256=sha,stable=stable)
                total += info.st_size; files += 1; unstable += int(not stable)
            else:
                continue
            encoded=json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(",",":"))
            handle.write(encoded+"\n"); leaves.append(hashlib.sha256(encoded.encode()).digest())
            if index % 25000 == 0:
                print(json.dumps({"entries":index,"bytes":total}),flush=True)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(partial,output)
    summary={
        "schema_version":1,"archive_root":str(root),"entries":len(leaves),"files":files,
        "symlinks":links,"bytes":total,"unstable_entries":unstable,
        "merkle_root_sha256":merkle_root(leaves),"redirects":redirects,
        "excluded_generated":["INVENTORY.jsonl","MERKLE_ROOT.json","RESTORE_DRILL.json"],
        "elapsed_seconds":round(time.time()-started,3),
    }
    (root/'MERKLE_ROOT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,ensure_ascii=False,sort_keys=True))
    return 0 if unstable == 0 else 2

if __name__ == "__main__":
    raise SystemExit(main())
