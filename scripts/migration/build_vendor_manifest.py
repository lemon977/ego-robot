#!/usr/bin/env python3
"""Bind each retained vendor runtime closure to a deterministic SHA-256 list."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb",buffering=1024*1024) as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()

def closure(root: Path) -> dict:
    encoded=[]; files=links=total=0
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts or path.is_dir():continue
        rel=path.relative_to(root).as_posix()
        if path.is_symlink():
            row={"path":rel,"type":"symlink","target":os.readlink(path)}; links+=1
        else:
            size=path.stat().st_size; row={"path":rel,"type":"file","bytes":size,"sha256":sha(path)};files+=1;total+=size
        encoded.append(json.dumps(row,sort_keys=True,separators=(",",":"),ensure_ascii=False))
    digest=hashlib.sha256(("\n".join(encoded)+"\n").encode()).hexdigest()
    return {"algorithm":"sha256-file-list-v1","sha256_file_list":digest,"regular_files":files,"symlinks":links,"regular_bytes":total}

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--manifest",type=Path,default=Path("manifests/vendor.json"));args=parser.parse_args()
    manifest=args.manifest.resolve();repo=manifest.parents[1];value=json.loads(manifest.read_text())
    for row in value["packages"]:
        runtime=repo/row["runtime_path"]
        observed=closure(runtime)
        current=row.get("runtime_closure",{})
        current.update(observed);row["runtime_closure"]=current
    manifest.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n")
    print(json.dumps({r["id"]:r["runtime_closure"]["sha256_file_list"] for r in value["packages"]},sort_keys=True))
    return 0
if __name__=="__main__":raise SystemExit(main())
