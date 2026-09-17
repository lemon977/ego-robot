#!/usr/bin/env python3
"""Validate repository-local Markdown links in the active documentation tree."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from urllib.parse import unquote

LINK = re.compile(r"!?\[[^]\n]*\]\(([^)\n]+)\)")
ACTIVE_DOC_ROOTS = ("docs/current", "docs/guides", "docs/reference", "docs/research/current", "docs/governance")

def candidates(root: Path, document: Path, raw: str) -> list[Path]:
    target = unquote(raw.strip().strip("<>").split("#", 1)[0])
    if not target:
        return []
    path = Path(target)
    if path.is_absolute():
        fixed = "/mnt/workspace/code/" + "chaoyang/"
        if target.startswith(fixed):
            return [root / target[len(fixed):]]
        return []
    return [document.parent / path, root / path]

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--report",type=Path)
    args=parser.parse_args(); root=args.root.resolve(); checked=0; broken=[]
    docs=[root / "README.md", root / "docs/README.md"]
    for relative in ACTIVE_DOC_ROOTS:
        start=root/relative
        if start.exists(): docs.extend(start.rglob("*.md"))
    for document in sorted(set(docs)):
        fenced=False
        for number,line in enumerate(document.read_text(encoding="utf-8",errors="ignore").splitlines(),1):
            if line.lstrip().startswith("```"):
                fenced=not fenced; continue
            if fenced: continue
            for match in LINK.finditer(line):
                raw=match.group(1).strip()
                if raw.startswith(("http://","https://","mailto:","data:","#")):
                    continue
                paths=candidates(root,document,raw)
                if not paths: continue
                checked += 1
                if not any(path.exists() for path in paths):
                    broken.append({"document":document.relative_to(root).as_posix(),"line":number,"target":raw})
    value={"schema_version":1,"markdown_files":len(set(docs)),"local_links_checked":checked,"broken":broken,"status":"PASS" if not broken else "FAIL"}
    rendered=json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(rendered)
    print(rendered,end="")
    return 0 if not broken else 1
if __name__=="__main__": raise SystemExit(main())
