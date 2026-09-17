#!/usr/bin/env python3
"""Verify an archive inventory and perform a staged, non-destructive restore drill."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, tempfile
from pathlib import Path

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb",buffering=1024*1024) as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()

def merkle(leaves:list[bytes])->str:
    if not leaves:return hashlib.sha256(b"").hexdigest()
    level=leaves
    while len(level)>1:
        if len(level)%2:level.append(level[-1])
        level=[hashlib.sha256(level[i]+level[i+1]).digest() for i in range(0,len(level),2)]
    return level[0].hex()

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--archive",type=Path,required=True);args=parser.parse_args()
    root=args.archive.resolve(strict=True); summary=json.loads((root/'MERKLE_ROOT.json').read_text())
    rows=[]; leaves=[]; missing=[]; mismatched=[]
    for raw in (root/'INVENTORY.jsonl').read_text(encoding='utf-8').splitlines():
        leaves.append(hashlib.sha256(raw.encode()).digest()); row=json.loads(raw); rows.append(row)
        path=root/row['path']
        if not (path.exists() or path.is_symlink()):missing.append(row['path']);continue
        info=path.lstat()
        if info.st_size!=row['bytes']:mismatched.append(row['path'])
        if row['type']=='symlink' and os.readlink(path)!=row['target']:mismatched.append(row['path'])
    observed_root=merkle(leaves)
    if observed_root!=summary['merkle_root_sha256'] or len(rows)!=summary['entries'] or missing or mismatched or summary['unstable_entries']:
        raise RuntimeError({'merkle':observed_root,'missing':missing[:10],'mismatched':mismatched[:10]})
    by_path={r['path']:r for r in rows}
    samples=['git/repository.bundle','git/worktree.patch','MOVES.tsv']
    staging=root/'.staging';staging.mkdir(exist_ok=True)
    drill=Path(tempfile.mkdtemp(prefix='restore-drill-',dir=staging))
    restored=[]
    for relative in samples:
        row=by_path[relative]; source=root/relative; target=drill/'samples'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        observed=sha(target)
        if observed!=row['sha256']:raise RuntimeError(f'sample SHA mismatch: {relative}')
        restored.append({'path':relative,'bytes':row['bytes'],'sha256':observed})
    bundle=root/'git/repository.bundle'
    verify=subprocess.run(['git','bundle','verify',str(bundle)],cwd=root,text=True,capture_output=True,check=False)
    if verify.returncode:raise RuntimeError(verify.stderr)
    clone=drill/'bundle-clone'
    subprocess.run(['git','clone','--no-checkout',str(bundle),str(clone)],cwd=root,check=True,capture_output=True,text=True)
    subprocess.run(['git','checkout','--detach','pre-clean-20260917-0aa69e9'],cwd=clone,check=True,capture_output=True,text=True)
    patch_checks=[]
    for relative in ('git/worktree.patch','git/index.patch'):
        patch=root/relative
        if patch.stat().st_size==0:
            patch_checks.append({'path':relative,'status':'SKIPPED_EMPTY'});continue
        result=subprocess.run(['git','apply','--check',str(patch)],cwd=clone,text=True,capture_output=True,check=False)
        patch_checks.append({'path':relative,'status':'PASS' if result.returncode==0 else 'FAIL','stderr':result.stderr[-2000:]})
        if result.returncode:raise RuntimeError(patch_checks[-1])
    result={'schema_version':1,'status':'PASS','inventory_entries':len(rows),'merkle_root_sha256':observed_root,'all_paths_and_sizes_verified':True,'sample_restores':restored,'bundle_verify':'PASS','patch_checks':patch_checks,'staging_path':str(drill)}
    (root/'RESTORE_DRILL.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,ensure_ascii=False,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
