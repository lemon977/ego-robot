#!/usr/bin/env python3
"""Publish a fresh V5.2.1 recovery preflight for frozen Clean Wave-0."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

PROJECT = Path(__file__).resolve().parents[3]
LEASE = PROJECT / "_run/current/GPU_LEASE.json"
PROPAINTER = PROJECT / "vendor/ProPainter"
PROPAINTER_WEIGHTS = PROJECT / "assets/models/vendor/propainter"
SAMPLE = PROJECT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1"
EXECUTION_AUTHORITY_NAME = "EXECUTION_AUTHORITY_V52_1.json"


def now(): return datetime.now().astimezone().isoformat(timespec="seconds")
def sha(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()
def ref(path: Path):
    path=path.resolve(strict=True); return {"path":str(path),"bytes":path.stat().st_size,"sha256":sha(path)}
def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))
def atomic(path: Path, value):
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(value,f,ensure_ascii=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
def new(path: Path,value):
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
def tree_bytes(path: Path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())
def gpu():
    line=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip().splitlines()[0]
    v=[x.strip() for x in line.split(",")]
    return {"index":int(v[0]),"name":v[1],"total_mib":int(v[2]),"used_mib":int(v[3]),"free_mib":int(v[4]),"utilization_percent":int(v[5])}
def conflicts():
    tokens=("train_usb_tict_ddp.py","ProPainter/inference_propainter.py","run_clean_synthetic_propainter_baseline.py","FoundationStereo")
    rows=[]
    for p in Path("/proc").iterdir():
        if not p.name.isdigit(): continue
        try: cmd=(p/"cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace").strip()
        except OSError: continue
        if cmd and any(t in cmd for t in tokens) and int(p.name)!=os.getpid(): rows.append({"pid":int(p.name),"cmd":cmd})
    return sorted(rows,key=lambda x:x["pid"])


def validate_execution_authority(root: Path) -> dict:
    authority_path = root / EXECUTION_AUTHORITY_NAME
    authority = load(authority_path)
    if authority.get("status") != "FROZEN_RECOVERABLE_WAIT_GPU":
        raise RuntimeError("Clean execution authority is not frozen")
    for name, expected in authority.get("scripts", {}).items():
        if ref(PROJECT / "tools" / name) != expected:
            raise RuntimeError(f"Clean execution script changed after freeze: {name}")
    return ref(authority_path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--plan-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,help="fresh successor preflight path; default is PLAN/PREFLIGHT.json")
    a=ap.parse_args()
    root=a.plan_root.resolve(strict=True); selection_path=root/"EXACT78_WAVE0_SELECTION.json"
    selection=load(selection_path); interface=load(root/"MACHINE_INTERFACE.json")
    execution_authority = validate_execution_authority(root)
    actual_selection=ref(selection_path)
    if actual_selection != interface["selection"]: raise RuntimeError("selection differs from frozen machine interface")
    if selection["counts"]["sessions"]!=58 or selection["counts"]["existing_clean"]!=4 or selection["counts"]["pending_clean"]!=54: raise RuntimeError("Wave0 count contract changed")
    sample_session="play_cards_0903_203"; sample_frames=151
    sample_paths=[SAMPLE/"sessions"/sample_session,SAMPLE/"real_donor_v1"/sample_session,SAMPLE/"propainter_v1"/sample_session]
    sample_bytes=sum(tree_bytes(x) for x in sample_paths); per_frame=(sample_bytes+sample_frames-1)//sample_frames
    passed_new=[]; failed_terminal=[]; unfinished=[]
    for row in selection["sessions"]:
        if row.get("existing_clean") is not None:
            continue
        sid=row["session_id"]
        if (root/"propainter_v1"/sid/"RESULT.json").is_file():
            passed_new.append(sid)
        elif (root/"clean_terminals"/sid/"RESULT.json").is_file():
            failed_terminal.append(sid)
        else:
            unfinished.append(row)
    pending_frames=sum(int(row["frame_count"]) for row in unfinished); multiplier=2.0
    projected=int(per_frame*pending_frames*multiplier); reserve=100*1024**3
    usage=shutil.disk_usage(root); disk_safe=usage.free>=projected+reserve
    disk={"schema_version":"exact78-clean-wave-disk-budget-v52.1","created_at":now(),"status":"PASS" if disk_safe else "HOLD_INSUFFICIENT_DISK",
          "sample":{"session":sample_session,"frames":sample_frames,"paths":[str(x) for x in sample_paths],"actual_bytes":sample_bytes,"bytes_per_frame_ceiling":per_frame},
          "projection":{"pending_sessions":len(unfinished),"pending_frames":pending_frames,"safety_multiplier":multiplier,"projected_bytes":projected,"reserve_bytes":reserve,"required_free_bytes":projected+reserve},
          "filesystem":{"total_bytes":usage.total,"used_bytes":usage.used,"free_bytes":usage.free},"hard_gate":disk_safe}
    disk_path=(a.output.parent if a.output else root)/"DISK_BUDGET_V52_1.json"
    if disk_path.exists(): raise RuntimeError(f"no-clobber disk budget exists: {disk_path}")
    disk_path.parent.mkdir(parents=True,exist_ok=True);new(disk_path,disk)
    lease=load(LEASE); gs=gpu(); procs=conflicts()
    weights={n:ref(PROPAINTER_WEIGHTS/n) for n in ("raft-things.pth","recurrent_flow_completion.pth","ProPainter.pth")}
    commit=subprocess.check_output(["git","-C",str(PROPAINTER),"rev-parse","HEAD"],text=True).strip()
    vendor_ok=commit=="e870e79321c31b733e2031af5aa2fb1fe3ac7eec"
    gpu_quiet=gs["used_mib"]<=8192 and gs["utilization_percent"]<=10 and not procs
    safe=disk_safe and lease.get("status")=="RELEASED" and vendor_ok and gpu_quiet
    status="PASS_READY_TO_START_GUARDIAN" if safe else "HOLD_SHARED_GPU_BUSY_OR_RUNTIME_GATE"
    pre={"schema_version":"exact78-clean-wave-preflight-v52.1","created_at":now(),"status":status,"selection":actual_selection,
         "execution_authority":execution_authority,
         "disk_budget":ref(disk_path),"current_disk_observation":disk,"central_gpu_lease":lease,"gpu_observation":gs,"conflicting_processes":procs,
         "vendor":{"root":str(PROPAINTER),"commit":commit,"commit_expected":"e870e79321c31b733e2031af5aa2fb1fe3ac7eec","weights":weights},
         "hard_gates":{"selection_immutable":True,"disk_budget":disk_safe,"central_lease_released":lease.get("status")=="RELEASED","vendor_pinned":vendor_ok,"gpu_quiet_no_conflicting_process":gpu_quiet},
         "recovery_counts":{"existing":4,"passed_new":len(passed_new),"failed_terminal":len(failed_terminal),"pending":len(unfinished)},
         "launch_authorized":safe,"claim_limit":"Recovery preflight only; no GPU, Clean pixels, Robot or training authority."}
    preflight_path=(a.output or (root/"PREFLIGHT.json")).resolve()
    if preflight_path.exists(): raise RuntimeError(f"no-clobber preflight exists: {preflight_path}")
    preflight_path.parent.mkdir(parents=True,exist_ok=True); new(preflight_path,pre)
    state={"schema_version":"exact78-clean-wave-state-v52.1","updated_at":now(),"status":status,"pid":None,"current_session":None,
           "counts":{"selected":58,"existing":4,"pending":len(unfinished),"completed_new":len(passed_new),"failed":len(failed_terminal)},"selection":actual_selection,
           "preflight":ref(preflight_path),"next_action":"START_GUARDIAN" if safe else "WAIT_EXTERNAL_GPU_TRAINING_AND_RERUN_FRESH_SUCCESSOR_PREFLIGHT"}
    atomic(root/"STATE.json",state)
    print(json.dumps({"status":status,"disk":disk,"gpu":gs,"conflicting_processes":procs,"selection":actual_selection},ensure_ascii=False,indent=2))
    return 0 if safe else 2

if __name__=="__main__": raise SystemExit(main())
