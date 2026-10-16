"""Regenerate the frozen 0916 Robot100 cohort with an inward-forearm reference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json
from chaoyang.ops import run_robot100_batch_baseline_v1 as predecessor
from chaoyang.pipeline.full_robot_review_inward_v3 import solve_full_chain
from chaoyang.pipeline.robot_renderer_eevee_fullchain import load_pinned_robot_assets

SELECTION = REPO_ROOT / "_run/current/robot100_batch_baseline_20260928/attempts/attempt_0001/SELECTION.json"
OUTPUT = REPO_ROOT / "_run/current/robot100_batch_baseline_20260928/attempts/attempt_0001/corrections/inward_forearm_reference_v2/successor_sessions"
REFERENCE_Q = np.asarray([
    [1.4681, -0.2737, -0.8942, -1.9858, -0.2013, 0.3534, 0.5685],
    [-1.4681, -0.2737, 0.8942, -1.9858, 0.2013, 0.3533, -0.5685],
], dtype=np.float64)


def run(row: dict, assets, mounts, hand_eval: int, arm_eval: int) -> dict:
    folder = OUTPUT / row["session_id"]
    result_path = folder / "RESULT.json"
    if result_path.is_file():
        result = load_json(result_path)
        if result.get("session_id") != row["session_id"]: raise RuntimeError("RESULT_IDENTITY_DRIFT")
        return result
    if folder.exists(): raise RuntimeError(f"PARTIAL_OUTPUT:{folder}")
    folder.mkdir(parents=True); started = time.monotonic()
    arrays, sensor = predecessor._load_motion(row, hand_eval)
    solved = solve_full_chain(arrays, assets, mounts, max_nfev=arm_eval, continuity_weight=.5, reference_q_arm=REFERENCE_Q)
    solved["source_video_frame_idx"] = arrays["source_video_frame_idx"].copy()
    motion = folder / "ROBOT_MOTION_INWARD_V2.npz"; np.savez_compressed(motion, **solved)
    video = folder / f"{row['session_id']}_上臂向外_小臂向内_Robot_V2.mp4"
    review = predecessor._render(row, solved, video)
    sides=[]
    for side in range(2):
        reference_root=float(abs(solved["reference_roots"][side,1,3]))
        wrist=float(abs(solved["T_actual_root"][0,side,1,3]))
        sides.append({"side":side,"reference_hand_root_lateral_m":reference_root,"actual_frame0_wrist_lateral_m":wrist})
    result={"schema_version":"CHAOYANG_ROBOT100_INWARD_SUCCESSOR_SESSION_V2","session_id":row["session_id"],
            "status":"COMPLETE_PENDING_HUMAN_REVIEW","predecessor_selection_row":row,"reference_q_arm":REFERENCE_Q.tolist(),
            "motion":artifact_ref(motion),"video":artifact_ref(video),"review":review,"sensor_metrics":sensor,
            "frame0_geometry":sides,"elapsed_seconds":time.monotonic()-started,"algorithm_quality_adopted":False,
            "claim_limit":"0916 sensor-driven inward-forearm virtual Robot successor; pending human review and no physical/control/training authority."}
    atomic_json(result_path,result); return result


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--worker-index",type=int,required=True)
    parser.add_argument("--worker-count",type=int,default=2); parser.add_argument("--max-hand-evaluations",type=int,default=24)
    parser.add_argument("--max-arm-evaluations",type=int,default=60); args=parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count <= 2: raise ValueError("WORKER_PARTITION")
    selection=load_json(SELECTION)
    if selection.get("count") != 100: raise RuntimeError("PREDECESSOR_SELECTION_DRIFT")
    OUTPUT.mkdir(parents=True,exist_ok=True); assets=load_pinned_robot_assets(REPO_ROOT); mounts,_=predecessor._mounts()
    rows=selection["rows"][args.worker_index::args.worker_count]; completed=[]; failed=[]
    for row in rows:
        try:
            result=run(row,assets,mounts,args.max_hand_evaluations,args.max_arm_evaluations)
            completed.append({"session_id":row["session_id"],"video":result["video"]})
            print(json.dumps({"session":row["session_id"],"status":"COMPLETE","count":len(completed)},ensure_ascii=False),flush=True)
        except Exception as exc:
            failed.append({"session_id":row["session_id"],"type":type(exc).__name__,"error":str(exc)})
            print(json.dumps({"session":row["session_id"],"status":"FAILED","error":str(exc)},ensure_ascii=False),flush=True)
    atomic_json(OUTPUT.parent/f"WORKER_{args.worker_index}.json",{"schema_version":"CHAOYANG_ROBOT100_INWARD_SUCCESSOR_WORKER_V2",
        "worker_index":args.worker_index,"worker_count":args.worker_count,"completed":completed,"failures":failed,"selection":artifact_ref(SELECTION)})
    return 0 if not failed else 3


if __name__ == "__main__": raise SystemExit(main())
