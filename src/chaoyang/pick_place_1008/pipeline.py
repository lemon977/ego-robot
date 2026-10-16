"""Finite raw-frame pipeline; sealed stages may be reused only by exact bindings."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib,html,json,os,time,traceback
from pathlib import Path
import numpy as np
from . import source
from chaoyang.exact78_baseline.stage_reuse import dependencies, ref, verify, current_producer

STAGE_FILES={
 "source":("source/SESSION_INPUT.json","source/source.npz","source/physical_eye.mp4"),
 "motion":("actions.npz","motion.json","frame_mapping.jsonl"),
 "gpu":("raw_rgb.npz","clean_rgb.npz","masks.npz","masks.json"),
 "render":("final.mp4","review.mp4","frame_states.jsonl","contact_events.json","render_validation.npz","SESSION.json"),
}
STAGE_CODE={
 "source":("pick_place_1008/source.py","pick_place_1008/camera.py","pipeline/pico_manus_motion_v2.py"),
 "motion":("pick_place_1008/pipeline.py","exact78_baseline/retarget.py","exact78_baseline/robot_model.py"),
 "gpu":("pick_place_1008/visual.py","pick_place_1008/pipeline.py","pick_place_1008/mask_seeds.py","pick_place_1008/sam_point_adapter.py"),
 "render":("pick_place_1008/render_stage.py","pick_place_1008/visual.py","pick_place_1008/source.py"),
}
def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".writing")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,
        default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v),allow_nan=False)+"\n")
    os.replace(temp,path)
def read(path):return json.loads(Path(path).read_text())
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()
@lru_cache(maxsize=12)
def stage_dependencies(repo_string,stage):
    repo=Path(repo_string)
    # Conservative candidate closure: stage reuse also pins imported helpers,
    # even when a helper is used only transitively or by a runtime branch.
    rows=[ref(path) for path in sorted((repo/"src/chaoyang/pick_place_1008").rglob("*.py")) if "__pycache__" not in path.parts]
    rows += [ref(repo/"src/chaoyang"/name) for name in STAGE_CODE[stage]]
    parent=read(repo/"baseline_v1/MANIFEST.json")
    for item in parent["sources"]:
        if item["role"]!="RUNTIME_PLATFORM_DEPENDENCY_ONLY":
            verify(item["source"]);rows.append(item["source"])
    if stage in ("motion","gpu","render"):rows+=dependencies(repo,stage)
    return sorted({v["path"]:v for v in rows}.values(),key=lambda v:v["path"])
def binding(repo,stage,row,config_ref,upstream,mask_seed_ref=None):
    value = {"stage":stage,"raw_source":row["source_refs"],"session_id":row["session_id"],
            "input_admission":{"status":row["input_status"],"time_boundary_resolution":row.get("time_boundary_resolution")},
            "raw_frame_count":row["raw_frame_count"],"camera_config":config_ref,
            "code_and_dependencies":stage_dependencies(str(repo),stage),"upstream":upstream}
    if stage=="gpu": value["visible_mask_seed"] = mask_seed_ref
    return value
def seal(repo,folder,stage,before):
    for item in before["code_and_dependencies"]:verify(item)
    receipt={"schema":"pick_place_1008_sealed_stage_v1","status":"COMPLETE",
             "binding":before,"binding_sha256":canonical(before),"producer":current_producer(repo),
             "files":{name:ref(folder/name) for name in STAGE_FILES[stage]}}
    path=folder/("STAGE_"+stage+".json")
    if path.exists():raise RuntimeError("IMMUTABLE_STAGE_ALREADY_EXISTS")
    write(path,receipt)
    return ref(path)
def find_reuse(repo,roots,sid,stage,expected):
    from .stage_reuse import find_reuse as sealed_lookup
    return sealed_lookup(repo,roots,sid,stage,expected,STAGE_FILES[stage])
def init_worker():
    import cv2
    cv2.setNumThreads(1)
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[name]="2"
def prepare_source_worker(source_dir_string,source_folder_string,config,raw_frame_count):
    source.prepare_session(Path(source_dir_string),Path(source_folder_string),config,raw_frame_count)
    return True
def motion_worker(repo_string,folder_string,source_folder_string):
    from chaoyang.exact78_baseline.retarget import retarget_session
    from chaoyang.exact78_baseline.robot_model import Robot
    repo,folder=Path(repo_string),Path(folder_string)
    session=source.load_prepared(Path(source_folder_string))
    tick=time.monotonic()
    robot=Robot(repo)
    result=retarget_session(session,robot)
    arm_names=[list(chain.joint_names) for chain in robot.arms]
    hand_names=[list(hand.names) for hand in robot.hands]
    dt=np.diff(session.qpc_ns).astype(float)/1e9
    velocity=np.diff(result.q_arm,axis=0)/dt[:,None,None]
    acceleration=np.diff(velocity,axis=0)/((dt[:-1]+dt[1:])/2)[:,None,None]
    speed_bad=np.zeros((session.frame_count,2),bool)
    accel_bad=np.zeros((session.frame_count,2),bool)
    speed_bad[1:]=np.any(abs(velocity)>3.6,axis=2)
    accel_bad[2:]=np.any(abs(acceleration)>54,axis=2)
    interval_observed=np.zeros_like(speed_bad)
    interval_observed[1:]=(session.hand_observed[:,:-1]&session.hand_observed[:,1:]).T
    np.savez_compressed(folder/"actions.npz",q_arm=result.q_arm,q_hand=result.q_hand,
        T_world_base=result.T_world_base,c2w=session.c2w,qpc_ns=session.qpc_ns,
        root_world=result.root_world,arm_position_mm=result.arm_position_mm,
        arm_rotation_deg=result.arm_rotation_deg,hand_bone_deg=result.hand_bone_deg,
        hand_source=session.hand_source,hand_observed=session.hand_observed,
        complete_valid=session.motion_valid["complete_valid"],
        arm_velocity_rad_s=velocity,arm_acceleration_rad_s2=acceleration,
        qpc_velocity_violation=speed_bad,qpc_acceleration_violation=accel_bad,
        qpc_interval_observed=interval_observed,
        q_arm_joint_names=np.asarray(arm_names,dtype="U64"),
        q_hand_joint_names=np.asarray(hand_names,dtype="U64"))
    value={"schema":"pick_place_1008_raw_motion_v1","status":"COMPLETE","session_id":session.session_id,
        "raw_frame_count":session.frame_count,"calibration":session.calibration,"metrics":result.metrics,
        "base":result.base_report,"actual_motion_source":"RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS",
        "action_contract":{"angle_unit":"radian","translation_unit":"metre","side_order":["left","right"],
            "q_arm_joint_names":arm_names,"q_hand_joint_names":hand_names,
            "T_world_base_direction":"p_world = T_world_base @ p_base",
            "c2w_direction":"p_world = c2w @ p_camera","auxiliary_arm_error_shape":"2,N"},
        "qpc_diagnostics":{"max_arm_velocity_rad_s":float(abs(velocity).max()),
            "max_arm_acceleration_rad_s2":float(abs(acceleration).max()),
            "limits_rad_s":3.6,"limits_rad_s2":54,
            "velocity_violation_raw_frame_indices":np.flatnonzero(speed_bad.any(axis=1)).tolist(),
            "acceleration_violation_raw_frame_indices":np.flatnonzero(accel_bad.any(axis=1)).tolist(),
            "velocity_violation_count_per_side":speed_bad.sum(axis=0).tolist(),
            "acceleration_violation_count_per_side":accel_bad.sum(axis=0).tolist(),
            "qualification":"QPC_FIXED_PARENT_NUMERIC_LIMITS_REVIEW_REQUIRED" if speed_bad.any() or accel_bad.any() else "QPC_LIMITS_PASSED",
            "invalid_intervals_remain_distinct":True},
        "elapsed_seconds":time.monotonic()-tick,"training_eligible":False,"control_ground_truth":False}
    write(folder/"motion.json",value)
    with (folder/"frame_mapping.jsonl").open("x") as stream:
        for frame in range(session.frame_count):
            row={"session_id":session.session_id,"raw_frame_index":frame,"raw_frame_id":frame,
                 "timestamp_semantics":"SOFTWARE_HOST_ARRIVAL_QPC_NOT_EXPOSURE_TRUTH"}
            for key,values in session.mapping.items():
                v=values[frame];row[key]=v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else v
            stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    return value
def load_arrays(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}
def pack_masks(masks):
    return {name:np.packbits(value.reshape(len(value),-1),axis=1) for name,value in masks.items()
            if isinstance(value,np.ndarray) and value.ndim==3 and value.dtype==bool}
def load_masks(folder):
    arrays=load_arrays(folder/"masks.npz")
    masks={name:np.unpackbits(value,axis=1,count=640*480).reshape(len(value),480,640).astype(bool)
           for name,value in arrays.items()}
    masks["metadata"]=read(folder/"masks.json")
    return masks
def render_worker(repo_string,folder_string,source_string,motion_string,gpu_string):
    from .render_stage import render_session
    repo,folder=Path(repo_string),Path(folder_string)
    session=source.load_prepared(Path(source_string))
    motion=load_arrays(Path(motion_string)/"actions.npz")
    masks=load_masks(Path(gpu_string));masks["raw_rgb"]=load_arrays(Path(gpu_string)/"raw_rgb.npz")["rgb"]
    clean=load_arrays(Path(gpu_string)/"clean_rgb.npz")["rgb"]
    masks["motion_info"]=read(Path(motion_string)/"motion.json")
    masks["upstream_refs"]={name:ref(path) for name,path in {
        "actions":Path(motion_string)/"actions.npz","frame_mapping":Path(motion_string)/"frame_mapping.jsonl",
        "motion_info":Path(motion_string)/"motion.json","raw_rgb":Path(gpu_string)/"raw_rgb.npz",
        "clean_rgb":Path(gpu_string)/"clean_rgb.npz","masks":Path(gpu_string)/"masks.npz",
        "mask_info":Path(gpu_string)/"masks.json"}.items()}
    return render_session(repo,folder,session,motion,clean,masks)
def gpu_stage(models,folder,session,visible_seeds=None,raw_rgb=None):
    from .visual import build_visual_masks,ProtectedProPainter
    raw=source.load_rgb_frames(session,640,480) if raw_rgb is None else raw_rgb
    masks=build_visual_masks(models["sam"],raw,session,visible_seeds=visible_seeds) if visible_seeds is not None else build_visual_masks(models["sam"],raw,session)
    failures=[problem for problem in masks["metadata"].get("quality_problems",[]) if problem.get("severity")=="QUALITY_FAILURE"]
    if visible_seeds is not None and masks["metadata"].get("point_tracking",{}).get("status")!="COMPLETE":
        failures.append({"code":"POINT_TRACKING_NOT_COMPLETE","severity":"QUALITY_FAILURE"})
    if failures:
        np.savez_compressed(folder/"raw_rgb.npz",rgb=raw)
        np.savez_compressed(folder/"masks.npz",**pack_masks(masks))
        write(folder/"MASK_QUALITY_FAILURE.json",{"status":"MASK_QUALITY_NOT_ADMITTED","quality_failures":failures,
            "metadata":masks["metadata"],"inpainting_performed":False,"training_eligible":False,"control_ground_truth":False})
        raise RuntimeError("MASK_QUALITY_NOT_ADMITTED")
    wrapped=ProtectedProPainter(models["propainter"])
    clean=wrapped.inpaint(raw,masks["removal"],masks["protected_visible"])
    validation=wrapped.last_validation
    if np.any(clean[masks["protected_visible"]]!=raw[masks["protected_visible"]]):
        raise RuntimeError("PROTECTED_VISIBLE_RGB_CHANGED")
    effective=wrapped.effective_masks
    if np.any(clean[~effective]!=raw[~effective]):raise RuntimeError("CLEAN_CHANGED_OUTSIDE_EFFECTIVE")
    masks["effective_removal"]=effective
    np.savez_compressed(folder/"raw_rgb.npz",rgb=raw)
    np.savez_compressed(folder/"clean_rgb.npz",rgb=clean)
    np.savez_compressed(folder/"masks.npz",**pack_masks(masks))
    info={"schema":"pick_place_1008_protected_clean_v1","status":"COMPLETE",
          "metadata":masks["metadata"],"clean_validation":validation,
          "protected_pixel_channel_difference_count":0,"outside_effective_difference_count":0,
          "source_pixel_domain":"DIRECT_ORIGINAL_VST_DECODE_CROP_RESIZE_RGB_UINT8",
          "mask_role_frames":{name:int(value.any(axis=(1,2)).sum()) for name,value in masks.items()
              if isinstance(value,np.ndarray) and value.ndim==3 and value.dtype==bool},
          "settings":models["propainter"].settings,"training_eligible":False,"control_ground_truth":False}
    write(folder/"masks.json",info);return info

def run(repo,output,manifest_path,config_path,session_ids,stage="full",reuse=(),mask_seed_index=None,render_workers=2,source_workers=1):
    repo,output=Path(repo),Path(output);started=time.monotonic()
    if render_workers not in (2,4):raise ValueError("BOUNDED_RENDER_WORKER_COUNT")
    if source_workers not in (1,2,4):raise ValueError("BOUNDED_SOURCE_WORKER_COUNT")
    if source_workers>1:
        producer=current_producer(repo);verify(producer["task_packet"]);packet=read(producer["task_packet"]["path"])
        minimum_threads=2*source_workers+(2 if stage=="full" else 0)
        if packet["budgets"].get("workers_max",0)<source_workers or packet["budgets"].get("cpu_threads_total",0)<minimum_threads:
            raise ValueError("SOURCE_WORKERS_EXCEED_FENCED_CPU_BUDGET")
    if render_workers==4 and stage=="full":
        producer=current_producer(repo);packet=read(producer["task_packet"]["path"])
        verify(producer["task_packet"])
        if packet["budgets"].get("workers_max",0)<4 or packet["budgets"].get("cpu_threads_total",0)<10:
            raise ValueError("FOUR_RENDER_WORKERS_REQUIRE_FENCED_CPU_BUDGET_10")

    output.mkdir(parents=True,exist_ok=False)
    manifest,config=read(manifest_path),read(config_path);config_ref=ref(config_path)
    seed_index_pin=ref(mask_seed_index) if mask_seed_index is not None else None
    available={row["session_id"]:row for row in manifest["sessions"]}
    if len(set(session_ids))!=len(session_ids) or not set(session_ids)<=set(available):raise ValueError("SESSION_SET_INVALID")
    status={sid:{"session_id":sid,"status":"PENDING","stages":{}} for sid in session_ids}
    locations={sid:{} for sid in session_ids};receipts={sid:{} for sid in session_ids}
    def publish():
        rows=[status[sid] for sid in session_ids]
        live=[{**value,"stage_receipts":dict(receipts[value["session_id"]])} for value in rows]
        live_index={"schema":"pick_place_1008_live_index_v1","status":"RUNNING",
            "sessions":live,"manifest":ref(manifest_path),"camera_config":config_ref,
            "expected_sessions":len(rows),"completed":sum(v["status"]=="COMPLETE" for v in rows),
            "elapsed_seconds":time.monotonic()-started}
        write(output/"PROGRESS.json",live_index)
        if not (output/"RESULT.json").is_file():write(output/"INDEX.json",live_index)
    def failed(sid,phase,error):
        status[sid].update(status="FAILED",failure_stage=phase,error=f"{type(error).__name__}:{error}")
        write(output/"sessions"/sid/("ERROR_"+phase+".json"),{"error":status[sid]["error"],"traceback":traceback.format_exc()})
        publish()
    for sid in session_ids:(output/"sessions"/sid).mkdir(parents=True)
    publish()
    # CPU source workers finish before motion/CUDA/render pools; budgets do not overlap.
    source_pool=None;source_futures={}
    def admit_source(sid,folder,found,receipt):
        locations[sid]["source"]=found/"source";receipts[sid]["source"]=receipt
        status[sid]["stages"]["source"]={"receipt":receipt,"folder":str(found),"reused":found!=folder}
    try:
        if source_workers>1:
            import multiprocessing
            source_pool=ProcessPoolExecutor(max_workers=source_workers,initializer=init_worker,mp_context=multiprocessing.get_context("spawn"))
        for sid in session_ids:
            folder=output/"sessions"/sid;row=available[sid]
            expected=binding(repo,"source",row,config_ref,{})
            try:
                if row["input_status"] not in ("READY_RAW_AUDIT","READY_RAW_WITH_INVALID_SHORT_GAP"):
                    raise RuntimeError("RAW_INPUT_NOT_ADMITTED")
                if row["input_status"]=="READY_RAW_WITH_INVALID_SHORT_GAP":
                    verify(row["time_boundary_resolution"])
                found,receipt=find_reuse(repo,reuse,sid,"source",expected)
                if found is None:
                    pins=row["source_refs"].values() if isinstance(row["source_refs"],dict) else row["source_refs"]
                    for pin in pins:verify(pin)
                    if source_pool is not None:
                        future=source_pool.submit(prepare_source_worker,str(Path(manifest["source_root"])/sid),str(folder/"source"),config,row["raw_frame_count"])
                        source_futures[future]=(sid,expected);status[sid]["status"]="SOURCE";publish();continue
                    prepare_source_worker(str(Path(manifest["source_root"])/sid),str(folder/"source"),config,row["raw_frame_count"])
                    found=folder;receipt=seal(repo,folder,"source",expected)
                admit_source(sid,folder,found,receipt)
            except Exception as error:failed(sid,"source",error)
            publish()
        for future in as_completed(source_futures):
            sid,expected=source_futures[future];folder=output/"sessions"/sid
            try:
                future.result();receipt=seal(repo,folder,"source",expected);admit_source(sid,folder,folder,receipt)
            except Exception as error:failed(sid,"source",error)
            publish()
    finally:
        if source_pool is not None:source_pool.shutdown(wait=True)
    if stage!="prepare":
        with ProcessPoolExecutor(max_workers=2,initializer=init_worker) as pool:
            futures={}
            for sid in session_ids:
                if status[sid]["status"]=="FAILED":continue
                folder=output/"sessions"/sid
                expected=binding(repo,"motion",available[sid],config_ref,{"source":receipts[sid]["source"]})
                try:
                    found,receipt=find_reuse(repo,reuse,sid,"motion",expected)
                    if found is not None:
                        locations[sid]["motion"]=found;receipts[sid]["motion"]=receipt
                        status[sid]["stages"]["motion"]={"receipt":receipt,"folder":str(found),"reused":True}
                    else:futures[pool.submit(motion_worker,str(repo),str(folder),str(locations[sid]["source"]))]=(sid,expected)
                except Exception as error:failed(sid,"motion_reuse",error)
                publish()
            for future in as_completed(futures):
                sid,expected=futures[future];folder=output/"sessions"/sid
                try:
                    future.result();receipt=seal(repo,folder,"motion",expected)
                    locations[sid]["motion"]=folder;receipts[sid]["motion"]=receipt
                    status[sid]["stages"]["motion"]={"receipt":receipt,"folder":str(folder),"reused":False}
                except Exception as error:failed(sid,"motion",error)
                publish()
    models=None
    if stage=="full":
        # Spawn CPU render children before any parent CUDA imports; no forked CUDA state.
        import multiprocessing
        pool=ProcessPoolExecutor(max_workers=render_workers,initializer=init_worker,mp_context=multiprocessing.get_context("spawn"))
        futures={}
        def drain(wait=False):
            for future in list(futures):
                if not wait and not future.done():continue
                sid,expected=futures.pop(future);folder=output/"sessions"/sid
                try:
                    future.result();receipt=seal(repo,folder,"render",expected)
                    locations[sid]["render"]=folder;receipts[sid]["render"]=receipt
                    status[sid]["stages"]["render"]={"receipt":receipt,"folder":str(folder)}
                except Exception as error:failed(sid,"render",error)
                publish()
        try:
            for sid in session_ids:
                if status[sid]["status"]=="FAILED":continue
                folder=output/"sessions"/sid
                try:
                    seed_ref=None
                    if mask_seed_index is not None:
                        verify(seed_index_pin)
                        from .mask_seeds import seed_reference
                        seed_ref=seed_reference(mask_seed_index,sid)
                    expected=binding(repo,"gpu",available[sid],config_ref,{"source":receipts[sid]["source"]},mask_seed_ref=seed_ref)
                    found,receipt=find_reuse(repo,reuse,sid,"gpu",expected)
                    if found is None:
                        if models is None:
                            from chaoyang.exact78_baseline.mask import Sam3Video
                            from chaoyang.exact78_baseline.clean import ProPainter
                            models={"sam":Sam3Video(repo),"propainter":ProPainter(repo)}
                        status[sid]["status"]="GPU";publish()
                        session=source.load_prepared(locations[sid]["source"])
                        visible_seeds=None;seed_rgb=None
                        if mask_seed_index is not None:
                            from .mask_seeds import load_visible_seed
                            seed_rgb=source.load_rgb_frames(session,640,480)
                            visible_seeds=load_visible_seed(mask_seed_index,sid,session,seed_rgb)
                            verify(seed_index_pin)
                            if visible_seeds["seed_reference"]!=seed_ref:
                                raise RuntimeError("VISIBLE_MASK_SEED_CHANGED_AFTER_BINDING")
                        gpu_stage(models,folder,session,visible_seeds=visible_seeds,raw_rgb=seed_rgb)
                        del seed_rgb
                        found=folder;receipt=seal(repo,folder,"gpu",expected)
                    locations[sid]["gpu"]=found;receipts[sid]["gpu"]=receipt
                    status[sid]["stages"]["gpu"]={"receipt":receipt,"folder":str(found),"reused":found!=folder}
                    render_binding=binding(repo,"render",available[sid],config_ref,dict(receipts[sid]))
                    reused,render_receipt=find_reuse(repo,reuse,sid,"render",render_binding)
                    if reused is not None:
                        locations[sid]["render"]=reused;receipts[sid]["render"]=render_receipt
                    else:
                        future=pool.submit(render_worker,str(repo),str(folder),str(locations[sid]["source"]),
                            str(locations[sid]["motion"]),str(found))
                        futures[future]=(sid,render_binding)
                    drain()
                except Exception as error:
                    failed(sid,"gpu",error)
                    if models:
                        import torch
                        torch.cuda.empty_cache()
            drain(True)
        finally:
            pool.shutdown(wait=True)
            if models:models["sam"].close()
    rows=[]
    for sid in session_ids:
        value=status[sid];loc=locations[sid]
        value["stage_receipts"]=dict(receipts[sid])
        if value["status"]!="FAILED":
            value["status"]="COMPLETE";value["raw_frame_count"]=available[sid]["raw_frame_count"]
            if stage=="full":
                rf=loc["render"];mf=loc["motion"];sr=read(rf/"SESSION.json")
                value["files"]={role:ref(path) for role,path in {
                    "final":rf/"final.mp4","review":rf/"review.mp4","actions":mf/"actions.npz",
                    "frame_mapping":mf/"frame_mapping.jsonl","frame_states":rf/"frame_states.jsonl",
                    "session_result":rf/"SESSION.json"}.items()}
                value.update(visual_status=sr.get("visual_status","PENDING_AI_REVIEW"),
                    severe_issues_open=sr.get("severe_issues_open",[]),accepted_defects=sr.get("accepted_defects",[]),
                    human_review_status="PENDING")
            value["stage_receipts"]=receipts[sid]
        rows.append(value)
    result={"schema":"pick_place_1008_pipeline_v1","stage":stage,
        "status":"COMPLETE" if all(v["status"]=="COMPLETE" for v in rows) else "INCOMPLETE",
        "sessions":rows,"sessions_requested":len(rows),
        "sessions_complete":sum(v["status"]=="COMPLETE" for v in rows),
        "frames_complete":sum(v.get("raw_frame_count",0) for v in rows),
        "manifest":ref(manifest_path),"camera_config":config_ref,"elapsed_seconds":time.monotonic()-started,
        "resource_recipe":{"render_workers":render_workers,"source_workers":source_workers,"worker_threads":2,"cpu_pools_overlap":False},"mask_seed_index":seed_index_pin,
        "training_eligible":False,"control_ground_truth":False,"human_review_status":"PENDING"}
    if seed_index_pin is not None:verify(seed_index_pin)
    write(output/"INDEX.json",result);write(output/"RESULT.json",result);publish()
    links="".join("<tr><td>"+html.escape(r["session_id"])+"</td><td>"+r["status"]+"</td><td>"+
        ("<a href='"+html.escape(os.path.relpath(r["files"]["review"]["path"],output))+"'>six panels</a> "+
        "<a href='"+html.escape(os.path.relpath(r["files"]["final"]["path"],output))+"'>final</a>" if "files" in r else "")+"</td></tr>" for r in rows)
    (output/"index.html").write_text("<!doctype html><meta charset='utf-8'><title>pick_place_1008</title><h1>Raw frame delivery</h1><p>Visual candidates; human review pending; training and control authority false.</p><table>"+links+"</table>")
    return result
