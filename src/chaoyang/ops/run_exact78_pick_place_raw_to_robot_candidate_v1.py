"""New native Controller/Head+MANUS -> frozen parent robot visual candidate route."""
from pathlib import Path
import argparse,json,os,secrets,signal,sys
def main(argv=None):
    from chaoyang.governance.exact78_operation_gate import authorize_operation_entry
    authorize_operation_entry("run_exact78_pick_place_raw_to_robot_candidate_v1",argv)
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--camera-config",type=Path)
    p.add_argument("--stage",choices=["projection","prepare","motion","full","review-evidence","camera-evidence","camera-fit","installation-fit","fixed-k-installation-fit","intrinsic-verification","projection-check","mask-probe","mask-seed-evidence"],default="full")
    p.add_argument("--delivery-index",type=Path)
    p.add_argument("--annotations",type=Path)
    p.add_argument("--mask-seeds",type=Path)
    p.add_argument("--seed-anchor-frames",type=Path)
    p.add_argument("--source-index",type=Path)
    p.add_argument("--cpu-render-workers",type=int,choices=[2,4],default=2)
    p.add_argument("--cpu-source-workers",type=int,choices=[1,2,4],default=1)
    p.add_argument("--projection-group",choices=["development","holdout"],default="development")
    p.add_argument("--camera-fit-report",type=Path)
    p.add_argument("--installation-fit-report",type=Path)
    p.add_argument("--sessions",default="");p.add_argument("--reuse-from",action="append",type=Path,default=[])
    p.add_argument("--output-root",required=True,type=Path);a=p.parse_args(argv)
    from chaoyang.paths import get_paths
    repo=get_paths().repo_root.resolve();output=a.output_root.resolve()
    if not output.is_relative_to(repo) or output.exists():p.error("fresh project output required")
    cache=output.parent/"model_cache";cache.mkdir(exist_ok=True)
    for name,sub in (("HF_HOME","hf"),("TORCH_HOME","torch"),("XDG_CACHE_HOME","xdg"),("NUMBA_CACHE_DIR","numba"),("MPLCONFIGDIR","mpl")):
        os.environ[name]=str(cache/sub);(cache/sub).mkdir(exist_ok=True)
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):os.environ[name]="2"
    signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
    if a.stage=="full" and not a.mask_seeds:
        p.error("--mask-seeds required for reviewed visible-point full production")
    if a.stage=="mask-seed-evidence":
        if not a.camera_config:p.error("--camera-config required")
        from chaoyang.pick_place_1008.seed_evidence import run_seed_evidence
        sessions=a.sessions.split(",") if a.sessions else [r["session_id"] for r in json.loads(a.manifest.read_text())["sessions"]]
        result=run_seed_evidence(repo,output,a.manifest,a.camera_config,sessions,frame_indices=json.loads(a.seed_anchor_frames.read_text()) if a.seed_anchor_frames else None)
    elif a.stage=="camera-evidence":
        if not a.camera_config:p.error("--camera-config required")
        from chaoyang.pick_place_1008.calibration_fit import prepare_calibration_evidence
        result=prepare_calibration_evidence(a.manifest,output,a.camera_config)
    elif a.stage=="camera-fit":
        if not a.camera_config or not a.annotations:p.error("--camera-config and --annotations required")
        from chaoyang.pick_place_1008.calibration_fit import run_camera_fit
        result=run_camera_fit(a.manifest,a.annotations,a.camera_config,output)
    elif a.stage=="installation-fit":
        if not a.camera_config or not a.annotations or not a.camera_fit_report:p.error("--camera-config, --annotations and --camera-fit-report required")
        from chaoyang.pick_place_1008.installation_fit import run_installation_fit
        result=run_installation_fit(a.manifest,a.annotations,a.camera_config,a.camera_fit_report,output)
    elif a.stage=="fixed-k-installation-fit":
        if not a.camera_config or not a.annotations or not a.source_index:
            p.error("fixed camera config, input annotation bundle, and fresh sealed source index required")
        from chaoyang.pick_place_1008.installation_fixed_k import run_fixed_k_installation_fit
        result=run_fixed_k_installation_fit(a.manifest,a.annotations,a.camera_config,a.source_index,output)
    elif a.stage=="intrinsic-verification":
        if not a.camera_config or not a.annotations or not a.camera_fit_report or not a.installation_fit_report:p.error("fixed inputs and both preceding fit reports required")
        from chaoyang.pick_place_1008.intrinsic_fit import run_intrinsic_verification
        result=run_intrinsic_verification(a.manifest,a.annotations,a.camera_config,a.camera_fit_report,a.installation_fit_report,output)
    elif a.stage=="projection-check":
        if not a.camera_config:p.error("--camera-config required")
        from chaoyang.pick_place_1008.projection_check import run_projection_check
        result=run_projection_check(a.manifest,a.camera_config,output,group=a.projection_group)
    elif a.stage=="mask-probe":
        if not a.source_index or not a.mask_seeds:p.error("sealed canonical source index and visible mask seeds required")
        from chaoyang.pick_place_1008.mask_probe import run_mask_probe
        from chaoyang.pipeline.causal_modal_mask_gpu_adapter_v71 import FileGpuLease
        task=output.relative_to(repo/"_run/current").parts[0]
        lease=FileGpuLease(lease_path=repo/"_run/current/GPU_LEASE.json",lock_path=repo/"_run/current/GPU_LEASE.lock",
                task_id=task,attempt_id="attempt_0001",gpu_id=0,executor_epoch=1,fencing_token=secrets.token_hex(16),min_free_mib=40960)
        try:lease.__enter__()
        except Exception as exc:
            from chaoyang.pick_place_1008.pipeline import write
            output.mkdir(parents=True)
            write(output/"RESOURCE_WAIT.json",{"status":"WAIT_GPU_RESOURCE","algorithm_started":False,"reason":str(exc)})
            return 75
        try:result=run_mask_probe(repo,output,a.source_index,a.mask_seeds,a.sessions.split(",") if a.sessions else ["001","008"])
        finally:lease.__exit__(None,None,None)
    elif a.stage=="review-evidence":
        if not a.delivery_index:p.error("--delivery-index required")
        from chaoyang.pick_place_1008.review import build_review_evidence
        result=build_review_evidence(a.delivery_index,output)
    elif a.stage=="projection":
        from chaoyang.pick_place_1008.calibration import run_projection_audit
        result=run_projection_audit(Path(json.loads(a.manifest.read_text())["source_root"]),output,
            json.loads(a.manifest.read_text()),dev_sessions=a.sessions.split(",") if a.sessions else ["001","003","008","100"])
    else:
        if not a.camera_config:p.error("--camera-config required")
        from chaoyang.pick_place_1008.pipeline import run,write
        sessions=a.sessions.split(",") if a.sessions else [r["session_id"] for r in json.loads(a.manifest.read_text())["sessions"]]
        if a.stage=="full":
            from chaoyang.pipeline.causal_modal_mask_gpu_adapter_v71 import FileGpuLease
            task=output.relative_to(repo/"_run/current").parts[0]
            lease=FileGpuLease(lease_path=repo/"_run/current/GPU_LEASE.json",lock_path=repo/"_run/current/GPU_LEASE.lock",
                task_id=task,attempt_id="attempt_0001",gpu_id=0,executor_epoch=1,fencing_token=secrets.token_hex(16),min_free_mib=40960)
            try:lease.__enter__()
            except Exception as exc:
                output.mkdir(parents=True)
                write(output/"RESOURCE_WAIT.json",{"status":"WAIT_GPU_RESOURCE","algorithm_started":False,"reason":str(exc)})
                return 75
            try:result=run(repo,output,a.manifest,a.camera_config,sessions,a.stage,a.reuse_from,mask_seed_index=a.mask_seeds,render_workers=a.cpu_render_workers,source_workers=a.cpu_source_workers)
            finally:lease.__exit__(None,None,None)
        else:result=run(repo,output,a.manifest,a.camera_config,sessions,a.stage,a.reuse_from,mask_seed_index=a.mask_seeds,render_workers=a.cpu_render_workers,source_workers=a.cpu_source_workers)
    print(json.dumps({"status":result.get("status"),"sessions_complete":result.get("sessions_complete"),"frames_complete":result.get("frames_complete")}),flush=True)
    return 0 if result.get("status") in ("COMPLETE","PASSED","PROJECTION_AUDIT_COMPLETE") else 2
if __name__=="__main__":raise SystemExit(main())
