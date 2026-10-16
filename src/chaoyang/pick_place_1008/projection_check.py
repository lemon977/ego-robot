"""Fixed-config raw projection checks on development or untouched holdout frames."""
from pathlib import Path
import copy,hashlib
import numpy as np
from . import audit, source
from .camera import normalize_camera
from .calibration import _candidate_panel, _native_frames

SETS={"development":("001","003","008","100"),"holdout":("007","199")}

def run_projection_check(manifest_path, config_path, output_root, *, group="development"):
    import cv2
    cv2.setNumThreads(1)
    if group not in SETS: raise ValueError("FIXED_PROJECTION_REVIEW_GROUP_REQUIRED")
    manifest=source.read_json(Path(manifest_path)); config=source.read_json(Path(config_path))
    config_ref=source.ref(Path(config_path))
    stage=Path(config_path).parent
    outer=source.read_json(stage.parent/"RESULT.json")
    result=source.read_json(stage/"RESULT.json")
    if outer.get("status")!="PASSED" or outer.get("required_outputs",{}).get("candidate/FITTED_CAMERA_CONFIG.json")!=config_ref or result.get("fitted_camera_config")!=config_ref or result.get("status")!="PASSED":
        raise ValueError("PROJECTION_CONFIG_REQUIRES_SEALED_PASSED_FIT")
    if config.get("fit_quality_status")!="PASSED":
        raise ValueError("PROJECTION_CONFIG_REQUIRES_PASSED_VISUAL_FIT")
    rows={r["session_id"]:r for r in manifest["sessions"]}
    out=source._project_output(Path(output_root));out.mkdir(parents=True,exist_ok=False)
    report={"schema":"pick_place_fixed_projection_check_v1","status":"PASSED","input_manifest":source.ref(Path(manifest_path)),"camera_config":config_ref,"fixed_group":group,"fitting_performed":False,"holdout_used_for_fit":False,"review_status":"AI_REVIEW_PENDING","training_eligible":False,"control_ground_truth":False,"sessions":[]}
    for sid in SETS[group]:
        row=rows[sid]
        if row["input_status"]!="READY_RAW_AUDIT":raise ValueError("PROJECTION_RAW_NOT_READY")
        for ref in row["source_refs"].values(): audit._verify_exact_ref(ref)
        raw=Path(row["source_refs"]["source_manifest"]["path"]).parent
        serial=row["device_serial"];selected=config["devices"][serial]
        if selected.get("fit_quality_status")!="PASSED":raise ValueError("PROJECTION_DEVICE_FIT_NOT_PASSED")
        model=normalize_camera(source.read_json(raw/"raw/camera_params.json"),source.read_json(raw/"raw/camera_params.meta.json"),device_serial=serial,camera_sha256=row["camera_snapshot_sha256"],config=selected,output_size=(1280,960))
        prior, prior_ref=source.read_installation(raw)
        if "installation_prior_override" in selected:
            prior=source.validated_installation_override(selected,prior_ref,serial)
            prior_ref["override_evidence"]=selected["installation_correction_evidence"]
        if selected.get("correction_evidence"):audit._verify_exact_ref(selected["correction_evidence"],project_only=True)
        qpc=source.timeline([v for _,v in source.json_rows(raw/"raw/vst.qpc.ts.jsonl")],"video")
        if len(qpc)!=row["raw_frame_count"]:raise ValueError("PROJECTION_NATIVE_COUNT_DRIFT")
        native,mapping,quality=source.load_raw_motion(raw,qpc,expected_serial=serial,prior=prior)
        indices=sorted({0,len(qpc)-1,*[int(round((len(qpc)-1)*x)) for x in (.25,.5,.75)]})
        frames=_native_frames(raw/"raw/vst.h264",indices);folder=out/sid;folder.mkdir()
        records=[]
        for frame in indices:
            panel,stats=_candidate_panel(frames[frame],model,native,frame)
            panel[480:]=0
            cv2.putText(panel,f"{sid} frame {frame} | fixed visual config | no fitting",(8,500),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1)
            cv2.putText(panel,"Blue LEFT / red RIGHT | source raw poses | anatomical proxy",(8,522),cv2.FONT_HERSHEY_SIMPLEX,.4,(240,240,240),1)
            name=folder/f"raw_projection_{frame:04d}.png"
            if not cv2.imwrite(str(name),panel):raise IOError("PROJECTION_IMAGE_WRITE_FAILED")
            records.append({"raw_frame_index":frame,"qpc_ns":int(qpc[frame]),"image":source.ref(name),"projection":stats})
        report["sessions"].append({"session_id":sid,"device_serial":serial,"raw_frame_count":len(qpc),"camera":model.calibration,"installation_prior":prior_ref,"used_prior_sha256":hashlib.sha256(__import__("json").dumps(prior,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"frames":records,"time_quality":quality})
    audit._write_json(out/"PROJECTION_CHECK.json",report)
    result={"schema":"pick_place_projection_check_result_v1","status":"PASSED","projection_check":source.ref(out/"PROJECTION_CHECK.json"),"review_status":"AI_REVIEW_PENDING","fitting_performed":False,"holdout_used_for_fit":False,"training_eligible":False,"control_ground_truth":False}
    audit._write_json(out/"RESULT.json",result)
    return result
