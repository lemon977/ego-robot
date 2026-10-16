import copy
import json
from pathlib import Path
import numpy as np
import pytest
from chaoyang.pick_place_1008 import audit, source
from chaoyang.pick_place_1008 import installation_fixed_k as fixed

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")

def producer(tmp_path,monkeypatch):
    repo=tmp_path/"repo";repo.mkdir();monkeypatch.setattr(audit,"PROJECT_ROOT",repo)
    task=fixed.K_TASK;c=repo/"_run/current"/task/"attempts/attempt_0001/candidate"
    c.mkdir(parents=True)
    inputs={}
    for key in ["input_manifest","base_camera_config","annotations","camera_fit_report","installation_fit_report"]:
        p=repo/"inputs"/(key+".json");write(p,{"authority":key});inputs[key]=source.ref(p)
    config=c/"FITTED_CAMERA_CONFIG.json";write(config,{"devices":{fixed.B_SERIAL:{"fit_quality_status":"PASSED"}}})
    report=c/"INTRINSIC_REPORT.json"
    value={"schema":"pick_place_device_encoded_intrinsic_verification_v1","status":"PASSED",**inputs,
        "K_fitted":True,"installation_prior_fitted":False,"camera_extrinsics_fitted":False,
        **{k:False for k in ["time_fitted","qpc_modified","scale_fitted","metric_scale_modified","lens_remap","holdout_used"]}}
    write(report,value)
    packet=repo/"tasks/current"/task/"TASK_PACKET.json"
    write(packet,{"task_id":task,"run_signature":"signed-job","execution_mode":"EXACT78_CANDIDATE",
        "required_outputs":["candidate/RESULT.json","candidate/INTRINSIC_REPORT.json","candidate/FITTED_CAMERA_CONFIG.json"],
        "inputs":{"files":list(inputs.values())}})
    result=c/"RESULT.json";outer=c.parent/"RESULT.json"
    def publish():
        write(result,{"status":"PASSED","intrinsic_report":source.ref(report),"fitted_camera_config":source.ref(config)})
        write(outer,{"status":"PASSED","task_id":task,"run_signature":"signed-job","validation_errors":[],
            "task_packet":source.ref(packet),"stage_result":source.ref(result),"required_outputs":{
                "candidate/RESULT.json":source.ref(result),"candidate/INTRINSIC_REPORT.json":source.ref(report),
                "candidate/FITTED_CAMERA_CONFIG.json":source.ref(config)}})
        refresh()
    def refresh():
        write(repo/"docs/governance/LONG_HORIZON_TASK_STATE.json",{"tasks":[{"task_id":task,"status":"PASSED",
            "pid":None,"proc_start_ticks":None,"run_signature":"signed-job","result":source.ref(outer)}]})
        write(repo/"tasks/receipts"/(task.upper()+".json"),{"status":"PASSED","result":source.ref(outer)})
    publish()
    return {"repo":repo,"task":task,"candidate":c,"report":report,"config":config,
            "packet":packet,"result":result,"outer":outer,"publish":publish,"refresh":refresh}

def sealed(p):
    return fixed._sealed_product(source.ref(p["report"]),"INTRINSIC_REPORT.json",
        "pick_place_device_encoded_intrinsic_verification_v1","intrinsic_report")

def test_genuine_sealed_encoded_producer_and_input_closure(tmp_path,monkeypatch):
    p=producer(tmp_path,monkeypatch);report,result,proof=sealed(p)
    got,exact=fixed._k_producer({"encoded_k_producer":proof},source.ref(p["config"]))
    assert got==report and exact==proof and result["status"]=="PASSED"

@pytest.mark.parametrize("change",["alive","duplicate","receipt","packetpath","signature","outputset","reportstatus","resultlink","packetinput"])
def test_rebound_illegal_producer_chain_is_rejected(tmp_path,monkeypatch,change):
    p=producer(tmp_path,monkeypatch)
    if change in ["alive","duplicate"]:
        q=p["repo"]/"docs/governance/LONG_HORIZON_TASK_STATE.json";v=json.loads(q.read_text())
        if change=="alive":v["tasks"][0]["pid"]=91
        else:v["tasks"].append(copy.deepcopy(v["tasks"][0]))
        write(q,v)
    elif change=="receipt":
        write(p["repo"]/"tasks/receipts"/(p["task"].upper()+".json"),{"status":"PASSED","result":source.ref(p["result"])})
    elif change=="reportstatus":
        v=json.loads(p["report"].read_text());v["status"]="REVIEW_REQUIRED";write(p["report"],v);p["publish"]()
    elif change=="packetinput":
        v=json.loads(p["packet"].read_text());v["inputs"]["files"]=[];write(p["packet"],v);p["publish"]()
    else:
        v=json.loads(p["outer"].read_text())
        if change=="packetpath":
            fake=p["repo"]/"unowned_packet.json";fake.write_bytes(p["packet"].read_bytes());v["task_packet"]=source.ref(fake)
        elif change=="signature":v["run_signature"]="different-job"
        elif change=="outputset":v["required_outputs"].pop("candidate/FITTED_CAMERA_CONFIG.json")
        elif change=="resultlink":v["stage_result"]=source.ref(p["report"])
        write(p["outer"],v);p["refresh"]()
    with pytest.raises((fixed.FixedKInstallationError,ValueError)):
        if change=="packetinput":
            _,_,proof=sealed(p);fixed._k_producer({"encoded_k_producer":proof},source.ref(p["config"]))
        else:sealed(p)

@pytest.mark.parametrize("domain",["time_fitted","scale_fitted","qpc_modified","camera_extrinsics_fitted","installation_prior_fitted","lens_remap","holdout_used"])
def test_sealed_encoded_producer_cannot_change_fixed_domains(tmp_path,monkeypatch,domain):
    p=producer(tmp_path,monkeypatch);v=json.loads(p["report"].read_text());v[domain]=True;write(p["report"],v);p["publish"]();_,_,proof=sealed(p)
    with pytest.raises(fixed.FixedKInstallationError,match="FIXED_DOMAINS"):
        fixed._k_producer({"encoded_k_producer":proof},source.ref(p["config"]))

def test_different_config_ref_cannot_borrow_passed_k(tmp_path,monkeypatch):
    p=producer(tmp_path,monkeypatch);_,_,proof=sealed(p);other=p["repo"]/"other.json";other.write_bytes(p["config"].read_bytes())
    with pytest.raises(fixed.FixedKInstallationError,match="PIN_MISMATCH"):
        fixed._k_producer({"encoded_k_producer":proof},source.ref(other))

def original_prior():
    return {"schema":"controller_to_wrist_v1","convention":"wrist_pose = compose_pose(controller_pose, transform)","controller_pose_frame":"pico_world_rh_x_forward_y_left_z_up","left":{"pos":[.02,.03,.04],"quat":[0,0,0,1]},"right":{"pos":[.05,.04,.03],"quat":[0,0,0,1]}}

def test_failed_configuration_keeps_every_original_field():
    base={"status":"FROZEN_VISUAL_ESTIMATE","fit_quality_status":"PASSED","devices":{
        "DEVICE_A":{"K_override":[[2,0,1],[0,2,1],[0,0,1]],"custom":"fixed"},
        fixed.B_SERIAL:{"K_override":[[3,0,1],[0,3,1],[0,0,1]],"fit_quality_status":"PASSED","custom":"keep"}}}
    old=copy.deepcopy(base)
    result=fixed._config_after(base,{"status":"REVIEW_REQUIRED"},original_prior(),{"unused":"report"})
    assert result==old and base==old and result is not base

def test_successful_configuration_changes_only_b_right_installation():
    base={"fit_quality_status":"PASSED","devices":{"A":{"key":[1,2,3]},fixed.B_SERIAL:{"fit_quality_status":"PASSED","K_override":[[500,0,301],[0,560,260],[0,0,1]]}}}
    prior=original_prior();original=copy.deepcopy(prior);m=source.installation_matrices(prior)[1].copy();m[0,3]+=.01
    result=fixed._config_after(base,{"status":"PASSED","T_controller_wrist_fitted":m.tolist()},prior,{"path":"report","bytes":1,"sha256":"a"*64})
    assert result["devices"]["A"]==base["devices"]["A"]
    for k,v in base["devices"][fixed.B_SERIAL].items():assert result["devices"][fixed.B_SERIAL][k]==v
    assert result["devices"][fixed.B_SERIAL]["installation_prior_override"]["left"]==prior["left"]
    assert prior==original and "installation_prior_override" not in base["devices"][fixed.B_SERIAL]

def test_image_manifest_duplicate_native_row_and_session_conflict_rejected():
    item={"session_id":"008","raw_frame_index":0}
    inspect={"schema":"pick_place_readonly_raw_image_inspection_v1","frames":[item,item],"gpu_used":False,"renderer_used":False,"fitting_performed":False,"training_eligible":False}
    with pytest.raises(fixed.FixedKInstallationError,match="DUPLICATE"):fixed._image_rows(inspect)
    original={"schema":"pick_place_calibration_source_images_v1","status":"PASSED","no_overlays":True,"lens_remap":False,"qpc_modified":False,"holdout_used":False,"sessions":[{"session_id":"008","images":[dict(item,session_id="100")]}]}
    with pytest.raises(fixed.FixedKInstallationError,match="SESSION_CONFLICT"):fixed._image_rows(original)

def test_new_consumer_requires_sealed_producer_legacy_path_preserved(tmp_path,monkeypatch):
    repo=tmp_path;monkeypatch.setattr(audit,"PROJECT_ROOT",repo)
    prior=original_prior();group=fixed.B_SERIAL+":"+fixed.PRIOR_SHA
    p=repo/"legacy.json";write(p,{"schema":"pick_place_installation_group_visual_fit_v1","groups":{group:{"status":"PASSED","device_serial":fixed.B_SERIAL,"original_prior_sha256":fixed.PRIOR_SHA,"prior_used":prior}}})
    selected={"installation_original_prior_sha256":fixed.PRIOR_SHA,"installation_group_key":group,"fit_quality_status":"PASSED","installation_correction_evidence":source.ref(p),"installation_prior_override":prior}
    assert source.validated_installation_override(selected,{"prior_sha256":fixed.PRIOR_SHA},fixed.B_SERIAL)==prior
    n=repo/"_run/current/exact78_pick_place_fake/attempts/attempt_0001/candidate"/fixed.REPORT_NAME
    write(n,{"schema":fixed.REPORT_SCHEMA,"status":"PASSED","groups":{group:{"status":"PASSED","prior_used":prior}}})
    selected["installation_correction_evidence"]=source.ref(n)
    with pytest.raises((ValueError,FileNotFoundError)):
        source.validated_installation_override(selected,{"prior_sha256":fixed.PRIOR_SHA},fixed.B_SERIAL)


def native_tip_fixture(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import cv2
    monkeypatch.setattr(audit,"PROJECT_ROOT",tmp_path)
    yy,xx=np.indices((480,640));a=np.stack([(xx%180).astype(np.uint8),(yy%180).astype(np.uint8),np.full((480,640),40,np.uint8)],axis=2)
    rgb=np.stack([a,np.clip(a.astype(int)+12,0,255).astype(np.uint8),np.clip(a.astype(int)+24,0,255).astype(np.uint8)])
    image=tmp_path/"original_native_1.png";assert cv2.imwrite(str(image),cv2.cvtColor(rgb[1],cv2.COLOR_RGB2BGR))
    video=tmp_path/"original.vst.h264";video.write_bytes(b"original encoded image source")
    qpcfile=tmp_path/"vst.qpc.ts.jsonl";qpcfile.write_text("100\n200\n300\n")
    rawtypes=["Invalid","MCP","PIP","DIP","TIP","MCP","PIP","IP","DIP","TIP","MCP","PIP","IP","DIP","TIP","MCP","PIP","IP","DIP","TIP","MCP","PIP","IP","DIP","TIP"]
    parents=[0,0,1,2,3,0,5,6,7,8,0,10,11,12,13,0,15,16,17,18,0,20,21,22,23]
    chains=["Hand"]+["Thumb"]*4+["Index"]*5+["Middle"]*5+["Ring"]*5+["Pinky"]*5
    sess=SimpleNamespace(session_id="pick_place_1008_100",qpc_ns=np.array([100,200,300],np.int64),calibration={"manus_joint_mapping":{
        "authority":"VERIFIED_SDK_RAW25_TO_CANONICAL_MANO21","raw_joint_types":rawtypes,"raw_parent_ids":parents,"raw_chain_types":chains}})
    head=np.tile(np.eye(4),(3,1,1));ctrl=np.tile(np.eye(4),(3,2,1,1));ctrl[:,:,2,3]=1.
    local=np.zeros((3,2,25,3));local[:,:,9]=[.009,.02,.03]
    native={"head":head,"controller":ctrl,"manus_local_25":local,"head_valid":np.ones(3,bool),"controller_valid":np.ones((3,2),bool),"hand_observed":np.ones((2,3),bool),"joint_observed_local":np.ones((3,2,21),bool)}
    anchored=tmp_path/"src/chaoyang/pick_place_1008/source.py"
    anchored.parent.mkdir(parents=True,exist_ok=True);anchored.write_bytes(Path(source.__file__).read_bytes())
    monkeypatch.setattr(source,"__file__",str(anchored))
    folder=tmp_path/"prepared/100/source";folder.mkdir(parents=True)
    prepared_video=folder/"native.mp4";prepared_video.write_bytes(b"prepared reference; RGB evidence validated separately")
    arrays=folder/"source.npz"
    np.savez(arrays,K=np.array([[1042.4497247810964,0,603.279959158785],[0,1135.6193426120842,521.9762600698414],[0,0,1]]),
        c2w=head,c2w_editor=head,hands=np.zeros((2,3,21,3)),hand_observed=np.ones((2,3),bool),hand_source=np.full((2,3),"RAW"),
        body=np.zeros((3,24,3)),body_observed=np.zeros(3,bool),gravity=np.tile([0,0,-1.],(3,1)),qpc_ns=sess.qpc_ns,
        **{"native__"+k:v for k,v in native.items()})
    sess.calibration.update(hand_joint_schema=source.JOINT_MAPPING_SCHEMA,manus25_to21=source.MAP_25_TO_21.tolist())
    write(folder/"SESSION_INPUT.json",{"session_id":sess.session_id,"status":"COMPLETE_RAW_INPUT","schema_version":source.RAW_INPUT_SCHEMA,
        "width":1280,"height":960,"fps":30.,"frame_count":3,"video":source.ref(prepared_video),"arrays":source.ref(arrays),"calibration":sess.calibration})
    loaded=source.load_prepared(folder)
    assert loaded.session_id=="pick_place_1008_100" and loaded.width==1280
    context={"session":loaded,"native":loaded.native_motion,"rgb":rgb,"row":{"session_id":"100","source_refs":{"video":source.ref(video),"video_qpc":source.ref(qpcfile)}}}
    obs={"session_id":"100","device_serial":fixed.B_SERIAL,"raw_frame_index":1,"qpc_ns":200,"image":source.ref(image),"side":"right","joint_index_21":8,"manus_node_id":9,"point_type":"GLOVE_TIP","semantics":"VISUAL_PROXY","visibility":"VISIBLE","side_confirmed":True,"joint_mapping_confirmed":True,"uv":[120.,260.],"annotation_uncertainty_px":12.}
    row={"session_id":"100","device_serial":fixed.B_SERIAL,"raw_frame_index":1,"qpc_ns":200,"image":source.ref(image),"source_vst":source.ref(video),"source_qpc":source.ref(qpcfile),"pixels_overlaid":False,"lens_remap":False}
    return obs,row,context

def test_original_native_png_qpc_sdk_leaf_and_raw_composition(tmp_path,monkeypatch):
    o,er,ctx=native_tip_fixture(tmp_path,monkeypatch)
    v=fixed._verified_native_tip(o,er,ctx,np.eye(4),True)
    assert v["image"]==o["image"] and v["sigma"]==12.
    np.testing.assert_array_equal(v["local"],[.009,.02,.03])
    np.testing.assert_array_equal(v["matrix"],ctx["native"]["controller"][1,1])

@pytest.mark.parametrize("change",["rebound_pixels","different_native_png","qpc_observation","qpc_row","serial","sdk_node","nonleaf","sdk_kind","sdk_chain","sdk_authority","held_head","held_controller","held_hand","held_joint","occluded","overlay","lens","wrong_raw_ref"])
def test_actual_native_evidence_rejection_even_when_image_refs_are_rebound(tmp_path,monkeypatch,change):
    import cv2
    o,er,ctx=native_tip_fixture(tmp_path,monkeypatch)
    if change=="rebound_pixels":
        image=Path(o["image"]["path"]);v=cv2.imread(str(image));v[20,30,0]^=1;assert cv2.imwrite(str(image),v)
        o["image"]=er["image"]=source.ref(image)  # valid new SHA, wrong original pixels
    elif change=="different_native_png":
        image=tmp_path/"native_2.png";assert cv2.imwrite(str(image),cv2.cvtColor(ctx["rgb"][2],cv2.COLOR_RGB2BGR));o["image"]=er["image"]=source.ref(image)
    elif change=="qpc_observation":o["qpc_ns"]+=1
    elif change=="qpc_row":er["qpc_ns"]+=1
    elif change=="serial":o["device_serial"]="ANOTHER_PICO"
    elif change=="sdk_node":o["manus_node_id"]=14
    elif change=="nonleaf":ctx["session"].calibration["manus_joint_mapping"]["raw_parent_ids"][10]=9
    elif change=="sdk_kind":ctx["session"].calibration["manus_joint_mapping"]["raw_joint_types"][9]="DIP"
    elif change=="sdk_chain":ctx["session"].calibration["manus_joint_mapping"]["raw_chain_types"][9]="Middle"
    elif change=="sdk_authority":ctx["session"].calibration["manus_joint_mapping"]["authority"]="UNQUALIFIED_VERSION"
    elif change=="held_head":ctx["native"]["head_valid"][1]=False
    elif change=="held_controller":ctx["native"]["controller_valid"][1,1]=False
    elif change=="held_hand":ctx["native"]["hand_observed"][1,1]=False
    elif change=="held_joint":ctx["native"]["joint_observed_local"][1,1,8]=False
    elif change=="occluded":o["visibility"]="OCCLUDED"
    elif change=="overlay":er["pixels_overlaid"]=True
    elif change=="lens":er["lens_remap"]=True
    elif change=="wrong_raw_ref":
        p=tmp_path/"other_original.h264";p.write_bytes(b"another device input");er["source_vst"]=source.ref(p)
    with pytest.raises(fixed.FixedKInstallationError):fixed._verified_native_tip(o,er,ctx,np.eye(4),True)
