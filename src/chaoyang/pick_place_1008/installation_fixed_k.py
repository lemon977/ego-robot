"""Fixed accepted encoded K; one shared right-controller installation correction.

This distinct route consumes visible development TIP proxies. Factory camera,
raw metric scale, timestamps, device A and all left installation values stay fixed.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from . import audit, source, installation_fit, calibration_fit
from .camera import rigid

B_SERIAL = "PA9410MGK8090485G"
PRIOR_SHA = "3d71684e9d29fad06c2c638ee80164a130fcb338512b9d40131f509ee76280d3"
K_TASK = "exact78_pick_place_encoded_intrinsic_verification_v1_20261009"
BUNDLE_SCHEMA = "pick_place_fixed_k_installation_input_bundle_v1"
REPORT_SCHEMA = "pick_place_fixed_encoded_k_right_installation_visual_fit_v1"
REPORT_NAME = "FIXED_K_INSTALLATION_REPORT.json"
FIT_FRAMES = {"008": [0, 199], "100": [0, 40, 120, 138]}
VALIDATION_FRAMES = {"008": [50, 100], "100": [50, 125]}
TIP_IDS = {4: 4, 8: 9, 12: 14, 16: 19, 20: 24}

class FixedKInstallationError(ValueError):
    pass

def _fail(message):
    raise FixedKInstallationError(message)

def _read(path):
    return source.read_json(Path(path))

def _exact(reference, expected=None, project_only=True):
    if (not isinstance(reference, dict) or set(reference) != {"path", "bytes", "sha256"}
            or not isinstance(reference.get("path"), str)
            or type(reference.get("bytes")) is not int or reference["bytes"] < 0
            or not isinstance(reference.get("sha256"), str) or len(reference["sha256"]) != 64):
        _fail("FIXED_K_EXACT_REFERENCE_REQUIRED")
    if expected is not None and reference["path"] != str(Path(expected)):
        _fail("FIXED_K_REFERENCE_OWNERSHIP")
    p = Path(reference["path"])
    if p.is_symlink() or any(x.is_symlink() for x in p.parents):
        _fail("FIXED_K_REFERENCE_SYMLINK")
    audit._verify_exact_ref(reference, project_only=project_only)
    return reference

def _sealed_product(report_ref, report_name, schema, report_field):
    repo = audit.PROJECT_ROOT
    _exact(report_ref)
    path = Path(report_ref["path"])
    try:
        task = path.relative_to(repo / "_run/current").parts[0]
    except ValueError:
        _fail("FIXED_K_PRODUCER_REPORT_PATH")
    if not task.startswith("exact78_pick_place_") or Path(task).name != task:
        _fail("FIXED_K_PRODUCER_TASK_ID")
    candidate = repo / "_run/current" / task / "attempts/attempt_0001/candidate"
    _exact(report_ref, candidate / report_name)
    outer_path, result_path = candidate.parent / "RESULT.json", candidate / "RESULT.json"
    outer, result = _read(outer_path), _read(result_path)
    ledger = _read(repo / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    rows = [r for r in ledger.get("tasks", []) if r.get("task_id") == task]
    if (len(rows) != 1 or rows[0].get("status") != "PASSED"
            or rows[0].get("pid") is not None or rows[0].get("proc_start_ticks") is not None):
        _fail("FIXED_K_PRODUCER_NOT_SEALED")
    _exact(rows[0]["result"], outer_path)
    packet_path = repo / "tasks/current" / task / "TASK_PACKET.json"
    _exact(outer.get("task_packet"), packet_path)
    packet = _read(packet_path)
    signature = outer.get("run_signature")
    if (not signature or signature != rows[0].get("run_signature")
            or signature != packet.get("run_signature")
            or outer.get("task_id") != task or packet.get("task_id") != task
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"
            or outer.get("status") != "PASSED" or outer.get("validation_errors") != []):
        _fail("FIXED_K_PRODUCER_IDENTITY_OR_SIGNATURE")
    mandatory = {"candidate/RESULT.json", "candidate/" + report_name,
                 "candidate/FITTED_CAMERA_CONFIG.json"}
    declared = outer.get("required_outputs", {})
    if set(declared) != mandatory or not mandatory <= set(packet.get("required_outputs", [])):
        _fail("FIXED_K_PRODUCER_OUTPUT_SET")
    for name, pin in declared.items():
        _exact(pin, candidate.parent / name)
    result_ref = source.ref(result_path)
    if (outer.get("stage_result") != result_ref or result.get("status") != "PASSED"
            or result.get(report_field) != report_ref
            or result.get("fitted_camera_config") != declared["candidate/FITTED_CAMERA_CONFIG.json"]):
        _fail("FIXED_K_PRODUCER_RESULT_LINK")
    receipt_path = repo / "tasks/receipts" / (task.upper() + ".json")
    receipt = _read(receipt_path)
    if receipt.get("result") != source.ref(outer_path) or receipt.get("status") != "PASSED":
        _fail("FIXED_K_PRODUCER_RECEIPT_LINK")
    report = _read(path)
    if report.get("schema") != schema or report.get("status") != "PASSED":
        _fail("FIXED_K_REPORT_SCHEMA_OR_STATUS")
    return report, result, {"task_id": task, "report": report_ref,
        "stage_result": result_ref, "executor_result": source.ref(outer_path),
        "task_packet": source.ref(packet_path), "terminal_receipt": source.ref(receipt_path)}

def _packet_refs(proof):
    packet = _read(proof["task_packet"]["path"])
    refs = []
    def walk(value):
        if isinstance(value,dict):
            if set(value)=={"path","bytes","sha256"}: refs.append(value)
            else:
                for item in value.values(): walk(item)
        elif isinstance(value,list):
            for item in value: walk(item)
    walk(packet.get("inputs",{}))
    return refs

def _require_packet_inputs(proof, references):
    inputs = _packet_refs(proof)
    if any(pin not in inputs for pin in references):
        _fail("FIXED_K_PRODUCER_PACKET_INPUT_LINK")

def _k_producer(bundle, camera_ref):
    declared = bundle.get("encoded_k_producer")
    if not isinstance(declared, dict):
        _fail("FIXED_K_ACCEPTED_ENCODED_PRODUCER_REQUIRED")
    report, result, proof = _sealed_product(declared.get("report"), "INTRINSIC_REPORT.json",
        "pick_place_device_encoded_intrinsic_verification_v1", "intrinsic_report")
    if declared != proof or proof["task_id"] != K_TASK or result["fitted_camera_config"] != camera_ref:
        _fail("FIXED_K_ENCODED_PRODUCER_PIN_MISMATCH")
    if (report.get("K_fitted") is not True or report.get("installation_prior_fitted") is not False
            or report.get("camera_extrinsics_fitted") is not False
            or any(report.get(k) is not False for k in
                   ("time_fitted", "qpc_modified", "scale_fitted", "metric_scale_modified", "lens_remap", "holdout_used"))):
        _fail("FIXED_K_ENCODED_PRODUCER_FIXED_DOMAINS")
    _require_packet_inputs(proof,[report[k] for k in ("input_manifest","base_camera_config","annotations","camera_fit_report","installation_fit_report")])
    return report, proof

def _data(data, minimum):
    if data.get("side") != "right" or data.get("device_serial") != B_SERIAL:
        _fail("FIXED_K_RIGHT_DEVICE_B_ONLY")
    names = ("controller_to_camera", "local_points", "uv", "session_ids",
             "raw_frame_indices", "joint_indices_21", "annotation_uncertainty_px")
    if any(k not in data for k in names):
        _fail("FIXED_K_PURE_DATA_FIELDS")
    m, p, uv, sigma = (np.asarray(data[k], float) for k in
                       ("controller_to_camera", "local_points", "uv", "annotation_uncertainty_px"))
    n = len(p)
    if (n < minimum or p.shape != (n,3) or m.shape != (n,4,4) or uv.shape != (n,2)
            or sigma.shape != (n,) or not all(np.isfinite(x).all() for x in (m,p,uv,sigma))
            or (sigma < 8).any() or (np.linalg.norm(p, axis=1) > 1).any()
            or (uv < 0).any() or (uv[:,0] >= 640).any() or (uv[:,1] >= 480).any()):
        _fail("FIXED_K_PURE_DATA_DOMAIN")
    if (not np.allclose(m[:,3], [0,0,0,1], atol=1e-6)
            or not np.allclose(m[:,:3,:3] @ m[:,:3,:3].transpose(0,2,1), np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(m[:,:3,:3]), 1, atol=1e-5)):
        _fail("FIXED_K_CONTROLLER_CAMERA_NOT_RIGID")
    sid, frames, joints = (list(data[k]) for k in ("session_ids", "raw_frame_indices", "joint_indices_21"))
    if any(len(x) != n for x in (sid,frames,joints)) or set(sid) != {"008","100"}:
        _fail("FIXED_K_TWO_DEVELOPMENT_SESSIONS_ONLY")
    identities = []
    for s,f,j in zip(sid,frames,joints):
        if (isinstance(f,(bool,np.bool_)) or not isinstance(f,(int,np.integer)) or f < 0
                or isinstance(j,(bool,np.bool_)) or not isinstance(j,(int,np.integer)) or j not in TIP_IDS):
            _fail("FIXED_K_NATIVE_FRAME_OR_TIP_DOMAIN")
        identities.append((s,int(f),int(j)))
    if len(set(identities)) != n:
        _fail("FIXED_K_DUPLICATE_TIP")
    if any(len({f for s,f,j in identities if s == x}) < 2 for x in {"008","100"}):
        _fail("FIXED_K_MULTIPLE_NATIVE_FRAMES_PER_SESSION")
    return {"m":m,"p":p,"uv":uv,"sigma":sigma,"sid":sid,"identities":identities}

def _k(k):
    k = np.asarray(k,float)
    if (k.shape != (3,3) or not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0
            or not np.array_equal(k[2],[0,0,1]) or k[0,1] != 0 or k[1,0] != 0):
        _fail("FIXED_K_CANONICAL_K_REQUIRED")
    return k

def _evaluation(d,k,seed,fitted):
    before,bz,_ = installation_fit._project(d["m"],d["p"],k,seed)
    after,az,_ = installation_fit._project(d["m"],d["p"],k,fitted)
    ids = np.asarray(d["sid"])
    per = {s:{"before":calibration_fit._pixel_metrics(before[ids==s],d["uv"][ids==s]),
              "after":calibration_fit._pixel_metrics(after[ids==s],d["uv"][ids==s])} for s in sorted(set(ids))}
    metrics = calibration_fit._pixel_metrics(after,d["uv"])
    passed = bool(np.all(az > .01) and metrics["p50_px"] <= 20
                  and all(x["after"]["p50_px"] <= 20 and x["after"]["p95_px"] <= 60 for x in per.values()))
    return {"status":"PASSED" if passed else "REVIEW_REQUIRED",
            "before":calibration_fit._pixel_metrics(before,d["uv"]),"after":metrics,
            "per_session":per,"nonpositive_depth_count":int((az <= .01).sum())}

def fit_and_validate_right_installation(fit_data, validation_data, k_image, original_right):
    fit, val = _data(fit_data,8), _data(validation_data,4)
    k, seed = _k(k_image), rigid(original_right,"fixed_k_original_right")
    if {(s,f) for s,f,j in fit["identities"]} & {(s,f) for s,f,j in val["identities"]}:
        _fail("FIXED_K_FIT_VALIDATION_WHOLE_FRAME_OVERLAP")
    solved = installation_fit.fit_installation_proxy_points(fit["m"],fit["p"],fit["uv"],k,
        seed,fit["sid"],fit["sigma"])
    fitted = np.asarray(solved["T_controller_wrist_fitted"],float)
    validation = _evaluation(val,k,seed,fitted)
    passed = solved["status"] == "PASSED" and validation["status"] == "PASSED"
    return {"status":"PASSED" if passed else "REVIEW_REQUIRED", "fit":solved,
            "validation":validation,"T_controller_wrist_before":seed.tolist(),
            "T_controller_wrist_fitted":fitted.tolist(),
            "accepted_installation":(fitted if passed else seed).tolist(),
            "validation_used_by_optimizer":False,"validation_is_blind":False}

def _image_rows(document):
    if document.get("schema") == "pick_place_calibration_source_images_v1":
        if (document.get("status") != "PASSED" or document.get("no_overlays") is not True
                or any(document.get(k) is not False for k in ("lens_remap","qpc_modified","holdout_used"))):
            _fail("FIXED_K_ORIGINAL_SOURCE_IMAGE_MANIFEST")
        rows = []
        for session in document["sessions"]:
            for image in session["images"]:
                if image.get("session_id",session["session_id"]) != session["session_id"]:
                    _fail("FIXED_K_IMAGE_ROW_SESSION_CONFLICT")
                rows.append(dict(image,session_id=session["session_id"]))
    elif document.get("schema") == "pick_place_readonly_raw_image_inspection_v1":
        if any(document.get(k) is not False for k in ("gpu_used","renderer_used","fitting_performed","training_eligible")):
            _fail("FIXED_K_INSPECTION_SOURCE_IMAGE_MANIFEST")
        rows = document.get("frames",[])
    else:
        _fail("FIXED_K_SOURCE_IMAGE_SCHEMA")
    result = {}
    for x in rows:
        key = (x.get("session_id"),x.get("raw_frame_index"))
        if key in result:
            _fail("FIXED_K_DUPLICATE_SOURCE_IMAGE_ROW")
        result[key] = x
    return result

def _verified_source_K(session, accepted_device):
    size = accepted_device.get("output_size")
    if (type(session.width) is not int or type(session.height) is not int
            or session.width <= 0 or session.height <= 0
            or [session.width,session.height] != size
            or not np.array_equal(session.K,accepted_device.get("K_output_fitted"))):
        _fail("FIXED_K_SOURCE_OUTPUT_K_OR_SIZE_MISMATCH")
    sx, sy = 640. / session.width, 480. / session.height
    if not np.isclose(sx,sy,rtol=0,atol=1e-12):
        _fail("FIXED_K_SOURCE_NONUNIFORM_RESIZE")
    k_image = np.diag([sx,sy,1.]) @ _k(session.K)
    if not np.allclose(k_image,accepted_device.get("K_image_fitted"),rtol=0,atol=1e-12):
        _fail("FIXED_K_SOURCE_IMAGE_K_SCALE_MISMATCH")
    return k_image

def _verified_native_tip(observation, evidence_row, context, factory_camera, verify_pixels=True):
    o, er, ctx = observation, evidence_row, context
    sess, native = ctx["session"], ctx["native"]
    expected_sid = ctx["row"].get("session_id")
    if expected_sid not in {"008","100"} or sess.session_id != "pick_place_1008_" + expected_sid:
        _fail("FIXED_K_PREPARED_NUMERIC_SESSION_NAMESPACE")
    f, j = o.get("raw_frame_index"), o.get("joint_index_21")
    if (type(f) is not int or not 0 <= f < len(sess.qpc_ns)
            or type(j) is not int or j not in TIP_IDS
            or o.get("manus_node_id") != TIP_IDS[j]
            or o.get("session_id") != expected_sid or er.get("session_id") != expected_sid
            or er.get("raw_frame_index") != f or o.get("side") != "right"
            or o.get("point_type") != "GLOVE_TIP" or o.get("semantics") != "VISUAL_PROXY"
            or o.get("visibility") != "VISIBLE" or o.get("side_confirmed") is not True
            or o.get("joint_mapping_confirmed") is not True):
        _fail("FIXED_K_NATIVE_VISIBLE_TIP_IDENTITY")
    if (o.get("device_serial") != B_SERIAL or er.get("device_serial") != B_SERIAL
            or o.get("qpc_ns") != int(sess.qpc_ns[f]) or er.get("qpc_ns") != int(sess.qpc_ns[f])
            or er.get("image") != o.get("image")):
        _fail("FIXED_K_IMAGE_DEVICE_NATIVE_QPC")
    image_ref = _exact(o.get("image"))
    if verify_pixels:
        pic = cv2.imread(image_ref["path"])
        if pic is None or not np.array_equal(cv2.cvtColor(pic,cv2.COLOR_BGR2RGB),ctx["rgb"][f]):
            _fail("FIXED_K_ORIGINAL_NATIVE_RGB_MISMATCH")
    if er.get("pixels_overlaid") is True or er.get("lens_remap") is True:
        _fail("FIXED_K_NATIVE_IMAGE_OPERATION_CHANGED")
    for rawkey in ("source_vst","source_qpc"):
        if rawkey in er:
            _exact(er[rawkey],project_only=False)
            if er[rawkey] not in list(ctx["row"]["source_refs"].values()):
                _fail("FIXED_K_INSPECTION_RAW_REFERENCE_MISMATCH")
    if (not native["head_valid"][f] or not native["controller_valid"][f,1]
            or not native["hand_observed"][1,f] or not native["joint_observed_local"][f,1,j]):
        _fail("FIXED_K_RAW_OBSERVATION_REQUIRED")
    topology, node = sess.calibration["manus_joint_mapping"], TIP_IDS[j]
    chains = {4:"Thumb",8:"Index",12:"Middle",16:"Ring",20:"Pinky"}
    if (topology.get("authority") != "VERIFIED_SDK_RAW25_TO_CANONICAL_MANO21"
            or topology["raw_joint_types"][node] != "TIP" or node in topology["raw_parent_ids"]
            or topology["raw_chain_types"][node] != chains[j] or int(source.MAP_25_TO_21[j]) != node):
        _fail("FIXED_K_VERIFIED_TIP_LEAF_REQUIRED")
    uv = np.asarray(o.get("uv"),float); sigma = float(o.get("annotation_uncertainty_px",8.))
    if (uv.shape!=(2,) or not np.isfinite(uv).all() or not 0<=uv[0]<640 or not 0<=uv[1]<480
            or not np.isfinite(sigma) or sigma<8):
        _fail("FIXED_K_VISIBLE_PROXY_PIXEL_OR_UNCERTAINTY")
    head,ctrl = native["head"][f],native["controller"][f,1]
    matrix = np.linalg.inv(np.asarray(factory_camera)) @ np.linalg.inv(head) @ ctrl
    return {"image":image_ref,"matrix":matrix,"local":native["manus_local_25"][f,1,node],"sigma":sigma}

def _validate_bundle(manifest_path, bundle_path, camera_config_path, source_index_path, verify_pixels):
    from .mask_probe import _sealed_index, _sealed_source
    paths = {"input_manifest":manifest_path,"camera_config":camera_config_path,"source_index":source_index_path}
    bundle = _read(bundle_path)
    if (bundle.get("schema") != BUNDLE_SCHEMA or any(bundle.get(k) is not False for k in
            ("physical_ground_truth","training_eligible","holdout_used"))):
        _fail("FIXED_K_BUNDLE_SCHEMA_OR_QUALIFICATION")
    scope = {"device_serial":B_SERIAL,"side":"right","session_ids":["008","100"],"original_prior_sha256":PRIOR_SHA}
    if bundle.get("scope") != scope:
        _fail("FIXED_K_FROZEN_RIGHT_GROUP_SCOPE")
    pins = {k:source.ref(Path(v)) for k,v in paths.items()}
    pins["annotations_bundle"] = source.ref(Path(bundle_path))
    for k,p in pins.items():
        _exact(p)
        if k != "annotations_bundle" and bundle.get(k) != p:
            _fail("FIXED_K_BUNDLE_PATH_PIN:"+k)
    k_report,k_proof = _k_producer(bundle,pins["camera_config"])
    if k_report.get("input_manifest") != pins["input_manifest"]:
        _fail("FIXED_K_ENCODED_MANIFEST_MISMATCH")
    config = _read(camera_config_path)
    selected = config.get("devices",{}).get(B_SERIAL,{})
    if (selected.get("fit_quality_status") != "PASSED" or "installation_prior_override" in selected
            or "T_head_camera_override" in selected):
        _fail("FIXED_K_ORIGINAL_INSTALLATION_FACTORY_CAMERA_REQUIRED")
    kr = k_report["devices"][B_SERIAL]
    if not np.array_equal(selected.get("K_override"),kr["K_output_fitted"]):
        _fail("FIXED_K_ACCEPTED_K_VALUES_CHANGED")
    split_ref = _exact(bundle.get("split_plan")); split = _read(split_ref["path"])
    annotation_refs = bundle.get("annotations")
    if (not isinstance(annotation_refs,dict) or set(annotation_refs) != {"static","dynamic","extra"}
            or annotation_refs.get("static") != k_report["annotations"]
            or split.get("annotations") != annotation_refs or split.get("current_K") != pins["camera_config"]
            or split.get("fit_native_frames") != FIT_FRAMES
            or split.get("validation_native_frames") != VALIDATION_FRAMES
            or split.get("fit_points") != 22 or split.get("validation_points") != 11
            or split.get("holdout_used") is not False or split.get("whole_frame_split_no_point_overlap") is not True):
        _fail("FIXED_K_FROZEN_SPLIT_PLAN")
    current,index_ref = _sealed_index(audit.PROJECT_ROOT,source_index_path)
    index_rows = {x["session_id"]:x for x in current["sessions"]}
    if len(index_rows) != len(current["sessions"]):
        _fail("FIXED_K_SOURCE_INDEX_DUPLICATE_SESSION")
    manifest = _read(manifest_path); manifest_rows = {x["session_id"]:x for x in manifest["sessions"]}
    contexts = {}
    for sid in ["008","100"]:
        folder = _sealed_source(audit.PROJECT_ROOT,Path(source_index_path).parent,sid,index_rows[sid]["stage_receipts"]["source"])
        sess = source.load_prepared(folder)
        with np.load(folder/"source.npz",allow_pickle=False) as z:
            native = {k[8:]:z[k].copy() for k in z.files if k.startswith("native__")}
        row = manifest_rows[sid]
        prior,prior_ref = source.read_installation(Path(row["source_refs"]["source_manifest"]["path"]).parent)
        _verified_source_K(sess,kr)
        if (row["device_serial"] != B_SERIAL or sess.calibration["device_serial"] != B_SERIAL
                or prior_ref["prior_sha256"] != PRIOR_SHA
                or sess.calibration["installation"]["prior_sha256"] != PRIOR_SHA
                or sess.calibration["installation"]["used_prior_sha256"] != PRIOR_SHA
                or not np.array_equal(sess.calibration["T_head_camera_used"],kr["T_head_camera_factory_fixed"])
                or not np.array_equal(native["manus25_to21"],source.MAP_25_TO_21)
                or sess.calibration["manus_joint_mapping"].get("authority") != "VERIFIED_SDK_RAW25_TO_CANONICAL_MANO21"):
            _fail("FIXED_K_CANONICAL_SOURCE_FIXED_DOMAINS")
        rgb = source.load_rgb_frames(sess,640,480) if verify_pixels else None
        contexts[sid] = {"session":sess,"native":native,"prior":prior,"prior_reference":prior_ref,
                         "row":row,"rgb":rgb,"source_receipt":index_rows[sid]["stage_receipts"]["source"]}
    if contexts["008"]["prior"] != contexts["100"]["prior"]:
        _fail("FIXED_K_SHARED_PRIOR_VALUES_DIFFER")
    records,seen,image_pins = [],set(),[]
    for kind,annotation_ref in annotation_refs.items():
        _exact(annotation_ref);doc = _read(annotation_ref["path"])
        if (doc.get("input_manifest") != pins["input_manifest"]
                or doc.get("base_camera_config") != k_report["base_camera_config"]
                or any(doc.get(k) is True for k in ("holdout_used","physical_ground_truth","training_eligible"))):
            _fail("FIXED_K_ANNOTATION_ORIGINAL_MANIFEST_OR_DOMAIN")
        evidence_ref = _exact(doc.get("source_images")); erows = _image_rows(_read(evidence_ref["path"]))
        image_pins.append(evidence_ref)
        for ai,o in enumerate(doc.get("observations",[])):
            sid,f,j = o.get("session_id"),o.get("raw_frame_index"),o.get("joint_index_21")
            if sid not in contexts or o.get("side") != "right":
                continue
            if f not in FIT_FRAMES[sid]+VALIDATION_FRAMES[sid]:
                continue
            if o.get("visibility") != "VISIBLE":
                _fail("FIXED_K_SELECTED_TIP_NOT_VISIBLE")
            if (type(f) is not int or type(j) is not int or j not in TIP_IDS
                    or o.get("manus_node_id") != TIP_IDS[j]
                    or o.get("point_type") != "GLOVE_TIP" or o.get("semantics") != "VISUAL_PROXY"
                    or o.get("side_confirmed") is not True or o.get("joint_mapping_confirmed") is not True
                    or o.get("device_serial") != B_SERIAL):
                _fail("FIXED_K_VISIBLE_SDK_TIP_IDENTITY")
            key = (sid,f,j)
            if key in seen: _fail("FIXED_K_DUPLICATE_SELECTED_TIP")
            seen.add(key);ctx=contexts[sid];sess=ctx["session"];native=ctx["native"]
            if (sid,f) not in erows: _fail("FIXED_K_NATIVE_IMAGE_ROW_MISSING")
            er=erows[sid,f]
            if (o.get("qpc_ns") != int(sess.qpc_ns[f]) or er.get("qpc_ns") != int(sess.qpc_ns[f])
                    or er.get("device_serial") != B_SERIAL or er.get("image") != o.get("image")):
                _fail("FIXED_K_IMAGE_DEVICE_NATIVE_QPC")
            checked=_verified_native_tip(o,er,ctx,kr["T_head_camera_factory_fixed"],verify_pixels)
            image_ref=checked["image"];image_pins.append(image_ref)
            matrix,local,sigma=checked["matrix"],checked["local"],checked["sigma"]
            node=TIP_IDS[j]
            record={"annotation_kind":kind,"annotation_index":ai,"annotation_ref":annotation_ref,
                "source_images":evidence_ref,"image":image_ref,"session_id":sid,"device_serial":B_SERIAL,
                "side":"right","raw_frame_index":f,"qpc_ns":int(sess.qpc_ns[f]),
                "joint_index_21":j,"manus_node_id":node,"uv":o["uv"],"annotation_uncertainty_px":sigma,
                "controller_to_camera":matrix.tolist(),"local_point_m":local.tolist()}
            records.append(record)
    def data_for(frames):
        rr=[x for x in records if x["raw_frame_index"] in frames[x["session_id"]]]
        data={"side":"right","device_serial":B_SERIAL,
            "controller_to_camera":[x["controller_to_camera"] for x in rr],"local_points":[x["local_point_m"] for x in rr],
            "uv":[x["uv"] for x in rr],"session_ids":[x["session_id"] for x in rr],
            "raw_frame_indices":[x["raw_frame_index"] for x in rr],"joint_indices_21":[x["joint_index_21"] for x in rr],
            "annotation_uncertainty_px":[x["annotation_uncertainty_px"] for x in rr]}
        _data(data,4);return data,rr
    fit,fr=data_for(FIT_FRAMES);val,vr=data_for(VALIDATION_FRAMES)
    if len(fr) != 22 or len(vr) != 11 or set((x["session_id"],x["raw_frame_index"]) for x in fr) != set((s,f) for s,ff in FIT_FRAMES.items() for f in ff) or set((x["session_id"],x["raw_frame_index"]) for x in vr) != set((s,f) for s,ff in VALIDATION_FRAMES.items() for f in ff):
        _fail("FIXED_K_FROZEN_22_11_COUNT_OR_FRAME_COVERAGE")
    pins.update(split_plan=split_ref,annotations=annotation_refs,source_images=image_pins)
    for reference in list(annotation_refs.values()) + [split_ref] + image_pins: _exact(reference)
    # Recheck the pins captured at entry, not fresh hashes of possibly changed input.
    for name in ("input_manifest","camera_config","source_index","annotations_bundle"):_exact(pins[name])
    return {"bundle":bundle,"pins":pins,"base_config":config,"contexts":contexts,
        "fit_data":fit,"validation_data":val,"fit_records":fr,"validation_records":vr,
        "encoded_k_producer":k_proof,"k_report":k_report}

def validate_input_bundle(manifest_path, annotations_bundle_path, camera_config_path, source_index_path):
    return _validate_bundle(manifest_path,annotations_bundle_path,camera_config_path,source_index_path,True)

def _config_after(base_config,fit_result,prior,report_ref):
    value=copy.deepcopy(base_config)
    if fit_result["status"] == "PASSED":
        value["installation_prior_changed"]=True
        value["installation_changed_scope"]="DEVICE_B_RIGHT_ONLY"
        used=copy.deepcopy(prior);m=np.asarray(fit_result["T_controller_wrist_fitted"])
        used["right"]["pos"]=m[:3,3].tolist();used["right"]["quat"]=Rotation.from_matrix(m[:3,:3]).as_quat().tolist()
        value["devices"][B_SERIAL].update(installation_prior_override=used,
            installation_group_key=B_SERIAL+":"+PRIOR_SHA,installation_original_prior_sha256=PRIOR_SHA,
            installation_correction_evidence=report_ref,installation_diagnostic_status="PASSED")
    return value

def run_fixed_k_installation_fit(manifest_path,annotations_bundle_path,camera_config_path,source_index_path,output_root):
    v=validate_input_bundle(manifest_path,annotations_bundle_path,camera_config_path,source_index_path)
    prior=v["contexts"]["008"]["prior"];seed=source.installation_matrices(prior)[1]
    k=v["k_report"]["devices"][B_SERIAL]["K_image_fitted"]
    fit=fit_and_validate_right_installation(v["fit_data"],v["validation_data"],k,seed)
    output=audit._output_path(output_root)
    if output.exists():_fail("FIXED_K_FRESH_OUTPUT_REQUIRED")
    output.mkdir(parents=True,exist_ok=False)
    group={"status":fit["status"],"device_serial":B_SERIAL,"side":"right","original_prior_sha256":PRIOR_SHA,
        "prior_before":prior,"prior_used":copy.deepcopy(prior),"fit":fit["fit"],"validation":fit["validation"],
        "T_controller_wrist_before":fit["T_controller_wrist_before"],"T_controller_wrist_fitted":fit["T_controller_wrist_fitted"]}
    if fit["status"]=="PASSED":
        m=np.asarray(fit["T_controller_wrist_fitted"]);group["prior_used"]["right"]["pos"]=m[:3,3].tolist();group["prior_used"]["right"]["quat"]=Rotation.from_matrix(m[:3,:3]).as_quat().tolist()
    report={"schema":REPORT_SCHEMA,"status":fit["status"],**v["pins"],"encoded_k_producer":v["encoded_k_producer"],
        "groups":{B_SERIAL+":"+PRIOR_SHA:group},"K_image_fixed":k,
        "T_head_camera_factory_fixed":v["k_report"]["devices"][B_SERIAL]["T_head_camera_factory_fixed"],
        "fit_correspondences":v["fit_records"],"validation_correspondences":v["validation_records"],
        "fit_native_frames":FIT_FRAMES,"validation_native_frames":VALIDATION_FRAMES,
        "source_receipts":{s:x["source_receipt"] for s,x in v["contexts"].items()},
        "implementation":{"module":source.ref(Path(__file__)),"source":source.ref(Path(source.__file__)),"core":source.ref(Path(installation_fit.__file__))},
        "fitted_variables":"ONE_SHARED_DEVICE_B_RIGHT_CONTROLLER_LOCAL_SIX_DOF",
        "installation_prior_fitted":True,"left_installation_fitted":False,"device_A_fitted":False,
        "K_fitted":False,"camera_fitted":False,"time_fitted":False,"scale_fitted":False,"qpc_modified":False,"lens_remap":False,"holdout_used":False,
        "validation_used_by_optimizer":False,"validation_is_blind":False,
        "validation_not_independent_of_original_K_fit":True,"original_native_rgb_verified":True,
        "physical_ground_truth":False,"control_ground_truth":False,"training_eligible":False,
        "claim_limit":"Fixed-K bounded visual proxy estimate; nonblind development validation. GPU actual visual review still required; no physical calibration/contact qualification."}
    for name in ("input_manifest","camera_config","source_index","annotations_bundle"):_exact(v["pins"][name])
    for pin in list(v["pins"]["annotations"].values())+[v["pins"]["split_plan"]]+v["pins"]["source_images"]:_exact(pin)
    audit._write_json(output/REPORT_NAME,report);report_ref=source.ref(output/REPORT_NAME)
    config=_config_after(v["base_config"],fit,prior,report_ref)
    if fit["status"]=="PASSED":audit._write_json(output/"FITTED_CAMERA_CONFIG.json",config)
    else:(output/"FITTED_CAMERA_CONFIG.json").write_bytes(Path(camera_config_path).read_bytes())
    result={"schema":"pick_place_fixed_k_right_installation_result_v1","status":fit["status"],
        "fixed_k_installation_report":report_ref,"fitted_camera_config":source.ref(output/"FITTED_CAMERA_CONFIG.json"),
        "fit_count":22,"validation_count":11,"installation_diagnostic_status":fit["status"],
        "K_fitted":False,"camera_fitted":False,"time_fitted":False,"scale_fitted":False,"holdout_used":False,
        "physical_ground_truth":False,"control_ground_truth":False,"training_eligible":False}
    audit._write_json(output/"RESULT.json",result);return result

def validate_override_evidence(selected_config,original_reference,device_serial):
    if device_serial != B_SERIAL or original_reference.get("prior_sha256") != PRIOR_SHA:
        _fail("FIXED_K_OVERRIDE_RIGHT_ORIGINAL_GROUP_REQUIRED")
    report_ref=_exact(selected_config.get("installation_correction_evidence"))
    report,result,proof=_sealed_product(report_ref,REPORT_NAME,REPORT_SCHEMA,"fixed_k_installation_report")
    for role, path in {"module":Path(__file__),"source":Path(source.__file__),"core":Path(installation_fit.__file__)}.items():
        _exact(report.get("implementation",{}).get(role),path)
    if set(report.get("implementation",{})) != {"module","source","core"}:
        _fail("FIXED_K_OVERRIDE_IMPLEMENTATION_CLOSURE")
    if (report.get("installation_prior_fitted") is not True or report.get("original_native_rgb_verified") is not True
            or any(report.get(k) is not False for k in ("left_installation_fitted","device_A_fitted","K_fitted","camera_fitted","time_fitted","scale_fitted","qpc_modified","lens_remap","holdout_used","validation_used_by_optimizer","training_eligible","control_ground_truth","physical_ground_truth"))):
        _fail("FIXED_K_OVERRIDE_FIXED_DOMAIN_DECLARATIONS")
    if (report.get("fit_native_frames")!=FIT_FRAMES or report.get("validation_native_frames")!=VALIDATION_FRAMES
            or report.get("fitted_variables")!="ONE_SHARED_DEVICE_B_RIGHT_CONTROLLER_LOCAL_SIX_DOF"):
        _fail("FIXED_K_OVERRIDE_FROZEN_SPLIT_DECLARATION")
    _require_packet_inputs(proof,[report[k] for k in ("input_manifest","annotations_bundle","camera_config","source_index")])
    v=_validate_bundle(report["input_manifest"]["path"],report["annotations_bundle"]["path"],report["camera_config"]["path"],report["source_index"]["path"],False)
    if any(report.get(k)!=v["pins"][k] for k in v["pins"]) or report.get("encoded_k_producer")!=v["encoded_k_producer"]:
        _fail("FIXED_K_OVERRIDE_INPUT_PRODUCER_CLOSURE")
    _exact(original_reference["source_hdf5"],project_only=False)
    original,reference=source.read_installation(Path(original_reference["source_hdf5"]["path"]).parent)
    if reference["prior_sha256"] != PRIOR_SHA or original != v["contexts"]["008"]["prior"]:
        _fail("FIXED_K_OVERRIDE_ORIGINAL_PRIOR_VALUES")
    key=B_SERIAL+":"+PRIOR_SHA;groups=report.get("groups",{});g=groups.get(key,{})
    if (set(groups)!={key} or g.get("status")!="PASSED" or g.get("device_serial")!=B_SERIAL or g.get("side")!="right"
            or g.get("original_prior_sha256")!=PRIOR_SHA or g.get("prior_before")!=original
            or g.get("prior_used")!=selected_config.get("installation_prior_override")
            or g.get("prior_used",{}).get("left")!=original["left"]
            or report.get("fit_correspondences")!=v["fit_records"] or report.get("validation_correspondences")!=v["validation_records"]):
        _fail("FIXED_K_OVERRIDE_GROUP_LEFT_OR_CORRESPONDENCE")
    seed=source.installation_matrices(original)[1];used=source.installation_matrices(g["prior_used"])[1]
    reported_fitted=rigid(g["T_controller_wrist_fitted"],"fixed_k_report_fitted")
    fit=g["fit"];delta=np.asarray(fit.get("delta_controller_rotvec_rad",[])+fit.get("delta_controller_translation_m",[]),float)
    bounds=np.array([.35]*3+[.15]*3)
    if (delta.shape!=(6,) or not np.isfinite(delta).all() or np.any(np.abs(delta)>=bounds-1e-4)
            or not np.allclose(installation_fit._transform(delta,seed),used,atol=1e-8)
            or not np.allclose(g["T_controller_wrist_before"],seed,atol=1e-8)
            or not np.allclose(g["T_controller_wrist_fitted"],used,atol=1e-8)
            or fit.get("status")!="PASSED" or fit.get("optimizer",{}).get("success") is not True
            or fit.get("boundary_hit_parameter_indices")!=[] or fit.get("jacobian_rank")!=6
            or not np.isfinite(float(fit.get("jacobian_condition",float("nan"))))
            or not 0 <= float(fit["jacobian_condition"]) < 1e8
            or fit.get("rotation_vector_component_bound_rad")!=.35 or fit.get("translation_component_bound_m")!=.15
            or fit.get("delta_applied_on_left_in_controller_frame") is not True):
        _fail("FIXED_K_OVERRIDE_FIT_TRANSFORM_OR_GATE")
    k=_k(v["k_report"]["devices"][B_SERIAL]["K_image_fitted"])
    if (not np.array_equal(report.get("K_image_fixed"),k)
            or not np.array_equal(report.get("T_head_camera_factory_fixed"),v["k_report"]["devices"][B_SERIAL]["T_head_camera_factory_fixed"])):
        _fail("FIXED_K_OVERRIDE_K_OR_FACTORY_CAMERA_CHANGED")
    f_eval=_evaluation(_data(v["fit_data"],8),k,seed,reported_fitted);val=_evaluation(_data(v["validation_data"],4),k,seed,reported_fitted)
    if f_eval["status"]!="PASSED" or val["status"]!="PASSED" or g.get("validation")!=val:
        _fail("FIXED_K_OVERRIDE_STATIC_RECOMPUTED_VALIDATION_FAILED")
    frozen_config=_read(result["fitted_camera_config"]["path"])
    expected=_config_after(v["base_config"],{"status":"PASSED","T_controller_wrist_fitted":reported_fitted.tolist()},original,report_ref)
    if frozen_config!=expected or selected_config!=frozen_config["devices"][B_SERIAL]:
        _fail("FIXED_K_OVERRIDE_FROZEN_CONFIG_FIELDS_CHANGED")
    return copy.deepcopy(g["prior_used"])
