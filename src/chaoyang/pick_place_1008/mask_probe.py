"""Finite observed-mask GPU probe. No inpainting, action or contact qualification."""
from pathlib import Path
import time,json
import cv2,numpy as np
from . import audit,source
from .pipeline import read,write,pack_masks
from .mask_seeds import load_visible_seed

def _sealed_index(repo,path):
    repo,path=Path(repo).resolve(),Path(path)
    reference=source.ref(path);audit._verify_exact_ref(reference,project_only=True)
    outer_path=path.parent.parent/"RESULT.json";outer=source.read_json(outer_path)
    task_id=outer.get("task_id")
    expected=repo/"_run/current"/str(task_id)/"attempts/attempt_0001/candidate/INDEX.json"
    if path!=expected or not str(task_id).startswith("exact78_pick_place_"):
        raise ValueError("MASK_PROBE_INDEX_OWNERSHIP")
    ledger=source.read_json(repo/"docs/governance/LONG_HORIZON_TASK_STATE.json")
    entries=[row for row in ledger.get("tasks",[]) if row.get("task_id")==task_id]
    if len(entries)!=1 or entries[0].get("status")!="PASSED" or entries[0].get("pid") is not None or entries[0].get("proc_start_ticks") is not None:
        raise ValueError("MASK_PROBE_PRODUCER_NOT_SEALED")
    audit._verify_exact_ref(entries[0]["result"],project_only=True)
    if entries[0]["result"]!=source.ref(outer_path):
        raise ValueError("MASK_PROBE_TERMINAL_LEDGER_PIN")
    if (outer.get("status")!="PASSED" or outer.get("validation_errors")!=[]
            or outer.get("required_outputs",{}).get("candidate/INDEX.json")!=reference
            or outer.get("run_signature")!=entries[0].get("run_signature")):
        raise ValueError("MASK_PROBE_INDEX_NOT_SEALED")
    expected_packet=repo/"tasks/current"/task_id/"TASK_PACKET.json"
    if outer["task_packet"].get("path")!=str(expected_packet):raise ValueError("MASK_PROBE_PACKET_OWNERSHIP")
    audit._verify_exact_ref(outer["task_packet"],project_only=True)
    packet=source.read_json(expected_packet)
    if packet.get("task_id")!=task_id or packet.get("run_signature")!=outer["run_signature"] or packet.get("execution_mode")!="EXACT78_CANDIDATE":
        raise ValueError("MASK_PROBE_PACKET_IDENTITY")
    value=source.read_json(path)
    if value.get("status")!="COMPLETE":raise ValueError("MASK_PROBE_SOURCE_INDEX_INCOMPLETE")
    return value,reference

def _sealed_source(repo,index_root,sid,source_ref):
    from .stage_reuse import _validate
    from .pipeline import STAGE_FILES
    audit._verify_exact_ref(source_ref,project_only=True)
    receipt=read(source_ref["path"])
    found,validated_ref=_validate(Path(repo),Path(index_root),sid,"source",receipt["binding"],STAGE_FILES["source"])
    if found is None or validated_ref!=source_ref:raise ValueError("MASK_PROBE_CANONICAL_SOURCE_NOT_SEALED")
    return found/"source"


def _save_point_diagnostics(folder, masks, shape):
    """Save independently owned raw model masks, without granting deletion authority."""
    from collections.abc import Mapping
    count, height, width = shape
    if any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError("POINT_DIAGNOSTICS_NATIVE_SHAPE")
    tracks = masks.get("point_object_tracks")
    records = masks.get("metadata", {}).get("point_tracking", {}).get("objects")
    if not isinstance(tracks, Mapping) or not isinstance(records, Mapping):
        raise ValueError("POINT_DIAGNOSTICS_EXPLICIT_RAW_MODEL_TRACKS_REQUIRED")
    ids = sorted(int(key) for key in records)
    if len(set(ids)) != len(ids) or any(not 1 <= obj <= 16 for obj in ids):
        raise ValueError("POINT_DIAGNOSTICS_OBJECT_IDS")
    if set(tracks) - set(ids):
        raise ValueError("POINT_DIAGNOSTICS_UNSEEDED_OBJECT")
    packed_size = (height * width + 7) // 8
    packed = np.zeros((len(ids), count, packed_size), np.uint8)
    nonempty = np.zeros((len(ids), count), bool)
    objects = []
    for position, obj in enumerate(ids):
        record = records[str(obj)]
        if record.get("role") not in {"glove", "controller", "mount", "attachment", "forearm", "cable", "human"} or record.get("side") not in {"left", "right"}:
            raise ValueError("POINT_DIAGNOSTICS_ROLE_SIDE")
        frames = tracks.get(obj, {})
        if not isinstance(frames, Mapping):
            raise ValueError("POINT_DIAGNOSTICS_FRAME_MAPPING")
        for frame, value in frames.items():
            if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)) or not 0 <= frame < count:
                raise ValueError("POINT_DIAGNOSTICS_FRAME_DOMAIN")
            value = np.asarray(value)
            if value.dtype != np.uint8 or value.shape != (packed_size,):
                raise ValueError("POINT_DIAGNOSTICS_PACKED_DOMAIN")
            padding = packed_size * 8 - height * width
            if padding and int(value[-1]) & ((1 << padding) - 1):
                raise ValueError("POINT_DIAGNOSTICS_PADDING_BITS")
            packed[position, int(frame)] = value
            nonempty[position, int(frame)] = bool(value.any())
        objects.append({"obj_id": obj, "role": record["role"], "side": record["side"],
                        "model_nonempty_raw_frame_indices": np.flatnonzero(nonempty[position]).tolist()})
    folder = Path(folder)
    arrays_path, info_path = folder / "point_object_masks.npz", folder / "POINT_OBJECT_DIAGNOSTICS.json"
    if arrays_path.exists() or info_path.exists():
        raise ValueError("POINT_DIAGNOSTICS_FRESH_OUTPUT_REQUIRED")
    np.savez_compressed(arrays_path, object_ids=np.asarray(ids, np.int64),
                        packed_masks=packed, model_nonempty=nonempty,
                        raw_shape=np.asarray(shape, np.int64))
    arrays_ref = source.ref(arrays_path)
    write(info_path, {"schema": "pick_place_raw_point_object_diagnostics_v1",
                     "packed_arrays": arrays_ref, "objects": objects,
                     "mask_interpretation": "RAW_SAM31_MODEL_PROPOSALS_BEFORE_CONSERVATIVE_COMPONENT_VETO",
                     "bitorder": "big", "raw_shape": list(shape),
                     "missing_model_masks_are_zero_not_hidden_observations": True,
                     "physical_ground_truth": False, "training_eligible": False,
                     "deletion_authority_added_by_diagnostics": False})
    return {"arrays": arrays_ref, "metadata": source.ref(info_path)}

def run_mask_probe(repo,output,source_index_path,seed_index_path,session_ids):
    from chaoyang.exact78_baseline.mask import Sam3Video
    from .visual import build_visual_masks
    repo,output=Path(repo),audit._output_path(output)
    current,source_pin=_sealed_index(repo,source_index_path)
    rows={r["session_id"]:r for r in current["sessions"]}
    if len(rows)!=len(current["sessions"]):raise ValueError("MASK_PROBE_SOURCE_SESSION_DUPLICATES")
    seed_index_pin=source.ref(Path(seed_index_path))
    if len(set(session_ids))!=len(session_ids) or not set(session_ids)<=set(rows):
        raise ValueError("MASK_PROBE_SESSION_SET")
    output.mkdir(parents=True,exist_ok=False);model=None;reports=[]
    try:
        for sid in session_ids:
            tick=time.monotonic();folder=output/sid;folder.mkdir()
            source_ref=rows[sid]["stage_receipts"]["source"]
            source_folder=_sealed_source(repo,Path(source_index_path).parent,sid,source_ref)
            session=source.load_prepared(source_folder)
            rgb=source.load_rgb_frames(session,640,480)
            if rgb.shape!=(session.frame_count,480,640,3) or rgb.dtype!=np.uint8:
                raise ValueError("MASK_PROBE_RAW_RGB_DOMAIN")
            audit._verify_exact_ref(seed_index_pin,project_only=True)
            seeds=load_visible_seed(seed_index_path,sid,session,rgb)
            audit._verify_exact_ref(seed_index_pin,project_only=True)
            if model is None:model=Sam3Video(repo)
            segmentation_tick=time.monotonic()
            masks=build_visual_masks(model,rgb,session,visible_seeds=seeds)
            segmentation_seconds=time.monotonic()-segmentation_tick
            np.savez_compressed(folder/"raw_rgb.npz",rgb=rgb)
            np.savez_compressed(folder/"masks.npz",**pack_masks(masks))
            write(folder/"masks.json",masks["metadata"])
            point_diagnostics=_save_point_diagnostics(folder,masks,tuple(rgb.shape[:3]))
            images=[]
            anchors=sorted({0,len(rgb)-1,*[int(round((len(rgb)-1)*x)) for x in (.25,.5,.75)],*([125] if sid=="008" and len(rgb)>125 else [])})
            for frame in anchors:
                raw=rgb[frame,:,:,::-1].copy();overlay=raw.copy()
                removal=masks["removal"][frame];protected=masks["protected_visible"][frame]
                overlay[removal]=(.45*overlay[removal]+.55*np.array([200,0,210])).astype(np.uint8)
                overlay[protected]=(.35*overlay[protected]+.65*np.array([240,210,0])).astype(np.uint8)
                montage=np.hstack([raw,overlay])
                image=folder/f"raw_mask_{frame:04d}.png"
                if not cv2.imwrite(str(image),montage):raise IOError("MASK_PROBE_IMAGE_WRITE")
                images.append({"raw_frame_index":frame,"qpc_ns":int(session.qpc_ns[frame]),"image":source.ref(image)})
            reports.append({"session_id":sid,"status":"EXECUTION_COMPLETE_REVIEW_PENDING",
                "frames":session.frame_count,"elapsed_seconds":time.monotonic()-tick,
                "segmentation_seconds":segmentation_seconds,"raw_rgb":source.ref(folder/"raw_rgb.npz"),
                "source_receipt":source_ref,"original_rgb_source":"DIRECT_ORIGINAL_VST_DECODE_CROP_RESIZE_RGB_UINT8","visible_seeds":seeds["seed_reference"],
                "masks":source.ref(folder/"masks.npz"),"mask_info":source.ref(folder/"masks.json"),
                "point_object_diagnostics":point_diagnostics,
                "images":images,"quality_problems":masks["metadata"].get("quality_problems",[])})
            write(output/"PROBE_PROGRESS.json",{"status":"RUNNING","sessions":reports})
    finally:
        if model is not None:model.close()
    result={"schema":"pick_place_mask_probe_v1","status":"COMPLETE","sessions":reports,
        "source_index":source_pin,"mask_seed_index":seed_index_pin,
        "inpainting_performed":False,"retargeting_performed":False,"review_status":"AI_REVIEW_PENDING",
        "training_eligible":False,"control_ground_truth":False}
    audit._verify_exact_ref(seed_index_pin,project_only=True)
    write(output/"PROBE.json",result);write(output/"RESULT.json",result)
    return result
