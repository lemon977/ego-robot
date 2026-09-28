"""C02 candidate 2: refine only hand-supported, card-excluded UNKNOWN pixels."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from chaoyang.ops import run_four_stream_completion_clean as c01
from chaoyang.ops import run_four_stream_completion_clean_mask_support as c02

HAND_ROOT="_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/masks_042_v1"
CARD_ROOT="_run/current/human_to_robot_quality_acceptance_20260924/attempts/attempt_0001/lanes/scene/POKER_076_091_CURRENT_CARD_SUPPORT"

def refine_masks(write, protect, unknown, hand, visible_card):
    arrays=[write,protect,unknown,hand,visible_card]
    if any(x.dtype!=np.bool_ or x.shape!=write.shape for x in arrays):
        raise ValueError("BOOLEAN_MATCHING_DOMAIN_REQUIRED")
    if np.any(write&protect) or np.any(write&unknown):
        raise ValueError("PREDECESSOR_MASK_CONFLICT")
    eligible=unknown&hand&~visible_card&~protect
    new_write=write|eligible
    remaining=unknown&~eligible
    conflict=unknown&hand&visible_card
    if np.any(new_write&protect) or np.any(eligible&visible_card) or np.any(conflict&new_write):
        raise RuntimeError("ROLE_GUARD_FAILED")
    return eligible,new_write,remaining,conflict

def load_mask(path):
    value=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if value is None or value.ndim!=2:
        raise RuntimeError("MASK_UNREADABLE")
    return value>0

def freeze(repo,lane):
    destination=lane/"C02_ROLE_INPUT_FIX2"
    destination.mkdir()
    old=c01.read(lane/"C01_FROZEN_INPUT.json")
    base=c01.read(repo/c01.BASE/"INPUT.json")
    hand_manifest=c01.read(repo/HAND_ROOT/"MASK_MANIFEST.json")
    card_result=c01.read(repo/CARD_ROOT/"RESULT.json")
    if hand_manifest.get("session_id")!="play_cards_0902_042" or hand_manifest.get("frame_count")!=171:
        raise RuntimeError("HAND_SESSION_DRIFT")
    if card_result.get("nonempty_frames")!=16:
        raise RuntimeError("CARD_SCOPE_DRIFT")
    rows=[]
    for row,source,card in zip(old["rows"],base["rows"],card_result["rows"],strict=True):
        fid=row["frame_id"]
        if source["frame_id"]!=fid or card["source_frame"]!=fid:
            raise RuntimeError("ROLE_FRAME_DRIFT")
        refs={k:row[k] for k in ("raw","write","protect","propagated_frame","residual_propagation_mask")}
        refs.update(unknown=source["unknown"],hand=c01.ref(repo/HAND_ROOT/"human_hand"/f"{fid:06d}.png"),visible_card=card["mask"])
        for value in refs.values():c01.verify(value)
        w,p,u,h,v=[load_mask(refs[k]["path"]) for k in ("write","protect","unknown","hand","visible_card")]
        if w.shape!=(960,1280):
            raise RuntimeError("ROLE_PIXEL_DOMAIN")
        e,n,remaining,conflict=refine_masks(w,p,u,h,v)
        files={}
        for name,pixels in (("eligible",e),("write",n),("unknown",remaining),("conflict",conflict)):
            path=destination/f"{fid:06d}_{name}.png"
            c01.save(path,pixels.astype(np.uint8)*255)
            files[name]=c01.ref(path)
        residual=load_mask(refs["residual_propagation_mask"]["path"])
        extra720=cv2.resize(e.astype(np.float32),(960,720),interpolation=cv2.INTER_LINEAR)>0
        model=residual|extra720
        path=destination/f"{fid:06d}_residual.png"
        c01.save(path,model.astype(np.uint8)*255);files["residual"]=c01.ref(path)
        rows.append({"frame_id":fid,"source":refs,"refined":files,
                     "eligible_px":int(e.sum()),"remaining_unknown_px":int(remaining.sum()),
                     "hand_card_unknown_conflict_px":int(conflict.sum()),
                     "new_residual_model_px":int((model&~residual).sum())})
    frozen={"schema_version":"C02_VISIBLE_ROLE_REFINEMENT_V1","task_id":c01.TASK,"candidate_ordinal":2,
            "session_id":"play_cards_0902_042","frame_range":[76,91],
            "c01_frozen":c01.ref(lane/"C01_FROZEN_INPUT.json"),
            "predecessor":c01.ref(lane/"C02_MASK_SUPPORT_FIX1/RESULT.json"),
            "hand_manifest":c01.ref(repo/HAND_ROOT/"MASK_MANIFEST.json"),
            "card_result":c01.ref(repo/CARD_ROOT/"RESULT.json"),
            "code":c01.ref(Path(__file__)),"support_code":c01.ref(Path(c02.__file__)),
            "formula":"eligible=UNKNOWN & HAND & ~VISIBLE_CARD & ~PROTECT; write2=write1|eligible",
            "role_authority":"MODEL_ROLE_CROSS_CHECK_PLUS_FIXED_FRAME_AI_REVIEW_NOT_INDEPENDENT_TRUTH",
            "geometry_input_allowed":False,"rows":rows}
    path=lane/"C02_ROLE_FROZEN_FIX2.json";c01.write_json(path,frozen)
    return {"frozen":str(path),"frames":len(rows),"eligible_pixels":sum(r["eligible_px"] for r in rows)}

def run(repo,lane):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise RuntimeError("CPU_ONLY")
    cv2.setNumThreads(2)
    frozen_path=lane/"C02_ROLE_FROZEN_FIX2.json";frozen=c01.read(frozen_path)
    for key in ("c01_frozen","predecessor","hand_manifest","card_result","code","support_code"):
        c01.verify(frozen[key])
    original=c01.read(lane/"C01_FROZEN_INPUT.json")
    for key in ("weight","runtime","source_manifest","propagation_manifest","predecessor"):
        c01.verify(original[key])
    for item in original["code"]:c01.verify(item)
    sys.path.insert(0,str(repo/c01.OLD/"environments/onnxruntime-1.22.1"))
    import onnxruntime as ort
    if ort.__version__!="1.22.1":raise RuntimeError("ORT_VERSION_DRIFT")
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session=ort.InferenceSession(str(repo/c01.WEIGHT),sess_options=options,providers=["CPUExecutionProvider"])
    ins={x.name:x for x in session.get_inputs()};outs=session.get_outputs()
    if set(ins)!={"image","mask"} or len(outs)!=1:raise RuntimeError("ONNX_NAMES")
    for node,channels in ((ins["image"],3),(ins["mask"],1),(outs[0],3)):
        if node.type!="tensor(float)" or not c01._lama_shape_matches(node.shape,channels):raise RuntimeError("ONNX_CONTRACT")
    out=lane/"C02_ROLE_FIX2";out.mkdir()
    for sub in ("clean","internal"):(out/sub).mkdir()
    started=time.monotonic();rows=[]
    for row in frozen["rows"]:
        for item in [*row["source"].values(),*row["refined"].values()]:c01.verify(item)
        fid=row["frame_id"];source=row["source"];refined=row["refined"]
        raw=c01.image(source["raw"]["path"]);prop=c01.image(source["propagated_frame"]["path"])
        mask=load_mask(refined["residual"]["path"])
        wrapper=c02.SupportSession(session,mask)
        generated=c01._apply_lama_residual(wrapper,outs[0].name,prop,mask)
        if np.any(generated[~mask]!=prop[~mask]):raise RuntimeError("RESIDUAL_WRITE_ESCAPE")
        write=load_mask(refined["write"]["path"]);protect=load_mask(source["protect"]["path"])
        unknown=load_mask(refined["unknown"]["path"]);conflict=load_mask(refined["conflict"]["path"])
        output=c01.composite_clean(raw,cv2.resize(generated,(1280,960)),write,protect)
        changed=np.any(raw!=output,axis=2)
        if np.any(changed&(~write|protect|unknown|conflict)):raise RuntimeError("ROLE_FINAL_WRITE_ESCAPE")
        c01.save(out/"clean"/f"{fid:06d}.png",output);c01.save(out/"internal"/f"{fid:06d}.png",generated)
        rows.append({"frame_id":fid,"clean":c01.ref(out/"clean"/f"{fid:06d}.png"),"probe":wrapper.probe,
                     "eligible_px":row["eligible_px"],"remaining_unknown_px":row["remaining_unknown_px"],
                     "outside_new_write_changed":0,"protected_changed":0,"remaining_unknown_changed":0,
                     "hand_card_conflict_changed":0})
    result={"schema_version":"C02_ROLE_RESULT_V1","task_id":c01.TASK,"candidate_ordinal":2,
            "execution":"REAL_CPU_LAMA_16_FRAME_REPLAY","structure":"PASS","quality":"PENDING_FIXED_WINDOW_REVIEW",
            "adoption":"NOT_ADOPTED","human_review":"NOT_PERFORMED","full171":"NOT_RUN",
            "frozen":c01.ref(frozen_path),"rows":rows,"elapsed_seconds":time.monotonic()-started,
            "gpu_used":False,"providers":session.get_providers(),"geometry_input_allowed":False}
    c01.write_json(out/"RESULT.json",result)
    return {"result":str(out/"RESULT.json"),"frames":len(rows)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("freeze","run"))
    p.add_argument("--repo-root",required=True);p.add_argument("--attempt",default="attempt_0001")
    args=p.parse_args();repo,lane=c01.safe_context(args.repo_root,args.attempt)
    print(json.dumps({"freeze":freeze,"run":run}[args.stage](repo,lane)))
    return 0
if __name__=="__main__":raise SystemExit(main())
