"""Registered CPU replay of existing Clean production consumer; no new algorithm candidate."""
from __future__ import annotations
import argparse,ast,inspect,os,sys,time
from pathlib import Path
import cv2,numpy as np
from chaoyang.ops import run_four_stream_completion_clean as c01
from chaoyang.ops import run_human_to_robot_shared_delivery_clean as production

OUTPUT="C01_PRODUCTION_CONSUMER_REPLAY_V1"
FROZEN="C01_PRODUCTION_CONSUMER_REPLAY_INPUT_V1.json"
LABELS={0:"RAW_OUTSIDE_WRITE",1:"FROZEN_PROPAGATION_NOT_DIRECT_OBSERVATION",2:"LAMA_GENERATED",3:"LINEAR_RESIZE_MIX_PROPAGATION_LAMA",4:"RAW_PROTECTED",5:"RAW_UNKNOWN_PRESERVED"}

def consumer_wiring():
    source=inspect.getsource(production.lama_candidate_infer)
    calls={n.func.id for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    if not {"_apply_lama_residual","composite_clean"}<=calls:raise RuntimeError("PRODUCTION_CALL_CHAIN_DRIFT")
    if production._apply_lama_residual is not c01._apply_lama_residual:raise RuntimeError("HELPER_IDENTITY_DRIFT")
    return {"entry":"run_human_to_robot_shared_delivery_clean.lama_candidate_infer",
      "consumer":"run_human_to_robot_shared_delivery_clean._apply_lama_residual",
      "compositor":"chaoyang.pipeline.v5_scene.composite_clean",
      "same_helper_object":True,"legacy_full_entry_not_executed":True,
      "full_entry_requires_gpu_propainter_and_old_task":"NOT_REVIVED",
      "scope":"CPU production post-propagation consumer; frozen ProPainter reused, not full pipeline rerun"}

def consume(session,name,prop,residual,raw,write,protect,unknown):
    if raw.shape!=(960,1280,3) or raw.dtype!=np.uint8:raise ValueError("RAW_DOMAIN")
    for m in (write,protect,unknown):
        if m.shape!=raw.shape[:2] or m.dtype!=bool:raise ValueError("MASK_DOMAIN")
    if (write&(protect|unknown)).any():raise ValueError("WRITE_CONFLICT")
    if prop.dtype!=np.uint8 or residual.dtype!=np.uint8:raise ValueError("INTERNAL_DTYPE")
    generated=production._apply_lama_residual(session,name,prop,residual)
    clean=production.composite_clean(raw,cv2.resize(generated,(1280,960)),write,protect)
    changed=np.any(clean!=raw,axis=2)
    if (changed&(~write|protect|unknown)).any():raise RuntimeError("WRITE_BOUNDARY")
    # Same linear resize support as actual final consumer. This labels direct source lineage, not semantic truth.
    fraction=cv2.resize((residual>0).astype(np.float32),(1280,960),interpolation=cv2.INTER_LINEAR)
    labels=np.zeros(write.shape,np.uint8)
    labels[write&(fraction==0)]=1;labels[write&(fraction>=1)]=2;labels[write&(fraction>0)&(fraction<1)]=3
    labels[unknown]=5;labels[protect]=4
    return clean,generated,labels,{"outside_write_changed":int((changed&~write).sum()),
      "protected_changed":int((changed&protect).sum()),"unknown_changed":int((changed&unknown).sum()),
      "source_pixel_counts":{str(k):int((labels==k).sum()) for k in LABELS},
      "raw_fallback_inside_write":False,"model_failure_policy":"RAISE_NO_RAW_SUCCESS_FALLBACK"}

def code_refs():
    return [c01.ref(Path(__file__)),c01.ref(Path(production.__file__)),c01.ref(Path(c01.__file__)),
            c01.ref(Path(inspect.getfile(production.composite_clean)))]

def freeze(repo,lane):
    wiring=consumer_wiring();old_path=lane/"C01_FROZEN_INPUT.json";old=c01.read(old_path)
    if old["session_id"]!="play_cards_0902_042" or old["frame_range"]!=[76,91]:raise RuntimeError("SESSION_RANGE")
    base=c01.read(c01.verify(old["source_manifest"]))
    if [r["frame_id"] for r in old["rows"]]!=list(range(76,92)) or [r["frame_id"] for r in base["rows"]]!=list(range(76,92)):raise RuntimeError("FRAME_MAPPING")
    for r in old["code"]:c01.verify(r)
    for k in ("weight","runtime","propagation_manifest","predecessor"):c01.verify(old[k])
    rows=[]
    for r,b in zip(old["rows"],base["rows"],strict=True):
        item=dict(r);item["unknown"]=c01.ref(c01.verify(b["unknown"]))
        item["c01_replay_clean"]=c01.ref(lane/"C01_ADAPTER_FIX1/clean"/f"{r['frame_id']:06d}.png")
        for v in item.values():
            if isinstance(v,dict):c01.verify(v)
        rows.append(item)
    c01.write_json(lane/FROZEN,{"schema_version":"C01_PRODUCTION_REPLAY_FROZEN_V1","task_id":c01.TASK,
      "frozen_input":c01.ref(old_path),"rows":rows,"weight":old["weight"],"runtime":old["runtime"],
      "code":code_refs(),"wiring":wiring,"algorithm_candidate_consumed":False,
      "change":"No algorithm change; production helper already fixed. Audit consumer wiring, pixel lineage and byte parity."})
    return {"frozen":str(lane/FROZEN),"frames":16}

def run(repo,lane):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU_ONLY")
    cv2.setNumThreads(2);data=c01.read(lane/FROZEN);c01.verify(data["frozen_input"])
    if data["code"]!=code_refs():raise RuntimeError("CODE_CLOSURE_DRIFT")
    for k in ("weight","runtime"):c01.verify(data[k])
    if [r["frame_id"] for r in data["rows"]]!=list(range(76,92)):raise RuntimeError("FRAME_MAPPING")
    # Validate whole finite input set before model execution or output creation.
    for r in data["rows"]:
        for v in r.values():
            if isinstance(v,dict):c01.verify(v)
    dest=lane/OUTPUT
    if dest.exists():raise FileExistsError(dest)
    runtime=repo/c01.OLD/"environments/onnxruntime-1.22.1";sys.path.insert(0,str(runtime))
    import onnxruntime as ort
    if ort.__version__!="1.22.1" or not Path(ort.__file__).resolve().is_relative_to(runtime.resolve()):raise RuntimeError("ORT_ORIGIN")
    opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;opt.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    core=ort.InferenceSession(str(c01.verify(data["weight"])),sess_options=opt,providers=["CPUExecutionProvider"])
    ins={n.name:n for n in core.get_inputs()};outs=core.get_outputs()
    if set(ins)!={"image","mask"} or len(outs)!=1:raise RuntimeError("ONNX_NAMES")
    for n,ch in ((ins["image"],3),(ins["mask"],1),(outs[0],3)):
        if n.type!="tensor(float)" or not c01._lama_shape_matches(n.shape,ch):raise RuntimeError("ONNX_SHAPE")
    audited=c01.AuditedSession(core);dest.mkdir()
    for folder in ("clean","internal","source_map"):(dest/folder).mkdir()
    started=time.monotonic();results=[]
    for r in data["rows"]:
        fid=r["frame_id"];raw=c01.image(r["raw"]["path"]);prop=c01.image(r["propagated_frame"]["path"])
        residual=c01.image(r["residual_propagation_mask"]["path"],True)
        write,protect,unknown=[c01.image(r[k]["path"],True)>0 for k in ("write","protect","unknown")]
        previous=len(audited.rows)
        clean,internal,labels,metrics=consume(audited,outs[0].name,prop,residual,raw,write,protect,unknown)
        expected=c01.image(r["c01_replay_clean"]["path"])
        parity=bool(np.array_equal(clean,expected))
        if not parity:raise RuntimeError(f"C01_PARITY_DRIFT:{fid}")
        for folder,array in (("clean",clean),("internal",internal),("source_map",labels)):c01.save(dest/folder/f"{fid:06d}.png",array)
        results.append({"frame_id":fid,"clean":c01.ref(dest/"clean"/f"{fid:06d}.png"),
          "source_map":c01.ref(dest/"source_map"/f"{fid:06d}.png"),**metrics,"same_pixels_as_frozen_c01":parity,
          "model_called":len(audited.rows)>previous,"probe":audited.rows[-1] if len(audited.rows)>previous else None})
    result={"schema_version":"C01_PRODUCTION_CONSUMER_REPLAY_V1","execution":"REAL_CPU_MODEL_POST_PROPAGATION_CONSUMER_REPLAY",
      "task_id":c01.TASK,"frozen":c01.ref(lane/FROZEN),"wiring":consumer_wiring(),"rows":results,
      "pixel_labels":LABELS,"elapsed_seconds":time.monotonic()-started,"gpu_used":False,"providers":core.get_providers(),
      "propainter_rerun":False,"algorithm_candidate_consumed":False,"numeric_adapter_regression":"PASS",
      "quality":"NOT_REEVALUATED_EXISTING_C01_VISUAL_REJECTED","adoption":"NOT_ADOPTED_AS_COMPLETE_CLEAN",
      "human_review":"NOT_PERFORMED","scope_limit":"No C02 or mask/resize change; no card fill from three edges; no raw fallback counted as successful removal."}
    c01.write_json(dest/"RESULT.json",result)
    return {"result":str(dest/"RESULT.json"),"frames":len(results)}

def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["freeze","run"]);p.add_argument("--repo-root",required=True);p.add_argument("--attempt",default="attempt_0001");a=p.parse_args()
    repo,lane=c01.safe_context(a.repo_root,a.attempt)
    print(__import__("json").dumps((freeze if a.stage=="freeze" else run)(repo,lane),ensure_ascii=False));return 0
if __name__=="__main__":raise SystemExit(main())
