"""C02 final budgeted candidate: suppress original hand support in model conditioning only."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
import cv2,numpy as np
from chaoyang.ops import run_four_stream_completion_clean as c01
from chaoyang.ops import run_four_stream_completion_clean_mask_support as c02
from chaoyang.ops import run_four_stream_completion_clean_production as consumer
OUT="C02_CONTEXT_MASK_FIX3";FROZEN="C02_CONTEXT_MASK_FIX3_FROZEN.json"

def hand_context(hand):
    if hand.shape!=(960,1280) or hand.dtype!=bool:raise ValueError("HAND_DOMAIN")
    h=cv2.resize(hand.astype(np.float32),(960,720),interpolation=cv2.INTER_LINEAR)>0
    return c02.support_mask(h)

class ContextSession(c02.SupportSession):
    def __init__(self,core,residual,hand):
        super().__init__(core,residual);self.previous_support=self.support.copy()
        self.hand_support=hand_context(hand)
        self.support=np.maximum(self.support,self.hand_support)
    def run(self,outputs,inputs):
        result=super().run(outputs,inputs)
        self.probe.update(candidate_ordinal=3,conditioning_added_from_hand_px=int(((self.support>0)&(self.previous_support==0)).sum()),
          unmasked_hand_support_px=int(((self.hand_support>0)&(self.support==0)).sum()),
          hand_support_authority="Existing segmentation hypothesis, not independent truth",
          output_write_scope_changed=False)
        return result

def visible_card_guard(raw,reference,write,protect,card):
    if raw.shape!=reference.shape or raw.dtype!=np.uint8 or reference.dtype!=np.uint8:
        raise ValueError("REFERENCE_DOMAIN")
    for m in (write,protect,card):
        if m.shape!=raw.shape[:2] or m.dtype!=bool:raise ValueError("GUARD_MASK_DOMAIN")
    guarded_write=write&~card;guarded_protect=protect|card
    fair=reference.copy();fair[card]=raw[card]
    return guarded_write,guarded_protect,fair,{"write_removed_px":int((write&card).sum()),
        "reference_restored_changed_px":int(np.any(reference[card]!=raw[card],axis=1).sum())}

def mask(ref):
    a=cv2.imread(str(c01.verify(ref)),cv2.IMREAD_UNCHANGED)
    if a is None:raise RuntimeError("MASK_UNREADABLE")
    return a>0

def closure():
    return [c01.ref(Path(m.__file__)) for m in (sys.modules[__name__],c01,c02,consumer)]+consumer.code_refs()

def budget(repo,lane):
    packet=c01.read(repo/"tasks/current"/c01.TASK/"TASK_PACKET.json")
    limit=packet["budgets"]["candidates_per_root_cause"]
    refs=[c01.ref(lane/p/"RESULT.json") for p in ("C02_MASK_SUPPORT_FIX1","C02_ROLE_FIX2")]
    ordinals=[c01.read(r["path"])["candidate_ordinal"] for r in refs]
    if limit!=3 or ordinals!=[1,2]:raise RuntimeError("C02_CANDIDATE_BUDGET")
    if (lane/OUT).exists():raise RuntimeError("THIRD_CANDIDATE_ALREADY_STARTED")
    return {"limit":limit,"already_executed_ordinals":ordinals,"remaining_before_run":1,"history":refs,
      "new_ordinal":3,"on_quality_failure":"ROOT_CAUSE_REVIEW_NO_FOURTH_RENAMED_CANDIDATE"}

def freeze(repo,lane):
    ledger=budget(repo,lane);prior=c01.ref(lane/"C02_ROLE_FROZEN_FIX2.json");data=c01.read(prior["path"])
    original=c01.read(c01.verify(data["c01_frozen"]))
    for r in data["rows"]:
        for v in [*r["source"].values(),*r["refined"].values()]:c01.verify(v)
        # Existing candidate2 overlaps visible-card mask on14 pixels; freeze explicit guard below.
    refs=closure()+original["code"]+[data["code"],data["support_code"]]
    for r in refs:c01.verify(r)
    input_dir=lane/"C02_CONTEXT_MASK_INPUT_FIX3";input_dir.mkdir()
    previous=c01.read(ledger["history"][-1]["path"])
    if [r["frame_id"] for r in previous["rows"]]!=list(range(76,92)):raise RuntimeError("PREVIOUS_FRAME_MAPPING")
    for r,pr in zip(data["rows"],previous["rows"],strict=True):
        fid=r["frame_id"];raw=c01.image(r["source"]["raw"]["path"])
        reference=c01.image(c01.verify(pr["clean"]))
        write,protect,fair,guard=visible_card_guard(raw,reference,mask(r["refined"]["write"]),mask(r["source"]["protect"]),mask(r["source"]["visible_card"]))
        changes={}
        for kind,array in (("write",write.astype(np.uint8)*255),("protect",protect.astype(np.uint8)*255),("fair_reference",fair)):
            path=input_dir/f"{fid:06d}_{kind}.png";c01.save(path,array);changes[kind]=c01.ref(path)
        r["candidate3"]={**changes,"guard":guard,"previous_clean":pr["clean"]}
    c01.write_json(lane/FROZEN,{"schema_version":"C02_CONTEXT_MASK_FIX3_FROZEN_V1","task_id":c01.TASK,
      "prior":prior,"predecessor_result":ledger["history"][-1],"candidate_budget":ledger,"candidate_ordinal":3,"rows":data["rows"],
      "code":refs,"original":data["c01_frozen"],"weight":original["weight"],"runtime":original["runtime"],
      "change":"Model inference512 mask adds existing hand support. Explicit visible-card output guard applies identically to new output and separately saved predecessor reference; old result unchanged.",
      "output_guard":"write3=write2&~visible_card; protect3=protect2|visible_card; unknown unchanged; no hidden geometry",
      "hypothesis":"Remaining originally-hand-supported known context may seed skin-shaped hallucination; much propagated support is already background, so improvement is not assumed.",
      "fixed_failure_frames":[76,84,91],"normal_window_regression_frames":[77,82,87],
      "geometry_input_allowed":False,"source_truth_claim":False,"full171_allowed":False})
    return {"frozen":str(lane/FROZEN)}

def run(repo,lane):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU_ONLY")
    cv2.setNumThreads(2);f=c01.read(lane/FROZEN)
    budget(repo,lane)
    for r in f["code"]:c01.verify(r)
    for k in ("prior","predecessor_result","original","weight","runtime"):c01.verify(f[k])
    data={"rows":f["rows"]}
    if [r["frame_id"] for r in data["rows"]]!=list(range(76,92)):raise RuntimeError("FRAME_MAPPING")
    for r in data["rows"]:
        for v in [*r["source"].values(),*r["refined"].values()]:c01.verify(v)
        for k in ("write","protect","fair_reference","previous_clean"):c01.verify(r["candidate3"][k])
    runtime=repo/c01.OLD/"environments/onnxruntime-1.22.1";sys.path.insert(0,str(runtime))
    import onnxruntime as ort
    if ort.__version__!="1.22.1" or not Path(ort.__file__).resolve().is_relative_to(runtime.resolve()):raise RuntimeError("ORT_ORIGIN")
    opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.inter_op_num_threads=1;opt.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    core=ort.InferenceSession(str(c01.verify(f["weight"])),sess_options=opt,providers=["CPUExecutionProvider"])
    ins={x.name:x for x in core.get_inputs()};outs=core.get_outputs()
    if set(ins)!={"image","mask"} or len(outs)!=1:raise RuntimeError("ONNX_NAMES")
    for n,ch in ((ins["image"],3),(ins["mask"],1),(outs[0],3)):
        if n.type!="tensor(float)" or not c01._lama_shape_matches(n.shape,ch):raise RuntimeError("ONNX_CONTRACT")
    dest=lane/OUT;dest.mkdir()
    c01.write_json(dest/"STARTED.json",{"candidate_ordinal":3,"pid":os.getpid(),"budget":f["candidate_budget"],"frozen":c01.ref(lane/FROZEN)})
    for sub in ("clean","internal","source_map","conditioning_mask"):(dest/sub).mkdir()
    started=time.monotonic();rows=[]
    for r in data["rows"]:
        fid=r["frame_id"];s=r["source"];t=r["refined"]
        raw=c01.image(s["raw"]["path"]);prop=c01.image(s["propagated_frame"]["path"]);res=mask(t["residual"]).astype(np.uint8)*255
        hand=mask(s["hand"]);write=mask(r["candidate3"]["write"]);protect=mask(r["candidate3"]["protect"]);unknown=mask(t["unknown"]);card=mask(s["visible_card"])
        wrapper=ContextSession(core,res,hand)
        output,internal,labels,metrics=consumer.consume(wrapper,outs[0].name,prop,res,raw,write,protect,unknown)
        if np.any(internal[res==0]!=prop[res==0]):raise RuntimeError("RESIDUAL_ESCAPE")
        if np.any(output[card]!=raw[card]):raise RuntimeError("VISIBLE_CARD_CHANGED")
        for sub,array in (("clean",output),("internal",internal),("source_map",labels),("conditioning_mask",(wrapper.support[0,0]*255).astype(np.uint8))):
            c01.save(dest/sub/f"{fid:06d}.png",array)
        rows.append({"frame_id":fid,"clean":c01.ref(dest/"clean"/f"{fid:06d}.png"),"source_map":c01.ref(dest/"source_map"/f"{fid:06d}.png"),
          "conditioning_mask":c01.ref(dest/"conditioning_mask"/f"{fid:06d}.png"),"probe":wrapper.probe,**metrics,"visible_card_changed":0,"outside_residual_changed":0,
          "fair_reference":r["candidate3"]["fair_reference"],"output_guard":r["candidate3"]["guard"]})
    result={"schema_version":"C02_CONTEXT_MASK_FIX3_RESULT_V1","candidate_ordinal":3,"candidate_budget":f["candidate_budget"],
      "task_id":c01.TASK,"frozen":c01.ref(lane/FROZEN),"rows":rows,"elapsed_seconds":time.monotonic()-started,
      "execution":"REAL_CPU_LAMA_16_FRAME_REPLAY","gpu_used":False,"providers":core.get_providers(),
      "quality":"PENDING_FIXED_FAILURE_AND_NORMAL_WINDOW_REVIEW","human_review":"NOT_PERFORMED","adoption":"NOT_ADOPTED",
      "pixel_labels":consumer.LABELS,"geometry_input_allowed":False,"training_input_allowed":False,
      "claim_limit":"Generated pixels are offline visual hypothesis only; visible object bytes preserved, hidden card content not recovered truth.",
      "next_action":"Review fixed76/84/91 and77/82/87 plus all16; on failure root-cause review, no fourth candidate."}
    c01.write_json(dest/"RESULT.json",result);return {"result":str(dest/"RESULT.json"),"frames":16}

def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["freeze","run"]);p.add_argument("--repo-root",required=True);p.add_argument("--attempt",default="attempt_0001")
    a=p.parse_args();repo,lane=c01.safe_context(consumer.canonical_repo(a.repo_root),a.attempt)
    print(json.dumps((freeze if a.stage=="freeze" else run)(repo,lane)));return 0
if __name__=="__main__":raise SystemExit(main())
