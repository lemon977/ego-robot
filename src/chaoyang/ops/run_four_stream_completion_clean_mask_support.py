"""C02 candidate 1: correct image/mask resize support; C01 remains immutable."""
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

def support_mask(residual):
    if residual.shape != (720, 960):
        raise ValueError("MASK_DOMAIN")
    binary = (residual > 0).astype(np.float32)
    alpha = cv2.resize(binary, (512, 512), interpolation=cv2.INTER_LINEAR)
    return (alpha > 0).astype(np.float32)[None, None]

class SupportSession:
    def __init__(self, session, residual):
        self.session = session
        self.support = support_mask(residual)
        self.expected_old = cv2.resize((residual > 0).astype(np.uint8), (512,512),
                                      interpolation=cv2.INTER_NEAREST).astype(np.float32)[None,None]
        self.probe = None
    def run(self, outputs, inputs):
        if not np.array_equal(inputs["mask"], self.expected_old):
            raise RuntimeError("UNEXPECTED_CALLER_MASK")
        actual = dict(inputs)
        actual["mask"] = self.support
        prediction = self.session.run(outputs, actual)
        arr = prediction[0]
        self.probe = {"model_called": True, "input_range": [float(actual["image"].min()),float(actual["image"].max())],
                      "output_range": [float(arr.min()),float(arr.max())], "output_p50": float(np.median(arr)),
                      "old_model_mask_px": int(self.expected_old.sum()), "new_model_mask_px": int(self.support.sum()),
                      "added_mask_px": int(((self.support > 0)&(self.expected_old == 0)).sum()),
                      "removed_mask_px": int(((self.support == 0)&(self.expected_old > 0)).sum()),
                      "gray_source_contribution_unmasked_px": 0}
        return prediction

def freeze(repo, lane):
    path = lane / "C02_FROZEN_INPUT.json"
    c01.write_json(path, {"schema_version":"C02_MASK_SUPPORT_FROZEN_V1", "task_id":c01.TASK,
                         "c01_frozen":c01.ref(lane/"C01_FROZEN_INPUT.json"),
                         "c01_result":c01.ref(lane/"C01_ADAPTER_FIX1/RESULT.json"),
                         "edge_evidence":c01.ref(lane/"C02_ATTRIBUTION/RESIZE_EDGE_PROBE.json"),
                         "code":c01.ref(Path(__file__)), "base_code":c01.ref(Path(c01.__file__)),
                         "change":"ONLY_512_INFERENCE_MASK_USES_LINEAR_SUPPORT_GT_ZERO",
                         "retained":"source/model/image_resize/color/720_residual_write/1280_write_and_protect",
                         "candidate_ordinal":1, "full171_allowed":False})
    return {"frozen":str(path)}

def run(repo, lane):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU_ONLY_REQUIRES_EMPTY_CUDA_VISIBLE_DEVICES")
    cv2.setNumThreads(2)
    binding_path = lane/"C02_FROZEN_INPUT.json"
    binding=c01.read(binding_path)
    for key in ("c01_frozen","c01_result","edge_evidence","code","base_code"):
        c01.verify(binding[key])
    data=c01.read(lane/"C01_FROZEN_INPUT.json")
    for key in ("weight","runtime","source_manifest","propagation_manifest","predecessor"):
        c01.verify(data[key])
    for item in data["code"]:
        c01.verify(item)
    sys.path.insert(0,str(repo/c01.OLD/"environments/onnxruntime-1.22.1"))
    import onnxruntime as ort
    if ort.__version__!="1.22.1":
        raise RuntimeError("ORT_VERSION_DRIFT")
    options=ort.SessionOptions()
    options.intra_op_num_threads=2
    options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session=ort.InferenceSession(str(repo/c01.WEIGHT),sess_options=options,providers=["CPUExecutionProvider"])
    ins={x.name:x for x in session.get_inputs()}
    outs=session.get_outputs()
    if set(ins)!={"image","mask"} or len(outs)!=1:
        raise RuntimeError("ONNX_NAMES")
    for node,channels in ((ins["image"],3),(ins["mask"],1),(outs[0],3)):
        if node.type!="tensor(float)" or not c01._lama_shape_matches(node.shape,channels):
            raise RuntimeError("ONNX_CONTRACT")
    out=lane/"C02_MASK_SUPPORT_FIX1"
    out.mkdir()
    for folder in ("clean","internal"):
        (out/folder).mkdir()
    started=time.monotonic()
    rows=[]
    for row in data["rows"]:
        for value in row.values():
            if isinstance(value,dict): c01.verify(value)
        fid=row["frame_id"]
        prop=c01.image(row["propagated_frame"]["path"])
        residual=c01.image(row["residual_propagation_mask"]["path"],True)
        wrapper=SupportSession(session,residual)
        internal=c01._apply_lama_residual(wrapper,outs[0].name,prop,residual)
        if np.any(internal[residual==0]!=prop[residual==0]):
            raise RuntimeError("RESIDUAL_WRITE_ESCAPE")
        raw=c01.image(row["raw"]["path"])
        write=c01.image(row["write"]["path"],True)>0
        protect=c01.image(row["protect"]["path"],True)>0
        clean=c01.composite_clean(raw,cv2.resize(internal,(1280,960)),write,protect)
        changed=np.any(clean!=raw,axis=2)
        if np.any(changed&~write) or np.any(changed&protect):
            raise RuntimeError("FINAL_WRITE_ESCAPE")
        c01.save(out/"clean"/f"{fid:06d}.png",clean)
        c01.save(out/"internal"/f"{fid:06d}.png",internal)
        rows.append({"frame_id":fid,"probe":wrapper.probe,"model_called":wrapper.probe is not None,
                     "clean":c01.ref(out/"clean"/f"{fid:06d}.png"),
                     "outside_residual_changed":0,"outside_write_changed":0,"protected_changed":0,
                     "white_write_px":int((np.all(clean>=250,axis=2)&write).sum())})
    result={"schema_version":"C02_MASK_SUPPORT_RESULT_V1","task_id":c01.TASK,
            "execution":"REAL_CPU_LAMA_16_FRAME_REPLAY","structure":"PASS",
            "quality":"PENDING_FIXED_WINDOW_REVIEW","human_review":"NOT_PERFORMED","adoption":"NOT_ADOPTED",
            "frozen":c01.ref(binding_path),"rows":rows,"elapsed_seconds":time.monotonic()-started,
            "gpu_used":False,"providers":session.get_providers(),"candidate_ordinal":1,
            "full171":"NOT_RUN","claim_limit":"Sampler support corrected; semantic quality not certified."}
    c01.write_json(out/"RESULT.json",result)
    return {"result":str(out/"RESULT.json"),"frames":len(rows)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("freeze","run"))
    parser.add_argument("--repo-root",required=True)
    parser.add_argument("--attempt",default="attempt_0001")
    args=parser.parse_args()
    repo,lane=c01.safe_context(args.repo_root,args.attempt)
    print(json.dumps({"freeze":freeze,"run":run}[args.stage](repo,lane)))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
