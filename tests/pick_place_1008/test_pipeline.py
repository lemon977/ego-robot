from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from chaoyang.pick_place_1008 import pipeline

def test_qpc_limits_use_actual_unequal_intervals(monkeypatch,tmp_path):
    n=4;qpc=np.array([1000000000,1010000000,1110000000,1210000000],np.int64)
    session=SimpleNamespace(frame_count=n,qpc_ns=qpc,hand_observed=np.ones((2,n),bool),
        hand_source=np.full((2,n),"RAW_COMPOSED"),motion_valid={"complete_valid":np.ones(n,bool)},
        c2w=np.tile(np.eye(4),(n,1,1)),calibration={},session_id="pick_place_1008_001",
        mapping={"video_qpc_ns":qpc,"frame_id":np.arange(n)})
    qa=np.zeros((n,2,7));qa[:,0,0]=[0,.1,.11,.12]
    result=SimpleNamespace(q_arm=qa,q_hand=np.zeros((n,2,22)),T_world_base=np.eye(4),
        root_world=np.zeros((n,2,3)),arm_position_mm=np.zeros((2,n)),arm_rotation_deg=np.zeros((2,n)),
        hand_bone_deg=np.zeros((2,n)),metrics={},base_report={})
    from chaoyang.exact78_baseline import retarget,robot_model
    monkeypatch.setattr(pipeline.source,"load_prepared",lambda _:session)
    monkeypatch.setattr(retarget,"retarget_session",lambda *_:result)
    monkeypatch.setattr(robot_model,"Robot",lambda _:SimpleNamespace(
        arms=[SimpleNamespace(joint_names=[f"a{j}" for j in range(7)]) for _ in range(2)],
        hands=[SimpleNamespace(names=[f"h{j}" for j in range(22)]) for _ in range(2)]))
    out=tmp_path/"motion";out.mkdir()
    value=pipeline.motion_worker(str(tmp_path),str(out),str(tmp_path/"source"))
    assert value["qpc_diagnostics"]["velocity_violation_raw_frame_indices"]==[1]
    assert value["qpc_diagnostics"]["acceleration_violation_raw_frame_indices"]==[2]
    assert value["action_contract"]["angle_unit"]=="radian"
    arrays=pipeline.load_arrays(out/"actions.npz")
    assert arrays["qpc_velocity_violation"][:,0].tolist()==[False,True,False,False]
    assert np.array_equal(arrays["q_arm"],qa)
    mapping=[json.loads(v) for v in (out/"frame_mapping.jsonl").read_text().splitlines()]
    assert [v["video_qpc_ns"] for v in mapping]==qpc.tolist()

def test_bad_source_reuse_does_not_cancel_other_session(monkeypatch,tmp_path):
    manifest=tmp_path/"manifest.json";config=tmp_path/"config.json"
    rows=[{"session_id":s,"raw_frame_count":3,"source_refs":{},"input_status":"READY_RAW_AUDIT"} for s in ["001","002"]]
    manifest.write_text(json.dumps({"source_root":str(tmp_path),"sessions":rows}));config.write_text("{}")
    monkeypatch.setattr(pipeline,"binding",lambda *args:{"sid":args[2]["session_id"]})
    def lookup(repo,roots,sid,stage,expected):
        if sid=="001":raise RuntimeError("TAMPERED_STAGE")
        return None,None
    monkeypatch.setattr(pipeline,"find_reuse",lookup)
    monkeypatch.setattr(pipeline.source,"prepare_session",lambda *args:None)
    def seal(repo,folder,stage,expected):
        p=folder/"fake_receipt.json";p.write_text("{}");return pipeline.ref(p)
    monkeypatch.setattr(pipeline,"seal",seal)
    result=pipeline.run(tmp_path,tmp_path/"out",manifest,config,["001","002"],"prepare")
    assert result["status"]=="INCOMPLETE"
    assert result["sessions_requested"]==2 and result["sessions_complete"]==1
    assert result["sessions"][0]["status"]=="FAILED" and result["sessions"][1]["status"]=="COMPLETE"
    assert pipeline.read(tmp_path/"out/INDEX.json")["sessions"][1]["stage_receipts"]["source"]
