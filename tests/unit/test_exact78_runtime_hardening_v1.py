"""CPU-only coverage for producer seals, cache compatibility and direct fences."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from chaoyang.exact78_baseline import stage_reuse as reuse
from chaoyang.governance.source_snapshot_closure_v1 import local_import_closure


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def session_fixture(root, identity="unit_0902_001", count=2):
    video = root / "raw/video.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"synthetic-rgb-source")
    return SimpleNamespace(session_id=identity, raw_dir=video.parent, video=video,
        frame_count=count, fps=30.0, width=640, height=480, calibration={}, K=np.eye(3),
        c2w=np.repeat(np.eye(4)[None], count, 0), c2w_editor=np.repeat(np.eye(4)[None], count, 0),
        hands=np.zeros((2,count,21,3)), hand_observed=np.ones((2,count),bool),
        hand_source=np.full((2,count),"OBSERVED"), body=np.zeros((count,24,3)),
        body_observed=np.ones(count,bool), gravity=np.array([0,0,-1]))


def sealed_fixture(root, monkeypatch, stage="motion"):
    sid="unit_0902_001"; task="exact78_takeover_unit_seal"
    attempt=root/"_run/current"/task/"attempts/attempt_0001"
    folder=attempt/"candidate/sessions"/sid
    folder.mkdir(parents=True)
    session=session_fixture(root, sid)
    for name in reuse.FILES[stage]:
        (folder/name).write_bytes(b"synthetic-output")
    if stage=="motion":
        np.savez(folder/"motion.npz",q_arm=np.zeros((2,2,7)),q_hand=np.zeros((2,2,22)))
    write(folder/reuse.FILES[stage][-1],{"session":sid,"status":"OK","frame_count":2})
    code=root/"code.py";code.write_text("# frozen code\n")
    monkeypatch.setattr(reuse,"dependencies",lambda root,stage:[reuse.ref(code)])
    packet_path=root/"tasks/current"/task/"TASK_PACKET.json"
    write(packet_path,{"task_id":task,"run_signature":"producer-signature"})
    producer={"task_id":task,"task_packet":reuse.ref(packet_path),"run_signature":"producer-signature",
              "result_path":str(attempt/"RESULT.json")}
    monkeypatch.setattr(reuse,"current_producer",lambda root:producer)
    expected=reuse.binding(root,stage,session)
    receipt=reuse.seal(root,folder,stage,session,before=expected)
    index=attempt/"candidate/INDEX.json"
    write(index,{"sessions":[{"session":sid,"status":"OK","frames":2,"stage_receipts":{stage:receipt}}]})
    result=attempt/"RESULT.json"
    write(result,{"status":"PASSED","task_packet":producer["task_packet"],
                  "required_outputs":{"candidate/INDEX.json":reuse.ref(index)}})
    ledger=root/"docs/governance/LONG_HORIZON_TASK_STATE.json"
    write(ledger,{"tasks":[{"task_id":task,"status":"PASSED","result":reuse.ref(result)}]})
    return folder,expected,session,ledger


@pytest.mark.parametrize("stage",["motion","gpu","render"])
def test_verified_sealed_producer_can_be_reused(tmp_path,monkeypatch,stage):
    folder,expected,_,_=sealed_fixture(tmp_path,monkeypatch,stage)
    assert reuse.validate(folder,stage,expected,tmp_path)


def test_legacy_ok_wrong_identity_is_not_a_seal(tmp_path,monkeypatch):
    from chaoyang.exact78_baseline.pipeline import Layout
    folder=tmp_path/"old/sessions/expected_session";folder.mkdir(parents=True)
    (folder/"motion.npz").write_bytes(b"arbitrary bytes")
    write(folder/"motion.json",{"status":"OK","session":"WRONG","frame_count":999})
    from chaoyang.exact78_baseline import pico_source
    monkeypatch.setattr(pico_source,"load_session",lambda *_:pytest.fail("Legacy files must not cause input/model loading"))
    assert Layout(tmp_path/"new",[tmp_path/"old"],repo_root=tmp_path).find("expected_session","motion") is None


@pytest.mark.parametrize("change",["input","dependencies","config","upstream"])
def test_intact_different_signature_is_cache_miss_not_consumer_input(tmp_path,monkeypatch,change):
    folder,expected,_,_=sealed_fixture(tmp_path,monkeypatch)
    expected=copy.deepcopy(expected)
    if change=="dependencies":
        expected[change].append({"different":"code"})
    else:
        expected[change]["changed"]=True
    assert reuse.validate(folder,"motion",expected,tmp_path) is False


@pytest.mark.parametrize("field,value",[("session","WRONG"),("frame_count",999),("stage","gpu")])
def test_claimed_seal_wrong_identity_or_count_is_rejected(tmp_path,monkeypatch,field,value):
    folder,expected,_,_=sealed_fixture(tmp_path,monkeypatch)
    path=reuse.receipt_path(folder,"motion");data=json.loads(path.read_text());data[field]=value;write(path,data)
    with pytest.raises(reuse.StageReuseDenied,match="IDENTITY_OR_STAGE"):
        reuse.validate(folder,"motion",expected,tmp_path)


def test_tampered_binding_does_not_become_an_admission(tmp_path,monkeypatch):
    folder,expected,_,_=sealed_fixture(tmp_path,monkeypatch)
    path=reuse.receipt_path(folder,"motion");data=json.loads(path.read_text());data["binding"]["config"]["width"]=1;write(path,data)
    with pytest.raises(reuse.StageReuseDenied,match="SIGNATURE_TAMPERED"):
        reuse.validate(folder,"motion",expected,tmp_path)


def test_changed_array_sha_is_rejected(tmp_path,monkeypatch):
    folder,expected,_,_=sealed_fixture(tmp_path,monkeypatch)
    (folder/"motion.npz").write_bytes(b"different array bytes")
    with pytest.raises(reuse.StageReuseDenied,match="SHA_OR_SIZE_DRIFT"):
        reuse.validate(folder,"motion",expected,tmp_path)


@pytest.mark.parametrize("state",["RUNNING","FAILED_RUNTIME_FINAL","PENDING"])
def test_partial_or_unsealed_producer_not_consumed(tmp_path,monkeypatch,state):
    folder,expected,_,ledger=sealed_fixture(tmp_path,monkeypatch)
    data=json.loads(ledger.read_text());data["tasks"][0]["status"]=state;write(ledger,data)
    with pytest.raises(reuse.StageReuseDenied,match="NOT_SEALED_PASSED"):
        reuse.validate(folder,"motion",expected,tmp_path)


def test_index_must_bind_exact_receipt_sha(tmp_path,monkeypatch):
    folder,expected,_,ledger=sealed_fixture(tmp_path,monkeypatch)
    path=reuse.receipt_path(folder,"motion");data=json.loads(path.read_text());data["claim_limit"]="tampered";write(path,data)
    with pytest.raises(reuse.StageReuseDenied,match="INDEX_DOES_NOT_BIND"):
        reuse.validate(folder,"motion",expected,tmp_path)


def test_pre_execution_binding_is_required_and_must_remain_equal(tmp_path,monkeypatch):
    folder,expected,session,_=sealed_fixture(tmp_path,monkeypatch)
    session.hands[0,0,0,0]=1
    with pytest.raises(reuse.StageReuseDenied,match="CHANGED_DURING_EXECUTION"):
        reuse.seal(tmp_path,folder,"motion",session,before=expected)


def test_source_fingerprint_detects_motion_change_without_video_change(tmp_path):
    session=session_fixture(tmp_path);before=reuse.input_fingerprint(session)
    session.c2w[0,0,3]=0.5
    assert before!=reuse.input_fingerprint(session)


def test_fingerprint_does_not_alias_mutable_calibration(tmp_path):
    session=session_fixture(tmp_path);session.calibration={"camera":{"mode":"left"}}
    before=reuse.input_fingerprint(session)
    session.calibration["camera"]["mode"]="right"
    assert before["calibration"]["camera"]["mode"]=="left"
    assert before!=reuse.input_fingerprint(session)


def test_stage_output_must_be_inside_producer_attempt(tmp_path,monkeypatch):
    _,expected,session,_=sealed_fixture(tmp_path,monkeypatch)
    with pytest.raises(reuse.StageReuseDenied,match="NOT_OWNED_BY_PRODUCER"):
        reuse.seal(tmp_path,tmp_path/"unowned","motion",session,before=expected)


def test_current_producer_requires_the_actual_frozen_writer_gate(tmp_path):
    import importlib.util
    path=Path(__file__).with_name("test_exact78_operation_gate.py")
    spec=importlib.util.spec_from_file_location("gate_fixture_for_stage_seal",path)
    fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
    paths=fixtures._fixture(tmp_path)
    assert reuse.current_producer(tmp_path)["task_id"]==fixtures.TASK_ID
    writer=json.loads(paths["writer"].read_text());writer["fencing_token"]="wrong-fence"
    write(paths["writer"],writer)
    with pytest.raises(fixtures.Exact78GateDenied,match="fencing identity"):
        reuse.current_producer(tmp_path)


def test_asset_bytes_are_verified_even_when_pin_document_is_unchanged(tmp_path,monkeypatch):
    import chaoyang.governance.source_snapshot_closure_v1 as closure
    monkeypatch.setattr(closure,"local_import_closure",lambda *_:[])
    names=("assets/robot/ROBOT_ASSET_PIN.json",
           "assets/robot/hardware_handoff/kaihand_flange_adapter_v1/excluded_reference_only/MOUNT_VISUAL_PROXY_CONTRACT.json",
           "assets/robot/V2相机固定法兰PRO版本.STL","assets/robot/part.STL")
    for name in names:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"asset")
    asset=tmp_path/names[-1];row=reuse.ref(asset);row["path"]=names[-1]
    write(tmp_path/names[0],{"files":[row]})
    assert str(asset) in {ref["path"] for ref in reuse.dependencies(tmp_path,"motion")}
    asset.write_bytes(b"drift")
    with pytest.raises(reuse.StageReuseDenied,match="SHA_OR_SIZE_DRIFT"):
        reuse.dependencies(tmp_path,"motion")


def test_source_closure_includes_function_local_and_relative_dependencies(tmp_path):
    pkg=tmp_path/"src/chaoyang";pkg.mkdir(parents=True)
    (pkg/"__init__.py").write_text("")
    (pkg/"entry.py").write_text("def main():\n from . import retirement\n")
    (pkg/"retirement.py").write_text("# required\n")
    found=local_import_closure(tmp_path,[pkg/"entry.py"])
    assert pkg/"retirement.py" in found


def test_snapshot_covers_all_three_dynamic_takeover_operation_roots():
    from chaoyang.governance.project_branch_navigation_v1 import snapshot_paths
    root=Path(__file__).resolve().parents[2]
    if not (root/"tasks/current/exact78_takeover_b2_full_v13/TASK_PACKET.json").exists():
        pytest.skip("Server-only snapshot inputs; isolated restore tests bind published roots")
    paths=set(snapshot_paths(root))
    for operation in ("baseline","audit","cleanup"):
        assert root/("src/chaoyang/ops/run_exact78_takeover_"+operation+"_v1.py") in paths
    assert root/"src/chaoyang/governance/operation_retirement.py" in paths


@pytest.mark.parametrize("operation",["baseline","audit","cleanup"])
def test_direct_function_is_fenced_before_parser_or_writes(tmp_path,operation):
    from chaoyang.governance.exact78_operation_gate import Exact78GateDenied
    module=importlib.import_module("chaoyang.ops.run_exact78_takeover_"+operation+"_v1")
    output=tmp_path/"must_not_exist"
    with pytest.raises(Exact78GateDenied):
        module.main(["--output-root",str(output)])
    assert not output.exists()


@pytest.mark.parametrize("operation",["baseline","audit","cleanup"])
def test_direct_python_module_is_fenced_before_writes(tmp_path,operation):
    root=Path(__file__).resolve().parents[2];output=tmp_path/"must_not_exist"
    result=subprocess.run([sys.executable,"-B","-m","chaoyang.ops.run_exact78_takeover_"+operation+"_v1",
        "--output-root",str(output)],cwd=root,capture_output=True,text=True,
        env={**os.environ,"PYTHONPATH":str(root/"src"),"PYTHONDONTWRITEBYTECODE":"1"},timeout=20)
    assert result.returncode!=0 and "EXACT78_GATE_DENIED" in result.stderr
    assert not output.exists()


def test_pipeline_cold_scaffold_then_sealed_three_stage_reuse(tmp_path,monkeypatch):
    """Exercise the real scheduler/index publisher with fake compute, never models."""
    from concurrent.futures import Future
    from chaoyang.exact78_baseline import pipeline, pico_source, mask, clean
    sid="unit_0902_001";task="exact78_takeover_scheduler_test"
    attempt=tmp_path/"_run/current"/task/"attempts/attempt_0001"
    output=attempt/"candidate";output.mkdir(parents=True)
    session=session_fixture(tmp_path,sid)
    code=tmp_path/"code.py";code.write_text("# scheduler test\n")
    monkeypatch.setattr(reuse,"dependencies",lambda root,stage:[reuse.ref(code)])
    monkeypatch.setattr(pico_source,"load_session",lambda identity:session)
    packet=tmp_path/"tasks/current"/task/"TASK_PACKET.json"
    write(packet,{"task_id":task,"run_signature":"scheduler-signature"})
    producer={"task_id":task,"task_packet":reuse.ref(packet),"run_signature":"scheduler-signature",
              "result_path":str(attempt/"RESULT.json")}
    monkeypatch.setattr(reuse,"current_producer",lambda root:producer)

    class InlinePool:
        def __init__(self,**kwargs):pass
        def submit(self,fn,*args):
            future=Future()
            try:future.set_result(fn(*args))
            except Exception as exc:future.set_exception(exc)
            return future
        def shutdown(self,wait=True):pass
    class FakeSam:
        def __init__(self,*args):pass
        def close(self):pass
    class FakePainter:
        def __init__(self,*args):pass
    monkeypatch.setattr(pipeline,"ProcessPoolExecutor",InlinePool)
    monkeypatch.setattr(mask,"Sam3Video",FakeSam)
    monkeypatch.setattr(clean,"ProPainter",FakePainter)
    calls=[]
    def stage(stage,folder,upstream=None):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);calls.append(stage)
        before=reuse.binding(tmp_path,stage,session,upstream=upstream)
        for name in reuse.FILES[stage]:(folder/name).write_bytes(b"synthetic-output")
        if stage=="motion":np.savez(folder/"motion.npz",q_arm=np.zeros((2,2,7)),q_hand=np.zeros((2,2,22)))
        info={"session":sid,"frame_count":2,"status":"OK"}
        if stage=="render":
            info.update(motion={"metrics":{"per_side":{side:{"tracking_pass_rate_observed":1.0} for side in ("left","right")}}},
                        render={"removal_uncovered_fraction":0.0})
        write(folder/reuse.FILES[stage][-1],info)
        reuse.seal(tmp_path,folder,stage,session,before=before,upstream=upstream)
        return {"session":sid,"stage":stage}
    monkeypatch.setattr(pipeline,"motion_stage",lambda repo,sid,folder:stage("motion",folder))
    monkeypatch.setattr(pipeline,"gpu_stage",lambda models,sid,folder:stage("gpu",folder))
    monkeypatch.setattr(pipeline,"render_stage",lambda repo,sid,folder,motion,gpu:stage("render",folder,
        {"motion":Path(motion)/"motion.npz","masks":Path(gpu)/"masks.npz","clean":Path(gpu)/"clean.mp4"}))
    result=pipeline.run(tmp_path,output,[sid],workers=1,reuse=[],objects=True,propainter_settings=None)
    assert result["counts"]=={"OK":1} and calls==["motion","gpu","render"]
    assert set(result["sessions"][0]["stage_receipts"])=={"motion","gpu","render"}
    write(attempt/"RESULT.json",{"status":"PASSED","task_packet":producer["task_packet"],
        "required_outputs":{"candidate/INDEX.json":reuse.ref(output/"INDEX.json")}})
    write(tmp_path/"docs/governance/LONG_HORIZON_TASK_STATE.json",{"tasks":[{"task_id":task,"status":"PASSED",
        "result":reuse.ref(attempt/"RESULT.json")}]})
    calls.clear()
    result=pipeline.run(tmp_path,tmp_path/"new_candidate",[sid],workers=1,reuse=[output],
        reuse_stages=("motion","gpu","render"),objects=True,propainter_settings=None)
    assert result["counts"]=={"OK":1} and calls==[]
