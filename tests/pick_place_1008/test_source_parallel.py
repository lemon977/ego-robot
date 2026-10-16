import json
from concurrent.futures import Future
from pathlib import Path
import pytest
from chaoyang.pick_place_1008 import pipeline

@pytest.mark.parametrize("workers",[0,3,5,100])
def test_unbounded_source_workers_rejected_without_output(tmp_path,workers):
    with pytest.raises(ValueError,match="BOUNDED_SOURCE_WORKER_COUNT"):
        pipeline.run(tmp_path,tmp_path/"out",tmp_path/"missing",tmp_path/"missing",[],"prepare",source_workers=workers)
    assert not (tmp_path/"out").exists()

@pytest.mark.parametrize("stage,budget",[("full",{"workers_max":4,"cpu_threads_total":8}),("prepare",{"workers_max":2,"cpu_threads_total":8}),("prepare",{"workers_max":4,"cpu_threads_total":6})])
def test_source_pool_cannot_exceed_registered_core_budget(monkeypatch,tmp_path,stage,budget):
    packet=tmp_path/"packet.json";packet.write_text(json.dumps({"budgets":budget}))
    pin=pipeline.ref(packet);monkeypatch.setattr(pipeline,"current_producer",lambda _:{"task_packet":pin})
    with pytest.raises(ValueError,match="SOURCE_WORKERS_EXCEED_FENCED_CPU_BUDGET"):
        pipeline.run(tmp_path,tmp_path/"out",tmp_path/"missing",tmp_path/"missing",[],stage,source_workers=4)
    assert not (tmp_path/"out").exists()

def test_parallel_source_failure_keeps_other_native_session_and_parent_seal(monkeypatch,tmp_path):
    manifest=tmp_path/"manifest.json";config=tmp_path/"config.json"
    rows=[{"session_id":s,"raw_frame_count":n,"source_refs":{},"input_status":"READY_RAW_AUDIT"} for s,n in [("001",3),("002",5)]]
    manifest.write_text(json.dumps({"source_root":str(tmp_path/"raw"),"sessions":rows}));config.write_text("{}")
    packet=tmp_path/"packet.json";packet.write_text(json.dumps({"budgets":{"workers_max":2,"cpu_threads_total":4}}))
    monkeypatch.setattr(pipeline,"current_producer",lambda _:{"task_packet":pipeline.ref(packet)})
    monkeypatch.setattr(pipeline,"binding",lambda *a,**k:{"sid":a[2]["session_id"]})
    monkeypatch.setattr(pipeline,"find_reuse",lambda *a:(None,None))
    events=[]
    def prepare(raw,folder,configuration,count):
        sid=Path(raw).name;events.append(("prepare",sid,count))
        if sid=="001":raise RuntimeError("SDK_TOPOLOGY_DRIFT")
        assert count==5 and configuration=={}
        return True
    monkeypatch.setattr(pipeline,"prepare_source_worker",prepare)
    def seal(repo,folder,stage,expected):
        assert expected["sid"]=="002" and stage=="source"
        events.append(("seal","002"))
        receipt=folder/"fake_source_receipt.json";receipt.write_text("{}");return pipeline.ref(receipt)
    monkeypatch.setattr(pipeline,"seal",seal)
    class CpuPool:
        def __init__(self,max_workers,initializer,mp_context):
            assert max_workers==2 and mp_context.get_start_method()=="spawn"
        def submit(self,call,*args):
            future=Future()
            try:future.set_result(call(*args))
            except BaseException as e:future.set_exception(e)
            return future
        def shutdown(self,wait):
            assert wait is True;events.append(("pool_shutdown",))
    monkeypatch.setattr(pipeline,"ProcessPoolExecutor",CpuPool)
    result=pipeline.run(tmp_path,tmp_path/"out",manifest,config,["001","002"],"prepare",source_workers=2)
    assert result["status"]=="INCOMPLETE" and result["sessions_complete"]==1 and result["frames_complete"]==5
    assert result["sessions"][0]["failure_stage"]=="source" and "SDK_TOPOLOGY_DRIFT" in result["sessions"][0]["error"]
    assert result["sessions"][1]["stages"]["source"]["receipt"]
    assert [e for e in events if e[0]=="seal"]==[("seal","002")]
    assert events[-1]==("pool_shutdown",)
    assert result["resource_recipe"]["source_workers"]==2 and result["resource_recipe"]["cpu_pools_overlap"] is False
