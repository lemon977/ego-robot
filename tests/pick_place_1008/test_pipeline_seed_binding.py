import json
from pathlib import Path
import pytest
from chaoyang.pick_place_1008 import pipeline

def test_different_visible_seed_cannot_reuse_gpu_but_keeps_raw_motion_cache(monkeypatch,tmp_path):
    monkeypatch.setattr(pipeline,"stage_dependencies",lambda *args:[])
    row={"session_id":"001","source_refs":{"video":{"sha256":"v"}},"input_status":"READY_RAW_AUDIT","raw_frame_count":5}
    first={"path":"/seed/a.json","bytes":1,"sha256":"a"}
    second={"path":"/seed/b.json","bytes":1,"sha256":"b"}
    a=pipeline.binding(tmp_path,"gpu",row,{},{"source":"source"},first)
    b=pipeline.binding(tmp_path,"gpu",row,{},{"source":"source"},second)
    assert pipeline.canonical(a)!=pipeline.canonical(b)
    assert pipeline.binding(tmp_path,"motion",row,{},{"source":"source"},first)==pipeline.binding(tmp_path,"motion",row,{},{"source":"source"},second)

@pytest.mark.parametrize("workers",[0,1,3,5,100])
def test_unbounded_render_count_rejected_before_output(tmp_path,workers):
    with pytest.raises(ValueError,match="BOUNDED_RENDER_WORKER_COUNT"):
        pipeline.run(tmp_path,tmp_path/"out",tmp_path/"no_manifest",tmp_path/"no_config",[],"full",render_workers=workers)
    assert not (tmp_path/"out").exists()

@pytest.mark.parametrize("budget",[{"workers_max":2,"cpu_threads_total":10},{"workers_max":4,"cpu_threads_total":8}])
def test_four_render_workers_cannot_exceed_registered_cpu_budget(monkeypatch,tmp_path,budget):
    packet=tmp_path/"packet.json";packet.write_text(json.dumps({"budgets":budget}))
    pin=pipeline.ref(packet)
    monkeypatch.setattr(pipeline,"current_producer",lambda _:{"task_packet":pin})
    with pytest.raises(ValueError,match="FOUR_RENDER_WORKERS_REQUIRE_FENCED_CPU_BUDGET_10"):
        pipeline.run(tmp_path,tmp_path/"out",tmp_path/"no_manifest",tmp_path/"no_config",[],"full",render_workers=4)
    assert not (tmp_path/"out").exists()


def test_source_cache_pins_transitive_candidate_and_frozen_parent_helpers(tmp_path):
    own=tmp_path/"src/chaoyang/pick_place_1008";own.mkdir(parents=True)
    for relative in pipeline.STAGE_CODE["source"]:
        file=tmp_path/"src/chaoyang"/relative;file.parent.mkdir(parents=True,exist_ok=True);file.write_text("source dependency")
    helper=own/"audit.py";helper.write_text("transitive helper original")
    parent=tmp_path/"src/chaoyang/exact78_baseline/pico_source.py";parent.parent.mkdir(parents=True);parent.write_text("frozen pose helper")
    manifest=tmp_path/"baseline_v1/MANIFEST.json";manifest.parent.mkdir()
    manifest.write_text(json.dumps({"sources":[{"role":"ALGORITHM","source":pipeline.ref(parent)}]}))
    pins=pipeline.stage_dependencies(str(tmp_path),"source")
    assert pipeline.ref(helper) in pins and pipeline.ref(parent) in pins
    helper.write_text("transitive helper modified")
    original=next(pin for pin in pins if pin["path"]==str(helper))
    with pytest.raises(Exception):pipeline.verify(original)
    pipeline.stage_dependencies.cache_clear()
    assert pipeline.ref(helper) in pipeline.stage_dependencies(str(tmp_path),"source")
