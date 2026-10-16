from pathlib import Path
import pytest
from chaoyang.pick_place_1008 import projection_check as p

def test_projection_rejects_unsealed_fit(monkeypatch,tmp_path):
    config=tmp_path/"candidate"/"FITTED_CAMERA_CONFIG.json"
    monkeypatch.setattr(p.source,"ref",lambda path:{"path":str(path),"bytes":1,"sha256":"a"})
    def read(path):
        if Path(path).name=="manifest.json":return {"sessions":[]}
        if Path(path)==config:return {"fit_quality_status":"PASSED"}
        return {"status":"RUNNING","required_outputs":{}}
    monkeypatch.setattr(p.source,"read_json",read)
    with pytest.raises(ValueError,match="SEALED_PASSED"):
        p.run_projection_check(tmp_path/"manifest.json",config,tmp_path/"out")

def test_projection_rejects_failed_quality_even_with_execution_pin(monkeypatch,tmp_path):
    config=tmp_path/"candidate"/"FITTED_CAMERA_CONFIG.json"
    reference={"path":str(config),"bytes":1,"sha256":"a"}
    monkeypatch.setattr(p.source,"ref",lambda path:reference)
    def read(path):
        path=Path(path)
        if path.name=="manifest.json":return {"sessions":[]}
        if path==config:return {"fit_quality_status":"REVIEW_REQUIRED"}
        if path.parent==config.parent:return {"status":"PASSED","fitted_camera_config":reference}
        return {"status":"PASSED","required_outputs":{"candidate/FITTED_CAMERA_CONFIG.json":reference}}
    monkeypatch.setattr(p.source,"read_json",read)
    with pytest.raises(ValueError,match="PASSED_VISUAL_FIT"):
        p.run_projection_check(tmp_path/"manifest.json",config,tmp_path/"out")

def test_projection_rejects_arbitrary_session_group(tmp_path):
    with pytest.raises(ValueError,match="FIXED_PROJECTION"):
        p.run_projection_check(tmp_path/"m",tmp_path/"c",tmp_path/"out",group="training")
