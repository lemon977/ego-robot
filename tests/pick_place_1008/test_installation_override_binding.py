import copy
import json
import pytest
from chaoyang.pick_place_1008 import source

def case(tmp_path):
    prior={"schema":"controller_to_wrist_v1","convention":"wrist_pose = compose_pose(controller_pose, transform)","controller_pose_frame":"pico_world_rh_x_forward_y_left_z_up",
           "left":{"pos":[0,0,0],"quat":[0,0,0,1]},"right":{"pos":[0,0,0],"quat":[0,0,0,1]}}
    serial="DEVICE_A";sha="original";key=serial+":"+sha
    report={"schema":"pick_place_installation_group_visual_fit_v1","groups":{key:{"status":"PASSED","device_serial":serial,"original_prior_sha256":sha,"prior_used":prior}}}
    path=tmp_path/"INSTALLATION_REPORT.json";path.write_text(json.dumps(report))
    cfg={"installation_original_prior_sha256":sha,"installation_group_key":key,"fit_quality_status":"PASSED","installation_prior_override":prior,"installation_correction_evidence":source.ref(path)}
    return cfg,{"prior_sha256":sha},serial,path

def test_valid_installation_override_exact_group(tmp_path):
    cfg,original,serial,_=case(tmp_path)
    assert source.validated_installation_override(cfg,original,serial)==cfg["installation_prior_override"]

@pytest.mark.parametrize("change",[{"installation_group_key":"DEVICE_B:original"},{"installation_original_prior_sha256":"other"},{"fit_quality_status":"REVIEW_REQUIRED"}])
def test_installation_override_rejects_wrong_original_group_or_quality(tmp_path,change):
    cfg,original,serial,_=case(tmp_path);cfg.update(change)
    with pytest.raises(source.SourceContractError,match="GROUP_OR_QUALITY"):
        source.validated_installation_override(cfg,original,serial)

def test_installation_override_rejects_reference_drift(tmp_path):
    cfg,original,serial,path=case(tmp_path);path.write_text(path.read_text()+" ")
    with pytest.raises(Exception,match="REFERENCE_DRIFT"):
        source.validated_installation_override(cfg,original,serial)

def test_installation_override_rejects_other_values_than_report(tmp_path):
    cfg,original,serial,_=case(tmp_path);cfg["installation_prior_override"]=copy.deepcopy(cfg["installation_prior_override"]);cfg["installation_prior_override"]["left"]["pos"]=[.1,0,0]
    with pytest.raises(source.SourceContractError,match="EVIDENCE_GROUP"):
        source.validated_installation_override(cfg,original,serial)
