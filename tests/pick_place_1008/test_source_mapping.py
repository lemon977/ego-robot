import copy
import json
import numpy as np
import pytest
from test_source import raw_fixture, prior, write_rows
from chaoyang.pick_place_1008 import source
from chaoyang.pipeline.pico_manus_motion_v2 import MAP_25_TO_21 as parent_map


def topology_row():
    return {"node_count":25, "node_ids":list(range(25)),
            "parent_ids":list(source.RAW_SDK_PARENT_IDS),
            "chain_types":list(source.RAW_SDK_CHAIN_TYPES),
            "joint_types":list(source.RAW_CAPTURE_JOINT_ALIASES)}


def test_parent_mapping_preserved_and_single_candidate_authority():
    assert parent_map.tolist() == [0,1,2,3,4,5,6,8,9,10,11,13,14,15,16,18,19,20,21,23,24]
    assert source.MAP_25_TO_21 is source.RAW_SDK_TO_CANONICAL21
    assert source.MAP_25_TO_21.tolist() == [0,1,2,3,4,6,7,8,9,11,12,13,14,16,17,18,19,21,22,23,24]
    assert not source.MAP_25_TO_21.flags.writeable
    assert source.CANONICAL_JOINT_NAMES_21[5:9] == ("Index_MCP","Index_PIP","Index_DIP","Index_TIP")
    assert source.CANONICAL_JOINT_NAMES_21[1:5] == ("Thumb_CMC","Thumb_MCP","Thumb_IP","Thumb_TIP")


@pytest.mark.parametrize("style", ["capture_alias", "sdk_names", "sdk_prefixed", "sdk_numeric"])
def test_known_topology_category_forms_explicitly_qualified(style):
    row = topology_row()
    if style == "sdk_names": row["joint_types"] = list(source.RAW_SDK_JOINT_TYPES)
    if style == "sdk_prefixed": row["joint_types"] = ["FingerJointType_"+v for v in source.RAW_SDK_JOINT_TYPES]
    if style == "sdk_numeric":
        names = ["Invalid","Metacarpal","Proximal","Intermediate","Distal","Tip"]
        row["joint_types"] = [names.index(v) for v in source.RAW_SDK_JOINT_TYPES]
    q = source.validate_manus_topology(row)
    assert ("ALIASES" in q) == (style == "capture_alias")


@pytest.mark.parametrize("field,value", [
    ("node_count",24), ("node_ids",[1]+list(range(1,25))),
    ("node_ids",[False]+list(range(1,25))),
    ("parent_ids",[0]*25), ("chain_types",["Hand"]*25),
    ("joint_types",["MCP"]*25), ("joint_types",[False]+[1]*24),
])
def test_unsupported_schema_never_hardcodes_old_ids(field,value):
    row = topology_row(); row[field] = value
    with pytest.raises(source.SourceContractError, match="TOPOLOGY"):
        source.validate_manus_topology(row)


def test_missing_parent_info_not_assumed():
    row=topology_row(); del row["parent_ids"]
    with pytest.raises(source.SourceContractError, match="TOPOLOGY_PARENTS"):
        source.validate_manus_topology(row)


def test_even_unselected_topology_drift_is_rejected(tmp_path):
    qpc,_,rows=raw_fixture(tmp_path)
    extra=copy.deepcopy(rows[-1]);extra["recv_qpc_ns"]=int(qpc[-1]+100_000_000)
    extra["parent_ids"][7]=5;rows.append(extra)
    write_rows(tmp_path/"raw/manus.jsonl",rows)
    with pytest.raises(source.SourceContractError, match="TOPOLOGY_PARENTS"):
        source.load_raw_motion(tmp_path,qpc,expected_serial="PICO_A",prior=prior())


def test_mixed_valid_category_conventions_rejected(tmp_path):
    qpc,_,rows=raw_fixture(tmp_path)
    rows[0]["joint_types"]=list(source.RAW_SDK_JOINT_TYPES)
    write_rows(tmp_path/"raw/manus.jsonl",rows)
    with pytest.raises(source.SourceContractError, match="CATEGORY_SCHEMA_DRIFT"):
        source.load_raw_motion(tmp_path,qpc,expected_serial="PICO_A",prior=prior())


def test_canonical_known_answer_keeps_actual_pip_bend_and_all_tips(tmp_path):
    qpc,_,rows=raw_fixture(tmp_path)
    positions=np.array(rows[0]["nodes"],float)[:,:3]
    positions[5]=[.025,.02,0.]  # palm metacarpal base, omitted
    positions[6]=[.10,.02,0.]   # true MCP
    positions[7]=[.14,.02,0.]   # true PIP, 90 degree bend here
    positions[8]=[.14,.045,0.]  # DIP
    positions[9]=[.14,.065,0.]  # TIP
    for row in rows:
        nodes=np.array(row["nodes"],float);nodes[:,:3]=positions;row["nodes"]=nodes.tolist()
    write_rows(tmp_path/"raw/manus.jsonl",rows)
    n,m,_=source.load_raw_motion(tmp_path,qpc,expected_serial="PICO_A",prior=prior())
    local21=n["hands"][0,0]-n["wrist"][0,0,:3,3]
    np.testing.assert_allclose(local21[5:9],positions[[6,7,8,9]])
    v=np.diff(local21[5:9],axis=0);cosine=np.dot(v[0],v[1])/np.linalg.norm(v[0])/np.linalg.norm(v[1])
    assert np.degrees(np.arccos(cosine)) == pytest.approx(90.)
    np.testing.assert_array_equal(n["manus_local_25"][0,0],positions)
    np.testing.assert_array_equal(source.MAP_25_TO_21[[4,8,12,16,20]],parent_map[[4,8,12,16,20]])
    np.testing.assert_allclose(local21[[4,8,12,16,20]],positions[[4,9,14,19,24]])
    np.testing.assert_array_equal(m["video_qpc_ns"],qpc)
    meta=source.joint_mapping_metadata(n)
    assert meta["raw_alias_conflict"] and meta["raw25_positions_preserved"]
    assert meta["raw_joint_types"] == list(source.RAW_CAPTURE_JOINT_ALIASES)
    assert meta["canonical_joint_names_21"][6] == "Index_PIP"


def test_parent_human_features_now_use_mcp_scale_not_cmc():
    from chaoyang.exact78_baseline.retarget import _human_features
    raw=np.zeros((25,3))
    raw[1:5]=[[.03,.04,0.],[.06,.04,0.],[.08,.04,0.],[.10,.04,0.]]
    for start,y in [(5,.03),(10,.01),(15,-.01),(20,-.03)]:
        raw[start:start+5]=[[.02,y,0.],[.10,y,0.],[.14,y,0.],[.16,y,.01],[.18,y,.02]]
    old=raw[parent_map];canonical=raw[source.MAP_25_TO_21]
    a=_human_features(old,1)["middle"];b=_human_features(canonical,1)["middle"]
    actual_mcp_scale=np.mean([np.sqrt(.1**2+y**2) for y in [.03,.01,-.01,-.03]])
    assert np.linalg.norm(b["tip"]) == pytest.approx(np.linalg.norm(raw[14])/actual_mcp_scale)
    assert np.linalg.norm(a["tip"]) > 3*np.linalg.norm(b["tip"])
    # Correct three phalanx bones begin MCP->PIP, rather than the palm bone.
    b_angle=np.degrees(np.arccos(np.clip(np.dot(b["bones"][0],b["bones"][1]),-1,1)))
    a_angle=np.degrees(np.arccos(np.clip(np.dot(a["bones"][0],a["bones"][1]),-1,1)))
    assert b_angle == pytest.approx(np.degrees(np.arctan2(.01,.02)))
    assert a_angle == pytest.approx(np.degrees(np.arctan2(.01,.06)))


def test_legacy_prepared_source_cannot_be_relabelled_canonical(tmp_path):
    (tmp_path/"SESSION_INPUT.json").write_text(json.dumps({"status":"COMPLETE_RAW_INPUT","schema_version":"pick-place-raw-input-v1","calibration":{}}))
    with pytest.raises(source.SourceContractError, match="CANONICAL21_SCHEMA"):
        source.load_prepared(tmp_path)
