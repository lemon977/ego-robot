import json
from pathlib import Path
import pytest
from chaoyang.governance.build_algorithm_baseline_v2_status import snapshot,alive,build_status,TASK_ID


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_snapshot_absent_and_bad_schema(tmp_path):
    p=tmp_path/'x';assert snapshot(p)['read_status']=='ABSENT'
    p.write_text('[');assert snapshot(p)['read_status']=='INVALID_JSON'
    p.write_text('[]');assert snapshot(p)['read_status']=='INVALID_SCHEMA'


def test_snapshot_records_observation_not_permanent_sha_claim(tmp_path):
    p=tmp_path/'x';put(p,{'status':'RUNNING'})
    value=snapshot(p)
    assert value['value']['status']=='RUNNING'
    assert 'sha256' not in value and value['observed_sha256']


def test_alive_checks_start_ticks_and_zombie(tmp_path):
    p=tmp_path/'123/stat';p.parent.mkdir()
    fields=['S']+['0']*18+['456']
    p.write_text('123 (name with spaces) '+' '.join(fields))
    assert alive(123,456,tmp_path)
    assert not alive(123,457,tmp_path)
    p.write_text('123 (name) '+' '.join(['Z']+fields[1:]))
    assert not alive(123,456,tmp_path)


def test_empty_lane_never_promotes_pass(tmp_path):
    state={'tasks':[{'task_id':TASK_ID,'status':'RUNNING'}]}
    result=build_status(tmp_path,state,'now')
    assert all(x['status']=='UNKNOWN_NOT_PASS' for x in result['lanes'].values())
    assert not result['parent']['routable'] and not result['training_eligible']


def test_completed_job_is_not_quality_adoption(tmp_path):
    put(tmp_path/f'_run/current/{TASK_ID}/attempts/attempt_0001/lanes/ai2/ROBOT_RUN_STATE.json',
        {'status':'EXECUTED','pid':123,'proc_start_ticks':1})
    result=build_status(tmp_path,{'tasks':[{'task_id':TASK_ID,'status':'RUNNING'}]},'now')
    lane=result['lanes']['ai2']
    assert lane['status']=='ROBOT_EXECUTED_PENDING_EVALUATION'
    assert not lane['algorithm_quality_pass']


def test_duplicate_parent_refused(tmp_path):
    with pytest.raises(ValueError):build_status(tmp_path,{'tasks':[{'task_id':TASK_ID}]*2},'now')
