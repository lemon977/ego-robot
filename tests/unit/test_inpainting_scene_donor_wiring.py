import pytest
from chaoyang.inpainting_candidate import v1_mco_worker
from chaoyang.inpainting_candidate.v1_scene_donor_worker import require_cpu_donor


def packet():
    return {'mode':'Clean','config':{'cpu_raw_donor':True,'donor_count':12},
       'budgets':{k:0 for k in ('gpu_count','model_calls_max','render_calls_max','training_steps_max')},
       'model_weights':[],'sessions':[{'session_key':'independent_synthetic','native_frames':6}]}


@pytest.mark.parametrize('key',['gpu_count','model_calls_max','render_calls_max','training_steps_max'])
def test_no_resource_bypass(key):
    p=packet();p['budgets'][key]=1
    with pytest.raises(ValueError,match='STRICT_CPU'):require_cpu_donor(p)


def test_cpu_donor_valid_contract():require_cpu_donor(packet())


def test_partial_frames_forbidden():
    p=packet();p['sessions'][0]['output_frames']=[0]
    with pytest.raises(ValueError,match='STRICT_CPU'):require_cpu_donor(p)


def test_route_dispatches_before_model_import(monkeypatch):
    import chaoyang.inpainting_candidate.v1_scene_donor_worker as module
    calls=[]
    monkeypatch.setattr(v1_mco_worker,'_authority',lambda p:lambda:None)
    monkeypatch.setattr(module,'execute_raw_donor',lambda p,authority:calls.append(p) or 0)
    p=packet();assert v1_mco_worker.execute_v1_mco_stage(p)==0 and calls==[p]
