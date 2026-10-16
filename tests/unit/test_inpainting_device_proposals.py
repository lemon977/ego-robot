import numpy as np
import pytest
from chaoyang.inpainting_candidate.v1_device_proposals import bind_seeded_roles,execute_generic_object_probe


def fixture():
    seed={'raw_frame_index':0,'objects':[
      {'obj_id':1,'role':'glove','side':'left','frame_idx':0,'visible_confirmed':True,
       'points_px':[[3,3],[8,8],[1,1]],'point_labels':[1,0,0]},
      {'obj_id':2,'role':'mount','side':'right','frame_idx':0,'visible_confirmed':True,
       'points_px':[[8,8],[3,3],[1,1]],'point_labels':[1,0,0]}]}
    cup=np.zeros((12,12),bool);cup[5,5]=True
    return seed,cup


def test_same_side_known_positive_is_eligibility_not_deletion():
    seed,cup=fixture();m=np.zeros_like(cup);m[8:10,8:10]=True
    r=bind_seeded_roles({'device':m},seed,cup)[0]
    assert r['status']=='CANDIDATE_V1_SEED_BOUND_WEARER' and r['side']=='right'
    assert not r['deletion_authorized'] and not r['semantic_ground_truth']


def test_table_background_and_cup_conflict_stay_unknown():
    seed,cup=fixture();table=np.ones_like(cup)
    r=bind_seeded_roles({'table':table},seed,cup)[0]
    assert r['status']=='UNKNOWN' and 'BACKGROUND_OR_OBJECT_POINT_CONFLICT' in r['reasons']
    assert 'VISIBLE_CUP_SUPPORT_CONFLICT' in r['reasons'] and 'SIDE_NOT_UNIQUE' in r['reasons']


def test_emergency_stop_without_positive_is_not_wearer():
    seed,cup=fixture();m=np.zeros_like(cup);m[10,1]=True
    r=bind_seeded_roles({'estop':m},seed,cup)[0]
    assert r['status']=='UNKNOWN' and 'NO_EXISTING_WEARER_POSITIVE_PIXEL' in r['reasons']


def test_cross_role_negative_is_not_mistaken_for_background():
    seed,cup=fixture();m=np.zeros_like(cup);m[8,8]=True
    r=bind_seeded_roles({'right_mount':m},seed,cup)[0]
    assert not r['background_conflicts']  # The other object's negative is this mount's positive.


def test_noninteger_or_outside_seed_not_rounded_or_clipped():
    seed,cup=fixture();seed['objects'][0]['points_px'][0]=[3.4,3]
    with pytest.raises(ValueError,match='NONINTEGER'):bind_seeded_roles({},seed,cup)


def test_unsupported_query_rejected_before_any_load():
    with pytest.raises(ValueError,match='STRICT_ONE_GENERIC'):
        execute_generic_object_probe({'mode':'Mask','config':{'generic_object_probe':True,'query':'VR controller'},
                                     'budgets':{},'sessions':[]},authority=lambda:pytest.fail('no models'))


def test_full_probe_worker_is_proposal_only_one_api(monkeypatch,tmp_path):
    import json
    from types import SimpleNamespace
    from chaoyang.governance import common
    from chaoyang.inpainting_candidate import model_stages,source_frames
    from chaoyang.governance.common import artifact_ref
    n=4;raw=np.full((n,480,640,3),80,np.uint8)
    np.savez_compressed(tmp_path/'RAW.npz',rgb=raw)
    def js(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return artifact_ref(p)
    rgb_ref=artifact_ref(tmp_path/'RAW.npz')
    mapping=js('MAP.json',[{'native_frame_index':i,'output_frame':i,'qpc_declared_ns':1000000000+i*33333333} for i in range(n)])
    source=js('SOURCE.json',{'stage':'Source','status':'SEALED_COMPLETE','session_id':'synthetic',
         'native_frames':n,'outputs':[rgb_ref,mapping]})
    calls=[]
    def track(frames,prompt,anchor):
        calls.append((len(frames),prompt,anchor));m=np.zeros((480,640),bool);m[200:220,300:320]=True
        return {7:{i:np.packbits(m) for i in range(n)}}
    monkeypatch.setattr(common,'REPO_ROOT',tmp_path);monkeypatch.setattr(source_frames,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(model_stages,'configure_project_caches',lambda p:{})
    def load(*args,**kw):
        assert kw['need_sam'] is True and kw['need_clean'] is False
        return {'sam':SimpleNamespace(track_text=track,aborted=[])}
    monkeypatch.setattr(model_stages,'load_frozen_models',load)
    packet={'task_id':'synthetic','run_signature':'a'*64,'mode':'Mask','config':{
        'generic_object_probe':True,'query':'object','anchor_frame':0},'budgets':{
        'gpu_count':1,'model_calls_max':1,'render_calls_max':0,'training_steps_max':0,'output_bytes_estimate':50*1024*1024},
        'sessions':[{'session_key':'synthetic','native_frames':n,'review_anchors':[0]}],
        'runtime_code':[],'model_weights':[],'inputs':{'frozen_source_raw':rgb_ref,'frozen_source_receipt':source,'frozen_frame_map':mapping},
        'output_dir':str(tmp_path/'out'),'review_dir':str(tmp_path/'review')}
    assert execute_generic_object_probe(packet,authority=lambda:None)==0
    assert calls==[(4,'object',0)]
    out=json.loads((tmp_path/'out/MODEL_RESULT.json').read_text())
    assert out['model_calls']==1 and out['deletion_mask_produced'] is False
    proposal=json.loads((tmp_path/'out/sessions/synthetic/Proposals/GENERIC_OBJECT_PROPOSALS.json').read_text())
    assert len(proposal['proposals'])==1 and proposal['identity_role']=='UNKNOWN'
    assert proposal['native_frames_submitted']==n and proposal['proposal_consumable']
