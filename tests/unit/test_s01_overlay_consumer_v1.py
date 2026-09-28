import numpy as np
import pytest
from chaoyang.pipeline.s01_overlay_consumer_v1 import consume,project,proxy_errors,projected_points,FRAMES


def fixture():
    contract={'schema_version':'S01_POINT_SEMANTICS_CONTRACT_V1','point_definitions':{k:{} for k in ['controller_tracked_origin','controller_housing_center','anatomical_wrist_center','manus_virtual_root']},
      'physical_calibration_verified':False,'adoption':'NOT_ADOPTED','sessions':{k:{} for k in ['097','098','101']},
      'metadata_validation':{'status':'REJECT_LEGACY_CONFIG_AS_COHERENT_PROVENANCE','errors':['INSTALLATION_FIT_METADATA_CONFLICT']}}
    config={'installation_fitted':False,'T_controller_wrist_estimate':np.stack([np.eye(4)]*2).tolist()}
    fit={'sides':[{'success':True}]}
    motion={'session_id':np.array('play_cards_0916_097'),'frame_id':np.arange(165),
      'source_video_frame_id':np.arange(165),'anatomical_side_names':np.array(['left','right']),
      'T_controller_wrist_estimate':np.stack([np.eye(4)]*2),'camera_image_size':np.array([1280,960]),
      'camera_source_index':np.array(1),'physical_calibration_verified':np.array(False)}
    return contract,config,fit,motion


def test_only_explicit_diagnostic_mode_consumes_rejected_provenance():
    c,f,i,m=fixture()
    assert consume(c,f,i,m,'REJECTED_PROVENANCE_DIAGNOSTIC')['adoption']=='NOT_ADOPTED'
    with pytest.raises(ValueError):consume(c,f,i,m,'CALIBRATED')
    f['installation_fitted']=True
    with pytest.raises(ValueError):consume(c,f,i,m,'REJECTED_PROVENANCE_DIAGNOSTIC')


@pytest.mark.parametrize('key,value',[('camera_source_index',0),('physical_calibration_verified',True),('session_id','play_cards_0916_102')])
def test_domain_scope_authority_rejection(key,value):
    c,f,i,m=fixture();m[key]=value
    with pytest.raises(ValueError):consume(c,f,i,m,'REJECTED_PROVENANCE_DIAGNOSTIC')


def test_projection_known_answer_invalid_depth_not_zero():
    uv,valid=project(np.array([[1.,2,2],[1,2,-1]]),np.array([[100,0,50],[0,100,60],[0,0,1]]))
    assert uv[0].tolist()==[100,160]
    assert valid.tolist()==[True,False] and np.isnan(uv[1]).all()


def test_proxy_residual_never_certifies_same_point_accuracy():
    p={'skeleton_uv':np.zeros((2,25,2)),'skeleton_valid':np.ones((2,25),bool)}
    ann={'points':[{'session':'play_cards_0916_097','frame':15,'side':'left','node_id':4,'visible':True,'uv_1280':[3,4],'sigma_px':20}]}
    r=proxy_errors(p,ann,15)[0]
    assert r['discrepancy_px']==5 and r['independent_accuracy'] is False and r['quality_pass'] is None
    p['skeleton_valid'][0,4]=False
    assert proxy_errors(p,ann,15)[0]['discrepancy_px'] is None


def test_frame_budget_not_expandable():
    assert len(FRAMES)==12
    with pytest.raises(ValueError):projected_points({},0)


def test_independent_wrist_and_head_masks_propagate():
    T=np.tile(np.eye(4),(165,2,1,1));T[:,:,2,3]=1
    m={'T_camera_wrist':T,'T_camera_controller':T,'manus_local_25_m':np.zeros((165,2,25,3)),
       'T_world_controller':T,'wrist_world_valid':np.ones((165,2),bool),
       'camera_K':np.eye(3),'joint_camera_valid_25':np.ones((165,2,25),bool),
       'head_valid':np.ones(165,bool),'wrist_camera_valid':np.ones((165,2),bool)}
    m['wrist_camera_valid'][15,0]=False;m['joint_camera_valid_25'][15,0]=False
    p=projected_points(m,15)
    assert np.isnan(p['wrist_uv'][0]).all() and p['wrist_valid'][1]
    assert np.isnan(p['skeleton_uv'][0]).all() and p['skeleton_valid'][1].all()
    m['head_valid'][15]=False
    assert not projected_points(m,15)['controller_valid'].any()
    assert not projected_points(m,15)['skeleton_valid'].any()
    m['head_valid'][15]=True;m['wrist_world_valid'][15,1]=False
    p=projected_points(m,15)
    assert p['controller_valid'].tolist()==[True,False]
    assert not p['wrist_valid'][1] and not p['skeleton_valid'][1].any()
    m['wrist_world_valid']=np.ones((165,2),float)
    with pytest.raises(ValueError,match='controller'):projected_points(m,15)
