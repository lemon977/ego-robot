from .contract import ROOT,TASK,SCOPE,need,read
def validate_saved_index(ix,result,packet,ctx,owner):
    need(ix==result and ix['producer']==owner and ix['contract']==ctx['spec']['contract'] and ix['task_packet']==owner['task_packet'],'SCREEN_COLLECTOR_MANIFEST_FENCE')
    need(ix['task_id']==TASK and ix['run_signature']==packet['run_signature']==owner['run_signature'] and ix['plan']==ctx['plan_ref'] and ix['spec']==ctx['plan']['approved_spec'],'SCREEN_COLLECTOR_CURRENT_SIGNATURE_PLAN_SPEC')
    need(ix['status']=='COMPLETE' and type(ix['paired_native_frames']) is int and ix['paired_native_frames']==12 and type(ix['control_rendered_views']) is int and type(ix['candidate_rendered_views']) is int and ix['control_rendered_views']==ix['candidate_rendered_views']==24 and [r['session_id'] for r in ix['sessions']]==SCOPE,'SCREEN_COLLECTOR_FIXED_SCOPE')
    for k in ['q_Source_Motion_Camera_masks_changed','FK_cache_changed','codec_or_precision_changed','Root_adoption']:need(ix[k] is False,'SCREEN_COLLECTOR_FALSE_FLAG:'+k)
    need(ix['physical_contact']=='UNKNOWN' and ix['human_review_status']=='PENDING','SCREEN_COLLECTOR_NO_TRUTH_OR_REVIEW_UPGRADE')
    for row in ix['sessions']:
        samples=ctx['contract']['sampled_native_frames'][row['session_id']]
        need(row['sampled_native_frames']==samples and len(row['rows'])==3,'SCREEN_COLLECTOR_THREE_NATIVES')
        need(all(type(p['native_frame']) is int for p in row['rows']) and [p['native_frame'] for p in row['rows']]==samples,'SCREEN_COLLECTOR_EXACT_NATIVE_SEQUENCE')
        for pair in row['rows']:
            need(len(pair['control'])==len(pair['candidate'])==2 and [v['view'] for v in pair['control']]==[v['view'] for v in pair['candidate']]==['ego','external'],'SCREEN_COLLECTOR_EXACT_TWO_ORDERED_VIEWS')
            need(all(type(v['native_frame']) is int and v['native_frame']==pair['native_frame'] for side in ['control','candidate'] for v in pair[side]),'SCREEN_COLLECTOR_VIEW_NATIVE')
    return True

def collect(packet,ctx,root=ROOT):
    from chaoyang.exact78_baseline.stage_reuse import current_producer
    owner=current_producer(root);need(owner['task_id']==TASK and owner['task_packet']==ctx['packet_ref'] and owner['run_signature']==packet['run_signature'],'SCREEN_COLLECTOR_FOUR_FIELDS')
    expected_result=str(root/'_run/current'/TASK/'attempts/attempt_0001/RESULT.json');need(owner['result_path']==expected_result,'SCREEN_COLLECTOR_OUTER_RESULT')
    from chaoyang.pick_place_1008 import source,pipeline
    from chaoyang.exact78_baseline import render as rd
    from chaoyang.native_frame_fk_render_candidate_v2.driver import array_pin,freeze_arrays
    import numpy as np,json
    folder=root/'_run/current'/TASK/'attempts/attempt_0001/candidate'
    ix=json.loads((folder/'INDEX.json').read_text());result=json.loads((folder/'RESULT.json').read_text())
    validate_saved_index(ix,result,packet,ctx,owner)
    pairs=0
    for row in ix['sessions']:
        sid=row['session_id'];case=ctx['contract']['cases'][sid];need(row['sampled_native_frames']==ctx['contract']['sampled_native_frames'][sid] and len(row['rows'])==3,'SCREEN_COLLECTOR_THREE_NATIVES')
        sf=__import__('pathlib').Path(case['reused_stage_receipts']['source']['path']).parent/'source';mf=__import__('pathlib').Path(case['reused_stage_receipts']['motion']['path']).parent/'actions.npz'
        session=source.load_prepared(sf);motion=pipeline.load_arrays(mf)
        need(row['q_array_pins']==freeze_arrays(motion),'SCREEN_COLLECTOR_CURRENT_FROZEN_Q')
        K=rd.scaled_intrinsics(session.K,640,480,(session.width,session.height));external=rd.external_camera(np.linalg.inv(session.c2w[0])@motion['T_world_base'])
        for pair in row['rows']:
            i=pair['native_frame'];need(type(pair['original_QPC_ns']) is int and pair['original_QPC_ns']==int(session.qpc_ns[i])==int(motion['qpc_ns'][i]),'SCREEN_COLLECTOR_INTEGER_QPC')
            need(pair['actual_BGR_alpha_roots_array_equal'] is True,'SCREEN_COLLECTOR_PAIR_RESULT')
            for a,b in zip(pair['control'],pair['candidate']):
                need(a['view']==b['view'] and all(a[k]==b[k] for k in ['native_frame','q_arm','q_hand','camera','K','BGR','alpha','roots']),'SCREEN_COLLECTOR_LITERAL_TUPLE')
                camera=np.linalg.inv(session.c2w[i])@motion['T_world_base'] if a['view']=='ego' else external;ki=K if a['view']=='ego' else rd.EXTERNAL_K
                need(a['q_arm']==array_pin(motion['q_arm'][i]) and a['q_hand']==array_pin(motion['q_hand'][i]) and a['camera']==array_pin(camera) and a['K']==array_pin(ki),'SCREEN_COLLECTOR_SAVED_CURRENT_CAMERA_K')
                pairs+=1
    need(pairs==24,'SCREEN_COLLECTOR_EXACT24')
    return dict(status='PASS',paired_native_frames=12,paired_views=24,source_qpc_q_K_camera_rebuilt=True,real_BGR_alpha_roots_comparison='WORKER_ACTUAL_ARRAY_EQUAL_AND_BOUND_SAVED_PAIRS',model_FK_or_Render_reexecuted=False,full_pipeline_speedgain='UNKNOWN')
