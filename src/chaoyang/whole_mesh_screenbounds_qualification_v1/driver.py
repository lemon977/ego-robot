from pathlib import Path
import json,time,hashlib,types,importlib.util,sys,os
from .contract import ROOT,TASK,SCOPE,BUDGETS,need,pin,read,validate_packet

def load_package(rows,name,root=ROOT):
 pins={Path(v['path']).name:v for v in rows};need(len(pins)==len(rows) and '__init__.py' in pins,'SCREEN_PACKAGE_UNIQUE')
 folders={str(Path(v['path']).parent) for v in rows};need(len(folders)==1,'SCREEN_ONE_PACKAGE_FOLDER')
 for v in rows:pin(v,root)
 folder=Path(next(iter(folders)));sp=importlib.util.spec_from_file_location(name,folder/'__init__.py',submodule_search_locations=[str(folder)])
 need(name not in sys.modules,'SCREEN_FRESH_EXPLICIT_NAMESPACE');m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m)
 return m
def renderer_asset_pins(robot):
    # Same actual-Task-only helper as the successful FK2 asset binder; no FK cache.
    from chaoyang.exact78_baseline.robot_model import FLANGE_MESH,MOUNT_PROXY
    from chaoyang.pick_place_1008.source import ref
    models=(robot.assets.tianji,robot.assets.left_hand,robot.assets.right_hand)
    paths={Path(model.path) for model in models}
    paths.update(Path(visual.mesh_path) for model in models for visual in model.visuals)
    paths.update((robot.repo_root/FLANGE_MESH,robot.repo_root/MOUNT_PROXY))
    return [ref(p) for p in sorted(paths)]
def run_qualification(ctx,producer,root=ROOT):
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','SCREEN_GPU0_ENV')
 for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:need(os.environ.get(key)=='2','SCREEN_CPU2_ENV')
 from chaoyang.exact78_baseline.stage_reuse import current_producer
 owner=current_producer(root);need(owner==producer and owner['task_id']==TASK,'SCREEN_CURRENT_FENCE')
 packet=read(owner['task_packet'],root);validate_packet(packet,ctx,root)
 # Real imports/Robot construction occur exclusively beyond the registered exact fence.
 import numpy as np,importlib
 from chaoyang.native_frame_fk_render_candidate_v2.driver import official_upstream,freeze_arrays,view_record,need_equal_outputs
 from chaoyang.native_frame_fk_render_candidate_v2.asset_inputs import validate_live_renderer_assets
 from chaoyang.pick_place_1008 import source,pipeline
 from chaoyang.exact78_baseline import render as rd
 from chaoyang.exact78_baseline.pipeline import _robot
 start=time.perf_counter();deadline=start+BUDGETS['wall_seconds_max']
 def guard():need(time.perf_counter()<deadline,'SCREEN_WALL600')
 def write(p,v):
  guard();p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
  return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 out=root/'_run/current'/TASK/'attempts/attempt_0001/candidate';need(not out.exists(),'SCREEN_FRESH_CANDIDATE');out.mkdir()
 c=ctx['contract'];load_package(c['control_runtime_code'],'screenbounds_frozen_T2_control_v1',root);load_package(c['candidate_runtime_code'],'screenbounds_conservative_candidate_v1',root)
 control=importlib.import_module('screenbounds_frozen_T2_control_v1.operation');cr=importlib.import_module('screenbounds_frozen_T2_control_v1.renderer');nr=importlib.import_module('screenbounds_conservative_candidate_v1.renderer')
 robot=_robot(root);validate_live_renderer_assets(renderer_asset_pins(robot),packet,c,root)
 results=[]
 for sid in SCOPE:
  guard();t=time.perf_counter();folders,case=official_upstream(root,c,sid,control);reuse=time.perf_counter()-t
  t=time.perf_counter();session=source.load_prepared(folders['source']/'source');motion=pipeline.load_arrays(folders['motion']/'actions.npz');load=time.perf_counter()-t
  before=freeze_arrays(motion);need(np.array_equal(motion['qpc_ns'],session.qpc_ns),'SCREEN_QPC_EXACT')
  K=rd.scaled_intrinsics(session.K,640,480,(session.width,session.height));external=rd.external_camera(np.linalg.inv(session.c2w[0])@motion['T_world_base'])
  renderers={'control':cr.LookupRobotRenderer(robot),'candidate':nr.LookupRobotRenderer(robot)};rows=[]
  for ordinal,i in enumerate(c['sampled_native_frames'][sid]):
   guard();qa=motion['q_arm'][i];qh=motion['q_hand'][i];camera=np.linalg.inv(session.c2w[i])@motion['T_world_base'];paired={};order=['control','candidate'] if ordinal%2==0 else ['candidate','control']
   for which in order:
    ren=renderers[which];saved=[];arrays=[];t=time.perf_counter()
    for view,cam,k,complete in [('ego',camera,K,False),('external',external,rd.EXTERNAL_K,True)]:
     guard();vt=time.perf_counter();output=ren.render(qa,qh,cam,k,complete_robot=complete);arrays.append(output)
     rec=view_record(output,qa,qh,cam,k,i,view);rec['render_call_seconds']=time.perf_counter()-vt
     if which=='candidate':rec['screen_bounds_trace']=ren.last_screenbounds_trace
     saved.append(rec)
    paired[which]=dict(arrays=arrays,records=saved,render_pair_seconds=time.perf_counter()-t)
   for x,y in zip(paired['control']['arrays'],paired['candidate']['arrays']):need_equal_outputs(x,y)
   rows.append(dict(native_frame=i,original_QPC_ns=int(session.qpc_ns[i]),pair_order=order,control=paired['control']['records'],candidate=paired['candidate']['records'],timing={w:paired[w]['render_pair_seconds'] for w in order},actual_BGR_alpha_roots_array_equal=True))
  need(before==freeze_arrays(motion),'SCREEN_FROZEN_Q_INPUT_ARRAY_MUTATION')
  results.append(dict(session_id=sid,sampled_native_frames=c['sampled_native_frames'][sid],rows=rows,q_array_pins=before,upstream=case['reused_stage_receipts'],timing_seconds=dict(reuse= reuse,load_arrays=load)))
 index=dict(schema='whole_mesh_screenbounds_fixed24_qualification_index_v1',status='COMPLETE',task_id=TASK,producer=owner,task_packet=owner['task_packet'],run_signature=packet['run_signature'],contract=ctx['spec']['contract'],plan=ctx['plan_ref'],spec=ctx['plan']['approved_spec'],sessions=results,paired_native_frames=12,control_rendered_views=24,candidate_rendered_views=24,all_paired_BGR_alpha_roots_exact=True,q_Source_Motion_Camera_masks_changed=False,FK_cache_changed=False,codec_or_precision_changed=False,physical_contact='UNKNOWN',human_review_status='PENDING',Root_adoption=False,full_pipeline_speedup='UNKNOWN',actual_wall_seconds=time.perf_counter()-start)
 write(out/'INDEX.json',index);write(out/'RESULT.json',index);return 0
