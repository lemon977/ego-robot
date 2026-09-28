"""Build the receipt-bound current four-stream status projection."""
from pathlib import Path
from chaoyang.governance.build_algorithm_baseline_v2_status import snapshot, alive
from chaoyang.governance.common import artifact_ref, load_json
TASK_PREFIX='four_stream_full_pipeline_v4'
V4_LANES=('exact78','controller_manus','hawor_retarget','huro')
V5_TASK='four_stream_visual_delivery_v5'
V5_LANES=('scene','sensor','motion','huro')
R1_TASK='human_to_robot_completion_r1_20260922'
R1_LANES=('lane1_scene','lane2_sensor','lane3_product','lane4_compare')
R2_TASK='human_to_robot_root_cause_gated_r2_20260923'
R2_LANES=('lane1_scene','lane2_motion','lane3_sensor','lane4_compare')
SHARED_TASK='human_to_robot_shared_hand_delivery_20260924'
SHARED_LANES=('hand_data','robot','clean','delivery')
LIVE={'PENDING','READY','CLAIMED','RUNNING','WAIT_GPU_RESOURCE'}
def _current_task(state):
 next_id=str((state.get('next_task') or {}).get('task_id',''))
 if next_id==V5_TASK:
  selected=next((x for x in state.get('tasks',[]) if x.get('task_id')==V5_TASK),None)
  if selected is None: raise RuntimeError('V5 next task missing from task state')
  return selected
 if next_id==R1_TASK:
  selected=next((x for x in state.get('tasks',[]) if x.get('task_id')==R1_TASK),None)
  if selected is None: raise RuntimeError('R1 next task missing from task state')
  return selected
 if next_id==R2_TASK:
  selected=next((x for x in state.get('tasks',[]) if x.get('task_id')==R2_TASK),None)
  if selected is None: raise RuntimeError('R2 next task missing from task state')
  return selected
 if next_id==SHARED_TASK:
  selected=next((x for x in state.get('tasks',[]) if x.get('task_id')==SHARED_TASK),None)
  if selected is None: raise RuntimeError('shared-hand next task missing from task state')
  return selected
 if not next_id:
  selected=next((x for x in state.get('tasks',[]) if x.get('task_id')==SHARED_TASK),None)
  if selected is not None:return selected
  selected=next((x for x in state.get('tasks',[]) if x.get('task_id')==V5_TASK),None)
  if selected is not None:return selected
 selected=next((x for x in state.get('tasks',[]) if x.get('task_id')==R1_TASK),None)
 if selected is not None:return selected
 selected=next((x for x in state.get('tasks',[]) if x.get('task_id')==R2_TASK),None)
 if selected is not None:return selected
 candidates=[x for x in state.get('tasks',[]) if str(x.get('task_id','')).startswith(TASK_PREFIX) and x.get('status') in LIVE]
 if candidates: return candidates[-1]
 candidates=[x for x in state.get('tasks',[]) if str(x.get('task_id','')).startswith(TASK_PREFIX)]
 if not candidates: return None
 return candidates[-1]
def build_status(root,state,at):
 root=Path(root); parent=_current_task(state); task=str(parent['task_id']); out=root/'_run/current'/task/'attempts/attempt_0001'; lanes={}
 if task==SHARED_TASK and (out/'RESULT.json').is_file():
  result=load_json(out/'RESULT.json')
  return {'schema_version':'HUMAN_TO_ROBOT_SHARED_HAND_DELIVERY_CURRENT_STATUS_V1','generated_at':at,
          'status':'PASS_NO_ACTIVE_TASKS','active_tasks':[],'latest_task':task,
          'latest_terminal_status':parent.get('status'),'result':artifact_ref(out/'RESULT.json'),
          'counts':result.get('counts',{}),
          'robot_qualified_scope':'NO_NEW_FULL_SCOPE; 3_OF_4_WINDOW_BLOCKS_FEASIBLE_NOT_ADOPTED',
          'clean_qualified_scope':'NONE; METHOD_QUALITY_NOT_EVALUATED_AFTER_ADAPTER_LIMIT',
          'data_label_quality_pass':False,'data_interface_pass':False,
          'training_eligible':False,'physical_deployable':False,'control_ground_truth':False,
          'external_metric_authority':False,
          'claim_limit':'Receipt-bound terminal projection; module evidence does not promote product or training eligibility.'}
 lane_names=R1_LANES if task==R1_TASK else (R2_LANES if task==R2_TASK else (SHARED_LANES if task==SHARED_TASK else (V5_LANES if task==V5_TASK else V4_LANES)))
 for lane in lane_names:
  s=snapshot(out/'lanes'/lane/'STATE.json'); v=s.pop('value',{}); w=v.get('writer',{})
  lanes[lane]={'status':v.get('status','UNKNOWN'),'current_action':v.get('current_action'),'blocker':v.get('blocker'),'updated_at':v.get('updated_at'),'writer_pid':w.get('pid'),'writer_live':alive(w.get('pid'),w.get('proc_start_ticks')),'claims':v.get('claims',{}),'latest_artifacts':v.get('latest_artifacts',[]),'source_snapshot':s}
 schema='HUMAN_TO_ROBOT_COMPLETION_R1_STATUS' if task==R1_TASK else ('HUMAN_TO_ROBOT_ROOT_CAUSE_GATED_R2_STATUS' if task==R2_TASK else ('FOUR_STREAM_STATUS_V5' if task==V5_TASK else 'FOUR_STREAM_FULL_PIPELINE_V4_STATUS'))
 return {'schema_version':schema,'generated_at':at,'parent_task_id':task,'parent_status':parent.get('status'),'run_root':str(out),'lanes':lanes,'training_eligible':False,'physical_deployment_authorized':False,'control_ground_truth':False,'external_metric_authority':False,'claim_limit':'Receipt-bound navigation only; lane execution is not quality adoption.'}
def publish_navigation_if_v4(root,state,at):
 root=Path(root); plan=root/'docs/current/PLAN.md'
 if not plan.exists() or not any(marker in plan.read_text() for marker in ('FOUR_STREAM_FULL_PIPELINE_V4','HUMAN_TO_ROBOT_BASELINE_V1','human_to_robot_completion_r1_20260922','human_to_robot_root_cause_gated_r2_20260923','human_to_robot_shared_hand_delivery_20260924')): return
 if _current_task(state) is None: return
 from chaoyang.governance.common import atomic_json
 atomic_json(root/'docs/current/STATUS.json',build_status(root,state,at))
