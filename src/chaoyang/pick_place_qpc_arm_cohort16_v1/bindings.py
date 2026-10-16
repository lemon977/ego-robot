"""Baseline S/G remain byte-identical; only new M/R get this typed recipe."""
from .approval import ref,exact
from chaoyang.pick_place_1008 import pipeline
RECIPE='FIXED_REAL_QPC_MINIMAL_L2_ARM_TRAJECTORY_PROJECTION_V1'
def family_binding(repo,stage,row,camera,upstream,approval,extra_code):
 if stage not in ['motion','render']:raise ValueError('QP_ONLY_NEW_MOTION_RENDER_BINDINGS')
 value=pipeline.binding(repo,stage,row,camera,upstream)
 value['code_and_dependencies']=sorted({x['path']:x for x in value['code_and_dependencies']+extra_code+approval['exact_existing_runtime_code']}.values(),key=lambda x:x['path'])
 value['qpc_arm_projection_recipe']={'recipe_id':RECIPE,'contract':approval['contract_ref'],'evaluation_selection':approval['evaluation_selection'],'session_id':row['session_id'],'source_GPU_original_exact_reuse':True,'q_hand_base_targets_QPC_camera_P_unchanged':True,'contact_from_same_new_saved_q_required':True,'human_review_status':'PENDING','physical_contact_ground_truth':False,'control_ground_truth':False,'training_eligible':False}
 return value
