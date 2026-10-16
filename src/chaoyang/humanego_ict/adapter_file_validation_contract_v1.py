"""Candidate CPU route, not production or hardware deployment adoption."""
from pathlib import Path
from chaoyang.governance.common import artifact_ref
from chaoyang.humanego_ict.derived_contract import candidate_contract as old_contract, ROUTE as OLD_ROUTE

OPERATION = 'audit_pick_place_humanego_ict_offline_envelope_v1'
ROUTE = 'pick_place_humanego_ict_offline_envelope_v1'
CANDIDATE = '_run/maintenance/ict_inference_deploy_completion_20261012/attempt_0001/lanes/adapter/offline_envelope_candidate.py'
CANDIDATE_SHA = '76b026cc5942e1210b7136b870860e3a7e3b7372bd8f09d17326fa0f61b38354'


def candidate_contract(root: Path) -> dict:
    paths = ['src/chaoyang/ops/'+OPERATION+'.py',
        'src/chaoyang/humanego_ict/adapter_file_validation_runtime_v1.py',
        'src/chaoyang/humanego_ict/adapter_file_validation_contract_v1.py',
        'src/chaoyang/humanego_ict/assumed_deployment_v1.py',
        'src/chaoyang/humanego_ict/deployment_adapter.py', CANDIDATE]
    previous = old_contract(root).get(OLD_ROUTE)
    if previous is None or any(not (root/p).is_file() for p in paths):
        return {}
    if artifact_ref(root/CANDIDATE)['sha256'] != CANDIDATE_SHA:
        raise RuntimeError('FROZEN_ENVELOPE_CANDIDATE_SHA_DRIFT')
    return {ROUTE: {'candidate_operations':[OPERATION], 'task_packet_required':True,
        'code_closure':previous['code_closure']+[artifact_ref(root/p) for p in paths],
        'execution_mode':'HUMANEGO_ICT_OFFLINE_ENVELOPE_FILE_VALIDATION_CPU',
        'source_policy':'FROZEN_ADMITTED_PROJECT_PAYLOAD_SNAPSHOT_RESEARCH_ONLY',
        'control_ground_truth':False,'physical_deployment_authorized':False,
        'current_runnable_operation':None,'model_calls_authorized':0,'gpu_count':0,
        'claim_limit':'Sealed local H50 predictions and current-only input bindings, strict REJECT retained; no model, future target decode, IK provider, feedback, network or hardware qualification.'}}
