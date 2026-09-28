"""Finite V3 operation closure, explicitly enumerated by the sole publisher."""
from pathlib import Path
import json
from chaoyang.governance.common import artifact_ref, validate_artifact_ref

TASK='four_stream_full_pipeline_v3'

def build_v3_contract(repo_root: Path) -> dict:
    root=repo_root/'_run/current'/TASK/'attempts/attempt_0001'
    registry=root/'CODE_REGISTRY.json'
    packet=repo_root/'tasks/current'/TASK/'TASK_PACKET.json'
    if not registry.is_file() or not packet.is_file():return {}
    value=json.loads(registry.read_text())
    for ref in value['code_closure']:
        if validate_artifact_ref(ref):raise ValueError('V3_CODE_CLOSURE_DRIFT: '+ref['path'])
    return {TASK:dict(execution_scope='ORIGINAL_SCENE_FULL_PIPELINE_VISUAL_RESEARCH',
       task_packet=artifact_ref(packet),code_registry=artifact_ref(registry),code_closure=value['code_closure'],
       operation_policy='EXPLICIT_SHA_BOUND_CODE_ONLY',weights='EXISTING_PINNED_MODELS_ONLY',
       candidate_signatures_per_lane_max=3,gpu_concurrent_owners_max=1,
       training_eligible=False,control_ground_truth=False,physical_deployment_authorized=False,
       claim_limit='Full stage execution is distinct from quality; ego-data consistency is not independent physical accuracy.')}
