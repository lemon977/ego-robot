"""Registration closure for full-source causal derived ICT production/training."""
from pathlib import Path
import json
from chaoyang.governance.common import artifact_ref
from .production_training_gate import (PREPARE, OPERATION, PREPARE_ROUTE, ROUTE,
                                      PREPARE_MODE, EXECUTION_MODE, SCHEMA, TRAINING_SCOPE)


def candidate_contract(root: Path) -> dict:
    common = ["src/chaoyang/ops/" + name + ".py" for name in (PREPARE, OPERATION)] + [
        "scripts/humanego_ict/production_runner_v1.py", "scripts/humanego_ict/train_production_pick_place.sh",
        *["src/chaoyang/humanego_ict/" + name + ".py" for name in (
            "production_source", "production_data", "production_renderer", "production_training_cli",
            "production_training_gate", "production_contract", "derived_runtime", "derived_contract", "retarget",
            "data", "runtime_gate", "derived_training_cli", "derived_training_gate", "derived_data", "inference",
            "checkpoint_export", "model", "training")],
        "src/chaoyang/cli/__init__.py", "src/chaoyang/human_ego/training/FlowMatchingModel.py",
        "src/chaoyang/human_ego/training/checkpoint_transfer.py", "src/chaoyang/human_ego/utils/atomic_io.py",
        "src/chaoyang/human_ego/utils/frozen_contract.py", "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
        "src/chaoyang/governance/gpu_lease_v71.py"]
    if any(not (root / path).is_file() for path in common):
        return {}
    audit_path = root / "_run/current/pick_place_humanego_ict_two_checkpoint_preparation_v1_20261011/attempts/attempt_0001/lane_rgb/DECODE_ASSET_READONLY_AUDIT_V1.json"
    if not audit_path.is_file():
        return {}
    audit = json.loads(audit_path.read_bytes())
    extra = [ref for name in ("code_refs", "decoder_refs", "extra_asset_refs", "selected_urdf_visual_refs") for ref in audit[name]]
    if any(artifact_ref(Path(ref["path"])) != ref for ref in extra):
        return {}
    closure = {ref["path"]: ref for ref in [*[artifact_ref(root / path) for path in common], *extra, artifact_ref(audit_path)]}
    base = {"code_closure": list(closure.values()), "task_packet_required": True,
        "manifest_schema": SCHEMA, "training_scope": TRAINING_SCOPE, "supervision_class": "DERIVED_RETARGET",
        "control_ground_truth": False, "physical_deployment_authorized": False, "current_runnable_operation": None,
        "qualification_status": "NOT_PASSED", "real_control_route_unchanged": True, "old_derived_route_unchanged": True,
        "causal_time_group_window_gates_unchanged": True, "source_replay": "FULL_RAW_LATEST_PAST",
        "storage_policy": "PROJECT_ONLY_MINIMUM_4GIB; ZERO_STATVFS_REQUIRES_BOUND_SAME_FILESYSTEM_WRITE_FSYNC_READBACK_WITHIN_1H; NOT_A_TOTAL_FREE_SPACE_GUARANTEE",
        "claim_limit": "Source-derived research checkpoints only; not measured controls, physical calibration or deployment approval. Independent finite producer/training packets, source/time/group/window evidence and a live fenced writer are mandatory."}
    return {PREPARE_ROUTE: dict(base, candidate_operation=PREPARE, execution_mode=PREPARE_MODE),
            ROUTE: dict(base, candidate_operation=OPERATION, execution_mode=EXECUTION_MODE)}
