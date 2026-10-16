"""Additive candidate contract for exact sealed-state ICT continuation.

This builder only records existing files.  It never runs a model, registers a
task or treats the old eight-hour training authorization as renewable.
"""
from __future__ import annotations

from pathlib import Path

from chaoyang.governance.common import artifact_ref

from .production_contract import candidate_contract as production_contract
from .production_training_gate import ROUTE as PRODUCTION_ROUTE
from .production_resume_gate import OPERATION, ROUTE, EXECUTION_MODE


RESUME_FILES = (
    "src/chaoyang/humanego_ict/production_resume.py",
    "src/chaoyang/humanego_ict/production_resume_gate.py",
    "src/chaoyang/humanego_ict/production_resume_cli.py",
    "src/chaoyang/humanego_ict/production_resume_contract.py",
    "src/chaoyang/ops/train_pick_place_humanego_ict_production_resume_v1.py",
    "scripts/humanego_ict/production_resume_runner_v1.py",
    "tests/humanego_ict/test_production_resume.py",
    "tests/humanego_ict/test_production_resume_gate.py",
    "tests/humanego_ict/test_production_resume_cli.py",
)


def candidate_contract(root: Path) -> dict:
    """Preserve the complete original producer/trainer closure, add a new route.

    Engineering may still be in progress when a governance heartbeat invokes
    this builder.  Missing new files mean no advertised resume route, not a
    launch or a failure of the existing production contract.  A sealed passing
    CPU engineering packet and its tested SHA closure are checked at admission
    by the independent resume gate and finite publisher.
    """
    prior = production_contract(root).get(PRODUCTION_ROUTE)
    if prior is None or any(not (root / name).is_file() for name in RESUME_FILES):
        return {}
    added = [artifact_ref(root / name) for name in RESUME_FILES]
    closure = {ref["path"]: ref for ref in [*prior["code_closure"], *added]}
    route = dict(prior,
        code_closure=list(closure.values()),
        candidate_operation=OPERATION,
        execution_mode=EXECUTION_MODE,
        current_runnable_operation=None,
        resume_support=True,
        parent_sealed_checkpoint_required=True,
        sealed_cpu_engineering_required=True,
        resume_state="MODEL_ADAMW_EMA_RANK_RNG_STEP_CURSOR_FULL_STATE",
        parent_code_closure_preserved=True,
        finite_task_wall_seconds_max=43200,
        old_eight_hour_authorization_not_extended=True,
        claim_limit=("Sealed-state research continuation only. New finite task, "
            "complete parent result/checkpoint/sidecar and passing SHA-bound CPU "
            "engineering are required. Source, recipe, numerical, time, causal "
            "group and physical-deployment gates remain unchanged."))
    return {ROUTE: route}
