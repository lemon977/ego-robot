import json
from copy import deepcopy
from pathlib import Path

import pytest

from pipeline.pico_geometry_mask_protocol import (
    PicoGeometryProtocolError,
    validate_anchor_provider_authority,
    validate_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
SHARED_PATH = "systems/mask/configs/pico_raw_point_geometry_successor_v1.json"


def load(path):
    return json.loads((ROOT / path).read_text())


@pytest.mark.parametrize(
        ("task_path", "plan_path"),
    [
        ("tasks/chips/config/mask_pico_geometry_roles_v1.json", "tasks/chips/runs/mask/20260903_provider_contact_refreeze_v2/FIT_CHIPS008_EVALUATION_PLAN.json"),
        ("tasks/chips/config/mask_pico_geometry_roles_v1.json", "tasks/chips/runs/mask/20260903_provider_contact_refreeze_v2/EVAL_CHIPS011_EVALUATION_PLAN.json"),
        ("tasks/poker/config/mask_pico_geometry_roles_v1.json", "tasks/poker/runs/mask/20260903_provider_contact_refreeze_v2/EVAL_POKER001_EVALUATION_PLAN.json"),
    ],
)
def test_frozen_fit_eval_bundles_pass(task_path, plan_path):
    report = validate_bundle(load(SHARED_PATH), load(task_path), load(plan_path), SHARED_PATH)
    assert report["status"] == "PASS_FROZEN_CPU_CONTRACT"


def test_manual_points_and_threshold_override_fail_closed():
    shared = load(SHARED_PATH)
    task = load("tasks/chips/config/mask_pico_geometry_roles_v1.json")
    plan = load("tasks/chips/runs/mask/20260903_provider_contact_refreeze_v2/FIT_CHIPS008_EVALUATION_PLAN.json")
    broken = deepcopy(shared)
    broken["tracker_anchor_generation"]["manual_per_frame_points_forbidden"] = False
    with pytest.raises(PicoGeometryProtocolError):
        validate_bundle(broken, task, plan, SHARED_PATH)
    broken_plan = deepcopy(plan)
    broken_plan["threshold_override"] = {"area": 0.5}
    with pytest.raises(PicoGeometryProtocolError):
        validate_bundle(shared, task, broken_plan, SHARED_PATH)
    broken_setup = deepcopy(shared)
    broken_setup["setup_world_anchor_generation"]["image_colour_or_intensity_used"] = True
    with pytest.raises(PicoGeometryProtocolError):
        validate_bundle(broken_setup, task, plan, SHARED_PATH)


def test_provider_is_fixed_per_side_and_g2_or_hawor_identity_is_required():
    shared = load(SHARED_PATH)
    plan = load("tasks/chips/runs/mask/20260903_provider_contact_refreeze_v2/FIT_CHIPS008_EVALUATION_PLAN.json")
    authority = {
        "schema_version": "mask-hand-anchor-provider-authority-v1",
        "status": "PASS_FIXED_PROVIDER_AUTHORITY",
        "task_id": "chips",
        "session_id": "get_potato_chips_0901_008",
        "split": "FIT",
        "selection_scope": "PER_SESSION_PER_PHYSICAL_SIDE",
        "silent_provider_mixing_forbidden": True,
        "providers_by_side": {
            side: {
                "provider_id": "PICO21",
                "wrist_index": 5,
                "eligibility_pass": True,
                "applicability_gate": "G2_PICO21_PRELIMINARY",
                "applicability_value": "PASS_NUMERIC_NOT_MANO21",
            }
            for side in ("left", "right")
        },
    }
    assert validate_anchor_provider_authority(authority, shared, plan)["status"].startswith("PASS")
    broken = deepcopy(authority)
    broken["providers_by_side"]["left"] = {
        "provider_id": "HAWOR_MANO21",
        "wrist_index": 0,
        "eligibility_pass": True,
        "numeric_gate_pass": True,
        "independent_identity_gate_pass": False,
    }
    with pytest.raises(PicoGeometryProtocolError, match="identity"):
        validate_anchor_provider_authority(broken, shared, plan)
