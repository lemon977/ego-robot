from __future__ import annotations

import inspect

from chaoyang.governance import current_r3_contracts
from chaoyang.governance.current_baseline_v2 import SPECS


def test_0915_campaign_is_sam31_only() -> None:
    source = inspect.getsource(current_r3_contracts.build_algorithm_contract)
    assert '"current": "SAM3.1"' in source
    assert '"challengers": []' in source
    assert '"selection_policy": "SAM3.1_ONLY_USER_LOCKED"' in source
    assert '"mask": "SAM3.1_ONLY"' in source


def test_0915_campaign_does_not_register_challenger_operations() -> None:
    source = inspect.getsource(current_r3_contracts.build_algorithm_contract)
    campaign = source.split('"processed_0915_full_funnel_v1"', 1)[1].split(
        '"handle_data_cleaning_v4_0916"', 1
    )[0]
    assert "sam2" not in campaign.lower()
    assert "cutie" not in campaign.lower()


def test_current_mask_stage_specs_bind_only_sam31() -> None:
    mask_specs = [item for item in SPECS if item["stage"] in {"Role Mask", "Object Mask"}]
    assert {item["stage"] for item in mask_specs} == {"Role Mask", "Object Mask"}
    for item in mask_specs:
        assert item["weights"] == ["assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"]
        policy = "\n".join([*item["limitations"], item["successor"]])
        assert "SAM3.1" in policy
        assert (
            "challenger task is authorized" in policy
            or "without registering an alternate model route" in policy
        )
