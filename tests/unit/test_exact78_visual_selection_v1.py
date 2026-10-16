"""Synthetic route selection checks; no project state or model is loaded."""

import unittest

from chaoyang.governance.exact78_visual_selection_v1 import (
    assert_selected_submit_allowed,
    filter_selected_operations,
)


TAKEOVER = {
    "run_exact78_takeover_audit_v1",
    "run_exact78_takeover_baseline_v1",
    "run_exact78_takeover_cleanup_v1",
}
OPERATIONS = TAKEOVER | {
    "run_exact78_general_p1_action_canary_v10",
    "run_exact78_cards042_fresh_robot_v1",
    "run_0902_legacy_candidate",
    "run_robot100_world_first_v10",
    "run_0916_manus_candidate",
    "run_exact78_inpainting_robot100_v1",
    "run_unrelated_operation",
}
OTHER = {
    "run_robot100_world_first_v10",
    "run_0916_manus_candidate",
    "run_exact78_inpainting_robot100_v1",
    "run_unrelated_operation",
}


def selected_policy(**overrides):
    policy = {
        "enabled": True,
        "baseline_id": "EXACT78_VISUAL_BASELINE_V1",
        "scope": "chips_cards_tracker_0901_0902",
        "allowed_operations": sorted(TAKEOVER),
    }
    policy.update(overrides)
    return {"exact78_visual_baseline_v1": {"execution_policy": policy}}


def nested_policy(**overrides):
    return {"special_status_contracts": selected_policy(**overrides)}


class Exact78VisualSelectionTests(unittest.TestCase):
    def test_selected_route_keeps_only_takeover_and_other_branches(self):
        self.assertEqual(filter_selected_operations(OPERATIONS, nested_policy()), TAKEOVER | OTHER)
        assert_selected_submit_allowed(nested_policy(), "unrelated-recipe")
        with self.assertRaisesRegex(ValueError, "EXACT78_VISUAL_BASELINE_V1"):
            assert_selected_submit_allowed(nested_policy(), "exact78-raw-to-robot")

    def test_nested_contract_takes_priority_over_synthetic_top_level(self):
        contract = selected_policy(enabled=False) | nested_policy()
        self.assertEqual(filter_selected_operations(OPERATIONS, contract), TAKEOVER | OTHER)
        with self.assertRaises(ValueError):
            assert_selected_submit_allowed(contract, "exact78-raw-to-robot")

    def test_missing_or_disabled_policy_preserves_existing_operations_and_submit(self):
        for contract in ({}, {"exact78_visual_baseline_v1": {}},
                         selected_policy(enabled=False), nested_policy(enabled=False)):
            with self.subTest(contract=contract):
                self.assertEqual(filter_selected_operations(OPERATIONS, contract), OPERATIONS)
                assert_selected_submit_allowed(contract, "exact78-raw-to-robot")

    def test_malformed_selection_closes_exact78_only(self):
        overrides = [
            {"baseline_id": "OLD_BASELINE"},
            {"scope": "chips_cards_tracker_0915"},
            {"allowed_operations": []},
            {"allowed_operations": ["run_exact78_general_p1_action_canary_v10"]},
            {"enabled": "true"},
        ]
        for override in overrides:
            with self.subTest(override=override):
                contract = nested_policy(**override)
                self.assertEqual(filter_selected_operations(OPERATIONS, contract), OTHER)
                with self.assertRaisesRegex(ValueError, "separate authorization"):
                    assert_selected_submit_allowed(contract, "exact78-raw-to-robot")
