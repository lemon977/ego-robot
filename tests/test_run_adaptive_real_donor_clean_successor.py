from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.depth_multiview_clean_successor import SOURCE_DONOR, SOURCE_UNSUPPORTED  # noqa: E402
from tools.run_adaptive_real_donor_clean_successor import (
    _require_authority_binding,
    connected_components,
    donor_label_edge_fraction,
    output_grade,
    timeline_pool,
)  # noqa: E402


def test_timeline_pool_searches_whole_clip_and_excludes_target() -> None:
    pool = timeline_pool(146, 293, 6)
    assert 0 in pool
    assert 292 in pool
    assert 146 not in pool
    assert len(pool) >= 45


def test_connected_components_are_largest_first() -> None:
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[4:8, 5:10] = True
    components = connected_components(mask)
    assert [int(value.sum()) for value in components] == [20, 4]


def test_donor_edge_metric_ignores_donor_to_raw_boundary() -> None:
    kinds = np.full((4, 5), SOURCE_UNSUPPORTED, dtype=np.uint8)
    frames = np.full((4, 5), -1, dtype=np.int32)
    kinds[:, :4] = SOURCE_DONOR
    frames[:, :4] = 7
    assert donor_label_edge_fraction(kinds, frames) == 0.0
    frames[:, 2:4] = 9
    assert donor_label_edge_fraction(kinds, frames) > 0.0


def test_role_specific_gate_prevents_aggregate_false_pass() -> None:
    metrics = {
        "supported_fraction": 0.70,
        "left_human_supported_fraction": 0.02,
        "right_human_supported_fraction": 0.92,
        "tracker_supported_fraction": 0.95,
        "donor_label_edge_fraction": 0.01,
    }
    gates = {
        "minimum_total_supported_fraction": 0.5,
        "minimum_left_human_supported_fraction": 0.5,
        "minimum_right_human_supported_fraction": 0.75,
        "minimum_tracker_supported_fraction": 0.8,
        "maximum_donor_label_edge_fraction": 0.12,
    }
    grade, authorized, hard = output_grade({"MASK": "B"}, metrics, gates)
    assert grade == "C"
    assert authorized is False
    assert hard["left_human_real_donor_coverage"] == "FAIL"


def test_all_quality_gates_with_grade_b_upstream_yields_b() -> None:
    metrics = {
        "supported_fraction": 0.80,
        "left_human_supported_fraction": 0.70,
        "right_human_supported_fraction": 0.90,
        "tracker_supported_fraction": 0.90,
        "donor_label_edge_fraction": 0.02,
    }
    gates = {
        "minimum_total_supported_fraction": 0.5,
        "minimum_left_human_supported_fraction": 0.5,
        "minimum_right_human_supported_fraction": 0.75,
        "minimum_tracker_supported_fraction": 0.8,
        "maximum_donor_label_edge_fraction": 0.12,
    }
    grade, authorized, hard = output_grade({"MASK": "B"}, metrics, gates)
    assert grade == "B"
    assert authorized is True
    assert set(hard.values()) == {"PASS"}


def test_poker_wait_status_authorizes_exact_captured_inputs() -> None:
    inputs = {
        "depth_agent_review": {"sha256": "depth"},
        "hawor_agent_review": {"sha256": "hawor"},
        "mask_result": {"sha256": "mask-result"},
        "mask_agent_review": {"sha256": "mask-review"},
        "mask_frame_manifest": {"sha256": "mask-frames"},
        "object_mask_manifest": {"sha256": "object-mask"},
        "object6d_baseline_result": {"sha256": "object6d"},
    }
    spec = {
        "task": "poker",
        "session": "play_cards_0902_042",
        "inputs": inputs,
    }
    snapshot = {
        "schema_version": "adaptive-clean-recovery-input-snapshot-v1",
        "run_id": "20260908_two_task_e2e_baseline_v1",
        "stage": "CLEAN",
        "task": "poker",
        "session": "play_cards_0902_042",
        "status": "CONSISTENT_AT_CAPTURE",
        "cross_document_checks": {"document_or_lineage_conflict_at_capture": False},
        "captured_stage_authority": {
            "clean_status": "NO_CURRENT_CLEAN_AUTHORITY_POKER_WAIT_UPSTREAM",
            "session_clean_status": "FRESH_CLEAN_EXECUTION_FROM_CURRENT_MASK_AND_OBJECT6D_IN_PROGRESS",
            "depth_review_sha256": "depth",
            "hawor_grade": "A",
            "hawor_review_sha256": "hawor",
            "mask_grade": "B",
            "mask_downstream_authorized": True,
            "mask_result_sha256": "mask-result",
            "mask_agent_review_sha256": "mask-review",
            "mask_frame_manifest_sha256": "mask-frames",
            "object_mask_manifest_sha256": "object-mask",
            "object6d_grade": "B",
            "object6d_downstream_authorized": True,
            "object6d_result_sha256": "object6d",
        },
    }
    _require_authority_binding(spec, snapshot)


def test_poker_snapshot_rejects_non_wait_clean_status() -> None:
    spec = {
        "task": "poker",
        "session": "play_cards_0902_042",
        "inputs": {
            name: {"sha256": name}
            for name in (
                "depth_agent_review",
                "hawor_agent_review",
                "mask_result",
                "mask_agent_review",
                "mask_frame_manifest",
                "object_mask_manifest",
                "object6d_baseline_result",
            )
        },
    }
    captured = {
        "clean_status": "NO_CURRENT_CLEAN_AUTHORITY",
        "session_clean_status": "RUNNING",
        "depth_review_sha256": "depth_agent_review",
        "hawor_grade": "A",
        "hawor_review_sha256": "hawor_agent_review",
        "mask_grade": "B",
        "mask_downstream_authorized": True,
        "mask_result_sha256": "mask_result",
        "mask_agent_review_sha256": "mask_agent_review",
        "mask_frame_manifest_sha256": "mask_frame_manifest",
        "object_mask_manifest_sha256": "object_mask_manifest",
        "object6d_grade": "B",
        "object6d_downstream_authorized": True,
        "object6d_result_sha256": "object6d_baseline_result",
    }
    snapshot = {
        "schema_version": "adaptive-clean-recovery-input-snapshot-v1",
        "run_id": "20260908_two_task_e2e_baseline_v1",
        "stage": "CLEAN",
        "task": "poker",
        "session": "play_cards_0902_042",
        "status": "CONSISTENT_AT_CAPTURE",
        "cross_document_checks": {"document_or_lineage_conflict_at_capture": False},
        "captured_stage_authority": captured,
    }
    import pytest
    from tools.run_adaptive_real_donor_clean_successor import AdaptiveCleanError

    with pytest.raises(AdaptiveCleanError, match="does not authorize"):
        _require_authority_binding(spec, snapshot)
