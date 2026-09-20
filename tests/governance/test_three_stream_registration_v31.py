from __future__ import annotations

from pathlib import Path

from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.governance.register_three_stream_stable_baseline_v31 import build_packet
from chaoyang.ops.run_three_stream_stable_baseline_v31 import _initial_lane_state


def test_parent_packet_is_single_weight_absent_and_finite() -> None:
    packet = build_packet()
    assert _validate_packet(packet) == []
    assert packet["weights"] == "ABSENT"
    assert packet["budgets"]["gpu_concurrent_owners_max"] == 1
    assert packet["budgets"]["e2_gpu_seconds_per_model_max"] == 12 * 60 * 60
    assert packet["stop_conditions"]


def test_lane_writer_roots_are_pairwise_isolated() -> None:
    roots = list(build_packet()["fencing"]["lane_writer_roots"].values())
    assert len(roots) == len(set(roots)) == 3
    resolved = [Path(root).parts for root in roots]
    assert all("attempt_0001" in parts for parts in resolved)


def test_new_lane_state_cannot_claim_quality_or_deployment() -> None:
    state = _initial_lane_state("ai1", "2026-09-20T00:00:00+08:00")
    assert state["status"] == "READY_CPU_PREFLIGHT"
    assert state["claims"]["PIPELINE_COMPLETE"] is False
    assert state["claims"]["NUMERIC_QUALITY_PASS"] is False
    assert state["claims"]["CONTROL_GROUND_TRUTH"] is False
    assert state["claims"]["PHYSICAL_DEPLOYABLE"] is False
