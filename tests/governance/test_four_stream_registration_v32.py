from __future__ import annotations

from pathlib import Path

from chaoyang.governance.register_four_stream_pretraining_baseline_v32 import build_packet
from chaoyang.governance.register_single_task_packet import _validate_packet
from chaoyang.ops.run_four_stream_pretraining_baseline_v32 import _lane_state


def test_v32_parent_is_finite_and_single_gpu_owner() -> None:
    packet = build_packet()
    assert _validate_packet(packet) == []
    assert packet["weights"] == "ABSENT"
    assert packet["budgets"]["gpu_concurrent_owners_max"] == 1
    assert packet["budgets"]["first_cycle_gpu_seconds_max"] == int(20.25 * 3600)


def test_v32_lane_writer_roots_are_isolated() -> None:
    roots = list(build_packet()["fencing"]["lane_writer_roots"].values())
    assert len(roots) == len(set(roots)) == 4
    assert all("attempt_0001" in Path(root).parts for root in roots)


def test_v32_initial_lane_state_cannot_claim_quality() -> None:
    state = _lane_state("ai4_huro", "2026-09-20T00:00:00+08:00")
    assert state["status"] == "READY_CPU_PREFLIGHT"
    assert state["claims"]["PIPELINE_COMPLETE"] is False
    assert state["claims"]["TRAINING_COMPLETE"] is False
    assert state["claims"]["CONTROL_GROUND_TRUTH"] is False
    assert state["claims"]["PHYSICAL_DEPLOYABLE"] is False
