"""One bounded 0915 Cards001 raw physical-left HaWoR->Robot candidate packet."""

from __future__ import annotations

from pathlib import Path

from chaoyang.governance.exact78_raw_two_source_executor_v1 import Target, main_for


TARGET = Target(
    task_id="exact78_raw_to_robot_t2_0915_cards001_v1_20260930",
    session=Path("/path/to/private_data/egodata/datasets/ego/chips_cards_hands_0915/cards_120_0915/001"),
    session_id="play_cards_0915_001", frame_count=150, fps_nominal=30,
    time_mode="TIMEBASE_UNVERIFIED", physical_eye="left_sourceIndex_1",
    receipt_name="EXACT78_RAW_TO_ROBOT_T2_0915_CARDS001_V1.json",
    execution_revision="EXACT78_RAW_T2_0915_CARDS001_V1",
    objective="Cold-start all 150 raw VST physical-left frames through fresh RGB HaWoR and complete camera-diagnostic Robot",
    claim_limit="0915 Cards001 RGB-only camera diagnostic with unverified QPC/Head clock; no visual/product/training PASS or world pose claim",
    wall_seconds_max=14400,
)


def main() -> int:
    return main_for(TARGET, Path(__file__))


if __name__ == "__main__":
    raise SystemExit(main())
