"""One bounded 0901 Chips013 native-25-FPS HaWoR->Robot candidate packet."""

from __future__ import annotations

from pathlib import Path

from chaoyang.governance.exact78_raw_two_source_executor_v1 import Target, main_for


TARGET = Target(
    task_id="exact78_raw_to_robot_t2_0901_chips013_v1_20260930",
    session=Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0901/potato_chips/get_potato_chips_0901_013"),
    session_id="get_potato_chips_0901_013", frame_count=870, fps_nominal=25,
    time_mode="TRACKER_CLIP_METADATA_ONLY", physical_eye="tracker_session_left_RGB",
    receipt_name="EXACT78_RAW_TO_ROBOT_T2_0901_CHIPS013_V1.json",
    execution_revision="EXACT78_RAW_T2_0901_CHIPS013_V1",
    objective="Cold-start all 870 native 25-FPS tracker RGB frames through fresh HaWoR and complete camera-diagnostic Robot",
    claim_limit="0901 Chips013 RGB-only camera diagnostic without proven world alignment; no visual/product/training PASS or metric-calibration claim",
    wall_seconds_max=21600,
)


def main() -> int:
    return main_for(TARGET, Path(__file__))


if __name__ == "__main__":
    raise SystemExit(main())
