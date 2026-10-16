"""T0 0901 Cards001 holdout: separate packet; results cannot tune the recipe."""

from __future__ import annotations

from pathlib import Path

from chaoyang.governance.common import REPO_ROOT, artifact_ref
from chaoyang.governance.exact78_raw_two_source_executor_v1 import Target, main_for


TARGET = Target(
    task_id="exact78_raw_to_robot_t2_0901_cards001_holdout_v1_20260930",
    session=Path("/path/to/private_data/egodata/datasets/ego/chips_cards_tracker_0901/playing_cards/play_cards_0901_001"),
    session_id="play_cards_0901_001", frame_count=645, fps_nominal=30,
    time_mode="TRACKER_CLIP_METADATA_ONLY", physical_eye="tracker_session_left_RGB",
    receipt_name="EXACT78_RAW_TO_ROBOT_T2_0901_CARDS001_HOLDOUT_V1.json",
    execution_revision="EXACT78_RAW_T2_0901_CARDS001_HOLDOUT_V1",
    objective="Evaluate preselected cross-group 0901 Cards001 holdout with the unchanged six-capture raw recipe",
    claim_limit="Preselected cross-group holdout, not wholly unseen blind test; no tuning from its result and no visual/product/training qualification",
    wall_seconds_max=21600,
)
ROLE = "holdout"
RECIPE_SHA = {
    "src/chaoyang/ops/run_exact78_raw_to_robot_v3.py": "c95bb7b6aca25555fec236339f0c8da9d409742748cc1e5b47a6ff4309e0844f",
    "src/chaoyang/ops/run_exact78_hawor_raw_v3.py": "8f116ee85817ab11e013772e8b14cbe9c8c9ec51c94e3cf8f6a41fa3322eef47",
    "src/chaoyang/pipeline/exact78_fresh_robot_stage_v2.py": "1876c5af135bd97d9e7227b076b1879ce402dc47704c260c15a6b3b0220072c9",
    "src/chaoyang/governance/exact78_raw_two_source_executor_v1.py": "cd1c618fa2d7cbf8fcf212ffd444a5c8cad2ea8c9f098b55c18a3d38e3616121",
}


def main() -> int:
    from chaoyang.ops.run_exact78_raw_to_robot_v3 import TARGETS
    if TARGETS[TARGET.session_id]["role"] != ROLE:
        raise RuntimeError("T0_SOURCE_ROLE_DRIFT")
    for relative, sha in RECIPE_SHA.items():
        if artifact_ref(REPO_ROOT / relative)["sha256"] != sha:
            raise RuntimeError(f"FROZEN_RECIPE_SHA_DRIFT:{relative}")
    return main_for(TARGET, Path(__file__))


if __name__ == "__main__":
    raise SystemExit(main())
