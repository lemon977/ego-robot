from pathlib import Path

from tools.prepare_robot_mount_visual_proxy_input_t0 import build


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_004_fit_input_is_deterministic_and_not_executed() -> None:
    manifest = build(PROJECT_ROOT)
    assert manifest["status"] == "CPU8_INPUT_AND_ALGORITHM_SKELETON_READY_NOT_FIT"
    assert manifest["deterministic_fit_set"] == {
        "selection": "ALL_FRAMES_WITH_BOTH_HAWOR_AND_R2_SIDES_VALID",
        "frame_count": 458,
        "first": 2,
        "last": 459,
        "excluded": [0, 1],
        "hawor_valid_by_side": {"left": 460, "right": 458},
        "r2_valid_by_side": {"left": 459, "right": 458},
    }
    assert set(manifest["execution_counters"].values()) == {0}
    assert manifest["mount_provenance"] == "PROVISIONAL_MOUNT_VISUAL_ONLY"
    assert manifest["contact_infeasible"] == "UNMEASURED"


def test_input_bytes_are_bound_to_known_read_only_sources() -> None:
    manifest = build(PROJECT_ROOT)
    inputs = manifest["inputs"]
    assert inputs["hawor_2d_targets"]["sha256"] == (
        "b54650b73d1cf9827414300b5685c2bb0436869ed31d4074d2c9a91ad2891cfa"
    )
    assert inputs["r2_hand_states"]["sha256"] == (
        "a111a02fa21d246628b932bccdcaaf8d12e4f13c2db4acb6808cf4f9c8ed3fa6"
    )
