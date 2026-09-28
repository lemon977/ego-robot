import numpy as np

from chaoyang.ops.run_human_to_robot_shared_delivery_robot import (
    POSITION_M,
    ROTATION_RAD,
    contiguous_segments,
    optimize_block,
    rolling_blocks,
)


def test_rolling_blocks_do_not_cross_gap_or_duplicate_output():
    edge = np.array([False, True, True, False, True, True, True], dtype=bool)
    valid = np.ones(7, dtype=bool)
    segments = contiguous_segments(valid, edge)
    assert [value.tolist() for value in segments] == [[0, 1, 2], [3, 4, 5, 6]]
    blocks = [block for segment in segments for block in rolling_blocks(segment, segment)]
    outputs = np.concatenate([block.output for block in blocks])
    assert outputs.tolist() == list(range(7))
    assert len(np.unique(outputs)) == 7
    for block in blocks:
        members = set(np.concatenate((block.past, block.output, block.future)).tolist())
        assert members <= set(segments[0].tolist()) or members <= set(segments[1].tolist())


def test_rolling_16_output_and_eight_frame_context_contract():
    segment = np.arange(64, dtype=np.int64)
    blocks = rolling_blocks(segment, np.arange(16, 48))
    assert [len(block.output) for block in blocks] == [16, 16]
    assert [len(block.past) for block in blocks] == [8, 8]
    assert [len(block.future) for block in blocks] == [8, 8]
    assert np.array_equal(np.concatenate([block.output for block in blocks]), np.arange(16, 48))


def test_joint_optimizer_preserves_hard_pose_gate_and_reduces_peak():
    # A two-joint toy FK.  Pose is the distance to a frame-specific scalar
    # target; the second pose component is an independent bounded rotation.
    frames = np.arange(4, dtype=np.int64)
    targets = np.array([0.00, 0.01, 0.02, 0.03])

    def pose(frame, q):
        return abs(float(q[0] - targets[frame])), abs(float(q[1]))

    initial = np.array([[0.00, 0.00], [0.018, 0.0], [0.002, 0.0], [0.048, 0.0]])
    fixed = np.array([[-0.01, 0.0], [-0.005, 0.0]])
    solved, diagnostic = optimize_block(
        initial, fixed, frames,
        np.array([-0.2, -0.1]), np.array([0.0, 0.1, 0.2, 0.3]),
        np.array([-1.0, -1.0]), np.array([1.0, 1.0]), pose,
    )
    assert diagnostic["usable"]
    assert diagnostic["selected_stage"] in {"VELOCITY", "ACCELERATION"}
    assert diagnostic["stage1_velocity_peak_range_per_s"] <= (
        diagnostic["initial_velocity_peak_range_per_s"] + 1e-6
    )
    for frame, q in zip(frames, solved, strict=True):
        position, rotation = pose(int(frame), q)
        assert position <= POSITION_M + 1e-8
        assert rotation <= ROTATION_RAD + 1e-8


def test_invalid_input_is_never_bridged_by_segment_builder():
    edge = np.array([False, True, True, True, True], dtype=bool)
    valid = np.array([True, True, False, True, True], dtype=bool)
    segments = contiguous_segments(valid, edge)
    assert [value.tolist() for value in segments] == [[0, 1], [3, 4]]
