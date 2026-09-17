from chaoyang.pipeline.hawor_causal_tail_inference import tail_indices


def test_tail_indices_never_use_future() -> None:
    for target in range(40):
        values = tail_indices(target)
        assert len(values) == 16
        assert max(values) == target
        assert all(value <= target for value in values)


def test_short_prefix_left_pads_first_frame() -> None:
    assert tail_indices(0) == [0] * 16
    assert tail_indices(2) == [0] * 13 + [0, 1, 2]


def test_segment_boundary_is_not_crossed() -> None:
    assert tail_indices(10, segment_start=10) == [10] * 16
    assert tail_indices(12, segment_start=10) == [10] * 13 + [10, 11, 12]
