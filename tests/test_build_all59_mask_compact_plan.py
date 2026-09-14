from tools.build_all59_mask_compact_plan import spatial_indices, temporal_windows


def test_spatial_indices_are_exact_unique_endpoints():
    values = spatial_indices(799)
    assert len(values) == len(set(values)) == 12
    assert values[0] == 0 and values[-1] == 798


def test_temporal_windows_are_three_disjoint_unit_stride_windows():
    windows = temporal_windows(645)
    assert len(windows) == 3
    used = set()
    for row in windows:
        frames = row["source_frames"]
        assert len(frames) == 12
        assert frames == list(range(frames[0], frames[0] + 12))
        assert not used.intersection(frames)
        used.update(frames)
