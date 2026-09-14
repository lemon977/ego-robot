from pathlib import Path

import pytest

from tools.run_clean_spatial12_bounded import (
    BoundedCleanError,
    tree_bytes,
    validate_spatial_contract,
)


def test_validate_spatial_contract_accepts_exact_bounded_inventory():
    frames = validate_spatial_contract(
        {
            "review_frames": list(range(12)),
            "temporal_windows": [],
            "bounded_resource_policy": {
                "spatial_frames_persisted_max": 12,
                "whole_archive_extraction_forbidden": True,
            },
        }
    )
    assert frames == list(range(12))


@pytest.mark.parametrize(
    "patch",
    [
        {"review_frames": list(range(11))},
        {"review_frames": [0] * 12},
        {"temporal_windows": [{"frames": list(range(12))}]},
        {"bounded_resource_policy": {"spatial_frames_persisted_max": 13, "whole_archive_extraction_forbidden": True}},
        {"bounded_resource_policy": {"spatial_frames_persisted_max": 12, "whole_archive_extraction_forbidden": False}},
    ],
)
def test_validate_spatial_contract_fails_closed(patch):
    value = {
        "review_frames": list(range(12)),
        "temporal_windows": [],
        "bounded_resource_policy": {
            "spatial_frames_persisted_max": 12,
            "whole_archive_extraction_forbidden": True,
        },
    }
    value.update(patch)
    with pytest.raises(BoundedCleanError):
        validate_spatial_contract(value)


def test_tree_bytes_ignores_symlinks(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"abcd")
    child = tmp_path / "child"
    child.mkdir()
    (child / "b.bin").write_bytes(b"123")
    (tmp_path / "link").symlink_to(tmp_path / "a.bin")
    assert tree_bytes(tmp_path) == 7
