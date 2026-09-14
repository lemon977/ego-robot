import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "task29_context_test",
    PROJECT / "tools/prepare_002_012_full_paired_context_t1.py",
)
assert SPEC is not None and SPEC.loader is not None
context = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = context
SPEC.loader.exec_module(context)


def test_frame_counts_and_independent_expected_states_are_frozen():
    assert context.SESSIONS == {"grap_a_cap_002": 393, "grap_a_cap_012": 364}
    assert context.EXPECTED["grap_a_cap_002"]["right"]["OUTSIDE_IMAGE"] == 112
    assert context.EXPECTED["grap_a_cap_012"]["left"]["OUTSIDE_IMAGE"] == 12
    assert context.EXPECTED["grap_a_cap_012"]["right"]["OUTSIDE_IMAGE"] == 101
    assert context.EXPECTED["grap_a_cap_012"]["right"]["MISSING"] == 11


def test_intervals_are_contiguous_and_lossless():
    assert context.intervals([1, 2, 3, 7, 9, 10]) == [[1, 3], [7, 7], [9, 10]]
    assert context.intervals([]) == []


def test_oob_distance_uses_nearest_valid_pixel_boundary():
    distance, direction = context.out_of_image(np.asarray([100.0, 960.1876220703125]))
    assert distance == pytest.approx(1.1876220703125)
    assert direction == "BOTTOM"
    distance, direction = context.out_of_image(np.asarray([-3.0, 970.0]))
    assert distance == pytest.approx(np.hypot(3.0, 11.0))
    assert direction == "LEFT+BOTTOM"


def test_contiguous_frame_ids_rejects_gaps(tmp_path: Path):
    for name in ("00000", "00002"):
        (tmp_path / name).mkdir()
    with pytest.raises(context.ContextError, match="noncontiguous"):
        context.contiguous_frame_ids(tmp_path)


def test_contiguous_frame_ids_rejects_hidden_or_noncanonical_entry(tmp_path: Path):
    (tmp_path / "00000").mkdir()
    (tmp_path / "bad").mkdir()
    with pytest.raises(context.ContextError, match="noncanonical"):
        context.contiguous_frame_ids(tmp_path)


def test_identity_rejects_intermediate_symlink(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "x.json").write_text("{}")
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        context.identity(tmp_path / "link/x.json")


def test_secure_create_direct_child_rejects_symlink_parent(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "audits"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        context.secure_create_direct_child(link, "attack")
    assert not (outside / "attack").exists()


def test_secure_create_direct_child_is_oexcl(tmp_path: Path):
    parent = tmp_path / "audits"
    parent.mkdir()
    child, first = context.secure_create_direct_child(parent, "v1")
    assert child.is_dir()
    assert first["st_dev"] > 0 and first["st_ino"] > 0
    with pytest.raises(FileExistsError):
        context.secure_create_direct_child(parent, "v1")


def test_exclusive_writer_rejects_overwrite(tmp_path: Path):
    path = tmp_path / "x.bin"
    context.exclusive_bytes(path, b"first")
    with pytest.raises(FileExistsError):
        context.exclusive_bytes(path, b"second")
    assert path.read_bytes() == b"first"
