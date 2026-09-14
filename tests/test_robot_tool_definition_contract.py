from __future__ import annotations

import copy
from pathlib import Path

import pytest

from pipeline.robot_tool_definition_contract import (
    ToolDefinitionError,
    load_pinned_tool_definition,
    read_regular_bytes,
    require_same_tool_definition,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_current_pin_yields_one_shared_145mm_definition() -> None:
    source = load_pinned_tool_definition(PROJECT_ROOT)
    record = source.as_manifest()
    assert source.left.xyz_m == (0.0, 0.0, 0.145)
    assert source.right.xyz_m == (0.0, -0.0, 0.145)
    assert source.urdf_sha256 == "3c3bdfa9aa397c55dea2b3bc94d42c3081d4292d041b1ff43573d175d5faf309"
    assert record["external_mjcf_consumed"] is False


def test_three_way_equality_requires_all_equal_records() -> None:
    record = load_pinned_tool_definition(PROJECT_ROOT).as_manifest()
    require_same_tool_definition(record, copy.deepcopy(record), copy.deepcopy(record))
    drifted = copy.deepcopy(record)
    drifted["tool_joints"]["left"]["xyz_m"] = (0.0, 0.0, 0.05)
    with pytest.raises(ToolDefinitionError, match="disagree"):
        require_same_tool_definition(record, drifted, record)
    with pytest.raises(ToolDefinitionError, match="all required"):
        require_same_tool_definition(record, record)


def test_no_follow_reader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"tool")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ToolDefinitionError, match="securely open"):
        read_regular_bytes(link)
