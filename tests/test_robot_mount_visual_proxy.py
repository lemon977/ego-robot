from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.robot_mount_visual_proxy import (
    CONTACT_INFEASIBLE,
    MOUNT_PROVENANCE,
    VisualFitInputContract,
    VisualMountProxyError,
    build_fit_manifest_skeleton,
    validate_session_constant_mounts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORD = {"path": "read-only-source", "bytes": 1, "sha256": "a" * 64}


def test_manifest_is_visual_only_and_binds_shared_tool_source() -> None:
    inputs = VisualFitInputContract("grap_a_cap_004", (0, 10, 20), RECORD, RECORD)
    manifest = build_fit_manifest_skeleton(PROJECT_ROOT, inputs)
    assert manifest["mount_provenance"] == MOUNT_PROVENANCE
    assert manifest["contact_infeasible"] == CONTACT_INFEASIBLE
    assert manifest["visual_only"] is True
    assert manifest["formal_consumer_allowed"] is False
    assert manifest["tool_definition"]["tool_joints"]["left"]["xyz_m"][2] == 0.145


def test_mount_must_be_one_constant_se3_per_side() -> None:
    mounts = validate_session_constant_mounts({"left": np.eye(4), "right": np.eye(4)})
    assert set(mounts) == {"left", "right"}
    with pytest.raises(VisualMountProxyError, match="session-constant"):
        validate_session_constant_mounts(
            {"left": np.repeat(np.eye(4)[None], 2, axis=0), "right": np.eye(4)}
        )


@pytest.mark.parametrize("session", ["grap_a_cap_025", "cross_session_blind"])
def test_forbidden_fit_partition_fails_closed(session: str) -> None:
    with pytest.raises(VisualMountProxyError, match="forbidden"):
        build_fit_manifest_skeleton(
            PROJECT_ROOT, VisualFitInputContract(session, (0,), RECORD, RECORD)
        )
