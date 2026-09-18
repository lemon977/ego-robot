from __future__ import annotations

import json
from pathlib import Path

from chaoyang.governance.withdraw_vst_remap_geometry_authority_v1 import (
    WITHDRAWN_CLAIMS,
    withdraw_remap_geometry_authority,
)


ROOT = Path("/mnt/workspace/code/chaoyang")


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _stage(authority: dict, name: str) -> dict:
    return next(row for row in authority["stages"] if row["stage"] == name)


def test_withdrawal_blocks_depth_object6d_and_contact_consumption() -> None:
    authority = _load("docs/governance/CURRENT_AUTHORITY_INDEX.json")
    confirmation = {
        "path": str(
            ROOT / "tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json"
        ),
        "bytes": 1,
        "sha256": "a" * 64,
    }
    withdrawn = withdraw_remap_geometry_authority(authority, confirmation)

    depth = _stage(withdrawn, "Depth")
    assert depth["passed"] == 0
    assert depth["blocked"] == depth["total"] == 58
    assert depth["authority_scope"] == "NO_CURRENT_DEPTH_AUTHORITY_WRONG_VST_IMAGE_DOMAIN"

    object6d = _stage(withdrawn, "Object6D")
    assert object6d["passed"] == 0
    assert object6d["blocked"] == object6d["total"] == 58
    assert "BLOCKED_UPSTREAM_DEPTH" in object6d["authority_scope"]

    contact = _stage(withdrawn, "Contact")
    assert contact["passed"] == 0
    assert contact["blocked"] == contact["total"]
    assert "OBJECT6D_WITHDRAWN" in contact["authority_scope"]

    claims = {
        (row["claim"], row["scope"]): row for row in withdrawn["claims"]
    }
    assert all(claims[key]["status"] == "WITHDRAWN" for key in WITHDRAWN_CLAIMS)


def test_withdrawal_is_idempotent() -> None:
    authority = _load("docs/governance/CURRENT_AUTHORITY_INDEX.json")
    confirmation = {
        "path": str(
            ROOT / "tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json"
        ),
        "bytes": 1,
        "sha256": "a" * 64,
    }
    once = withdraw_remap_geometry_authority(authority, confirmation)
    twice = withdraw_remap_geometry_authority(once, confirmation)
    assert twice == once
