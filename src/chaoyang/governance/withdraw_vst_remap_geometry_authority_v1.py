"""Withdraw current geometry authority derived from lens-remapped VST video.

Decoded VST pixels are already undistorted.  Historical files remain immutable,
but results produced after applying ``equiDis62`` to those pixels cannot retain
current Depth, Object6D, Contact, or Occlusion consumption authority.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    artifact_ref,
    load_json,
    publish_bundle,
)


CONFIRMATION_PATH = (
    REPO_ROOT / "tasks/receipts/VST_ENCODED_VIDEO_NO_LENS_UNDISTORTION_V1.json"
)
WRONG_DOMAIN_WORKER_PATH = (
    REPO_ROOT
    / "src/chaoyang/ops/run_exact78_foundationstereo_corrected_depth_worker.py"
)

WITHDRAWN_CLAIMS = frozenset(
    {
        (
            "FoundationStereo disparity-to-depth formula and registration chain are "
            "internally closed for the current visual Depth products.",
            "exact78 metric-ready Depth",
        ),
        (
            "Chips034 right hand shows a persistent negative HaWoR-versus-Stereo Z "
            "discrepancy.",
            "get_potato_chips_0902_034/right",
        ),
        (
            "The Chips034 discrepancy is primarily a HaWoR absolute-Z placement error.",
            "get_potato_chips_0902_034/right",
        ),
        (
            "Poker042 can produce a hypothesis-only human-contact sidecar without "
            "waiting for Clean or Robot rendering.",
            "play_cards_0902_042/human-contact-input-gate",
        ),
        (
            "Poker042 has a full-session human-contact hypothesis sidecar while formal "
            "Object6D remains unchanged.",
            "play_cards_0902_042/HUMAN_CONTACT_HYPOTHESIS_V1",
        ),
        (
            "087真实会话已闭合Robot optical-Z与Stereo可见物体表面的4帧前后关系canary；"
            "接触窄带已知覆盖81.47%，UNKNOWN 18.53%。",
            "get_potato_chips_0902_087_visible_surface_occlusion_canary",
        ),
        (
            "Occlusion compositor只在Robot/物体真实重叠区要求Stereo排序后，087连续接触窗的"
            "物体条件保留率中位数由77.48%提高到95.92%，但仍未达到99% Silver门。",
            "get_potato_chips_0902_087_occlusion_visible_object_retention_successor",
        ),
        (
            "087连续24帧可见表面z-buffer successor的物体条件保留率按像素加权为97.01%，"
            "已消除非重叠物体被深度门误降背景的问题，但仍未达到99% Silver门。",
            "get_potato_chips_0902_087_occlusion_visible_surface_24frame_v4",
        ),
        (
            "Chips087连续24帧可见表面Robot/Object光学Z诊断在边缘邻域一致性约束后达到"
            "99.72%条件物体像素保留率，known coverage 84.12%，UNKNOWN 15.88%。",
            "get_potato_chips_0902_087/frames_92_115/visible_surface_occlusion_R7_2",
        ),
    }
)

WITHDRAWAL_LIMIT = (
    "Withdrawn from current consumption: the supporting Stereo/Object6D evidence was "
    "derived after applying forbidden lens remapping to already-undistorted VST pixels."
)


def _append_evidence(rows: list[dict[str, Any]], reference: Mapping[str, Any]) -> None:
    by_path = {str(item["path"]): dict(item) for item in rows}
    by_path[str(reference["path"])] = dict(reference)
    rows[:] = [by_path[path] for path in sorted(by_path)]


def _stage(authority: Mapping[str, Any], name: str) -> dict[str, Any]:
    matches = [item for item in authority.get("stages", []) if item.get("stage") == name]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name!r} stage, found {len(matches)}")
    return matches[0]


def validate_confirmation(
    confirmation: Mapping[str, Any], worker_reference: Mapping[str, Any]
) -> None:
    if confirmation.get("status") != "CONFIRMED_ENCODED_VIDEO_ALREADY_UNDISTORTED":
        raise RuntimeError("VST encoded-video confirmation status is not authoritative")
    if confirmation.get("scope") != "ALL_DECODED_VST_VIDEO_PIXELS_CONSUMED_BY_CURRENT_PIPELINE":
        raise RuntimeError("VST encoded-video confirmation scope is incomplete")
    if confirmation.get("admitted_visual_transform") != "SOURCE_INDEX_CROP_THEN_RESIZE_ONLY":
        raise RuntimeError("VST admitted transform is not crop-then-resize-only")
    expected = confirmation.get("withdrawn_current_authority", {}).get("offending_worker")
    if expected != dict(worker_reference):
        raise RuntimeError("wrong-domain worker artifact does not match confirmation receipt")


def withdraw_remap_geometry_authority(
    authority: Mapping[str, Any], confirmation_reference: Mapping[str, Any]
) -> dict[str, Any]:
    """Return an idempotently withdrawn authority ledger."""

    updated = copy.deepcopy(authority)

    depth = _stage(updated, "Depth")
    depth.update(
        total=58,
        passed=0,
        grade_c=0,
        running=0,
        blocked=58,
        authority_scope="NO_CURRENT_DEPTH_AUTHORITY_WRONG_VST_IMAGE_DOMAIN",
        denominator_semantics=(
            "The historical 58 remap-derived terminals are retained as files but all "
            "58 are blocked from current consumption because decoded VST video was "
            "lens-undistorted a second time."
        ),
    )
    _append_evidence(depth.setdefault("evidence", []), confirmation_reference)

    object6d = _stage(updated, "Object6D")
    object6d.update(
        total=58,
        passed=0,
        grade_c=0,
        running=0,
        blocked=58,
        authority_scope=(
            "NO_CURRENT_OBJECT6D_AUTHORITY_BLOCKED_UPSTREAM_DEPTH_WRONG_VST_IMAGE_DOMAIN"
        ),
        denominator_semantics=(
            "The historical 58 Object6D terminals depended on withdrawn remap-derived "
            "Depth and therefore have no current downstream authority."
        ),
    )
    _append_evidence(object6d.setdefault("evidence", []), confirmation_reference)

    contact = _stage(updated, "Contact")
    contact.update(
        authority_scope="NO_CURRENT_CONTACT_AUTHORITY_UPSTREAM_OBJECT6D_WITHDRAWN",
        passed=0,
        running=0,
        blocked=int(contact.get("total", 0)),
    )
    contact["denominator_semantics"] = (
        "The two historical hypotheses are blocked from current consumption because "
        "their Object6D/Stereo geometry input has been withdrawn; neither was contact truth."
    )
    _append_evidence(contact.setdefault("evidence", []), confirmation_reference)

    seen: set[tuple[str, str]] = set()
    for claim in updated.get("claims", []):
        key = (str(claim.get("claim")), str(claim.get("scope")))
        if key not in WITHDRAWN_CLAIMS:
            continue
        claim["status"] = "WITHDRAWN"
        claim["claim_limit"] = WITHDRAWAL_LIMIT
        _append_evidence(claim.setdefault("evidence", []), confirmation_reference)
        seen.add(key)
    missing = WITHDRAWN_CLAIMS - seen
    if missing:
        raise RuntimeError(f"withdrawal claim set is incomplete: {sorted(missing)!r}")

    old_decision = "Formal Object6D remains DIRECT_OBSERVED_ONLY and KEEP_INVALID."
    new_decision = (
        "Formal Object6D remains DIRECT_OBSERVED_ONLY and KEEP_INVALID; current authority "
        "is blocked until zero-lens-undistortion encoded-domain Depth is admitted."
    )
    decisions = list(updated.get("decisions", []))
    if old_decision in decisions:
        decisions[decisions.index(old_decision)] = new_decision
    elif new_decision not in decisions:
        raise RuntimeError("Object6D authority decision is missing")
    updated["decisions"] = decisions
    return updated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", type=int, required=True)
    args = parser.parse_args()

    current_receipt = load_json(RECEIPT_PATH)
    if int(current_receipt["governance_revision"]) != args.expected_revision:
        raise RuntimeError(
            "CAS revision mismatch before withdrawal: "
            f"expected={args.expected_revision} "
            f"current={current_receipt['governance_revision']}"
        )
    confirmation = load_json(CONFIRMATION_PATH)
    worker_reference = artifact_ref(WRONG_DOMAIN_WORKER_PATH)
    validate_confirmation(confirmation, worker_reference)
    confirmation_reference = artifact_ref(CONFIRMATION_PATH)

    authority = withdraw_remap_geometry_authority(
        load_json(AUTHORITY_PATH), confirmation_reference
    )
    published = publish_bundle(
        authority,
        load_json(TASK_STATE_PATH),
        event_type="VST_REMAP_GEOMETRY_AUTHORITY_WITHDRAWN_V1",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
    )
    print(
        json.dumps(
            {
                "status": "PUBLISHED",
                "governance_revision": published["governance_revision"],
                "generation_id": published["generation_id"],
                "depth_authorized": 0,
                "object6d_authorized": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
