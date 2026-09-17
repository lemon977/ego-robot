"""Frozen selector for the isolated `an arm` SAM3.1 candidate.

The selector chooses whole raw SAM instances.  It never combines or edits mask
pixels.  Object6D is a rejection/protection signal only.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_DIAGNOSTIC_PATH = Path(__file__).with_name("sam31_detailed_text_prompt_probe.py")
_SPEC = importlib.util.spec_from_file_location(
    "sam31_detailed_text_prompt_probe_for_an_arm_selector", _DIAGNOSTIC_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load raw-instance diagnostics: {_DIAGNOSTIC_PATH}")
_DIAGNOSTICS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _DIAGNOSTICS
_SPEC.loader.exec_module(_DIAGNOSTICS)
raw_instance_metrics = _DIAGNOSTICS.raw_instance_metrics


PRODUCER_ID = "sam31_an_arm_ego_select_v1"
TEXT_PROMPT = "an arm"
DEVELOPMENT_FRAMES = tuple(range(228, 243))


class AnArmSelectorContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class SelectorConfig:
    min_joint_support_ratio: float = 0.20
    min_side_margin: float = 0.05
    max_object_overlap_over_instance: float = 0.12


@dataclass(frozen=True)
class SelectedSide:
    side: str
    raw_offset: int | None
    instance_id: int | None
    mask: np.ndarray | None
    evidence: dict[str, Any] | None
    failure_reason: str | None


@dataclass(frozen=True)
class FrameSelection:
    left: SelectedSide
    right: SelectedSide
    frame_complete: bool
    failure_attribution: str | None
    raw_instance_evidence: tuple[dict[str, Any], ...]
    background_rejections: tuple[dict[str, Any], ...]
    pixels_created_or_edited: int = 0


def _side_eligible(
    evidence: dict[str, Any], side: str, config: SelectorConfig
) -> tuple[bool, list[str]]:
    other = "right" if side == "left" else "left"
    own = evidence["sides"][side]
    other_evidence = evidence["sides"][other]
    reasons = []
    if own["joint_support_ratio"] < config.min_joint_support_ratio:
        reasons.append("INSUFFICIENT_EGO_JOINT_SUPPORT")
    if not own["wrist_supported"]:
        reasons.append("EGO_WRIST_NOT_SUPPORTED")
    if not own["boundary_supported"]:
        reasons.append("EGO_BOUNDARY_NOT_SUPPORTED")
    if (
        own["joint_support_ratio"]
        < other_evidence["joint_support_ratio"] + config.min_side_margin
    ):
        reasons.append("NOT_SIDE_SPECIFIC")
    if evidence["object6d_overlap_over_instance"] > config.max_object_overlap_over_instance:
        reasons.append("OBJECT6D_PROTECTION_REJECTION")
    return not reasons, reasons


def _rank(evidence: dict[str, Any], side: str) -> tuple[float, ...]:
    own = evidence["sides"][side]
    other = "right" if side == "left" else "left"
    return (
        own["joint_support_ratio"],
        own["joint_support_ratio"] - evidence["sides"][other]["joint_support_ratio"],
        evidence["score"],
        -evidence["object6d_overlap_over_instance"],
        -evidence["area_ratio"],
        -float(evidence["raw_instance_offset"]),
    )


def select_raw_instances(
    masks: np.ndarray,
    scores: np.ndarray,
    instance_ids: np.ndarray,
    *,
    joints_by_side: np.ndarray,
    cad_object_mask: np.ndarray,
    config: SelectorConfig = SelectorConfig(),
) -> FrameSelection:
    raw_masks = np.asarray(masks, dtype=bool)
    raw_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    raw_ids = np.asarray(instance_ids).reshape(-1)
    if raw_masks.ndim != 3 or len(raw_masks) != len(raw_scores) or len(raw_masks) != len(raw_ids):
        raise AnArmSelectorContractError("raw SAM instance shape drift")
    if raw_masks.shape[1:] != np.asarray(cad_object_mask).shape:
        raise AnArmSelectorContractError("Object6D mask shape drift")

    evidence_items = []
    background_rejections = []
    eligible: dict[str, list[tuple[dict[str, Any], np.ndarray]]] = {
        "left": [],
        "right": [],
    }
    for offset, (mask, score, instance_id) in enumerate(
        zip(raw_masks, raw_scores, raw_ids, strict=True)
    ):
        evidence = raw_instance_metrics(
            mask,
            score=float(score),
            instance_id=int(instance_id),
            joints_by_side=joints_by_side,
            cad_object_mask=cad_object_mask,
        )
        evidence["raw_instance_offset"] = offset
        evidence["per_side_selector"] = {}
        any_ego_hand_wrist = False
        for side in ("left", "right"):
            own = evidence["sides"][side]
            any_ego_hand_wrist |= bool(
                own["joint_support_ratio"] >= config.min_joint_support_ratio
                and own["wrist_supported"]
            )
            is_eligible, reasons = _side_eligible(evidence, side, config)
            evidence["per_side_selector"][side] = {
                "eligible": is_eligible,
                "rejection_reasons": reasons,
            }
            if is_eligible:
                eligible[side].append((evidence, mask))
        if not any_ego_hand_wrist:
            background_rejections.append(
                {
                    "raw_instance_offset": offset,
                    "instance_id": int(instance_id),
                    "reason": "NO_AUTHORITY_EGO_HAND_WRIST_SUPPORT",
                }
            )
        evidence_items.append(evidence)

    selections = {}
    for side in ("left", "right"):
        candidates = eligible[side]
        if not candidates:
            recalled = any(
                item["sides"][side]["joint_support_ratio"]
                >= config.min_joint_support_ratio
                and item["sides"][side]["wrist_supported"]
                for item in evidence_items
            )
            reason = (
                "SELECTION_RULE_REJECTED_RECALLED_INSTANCE"
                if recalled
                else "PROMPT_RECALL_INSUFFICIENT"
            )
            selections[side] = SelectedSide(side, None, None, None, None, reason)
            continue
        evidence, mask = max(candidates, key=lambda item: _rank(item[0], side))
        selections[side] = SelectedSide(
            side,
            int(evidence["raw_instance_offset"]),
            int(evidence["instance_id"]),
            mask.copy(),
            evidence,
            None,
        )

    complete = selections["left"].mask is not None and selections["right"].mask is not None
    if complete:
        attribution = None
    else:
        reasons = {selections[side].failure_reason for side in ("left", "right")}
        reasons.discard(None)
        attribution = "+".join(sorted(reasons))
    return FrameSelection(
        left=selections["left"],
        right=selections["right"],
        frame_complete=complete,
        failure_attribution=attribution,
        raw_instance_evidence=tuple(evidence_items),
        background_rejections=tuple(background_rejections),
    )
