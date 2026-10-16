"""One candidate: unknown generic objects do not supply POINT exclusion prompts.

Pure, source-agnostic bundle transformation. No model, path, projection, new
positive support, identity promotion, threshold change, or anchor selection.
"""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

BUNDLE_SCHEMA = "inpainting-automatic-v1-rgb-seed-bundle-v1"
POLICY_KEY = "generic_object_negative_policy"


class NeutralObjectPromptError(ValueError):
    """A prompt bundle cannot be transformed without exact source alignment."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise NeutralObjectPromptError(code)


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def neutralize_generic_object_negatives(bundle: dict) -> dict:
    """Deep-copy a V1 bundle and remove only object-sourced label-0 points.

    Negative sources align with label-0 points in encounter order, including
    when positive and negative points are interleaved. Existing non-point
    fields are preserved verbatim. A pre-existing policy receipt is rejected
    so repeat application cannot overwrite the original removal audit.
    """
    _require(isinstance(bundle, dict) and bundle.get("schema") == BUNDLE_SCHEMA,
             "NEUTRAL_OBJECT_AUTOMATIC_BUNDLE_REQUIRED")
    _require(POLICY_KEY not in bundle, "NEUTRAL_OBJECT_POLICY_ALREADY_APPLIED")
    _require(isinstance(bundle.get("objects"), list), "NEUTRAL_OBJECT_OBJECTS_REQUIRED")
    result = deepcopy(bundle)
    removed = []
    positive_before = negative_before = negative_after = 0
    seen = set()
    for obj in result["objects"]:
        _require(isinstance(obj, dict), "NEUTRAL_OBJECT_OBJECT_MAPPING_REQUIRED")
        obj_id, frame = obj.get("obj_id"), obj.get("frame_idx")
        _require(_nonnegative_int(obj_id) and obj_id > 0 and obj_id not in seen,
                 "NEUTRAL_OBJECT_UNIQUE_POSITIVE_OBJ_ID_REQUIRED")
        _require(_nonnegative_int(frame), "NEUTRAL_OBJECT_NATIVE_ANCHOR_REQUIRED")
        seen.add(obj_id)
        points = obj.get("points_px")
        labels = obj.get("point_labels")
        sources = obj.get("negative_proposal_sources")
        _require(isinstance(points, list) and isinstance(labels, list)
                 and len(points) == len(labels), "NEUTRAL_OBJECT_POINT_LABEL_ALIGNMENT")
        _require(all(type(label) is int and label in (0, 1) for label in labels),
                 "NEUTRAL_OBJECT_BINARY_INTEGER_LABELS_REQUIRED")
        _require(1 in labels, "NEUTRAL_OBJECT_EXISTING_POSITIVE_REQUIRED")
        _require(isinstance(sources, list) and len(sources) == labels.count(0),
                 "NEUTRAL_OBJECT_NEGATIVE_SOURCE_ALIGNMENT")
        for point in points:
            _require(isinstance(point, list) and len(point) == 2
                     and all(type(v) in (int, float) and isfinite(v) for v in point),
                     "NEUTRAL_OBJECT_FINITE_XY_REQUIRED")
        for source in sources:
            _require(isinstance(source, dict) and isinstance(source.get("role"), str)
                     and bool(source["role"])
                     and _nonnegative_int(source.get("track_id"))
                     and isinstance(source.get("source"), str) and bool(source["source"]),
                     "NEUTRAL_OBJECT_SOURCE_IDENTITY_REQUIRED")
        kept_points, kept_labels, kept_sources = [], [], []
        source_index = 0
        for point_index, (point, label) in enumerate(zip(points, labels)):
            if label == 1:
                positive_before += 1
            else:
                source = sources[source_index]
                negative_before += 1
                if source["role"] == "object":
                    removed.append({
                        "obj_id": obj_id, "frame_idx": frame,
                        "point_index_original": point_index,
                        "negative_source_index_original": source_index,
                        "point_px": deepcopy(point),
                        "negative_proposal_source": deepcopy(source),
                    })
                    source_index += 1
                    continue
                kept_sources.append(source)
                negative_after += 1
                source_index += 1
            kept_points.append(point)
            kept_labels.append(label)
        obj["points_px"] = kept_points
        obj["point_labels"] = kept_labels
        obj["negative_proposal_sources"] = kept_sources
    result[POLICY_KEY] = {
        "schema": "inpainting-neutral-object-negative-policy-v1",
        "policy": "OMIT_GENERIC_OBJECT_EXCLUSION_PROMPTS",
        "generic_object_identity": "UNKNOWN_NOT_DEVICE",
        "point_scope": "ANCHOR_POINTS_ONLY",
        "device_promotion": False,
        "threshold_changes": False,
        "positive_points_unchanged": True,
        "anchors_unchanged": True,
        "instance_ids_and_order_unchanged": True,
        "non_object_negatives_unchanged": True,
        "input_bindings_unchanged": True,
        "counts": {
            "objects": len(result["objects"]),
            "positive_before": positive_before,
            "positive_after": sum(obj["point_labels"].count(1) for obj in result["objects"]),
            "negative_before": negative_before,
            "negative_after": negative_after,
            "generic_object_negatives_removed": len(removed),
        },
        "removed_negatives": removed,
        "physical_segmentation_ground_truth": False,
        "human_review": "PENDING",
    }
    return result
