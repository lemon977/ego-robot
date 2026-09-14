"""Fail-closed adapter for the pinned official SAM 3.1 box-prompt API.

This is a separate finite adapter, not a modification of the official source or
of ``sam31_compat_adapter_v1``.  It exposes only start, exactly two positive
normalized XYWH boxes on one frame, and close.  It has no text/point fallback.
"""

from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any, Mapping

from pipeline.sam31_compat_adapter_v1 import (
    Sam31PinnedIdentity,
    validate_pinned_signatures,
)


ADAPTER_ID = "sam31_geometric_box_adapter_v1"
FIXED_OUTPUT_PROB_THRESHOLD = 0.5
FIXED_BOX_COUNT = 2
FIXED_BOX_LABELS = (1, 1)


class Sam31GeometricBoxContractError(RuntimeError):
    """Raised for request, identity, signature, or prompt-contract drift."""


def _validate_request_keys(
    request: Mapping[str, Any], required: set[str], optional: set[str]
) -> None:
    keys = set(request)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise Sam31GeometricBoxContractError(
            f"request missing keys: {sorted(missing)}"
        )
    if unknown:
        raise Sam31GeometricBoxContractError(
            f"request has unknown keys: {sorted(unknown)}"
        )


def _strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise Sam31GeometricBoxContractError(f"{name} must be bool")
    return value


def validate_two_positive_boxes(boxes: Any, labels: Any) -> list[list[float]]:
    """Return validated normalized XYWH boxes without changing coordinates."""
    if not isinstance(boxes, (list, tuple)) or len(boxes) != FIXED_BOX_COUNT:
        raise Sam31GeometricBoxContractError("exactly two boxes are required")
    if not isinstance(labels, (list, tuple)) or tuple(labels) != FIXED_BOX_LABELS:
        raise Sam31GeometricBoxContractError("box labels must be exactly [1, 1]")
    normalized: list[list[float]] = []
    for index, box in enumerate(boxes):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise Sam31GeometricBoxContractError(
                f"box {index} must be [xmin,ymin,width,height]"
            )
        if any(type(value) not in (int, float) for value in box):
            raise Sam31GeometricBoxContractError(
                f"box {index} coordinates must be numeric"
            )
        xmin, ymin, width, height = map(float, box)
        if not all(math.isfinite(v) for v in (xmin, ymin, width, height)):
            raise Sam31GeometricBoxContractError(
                f"box {index} coordinates must be finite"
            )
        if xmin < 0 or ymin < 0 or width <= 0 or height <= 0:
            raise Sam31GeometricBoxContractError(
                f"box {index} must have nonnegative origin and positive extent"
            )
        if xmin > 1 or ymin > 1 or xmin + width > 1 or ymin + height > 1:
            raise Sam31GeometricBoxContractError(
                f"box {index} must remain inside normalized image bounds"
            )
        normalized.append([xmin, ymin, width, height])
    return normalized


class Sam31GeometricBoxAdapterV1:
    """Finite box-prompt router over the pinned official multiplex model."""

    def __init__(
        self,
        predictor: Any,
        *,
        identity: Sam31PinnedIdentity = Sam31PinnedIdentity(),
    ) -> None:
        if identity != Sam31PinnedIdentity():
            raise Sam31GeometricBoxContractError("pinned identity drift")
        if identity.use_rope_real is not False or identity.strict_state_dict_load is not True:
            raise Sam31GeometricBoxContractError("strict model identity drift")
        try:
            self.signature_evidence = validate_pinned_signatures(predictor)
        except Exception as exc:
            raise Sam31GeometricBoxContractError(str(exc)) from exc
        self.predictor = predictor
        self.model = predictor.model
        self.identity = identity
        self._sessions: dict[str, dict[str, Any]] = {}

    def handle_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        if request_type == "start_session":
            _validate_request_keys(
                request,
                {"type", "resource_path"},
                {
                    "session_id",
                    "offload_video_to_cpu",
                    "offload_state_to_cpu",
                    "async_loading_frames",
                },
            )
            resource_path = request["resource_path"]
            if not isinstance(resource_path, str) or not resource_path:
                raise Sam31GeometricBoxContractError(
                    "resource_path must be non-empty str"
                )
            if not Path(resource_path).is_absolute():
                raise Sam31GeometricBoxContractError(
                    "resource_path must be an absolute isolated input path"
                )
            offload_video = _strict_bool(
                request.get("offload_video_to_cpu", False),
                "offload_video_to_cpu",
            )
            offload_state = _strict_bool(
                request.get("offload_state_to_cpu", False),
                "offload_state_to_cpu",
            )
            async_loading = _strict_bool(
                request.get("async_loading_frames", False),
                "async_loading_frames",
            )
            if offload_state:
                raise Sam31GeometricBoxContractError(
                    "offload_state_to_cpu=true has no pinned multiplex equivalent"
                )
            session_id = request.get("session_id") or str(uuid.uuid4())
            if not isinstance(session_id, str) or not session_id:
                raise Sam31GeometricBoxContractError(
                    "session_id must be non-empty str"
                )
            if session_id in self._sessions:
                raise Sam31GeometricBoxContractError("session_id already exists")
            state = self.model.init_state(
                resource_path=resource_path,
                offload_video_to_cpu=offload_video,
                async_loading_frames=async_loading,
            )
            self._sessions[session_id] = state
            return {
                "session_id": session_id,
                "adapter_id": ADAPTER_ID,
                "parameter_mapping": {
                    "resource_path": "resource_path",
                    "offload_video_to_cpu": "offload_video_to_cpu",
                    "async_loading_frames": "async_loading_frames",
                    "offload_state_to_cpu": "validated_false_no_model_argument",
                },
            }
        if request_type == "add_geometric_box_prompt":
            _validate_request_keys(
                request,
                {
                    "type",
                    "session_id",
                    "frame_index",
                    "boxes_xywh",
                    "box_labels",
                },
                {"output_prob_thresh"},
            )
            state = self._session(request["session_id"])
            frame_index = request["frame_index"]
            if not isinstance(frame_index, int) or frame_index < 0:
                raise Sam31GeometricBoxContractError(
                    "frame_index must be non-negative int"
                )
            threshold = request.get(
                "output_prob_thresh", FIXED_OUTPUT_PROB_THRESHOLD
            )
            if type(threshold) not in (int, float) or not math.isfinite(threshold):
                raise Sam31GeometricBoxContractError(
                    "output_prob_thresh must be finite numeric"
                )
            if float(threshold) != FIXED_OUTPUT_PROB_THRESHOLD:
                raise Sam31GeometricBoxContractError(
                    "output_prob_thresh must remain fixed at 0.5"
                )
            boxes = validate_two_positive_boxes(
                request["boxes_xywh"], request["box_labels"]
            )
            frame_index, outputs = self.model.add_prompt(
                inference_state=state,
                frame_idx=frame_index,
                text_str=None,
                clear_old_points=True,
                points=None,
                point_labels=None,
                boxes_xywh=boxes,
                box_labels=list(FIXED_BOX_LABELS),
                clear_old_boxes=True,
                output_prob_thresh=FIXED_OUTPUT_PROB_THRESHOLD,
                obj_id=None,
                rel_coordinates=True,
            )
            return {
                "frame_index": frame_index,
                "outputs": outputs,
                "adapter_id": ADAPTER_ID,
                "prompt_type": "official_sam31_positive_boxes",
                "boxes_xywh": boxes,
                "box_labels": list(FIXED_BOX_LABELS),
                "output_prob_thresh": FIXED_OUTPUT_PROB_THRESHOLD,
            }
        if request_type == "close_session":
            _validate_request_keys(request, {"type", "session_id"}, set())
            session_id = request["session_id"]
            state = self._sessions.pop(session_id, None)
            if state is None:
                raise Sam31GeometricBoxContractError("unknown session_id")
            if isinstance(state, dict):
                state.clear()
            return {"is_success": True}
        raise Sam31GeometricBoxContractError(
            f"unsupported request type: {request_type!r}"
        )

    def _session(self, session_id: Any) -> dict[str, Any]:
        if not isinstance(session_id, str) or not session_id:
            raise Sam31GeometricBoxContractError(
                "session_id must be non-empty str"
            )
        state = self._sessions.get(session_id)
        if state is None:
            raise Sam31GeometricBoxContractError("unknown session_id")
        return state

