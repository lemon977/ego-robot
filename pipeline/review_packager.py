"""Fail-closed RAW/MASK/CLEAN review media interface.

Synthetic dry-runs are explicitly non-publishable. Formal candidate packaging
requires a digest-verified independent MASK QA artifact and a schema capable of
recording that reference; the current schema cannot, so formal packaging stays
blocked without writing media.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from .reveal_labels import (
    EvidenceIntegrityError,
    EvidenceRef,
    _bool_plane,
    _rgb_plane,
    load_verified_reveal_policy,
)


REVIEW_LAYOUT = ("RAW", "H_O_U_OVERLAY", "CLEAN", "DONOR_COVERAGE_DIFFERENCE")
BLOCKED_STAGES = (
    "OBJECT6D_REFINEMENT",
    "RETARGET",
    "BASE_IK",
    "RENDERER",
    "COMPOSITOR",
    "HARMONIZER",
    "FINAL_ROBOT",
)


class FormalReviewPackageBlocked(RuntimeError):
    """A required authorization/binding cannot be represented or verified."""


class ReviewMediaSink(Protocol):
    def write_stress_frame(self, frame_index: int, four_panel_rgb: np.ndarray) -> EvidenceRef:
        ...

    def write_full_preview(
        self, frames: Iterable[np.ndarray], *, fps: float, layout: Sequence[str]
    ) -> EvidenceRef:
        ...


@dataclass(frozen=True)
class ReviewFrameInput:
    frame_index: int
    timestamp_ns: int
    video_time_s: float
    raw_rgb: np.ndarray
    overlay_rgb: np.ndarray
    clean_rgb: np.ndarray
    donor_diagnostic_rgb: np.ndarray
    unsupported_mask: np.ndarray
    route_ids: tuple[str, ...]
    coverage: Mapping[str, float | int]
    hold: bool
    raw_ref: EvidenceRef
    overlay_ref: EvidenceRef
    clean_ref: EvidenceRef
    donor_diagnostic_ref: EvidenceRef
    strata: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReviewArtifactInputs:
    source_manifest_ref: EvidenceRef
    canary_set_ref: EvidenceRef
    mask_evidence_ref: EvidenceRef
    clean_layers_ref: EvidenceRef
    clean_coverage_ref: EvidenceRef
    task_card_ref: EvidenceRef
    calibration_overlay_ref: EvidenceRef
    reveal_policy_ref: EvidenceRef


@dataclass(frozen=True)
class FormalReviewAuthorization:
    independent_mask_qa_ref: EvidenceRef


def _assert_scope(ref: EvidenceRef, *, staging_required: bool) -> None:
    normalized = ref.path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if "processed" in parts:
        raise EvidenceIntegrityError(f"processed artifact is forbidden in G2 review: {ref.path}")
    if staging_required and "_run" not in parts:
        raise EvidenceIntegrityError(f"candidate review artifact is outside _run: {ref.path}")


def _verify_artifact_json(
    ref: EvidenceRef,
    *,
    schema_version: str,
    producer: str,
    session_id: str,
    product_line: str,
    artifact_state: str,
) -> Mapping[str, object]:
    if ref.schema_version != schema_version or ref.producer != producer:
        raise EvidenceIntegrityError("artifact EvidenceRef producer/schema mismatch")
    payload = ref.verified_json()
    expected = {
        "schema_version": schema_version,
        "producer": producer,
        "session_id": session_id,
        "product_line": product_line,
        "artifact_state": artifact_state,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise EvidenceIntegrityError(f"artifact binding mismatch: {name}")
    return payload


def _verify_source_manifest(
    ref: EvidenceRef,
    *,
    session_id: str,
    frame_count: int,
    fps: float,
    formal: bool,
) -> None:
    if ref.producer != "source_resolver" or ref.schema_version != "verified-source-manifest-v1":
        raise EvidenceIntegrityError("source manifest EvidenceRef producer/schema mismatch")
    payload = ref.verified_json()
    if payload.get("schema_version") != "verified-source-manifest-v1":
        raise EvidenceIntegrityError("source manifest schema mismatch")
    expected_status = "VERIFIED_G0_CALIBRATION_INPUT" if formal else "SYNTHETIC_TEST_ONLY"
    if payload.get("document_status") != expected_status:
        raise EvidenceIntegrityError("source manifest status does not match execution mode")
    authorized = payload.get("authorized_calibration_sessions")
    if not isinstance(authorized, list) or session_id not in authorized:
        raise EvidenceIntegrityError("source manifest does not authorize review session")
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        raise EvidenceIntegrityError("source manifest has no sessions")
    matches = [item for item in sessions if isinstance(item, dict) and item.get("session_id") == session_id]
    if len(matches) != 1:
        raise EvidenceIntegrityError("source manifest session binding is not unique")
    if matches[0].get("frame_count") != frame_count or matches[0].get("fps") != fps:
        raise EvidenceIntegrityError("source manifest frame_count/FPS binding mismatch")


def verify_independent_mask_qa(
    *,
    authorization: FormalReviewAuthorization,
    mask_manifest_ref: EvidenceRef,
    session_id: str,
    product_line: str,
) -> Mapping[str, object]:
    qa_ref = authorization.independent_mask_qa_ref
    if qa_ref.schema_version != "mask-independent-qa-v1":
        raise EvidenceIntegrityError("independent MASK QA schema mismatch")
    if qa_ref.producer != "independent_mask_qa":
        raise EvidenceIntegrityError("independent MASK QA producer mismatch")
    qa = qa_ref.verified_json()
    expected = {
        "schema_version": "mask-independent-qa-v1",
        "producer": "independent_mask_qa",
        "status": "PASS",
        "session_id": session_id,
        "product_line": product_line,
        "mask_manifest_sha256": mask_manifest_ref.sha256,
    }
    for name, value in expected.items():
        if qa.get(name) != value:
            raise EvidenceIntegrityError(f"independent MASK QA binding mismatch: {name}")
    return qa


def compose_four_panel(frame: ReviewFrameInput) -> np.ndarray:
    raw = _rgb_plane(frame.raw_rgb, name="review RAW")
    overlay = _rgb_plane(frame.overlay_rgb, name="review H/O/U overlay")
    clean = _rgb_plane(frame.clean_rgb, name="review CLEAN")
    donor = _rgb_plane(frame.donor_diagnostic_rgb, name="review donor diagnostic").copy()
    hold_mask = _bool_plane(frame.unsupported_mask, name="review unsupported mask")
    if any(image.shape != raw.shape for image in (overlay, clean, donor)):
        raise ValueError("all four review panels must have equal dimensions")
    if hold_mask.shape != raw.shape[:2]:
        raise ValueError("review unsupported mask dimensions differ from panels")
    donor[hold_mask] = np.array([255, 0, 255], dtype=np.uint8)
    return np.concatenate((raw, overlay, clean, donor), axis=1)


def _validate_frame(
    frame: ReviewFrameInput,
    *,
    allowed_routes: set[str],
    verify_refs: bool,
) -> None:
    if frame.frame_index < 0 or frame.timestamp_ns < 0:
        raise ValueError("frame index/timestamp must be non-negative")
    if not math.isfinite(frame.video_time_s) or frame.video_time_s < 0:
        raise ValueError("video time must be finite and non-negative")
    if len(set(frame.route_ids)) != len(frame.route_ids):
        raise ValueError("frame route ids must be unique")
    if any(route not in allowed_routes for route in frame.route_ids):
        raise ValueError("frame contains route outside verified registry")
    unsupported = _bool_plane(frame.unsupported_mask, name="review unsupported mask")
    unsupported_count = int(np.count_nonzero(unsupported))
    required = {
        "object_reveal_coverage",
        "background_reveal_coverage",
        "unsupported_pixel_count",
    }
    if set(frame.coverage) != required:
        raise ValueError("frame coverage keys are incomplete or unexpected")
    for key in ("object_reveal_coverage", "background_reveal_coverage"):
        value = frame.coverage[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise ValueError(f"invalid frame coverage ratio: {key}")
    if frame.coverage["unsupported_pixel_count"] != unsupported_count:
        raise ValueError("coverage unsupported count disagrees with HOLD mask")
    if frame.hold != (unsupported_count > 0):
        raise ValueError("frame HOLD state disagrees with unsupported mask")
    compose_four_panel(frame)
    if verify_refs:
        for ref in (frame.raw_ref, frame.overlay_ref, frame.clean_ref, frame.donor_diagnostic_ref):
            _assert_scope(ref, staging_required=ref is not frame.raw_ref)
            ref.verified_bytes()


def _preflight(
    *,
    session_id: str,
    product_line: str,
    frames: Sequence[ReviewFrameInput],
    stress_frame_indices: Sequence[int],
    fps: float,
    artifacts: ReviewArtifactInputs,
    formal: bool,
) -> None:
    if not session_id or not product_line:
        raise ValueError("session_id and product_line must be non-empty")
    if len(frames) != 460 or fps != 30:
        raise ValueError("current G2 review contract requires exactly 460 frames at 30 FPS")
    if not 12 <= len(stress_frame_indices) <= 24 or len(set(stress_frame_indices)) != len(
        stress_frame_indices
    ):
        raise ValueError("stress review must contain 12--24 unique frames")
    policy = load_verified_reveal_policy(artifacts.reveal_policy_ref)
    artifact_refs = (
        artifacts.canary_set_ref,
        artifacts.mask_evidence_ref,
        artifacts.clean_layers_ref,
        artifacts.clean_coverage_ref,
        artifacts.task_card_ref,
        artifacts.calibration_overlay_ref,
        artifacts.reveal_policy_ref,
    )
    _verify_source_manifest(
        artifacts.source_manifest_ref,
        session_id=session_id,
        frame_count=len(frames),
        fps=fps,
        formal=formal,
    )
    for ref in artifact_refs:
        _assert_scope(ref, staging_required=formal)
        ref.verified_bytes()

    canary = _verify_artifact_json(
        artifacts.canary_set_ref,
        schema_version="canary-set-v1",
        producer="session_profiler",
        session_id=session_id,
        product_line=product_line,
        artifact_state=("G2_CALIBRATION_CANDIDATE" if formal else "SYNTHETIC_TEST_ONLY"),
    )
    _verify_artifact_json(
        artifacts.mask_evidence_ref,
        schema_version="mask-evidence-v1",
        producer="mask_producer",
        session_id=session_id,
        product_line=product_line,
        artifact_state=("G2_CALIBRATION_CANDIDATE" if formal else "SYNTHETIC_TEST_ONLY"),
    )
    _verify_artifact_json(
        artifacts.clean_layers_ref,
        schema_version="clean-layers-v1",
        producer="clean_background_producer",
        session_id=session_id,
        product_line=product_line,
        artifact_state=("G2_CALIBRATION_CANDIDATE" if formal else "SYNTHETIC_TEST_ONLY"),
    )
    _verify_artifact_json(
        artifacts.clean_coverage_ref,
        schema_version="clean-coverage-v1",
        producer="clean_background_producer",
        session_id=session_id,
        product_line=product_line,
        artifact_state=("G2_CALIBRATION_CANDIDATE" if formal else "SYNTHETIC_TEST_ONLY"),
    )
    canary_frames = canary.get("frames")
    if not isinstance(canary_frames, list):
        raise EvidenceIntegrityError("canary manifest has no frame list")
    if canary.get("frame_count") != len(canary_frames):
        raise EvidenceIntegrityError("canary manifest frame_count mismatch")
    canary_map: dict[int, tuple[str, ...]] = {}
    for entry in canary_frames:
        if not isinstance(entry, dict) or not isinstance(entry.get("frame_index"), int):
            raise EvidenceIntegrityError("invalid canary frame entry")
        strata = entry.get("strata")
        if not isinstance(strata, list) or not strata or any(not isinstance(item, str) for item in strata):
            raise EvidenceIntegrityError("invalid canary strata")
        canary_map[entry["frame_index"]] = tuple(strata)
    if set(canary_map) != set(stress_frame_indices):
        raise EvidenceIntegrityError("stress frame list does not equal authenticated canary set")

    previous_timestamp = -1
    previous_video_time = -1.0
    for expected_index, frame in enumerate(frames):
        if frame.frame_index != expected_index:
            raise ValueError("review frames must be contiguous, ordered and zero-based")
        _validate_frame(frame, allowed_routes=set(policy.route_registry), verify_refs=True)
        if frame.timestamp_ns < previous_timestamp or frame.video_time_s <= previous_video_time:
            raise ValueError("review frame time order is invalid")
        expected_time = expected_index / fps
        if not math.isclose(frame.video_time_s, expected_time, rel_tol=0, abs_tol=1e-6):
            raise ValueError("review frame video_time_s disagrees with measured FPS")
        previous_timestamp = frame.timestamp_ns
        previous_video_time = frame.video_time_s
    by_index = {frame.frame_index: frame for frame in frames}
    for frame_index, strata in canary_map.items():
        frame = by_index[frame_index]
        if frame.strata != strata or len(set(frame.strata)) != len(frame.strata):
            raise EvidenceIntegrityError("review stress strata disagree with canary manifest")


def _write_media(
    *, frames: Sequence[ReviewFrameInput], stress_frame_indices: Sequence[int], fps: float, sink: ReviewMediaSink
) -> tuple[list[Mapping[str, object]], EvidenceRef]:
    by_index = {frame.frame_index: frame for frame in frames}
    records: list[Mapping[str, object]] = []
    for index in stress_frame_indices:
        frame = by_index[index]
        panel_ref = sink.write_stress_frame(index, compose_four_panel(frame))
        panel_ref.verified_bytes()
        records.append(
            {
                "frame_index": index,
                "strata": list(frame.strata),
                "four_panel_image": panel_ref.to_dict(),
                "raw_ref": frame.raw_ref.to_dict(),
                "overlay_ref": frame.overlay_ref.to_dict(),
                "clean_ref": frame.clean_ref.to_dict(),
                "donor_diagnostic_ref": frame.donor_diagnostic_ref.to_dict(),
            }
        )
    consumed = 0

    def full_frames() -> Iterable[np.ndarray]:
        nonlocal consumed
        for frame in frames:
            consumed += 1
            yield compose_four_panel(frame)

    full_ref = sink.write_full_preview(full_frames(), fps=fps, layout=REVIEW_LAYOUT)
    if consumed != len(frames):
        raise ValueError("media sink did not consume complete preview sequence")
    full_ref.verified_bytes()
    return records, full_ref


def exercise_synthetic_review_dry_run(
    *,
    session_id: str,
    product_line: str,
    frames: Sequence[ReviewFrameInput],
    stress_frame_indices: Sequence[int],
    fps: float,
    artifacts: ReviewArtifactInputs,
    sink: ReviewMediaSink,
) -> Mapping[str, object]:
    _preflight(
        session_id=session_id,
        product_line=product_line,
        frames=frames,
        stress_frame_indices=stress_frame_indices,
        fps=fps,
        artifacts=artifacts,
        formal=False,
    )
    stress_records, full_ref = _write_media(
        frames=frames, stress_frame_indices=stress_frame_indices, fps=fps, sink=sink
    )
    return {
        "schema_version": "synthetic-mask-clean-review-dry-run-v1",
        "artifact_state": "SYNTHETIC_TEST_ONLY",
        "publishable": False,
        "formal_package_status": "BLOCKED",
        "stress_frames": stress_records,
        "full_preview_ref": full_ref.to_dict(),
    }


def package_mask_clean_review(
    *,
    session_id: str,
    product_line: str,
    frames: Sequence[ReviewFrameInput],
    stress_frame_indices: Sequence[int],
    fps: float,
    artifacts: ReviewArtifactInputs,
    authorization: FormalReviewAuthorization,
    review_schema_ref: EvidenceRef,
    sink: ReviewMediaSink,
    created_at: str | None = None,
) -> Mapping[str, object]:
    _preflight(
        session_id=session_id,
        product_line=product_line,
        frames=frames,
        stress_frame_indices=stress_frame_indices,
        fps=fps,
        artifacts=artifacts,
        formal=True,
    )
    verify_independent_mask_qa(
        authorization=authorization,
        mask_manifest_ref=artifacts.mask_evidence_ref,
        session_id=session_id,
        product_line=product_line,
    )
    if (
        review_schema_ref.producer != "contract_owner"
        or review_schema_ref.schema_version != "json-schema-2020-12"
    ):
        raise EvidenceIntegrityError("review schema EvidenceRef producer/schema mismatch")
    review_schema = review_schema_ref.verified_json()
    properties = review_schema.get("properties")
    required = review_schema.get("required")
    if (
        not isinstance(properties, dict)
        or "mask_independent_qa_ref" not in properties
        or not isinstance(required, list)
        or "mask_independent_qa_ref" not in required
    ):
        raise FormalReviewPackageBlocked(
            "FORMAL_REVIEW_PACKAGE_BLOCKED: current schema cannot record independent MASK QA ref"
        )

    records, full_ref = _write_media(
        frames=frames, stress_frame_indices=stress_frame_indices, fps=fps, sink=sink
    )
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("created_at must include timezone")
    return {
        "schema_version": "mask-clean-review-package-v1",
        "artifact_state": "G2_CALIBRATION_CANDIDATE",
        "producer": "mask_clean_review_packager",
        "session_id": session_id,
        "product_line": product_line,
        "source_manifest_ref": artifacts.source_manifest_ref.to_dict(),
        "canary_set_ref": artifacts.canary_set_ref.to_dict(),
        "mask_evidence_ref": artifacts.mask_evidence_ref.to_dict(),
        "mask_independent_qa_ref": authorization.independent_mask_qa_ref.to_dict(),
        "clean_layers_ref": artifacts.clean_layers_ref.to_dict(),
        "clean_coverage_ref": artifacts.clean_coverage_ref.to_dict(),
        "task_card_ref": artifacts.task_card_ref.to_dict(),
        "calibration_overlay_ref": artifacts.calibration_overlay_ref.to_dict(),
        "stress_frames": records,
        "full_preview": {
            "video_ref": full_ref.to_dict(),
            "layout": list(REVIEW_LAYOUT),
            "frame_count": len(frames),
            "fps": fps,
            "contains_hold_overlay": True,
        },
        "review_state": "AWAITING_HUMAN_MASK_CLEAN_REVIEW",
        "approval_ref": None,
        "formal_pass_claim_allowed": False,
        "blocked_stages": list(BLOCKED_STAGES),
        "created_at": timestamp,
    }
