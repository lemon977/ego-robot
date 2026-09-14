from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pytest

from pipeline.reveal_labels import EvidenceIntegrityError, EvidenceRef
from pipeline.review_packager import (
    FormalReviewAuthorization,
    REVIEW_LAYOUT,
    ReviewArtifactInputs,
    ReviewFrameInput,
    exercise_synthetic_review_dry_run,
    package_mask_clean_review,
)
from pipeline.tests.evidence_test_utils import policy_ref, write_bytes_ref, write_json_ref


SESSION = "synthetic_session"
PRODUCT = "004_CONTACT_GOLD"


class FileTestSink:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.stress_count = 0
        self.preview_count = 0
        self.first_stress: np.ndarray | None = None

    def write_stress_frame(self, frame_index: int, four_panel_rgb: np.ndarray) -> EvidenceRef:
        self.stress_count += 1
        if self.first_stress is None:
            self.first_stress = four_panel_rgb.copy()
        return write_bytes_ref(
            self.root,
            f"_run/dry/media/stress_{frame_index:06d}.bin",
            four_panel_rgb.tobytes(),
            producer="mask_clean_review_packager",
        )

    def write_full_preview(
        self,
        frames: Iterable[np.ndarray],
        *,
        fps: float,
        layout: Sequence[str],
    ) -> EvidenceRef:
        digest = hashlib.sha256()
        for frame in frames:
            self.preview_count += 1
            digest.update(frame.tobytes())
        assert fps == 30
        assert tuple(layout) == REVIEW_LAYOUT
        return write_bytes_ref(
            self.root,
            "_run/dry/media/preview.bin",
            digest.digest(),
            producer="mask_clean_review_packager",
        )


def _artifact_ref(
    root: Path,
    name: str,
    schema: str,
    producer: str,
    artifact_state: str,
    **extra: object,
) -> EvidenceRef:
    return write_json_ref(
        root,
        f"_run/dry/{name}.json",
        {
            "schema_version": schema,
            "producer": producer,
            "session_id": SESSION,
            "product_line": PRODUCT,
            "artifact_state": artifact_state,
            **extra,
        },
        producer=producer,
        schema_version=schema,
    )


def _artifacts(root: Path, *, formal: bool = False) -> ReviewArtifactInputs:
    state = "G2_CALIBRATION_CANDIDATE" if formal else "SYNTHETIC_TEST_ONLY"
    canary = _artifact_ref(
        root,
        "canary",
        "canary-set-v1",
        "session_profiler",
        state,
        frame_count=12,
        frames=[{"frame_index": index, "strata": ["session_boundary"]} for index in range(12)],
    )
    return ReviewArtifactInputs(
        source_manifest_ref=write_json_ref(
            root,
            "source_manifest.json",
            {
                "schema_version": "verified-source-manifest-v1",
                "document_status": (
                    "VERIFIED_G0_CALIBRATION_INPUT" if formal else "SYNTHETIC_TEST_ONLY"
                ),
                "authorized_calibration_sessions": [SESSION],
                "sessions": [{"session_id": SESSION, "frame_count": 460, "fps": 30}],
            },
            producer="source_resolver",
            schema_version="verified-source-manifest-v1",
        ),
        canary_set_ref=canary,
        mask_evidence_ref=_artifact_ref(
            root, "mask", "mask-evidence-v1", "mask_producer", state
        ),
        clean_layers_ref=_artifact_ref(
            root,
            "clean_layers",
            "clean-layers-v1",
            "clean_background_producer",
            state,
        ),
        clean_coverage_ref=_artifact_ref(
            root,
            "clean_coverage",
            "clean-coverage-v1",
            "clean_background_producer",
            state,
        ),
        task_card_ref=write_bytes_ref(
            root, "_run/dry/TASK_CARD.yaml", b"synthetic-task", producer="task_card"
        ),
        calibration_overlay_ref=write_bytes_ref(
            root, "_run/dry/overlay.json", b"synthetic-overlay", producer="task_card"
        ),
        reveal_policy_ref=policy_ref(root / "_run/dry"),
    )


def _frames(root: Path) -> list[ReviewFrameInput]:
    raw_ref = write_bytes_ref(root, "raw.bin", b"raw", producer="source_resolver")
    overlay_ref = write_bytes_ref(
        root, "_run/dry/overlay.bin", b"overlay", producer="mask_producer"
    )
    clean_ref = write_bytes_ref(
        root, "_run/dry/clean.bin", b"clean", producer="clean_background_producer"
    )
    diagnostic_ref = write_bytes_ref(
        root,
        "_run/dry/diagnostic.bin",
        b"diagnostic",
        producer="clean_background_producer",
    )
    frames: list[ReviewFrameInput] = []
    for index in range(460):
        image = np.full((2, 3, 3), index % 255, np.uint8)
        unsupported = np.zeros((2, 3), bool)
        if index == 0:
            unsupported[0, 0] = True
        frames.append(
            ReviewFrameInput(
                frame_index=index,
                timestamp_ns=int(index * 1_000_000_000 / 30),
                video_time_s=index / 30,
                raw_rgb=image,
                overlay_rgb=np.full_like(image, 50),
                clean_rgb=np.full_like(image, 100),
                donor_diagnostic_rgb=np.full_like(image, 150),
                unsupported_mask=unsupported,
                route_ids=("ANALYTIC_GEOMETRY_TEXTURE", "TEMPORAL_DONOR_ATLAS"),
                coverage={
                    "object_reveal_coverage": 1.0,
                    "background_reveal_coverage": 1.0,
                    "unsupported_pixel_count": int(np.count_nonzero(unsupported)),
                },
                hold=bool(np.any(unsupported)),
                raw_ref=raw_ref,
                overlay_ref=overlay_ref,
                clean_ref=clean_ref,
                donor_diagnostic_ref=diagnostic_ref,
                strata=("session_boundary",) if index < 12 else (),
            )
        )
    return frames


def _qa_ref(root: Path, mask_ref: EvidenceRef, **changes: object) -> EvidenceRef:
    payload: dict[str, object] = {
        "schema_version": "mask-independent-qa-v1",
        "producer": "independent_mask_qa",
        "status": "PASS",
        "session_id": SESSION,
        "product_line": PRODUCT,
        "mask_manifest_sha256": mask_ref.sha256,
    }
    payload.update(changes)
    return write_json_ref(
        root,
        "_run/dry/mask_independent_qa.json",
        payload,
        producer="independent_mask_qa",
        schema_version="mask-independent-qa-v1",
    )


def _review_schema_ref() -> EvidenceRef:
    path = Path(__file__).parents[2] / "contracts/mask_clean_review_package.schema.json"
    return EvidenceRef.from_file(
        str(path), producer="contract_owner", schema_version="json-schema-2020-12"
    )


def test_synthetic_dry_run_is_unambiguously_non_publishable(tmp_path: Path) -> None:
    sink = FileTestSink(tmp_path)
    result = exercise_synthetic_review_dry_run(
        session_id=SESSION,
        product_line=PRODUCT,
        frames=_frames(tmp_path),
        stress_frame_indices=list(range(12)),
        fps=30,
        artifacts=_artifacts(tmp_path),
        sink=sink,
    )
    assert result["artifact_state"] == "SYNTHETIC_TEST_ONLY"
    assert result["publishable"] is False
    assert result["formal_package_status"] == "BLOCKED"
    assert "review_state" not in result
    assert sink.stress_count == 12
    assert sink.preview_count == 460
    assert sink.first_stress is not None
    assert np.all(sink.first_stress[0, 9] == [255, 0, 255])


def test_digest_bound_schema_allows_g2_review_package_after_independent_mask_qa(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path, formal=True)
    qa = _qa_ref(tmp_path, artifacts.mask_evidence_ref)
    sink = FileTestSink(tmp_path)
    result = package_mask_clean_review(
        session_id=SESSION,
        product_line=PRODUCT,
        frames=_frames(tmp_path),
        stress_frame_indices=list(range(12)),
        fps=30,
        artifacts=artifacts,
        authorization=FormalReviewAuthorization(qa),
        review_schema_ref=_review_schema_ref(),
        sink=sink,
    )
    assert result["mask_independent_qa_ref"]["sha256"] == qa.sha256
    assert result["mask_independent_qa_ref"]["producer"] == "independent_mask_qa"
    assert result["review_state"] == "AWAITING_HUMAN_MASK_CLEAN_REVIEW"
    assert result["formal_pass_claim_allowed"] is False
    assert sink.stress_count == 12
    assert sink.preview_count == 460


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("status", "FAIL"),
        ("producer", "mask_producer"),
        ("schema_version", "unknown-v1"),
        ("session_id", "other_session"),
        ("product_line", "EXACT78_R2_VISUAL_DOMAIN"),
        ("mask_manifest_sha256", "0" * 64),
    ],
)
def test_independent_qa_binding_mismatch_blocks_before_media(
    tmp_path: Path, field: str, wrong: object
) -> None:
    artifacts = _artifacts(tmp_path, formal=True)
    qa = _qa_ref(tmp_path, artifacts.mask_evidence_ref, **{field: wrong})
    # Producer/schema mismatches must also be reflected by the digest ref itself.
    if field in {"producer", "schema_version"}:
        qa = replace(
            qa,
            producer=str(wrong) if field == "producer" else qa.producer,
            schema_version=str(wrong) if field == "schema_version" else qa.schema_version,
        )
    sink = FileTestSink(tmp_path)
    with pytest.raises(EvidenceIntegrityError, match="independent MASK QA"):
        package_mask_clean_review(
            session_id=SESSION,
            product_line=PRODUCT,
            frames=_frames(tmp_path),
            stress_frame_indices=list(range(12)),
            fps=30,
            artifacts=artifacts,
            authorization=FormalReviewAuthorization(qa),
            review_schema_ref=_review_schema_ref(),
            sink=sink,
        )
    assert sink.stress_count == 0


def test_bad_canary_or_coverage_blocks_before_media_write(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    frames = _frames(tmp_path)
    frames[0] = replace(
        frames[0],
        coverage={
            "object_reveal_coverage": 1.0,
            "background_reveal_coverage": 1.0,
            "unsupported_pixel_count": 0,
        },
    )
    sink = FileTestSink(tmp_path)
    with pytest.raises(ValueError, match="unsupported count"):
        exercise_synthetic_review_dry_run(
            session_id=SESSION,
            product_line=PRODUCT,
            frames=frames,
            stress_frame_indices=list(range(12)),
            fps=30,
            artifacts=artifacts,
            sink=sink,
        )
    assert sink.stress_count == 0


def test_tampered_qa_file_blocks_before_media_write(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path, formal=True)
    qa = _qa_ref(tmp_path, artifacts.mask_evidence_ref)
    Path(qa.path).write_text("{}", encoding="utf-8")
    sink = FileTestSink(tmp_path)
    with pytest.raises(EvidenceIntegrityError, match="byte count|digest mismatch"):
        package_mask_clean_review(
            session_id=SESSION,
            product_line=PRODUCT,
            frames=_frames(tmp_path),
            stress_frame_indices=list(range(12)),
            fps=30,
            artifacts=artifacts,
            authorization=FormalReviewAuthorization(qa),
            review_schema_ref=_review_schema_ref(),
            sink=sink,
        )
    assert sink.stress_count == 0


def test_packager_has_no_generation_specific_path_hardcoding() -> None:
    source = (Path(__file__).parents[1] / "review_packager.py").read_text(encoding="utf-8").lower()
    assert "v7" not in source
    assert "v8" not in source
