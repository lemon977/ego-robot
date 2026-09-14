from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from pipeline.raw_source import StrictRawResolver
from pipeline.session_context import (
    ProfilerConfig,
    _projected_contact_score,
    build_session_context,
)


NAMES = [
    "thumb_fingertip",
    "index_fingertip",
    "middle_fingertip",
    "ring_fingertip",
    "pinky_fingertip",
    "wrist",
    "thumb_intermediate",
    "thumb_distal",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "ring_proximal",
    "ring_intermediate",
    "ring_distal",
    "pinky_proximal",
    "pinky_intermediate",
    "pinky_distal",
    "palm_center",
]


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _make_source(tmp_path: Path, frame_count: int = 12):
    raw = tmp_path / "raw"
    all_data = raw / "dataset" / "session" / "preprocess" / "all_data"
    frames = []
    for index in range(frame_count):
        root = all_data / f"{index:05d}"
        root.mkdir(parents=True)
        image = np.tile(np.arange(64, dtype=np.uint8), (48, 1))
        image = np.roll(image, index, axis=1)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        image_payload = encoded.tobytes()
        image_path = root / "rgb.png"
        image_path.write_bytes(image_payload)
        base = np.stack(
            [np.linspace(10, 30, 21) + index * 0.1, np.linspace(15, 35, 21)], axis=1
        )
        base[5] = [18 + index * 0.1, 35]
        base[8] = [14 + index * 0.1, 25]
        base[17] = [27 + index * 0.1, 25]
        base[20] = [20 + index * 0.1, 29]
        hands = {}
        for side, offset in (("left", 0), ("right", 24)):
            points = base.copy()
            points[:, 0] += offset
            hands[side] = {
                "joint_names": NAMES,
                "keypoints_2d": points.tolist(),
                "keypoints_3d_camera": np.column_stack(
                    [(points[:, 0] - 32) / 100, (points[:, 1] - 24) / 100, np.full(21, 0.5)]
                ).tolist(),
            }
        metadata = {
            "metadata": {
                "idx": index,
                "w": 64,
                "h": 48,
                "fps": 30.0,
                "k": [[32.0, 0.0, 31.5], [0.0, 32.0, 23.5], [0.0, 0.0, 1.0]],
            },
            "entities": {"hands": hands, "objects": {}},
        }
        metadata_payload = json.dumps(metadata).encode()
        metadata_path = root / "training_data.json"
        metadata_path.write_bytes(metadata_payload)
        frames.append(
            {
                "frame_index": index,
                "timestamp_ns": 1_000_000_000 + index * 33_333_333,
                "video_time_s": index / 30,
                "image": {
                    "path": str(image_path),
                    "kind": "regular_file",
                    "bytes": len(image_payload),
                    "sha256": _sha(image_payload),
                },
                "metadata": {
                    "path": str(metadata_path),
                    "kind": "regular_file",
                    "bytes": len(metadata_payload),
                    "sha256": _sha(metadata_payload),
                },
                "width": 64,
                "height": 48,
            }
        )
    manifest = {
        "schema_version": "candidate-source-manifest-v0",
        "status": "TEST_CANDIDATE",
        "raw_root": str(raw),
        "source_root": str(raw / "dataset"),
        "no_fallback": True,
        "sessions": [
            {
                "session_id": "session_beta",
                "source_session": str(raw / "dataset" / "session"),
                "selector": "preprocess/all_data/<frame_index:05d>/rgb.png",
                "selector_policy": "regular_file_only_no_symlink_no_fallback",
                "frame_order_policy": "ascending_zero_based_contiguous_frame_index",
                "timestamp_policy": "synthetic_video_clock",
                "width": 64,
                "height": 48,
                "fps": 30.0,
                "frame_count": frame_count,
                "frames": frames,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_payload = json.dumps(manifest).encode()
    manifest_path.write_bytes(manifest_payload)
    profile_path = tmp_path / "profile.yaml"
    profile_payload = yaml.safe_dump({"document_status": "DRAFT", "execution_ready": False}).encode()
    profile_path.write_bytes(profile_payload)
    task_card_path = tmp_path / "task_card.yaml"
    task_card_payload = yaml.safe_dump(
        {
            "status": "DRAFT",
            "dimensionless_calibration_overlay": {"test_ratio": 0.5},
        }
    ).encode()
    task_card_path.write_bytes(task_card_payload)
    return (
        raw,
        manifest_path,
        manifest_payload,
        profile_path,
        profile_payload,
        task_card_path,
        task_card_payload,
    )


def test_builds_blocked_draft_context_without_visual_outputs(tmp_path: Path) -> None:
    (
        raw,
        manifest,
        manifest_payload,
        profile,
        profile_payload,
        task_card,
        task_card_payload,
    ) = _make_source(tmp_path)
    resolver = StrictRawResolver(
        manifest_path=manifest,
        expected_manifest_sha256=_sha(manifest_payload),
        raw_root=raw,
        session_id="session_beta",
        allowed_manifest_statuses=("TEST_CANDIDATE",),
    )
    result = build_session_context(
        resolver,
        execution_mode="G2_CALIBRATION",
        product_line="004_CONTACT_GOLD",
        diagnostic_track="HAND_ONLY_DIAGNOSTIC",
        profile_path=str(profile),
        expected_profile_sha256=_sha(profile_payload),
        task_card_path=str(task_card),
        expected_task_card_sha256=_sha(task_card_payload),
        calibration_overlay_path=str(task_card),
        expected_calibration_overlay_sha256=_sha(task_card_payload),
        config=ProfilerConfig(
            analysis_scale=0.5,
            forearm_width_ratio=2.0,
            contact_radius_ratio=0.2,
            contact_projection_ratio=0.15,
            count_min=12,
            count_max=24,
            target_count=12,
            min_gap_seconds=0.03,
            stable_contact_seconds=0.2,
            pre_contact_seconds=0.2,
            contact_on_threshold=0.5,
        ),
    )
    assert result.context["execution_allowed"] is False
    assert "OBJECT_EVIDENCE_NOT_PINNED" in result.context["execution_blockers"]
    assert result.context["canary_selection"]["manual_review_required"] is True
    assert len(result.selection.frames) == 12
    assert result.context["measurements"]["donor_coverage"]["measurement_status"] == "MISSING"
    assert result.context["measurements"]["forearm_width_px"]["measurement_status"] == "MEASURED"


def test_pipeline_source_contains_no_session_literal_or_branch() -> None:
    package_root = Path(__file__).parents[1]
    production_sources = [
        package_root / "raw_source.py",
        package_root / "session_context.py",
        package_root / "stress_frames.py",
    ]
    for path in production_sources:
        source = path.read_text()
        assert "grap_a_cap_" not in source
        assert "session_id ==" not in source
        assert "session_id in" not in source


def test_projected_contact_score_uses_object_relative_distance() -> None:
    object_mask = np.zeros((50, 50), dtype=np.uint8)
    object_mask[20:30, 20:30] = 1
    inside = {"left": {"points_2d": np.array([[49.0, 49.0]])}}
    outside = {"left": {"points_2d": np.array([[80.0, 80.0]])}}

    inside_score = _projected_contact_score(
        inside,
        object_mask,
        source_width=100,
        source_height=100,
        object_scale=20.0,
        contact_projection_ratio=0.2,
    )
    outside_score = _projected_contact_score(
        outside,
        object_mask,
        source_width=100,
        source_height=100,
        object_scale=20.0,
        contact_projection_ratio=0.2,
    )

    assert inside_score == 1.0
    assert outside_score == 0.0
