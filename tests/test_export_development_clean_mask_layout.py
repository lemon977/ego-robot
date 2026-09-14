from __future__ import annotations

import argparse
import errno
from io import BytesIO
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_eligible68_batch_shard_plan_t0 as eligible_scope  # noqa: E402
from tools import export_development_clean_mask_layout as exporter  # noqa: E402
from tools import run_deterministic_table_reprojection_clean as runner  # noqa: E402
from tools import run_raw_apriltag_session_static_table_plane as raw_plane  # noqa: E402


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _png(path: Path, value: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(value).save(output, format="PNG")
    payload = output.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _raw_plane_fixture(tmp_path: Path) -> tuple[Path, Path]:
    session = "grap_a_cap_004"
    raw_root = tmp_path / session / "preprocess" / "all_data"
    intrinsic = np.asarray(
        [[100.0, 0.0, 7.5], [0.0, 100.0, 7.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    c2w = np.eye(4, dtype=np.float64)
    for frame in range(15):
        _png(raw_root / f"{frame:05d}" / "rgb.png", np.zeros((16, 16, 3), np.uint8))
        _json(
            raw_root / f"{frame:05d}" / "training_data.json",
            {
                "metadata": {
                    "w": 16,
                    "h": 16,
                    "k": intrinsic.tolist(),
                    "c2w": c2w.tolist(),
                }
            },
        )

    def observer(_rgb: np.ndarray, camera: np.ndarray) -> dict[str, object]:
        transform = np.eye(4, dtype=np.float64)
        transform[2, 3] = 2.0
        corners_camera = raw_plane.transform_points(
            transform, raw_plane.tag_corners_local(0.1)
        )
        projected = corners_camera @ camera.T
        corners = projected[:, :2] / projected[:, 2, None]
        return {
            "accepted": True,
            "reason": None,
            "detected_ids": [0],
            "tag_to_camera": transform,
            "corners_uv": corners,
            "corner_reprojection_errors_px": np.zeros(4, dtype=np.float64),
            "corner_reprojection_mean_px": 0.0,
            "corner_reprojection_max_px": 0.0,
        }

    output_parent = tmp_path / "plane-output"
    output_parent.mkdir()
    raw_plane.run(
        raw_root=raw_root,
        session_id=session,
        output_parent=output_parent,
        output_name="candidate",
        frame_observer=observer,
    )
    return raw_root, output_parent / "candidate" / "TABLE_PLANE_STATIC.json"


def _source_fixture(
    root: Path,
    *,
    zero_frame: int | None = None,
    profile: str = "q2",
    unavailable_status: str = "HOLD",
    skipped_frames: tuple[int, ...] = (),
) -> Path:
    session = "grap_a_cap_004"
    denominator = 7
    left_accept = {0, 1, 2, 3, 4, 5}
    right_accept = {0, 1, 3, 4, 5, 6}
    skipped = set(skipped_frames)
    processed = denominator - len(skipped)
    left_count = len(left_accept - skipped)
    right_count = len(right_accept - skipped)
    bilateral_count = len((left_accept & right_accept) - skipped)
    manifest = {
        "schema_version": exporter.SOURCE_SCHEMA,
        "status": exporter.SOURCE_STATUS,
        "run_id": root.name,
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "governance_bypassed": ["development-only fixture"],
        "governance_retained": ["human review required"],
        "route_of_evidence": "A_PRIME_SELECTOR_PLUS_HAWOR_OBJECT6D_CONTACT_PHASE",
        "selectors": ["task26_d1", "task26_d1_d4"],
        "detection_mode": "dual",
        "d4_applied": False,
        "context_root": f"archive/legacy_runs/unclassified/claude_generic_context_v1/{session}",
        "task29_module_sha256": "1" * 64,
        "pinned_build": {"fixture": True},
        "started_at": "2026-08-30T00:00:00+08:00",
        "sessions": {
            session: {
                "frames_processed": processed,
                "seconds": 1.0,
                "seconds_per_frame": 1.0 / denominator,
                "per_side_counts": {
                    "task26_d1": {
                        "left": {
                            "accept": left_count,
                            "reject": processed - left_count,
                        },
                        "right": {
                            "accept": right_count,
                            "reject": processed - right_count,
                        },
                    },
                    "task26_d1_d4": {
                        "left": {
                            "accept": left_count,
                            "reject": processed - left_count,
                        },
                        "right": {
                            "accept": right_count,
                            "reject": processed - right_count,
                        },
                    },
                },
                "dual_pass_d1": bilateral_count,
                "dual_pass_d1_d4": bilateral_count,
                "dual_pass_rate_d1": round(bilateral_count / processed, 4),
                "dual_pass_rate_d1_d4": round(bilateral_count / processed, 4),
                "d4_flip_count": 0,
                "d4_flips": [],
                "video": str(root / "d1_d4_live" / f"RAW_D1_D4_{session}.mp4"),
                "timing": {
                    "prompt": "an arm",
                    "method": "DUAL_PROPAGATE_PLUS_PER_FRAME_UNION_OF_CANDIDATES",
                    "instance_counts": [1] * denominator,
                    "total_seconds": 1.0,
                    "seconds_per_frame": 1.0 / denominator,
                    "frames_streamed": denominator,
                },
                "skipped_object6d_nonfinite": {
                    "frames": list(skipped_frames),
                    "count": len(skipped_frames),
                    "fraction_of_session": round(len(skipped_frames) / denominator, 4),
                },
            }
        },
        "propagation_frame0_duplicate_delta": {},
        "finished_at": "2026-08-30T00:00:01+08:00",
    }
    if profile in {"q", "legacy-dual"}:
        del manifest["sessions"][session]["skipped_object6d_nonfinite"]
    if profile == "legacy-dual":
        del manifest["d4_applied"]
        del manifest["context_root"]
        del manifest["task29_module_sha256"]
    manifest_path = root / "MANIFEST.json"
    _json(manifest_path, manifest)
    for frame in range(denominator):
        if frame in skipped:
            continue
        audit_sides: dict[str, object] = {}
        for side, accepted in (
            ("left", frame in left_accept),
            ("right", frame in right_accept),
        ):
            area = 0
            if accepted:
                mask = np.zeros((4, 6), dtype=np.uint8)
                if frame != zero_frame:
                    mask[(frame + (side == "right")) % 4, frame % 6] = 255
                    area = 1
                _png(
                    root
                    / "d1_d4_live"
                    / "formal_masks"
                    / "task26_d1_d4"
                    / side
                    / session
                    / f"frame_{frame:05d}.png",
                    mask,
                )
            stage = {
                "status": "ACCEPT" if accepted else unavailable_status,
                "accepted_offset": 0 if accepted else None,
                "accepted_stream": "PROPAGATE" if accepted else None,
                "accepted": accepted,
                "area": area,
                "reasons": [],
            }
            audit_sides[side] = {
                "task26_d1": dict(stage),
                "task26_d1_d4": dict(stage),
                "d4_application_status": (
                    "NO_CHANGE_SEPARATED" if profile == "legacy-dual" else "None"
                ),
                "d4_flip": False,
            }
        _json(
            root / "d1_d4_live" / "frames" / session / f"frame_{frame:05d}.json",
            {"session_id": session, "frame_index": frame, "sides": audit_sides},
        )
    return manifest_path


def _export(tmp_path: Path) -> tuple[Path, Path, Path]:
    source_manifest = _source_fixture(tmp_path / "source")
    output = tmp_path / "exports" / "mask-layout"
    output.parent.mkdir()
    manifest_path = exporter.export_layout(
        session_id="grap_a_cap_004",
        session_frame_count=7,
        source_run_manifest=source_manifest,
        preregistered_bilateral_ranges="0-1,3-5",
        output_root=output,
        expected_width=6,
        expected_height=4,
    )
    return source_manifest, output, manifest_path


def _development_run_fixture(
    tmp_path: Path,
) -> tuple[argparse.Namespace, Path, Path]:
    _, mask_root, mask_manifest = _export(tmp_path)
    raw_root = tmp_path / "raw"
    for frame in (3, 4, 5):
        rgb = np.full((4, 6, 3), 30 + frame, dtype=np.uint8)
        _png(raw_root / f"{frame:05d}" / "rgb.png", rgb)
        _json(
            raw_root / f"{frame:05d}" / "training_data.json",
            {
                "metadata": {
                    "idx": frame,
                    "camera_model": "rectified_pinhole",
                    "w": 6,
                    "h": 4,
                    "fps": 30,
                    "k": [[4.0, 0.0, 2.5], [0.0, 4.0, 1.5], [0.0, 0.0, 1.0]],
                    "c2w": [
                        [1.0, 0.0, 0.0, 0.001 * frame],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
            },
        )
    plane_path = tmp_path / "PLANE.json"
    _json(
        plane_path,
        {
            "schema_version": "table-plane-static-candidate-v1",
            "session_id": "grap_a_cap_004",
            "plumbing_only": True,
            "formal_consumer_allowed": False,
            "is_clean_candidate": False,
            "candidate_requires_human_review": True,
            "estimation_quality": {"pass": True},
            "plane": {
                "cardinality": 1,
                "per_frame_offsets_present": False,
                "normal": [0.0, 0.0, 1.0],
                "offset_m": -2.0,
            },
        },
    )
    output = tmp_path / "clean-output"
    args = argparse.Namespace(
        execution_mode="DEVELOPMENT_ONLY",
        formal_authority_manifest=None,
        expected_cohort_id=None,
        raw_root=raw_root,
        mask_root=mask_root,
        mask_source_manifest=mask_manifest,
        plane=plane_path,
        session_id="grap_a_cap_004",
        output_root=output,
        start_frame=3,
        frame_count=3,
        session_frame_count=7,
    )
    return args, output, raw_root


def test_runner_accepts_exact_raw_apriltag_plane_and_binds_full_session(
    tmp_path: Path,
) -> None:
    raw_root, plane_path = _raw_plane_fixture(tmp_path)

    plane, _, _, refs = runner.load_plane(
        plane_path,
        "grap_a_cap_004",
        expected_raw_root=raw_root,
        expected_session_frame_count=15,
    )

    assert plane.offset_m == pytest.approx(-2.0)
    assert refs is not None
    assert tuple(refs) == tuple(range(15))
    assert refs[0]["rgb"]["path"] == str(raw_root / "00000" / "rgb.png")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.update(artifact_state="HUMAN_REVIEW_COMPLETE"),
            "artifact_state mismatch",
        ),
        (
            lambda value: value.update(is_clean_candidate=False),
            "schema branch fields mismatch",
        ),
        (
            lambda value: value["estimation_quality"].update(fit_gate_pass=1),
            "fit gate/threshold mismatch",
        ),
        (
            lambda value: value["frames"][0].update(frame_index=True),
            "frame order mismatch",
        ),
        (
            lambda value: value["frames"][0]["inputs"]["rgb"].update(
                path=value["frames"][1]["inputs"]["rgb"]["path"]
            ),
            "reference path mismatch",
        ),
        (
            lambda value: value["estimation_quality"]["metrics"].update(
                residual_max_abs_m=0.014
            ),
            "quality metric residual_max_abs_m mismatch",
        ),
    ],
)
def test_runner_rejects_raw_apriltag_plane_near_miss_and_type_alias_attacks(
    tmp_path: Path,
    mutate: object,
    message: str,
) -> None:
    raw_root, plane_path = _raw_plane_fixture(tmp_path)
    value = json.loads(plane_path.read_text())
    mutate(value)  # type: ignore[operator]
    _json(plane_path, value)

    with pytest.raises(runner.CleanRunError, match=message):
        runner.load_plane(
            plane_path,
            "grap_a_cap_004",
            expected_raw_root=raw_root,
            expected_session_frame_count=15,
        )


def test_runner_rejects_raw_plane_hardlink_and_cross_stage_swap(
    tmp_path: Path,
) -> None:
    raw_root, plane_path = _raw_plane_fixture(tmp_path)
    rgb_path = raw_root / "00000" / "rgb.png"
    os.link(rgb_path, rgb_path.with_name("alias.png"))
    with pytest.raises(runner.CleanRunError, match="single-link"):
        runner.load_plane(
            plane_path,
            "grap_a_cap_004",
            expected_raw_root=raw_root,
            expected_session_frame_count=15,
        )
    rgb_path.with_name("alias.png").unlink()
    _, _, _, refs = runner.load_plane(
        plane_path,
        "grap_a_cap_004",
        expected_raw_root=raw_root,
        expected_session_frame_count=15,
    )
    assert refs is not None
    payload = rgb_path.read_bytes()
    replacement = rgb_path.with_name("replacement.png")
    replacement.write_bytes(payload)
    os.replace(replacement, rgb_path)
    mask_root = tmp_path / "legacy-masks"
    zero = np.zeros((16, 16), dtype=np.uint8)
    _png(mask_root / "left" / "frame_00000.png", zero)
    _png(mask_root / "right" / "frame_00000.png", zero)
    with pytest.raises(runner.CleanRunError, match="lineage changed before CLEAN"):
        runner.preflight_development_inputs(
            raw_root=raw_root,
            mask_root=mask_root,
            mask_layout=None,
            frames=(0,),
            expected_plane_input_refs=refs,
        )


def _development_camera_payload(fps: object) -> bytes:
    return json.dumps(
        {
            "metadata": {
                "idx": 0,
                "camera_model": "rectified_pinhole",
                "w": 16,
                "h": 12,
                "fps": fps,
                "k": [[10.0, 0.0, 7.5], [0.0, 10.0, 5.5], [0.0, 0.0, 1.0]],
                "c2w": np.eye(4, dtype=np.float64).tolist(),
            }
        }
    ).encode()


@pytest.mark.parametrize("fps", [30, 30.0])
def test_development_camera_canonicalizes_exact_integral_fps_union(
    fps: object,
) -> None:
    _, metadata = runner._development_camera_from_payload(
        _development_camera_payload(fps), frame=0
    )

    assert metadata["fps"] == 30
    assert type(metadata["fps"]) is int


@pytest.mark.parametrize(
    "fps", [True, False, 0, -1, 30.5, float("nan"), float("inf"), "30", None]
)
def test_development_camera_rejects_nonintegral_or_aliased_fps(fps: object) -> None:
    with pytest.raises(
        runner.CleanRunError, match="camera metadata authority mismatch"
    ):
        runner._development_camera_from_payload(
            _development_camera_payload(fps), frame=0
        )


def test_literal_development_allowlist_is_exact_frozen_eligible68() -> None:
    assert runner.DEVELOPMENT_ELIGIBLE68_SESSION_IDS == eligible_scope.ELIGIBLE68
    assert len(runner.ALLOWED_SESSIONS) == 68
    assert runner.ALLOWED_SESSIONS.isdisjoint(eligible_scope.FORBIDDEN10)


def test_export_preserves_indices_and_denominator_with_pending_truth(
    tmp_path: Path,
) -> None:
    source_manifest, output, manifest_path = _export(tmp_path)
    manifest = json.loads(manifest_path.read_text())

    assert manifest["selected_frame_indices"] == [3, 4, 5]
    assert manifest["selected_contiguous_range"] == {
        "start_frame": 3,
        "end_frame": 5,
        "frame_count": 3,
    }
    assert manifest["session_frame_count"] == 7
    assert manifest["processed_frame_count"] == 3
    assert manifest["pending_frame_indices"] == [0, 1, 2, 6]
    assert manifest["source_unavailable_frame_indices"] == [2, 6]
    assert manifest["available_not_selected_frame_indices"] == [0, 1]
    assert sorted(path.name for path in (output / "left").iterdir()) == [
        "frame_00003.png",
        "frame_00004.png",
        "frame_00005.png",
    ]
    for frame in (3, 4, 5):
        for side in ("left", "right"):
            source = (
                source_manifest.parent
                / "d1_d4_live"
                / "formal_masks"
                / "task26_d1_d4"
                / side
                / "grap_a_cap_004"
                / f"frame_{frame:05d}.png"
            )
            assert (output / side / f"frame_{frame:05d}.png").read_bytes() == (
                source.read_bytes()
            )

    layout = runner.preflight_development_mask_layout(
        manifest_path,
        mask_root=output,
        expected_session_id="grap_a_cap_004",
        expected_session_frame_count=7,
        expected_target_frames=(3, 4, 5),
    )
    assert layout is not None
    assert layout.selected_frames == (3, 4, 5)
    assert layout.pending_frames == (0, 1, 2, 6)


def test_export_rejects_inventory_drift_before_output(tmp_path: Path) -> None:
    source_manifest = _source_fixture(tmp_path / "source")
    output = tmp_path / "exports" / "must-not-exist"
    output.parent.mkdir()

    with pytest.raises(exporter.MaskExportError, match="differs from explicit"):
        exporter.export_layout(
            session_id="grap_a_cap_004",
            session_frame_count=7,
            source_run_manifest=source_manifest,
            preregistered_bilateral_ranges="0-1,3-4",
            output_root=output,
            expected_width=6,
            expected_height=4,
        )
    assert not output.exists()


def test_export_accepts_authority_accepted_empty_binary_mask(tmp_path: Path) -> None:
    source_manifest = _source_fixture(tmp_path / "source", zero_frame=3)
    output = tmp_path / "exports" / "empty-mask-is-authority-available"
    output.parent.mkdir()

    manifest_path = exporter.export_layout(
        session_id="grap_a_cap_004",
        session_frame_count=7,
        source_run_manifest=source_manifest,
        preregistered_bilateral_ranges="0-1,3-5",
        output_root=output,
        expected_width=6,
        expected_height=4,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["selected_frame_indices"] == [3, 4, 5]
    assert (
        np.count_nonzero(np.asarray(Image.open(output / "left" / "frame_00003.png")))
        == 0
    )


@pytest.mark.parametrize("profile", ["q", "legacy-dual"])
def test_export_accepts_only_literal_legacy_source_profiles(
    tmp_path: Path, profile: str
) -> None:
    source_manifest = _source_fixture(tmp_path / "source", profile=profile)
    output = tmp_path / "exports" / "legacy-layout"
    output.parent.mkdir()

    manifest_path = exporter.export_layout(
        session_id="grap_a_cap_004",
        session_frame_count=7,
        source_run_manifest=source_manifest,
        preregistered_bilateral_ranges="0-1,3-5",
        output_root=output,
        expected_width=6,
        expected_height=4,
    )
    layout = runner.preflight_development_mask_layout(
        manifest_path,
        mask_root=output,
        expected_session_id="grap_a_cap_004",
        expected_session_frame_count=7,
        expected_target_frames=(3, 4, 5),
    )
    assert layout is not None
    assert layout.selected_frames == (3, 4, 5)


def test_export_preserves_reject_as_unavailable_not_a_mask(tmp_path: Path) -> None:
    source_manifest = _source_fixture(tmp_path / "source", unavailable_status="REJECT")
    output = tmp_path / "exports" / "reject-layout"
    output.parent.mkdir()

    manifest_path = exporter.export_layout(
        session_id="grap_a_cap_004",
        session_frame_count=7,
        source_run_manifest=source_manifest,
        preregistered_bilateral_ranges="0-1,3-5",
        output_root=output,
        expected_width=6,
        expected_height=4,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_frame_closure"][2]["sides"]["right"] == {
        "status": "REJECT",
        "source_mask": None,
    }
    assert manifest["source_frame_closure"][2]["bilateral_available"] is False
    assert runner.preflight_development_mask_layout(
        manifest_path,
        mask_root=output,
        expected_session_id="grap_a_cap_004",
        expected_session_frame_count=7,
        expected_target_frames=(3, 4, 5),
    )


def test_export_explicit_skips_remain_denominator_pending_without_audit(
    tmp_path: Path,
) -> None:
    source_manifest = _source_fixture(tmp_path / "source", skipped_frames=(6,))
    output = tmp_path / "exports" / "skipped-layout"
    output.parent.mkdir()

    manifest_path = exporter.export_layout(
        session_id="grap_a_cap_004",
        session_frame_count=7,
        source_run_manifest=source_manifest,
        preregistered_bilateral_ranges="0-1,3-5",
        output_root=output,
        expected_width=6,
        expected_height=4,
    )
    manifest = json.loads(manifest_path.read_text())
    skipped = manifest["source_frame_closure"][6]
    assert skipped["audit"] is None
    assert skipped["bilateral_available"] is False
    assert skipped["sides"] == {
        side: {"status": "SKIPPED_OBJECT6D_NONFINITE", "source_mask": None}
        for side in ("left", "right")
    }
    assert manifest["session_frame_count"] == 7
    assert manifest["pending_frame_indices"] == [0, 1, 2, 6]
    assert manifest["verification"]["source_audit_json_parse_count"] == 6
    assert runner.preflight_development_mask_layout(
        manifest_path,
        mask_root=output,
        expected_session_id="grap_a_cap_004",
        expected_session_frame_count=7,
        expected_target_frames=(3, 4, 5),
    )


def test_export_rejects_missing_audit_not_declared_skipped(tmp_path: Path) -> None:
    source_manifest = _source_fixture(tmp_path / "source")
    audit = source_manifest.parent / "d1_d4_live/frames/grap_a_cap_004/frame_00006.json"
    audit.unlink()
    output = tmp_path / "exports" / "must-not-exist"
    output.parent.mkdir()

    with pytest.raises(exporter.MaskExportError, match="audit .*inventory mismatch"):
        exporter.export_layout(
            session_id="grap_a_cap_004",
            session_frame_count=7,
            source_run_manifest=source_manifest,
            preregistered_bilateral_ranges="0-1,3-5",
            output_root=output,
            expected_width=6,
            expected_height=4,
        )
    assert not output.exists()


def test_export_rejects_audit_present_for_explicit_skip(tmp_path: Path) -> None:
    source_manifest = _source_fixture(tmp_path / "source", skipped_frames=(6,))
    source_audit = (
        source_manifest.parent / "d1_d4_live/frames/grap_a_cap_004/frame_00005.json"
    )
    skipped_audit = source_audit.with_name("frame_00006.json")
    value = json.loads(source_audit.read_text())
    value["frame_index"] = 6
    _json(skipped_audit, value)
    output = tmp_path / "exports" / "must-not-exist"
    output.parent.mkdir()

    with pytest.raises(exporter.MaskExportError, match="audit .*inventory mismatch"):
        exporter.export_layout(
            session_id="grap_a_cap_004",
            session_frame_count=7,
            source_run_manifest=source_manifest,
            preregistered_bilateral_ranges="0-1,3-5",
            output_root=output,
            expected_width=6,
            expected_height=4,
        )
    assert not output.exists()


def test_export_rejects_legacy_d4_status_near_miss(tmp_path: Path) -> None:
    source_manifest = _source_fixture(tmp_path / "source", profile="legacy-dual")
    audit_path = (
        source_manifest.parent / "d1_d4_live/frames/grap_a_cap_004/frame_00000.json"
    )
    audit = json.loads(audit_path.read_text())
    audit["sides"]["left"]["d4_application_status"] = "None"
    _json(audit_path, audit)
    output = tmp_path / "exports" / "must-not-exist"
    output.parent.mkdir()

    with pytest.raises(runner.CleanRunError, match="legacy D4 audit authority"):
        exporter.export_layout(
            session_id="grap_a_cap_004",
            session_frame_count=7,
            source_run_manifest=source_manifest,
            preregistered_bilateral_ranges="0-1,3-5",
            output_root=output,
            expected_width=6,
            expected_height=4,
        )
    assert not output.exists()


def test_export_rejects_accept_without_real_file_before_output(tmp_path: Path) -> None:
    source_manifest = _source_fixture(tmp_path / "source")
    missing = (
        source_manifest.parent
        / "d1_d4_live"
        / "formal_masks"
        / "task26_d1_d4"
        / "left"
        / "grap_a_cap_004"
        / "frame_00003.png"
    )
    missing.unlink()
    output = tmp_path / "exports" / "must-not-exist"
    output.parent.mkdir()

    with pytest.raises(runner.CleanRunError, match="cannot be opened no-follow"):
        exporter.export_layout(
            session_id="grap_a_cap_004",
            session_frame_count=7,
            source_run_manifest=source_manifest,
            preregistered_bilateral_ranges="0-1,3-5",
            output_root=output,
            expected_width=6,
            expected_height=4,
        )
    assert not output.exists()


def test_runner_preflight_rejects_canonical_payload_tamper(tmp_path: Path) -> None:
    _, output, manifest_path = _export(tmp_path)
    target = output / "right" / "frame_00004.png"
    target.write_bytes(target.read_bytes() + b"tamper")

    with pytest.raises(runner.CleanRunError, match="bytes/SHA-256/inode mismatch"):
        runner.preflight_development_mask_layout(
            manifest_path,
            mask_root=output,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=7,
            expected_target_frames=(3, 4, 5),
        )


def test_runner_preflight_rejects_unmanifested_canonical_file(tmp_path: Path) -> None:
    _, output, manifest_path = _export(tmp_path)
    (output / "left" / "frame_00002.png").write_bytes(
        (output / "left" / "frame_00003.png").read_bytes()
    )

    with pytest.raises(runner.CleanRunError, match="side inventory mismatch"):
        runner.preflight_development_mask_layout(
            manifest_path,
            mask_root=output,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=7,
            expected_target_frames=(3, 4, 5),
        )


def test_runner_range_mismatch_fails_before_clean_output(tmp_path: Path) -> None:
    _, mask_root, manifest_path = _export(tmp_path)
    plane_path = tmp_path / "PLANE.json"
    _json(
        plane_path,
        {
            "schema_version": "table-plane-static-candidate-v1",
            "session_id": "grap_a_cap_004",
            "plumbing_only": True,
            "formal_consumer_allowed": False,
            "is_clean_candidate": False,
            "candidate_requires_human_review": True,
            "estimation_quality": {"pass": True},
            "plane": {
                "cardinality": 1,
                "per_frame_offsets_present": False,
                "normal": [0.0, 0.0, 1.0],
                "offset_m": -2.0,
            },
        },
    )
    output = tmp_path / "clean-must-not-exist"
    args = argparse.Namespace(
        execution_mode="DEVELOPMENT_ONLY",
        formal_authority_manifest=None,
        expected_cohort_id=None,
        raw_root=tmp_path / "raw-not-needed-for-fast-failure",
        mask_root=mask_root,
        mask_source_manifest=manifest_path,
        plane=plane_path,
        session_id="grap_a_cap_004",
        output_root=output,
        start_frame=0,
        frame_count=2,
        session_frame_count=7,
    )

    with pytest.raises(runner.CleanRunError, match="requested frame range differs"):
        runner.run(args)
    assert not output.exists()


def test_export_rejects_hardlink_alias_before_output(tmp_path: Path) -> None:
    source_manifest = _source_fixture(tmp_path / "source")
    source_root = source_manifest.parent
    left = (
        source_root
        / "d1_d4_live/formal_masks/task26_d1_d4/left/grap_a_cap_004"
        / "frame_00003.png"
    )
    right = (
        source_root
        / "d1_d4_live/formal_masks/task26_d1_d4/right/grap_a_cap_004"
        / "frame_00003.png"
    )
    right.unlink()
    os.link(left, right)
    output = tmp_path / "exports" / "must-not-exist"
    output.parent.mkdir()

    with pytest.raises(runner.CleanRunError, match="single-link"):
        exporter.export_layout(
            session_id="grap_a_cap_004",
            session_frame_count=7,
            source_run_manifest=source_manifest,
            preregistered_bilateral_ranges="0-1,3-5",
            output_root=output,
            expected_width=6,
            expected_height=4,
        )
    assert not output.exists()


@pytest.mark.parametrize("kind", ["audit", "mask"])
def test_export_rejects_extra_source_inventory_before_output(
    tmp_path: Path, kind: str
) -> None:
    source_manifest = _source_fixture(tmp_path / "source")
    if kind == "audit":
        _json(
            source_manifest.parent
            / "d1_d4_live/frames/grap_a_cap_004/frame_99999.json",
            {},
        )
    else:
        source = (
            source_manifest.parent
            / "d1_d4_live/formal_masks/task26_d1_d4/left/grap_a_cap_004"
            / "frame_00000.png"
        )
        extra = source.with_name("frame_99999.png")
        extra.write_bytes(source.read_bytes())
    output = tmp_path / "exports" / "must-not-exist"
    output.parent.mkdir()

    with pytest.raises(
        (exporter.MaskExportError, runner.CleanRunError), match="inventory"
    ):
        exporter.export_layout(
            session_id="grap_a_cap_004",
            session_frame_count=7,
            source_run_manifest=source_manifest,
            preregistered_bilateral_ranges="0-1,3-5",
            output_root=output,
            expected_width=6,
            expected_height=4,
        )
    assert not output.exists()


def test_export_rolls_back_private_stage_on_mid_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest = _source_fixture(tmp_path / "source")
    parent = tmp_path / "exports"
    parent.mkdir()
    output = parent / "must-not-exist"
    original = runner.AtomicDevelopmentDirectory.write
    calls = 0

    def fail_second_write(
        self: runner.AtomicDevelopmentDirectory,
        component: str | None,
        name: str,
        payload: bytes,
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise runner.CleanRunError("synthetic mid-write failure")
        return original(self, component, name, payload)

    monkeypatch.setattr(runner.AtomicDevelopmentDirectory, "write", fail_second_write)
    with pytest.raises(exporter.MaskExportError, match="synthetic mid-write failure"):
        exporter.export_layout(
            session_id="grap_a_cap_004",
            session_frame_count=7,
            source_run_manifest=source_manifest,
            preregistered_bilateral_ranges="0-1,3-5",
            output_root=output,
            expected_width=6,
            expected_height=4,
        )
    assert not output.exists()
    assert not list(parent.glob(".clean-stage-*.tmp"))


def test_cpfs_einval_uses_locked_same_parent_noreplace_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest = _source_fixture(tmp_path / "source")
    output = tmp_path / "exports" / "cpfs-compatible-layout"
    output.parent.mkdir()

    def cpfs_unsupported(*_args: object) -> None:
        raise OSError(errno.EINVAL, "synthetic CPFS renameat2 rejection")

    monkeypatch.setattr(runner, "_renameat2_noreplace", cpfs_unsupported)
    manifest_path = exporter.export_layout(
        session_id="grap_a_cap_004",
        session_frame_count=7,
        source_run_manifest=source_manifest,
        preregistered_bilateral_ranges="0-1,3-5",
        output_root=output,
        expected_width=6,
        expected_height=4,
    )

    assert manifest_path == output / "MASK_LAYOUT_MANIFEST.json"
    assert manifest_path.is_file()
    assert not list(output.parent.glob(".clean-stage-*.tmp"))


def test_cpfs_fallback_refuses_foreign_target_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest = _source_fixture(tmp_path / "source")
    output = tmp_path / "exports" / "foreign-target"
    output.parent.mkdir()
    original_flock = runner.fcntl.flock
    injected = False

    def cpfs_unsupported(*_args: object) -> None:
        raise OSError(errno.EINVAL, "synthetic CPFS renameat2 rejection")

    def inject_target(descriptor: int, operation: int) -> None:
        nonlocal injected
        if operation == runner.fcntl.LOCK_EX | runner.fcntl.LOCK_NB and not injected:
            output.mkdir()
            (output / "FOREIGN.txt").write_text("must survive")
            injected = True
        original_flock(descriptor, operation)

    monkeypatch.setattr(runner, "_renameat2_noreplace", cpfs_unsupported)
    monkeypatch.setattr(runner.fcntl, "flock", inject_target)
    with pytest.raises(FileExistsError):
        exporter.export_layout(
            session_id="grap_a_cap_004",
            session_frame_count=7,
            source_run_manifest=source_manifest,
            preregistered_bilateral_ranges="0-1,3-5",
            output_root=output,
            expected_width=6,
            expected_height=4,
        )

    assert (output / "FOREIGN.txt").read_text() == "must survive"
    assert set(path.name for path in output.iterdir()) == {"FOREIGN.txt"}
    assert not list(output.parent.glob(".clean-stage-*.tmp"))


def test_cpfs_einval_publishes_regular_single_link_video_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "CLEAN_DEVELOPMENT_ONLY.mp4"
    clean_payloads = [
        runner.encode_png(np.full((4, 6, 3), value, dtype=np.uint8), "RGB")
        for value in (17, 83)
    ]

    def cpfs_unsupported(*_args: object) -> None:
        raise OSError(errno.EINVAL, "synthetic CPFS renameat2 rejection")

    monkeypatch.setattr(runner, "_renameat2_noreplace", cpfs_unsupported)
    reference = runner.encode_video(
        clean_payloads,
        output,
        width=6,
        height=4,
        fps=30,
    )

    info = output.stat()
    assert info.st_nlink == 1
    assert reference["path"] == str(output)
    assert reference["bytes"] == info.st_size
    assert reference["frames"] == 2
    assert reference["full_decode"] is True
    assert not list(tmp_path.glob(".*.partial-*.mp4"))


def test_cpfs_regular_file_fallback_refuses_foreign_target_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / ".video.partial.mp4"
    target = tmp_path / "video.mp4"
    source.write_bytes(b"private encoded video")
    target.write_bytes(b"foreign video must survive")
    source_before = source.stat()
    target_before = target.stat()

    def cpfs_unsupported(*_args: object) -> None:
        raise OSError(errno.EINVAL, "synthetic CPFS renameat2 rejection")

    monkeypatch.setattr(runner, "_renameat2_noreplace", cpfs_unsupported)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd = os.open(tmp_path, flags)
    try:
        with pytest.raises(FileExistsError):
            runner._rename_noreplace(
                parent_fd,
                source.name,
                parent_fd,
                target.name,
            )
    finally:
        os.close(parent_fd)

    assert source.read_bytes() == b"private encoded video"
    assert target.read_bytes() == b"foreign video must survive"
    assert (source.stat().st_dev, source.stat().st_ino) == (
        source_before.st_dev,
        source_before.st_ino,
    )
    assert (target.stat().st_dev, target.stat().st_ino) == (
        target_before.st_dev,
        target_before.st_ino,
    )


def test_export_rejects_injected_extra_output_before_atomic_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest = _source_fixture(tmp_path / "source")
    parent = tmp_path / "exports"
    parent.mkdir()
    output = parent / "must-not-exist"
    original = runner.AtomicDevelopmentDirectory.write_json

    def inject_extra(
        self: runner.AtomicDevelopmentDirectory,
        component: str | None,
        name: str,
        value: dict[str, object],
        *,
        compact: bool = False,
    ) -> tuple[dict[str, object], bytes]:
        result = original(self, component, name, value, compact=compact)
        if name == "MASK_LAYOUT_MANIFEST.json":
            self.write(None, "UNMANIFESTED.bin", b"x")
        return result

    monkeypatch.setattr(runner.AtomicDevelopmentDirectory, "write_json", inject_extra)
    with pytest.raises(exporter.MaskExportError, match="root inventory mismatch"):
        exporter.export_layout(
            session_id="grap_a_cap_004",
            session_frame_count=7,
            source_run_manifest=source_manifest,
            preregistered_bilateral_ranges="0-1,3-5",
            output_root=output,
            expected_width=6,
            expected_height=4,
        )
    assert not output.exists()
    assert not list(parent.glob(".clean-stage-*.tmp"))


def test_export_rejects_source_root_swap_via_held_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest = _source_fixture(tmp_path / "source")
    source_root = source_manifest.parent
    moved_root = tmp_path / "source-original"
    output = tmp_path / "exports" / "must-not-exist"
    output.parent.mkdir()
    original = runner.StrictDevelopmentTree.read
    swapped = False

    def swap_after_first_audit(
        self: runner.StrictDevelopmentTree, relative: Path | str, label: str
    ) -> tuple[runner.BoundArtifact, bytes]:
        nonlocal swapped
        result = original(self, relative, label)
        if (
            not swapped
            and self.root == source_root
            and str(relative).endswith("frame_00000.json")
        ):
            source_root.rename(moved_root)
            source_root.mkdir()
            swapped = True
        return result

    monkeypatch.setattr(runner.StrictDevelopmentTree, "read", swap_after_first_audit)
    with pytest.raises(runner.CleanRunError, match="ancestor identity changed"):
        exporter.export_layout(
            session_id="grap_a_cap_004",
            session_frame_count=7,
            source_run_manifest=source_manifest,
            preregistered_bilateral_ranges="0-1,3-5",
            output_root=output,
            expected_width=6,
            expected_height=4,
        )
    assert not output.exists()


@pytest.mark.parametrize("tamper", ["manifest-inode", "same-bytes-replacement"])
def test_runner_preflight_rejects_exact_identity_tamper(
    tmp_path: Path, tamper: str
) -> None:
    _, output, manifest_path = _export(tmp_path)
    if tamper == "manifest-inode":
        manifest = json.loads(manifest_path.read_text())
        manifest["adapter_implementation"]["inode"] += 1
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    else:
        target = output / "left" / "frame_00004.png"
        payload = target.read_bytes()
        replacement = target.with_name("replacement.png")
        replacement.write_bytes(payload)
        os.replace(replacement, target)

    with pytest.raises(runner.CleanRunError, match="inode mismatch"):
        runner.preflight_development_mask_layout(
            manifest_path,
            mask_root=output,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=7,
            expected_target_frames=(3, 4, 5),
        )


def test_runner_canonical_success_uses_only_cached_selected_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, output, _ = _development_run_fixture(tmp_path)

    def forbidden_path_mask_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("combined_mask path reread is forbidden")

    monkeypatch.setattr(runner, "combined_mask", forbidden_path_mask_read)
    manifest_path = runner.run(args)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["terminal"] is True
    assert manifest["target_frames"] == [3, 4, 5]
    assert manifest["pending_frame_indices"] == [0, 1, 2, 6]
    assert manifest["donor_policy"]["inventory_frame_indices"] == [3, 4, 5]
    assert set(manifest["donor_policy"]["donor_frames_read"]) <= {3, 4, 5}
    assert ".clean-stage-" not in manifest_path.read_text()
    assert output.exists()
    assert not list(output.parent.glob(".clean-stage-*.tmp"))


def test_runner_canonical_clean_shard_accounts_true_session_complement(
    tmp_path: Path,
) -> None:
    args, output, _ = _development_run_fixture(tmp_path)
    args.frame_count = 2

    manifest_path = runner.run(args)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["target_frames"] == [3, 4]
    assert manifest["pending_frame_indices"] == [0, 1, 2, 5, 6]
    assert manifest["pending_frame_count"] == 5
    assert manifest["partial_session_status"] == "PENDING_OUTSIDE_THIS_CLEAN_SHARD"
    assert manifest["donor_policy"]["inventory_frame_indices"] == [3, 4, 5]
    assert manifest["verification"]["session_denominator_accounted"] is True
    assert output.exists()


def test_preflight_rejects_noncontiguous_subset_of_selected_interval(
    tmp_path: Path,
) -> None:
    _, output, manifest_path = _export(tmp_path)

    with pytest.raises(runner.CleanRunError, match="selected contiguous interval"):
        runner.preflight_development_mask_layout(
            manifest_path,
            mask_root=output,
            expected_session_id="grap_a_cap_004",
            expected_session_frame_count=7,
            expected_target_frames=(3, 5),
        )


def test_runner_raw_failure_and_mid_output_failure_leave_no_public_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, output, raw_root = _development_run_fixture(tmp_path)
    (raw_root / "00004" / "rgb.png").unlink()
    with pytest.raises(runner.CleanRunError, match="cannot be opened no-follow"):
        runner.run(args)
    assert not output.exists()
    assert not list(output.parent.glob(".clean-stage-*.tmp"))

    args, output, _ = _development_run_fixture(tmp_path / "second")

    def fail_video(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise runner.CleanRunError("synthetic video failure")

    monkeypatch.setattr(runner, "encode_video", fail_video)
    with pytest.raises(runner.CleanRunError, match="synthetic video failure"):
        runner.run(args)
    assert not output.exists()
    assert not list(output.parent.glob(".clean-stage-*.tmp"))
