from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "tools/run_quad_review_sample.py"
PENDING_RUNNER = PROJECT / "tools/run_quad_pending_lineage_manifest.py"
SPEC = importlib.util.spec_from_file_location("quad_review_sample_lineage_tested", RUNNER)
assert SPEC and SPEC.loader
quad = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quad
SPEC.loader.exec_module(quad)


def _payload_ref(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _json_ref(path: Path, value: object) -> dict[str, object]:
    return _payload_ref(path, (json.dumps(value, sort_keys=True) + "\n").encode())


def _png_ref(path: Path, image: np.ndarray) -> dict[str, object]:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return _payload_ref(path, encoded.tobytes())


def _fixture(tmp_path: Path, frame_count: int = 2) -> tuple[dict[str, object], Path]:
    candidate_root = tmp_path / "project" / "_run" / "candidate"
    raw_root = candidate_root / "raw"
    key_root = candidate_root / "W1"
    lineage_root = tmp_path / "project" / "_run" / "lineage"
    per_frame = [{"frame_index": index, "sides": {}} for index in range(frame_count)]
    per_frame_ref = _json_ref(key_root / "PER_FRAME.json", per_frame)
    candidate = {
        "session": "grap_a_cap_004",
        "frame_count": frame_count,
        "claim_limit": "CANDIDATE_REQUIRES_HUMAN_REVIEW_NOT_A_MASK_PASS",
        "formal_consumer_allowed": False,
        "preflight": {"inputs_root": str(raw_root.resolve())},
        "prompt_candidates": {"W1": {}},
    }
    candidate_ref = _json_ref(candidate_root / "MANIFEST.json", candidate)
    frames: list[dict[str, object]] = []
    for index in range(frame_count):
        raw = np.full((6, 8, 3), (index * 30, 80, 120), dtype=np.uint8)
        left = np.zeros((6, 8), dtype=np.uint8)
        right = np.zeros((6, 8), dtype=np.uint8)
        left[1:3, 1:3] = 255
        right[3:5, 5:7] = 255
        raw_ref = _png_ref(raw_root / f"{index:05d}.png", raw)
        left_ref = _png_ref(
            key_root / "union_masks" / "left" / f"frame_{index:05d}.png", left
        )
        right_ref = _png_ref(
            key_root / "union_masks" / "right" / f"frame_{index:05d}.png", right
        )
        mask_ref = _png_ref(
            lineage_root / "mask_candidate_display" / f"{index:05d}.png",
            cv2.bitwise_or(left, right),
        )
        frames.append(
            {
                "frame_index": index,
                "raw": raw_ref,
                "mask": {
                    "status": quad.MASK_STATUS,
                    "representation": "DISPLAY_ONLY_BOOLEAN_OR_OF_CANDIDATE_SIDE_MASKS",
                    "source_raw_sha256": raw_ref["sha256"],
                    "image": mask_ref,
                    "sources": {"left": left_ref, "right": right_ref},
                },
                "clean": {"status": "PENDING", "reason": quad.CLEAN_PENDING},
                "robot": {"status": "PENDING", "reason": quad.ROBOT_PENDING},
            }
        )
    manifest: dict[str, object] = {
        "schema_version": "quad-frame-lineage-v1",
        "created_at": "2026-08-30T00:00:00+08:00",
        "session_id": "grap_a_cap_004",
        "frame_count": frame_count,
        "fps": 30,
        "resolution": [8, 6],
        "lineage_id": "",
        "claim_limit": "LAYOUT_REVIEW_ONLY_NO_ACCEPTANCE_OR_FORMAL_CONSUMPTION",
        "mask_claim": quad.MASK_STATUS,
        "source_producer": {
            "candidate_manifest": candidate_ref,
            "candidate_per_frame": per_frame_ref,
            "candidate_key": "W1",
            "candidate_claim_limit": candidate["claim_limit"],
        },
        "panel_policy": {
            "raw": "EXACT_SOURCE_FRAME",
            "mask": "CURRENT_CANDIDATE_OVERLAY_NOT_ACCEPTED",
            "clean": "SAME_LINEAGE_OR_EXPLICIT_PENDING_NO_HISTORICAL_FALLBACK",
            "robot": "SAME_LINEAGE_OR_EXPLICIT_PENDING",
        },
        "frames": frames,
    }
    manifest["lineage_id"] = quad.compute_lineage_id(manifest)
    manifest_path = lineage_root / "LINEAGE_MANIFEST.json"
    _json_ref(manifest_path, manifest)
    return manifest, manifest_path


def _ref_existing_copy(path: Path, source_ref: dict[str, object]) -> dict[str, object]:
    return _payload_ref(path, Path(str(source_ref["path"])).read_bytes())


def _attach_ready_clean_and_robot(
    manifest: dict[str, object], manifest_path: Path
) -> None:
    for frame in manifest["frames"]:
        index = frame["frame_index"]
        raw_sha = frame["raw"]["sha256"]
        mask_sha = frame["mask"]["image"]["sha256"]
        clean_ref = _png_ref(
            manifest_path.parent / "clean" / f"{index:05d}.png",
            np.full((6, 8, 3), (20, 30, 40), dtype=np.uint8),
        )
        unresolved_ref = _png_ref(
            manifest_path.parent / "unresolved" / f"{index:05d}.png",
            np.zeros((6, 8), dtype=np.uint8),
        )
        clean_provenance = {
            "target_frame": index,
            "source_raw_sha256": raw_sha,
            "consumed_mask_sha256": mask_sha,
            "clean_rgb_sha256": clean_ref["sha256"],
            "unresolved_sha256": unresolved_ref["sha256"],
        }
        clean_provenance_ref = _json_ref(
            manifest_path.parent / "clean_provenance" / f"{index:05d}.json",
            clean_provenance,
        )
        frame["clean"] = {
            "status": "READY_LINEAGE_BOUND",
            "image": clean_ref,
            "unresolved": unresolved_ref,
            "provenance": clean_provenance_ref,
            "source_raw_sha256": raw_sha,
            "consumed_mask_sha256": mask_sha,
        }
        robot_ref = _png_ref(
            manifest_path.parent / "robot" / f"{index:05d}.png",
            np.full((6, 8, 3), (50, 60, 70), dtype=np.uint8),
        )
        robot_provenance = {
            "target_frame": index,
            "source_raw_sha256": raw_sha,
            "source_mask_sha256": mask_sha,
            "source_clean_sha256": clean_ref["sha256"],
            "robot_rgb_sha256": robot_ref["sha256"],
        }
        robot_provenance_ref = _json_ref(
            manifest_path.parent / "robot_provenance" / f"{index:05d}.json",
            robot_provenance,
        )
        frame["robot"] = {
            "status": "READY_LINEAGE_BOUND",
            "image": robot_ref,
            "provenance": robot_provenance_ref,
            "source_raw_sha256": raw_sha,
            "source_mask_sha256": mask_sha,
            "source_clean_sha256": clean_ref["sha256"],
        }
    manifest["lineage_id"] = quad.compute_lineage_id(manifest)


def test_valid_pending_manifest_is_one_lineage_and_explicitly_nonaccepted(
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _fixture(tmp_path)
    stats = quad.validate_lineage_manifest(manifest, manifest_path=manifest_path)
    assert stats == {
        "clean_pending": 2,
        "clean_ready": 0,
        "robot_pending": 2,
        "robot_ready": 0,
    }
    assert manifest["mask_claim"] == "CURRENT_CANDIDATE_NOT_ACCEPTED"


def test_ready_panels_must_and_can_bind_the_exact_per_frame_chain(tmp_path: Path) -> None:
    manifest, manifest_path = _fixture(tmp_path)
    _attach_ready_clean_and_robot(manifest, manifest_path)
    stats = quad.validate_lineage_manifest(manifest, manifest_path=manifest_path)
    assert stats == {
        "clean_pending": 0,
        "clean_ready": 2,
        "robot_pending": 0,
        "robot_ready": 2,
    }


def test_arbitrary_raw_mix_is_rejected_even_when_attacker_rehashes(
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _fixture(tmp_path)
    tampered = copy.deepcopy(manifest)
    original = tampered["frames"][0]["raw"]
    mixed_ref = _ref_existing_copy(tmp_path / "other_lineage" / "00000.png", original)
    tampered["frames"][0]["raw"] = mixed_ref
    tampered["frames"][0]["mask"]["source_raw_sha256"] = mixed_ref["sha256"]
    tampered["lineage_id"] = quad.compute_lineage_id(tampered)
    with pytest.raises(quad.QuadSampleError, match="raw path"):
        quad.validate_lineage_manifest(tampered, manifest_path=manifest_path)


def test_arbitrary_mask_mix_is_rejected_even_when_bytes_match(
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _fixture(tmp_path)
    tampered = copy.deepcopy(manifest)
    original = tampered["frames"][0]["mask"]["sources"]["left"]
    mixed_ref = _ref_existing_copy(tmp_path / "other_candidate" / "frame_00000.png", original)
    tampered["frames"][0]["mask"]["sources"]["left"] = mixed_ref
    tampered["lineage_id"] = quad.compute_lineage_id(tampered)
    with pytest.raises(quad.QuadSampleError, match="sources.left path"):
        quad.validate_lineage_manifest(tampered, manifest_path=manifest_path)


def test_historical_clean_cannot_replace_explicit_pending(tmp_path: Path) -> None:
    manifest, manifest_path = _fixture(tmp_path)
    tampered = copy.deepcopy(manifest)
    tampered["frames"][0]["clean"]["reason"] = "HISTORICAL_CLEAN_DISPLAY_ONLY"
    tampered["lineage_id"] = quad.compute_lineage_id(tampered)
    with pytest.raises(quad.QuadSampleError, match="dishonest pending reason"):
        quad.validate_lineage_manifest(tampered, manifest_path=manifest_path)


def test_clean_must_bind_exact_mask_sha_consumed_on_that_frame(tmp_path: Path) -> None:
    manifest, manifest_path = _fixture(tmp_path)
    tampered = copy.deepcopy(manifest)
    dummy = {"path": str((tmp_path / "unused").resolve()), "bytes": 1, "sha256": "a" * 64}
    tampered["frames"][0]["clean"] = {
        "status": "READY_LINEAGE_BOUND",
        "image": dummy,
        "unresolved": dummy,
        "provenance": dummy,
        "source_raw_sha256": tampered["frames"][0]["raw"]["sha256"],
        "consumed_mask_sha256": "f" * 64,
    }
    tampered["lineage_id"] = quad.compute_lineage_id(tampered)
    with pytest.raises(quad.QuadSampleError, match="CLEAN consumed mask SHA"):
        quad.validate_lineage_manifest(tampered, manifest_path=manifest_path)


def test_mask_cannot_claim_accepted_or_formal_status(tmp_path: Path) -> None:
    manifest, manifest_path = _fixture(tmp_path)
    tampered = copy.deepcopy(manifest)
    tampered["frames"][0]["mask"]["status"] = "ACCEPTED"
    tampered["lineage_id"] = quad.compute_lineage_id(tampered)
    with pytest.raises(quad.QuadSampleError, match="falsely claims acceptance"):
        quad.validate_lineage_manifest(tampered, manifest_path=manifest_path)


def test_legacy_independent_panel_cli_is_gone() -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--help"], text=True, capture_output=True, check=True
    )
    assert "--lineage-manifest" in result.stdout
    assert "--raw" not in result.stdout
    assert "--mask-overlay" not in result.stdout
    assert "--clean" not in result.stdout


def _dual_producer_fixture(tmp_path: Path, frame_count: int = 2) -> tuple[Path, Path]:
    session = "grap_a_cap_002"
    candidate_root = tmp_path / "project" / "_run" / "dual_candidate"
    component = candidate_root / "d1_d4_live"
    raw_root = tmp_path / "egodata" / session / "preprocess" / "all_data"
    raw_records: list[dict[str, object]] = []
    for index in range(frame_count):
        raw = np.full((6, 8, 3), (index * 30, 80, 120), dtype=np.uint8)
        raw_ref = _png_ref(raw_root / f"{index:05d}" / "rgb.png", raw)
        raw_records.append(
            {
                "frame_index": index,
                **raw_ref,
                "full_decode": True,
                "width": 8,
                "height": 6,
                "mode": "RGB",
            }
        )
        masks = {
            "left": np.pad(np.full((2, 2), 255, dtype=np.uint8), ((1, 3), (1, 5))),
            "right": np.pad(np.full((2, 2), 255, dtype=np.uint8), ((3, 1), (5, 1))),
        }
        inventory: list[dict[str, object]] = []
        selections: dict[str, dict[str, object]] = {}
        for offset, side in enumerate(("left", "right")):
            mask = masks[side]
            _png_ref(
                component
                / "formal_masks"
                / "task26_d1_d4"
                / side
                / session
                / f"frame_{index:05d}.png",
                mask,
            )
            mask_digest = quad._mask_sha256(mask)
            raw_evidence = {
                "frame_id": f"{session}:{index}",
                "instance_id": offset + 10,
                "mask_sha256": mask_digest,
                "source_kind": "SAM_RAW_INSTANCE",
            }
            raw_evidence_ref = _json_ref(
                component
                / "raw_evidence"
                / session
                / f"frame_{index:05d}_offset_{offset:03d}.json",
                raw_evidence,
            )
            inventory.append(
                {
                    "raw_instance_offset": offset,
                    "instance_id": offset + 10,
                    "mask_sha256": mask_digest,
                    "area": int(np.count_nonzero(mask)),
                    "shared_raw_source": {
                        **raw_evidence_ref,
                        "source_kind": "SAM_RAW_INSTANCE",
                    },
                }
            )
            selections[side] = {
                "task26_d1_d4": {
                    "status": "ACCEPT",
                    "accepted_offset": offset,
                    "accepted_stream": "PROPAGATE" if side == "left" else "PER_FRAME",
                    "accepted": True,
                    "area": int(np.count_nonzero(mask)),
                    "reasons": [],
                }
            }
        _json_ref(
            component / "raw_inventory" / session / f"frame_{index:05d}.json",
            inventory,
        )
        _json_ref(
            component / "frames" / session / f"frame_{index:05d}.json",
            {"session_id": session, "frame_index": index, "sides": selections},
        )
    raw_manifest = {
        "schema_version": "eligible68-dual-raw-input-context-v1",
        "session_count": 1,
        "frame_count": frame_count,
        "no_fallback": True,
        "labels_read": 0,
        "forbidden_session_frames_read": 0,
        "sessions": {session: {"frame_count": frame_count, "frames": raw_records}},
    }
    raw_manifest_path = tmp_path / "context" / session / "RAW_INPUT_MANIFEST.json"
    _json_ref(raw_manifest_path, raw_manifest)
    candidate = {
        "schema_version": "claude-d1-d4-live-producer-v1",
        "status": "COMPLETE_UNREVIEWED",
        "detection_mode": "dual",
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "sessions": {session: {"frames_processed": frame_count}},
    }
    candidate_path = candidate_root / "MANIFEST.json"
    _json_ref(candidate_path, candidate)
    return candidate_path, raw_manifest_path


def _generate_dual_lineage(tmp_path: Path) -> tuple[dict[str, object], Path]:
    candidate_path, raw_manifest_path = _dual_producer_fixture(tmp_path)
    output = tmp_path / "project" / "_run" / "dual_lineage"
    subprocess.run(
        [
            sys.executable,
            str(PENDING_RUNNER),
            "--candidate-manifest",
            str(candidate_path),
            "--candidate-key",
            "task26_d1_d4",
            "--session-id",
            "grap_a_cap_002",
            "--raw-input-manifest",
            str(raw_manifest_path),
            "--output-dir",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    lineage_path = output / "LINEAGE_MANIFEST.json"
    return json.loads(lineage_path.read_text()), lineage_path


def test_dual_review_layout_binds_manifest_raw_decision_inventory_and_masks(
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _generate_dual_lineage(tmp_path)
    stats = quad.validate_lineage_manifest(manifest, manifest_path=manifest_path)
    assert stats == {
        "clean_pending": 2,
        "clean_ready": 0,
        "robot_pending": 2,
        "robot_ready": 0,
    }
    assert manifest["source_producer"]["source_layout"] == quad.DUAL_SOURCE_LAYOUT
    assert all(
        set(frame["mask"]["candidate_evidence"]) == {
            "decision",
            "raw_inventory",
            "selected",
        }
        for frame in manifest["frames"]
    )


def test_dual_review_rejects_selected_offset_drift_even_when_lineage_is_rehashed(
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _generate_dual_lineage(tmp_path)
    tampered = copy.deepcopy(manifest)
    tampered["frames"][0]["mask"]["candidate_evidence"]["selected"]["left"][
        "accepted_offset"
    ] = 1
    tampered["lineage_id"] = quad.compute_lineage_id(tampered)
    with pytest.raises(quad.QuadSampleError, match="accepted selection"):
        quad.validate_lineage_manifest(tampered, manifest_path=manifest_path)


def test_dual_review_rejects_mask_that_does_not_match_selected_raw_inventory(
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _generate_dual_lineage(tmp_path)
    tampered = copy.deepcopy(manifest)
    left_ref = tampered["frames"][0]["mask"]["sources"]["left"]
    changed = np.zeros((6, 8), dtype=np.uint8)
    changed[0:2, 0:2] = 255
    replacement = _png_ref(Path(str(left_ref["path"])), changed)
    tampered["frames"][0]["mask"]["sources"]["left"] = replacement
    tampered["lineage_id"] = quad.compute_lineage_id(tampered)
    with pytest.raises(quad.QuadSampleError, match="mask identity"):
        quad.validate_lineage_manifest(tampered, manifest_path=manifest_path)


def test_every_rendered_frame_has_visible_development_banner_at_bottom(
    tmp_path: Path,
) -> None:
    manifest, _ = _generate_dual_lineage(tmp_path)
    for frame in manifest["frames"]:
        rendered = quad._render_frame(frame, (6, 8))
        assert rendered.shape == (448, 1920, 3)
        banner = rendered[-quad.DEVELOPMENT_BANNER_HEIGHT :]
        assert float((banner[:, :, 2] > 70).mean()) > 0.98
        assert int(np.all(banner > 170, axis=2).sum()) > 1_500
        former_top_banner = rendered[: quad.DEVELOPMENT_BANNER_HEIGHT]
        assert float((former_top_banner[:, :, 2] > 70).mean()) < 0.98


def test_renderer_result_binds_artifact_success_and_zero_operation_counters(
    tmp_path: Path,
) -> None:
    _, lineage_path = _generate_dual_lineage(tmp_path)
    output = tmp_path / "project" / "_run" / "render" / "review.mp4"
    result_path = output.with_name("RESULT.json")
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--lineage-manifest",
            str(lineage_path),
            "--output",
            str(output),
            "--result-manifest",
            str(result_path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(result_path.read_text())
    assert result["completion_mode"] == "ARTIFACT_EXISTS"
    assert result["visible_banner_position"] == "BOTTOM_40_PIXELS_ALL_FRAMES"
    assert result["operation_counter_scope"] == (
        "QUAD_RENDERER_ONLY_SOURCE_PRODUCER_OPERATIONS_REMAIN_IN_LINEAGE"
    )
    assert result["operation_counters"] == quad.ZERO_OPERATION_COUNTERS
    assert result["operation_counters"]["connected_component_fill_calls"] == 0
    lineage = json.loads(lineage_path.read_text())
    assert any(
        frame["mask"]["candidate_evidence"]["selected"]["left"][
            "accepted_stream"
        ]
        == "PROPAGATE"
        for frame in lineage["frames"]
    )
    assert result["operation_counters"]["temporal_propagation_calls"] == 0
    assert [item["artifact"] for item in result["success_evidence"]] == [
        "LINEAGE_MANIFEST",
        "REVIEW_MP4",
    ]
    lineage_evidence, video_evidence = result["success_evidence"]
    assert lineage_evidence["json_parse"] is True
    assert lineage_evidence["exact_top_level_keys"] == sorted(
        quad.LINEAGE_MANIFEST_KEYS
    )
    assert video_evidence["bytes"] == output.stat().st_size > 0
    assert video_evidence["expected_frames"] == 2
    assert video_evidence["decoded_frames"] == 2
    assert video_evidence["full_decode"] is True
