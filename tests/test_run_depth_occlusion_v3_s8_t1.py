from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import pytest

from pipeline.depth_occlusion_v3 import supersampled_intrinsics
from tools.run_depth_occlusion_v3_s8_t1 import (
    INPUT_MANIFEST_SCHEMA,
    MOUNT_PROVENANCE,
    OBJECT6D_004,
    _array_digest,
    build_cpu_freeze,
    execute,
    validate_input_manifest,
)


PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "relative_script",
    [
        "tools/run_depth_occlusion_v3_s8_t1.py",
        "tools/render_robot_eevee_fullchain_worker.py",
    ],
)
def test_robot_cli_help_bootstraps_project_outside_repo(
    tmp_path: Path,
    relative_script: str,
) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""
    completed = subprocess.run(
        [sys.executable, str(PROJECT / relative_script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def _record(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _manifest() -> dict[str, object]:
    record = {"path": "_run/input.bin", "bytes": 1, "sha256": "a" * 64}
    return {
        "schema_version": INPUT_MANIFEST_SCHEMA,
        "session_id": "grap_a_cap_004",
        "frame_name": "00240",
        "frame_index": 240,
        "frame_bundle": dict(record),
        "object6d": dict(record),
        "source_lineage": {
            "clean": dict(record),
            "object_texture": dict(record),
            "robot_render": dict(record),
        },
        "object_texture_source": "raw_observation",
        "candidate_requires_human_review": True,
        "next_bucket_blocked": True,
        "advancement_authorized": False,
        "formal_consumer_allowed": False,
        "mount_provenance": MOUNT_PROVENANCE,
        "contact_infeasible": "UNMEASURED",
    }


def test_cpu_freeze_binds_frozen_spec_shared_io_and_deleted_history_truthfully() -> (
    None
):
    freeze = build_cpu_freeze(PROJECT)
    assert (
        freeze["status"]
        == "CPU_READY_WAITING_S7_AND_CLEAN_CANDIDATE_INPUTS_NO_EXECUTION"
    )
    assert freeze["shared_immutable_io"]["sha256"] == (
        "e9af8d38146dee0813d415330fb626a295482740b160435fac544a045d74e57e"
    )
    assert freeze["frozen_algorithm"]["occlusion_epsilon_m"] == 0.003
    assert freeze["frozen_algorithm"]["supersample"] == 2
    assert (
        freeze["historical_evidence"]["deleted_depth_video"]["not_reverified"] is True
    )
    assert freeze["historical_evidence"]["user_visual_status"] == (
        "USER_VISUAL_HOLD_THUMB_GEOMETRY"
    )
    assert set(freeze["execution_counters"].values()) == {0}


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"session_id": "grap_a_cap_025"}, "forbidden"),
        ({"object_texture_source": "cad_material"}, "RAW/donor"),
        ({"advancement_authorized": True}, "candidate-only"),
        ({"mount_provenance": "MECHANICAL_CALIBRATION"}, "visual-only"),
    ],
)
def test_input_manifest_forbidden_semantics_fail_closed(
    mutation: dict, match: str
) -> None:
    manifest = _manifest()
    manifest.update(mutation)
    with pytest.raises(Exception, match=match):
        validate_input_manifest(manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        {"frame_bundle": {"path": "_run/input.bin", "bytes": True, "sha256": "a" * 64}},
        {"frame_bundle": {"path": "_run/input.bin", "bytes": 1, "sha256": "z" * 64}},
    ],
)
def test_input_manifest_bound_record_shape_fails_closed(mutation: dict) -> None:
    manifest = _manifest()
    manifest.update(mutation)
    with pytest.raises(Exception, match="bound frame_bundle"):
        validate_input_manifest(manifest)


def test_cpu_candidate_execute_writes_source_and_depth_evidence_without_gpu() -> None:
    with tempfile.TemporaryDirectory(
        prefix="s8_test_", dir=PROJECT / "_run"
    ) as directory:
        root = Path(directory)
        height, width = 2, 3
        high = (height * 2, width * 2)
        camera = np.array([[3.0, 0, 1.0], [0, 3.0, 0.5], [0, 0, 1.0]])
        clean = np.zeros((height, width, 3), dtype=np.uint8)
        object_rgb = np.full((*high, 3), (200, 20, 10), dtype=np.uint8)
        robot_rgb = np.full((*high, 3), (10, 20, 200), dtype=np.uint8)
        robot_alpha = np.ones(high, dtype=np.float32)
        k2 = supersampled_intrinsics(camera)
        u = np.arange(high[1])[None, :]
        v = np.arange(high[0])[:, None]
        ray_norm = np.sqrt(
            1.0 + ((u - k2[0, 2]) / k2[0, 0]) ** 2 + ((v - k2[1, 2]) / k2[1, 1]) ** 2
        )
        robot_range = (2.0 * ray_norm).astype(np.float32)
        arrays = {
            "clean_rgb": clean,
            "object_rgb_2x": object_rgb,
            "robot_rgb_2x": robot_rgb,
            "robot_range_m_2x": robot_range,
            "robot_alpha_2x": robot_alpha,
            "camera_intrinsics": camera,
        }
        bundle_buffer = io.BytesIO()
        np.savez(
            bundle_buffer,
            schema_version=np.asarray("s8-depth-occlusion-v3-frame-bundle-v1"),
            session_id=np.asarray("grap_a_cap_004"),
            frame_name=np.asarray("00240"),
            frame_index=np.asarray(240, dtype=np.int64),
            object_texture_source=np.asarray("raw_observation"),
            **arrays,
        )
        bundle = root / "frame_bundle.npz"
        bundle.write_bytes(bundle_buffer.getvalue())
        lineage = {}
        for name in ("clean", "object_texture", "robot_render"):
            path = root / f"{name}.manifest"
            path.write_text(name, encoding="utf-8")
            lineage[name] = _record(path)
        manifest = _manifest()
        manifest["frame_bundle"] = _record(bundle)
        manifest["object6d"] = _record(PROJECT / OBJECT6D_004)
        manifest["source_lineage"] = lineage
        manifest["bundle_array_sha256"] = {
            name: _array_digest(array) for name, array in arrays.items()
        }
        manifest_path = root / "INPUT_MANIFEST.json"
        manifest_payload = (json.dumps(manifest, sort_keys=True) + "\n").encode()
        manifest_path.write_bytes(manifest_payload)
        result = execute(
            PROJECT,
            manifest_path,
            hashlib.sha256(manifest_payload).hexdigest(),
            len(manifest_payload),
            root / "output",
        )
        assert result["status"] == "CANDIDATE_REQUIRES_HUMAN_REVIEW_THUMB_GEOMETRY_HOLD"
        assert result["metrics"]["robotrgb_changed_from_clean_pixels"] > 0
        assert result["algorithm"]["object_index_role"] == "QA_ONLY_NOT_DECISION"
        decoded = cv2.imread(str(root / "output/robot_rgb.png"), cv2.IMREAD_COLOR)
        assert decoded.shape[:2] == (height, width)
        assert (root / "output/source_map.npy").is_file()
        assert (root / "output/depth_evidence_2x.npz").is_file()
