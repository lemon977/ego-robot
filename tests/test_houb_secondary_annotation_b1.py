from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from tools.build_houb_secondary_annotation_b1 import (
    SELECTED_FRAMES,
    build_package,
)
from tools.houb_secondary_annotation_app import SecondaryPackage
from tools.validate_houb_secondary_labels_b1 import validate


def _source(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "source"
    frames = []
    for benchmark_index, frame_index in enumerate(range(228, 243)):
        relative = f"annotation_assets/rgb/grap_a_cap_004/{frame_index:05d}.png"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (6, 4), (frame_index % 255, 20, 30))
        image.save(path)
        payload = path.read_bytes()
        frames.append(
            {
                "annotation_asset": {
                    "bytes": len(payload),
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "benchmark_index": benchmark_index,
                "frame_index": frame_index,
                "height": 4,
                "panel_role": "DEVELOPMENT_KNOWN_WINDOW",
                "session_id": "grap_a_cap_004",
                "width": 6,
            }
        )
    manifest_bytes = (json.dumps({"frames": frames}, sort_keys=True) + "\n").encode()
    (root / "BENCHMARK_MANIFEST.json").write_bytes(manifest_bytes)
    return root, hashlib.sha256(manifest_bytes).hexdigest()


def _write_second_pass_labels(root: Path) -> None:
    manifest = json.loads((root / "SECONDARY_ANNOTATION_MANIFEST.json").read_text())
    for frame in manifest["frames"]:
        path = root / frame["expected_secondary_label_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("P", (6, 4), 0)
        image.putpixel((0, 0), 1)
        image.putpixel((1, 0), 2)
        image.putpalette([0] * 768)
        image.save(path)


def test_builder_selects_fixed_eight_raw_only_frames_and_is_idempotent(
    tmp_path: Path,
) -> None:
    source, source_sha = _source(tmp_path)
    output = tmp_path / "secondary"
    manifest, manifest_sha = build_package(source, output, source_sha)
    second, second_sha = build_package(source, output, source_sha)
    encoded = (output / "SECONDARY_ANNOTATION_MANIFEST.json").read_text()

    assert manifest == second
    assert manifest_sha == second_sha
    assert tuple(manifest["selection"]["frames"]) == SELECTED_FRAMES
    assert manifest["first_pass_independence"]["ui_input"] == "RAW_ONLY"
    assert manifest["first_pass_independence"]["first_pass_label_pixels_read"] is False
    assert "expected_human_label_path" not in encoded
    assert "human_labels/" not in encoded
    assert not (output / "second_pass_labels").exists()
    package = SecondaryPackage(output, manifest_sha)
    assert [record.frame_index for record in package.records] == list(SELECTED_FRAMES)


def test_validator_stays_need_human_until_all_eight_are_present(tmp_path: Path) -> None:
    source, source_sha = _source(tmp_path)
    output = tmp_path / "secondary"
    _, manifest_sha = build_package(source, output, source_sha)
    report, code = validate(output, manifest_sha)
    assert code == 2
    assert report["status"] == "NEED_HUMAN_SECOND_PASS"
    assert report["passed_label_count"] == 0
    assert report["thresholds_defined"] is False

    _write_second_pass_labels(output)
    report, code = validate(output, manifest_sha)
    assert code == 0
    assert report["status"] == "READY_FOR_NOISE_FLOOR_COMPUTATION"
    assert report["passed_label_count"] == 8
    assert report["noise_floor_computed"] is False
    assert report["candidate_evaluation_allowed"] is False
