from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from tools.validate_houb_labels import validate_package


def _benchmark(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "package"
    root.mkdir()
    frames = []
    for index in range(60):
        frames.append(
            {
                "session_id": "fixture",
                "frame_index": index,
                "width": 4,
                "height": 3,
                "expected_human_label_path": f"human_labels/palette_png/fixture/{index:05d}.png",
            }
        )
    payload = (json.dumps({"frames": frames}, sort_keys=True) + "\n").encode()
    (root / "BENCHMARK_MANIFEST.json").write_bytes(payload)
    return root, hashlib.sha256(payload).hexdigest()


def _write_labels(root: Path, *, fill: int = 0, mode: str = "P") -> None:
    label_root = root / "human_labels/palette_png/fixture"
    label_root.mkdir(parents=True)
    for index in range(60):
        image = Image.new(mode, (4, 3), fill)
        if mode == "P":
            image.putpalette([0] * 768)
        image.save(label_root / f"{index:05d}.png")


def test_missing_labels_fail_closed(tmp_path: Path) -> None:
    root, manifest_sha = _benchmark(tmp_path)
    report, exit_code = validate_package(
        package_root=root, expected_benchmark_sha256=manifest_sha
    )
    assert exit_code == 2
    assert report["status"] == "AWAITING_HUMAN_HOUB_ANNOTATION"
    assert report["candidate_evaluation_allowed"] is False
    assert report["passed_label_count"] == 0


def test_all_exhaustive_palette_labels_complete_only_annotation_gate(tmp_path: Path) -> None:
    root, manifest_sha = _benchmark(tmp_path)
    _write_labels(root, fill=0, mode="P")
    report, exit_code = validate_package(
        package_root=root, expected_benchmark_sha256=manifest_sha
    )
    assert exit_code == 0
    assert report["status"] == "PASS_HUMAN_HOUB_ANNOTATION"
    assert report["annotation_gate_complete"] is True
    assert report["auditor_freeze_task_eligible"] is True
    assert report["candidate_evaluation_allowed"] is False
    assert report["passed_label_count"] == 60


def test_unlabeled_255_and_non_palette_each_hold(tmp_path: Path) -> None:
    root, manifest_sha = _benchmark(tmp_path)
    _write_labels(root, fill=255, mode="P")
    report, exit_code = validate_package(
        package_root=root, expected_benchmark_sha256=manifest_sha
    )
    assert exit_code == 2
    assert report["passed_label_count"] == 0
    assert all("255" in error.get("detail", "") for error in report["errors"])

    label = root / "human_labels/palette_png/fixture/00000.png"
    Image.new("L", (4, 3), 0).save(label)
    report, exit_code = validate_package(
        package_root=root, expected_benchmark_sha256=manifest_sha
    )
    assert exit_code == 2
    assert any("mode must be" in error.get("detail", "") for error in report["errors"])


def test_manifest_digest_drift_holds_even_with_valid_labels(tmp_path: Path) -> None:
    root, manifest_sha = _benchmark(tmp_path)
    _write_labels(root, fill=0, mode="P")
    report, exit_code = validate_package(
        package_root=root, expected_benchmark_sha256="0" * 64
    )
    assert exit_code == 2
    assert report["benchmark_manifest_ref"]["observed_sha256"] == manifest_sha
    assert any(error["code"] == "BENCHMARK_MANIFEST_SHA_MISMATCH" for error in report["errors"])
