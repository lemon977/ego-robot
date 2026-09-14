from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.build_houb_benchmark_v2 import build
from tools.houb_annotation_app import BenchmarkPackage
from tools.validate_houb_labels_v2 import validate_package


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "_run/g2_mask_benchmark_v2"
BENCHMARK_SHA256 = "60b9df265169bb2a4d575bfc3a836039c9ba0d618922a6f8b08dbf7b44b9fde1"
RUN_MANIFEST_SHA256 = "0692d73123c8ad710367da3277d0a907620e8c76a9b8f21e1ae809de6d6c5ce4"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_identity_partitions_and_final_blind_holdout() -> None:
    manifest = json.loads((PACKAGE / "BENCHMARK_MANIFEST.json").read_text(encoding="utf-8"))
    assert _sha(PACKAGE / "BENCHMARK_MANIFEST.json") == BENCHMARK_SHA256
    assert _sha(PACKAGE / "run_manifest.json") == RUN_MANIFEST_SHA256
    assert manifest["schema_version"] == "g2-mask-houb-benchmark-v2"
    assert manifest["counts"]["frames"] == 45
    assert manifest["counts"]["development_frames"] == 15
    assert manifest["counts"]["same_session_blind_frames"] == 15
    assert manifest["counts"]["cross_session_blind_frames"] == 15
    assert {frame["session_id"] for frame in manifest["frames"]} == {
        "grap_a_cap_004",
        "grap_a_cap_025",
    }
    assert all(frame["session_id"] != "grap_a_cap_149" for frame in manifest["frames"])
    assert manifest["selection_policy"]["held_out_final_blind_session"] == "grap_a_cap_149"
    assert manifest["selection_policy"]["reselection_performed"] is False
    assert manifest["annotation_gate"]["candidate_evaluation_allowed"] is False


def test_v2_completed_labels_pass_validation_without_authorizing_promotion() -> None:
    report, exit_code = validate_package(
        package_root=PACKAGE,
        expected_benchmark_sha256=BENCHMARK_SHA256,
    )
    assert exit_code == 0
    assert report["status"] == "PASS_HUMAN_HOUB_ANNOTATION"
    assert report["expected_label_count"] == 45
    assert report["passed_label_count"] == 45
    assert report["missing_or_invalid_label_count"] == 0
    assert report["candidate_evaluation_allowed"] is False
    assert report["promotion_allowed"] is False


def test_v2_ui_reads_all_completed_labels_and_rejects_locked_overwrite() -> None:
    package = BenchmarkPackage(PACKAGE)
    assert len(package.records) == 45
    locked = [record for record in package.records if record.locked_label_sha256 is not None]
    assert [record.frame_index for record in locked] == [228, 229, 230, 231, 232]
    assert all(record.label_path.is_file() for record in package.records)
    assert package.progress_text() == "### 已落盘（尚未代表 QA 合格）：45/45 帧"
    with pytest.raises(ValueError, match="已审核迁移标签"):
        package.save(0, None, True)


def test_v2_builder_never_overwrites_existing_package() -> None:
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        build(ROOT, PACKAGE)
