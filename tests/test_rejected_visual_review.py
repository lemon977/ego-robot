from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from tools.build_rejected_visual_review import build, sha256_file, validate


def _png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color).save(path)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    _png(project / "raw" / "00001.png", (30, 40, 50))
    for name, color in (("H_core.png", (255, 255, 255)), ("O_visible_core.png", (0, 0, 0)), ("U_contact.png", (0, 0, 0))):
        _png(project / "candidate" / "00001" / name, color)
    _png(project / "existing.png", (12, 34, 56))
    qa = {
        "frame_scope": {"scope": "FRAME_SET", "frame_indices": [1]},
        "symptoms": ["synthetic rejection"],
    }
    (project / "qa.json").write_text(json.dumps(qa), encoding="utf-8")
    spec = {
        "schema_version": "rejected-visual-review-v1",
        "items": [
            {
                "asset_id": "v1_representative",
                "version": "v1",
                "title": "representative",
                "kind": "representative_mask_sheet",
                "qa_ref": "qa.json",
                "representative_frame_indices": [1],
                "raw_template": "raw/{frame05}.png",
                "mask_template": "candidate/{frame05}/{mask_name}",
                "thumbnail_size": [16, 12],
            },
            {
                "asset_id": "v2_existing",
                "version": "v2",
                "title": "existing",
                "kind": "copy",
                "qa_ref": "qa.json",
                "source_path": "existing.png",
            },
        ],
    }
    spec_path = project / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return project, spec_path


def test_build_is_immutable_and_valid(tmp_path: Path) -> None:
    project, spec = _fixture(tmp_path)
    output = project / "_run" / "review_v1"
    result = build(spec, output, project)
    assert result["verdict"] == "PASS"
    assert result["asset_count"] == 2
    manifest = json.loads((output / "MANIFEST.json").read_text())
    assert all(asset["consumer_allowed"] is False for asset in manifest["assets"])
    assert "REPRESENTATIVE" in (output / "INDEX.html").read_text()
    assert not ((output / "assets" / "01_v1_representative.png").stat().st_mode & 0o222)


def test_refuses_overwrite(tmp_path: Path) -> None:
    project, spec = _fixture(tmp_path)
    output = project / "_run" / "review_v1"
    build(spec, output, project)
    with pytest.raises(FileExistsError):
        build(spec, output, project)


def test_validation_detects_copy_tamper(tmp_path: Path) -> None:
    project, spec = _fixture(tmp_path)
    output = project / "_run" / "review_v1"
    build(spec, output, project)
    asset = output / "assets" / "02_v2_existing.png"
    asset.chmod(0o644)
    asset.write_bytes(b"tampered")
    assert validate(output)["verdict"] == "FAIL"


def test_manifest_digest_binds_manifest(tmp_path: Path) -> None:
    project, spec = _fixture(tmp_path)
    output = project / "_run" / "review_v1"
    result = build(spec, output, project)
    digest = (output / "MANIFEST.sha256").read_text().split()[0]
    assert digest == sha256_file(output / "MANIFEST.json") == result["manifest_sha256"]
