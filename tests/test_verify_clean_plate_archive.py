import json
import sys
import zipfile

from tools.verify_clean_plate_archive import main


def test_complete_crc_archive_issues_identity_only_authority(tmp_path, monkeypatch):
    archive = tmp_path / "base_001.zip"
    members = {
        "base_001/clip_manifest.json": b"{}",
        "base_001/camera_params.json": b"{\"camera\": true}",
        "base_001/source_stereo/CameraRecord_base_001_stereo.mp4": b"synthetic-video",
    }
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name, value in members.items():
            package.writestr(name, value)
    extracted = tmp_path / "critical"
    extracted.mkdir()
    for name, value in members.items():
        (extracted / name.rsplit("/", 1)[-1]).write_bytes(value)
    output = tmp_path / "AUTHORITY.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_clean_plate_archive.py",
            "--archive", str(archive),
            "--critical-extracted-root", str(extracted),
            "--minimum-age-seconds", "0",
            "--output", str(output),
        ],
    )
    assert main() == 0
    report = json.loads(output.read_text())
    assert report["status"] == "PASS_COMPLETE_STABLE_FULL_CRC"
    assert report["full_crc"] == "PASS_ALL_MEMBERS"
    assert report["donor_archive_identity_verified"] is True
    assert report["same_session_same_camera_donor"] is None
    assert report["consumer_camera_compatibility_pending"] is True
    assert all(row["extracted_matches_archive"] for row in report["critical_members"].values())
