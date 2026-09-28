"""Publication regression: close staging writer without weakening identity checks."""
import os
from pathlib import Path
import pytest
from chaoyang.human_ego.utils import frozen_contract as fc

def test_staging_writer_closed_before_recovery(tmp_path, monkeypatch):
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"not a trained model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    recover = fc._recover_runtime_checkpoint_authority_transaction
    seen = []
    def checked(authority, parent, parent_fd, parent_stat, encoded, expected_staging_identity=None):
        staging = parent / fc._authority_staging_name(authority)
        assert expected_staging_identity == (staging.stat().st_dev, staging.stat().st_ino)
        if Path("/proc/self/fd").exists():
            for entry in Path("/proc/self/fd").iterdir():
                try:
                    assert os.readlink(entry) != str(staging)
                except FileNotFoundError:
                    pass
        seen.append(True)
        return recover(authority, parent, parent_fd, parent_stat, encoded,
                       expected_staging_identity=expected_staging_identity)
    monkeypatch.setattr(fc, "_recover_runtime_checkpoint_authority_transaction", checked)
    result = fc.write_runtime_checkpoint_authority(checkpoint,
        run_manifest_reference=fc.file_reference(manifest),
        dataset_stats_reference=fc.file_reference(manifest))
    assert seen == [True]
    assert result.stat().st_nlink == 1

def test_recovery_refuses_replaced_staging_identity(tmp_path):
    authority = tmp_path / "fixture.pt.authority.json"
    staging = tmp_path / fc._authority_staging_name(authority)
    staging.write_bytes(b"{}")
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        s = staging.stat()
        with pytest.raises(ValueError, match="staging identity drift"):
            fc._recover_runtime_checkpoint_authority_transaction(
                authority, tmp_path, fd, os.fstat(fd), b"{}",
                expected_staging_identity=(s.st_dev, s.st_ino + 1))
        assert not authority.exists()
        assert staging.read_bytes() == b"{}"
    finally:
        os.close(fd)
