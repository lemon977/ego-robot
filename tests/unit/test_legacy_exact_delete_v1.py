"""Deletion integrity and negative entry checks; never import backup code."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.governance.baseline_v1_source_release import verify_bundle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "_run/current/legacy_exact_delete_c1_v5_20261008/attempts/attempt_0001"


def test_exact_target_count_and_all_files_absent():
    saved = load_json(OUT / "BACKUP_VERIFIED.json")
    assert len(saved["rows"]) == 18
    assert sum(r["bytes"] for r in saved["rows"]) == 138062
    assert not any((ROOT/r["path"]).exists() for r in saved["rows"])


def test_original_byte_backup_complete_including_unique_untracked_source():
    saved = load_json(OUT / "BACKUP_VERIFIED.json")
    assert artifact_ref(OUT/"BACKUP.tar.gz") == saved["backup"]
    assert saved["restore_requires_new_task"] is True
    with tarfile.open(OUT/"BACKUP.tar.gz", "r:gz") as archive:
        assert len(archive.getmembers()) == 19
        for row in saved["rows"]:
            payload = archive.extractfile(row["path"]).read()
            assert len(payload) == row["bytes"]
            assert hashlib.sha256(payload).hexdigest() == row["sha256"]
        assert archive.extractfile("THIRD_PARTY_NOTICES.md").read() == (ROOT/"THIRD_PARTY_NOTICES.md").read_bytes()
    assert sum(not r["git"]["tracked"] for r in saved["rows"]) == 14


def test_selected_source_bundle_and_original_runtime_closure_unchanged():
    result = verify_bundle(ROOT, ROOT/"baseline_v1")
    assert result["copied_files_sha_verified"] == 163
    assert not result["algorithm_changed"]


@pytest.mark.parametrize("name", [
    "build_exact78_v4_cohort_audit", "record_exact78_v4_audit_state",
    "run_human_to_robot_r2_local_huro_same_background", "run_robot100_retire_stale_navigation",
])
def test_old_command_refused_without_import_or_execution(name):
    from chaoyang.cli import _run_module
    with pytest.raises(SystemExit, match="RETIRED_OPERATION"):
        _run_module(name, [])
    assert "chaoyang.ops."+name not in sys.modules


def test_deleted_cli_v10_cannot_run():
    assert importlib.util.find_spec("chaoyang.cli_v10.__main__") is None
    result = subprocess.run([sys.executable, "-B", "-m", "chaoyang.cli_v10"], cwd=ROOT,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0


def test_old_autopilot_stopped_by_exact_identity_only():
    value = load_json(ROOT/"_run/current/legacy_exact_delete_c1_v4_20261008/attempts/attempt_0001/OLD_WRITER_STOP.json")
    assert value["status"] == "STOPPED"
    assert value["before"]["pid"] == 35315
    assert value["before"]["proc_start_ticks"] == 371570676
    assert value["signal"] == "SIGTERM"
    from chaoyang.governance.publish_legacy_exact_delete_v1 import identity
    state = identity(35315)
    assert state is None or state["proc_start_ticks"] != 371570676 or state["state"] == "Z"


def test_c1_command_is_exactly_registered():
    packet = load_json(ROOT/"tasks/current/legacy_exact_delete_c1_v5_20261008/TASK_PACKET.json")
    assert packet["commands"] == [sys.executable+" -B -m chaoyang.governance.publish_legacy_exact_delete_v1 --c1-only"]
