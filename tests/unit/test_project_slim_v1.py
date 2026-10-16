"""No models: exact cache deletion, old-loop fencing, and document pointers."""
from pathlib import Path
from datetime import datetime
import json
import re
import subprocess
import sys

import pytest

from chaoyang.governance.common import artifact_ref, load_json
from chaoyang.governance.project_slim_cache_proof_v1 import original, root_cwd_holders
from test_frozen_bundle_runtime_separation_v1 import (
    audit_deleted_generations, checked_ref,
    verify_frozen_release, verify_pinned_runtime,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"_run/current/project_slim_c1_v2_20261008/attempts/attempt_0001"


def test_exact_deletion_receipt_and_preserved_orphan_cache():
    checked_ref(OUT/"RESULT.json", {
        "bytes": 112676,
        "sha256": "5153f353e6938ab0e05cbf10016a02b50963781e92ec836a689b8f900c198293",
    }, "HISTORICAL_DELETE_RESULT_DRIFT")
    result = load_json(OUT/"RESULT.json")
    checked_ref(OUT/"DELETE_RECEIPT.json", result["outputs"]["DELETE_RECEIPT.json"], "HISTORICAL_DELETE_RECEIPT_DRIFT")
    deleted = load_json(OUT/"DELETE_RECEIPT.json")
    assert deleted["files_deleted"] == 5261
    assert deleted["bytes_deleted"] == 93443713
    assert deleted["allocated_bytes_deleted"] == 105481728
    checked_ref(deleted["proof"]["path"], deleted["proof"], "HISTORICAL_DELETE_PROOF_DRIFT")
    proof = load_json(Path(deleted["proof"]["path"]))
    journal_ref = result["outputs"]["DELETED.jsonl"]
    checked_ref(journal_ref["path"], journal_ref, "HISTORICAL_DELETE_JOURNAL_DRIFT")
    journal = [json.loads(line) for line in Path(journal_ref["path"]).read_text().splitlines()]
    assert journal == deleted["deleted"]
    runtime_audit = verify_pinned_runtime(ROOT, load_json(ROOT/"baseline_v1/MANIFEST.json"))
    runtime = runtime_audit["current_wrapper_refs"]
    # Receipt filesystem mtime is a retained generation witness, not a made-up
    # signed camera/acquisition timestamp. Old proof creation is SHA-bound.
    sealed_after_ns = max((OUT/"DELETE_RECEIPT.json").stat().st_mtime_ns,
                          int(datetime.fromisoformat(proof["created_at"]).timestamp() * 1_000_000_000))
    audited = audit_deleted_generations(ROOT, deleted, proof, sealed_after_ns, runtime)
    assert audited["historical_deleted"] == 5261
    assert audited["currently_absent"] + len(audited["regenerated_generations"]) == 5261
    blocked = [row for row in proof["rows"] if row["status"] != "ELIGIBLE_C1"]
    assert len(blocked) == 467
    assert all(artifact_ref(ROOT/row["path"]) == row["file"] for row in blocked)
    assert len(proof["preserved_symlinks"]) == 132
    assert all((ROOT/row["path"]).is_symlink() for row in proof["preserved_symlinks"])


def test_frozen_v1_bundle_preserves_its_published_source_bytes():
    assert verify_frozen_release(ROOT, ROOT/"baseline_v1")["copied_files_sha_verified"] == 163


@pytest.mark.parametrize("script", ["autopilot_watchdog.sh", "progress_pulse.sh"])
def test_old_auxiliary_script_refuses_before_writing(script):
    base = ROOT/"_run/caches/takeover_autopilot"
    names = ("HEARTBEAT", "PULSE_HEARTBEAT", "watchdog.pid", "pulse.pid")
    before = {n: (base/n).read_bytes() if (base/n).exists() else None for n in names}
    result = subprocess.run(["bash", "scripts/takeover/"+script], cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 64
    assert "LEGACY_TAKEOVER_HELPER_RETIRED" in result.stderr
    assert before == {n: (base/n).read_bytes() if (base/n).exists() else None for n in names}


def test_only_precise_old_helpers_and_sleep_children_signaled():
    # This is a pre-reboot historical receipt, not a current PID liveness gate.
    checked_ref(OUT/"STOP_LOOPS.json", {
        "bytes": 1772,
        "sha256": "b406ab643d5873f75ea8378a027ac1837a28176d5b6a5ef97a55f99beeadf43a",
    }, "HISTORICAL_STOP_RECEIPT_DRIFT")
    receipt = load_json(OUT/"STOP_LOOPS.json")
    assert receipt["status"] == "PASS"
    assert receipt["other_processes_signaled"] == 0
    assert receipt["manus_supervisor_preserved"] is True
    expected = {643811: 352439993, 919396: 364543469}
    assert {row["parent"]["pid"] for row in receipt["rows"]} == set(expected)
    for row in receipt["rows"]:
        assert row["status"] == "STOPPED"
        pid = row["parent"]["pid"]
        assert row["parent"]["proc_start_ticks"] == expected[pid]
        assert len(row["children"]) <= 1
        assert all(p["argv"][0] == "sleep" for p in row["children"])
        assert row["parent"]["argv"] in [
            ["bash", "scripts/takeover/autopilot_watchdog.sh"],
            ["bash", "scripts/takeover/progress_pulse.sh"],
        ]
        expected_signals = {(p["pid"], p["proc_start_ticks"])
                            for p in [row["parent"]] + row["children"]}
        assert {(p["pid"], p["proc_start_ticks"]) for p in row["signaled"]} == expected_signals
        assert row["signal"] == "SIGTERM_PIDFD"


def test_deep_cache_files_blocked_by_root_cwd():
    root = ".cache/pycache"
    records = [{"pid": 999, "cwd": str(ROOT/root)}, {"pid": 998, "cwd": str(ROOT/root/"deep")},
               {"pid": 997, "cwd": str(ROOT/root)+"_other"}, {"pid": 996, "cwd": str(ROOT)}]
    assert [p["pid"] for p in root_cwd_holders(records, [root])] == [999, 998]


def test_bytecode_requires_existing_original_python_source(tmp_path):
    source = tmp_path/"sample.py"; source.write_text("VALUE = 1\n")
    cache = tmp_path/"__pycache__"/"sample.cpython-311.pyc"
    assert original(cache, "src/test/__pycache__") == source
    assert original(cache.with_name("missing.cpython-311.pyc"), "src/test/__pycache__") is None


def test_humanego_document_links_and_new_default_reference_resolve():
    document = ROOT/"src/chaoyang/human_ego/README.md"
    links = re.findall(r"\]\(([^)]+)\)", document.read_text())
    assert len(links) == 5
    assert all((document.parent/link).resolve().exists() for link in links)
    assert "不是新阶段 Raw→Robot" in document.read_text()


def test_c1_exact_command_and_sealed_c0_proof():
    packet = load_json(ROOT/"tasks/current/project_slim_c1_v2_20261008/TASK_PACKET.json")
    assert " --phase c1 --revision v2 --proof " in packet["commands"][0]
    assert packet["commands"][0].startswith(sys.executable+" -B -m ")
    proof = packet["inputs"]["proof"]
    assert artifact_ref(proof["path"]) == proof
    result = load_json(Path(proof["path"]).parent/"RESULT.json")
    assert result["status"] == "PASSED" and result["outputs"]["PROOF.json"] == proof
