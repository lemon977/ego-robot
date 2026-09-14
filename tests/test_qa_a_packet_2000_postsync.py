from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import types

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import qa_a_packet_2000_postsync as qa  # noqa: E402


def _ref(role: str, path: Path) -> qa.ExactRef:
    payload = path.read_bytes()
    return qa.ExactRef(role, path, len(payload), hashlib.sha256(payload).hexdigest())


def _matrix() -> dict[str, object]:
    return {
        "grap_a_cap_004_all_columns": "NOT_READY",
        "grap_a_cap_002_all_columns": "NOT_READY",
        "current_coverage_admitted_sessions": 0,
        "current_coverage_total_sessions": 68,
        "current_coverage_report": "NOT_PRODUCED",
        "E68I": False,
        "eta": "UNMEASURED",
        "command_admission": 0,
        "gpu_admission": 0,
        "pixel_or_selector_admission": 0,
    }


def _post(verdicts: dict[str, dict[str, str]]) -> dict[str, object]:
    return {
        "decision_group_count": 3,
        "all_a_holds": "UNCHANGED_OWNER_DECISIONS_REQUIRED",
        "bq05": "DEGRADED_TO_A_ACTIVATION_HOLD",
        "coverage": "CURRENT_0_OF_68_NOT_PRODUCED_FUTURE_2A_P0_2P1_HOLD_FORBIDDEN",
        "grap_a_cap_004_all_columns": "NOT_READY",
        "grap_a_cap_002_all_columns": "NOT_READY",
        "E68I": False,
        "eta": "UNMEASURED",
        "execution_gpu_pixel_queue_admission": 0,
        "snapshot_to_postlive_strict_prefix": "4/4",
        "finalizer_terminal_status": qa.TERMINAL_SUCCESS,
        "qa_dispositions": verdicts,
    }


def _gap() -> dict[str, object]:
    return {
        "status": "CONTRADICTED",
        "requirement_count": 131,
        "leaf_state_counts": {"PROVEN": 3, "CONTRADICTED": 44, "INCOMPLETE": 84, "MISSING": 0},
        "admission_and_governance": {"execution_gpu_pixel_queue_admission": 0},
        "completion_conclusion": {"goal_complete": False},
    }


def test_exact_held_read_rejects_sha_drift(tmp_path: Path) -> None:
    path = tmp_path / "subject.json"
    path.write_text("{}\n")
    ref = _ref("subject", path)
    path.write_text('{"drift":true}\n')
    with pytest.raises(qa.AuditError, match="bytes/SHA mismatch"):
        qa._read_held(ref, tmp_path)


def test_runtime_path_alias_rejected() -> None:
    rows = [[role, f"/tmp/{role}", "1", "0" * 64] for role in qa.ROLE_ORDER]
    rows[-1][1] = rows[-2][1]
    with pytest.raises(qa.AuditError, match="path alias"):
        qa._parse_ref_rows(rows)


def test_hardlink_rejected_by_nlink_one_gate(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(b"same")
    os.link(first, second)
    with pytest.raises(qa.AuditError, match="nlink is not 1"):
        qa._read_held(_ref("hardlink", second), tmp_path)


def test_stale_predecessor_row_contract_rejected() -> None:
    records: dict[str, qa.HeldRecord] = {}
    base_rows = [{"id": f"stable-{index}", "path": f"/p/s{index}", "bytes": 1,
                  "sha256": "0" * 64} for index in range(99)]
    for index, spec in enumerate(qa.DOC_SPECS):
        _, snapshot_role, post_role, stale_id, _, _, _ = spec
        snapshot = qa.HeldRecord(f"/p/snapshot{index}", b"x", 1, "1" * 64,
                                 1, 100 + index, 1, "0440")
        live = qa.HeldRecord(f"/p/live{index}", b"x+y", 3, "2" * 64,
                             1, 200 + index, 1, "0440")
        records[snapshot_role], records[post_role] = snapshot, live
        base_rows.append({"id": stale_id, "path": live.path, "bytes": snapshot.bytes,
                          "sha256": snapshot.sha256})
    base_rows[-1]["path"] = "/p/wrong-stale-path"
    with pytest.raises(qa.AuditError, match="stale row contract changed"):
        qa._catalog_expected({"canonical_references": base_rows}, records)


def test_positive_delivery_readiness_rejected() -> None:
    matrix = _matrix()
    matrix["grap_a_cap_004_all_columns"] = "READY"
    with pytest.raises(qa.AuditError, match="matrix readiness"):
        qa._require_matrix_hold(matrix)


def test_nonzero_admission_rejected() -> None:
    verdicts = {"q": {"field": "overall_verdict.verdict", "value": "PASS_Q"}}
    post = _post(verdicts)
    post["execution_gpu_pixel_queue_admission"] = 1
    with pytest.raises(qa.AuditError, match="post-2000 owner HOLD"):
        qa._require_post_hold(post, verdicts, qa.TERMINAL_SUCCESS)


def test_conflicting_qa_disposition_rejected() -> None:
    payload = {"overall_verdict": {"verdict": "PASS_EXACT"}, "status": "HOLD_CONFLICT"}
    with pytest.raises(qa.AuditError, match="conflicting top-level"):
        qa._exact_qa(payload, "PASS_EXACT", "synthetic", unambiguous=True)


def test_gap_exact_131_partition_and_drift() -> None:
    assert qa._require_gap_counts(_gap())["INCOMPLETE"] == 84
    changed = _gap()
    changed["leaf_state_counts"]["PROVEN"] = 4  # type: ignore[index]
    with pytest.raises(qa.AuditError, match="131=3/44/84/0"):
        qa._require_gap_counts(changed)


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    audit_root = tmp_path / "audits"
    audit_root.mkdir()
    output = audit_root / "INDEPENDENT_QA_A_CLASS_DECISION_QUEUE_2000_V1_T0_V1.json"
    paths = qa.AuditPaths(tmp_path, tmp_path, tmp_path, audit_root, tmp_path / "packet", output)
    first = qa.publish({"overall_verdict": {"verdict": qa.PASS_VERDICT}}, paths)
    before = output.read_bytes()
    with pytest.raises(qa.AuditError, match="exclusive QA publication failed"):
        qa.publish({"different": True}, paths)
    assert output.read_bytes() == before
    assert first["publication"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert output.stat().st_mode & 0o777 == 0o440


def test_source_held_compile_exec_with_registered_module() -> None:
    source = PROJECT_ROOT / "tools/qa_a_packet_2000_postsync.py"
    payload = source.read_bytes()
    name = "qa_a_packet_2000_postsync_held_synthetic"
    module = types.ModuleType(name)
    module.__file__ = str(source)
    sys.modules[name] = module
    try:
        exec(compile(payload, str(source), "exec"), module.__dict__)
        assert module.__dict__["PASS_VERDICT"] == qa.PASS_VERDICT
    finally:
        sys.modules.pop(name, None)


def test_no_discovery_media_gpu_queue_or_process_calls() -> None:
    source = (PROJECT_ROOT / "tools/qa_a_packet_2000_postsync.py").read_text()
    forbidden = ("os.walk(", "os.listdir(", "os.scandir(", ".rglob(", "glob.glob(",
                 "subprocess.", "os.kill(", "nvidia-smi", "cv2.", "ffmpeg")
    assert [token for token in forbidden if token in source] == []
