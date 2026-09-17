from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from chaoyang.ops import audit_robot_reach_v75 as reach
from chaoyang.ops import audit_robot_target_chain_v75 as target
from chaoyang.ops import profile_robot_pipeline_v75 as profile
from chaoyang.ops import run_exact78_robot_ready_batches_v75 as runner_v75
from chaoyang.ops.robot_target_reach_v75 import RobotTargetContractError


ROOT = Path(__file__).resolve().parents[1]


def test_target_and_reach_keep_terminal_status_separate_from_robot_tier(tmp_path, monkeypatch):
    n = 8
    human = np.zeros((2, n, 21, 3), dtype=np.float64)
    target_tf = np.tile(np.eye(4), (n, 2, 1, 1)).astype(np.float64)
    actual_tf = target_tf.copy()
    for side in range(2):
        delta = np.linspace(0.0, 0.02, n)
        human[side, :, 0, 0] = delta
        target_tf[:, side, 0, 3] = delta
        actual_tf[:, side, 0, 3] = 0.75 * delta
    hawor = tmp_path / "hawor.npz"
    states = tmp_path / "states.npz"
    detail = tmp_path / "arm.json"
    np.savez_compressed(
        hawor,
        joints_3d_world=human,
        original_frame_indices=np.arange(n, dtype=np.int32),
    )
    np.savez_compressed(
        states,
        T_target_hand_root_world=target_tf,
        T_actual_hand_root_world=actual_tf,
        valid_side_frame=np.ones((2, n), dtype=bool),
        source_frames=np.arange(n, dtype=np.int32),
        T_world_base=np.eye(4),
    )
    detail.write_text(
        json.dumps({"task": "chips", "session": "chips_fixture", "gates": {"pose_branch_all_observed": False}}),
        encoding="utf-8",
    )
    target_result = tmp_path / "target.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_robot_target_chain_v75.py",
            "--task",
            "chips",
            "--session",
            "chips_fixture",
            "--hawor",
            str(hawor),
            "--arm-result",
            str(detail),
            "--arm-states",
            str(states),
            "--output",
            str(target_result),
        ],
    )
    assert target.main() == 0
    target_payload = json.loads(target_result.read_text())
    assert target_payload["terminal_status"] == "PASSED"
    assert target_payload["robot_tier"] == "NONE"
    assert target_payload["evidence_class"] == "DIAGNOSTIC_PROXY"
    assert target_payload["object6d_consumed"] is False
    assert target_payload["workspace_bounds_consumed"] is False
    assert target_payload["soft_pose_similarity_pass"] is False

    reach_result = tmp_path / "reach.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_robot_reach_v75.py", "--target-audit", str(target_result), "--output", str(reach_result)],
    )
    assert reach.main() == 0
    reach_payload = json.loads(reach_result.read_text())
    assert reach_payload["terminal_status"] == "PASSED"
    assert reach_payload["robot_tier"] == "NONE"
    assert reach_payload["evidence_class"] == "DIAGNOSTIC_PROXY"
    assert reach_payload["candidate_robot_tier"] == "POSE_ONLY_VISUAL"
    assert reach_payload["hard_geometry_pass"] is False
    assert reach_payload["visual_train_eligible"] is True
    metrics = target_payload["sides"][0]["metrics"]
    assert "target_tracking_large_error_ratio" in metrics
    assert "workspace_clipping_frame_ratio_proxy" not in metrics
    gates = reach_payload["side_gates"][0]["eligibility"]
    assert "target_tracking_large_error_ratio_le_0p2" in gates
    assert "workspace_clipping_le_0p2" not in gates


def test_profile_reports_worker_timing_without_authority(tmp_path, monkeypatch):
    root = tmp_path / "run"
    phase = root / "batch_001" / "render"
    phase.mkdir(parents=True)
    (phase / "RESULT.json").write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "session": "s1",
                        "status": "PASS",
                        "returncode": 0,
                        "started_at": "2026-09-15T00:00:00+08:00",
                        "finished_at": "2026-09-15T00:00:10+08:00",
                        "frame_count": 100,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "profile.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_robot_pipeline_v75.py", "--run-root", str(root), "--output", str(output)],
    )
    assert profile.main() == 0
    value = json.loads(output.read_text())
    assert value["terminal_status"] == "PASSED"
    assert value["robot_tier"] == "NONE"
    assert value["phases"]["render"]["duration_s_p50"] == 10.0
    assert value["phases"]["render"]["frames_per_worker_hour"] == 36000.0
    assert value["authority"] is False


def test_matrix_and_robot_entrypoint_help_are_read_only_from_arbitrary_cwd(tmp_path):
    scripts = [
        "src/chaoyang/ops/build_robot_geometry_terminal_matrix_v71.py",
        "src/chaoyang/ops/build_exact78_final_terminal_matrix_v71.py",
        "src/chaoyang/ops/run_exact78_robot_ready_batches_v75.py",
        "src/chaoyang/ops/preflight_adopt_robot_v74_to_v75.py",
        "src/chaoyang/ops/audit_robot_target_chain_v75.py",
        "src/chaoyang/ops/audit_robot_reach_v75.py",
        "src/chaoyang/ops/profile_robot_pipeline_v75.py",
    ]
    for relative in scripts:
        completed = subprocess.run(
            [sys.executable, str(ROOT / relative), "--help"],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, f"{relative}: {completed.stderr}"
    assert list(tmp_path.iterdir()) == []


def test_v75_signed_adopt_preflight_excludes_nine_completed_sessions_from_execution():
    path = ROOT / "tasks/receipts/ROBOT_V75_ADOPT_PREFLIGHT_CLEAN_BASELINE.json"
    preflight, rows = runner_v75.verify_adopt_preflight(path)
    selection_path = Path(preflight["selection"]["path"])
    frozen = json.loads(selection_path.read_text(encoding="utf-8"))["sessions"]
    adopted, remaining = runner_v75.split_frozen_sessions(frozen, rows)
    assert len(frozen) == 25
    assert len(adopted) == 9
    assert len(remaining) == 16
    assert not set(adopted) & set(remaining)
    assert set(adopted) | set(remaining) == set(frozen)


def test_v75_refuses_adopt_rows_outside_frozen_selection():
    with pytest.raises(RobotTargetContractError, match="not a subset"):
        runner_v75.split_frozen_sessions(["a", "b"], [{"session": "c"}])


def test_v75_refuses_remaining_session_that_is_no_longer_robot_ready():
    matrix = {
        "rows": [
            {"session_id": "adopted", "robot_current_state": "FAILED_QUALITY_C"},
            {"session_id": "execute", "robot_current_state": "BLOCKED_PREREQ"},
        ]
    }
    with pytest.raises(RobotTargetContractError, match="no longer Robot-ready"):
        runner_v75.validate_execution_readiness(matrix, ["adopted", "execute"], ["execute"])


def test_v75_allows_adopted_history_but_requires_remaining_ready():
    matrix = {
        "rows": [
            {"session_id": "adopted", "robot_current_state": "FAILED_QUALITY_C"},
            {"session_id": "execute", "robot_current_state": "READY_FOR_ROBOT_CURRENT_DRAFT"},
        ]
    }
    rows = runner_v75.validate_execution_readiness(
        matrix, ["adopted", "execute"], ["execute"]
    )
    assert set(rows) == {"adopted", "execute"}


def test_v75_terminal_publisher_binds_result_and_retries_cas(tmp_path, monkeypatch):
    result = tmp_path / "RESULT.json"
    result.write_text("{}", encoding="utf-8")
    receipt = tmp_path / "docs/governance/CURRENT_STATUS_RECEIPT.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"governance_revision": 7}), encoding="utf-8")
    monkeypatch.setattr(runner_v75, "ROOT", tmp_path)
    responses = [
        subprocess.CompletedProcess([], 2, "", "CAS revision mismatch"),
        subprocess.CompletedProcess([], 0, '{"governance_revision": 8}', ""),
    ]
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return responses.pop(0)

    monkeypatch.setattr(runner_v75.subprocess, "run", fake_run)
    monkeypatch.setattr(runner_v75.time, "sleep", lambda _: None)
    published = runner_v75.publish_task_terminal("robot_geometry_expansion_v75", result, "done")
    assert published["governance_revision"] == 8
    assert len(calls) == 2
    assert "--result" in calls[-1]
    assert "PASSED" in calls[-1]


def test_v75_current_packet_registration_requires_exact_packet_sha(tmp_path, monkeypatch):
    packet = tmp_path / "attempt/TASK_PACKET.json"
    packet.parent.mkdir(parents=True)
    packet.write_text('{"task_id":"robot_geometry_expansion_v75"}', encoding="utf-8")
    index = tmp_path / "index.json"
    packet_sha = runner_v75.artifact_ref(packet)["sha256"]
    index.write_text(
        json.dumps(
            {
                "task_packets": [
                    {
                        "task_id": "robot_geometry_expansion_v75",
                        "packet_path": str(packet),
                        "packet_sha256": packet_sha,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pointer = tmp_path / "docs/governance/CURRENT_V71_TASK_PACKET_INDEX.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(
        json.dumps(
            {
                "index_path": str(index),
                "index_sha256": runner_v75.artifact_ref(index)["sha256"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner_v75, "ROOT", tmp_path)
    registered = runner_v75.verify_current_packet_registration(
        packet.resolve(), "robot_geometry_expansion_v75"
    )
    assert registered["packet"]["sha256"] == packet_sha

    value = json.loads(index.read_text(encoding="utf-8"))
    value["task_packets"][0]["packet_sha256"] = "0" * 64
    index.write_text(json.dumps(value), encoding="utf-8")
    pointer.write_text(
        json.dumps(
            {
                "index_path": str(index),
                "index_sha256": runner_v75.artifact_ref(index)["sha256"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RobotTargetContractError, match="does not bind"):
        runner_v75.verify_current_packet_registration(
            packet.resolve(), "robot_geometry_expansion_v75"
        )


def test_v75_task_row_must_bind_exact_task_packet(tmp_path):
    packet = tmp_path / "TASK_PACKET.json"
    packet.write_text("{}", encoding="utf-8")
    with pytest.raises(RobotTargetContractError, match="does not bind"):
        runner_v75.validate_task_row_registration(
            {"task_packet": {"path": str(packet), "bytes": 2, "sha256": "0" * 64}},
            packet.resolve(),
        )
