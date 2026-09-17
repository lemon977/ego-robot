"""CPU-only contract tests for the Clean-independent pose-only runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chaoyang.ops import run_rc1_pose_only_offline_generic_v1 as runner


def _json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    return path


def _packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    session, frames = "play_cards_0901_017", 241
    dummy = tmp_path / "dummy.bin"
    dummy.write_bytes(b"pinned")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source fixture")
    hawor_npz = tmp_path / "hawor.npz"
    hawor_npz.write_bytes(b"npz fixture")
    hawor_result = _json(tmp_path / "hawor.json", {
        "inputs": {"source_video": runner.ref(source)},
        "outputs": {"npz": runner.ref(hawor_npz)},
    })
    role_result = _json(tmp_path / "role.json", {"status": "TERMINAL_GRADE_B"})
    object_result = _json(tmp_path / "object.json", {"status": "TERMINAL_GRADE_B"})
    stage_refs = {
        "hawor": runner.ref(hawor_result),
        "role_mask": runner.ref(role_result),
        "object_mask": runner.ref(object_result),
    }
    selection = _json(tmp_path / "selection.json", {"rows": [{"session_id": session, "task": "poker"}]})
    matrix = _json(tmp_path / "matrix.json", {"rows": [{
        "session_id": session, "task": "poker", "frame_count": frames,
        "three_upstream_ab": True,
        **{name: {"grade": "B", "result": item} for name, item in stage_refs.items()},
    }]})
    pins = {
        "selection": runner.ref(selection), "matrix": runner.ref(matrix),
        "hawor_result": runner.ref(hawor_result), "hawor_npz": runner.ref(hawor_npz),
        "role_result": runner.ref(role_result), "object_result": runner.ref(object_result),
        "raw_video": runner.ref(source), "template_states": runner.ref(dummy),
        "template_hawor": runner.ref(dummy), "robot_asset_pin": runner.ref(dummy),
    }
    code = {name: runner.ref(Path(runner.__file__)) for name in runner.REQUIRED_CODE}
    packet = {
        "schema_version": runner.SCHEMA, "task_id": "test-generic-pose-only",
        "session_id": session, "task": "poker", "frame_count": frames,
        "mode": "OFFLINE_VISUAL", "inputs": pins, "code": code,
        "initialization_provenance": {"mode": "HISTORICAL_CROSS_SESSION_DIAGNOSTIC_ONLY",
                                      "template_source_session_ids": {"accepted_states": "play_cards_0902_042",
                                                                      "accepted_hawor": "play_cards_0902_042"}},
        "config": {"collision_frames": frames, "render_fps": "source_fps",
                   "task_base_backoff_m": 0.26, "placement_label": "TEST_ASSUMPTION"},
        "budget": {"runtime_attempts": 1, "gpu_bytes": 0, "wall_cap_s": 5400},
        "model_weights": "PINNED_URDF_ASSET_PIN", "calibration": "ABSENT_POSE_ONLY",
    }
    monkeypatch.setattr(runner, "source_probe", lambda _: {"frames": frames, "fps": "25/1", "width": 1280, "height": 960})
    return _json(tmp_path / "packet.json", packet)


def test_preflight_checks_inputs_then_blocks_cross_session_initializer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet = _packet(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="CROSS_SESSION_TEMPLATE_BLOCKED"):
        runner.preflight(packet)
    assert runner.source_probe(Path(json.loads(packet.read_text())["inputs"]["raw_video"]["path"]))["fps"] == "25/1"


def test_safe_label_cannot_bypass_v3_implementation_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet = _packet(tmp_path, monkeypatch)
    data = json.loads(packet.read_text())
    data["initialization_provenance"]["mode"] = "INDEPENDENT_TASK_PRESET_NEUTRAL_OR_SAME_SESSION_PREFIX"
    _json(packet, data)
    with pytest.raises(RuntimeError, match="INDEPENDENT_PLACEMENT_RUNNER_NOT_IMPLEMENTED"):
        runner.preflight(packet)


def test_preflight_refuses_tampered_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet = _packet(tmp_path, monkeypatch)
    Path(json.loads(packet.read_text())["inputs"]["hawor_npz"]["path"]).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="bytes/SHA mismatch"):
        runner.preflight(packet)


def test_preflight_refuses_wrong_session_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet = _packet(tmp_path, monkeypatch)
    data = json.loads(packet.read_text())
    data["session_id"] = "play_cards_0901_018"
    _json(packet, data)
    with pytest.raises(RuntimeError, match="session not unique"):
        runner.preflight(packet)


def test_nonconsecutive_collision_frame_ids_are_not_first_n() -> None:
    eligible = [0, 4, 9, 17]
    runner.require_exact_audited_frame_ids(eligible, [0, 4, 9, 17])
    with pytest.raises(RuntimeError, match="exact nonconsecutive"):
        runner.require_exact_audited_frame_ids(eligible, [0, 1, 2, 3])


def test_signature_changes_with_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet = json.loads(_packet(tmp_path, monkeypatch).read_text())
    before = runner.producer_signature(packet)
    packet["config"]["task_base_backoff_m"] = 0.24
    assert runner.producer_signature(packet) != before


def test_signature_changes_with_initialization_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet = json.loads(_packet(tmp_path, monkeypatch).read_text())
    before = runner.producer_signature(packet)
    packet["initialization_provenance"]["template_source_session_ids"]["accepted_states"] = "play_cards_0902_043"
    assert runner.producer_signature(packet) != before


def test_hard_gate_vs_soft_pose_and_unknown_frames() -> None:
    arm = {"gates": {"all_observed_rows_pose_branch": False, "base_fixed": True,
                     "motion_gain_exact_one": True, "missing_frames_unknown_not_filled": True,
                     "per_side_anchor_is_observed": True}}
    hand = {"gates": {"anatomy_all_observed": False, "velocity": True, "acceleration": True,
                      "missing_unknown_not_filled": True, "thumb_independent_q0_to_q5": True,
                      "four_finger_chain_semantics": True}}
    collision = {"status": "PASS_DEVELOPMENT_COLLISION_AUDIT"}
    result = runner.classify_digital_geometry(frames=3, bilateral_valid_count=2,
                                              arm_result=arm, hand_result=hand, collision_result=collision)
    assert result["terminal_status"] == "PASSED_DIAGNOSTIC"
    assert result["unknown_frames"] == 1
    assert result["soft_human_pose_pass"] is False
    assert result["soft_arm_pose_branch_pass"] is False
    arm["gates"]["base_fixed"] = False
    assert runner.classify_digital_geometry(frames=3, bilateral_valid_count=3,
        arm_result=arm, hand_result=hand, collision_result=collision)["terminal_status"] == "FAILED_QUALITY_C_DIAGNOSTIC_VIDEO"
