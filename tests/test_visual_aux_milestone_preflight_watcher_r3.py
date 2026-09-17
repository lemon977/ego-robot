import json
from pathlib import Path
import subprocess
import sys

from chaoyang.ops import watch_visual_aux_milestone_preflight_r3 as watcher


ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def fixture(tmp_path: Path, processed: int, *, ledger: bool = False) -> list[str]:
    state = write(tmp_path / "robot.json", {"status": "RUNNING", "processed_sessions": processed, "expected_sessions": 25})
    receipt = write(tmp_path / "receipt.json", {"governance_revision": 1})
    contract = write(tmp_path / "contract.json", {"stages": [{"stage": "HumanEgo Aux", "algorithm_id": "H50", "weights": "ABSENT"}]})
    eligibility = write(tmp_path / "eligibility.json", {"status": "PASS_ELIGIBILITY_INDEX_BUILT", "summaries": {task: {split: {"ready_sessions": 0, "eligible_h50_windows": 0} for split in ("train", "validation")} for task in ("chips", "poker")}})
    prior = write(tmp_path / "prior.json", {"pair_status": {"chips": "BLOCKED_DATA_VOLUME", "poker": "BLOCKED_DATA_VOLUME"}})
    capacity = write(tmp_path / "capacity.json", {"tasks": {task: {"capacity_status": "BLOCKED_DATA_VOLUME_CAPACITY", "train": {}, "validation": {}} for task in ("chips", "poker")}})
    packets = {}
    for task in ("chips", "poker"):
        packets[task] = write(tmp_path / f"{task}_packet.json", {"task_id": f"visual_aux_{task}_pair_v1"})
    ledger_path = tmp_path / "causal_ledger.json"
    if ledger:
        write(ledger_path, {"status": "PASS"})
    packet = write(tmp_path / "watcher_packet.json", {"task_id": "visual_aux_milestone_preflight_watcher_r3"})
    launch = write(tmp_path / "launch.json", {"task_id": "visual_aux_milestone_preflight_watcher_r3", "command": []})
    return [
        sys.executable, str(ROOT / "src/chaoyang/ops/watch_visual_aux_milestone_preflight_r3.py"),
        "--robot-state", str(state), "--output-root", str(tmp_path / "output"),
        "--status-receipt", str(receipt), "--algorithm-contract", str(contract),
        "--eligibility-index", str(eligibility), "--prior-result", str(prior),
        "--capacity-preflight", str(capacity), "--chips-task-packet", str(packets["chips"]),
        "--poker-task-packet", str(packets["poker"]), "--robotized-ledger", str(ledger_path),
        "--task-packet", str(packet), "--launch-command", str(launch),
        "--milestones", "12,18,25", "--once",
    ]


def test_before_first_milestone_only_updates_local_state(tmp_path):
    completed = subprocess.run(fixture(tmp_path, 9), cwd=tmp_path, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    state = json.loads((tmp_path / "output/AUTOMATION_STATE.json").read_text())
    assert state["status"] == "POLL_COMPLETE"
    assert state["completed_milestones"] == []
    assert state["started_training"] is False
    assert not (tmp_path / "output/milestones").exists()
    setup = json.loads((tmp_path / "output/setup/attempt_0001/RESULT.json").read_text())
    assert setup["execution_status"] == "PASSED_SETUP"
    assert setup["started_training"] is False


def test_crossed_milestones_are_immutable_and_block_without_causal_proof(tmp_path):
    command = fixture(tmp_path, 18)
    completed = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    state = json.loads((tmp_path / "output/AUTOMATION_STATE.json").read_text())
    assert state["completed_milestones"] == [12, 18]
    for milestone in (12, 18):
        attempt = tmp_path / f"output/milestones/processed_{milestone:04d}/attempt_0001"
        result = json.loads((attempt / "RESULT.json").read_text())
        assert result["terminal_status"] == "BLOCKED_PREREQ"
        assert result["started_training"] is False
        assert result["mutated_current_governance"] is False
        for task in ("chips", "poker"):
            pair = json.loads((attempt / task / "RESULT.json").read_text())
            assert pair["terminal_status"] == "BLOCKED_PREREQ"
            assert pair["checkpoint_count"] == 0
    second = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, check=False)
    assert second.returncode == 0, second.stderr
    assert not (tmp_path / "output/milestones/processed_0012/attempt_0002").exists()


def test_parser_normalizes_and_validates_milestones():
    assert watcher.parse_milestones("25,12,18,12") == (12, 18, 25)


def test_source_cannot_start_training_or_mutate_governance():
    source = (ROOT / "src/chaoyang/ops/watch_visual_aux_milestone_preflight_r3.py").read_text()
    assert "run_v71_post_robot_visual_aux" not in source
    assert "run_gpu_command_with_v71_lease" not in source
    assert "chaoyang.governance.update_task_state" not in source
    assert "preflight_visual_aux_pairs_r3.py" in source
