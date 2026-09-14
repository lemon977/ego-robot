import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "task33_cpu_plan_test", ROOT / "tools/prepare_002_d1_wearable_conditional_p3_t1.py"
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_real_contract_preserves_all_four_prompts_in_frozen_order():
    plan = runner.plan(ROOT)
    matrix = plan["identity_prompt_matrix"]
    assert [(row["identity"], row["prompt"]) for row in matrix] == [
        ("D1", "an arm"),
        ("W1", "a wrist-worn object"),
        ("W2", "a wearable object around a wrist"),
        ("W3", "a wrist accessory"),
        ("W4", "an object worn on a wrist"),
    ]
    assert all(row["full_session_calls"] == 1 for row in matrix)
    assert plan["wearable_prompt_execution"] == "ALL_FOUR_SEPARATE_NO_SELECTION_NO_FALLBACK"
    assert plan["gpu_admission_allowed"] is False
    assert plan["run_creation_allowed"] is False


def test_missing_future_dependencies_stay_hardblocked(tmp_path: Path):
    contract = ROOT / runner.PROMPT_CONTRACT
    (tmp_path / "contracts").mkdir()
    (tmp_path / "contracts" / contract.name).write_bytes(contract.read_bytes())
    plan = runner.plan(tmp_path)
    assert plan["status"] == "CPU_PREPARED_GPU_HARDBLOCKED"
    assert all(
        row["state"] == "WAITING_FOR_VERSIONED_EVIDENCE"
        for row in plan["dependencies"].values()
    )


def test_failed_or_cpu_qa_status_cannot_satisfy_result_dependency(tmp_path: Path):
    path = tmp_path / runner.P1_RESULT_QA
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "task29-independent-cpu-qa-v3",
                "status": "PASS_TASK29_V3_CPU_QA_P0_ZERO_GPU_ADMISSION_ALLOWED",
            }
        ),
        encoding="utf-8",
    )
    state = runner.optional_state(
        tmp_path,
        runner.P1_RESULT_QA,
        schema="task29-independent-result-qa-v3",
        status="PASS_TASK29_V3_RESULT_INTEGRITY_EXACT",
    )
    assert state["state"] == "HOLD_NONCANONICAL_OR_FAILED"


def test_result_driven_prompt_key_is_rejected(tmp_path: Path):
    contract = json.loads((ROOT / runner.PROMPT_CONTRACT).read_text())
    contract["chosen_prompt"] = "W1"
    path = tmp_path / runner.PROMPT_CONTRACT
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(contract), encoding="utf-8")
    runner.PROMPT_CONTRACT_SHA = runner.hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(runner.Task33Error, match="result-driven"):
        runner.frozen_prompts(tmp_path)


@pytest.mark.parametrize("command", ["admit", "run"])
def test_current_version_cannot_admit_or_run(command, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["task33", command])
    with pytest.raises(runner.Task33Error, match="hardblocked"):
        runner.main()
