from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_supervisor_is_cpu_only_and_uses_governance_cas() -> None:
    text = (ROOT / "src/chaoyang/ops/run_v71_finite_convergence_supervisor.py").read_text()
    assert "gpu_work_launched\": False" in text
    assert "chaoyang.governance.update_task_state" in text
    assert "build_exact78_final_terminal_matrix_v71.py" in text
    assert "nvidia-smi" not in text
    assert "run_exact78_robot_ready_batches" not in text
