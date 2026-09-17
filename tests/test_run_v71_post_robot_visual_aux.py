import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from chaoyang.ops import run_v71_post_robot_visual_aux as orchestrator


ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_is_finite_and_claim_limited(tmp_path):
    upstream = tmp_path / "robot.json"
    upstream.write_text(json.dumps({"status": "WAITING"}))
    output = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable, "src/chaoyang/ops/run_v71_post_robot_visual_aux.py",
            "--dry-run", "--robot-state", str(upstream), "--output-root", str(output),
        ], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    state = json.loads((output / "AUTOMATION_STATE.json").read_text())
    assert state["status"] == "DRY_RUN_READY"
    assert "no real action" in state["claim_limit"]


def test_source_enforces_causal_pairs_and_minimums():
    source = (ROOT / "src/chaoyang/ops/run_v71_post_robot_visual_aux.py").read_text()
    assert '"--causal-training"' in source
    assert '"train": {"sessions": 16, "windows": 256}' in source
    assert '"validation": {"sessions": 3, "windows": 48}' in source
    assert 'for task in ("chips", "poker")' in source
    assert '"--priority", "CHECKPOINT_TRAINING"' in source
    assert '"BLOCKED_DATA_VOLUME"' in source


def result_metrics(*, full_ade: float, hard_ade: float, hard_fde: float) -> dict:
    return {
        "pair_dataset_signature": "c" * 64,
        "occlusion_hardset": {
            "path": "/frozen/hardset.json",
            "bytes": 100,
            "sha256": "a" * 64,
        },
        "best_validation_metrics": {"ADE_2D_px": full_ade},
        "best_occlusion_hardset_metrics": {
            "ADE_2D_px": hard_ade,
            "FDE_2D_px": hard_fde,
        },
    }


def test_value_gate_requires_both_hardset_metrics_and_full_validation_limit():
    raw = result_metrics(full_ade=100.0, hard_ade=100.0, hard_fde=100.0)
    passing = result_metrics(full_ade=101.0, hard_ade=94.0, hard_fde=94.0)
    result = orchestrator.visual_aux_value_gate(raw, passing)
    assert result["status"] == "PASS_FROZEN_HARDSET_AND_FULL_VALIDATION"
    assert all(result["gates"].values())

    bad_fde = result_metrics(full_ade=101.0, hard_ade=94.0, hard_fde=96.0)
    assert orchestrator.visual_aux_value_gate(raw, bad_fde)["status"] == "NO_GO_DIAGNOSTIC_ONLY"
    bad_full = result_metrics(full_ade=103.0, hard_ade=94.0, hard_fde=94.0)
    assert orchestrator.visual_aux_value_gate(raw, bad_full)["status"] == "NO_GO_DIAGNOSTIC_ONLY"


def test_value_gate_rejects_different_frozen_hardsets():
    raw = result_metrics(full_ade=100.0, hard_ade=100.0, hard_fde=100.0)
    robot = result_metrics(full_ade=101.0, hard_ade=94.0, hard_fde=94.0)
    robot["occlusion_hardset"] = {**robot["occlusion_hardset"], "sha256": "b" * 64}
    with pytest.raises(RuntimeError, match="same frozen hardset"):
        orchestrator.visual_aux_value_gate(raw, robot)


def test_missing_hardset_closes_before_waiting_or_training(tmp_path):
    upstream = tmp_path / "robot.json"
    upstream.write_text(json.dumps({"status": "RUNNING"}))
    output = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "src/chaoyang/ops/run_v71_post_robot_visual_aux.py"),
            "--robot-state",
            str(upstream),
            "--output-root",
            str(output),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.returncode == 3, completed.stderr
    state = json.loads((output / "AUTOMATION_STATE.json").read_text())
    assert state["status"] == "BLOCKED_PREREQ"
    assert state["reason"] == "FROZEN_OCCLUSION_HARDSET_INDEX_REQUIRED_BEFORE_TRAINING"
    assert not (output / "training").exists()


@pytest.mark.parametrize(
    "relative_script",
    [
        "src/chaoyang/human_ego/tools/build_visual_aux_session_bundle_v54.py",
        "src/chaoyang/human_ego/tools/train_visual_aux_future2d_v53.py",
        "src/chaoyang/ops/run_v71_post_robot_visual_aux.py",
    ],
)
def test_cli_help_runs_from_arbitrary_cwd_without_writes(tmp_path, relative_script):
    before = set(tmp_path.iterdir())
    completed = subprocess.run(
        [sys.executable, str(ROOT / relative_script), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": str(tmp_path / "nonexistent-mpl-cache"),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    after = set(tmp_path.iterdir())
    # --help exits before business output creation. Matplotlib is allowed to
    # create only the explicitly redirected cache directory.
    assert after - before <= {tmp_path / "nonexistent-mpl-cache"}


def test_worker_does_not_mutate_current_governance():
    source = (ROOT / "src/chaoyang/ops/run_v71_post_robot_visual_aux.py").read_text()
    assert "chaoyang.governance.update_task_state" not in source
    assert "AGGREGATOR_UPDATE_REQUESTS.json" in source
