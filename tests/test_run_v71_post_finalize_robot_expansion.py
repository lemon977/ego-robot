import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_is_bounded_and_claim_limited(tmp_path):
    finalizer = tmp_path / "finalizer.json"
    finalizer.write_text(json.dumps({"status": "WAITING"}))
    output = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable, "src/chaoyang/ops/run_v71_post_finalize_robot_expansion.py",
            "--dry-run", "--output-root", str(output), "--finalizer-state", str(finalizer),
        ], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    state = json.loads((output / "AUTOMATION_STATE.json").read_text())
    assert state["status"] == "DRY_RUN_READY"
    assert "no Contact/Occlusion/Robot/action/physical authority" in state["claim_limit"]


def test_automation_uses_new_task_and_timeout_bounded_runner():
    source = (ROOT / "src/chaoyang/ops/run_v71_post_finalize_robot_expansion.py").read_text()
    assert '"--task-id", "robot_geometry_v1"' in source
    assert '"--phase-timeout-seconds", "7200"' in source
    assert "start_new_session=True" in source
