import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_is_claim_limited(tmp_path):
    upstream = tmp_path / "upstream.json"
    upstream.write_text(json.dumps({"status": "WAITING"}))
    hard_soft = tmp_path / "hard_soft.json"
    hard_soft.write_text(json.dumps({"status": "RUNNING"}))
    output = tmp_path / "out"
    completed = subprocess.run(
        [sys.executable, "src/chaoyang/ops/run_v71_post_robot_visual_tier.py", "--dry-run", "--upstream-state", str(upstream), "--hard-soft-state", str(hard_soft), "--output-root", str(output)],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    state = json.loads((output / "AUTOMATION_STATE.json").read_text())
    assert state["status"] == "DRY_RUN_READY"
    assert state["hard_soft_state"] == str(hard_soft.resolve())
    assert "no metric/Contact/physical authority" in state["claim_limit"]


def test_source_runs_visual_clean_before_pose_only_robot():
    source = (ROOT / "src/chaoyang/ops/run_v71_post_robot_visual_tier.py").read_text()
    assert "post_finalize_robot_expansion_R7_3/AUTOMATION_STATE.json" in source
    assert "run_visual_tier_causal_clean_v71.py" in source
    assert "run_exact78_robot_ready_batches_v53.py" in source
    assert "build_visual_aux_eligibility_index_v56.py" in source
    assert "WAITING_HARD_SOFT_TERMINAL_FOR_COMBINED_ELIGIBILITY" in source
    assert 'hard_soft.get("status") == "TERMINAL"' in source
    assert '"metric_geometry": False' in source
    assert '"robot_geometry_v1"' in source
    assert '"PASSED" if passed else "FAILED_RUNTIME_FINAL"' in source
    assert '"robotized_compositor_causal_v1"' in source
    assert '"BLOCKED_PREREQ" if passed else "FAILED_RUNTIME_FINAL"' in source
    assert "Eight calibration-missing" not in source
