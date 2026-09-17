from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "src/chaoyang/ops/preflight_visual_aux_pairs_r3.py"


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_missing_causal_ledger_closes_pair_blocked(tmp_path: Path) -> None:
    status = tmp_path / "status.json"
    packet = tmp_path / "packet.json"
    algorithm = tmp_path / "algorithm.json"
    eligibility = tmp_path / "eligibility.json"
    prior = tmp_path / "prior.json"
    output = tmp_path / "attempt" / "chips"
    dump(status, {"governance_revision": 7})
    dump(packet, {"task_id": "visual_aux_chips_pair_v1"})
    dump(
        algorithm,
        {
            "stages": [
                {"stage": "HumanEgo Aux", "algorithm_id": "future2d", "weights": "NOT_YET_PUBLISHED"}
            ]
        },
    )
    dump(
        eligibility,
        {
            "status": "PASS_ELIGIBILITY_INDEX_BUILT",
            "summaries": {
                "chips": {
                    "train": {"ready_sessions": 0, "eligible_h50_windows": 0},
                    "validation": {"ready_sessions": 0, "eligible_h50_windows": 0},
                }
            },
        },
    )
    dump(prior, {"pair_status": {"chips": "BLOCKED_DATA_VOLUME"}})
    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--task", "chips",
            "--output", str(output),
            "--status-receipt", str(status),
            "--task-packet", str(packet),
            "--algorithm-contract", str(algorithm),
            "--eligibility-index", str(eligibility),
            "--prior-result", str(prior),
            "--robotized-ledger", str(tmp_path / "missing-ledger.json"),
        ],
        check=True,
        cwd=ROOT,
    )
    result = json.loads((output / "RESULT.json").read_text())
    assert result["terminal_status"] == "BLOCKED_PREREQ"
    assert result["checkpoint_count"] == 0
    assert result["control_ground_truth"] is False
    assert {row["gate"] for row in result["blockers"]} >= {
        "formal_causal_robotized_rgb_ledger",
        "formal_occlusion_silver_binding",
        "causal_compositor_binding",
    }
    manifest = json.loads((output / "ARTIFACT_MANIFEST.json").read_text())
    assert set(manifest["artifacts"]) == {
        "RESULT.json", "METRICS.json", "RUN_RECEIPT.json", "DECISION.md", "NEXT_ACTION.json"
    }


def test_output_is_immutable(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--task", "chips",
            "--output", str(output),
            "--status-receipt", str(tmp_path / "unused-status.json"),
            "--task-packet", str(tmp_path / "unused-packet.json"),
            "--algorithm-contract", str(tmp_path / "unused-algorithm.json"),
            "--eligibility-index", str(tmp_path / "unused-eligibility.json"),
            "--prior-result", str(tmp_path / "unused-prior.json"),
            "--robotized-ledger", str(tmp_path / "unused-ledger.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "immutable output directory already exists" in completed.stderr
