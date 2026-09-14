from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "build_workspace_organization_inventory.py"


def test_inventory_is_read_only_classifies_and_finds_external_reference(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    old_run = workspace / "_run" / "example_mask_poker_v1"
    successor = workspace / "_run" / "example_mask_poker_v2"
    docs = workspace / "docs"
    output = workspace / "inventory"
    old_run.mkdir(parents=True)
    successor.mkdir(parents=True)
    docs.mkdir()
    (old_run / "SUPERSEDED.md").write_text(
        "Superseded by _run/example_mask_poker_v2\n", encoding="utf-8"
    )
    (successor / "RESULT.json").write_text('{"status":"PASS"}\n', encoding="utf-8")
    (docs / "reference.md").write_text(
        "See `_run/example_mask_poker_v1/RESULT.json`.\n", encoding="utf-8"
    )

    subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--workspace",
            str(workspace),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    inventory = json.loads((output / "WORKSPACE_INVENTORY.json").read_text())
    graph = json.loads((output / "LEGACY_RUN_REFERENCE_GRAPH.json").read_text())
    rows = {row["name"]: row for row in inventory["legacy_runs"]}
    refs = {row["target_run"]: row for row in graph["targets"]}
    assert inventory["counts"]["legacy_run_directories"] == 2
    assert rows["example_mask_poker_v1"]["system"] == "mask"
    assert rows["example_mask_poker_v1"]["tasks"] == ["poker"]
    assert rows["example_mask_poker_v1"]["named_successors_from_shallow_text"] == [
        "example_mask_poker_v2"
    ]
    assert refs["example_mask_poker_v1"]["external_referrer_files"] == 1
    assert old_run.is_dir()
    assert successor.is_dir()

    repeated = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--workspace",
            str(workspace),
            "--output-dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr
