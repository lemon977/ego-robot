from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "allocate_task_run.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_registry(workspace: Path) -> None:
    system_path = workspace / "systems" / "mask" / "system.manifest.json"
    system_path.parent.mkdir(parents=True)
    system_path.write_text(
        '{"schema_version":"chaoyang-system-manifest-v1","system_id":"mask"}\n',
        encoding="utf-8",
    )
    task_path = workspace / "tasks" / "demo" / "task.manifest.json"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(
        json.dumps(
            {
                "task_id": "demo",
                "run_root_templates": {
                    "mask": "tasks/demo/runs/mask/{run_id}",
                    "clean": "tasks/demo/runs/clean/{run_id}",
                    "robot": "tasks/demo/runs/robot/{run_id}",
                    "pipeline": "tasks/demo/runs/pipeline/{pipeline_run_id}",
                },
                "system_refs": {
                    "mask": {
                        "path": "systems/mask/system.manifest.json",
                        "manifest_sha256": sha256(system_path),
                        "algorithm_id": "mask.demo.v1",
                        "weight_ids": ["weight.demo"],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_allocate_task_run_is_system_scoped_and_refuses_overwrite(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    make_registry(workspace)
    command = [
        sys.executable,
        str(TOOL),
        "--workspace",
        str(workspace),
        "--task",
        "demo",
        "--system",
        "mask",
        "--run-id",
        "20260902_smoke_v1",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    allocation_path = Path(completed.stdout.strip())
    allocation = json.loads(allocation_path.read_text())
    assert allocation_path == (
        workspace
        / "tasks"
        / "demo"
        / "runs"
        / "mask"
        / "20260902_smoke_v1"
        / "RUN_MANIFEST.json"
    )
    assert allocation["task_id"] == "demo"
    assert allocation["system_id"] == "mask"
    assert allocation["system_manifest"]["weight_ids"] == ["weight.demo"]
    assert allocation["output_policy"]["overwrite_allowed"] is False

    repeated = subprocess.run(command, check=False, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "File exists" in repeated.stderr


def test_allocate_task_run_rejects_path_traversal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    make_registry(workspace)
    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--workspace",
            str(workspace),
            "--task",
            "demo",
            "--system",
            "mask",
            "--run-id",
            "../escape",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "invalid run id" in completed.stderr
    assert not (workspace / "tasks" / "demo" / "runs").exists()
