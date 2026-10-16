"""Read-only CPFS publisher metadata regression in an isolated CPU process."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path("/path/to/chaoyang")
CPFS_ROOT = Path("/path/to/private_workspace/code/chaoyang")
FIX_ROOT = PROJECT_ROOT / "_run/maintenance/joint26_cpfs_smoke_20261013/fix_a01"


_READ_ONLY_PROBE = r"""
import hashlib
import importlib.util
import json
import pathlib
import sys

original_path = pathlib.Path
root, unknown_path = map(original_path, sys.argv[1:3])
board_path = root / "docs/current/PROJECT_BRANCH_MAP.json"
board_before = board_path.read_bytes()
unknown_before = unknown_path.read_bytes()
sys.path.insert(0, str(root / "src"))
from chaoyang.humanego_ict import governance_mount_compat as compat
spec = importlib.util.spec_from_file_location(
    "_read_only_cpfs_joint26_runner", root / "scripts/humanego_ict/joint26_runner_v1.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
assert runner.ROOT == root and runner.MOUNT_COMPAT["enabled"] is True
assert original_path(compat.__file__).is_relative_to(root / "src")
from chaoyang.governance import common
from chaoyang.governance import project_branch_navigation_v1 as navigation
board = navigation.load_json(board_path)
built = navigation.build_branch_map(root)
# Synthetic empty activity avoids opening or changing the real task ledger.
# The adopted board and all build_branch_map sources are the real CPFS files.
live = navigation.build_live_status(root, {"tasks": []}, "SYNTHETIC_CPU_READ_ONLY")
unknown = common.load_json(unknown_path)
board_after = board_path.read_bytes()
unknown_after = unknown_path.read_bytes()
print(json.dumps({
    "cpfs_source_imported": True,
    "runtime_root": str(runner.ROOT),
    "board_matches_actual_builder": board == built,
    "original_board_needed_metadata_view": json.loads(board_before) != board,
    "live_schema": live["schema_version"],
    "selected_routes": live["counts"]["selected_visual_routes"],
    "research_flags_unchanged": live["training_eligible"] is False
        and live["control_ground_truth"] is False,
    "board_bytes_unchanged": board_before == board_after,
    "board_sha_unchanged": hashlib.sha256(board_before).hexdigest()
        == hashlib.sha256(board_after).hexdigest(),
    "global_path_unchanged": pathlib.Path is original_path,
    "unknown_view_unchanged": unknown == json.loads(unknown_before),
    "unknown_bytes_unchanged": unknown_before == unknown_after,
    "unknown_sha_unchanged": hashlib.sha256(unknown_before).hexdigest()
        == hashlib.sha256(unknown_after).hexdigest(),
}, sort_keys=True))
"""


@pytest.fixture(scope="module")
def cpfs_metadata_probe(tmp_path_factory):
    """Import only; never call execute, publish, a model, or a GPU operation."""
    scratch = tmp_path_factory.mktemp("cpfs_metadata_probe")
    assert scratch.resolve().is_relative_to(FIX_ROOT.resolve()), "FIX_ATTEMPT_FIXTURES_REQUIRED"
    unknown = scratch / "unknown_metadata_view.json"
    unknown.write_text(json.dumps({
        "path": str(PROJECT_ROOT / "docs/current/PROJECT_BRANCH_MAP.json"),
        "nested": {"path": str(PROJECT_ROOT / "signed_input.json"),
                   "bytes": 7, "sha256": "0" * 64},
    }), encoding="utf-8")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(CPFS_ROOT / "src"),
                       TMPDIR=str(scratch), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-B", "-c", _READ_ONLY_PROBE,
                             str(CPFS_ROOT), str(unknown)],
                            cwd=CPFS_ROOT, env=environment, capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, (result.stdout + result.stderr)[-1800:]
    return json.loads(result.stdout.strip())


def test_real_cpfs_branch_board_matches_actual_live_status_builder(cpfs_metadata_probe):
    probe = cpfs_metadata_probe
    assert probe["cpfs_source_imported"]
    assert probe["runtime_root"] == str(CPFS_ROOT)
    assert probe["original_board_needed_metadata_view"]
    assert probe["board_matches_actual_builder"]
    assert probe["live_schema"] == "PROJECT_BRANCH_LIVE_STATUS_V1"
    assert probe["selected_routes"] == 2
    assert probe["research_flags_unchanged"]


def test_metadata_view_preserves_real_board_and_global_path_class(cpfs_metadata_probe):
    assert cpfs_metadata_probe["board_bytes_unchanged"]
    assert cpfs_metadata_probe["board_sha_unchanged"]
    assert cpfs_metadata_probe["global_path_unchanged"]


def test_unknown_metadata_view_keeps_signed_paths_and_bytes(cpfs_metadata_probe):
    assert cpfs_metadata_probe["unknown_view_unchanged"]
    assert cpfs_metadata_probe["unknown_bytes_unchanged"]
    assert cpfs_metadata_probe["unknown_sha_unchanged"]
