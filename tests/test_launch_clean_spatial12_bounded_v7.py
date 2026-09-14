from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tools import launch_clean_spatial12_bounded_v7 as launcher


def adapted_result(root: Path, *, status: str = "PASS_CANARY_ONLY") -> Path:
    root.mkdir(parents=True)
    (root / "RESULT.json").write_text(
        json.dumps(
            {
                "status": status,
                "v7_producer_adapter": {
                    "sha256": launcher.ADAPTER_SHA256,
                    "only_replaced_symbol": "donor_pico_exclusion",
                    "present_side_semantics": "UNCHANGED_ORIGINAL_GEOMETRY",
                    "absent_side_semantics": "ABSENT_NO_AUTHORIZED_SUPPORT",
                    "absent_side_visible_joint_count": 0,
                    "absent_side_fake_points_or_mask_support": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def test_real_static_pins_adapter_original_and_v6() -> None:
    frames, evidence = launcher.validate_static()
    assert len(frames) == 12
    pins = evidence["v7_pins"]
    assert pins["V7 producer adapter"]["sha256"] == launcher.ADAPTER_SHA256
    assert pins["V6 predecessor"]["sha256"] == launcher.V6_SHA256
    assert (
        pins["original producer implementation"]["sha256"]
        == launcher.ORIGINAL_IMPLEMENTATION_SHA256
    )


def test_fixed_argv_uses_adapter_directly_in_ram() -> None:
    output = Path("/dev/shm/v7-test/producer")
    argv = launcher.fixed_producer_argv(project=launcher.PROJECT, ram_output=output)
    assert argv[0] == str(launcher.PROJECT / launcher.v6.v5.LOCAL_LAUNCHER_RELATIVE)
    assert argv[1] == str(launcher.PROJECT / launcher.ADAPTER_RELATIVE)
    assert argv[argv.index("--output") + 1] == str(output)
    assert str(launcher.v6.v4.BASE_RUNNER) not in argv


def test_result_requires_adapter_evidence_and_normalizes_status(tmp_path: Path) -> None:
    root = adapted_result(tmp_path / "candidate")
    status = launcher.normalize_and_annotate_result(
        root,
        donor_before={"status": "PASS_BEFORE"},
        donor_after={"status": "PASS_AFTER"},
    )
    result = json.loads((root / "RESULT.json").read_text(encoding="utf-8"))
    assert status == "PASS_SPATIAL12_DIAGNOSTIC_ONLY"
    assert result["consumption_authorized"] is False
    gpu = result["v7_cpu_only_gpu_coexistence"]
    assert gpu["external_gpu_present"] is True
    assert gpu["does_not_authorize_or_use_gpu"] is True
    assert gpu["cuda_visible_devices"] == ""
    assert gpu["gpu_calls"] == 0


def test_missing_or_drifted_adapter_evidence_fails_closed(tmp_path: Path) -> None:
    root = adapted_result(tmp_path / "candidate")
    result_path = root / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["v7_producer_adapter"]["absent_side_fake_points_or_mask_support"] = True
    result_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(launcher.CleanV7LaunchHold, match="adapter RESULT"):
        launcher.normalize_and_annotate_result(
            root, donor_before={}, donor_after={}
        )


def test_v7_output_is_fresh_task_run_not_v3_producer() -> None:
    assert launcher.FINAL_OUTPUT_RELATIVE.parent == launcher.RUN_RELATIVE
    assert launcher.FINAL_OUTPUT_RELATIVE != launcher.v4.OUTPUT_RELATIVE
    assert launcher.OWNED_PUBLISHING_RELATIVE.parent == launcher.RUN_RELATIVE
    assert launcher.FAILURE_EVIDENCE_RELATIVE == launcher.RUN_RELATIVE


def test_direct_validate_only_works_outside_repo_cwd(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            str(launcher.PROJECT / launcher.v6.v5.LOCAL_LAUNCHER_RELATIVE),
            str(Path(launcher.__file__).resolve()),
            "--validate-only",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "PASS_V7_STATIC_VALIDATE_ONLY"
    assert result["execution_performed"] is False
    assert result["gpu_calls"] == 0
