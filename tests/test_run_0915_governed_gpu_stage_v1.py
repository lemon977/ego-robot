from __future__ import annotations

import json

from chaoyang.ops import run_0915_governed_gpu_stage_v1 as subject


def test_gpu_stages_are_finite_and_use_expected_pinned_launchers(tmp_path) -> None:
    assert set(subject.GPU_TASKS) == {
        "0915_hawor_full_v1",
        "0915_foundationstereo_full_v1",
        "0915_sam31_mask_full_v1",
    }
    hawor = subject.build_worker_command("0915_hawor_full_v1", tmp_path / "hawor")
    stereo = subject.build_worker_command("0915_foundationstereo_full_v1", tmp_path / "depth")
    sam = subject.build_worker_command("0915_sam31_mask_full_v1", tmp_path / "sam31")
    assert hawor[0].endswith("hawor_python.sh")
    assert stereo[0].endswith("foundationstereo_gpu_python.sh")
    assert sam[1].endswith("run_0915_sam31_persistent_masks_v2.py")


def test_sam_gpu_command_is_sam31_only(tmp_path) -> None:
    command = subject.build_worker_command("0915_sam31_mask_full_v1", tmp_path / "sam31")
    encoded = json.dumps(command).lower()
    assert "sam31" in encoded
    assert "sam2" not in encoded
    assert "cutie" not in encoded


def test_worker_output_placeholder_is_fully_resolved(tmp_path) -> None:
    for task_id in subject.GPU_TASKS:
        command = subject.build_worker_command(task_id, tmp_path / task_id)
        assert "{worker_output}" not in command
