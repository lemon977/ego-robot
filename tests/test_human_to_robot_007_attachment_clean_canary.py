from __future__ import annotations

import inspect

import cv2
import numpy as np

from chaoyang.ops.run_human_to_robot_007_attachment_clean_canary import (
    ATTACH, OBJECT, REBOUND, complaint_coverage, main, seed_model_for_full_write,
)
from chaoyang.ops.run_human_to_robot_source_coverage_rebind_007 import merge_support


def test_all_frozen_attachment_frames_exist_and_frame184_covers_complaints() -> None:
    for frame in range(181, 197):
        value = cv2.imread(str(ATTACH / "candidate_masks" / f"{frame:06d}.png"), 0)
        assert value is not None and value.shape == (960, 1280) and np.any(value)
    anchor = cv2.imread(str(ATTACH / "candidate_masks/000184.png"), 0) > 0
    assert all(complaint_coverage(anchor).values())


def test_actual_merge_keeps_old_protect_and_reports_object_conflict() -> None:
    frame, local = 181, 0
    attachment = cv2.imread(str(ATTACH / "candidate_masks" / f"{frame:06d}.png"), 0) > 0
    current = cv2.imread(str(REBOUND / "model_masks" / f"{local:06d}.png"), 0) > 0
    write = cv2.imread(str(REBOUND / "write" / f"{local:06d}.png"), 0) > 0
    protect = cv2.imread(str(REBOUND / "protect" / f"{local:06d}.png"), 0) > 0
    task_object = cv2.imread(str(OBJECT / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED) > 0
    model, new_write = merge_support(current, attachment, write, protect)
    assert model.shape == (720, 960) and new_write.shape == (960, 1280)
    assert np.all(new_write[write]) and np.all(new_write[attachment])
    assert not np.any(new_write & protect)
    assert int(np.count_nonzero(attachment & task_object)) > 0


def test_runner_has_real_model_and_byte_exact_repaste_checks() -> None:
    source = inspect.getsource(main)
    assert "subprocess.run(command, cwd=VENDOR" in source
    assert '"--mask_dilation", "0"' in source
    assert "changed & ~write" in source and "changed & protect" in source
    assert "attachment_task_object_overlap_pixels" in source
    assert "if DEST.exists()" in source


def test_changed_input_model_covers_full_write_even_at_protection_boundary() -> None:
    for frame in (190, 191, 192, 193):
        local = frame - 181
        attachment = cv2.imread(str(ATTACH / "candidate_masks" / f"{frame:06d}.png"), 0) > 0
        current = cv2.imread(str(REBOUND / "model_masks" / f"{local:06d}.png"), 0) > 0
        write = cv2.imread(str(REBOUND / "write" / f"{local:06d}.png"), 0) > 0
        protect = cv2.imread(str(REBOUND / "protect" / f"{local:06d}.png"), 0) > 0
        admissible = attachment & ~protect
        seeded = seed_model_for_full_write(current, write | admissible)
        model, new_write = merge_support(seeded, admissible, write, protect)
        upsampled = cv2.resize(model.astype(np.uint8), (1280, 960), interpolation=cv2.INTER_NEAREST) > 0
        assert np.all(upsampled[new_write])
        assert not np.any(new_write & protect)
