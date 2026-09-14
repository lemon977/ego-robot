from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


RUNNER = Path(__file__).parents[1] / "tools/run_exact78_train010_raw_inventory_gpu.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("exact78_train010_raw_runner_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scope_is_single_train_session_and_excludes_blind_population():
    runner = load_runner()
    ref = runner.validate_split()
    assert runner.SESSION == "grap_a_cap_010"
    assert runner.SESSION != "grap_a_cap_025"
    assert ref["sha256"] == runner.SPLIT_SHA


def test_outputs_are_unmodified_individual_instances():
    runner = load_runner()
    masks = np.zeros((2, 1, runner.HEIGHT, runner.WIDTH), dtype=np.bool_)
    masks[0, 0, 4, 5] = True
    masks[1, 0, 8, 9] = True
    observed, scores, ids = runner.outputs_to_instances(
        {"out_binary_masks": masks, "out_probs": [0.7, 0.8], "out_obj_ids": [4, 7]}
    )
    assert np.array_equal(observed[:, [4, 8], [5, 9]], np.array([[True, False], [False, True]]))
    assert scores.tolist() == [0.7, 0.8]
    assert ids.tolist() == [4, 7]


def test_duplicate_instance_ids_fail_closed():
    runner = load_runner()
    masks = np.zeros((2, runner.HEIGHT, runner.WIDTH), dtype=np.bool_)
    with pytest.raises(runner.RawInventoryError, match="score/id"):
        runner.outputs_to_instances(
            {"out_binary_masks": masks, "out_probs": [0.7, 0.8], "out_obj_ids": [4, 4]}
        )


def test_checkpoint_path_replacement_is_detected(tmp_path, monkeypatch):
    runner = load_runner()
    path = tmp_path / "model.pt"
    payload = b"frozen-checkpoint-a"
    path.write_bytes(payload)
    monkeypatch.setattr(runner, "CHECKPOINT_BYTES", len(payload))
    monkeypatch.setattr(runner, "CHECKPOINT_SHA", hashlib.sha256(payload).hexdigest())
    descriptor, ref = runner.open_checkpoint(path)
    replacement = tmp_path / "replacement.pt"
    replacement.write_bytes(b"changed-checkpoint-b")
    os.replace(replacement, path)
    try:
        with pytest.raises(runner.RawInventoryError, match="path/inode"):
            runner.checkpoint_unchanged(descriptor, path, ref)
    finally:
        os.close(descriptor)


def test_run_attempts_are_fresh_direct_children(tmp_path, monkeypatch):
    runner = load_runner()
    (tmp_path / "_run").mkdir()
    (tmp_path / "_run" / f"{runner.RUN_PREFIX}_a1").mkdir()
    monkeypatch.setattr(runner, "PROJECT", tmp_path)
    result = runner.create_run_root()
    assert result.name == f"{runner.RUN_PREFIX}_a2"
    assert result.parent == tmp_path / "_run"


def test_two_frame_stage_binds_same_bytes_and_canonical_inventory(tmp_path, monkeypatch):
    runner = load_runner()
    raw_root = tmp_path / "source"
    rows = []
    total = 0
    canonical = hashlib.sha256()
    for frame in range(2):
        directory = raw_root / f"{frame:05d}"
        directory.mkdir(parents=True)
        path = directory / "rgb.png"
        Image.new("RGB", (runner.WIDTH, runner.HEIGHT), (frame, 2, 3)).save(path)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        rows.append((frame, len(payload), digest))
        total += len(payload)
        canonical.update(f"{frame:05d} {len(payload)} {digest}\n".encode())
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(runner, "RAW_ROOT", raw_root)
    monkeypatch.setattr(runner, "FRAME_COUNT", 2)
    monkeypatch.setattr(runner, "EXPECTED_RAW_BYTES", total)
    monkeypatch.setattr(runner, "EXPECTED_RAW_SHA_LINES", canonical.hexdigest())
    input_root, manifest_ref = runner.stage_raw_inputs(run_root)
    assert manifest_ref["sha256"]
    for frame, size, digest in rows:
        staged = input_root / f"{frame:05d}.png"
        assert staged.stat().st_size == size
        assert hashlib.sha256(staged.read_bytes()).hexdigest() == digest


def test_runner_has_no_selector_or_pixel_combination_path():
    source = RUNNER.read_text(encoding="utf-8")
    assert "select_frame(" not in source
    assert "morphology" not in source
    assert "object6d" not in source.lower()
    assert 'PROMPT = "an arm"' in source
    assert "COMPLETE_PROPAGATION_STREAM_ONLY" in source
