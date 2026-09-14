from __future__ import annotations

from pathlib import Path

from tools.train_embodiment import resolve_trainer_config


def test_launcher_resolves_the_exact_trainer_cli_fields(tmp_path: Path) -> None:
    reviewed = {
        "task": "grap_a_cap",
        "image_size": [240, 320],
        "img_name": "rgb.png",
        "epochs": 400,
        "persistent_workers": False,
        "model_h_weighting": "uniform",
        "model_h_beta": 0.0,
    }
    train = tmp_path / "grap_a_cap_001/09_humanego_adapter"
    validation = tmp_path / "grap_a_cap_002/09_humanego_adapter"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    resolved = resolve_trainer_config(
        reviewed,
        train_paths=[train],
        validation_paths=[validation],
        run_dir=run_dir,
        run_root=tmp_path,
        selected_batch=64,
        evaluation_batch=32,
        epochs=285,
        overfit_session=None,
        scratch=False,
    )
    assert resolved["MPS_PATHS_TRAIN"] == [str(train)]
    assert resolved["MPS_PATHS_EVAL"] == [str(validation)]
    assert resolved["out_dir"] == str(run_dir)
    assert resolved["run_root"] == str(tmp_path)
    assert resolved["batch_size"] == 64
    assert resolved["eval_batch_size"] == 32
    assert resolved["epochs"] == 285
    assert resolved["image_size"] == (240, 320)
    assert resolved["model_horizon_weighting"] == "uniform"
    assert resolved["model_horizon_beta"] == 0.0
    assert resolved["device"] == "cuda"
    assert resolved["skip_visual_eval"] is True
