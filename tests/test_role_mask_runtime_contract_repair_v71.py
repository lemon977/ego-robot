import json
from pathlib import Path

from chaoyang.ops import run_role_mask_runtime_contract_repair_v71 as runner


def test_validated_keeps_model_input_fps_and_records_container_diagnostic(monkeypatch, tmp_path):
    config = {
        "frame_count": 1,
        "raw_all_data": str(tmp_path / "all_data"),
        "raw_video": {"path": str(tmp_path / "raw.mp4"), "bytes": 1, "sha256": "x"},
        "hawor_npz": {"path": str(tmp_path / "hawor.npz"), "bytes": 1, "sha256": "y"},
        "fps": 30.0,
        "anchor_frame": 0,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    frame = tmp_path / "all_data/00000/rgb.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")
    (tmp_path / "raw.mp4").write_bytes(b"v")
    (tmp_path / "hawor.npz").write_bytes(b"n")
    monkeypatch.setattr(runner, "load", lambda _: config)
    monkeypatch.setattr(runner.cv2, "imread", lambda *args, **kwargs: __import__("numpy").zeros((960, 1280, 3), dtype="uint8"))
    monkeypatch.setattr(runner.legacy.canary, "verify_ref", lambda value, _: Path(value["path"]))

    class Capture:
        def __init__(self, _): pass
        def get(self, key):
            return 1 if key == runner.cv2.CAP_PROP_FRAME_COUNT else 25.0
        def release(self): pass

    monkeypatch.setattr(runner.cv2, "VideoCapture", Capture)

    class Archive:
        def __enter__(self):
            import numpy as np
            return {"joints_2d": np.zeros((2, 1, 21, 2)), "fps": np.array(25.0), "anatomical_side_names": np.array(["left", "right"])}
        def __exit__(self, *args): pass

    monkeypatch.setattr(runner.np, "load", lambda *args, **kwargs: Archive())
    result = runner.validated(config_path, tmp_path / "fresh")
    assert result["fps"] == 30.0
    assert result["raw_video_metadata_diagnostic"]["declared_fps"] == 25.0
    assert result["raw_video_metadata_diagnostic"]["fps_matches_model_input"] is False
    assert result["raw_video_metadata_diagnostic"]["hawor_declared_fps"] == 25.0
    assert result["raw_video_metadata_diagnostic"]["hawor_fps_matches_model_input"] is False
