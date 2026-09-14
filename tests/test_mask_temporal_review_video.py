import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.mask_temporal_review_video import (
    MaskTemporalReviewError,
    ROLE_NAMES,
    author_temporal_reviews,
)


def _write_fixture(tmp_path: Path, *, automatic_pass: bool = True) -> tuple[Path, Path]:
    session = tmp_path / "session"
    all_data = session / "preprocess" / "all_data"
    run_root = tmp_path / "run"
    events = (("early", 0), ("middle", 20), ("late", 40))
    windows = []
    result_windows = []
    for event, start in events:
        frames = list(range(start, start + 12))
        windows.append({"event": event, "source_frames": frames})
        rows = []
        for slot, frame in enumerate(frames):
            frame_root = all_data / f"{frame:05d}"
            frame_root.mkdir(parents=True)
            image = np.full((48, 64, 3), (25 + frame, 70, 110), dtype=np.uint8)
            assert cv2.imwrite(str(frame_root / "rgb.png"), image)
            role_masks: dict[str, np.ndarray] = {}
            for role in ROLE_NAMES:
                role_masks[role] = np.zeros((48, 64), dtype=np.uint8)
            role_masks["left_human"][8:25, 5:16] = 255
            role_masks["right_human"][8:25, 48:59] = 255
            role_masks["left_tracker"][23:29, 10:17] = 255
            role_masks["right_tracker"][23:29, 47:54] = 255
            role_masks["task_object"][27:40, 25:39] = 255
            role_masks["fixture"][39:45, 20:44] = 255
            removal_raw = np.logical_or.reduce([role_masks[name] != 0 for name in ROLE_NAMES[:4]])
            protected = np.logical_or.reduce([role_masks[name] != 0 for name in ROLE_NAMES[4:]])
            role_masks["removal_union"] = ((removal_raw & ~protected) * 255).astype(np.uint8)
            for role, mask in role_masks.items():
                destination = run_root / "temporal" / event / role / f"{frame:05d}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                assert cv2.imwrite(str(destination), mask)
            rows.append({"slot": slot, "source_frame": frame, "status": "PASS"})
        result_windows.append({"event": event, "source_frames": frames, "status": "PASS", "frames": rows})
    plan = {
        "task_id": "chips",
        "session_id": "fixture_001",
        "split": "FIT",
        "canonical_session_path": str(session),
        "all_data_relative_path": "preprocess/all_data",
        "temporal_windows": windows,
    }
    plan_path = tmp_path / "PLAN.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    result = {
        "schema_version": "pico-raw-point-geometry-mask-canary-result-v1",
        "task_id": "chips",
        "session_id": "fixture_001",
        "split": "FIT",
        "automatic_gate_pass": automatic_pass,
        "temporal": {"windows": result_windows},
    }
    run_root.mkdir(exist_ok=True)
    (run_root / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
    return run_root, plan_path


def test_authors_three_role_union_videos_and_fully_decodes(tmp_path: Path) -> None:
    run_root, plan = _write_fixture(tmp_path)
    result = author_temporal_reviews(
        run_root=run_root,
        evaluation_plan_path=plan,
        output_root=tmp_path / "review",
        fps=6.0,
        panel_width=160,
    )
    assert result["status"] == "PASS_FULL_DECODE_REVIEW_ONLY"
    assert result["consumption_authorized"] is False
    assert len(result["windows"]) == 3
    assert all(window["decode_audit"]["full_decode"] for window in result["windows"])
    assert all(window["decode_audit"]["decoded_frames"] == 12 for window in result["windows"])
    assert all(window["decode_audit"]["codec_name"] == "h264" for window in result["windows"])
    assert Path(result["poster"]["path"]).is_file()
    persisted = json.loads((tmp_path / "review" / "RESULT.json").read_text(encoding="utf-8"))
    assert persisted["render_contract"]["windows"] == 3
    assert persisted["render_contract"]["gpu_calls"] == 0


def test_refuses_canary_that_did_not_pass_automatic_gate(tmp_path: Path) -> None:
    run_root, plan = _write_fixture(tmp_path, automatic_pass=False)
    with pytest.raises(MaskTemporalReviewError, match="automatic Mask gate PASS"):
        author_temporal_reviews(
            run_root=run_root,
            evaluation_plan_path=plan,
            output_root=tmp_path / "review",
            panel_width=160,
        )


def test_refuses_removal_union_semantic_drift(tmp_path: Path) -> None:
    run_root, plan = _write_fixture(tmp_path)
    corrupt = run_root / "temporal" / "early" / "removal_union" / "00000.png"
    mask = cv2.imread(str(corrupt), cv2.IMREAD_GRAYSCALE)
    mask[0, 0] = 255
    assert cv2.imwrite(str(corrupt), mask)
    with pytest.raises(MaskTemporalReviewError, match="removal_union semantic mismatch"):
        author_temporal_reviews(
            run_root=run_root,
            evaluation_plan_path=plan,
            output_root=tmp_path / "review",
            panel_width=160,
        )
