"""No Task Packet registration, render, video write, or model call."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from chaoyang.governance import execute_exact78_cards042_clean_robot_diagnostic_v1 as runner
from chaoyang.ops import run_exact78_cards042_clean_robot_diagnostic_v1 as operation
from chaoyang.pipeline import exact78_clean_robot_diagnostic_cards042_v1 as candidate


@pytest.fixture(scope="module")
def frozen() -> tuple[dict, dict]:
    spec = runner.frozen_spec()
    return spec, candidate.prepare_binding(spec["inputs"])


def test_packet_is_finite_cpu_only_and_unregistered(frozen: tuple[dict, dict]) -> None:
    spec, _ = frozen
    packet = runner.packet_value(runner.virtual_json_ref(runner.SPEC, spec), spec)
    assert packet["task_id"] == operation.TASK
    assert packet["write_set"][0] == str(runner.OUT.relative_to(runner.REPO_ROOT))
    assert packet["budgets"]["cpu_threads_total"] == 2
    assert packet["budgets"]["gpu_count"] == 0
    assert packet["budgets"]["session_attempts_max"] == 1
    assert packet["config"]["scene_version"] == "SEALED_FRESH_V1_NOT_V4"
    assert packet["config"]["model_calls"] == 0
    assert packet["config"]["object_mask_occlusion_authority"] is False
    assert packet["inputs"]["sealed_fresh_outers"] == spec["inputs"]
    assert packet["commands"] == [runner.COMMAND]
    assert len(packet["read_set"]) <= 8
    assert not runner.OUT.exists()
    assert not runner.PACKET.exists()


def test_without_route_operation_rejects_before_pipeline() -> None:
    with pytest.raises(operation.CompositeOperationError, match="COMPOSITE_NOT_ROUTABLE"):
        operation.authorize(runner.SPEC)


def test_encoder_is_single_thread_without_starting_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_popen(command, **kwargs):
        seen.update(command=command, kwargs=kwargs)
        return object()

    monkeypatch.setattr(operation.subprocess, "Popen", fake_popen)
    path = runner.OUT / "candidate/videos/dry_run.mp4"
    temp = path.parent / "tmp"
    operation._single_thread_encoder(path, (640, 480), temp)
    assert seen["command"][seen["command"].index("-threads") + 1] == "1"
    assert "-fps_mode" not in seen["command"]
    assert seen["kwargs"]["env"]["TMPDIR"] == str(temp)


def test_review_panel_is_display_only(frozen: tuple[dict, dict]) -> None:
    _, binding = frozen
    synthetic = np.full((480, 640, 3), 140, np.uint8)
    panel, state = operation.review_panel(binding, 0, synthetic)
    assert panel.shape == (960, 1280, 3)
    assert state["mask_geometry_authority"] is False
    assert state["raw"] == binding["raw_rows"][0]["model_input"]
    assert state["clean"] == binding["clean_rows"][0]["clean"]
    assert len(state["review_panel_array_sha256"]) == 64
    with pytest.raises(operation.CompositeOperationError, match="COMPOSITE_FRAME_DOMAIN_DRIFT"):
        operation.review_panel(binding, 0, np.empty((2, 2, 3), np.uint8))


def _probe_for(path: Path) -> dict:
    command = ["/usr/bin/ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries",
               "stream=width,height,avg_frame_rate,r_frame_rate,time_base,duration_ts,nb_frames:frame=best_effort_timestamp,pkt_duration",
               "-of", "json", str(path)]
    return json.loads(subprocess.run(command, capture_output=True, text=True,
                                     check=True).stdout)


def test_existing_sealed_ffmpeg44_video_passes_exact_cpu_verifier(frozen: tuple[dict, dict]) -> None:
    _, binding = frozen
    old = runner.load_json(Path(binding["render"]["candidate_result"]["path"]))
    path = Path(old["videos"]["ego_robot"]["video"]["path"])
    observed = operation.verify_video(path, 171, (640, 480), 30)
    assert observed["decoded_frames"] == 171
    assert observed["video"] == old["videos"]["ego_robot"]["video"]


@pytest.mark.parametrize("fault", ["missing_frame", "bad_pts", "bad_fps"])
def test_video_verifier_rejects_missing_frame_pts_and_fps(
        frozen: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    _, binding = frozen
    old = runner.load_json(Path(binding["render"]["candidate_result"]["path"]))
    path = Path(old["videos"]["ego_robot"]["video"]["path"])
    payload = deepcopy(_probe_for(path))
    if fault == "missing_frame":
        payload["frames"].pop()
    elif fault == "bad_pts":
        payload["frames"][75]["best_effort_timestamp"] = "-1"
    else:
        payload["streams"][0]["avg_frame_rate"] = "25/1"

    class Process:
        stdout = json.dumps(payload)

    monkeypatch.setattr(operation.subprocess, "run", lambda *_args, **_kwargs: Process())
    with pytest.raises(operation.CompositeOperationError):
        operation.verify_video(path, 171, (640, 480), 30)


def test_candidate_claims_cannot_be_upgraded_by_video_metadata() -> None:
    # The terminal verifier rejects content claims before looking at MP4s.
    spec = runner.frozen_spec()
    fake = {"status": "COMPLETE_DIAGNOSTIC_VIDEO_NOT_VISUAL_PASS",
            "session_id": candidate.SESSION, "frame_count": 171,
            "fps_nominal": 30, "diagnostic_video_complete": True,
            "inputs": spec["inputs"],
            "time_mode": "CLIP_INDEX_ONLY_TIMEBASE_UNVERIFIED",
            "geometry_decision": "SCENE_FRONT",
            "display_policy": candidate.DISPLAY_POLICY,
            **{key: False for key in ("scene_depth_consumed", "object_mask_consumed_for_occlusion",
                                       "visual_qualified", "product_qualified",
                                       "training_eligible", "control_ground_truth")}}
    assert not runner.candidate_scope_valid(fake, spec)
    fake["geometry_decision"] = "UNKNOWN"
    assert runner.candidate_scope_valid(fake, spec)
    fake["visual_qualified"] = True
    assert not runner.candidate_scope_valid(fake, spec)
    # The actual candidate verifier reads only the future attempt path; no
    # attempt exists yet, hence any fabricated metadata cannot pass/seal.
    result, errors = runner.verify_candidate(spec)
    assert result is None and errors == ["CANDIDATE_RESULT_MISSING"]
