from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "tools/build_quad_visual_review_index_t1.py"
SPEC = importlib.util.spec_from_file_location("quad_visual_review_index_tested", RUNNER)
assert SPEC and SPEC.loader
visual = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = visual
SPEC.loader.exec_module(visual)


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_json(path: Path, value: dict[str, object]) -> dict[str, object]:
    return _write(path, (json.dumps(value, sort_keys=True) + "\n").encode())


def _rows(frame_count: int) -> list[dict[str, object]]:
    rows = []
    for frame in range(frame_count):
        status = "ACCEPT" if frame < frame_count // 2 else "REJECT"
        rows.append(
            {
                "frame_index": frame,
                "adjacent_mask_iou": {
                    "left": None if frame == 0 else 1.0 - frame / (frame_count * 2),
                    "right": None if frame == 0 else 1.0 - frame / (frame_count * 3),
                },
                "selector_status": {"left": status, "right": "ACCEPT"},
                "stress": {
                    axis: {
                        "left": frame / max(1, frame_count - 1),
                        "right": (frame_count - frame - 1) / max(1, frame_count - 1),
                    }
                    for axis in visual.STRESS_KEYS
                },
            }
        )
    return rows


def _quad_run(
    session: str,
    frame_count: int,
    successor_freeze_ref: dict[str, object],
    video_ref: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "quad-review-video-run-manifest-v1",
        "status": "CANDIDATE_REVIEW_VIDEO_COMPLETE_NOT_BASELINE",
        "session_id": session,
        "frame_count": frame_count,
        "fps": 30,
        "layout": list(visual.LAYOUT),
        "source_resolution": [64, 48],
        "display_contract": {},
        "frame_policy": "EXACT_LOCKSTEP_NO_FILL_NO_REPEAT_NO_INTERPOLATION",
        "pixel_claim": "DISPLAY_ASSEMBLY_ONLY_NO_MASK_CLEAN_ROBOT_SOURCE_PIXELS_CREATED",
        "input_manifest": {"path": "/x", "bytes": 1, "sha256": "a" * 64},
        "cpu_freeze": successor_freeze_ref,
        "independent_input_qa": {"path": "/x", "bytes": 1, "sha256": "b" * 64},
        "panels": [
            {"role": role, "admitted_status": "CANDIDATE_ACCEPTABLE"}
            for role in visual.LAYOUT
        ],
        "video": video_ref,
        "video_probe": {},
        "encoder_terminal": {},
        "decoder_terminals": [],
        "wall_seconds": 1.0,
        "mount_provenance": "PROVISIONAL_MOUNT_VISUAL_ONLY",
        "contact_infeasible": "UNMEASURED",
        **visual.GOVERNANCE,
    }


def _release(
    project: Path, component: str, hold_ref: dict[str, object]
) -> dict[str, object]:
    successor = _write_json(
        project / "_run" / f"{component}_SUCCESSOR.json", {"component": component}
    )
    value = {
        "schema_version": "quad-upstream-a-hold-release-v1",
        "producer": "INDEPENDENT_CPU_QA",
        "status": "PASS",
        "a_class_p0": False,
        "component": component,
        "supersedes_hold": hold_ref,
        "successor_producer_manifest": successor,
        "quad_consumer_allowed": True,
    }
    return _write_json(project / "audits" / f"{component}_RELEASE.json", value)


def _fixture_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, object, dict[str, object]]:
    project = tmp_path / "project"
    run_root = project / "_run"
    run_root.mkdir(parents=True)
    strict = visual.load_strict_io(PROJECT)
    session = "synthetic_quad"
    frame_count = 24
    monkeypatch.setitem(visual.SESSIONS, session, {"frame_count": frame_count, "fps": 30})
    successor_freeze = _write_json(run_root / "CPU9_SUCCESSOR_FREEZE.json", {"ready": True})
    forbidden_freeze = _write_json(run_root / "CPU9_FORBIDDEN_FREEZE.json", {"hold": True})
    forbidden_qa = _write_json(run_root / "CPU9_FORBIDDEN_QA.json", {"hold": True})
    cpu9_a_hold = _write_json(project / "audits" / "CPU9_A_HOLD.json", {"hold": True})
    hold_refs = {
        component: _write_json(project / "audits" / f"{component}_HOLD.json", {"hold": True})
        for component in ("S7", "S8")
    }
    video = _write(run_root / "quad" / "QUAD.mp4", b"verified-quad-video")
    run = _quad_run(session, frame_count, successor_freeze, video)
    run_ref = _write_json(run_root / "quad" / "RUN_MANIFEST.json", run)
    releases = {
        component: _release(project, component, hold_refs[component])
        for component in ("S7", "S8")
    }
    cpu9_qa = {
        "schema_version": "cpu9-quad-run-independent-qa-v1",
        "producer": "INDEPENDENT_CPU_QA",
        "status": "PASS",
        "a_class_p0": False,
        "quad_run_manifest": run_ref,
        "quad_video": video,
        "session_id": session,
        "frame_count": frame_count,
        "fps": 30,
        "layout": list(visual.LAYOUT),
        "cpu9_successor_freeze": successor_freeze,
        "supersedes_forbidden_cpu9_author_freeze": forbidden_freeze,
        "supersedes_forbidden_cpu9_author_qa": forbidden_qa,
        "supersedes_cpu9_a_hold": cpu9_a_hold,
        "executable_lineage_recomputed": True,
        "upstream_hold_releases": releases,
        **visual.GOVERNANCE,
    }
    cpu9_qa_ref = _write_json(project / "audits" / "CPU9_OUTPUT_QA.json", cpu9_qa)
    diagnostics = {
        "schema_version": "cpu10-quad-visual-diagnostics-v1",
        "producer": "CPU_DIAGNOSTIC_ONLY",
        "status": "COMPLETE",
        "session_id": session,
        "frame_count": frame_count,
        "quad_run_manifest": run_ref,
        "metric_contract": {
            "adjacent_mask_iou": "PER_SIDE_RAW_UNALIGNED_IOU_NULL_ONLY_AT_FRAME_ZERO",
            "selector_status": "PER_SIDE_FROZEN_PRODUCER_STATUS_STRING",
            "wrist_wearable_stress": "PER_SIDE_NORMALIZED_ZERO_TO_ONE_HIGH_IS_STRESS",
            "contact_boundary_stress": "PER_SIDE_NORMALIZED_ZERO_TO_ONE_HIGH_IS_STRESS",
            "thumb_embedding_stress": "PER_SIDE_NORMALIZED_ZERO_TO_ONE_HIGH_IS_STRESS",
            "diagnostic_only_no_pixel_mutation": True,
        },
        "frames": _rows(frame_count),
    }
    diagnostics_ref = _write_json(project / "audits" / "DIAGNOSTICS.json", diagnostics)
    diagnostics_qa = {
        "schema_version": "cpu10-quad-visual-diagnostics-independent-qa-v1",
        "producer": "INDEPENDENT_CPU_QA",
        "status": "PASS",
        "a_class_p0": False,
        "diagnostics": diagnostics_ref,
        "quad_run_manifest": run_ref,
        "session_id": session,
        "frame_count": frame_count,
        "selection_metrics_recomputed": True,
    }
    diagnostics_qa_ref = _write_json(project / "audits" / "DIAGNOSTICS_QA.json", diagnostics_qa)
    runner_sha = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    freeze = {
        "schema_version": "cpu10-quad-visual-index-cpu-freeze-v1",
        "status": "CPU_READY_ADMISSION_CLOSED_UNTIL_S7_S8_A_HOLDS_SUPERSEDED",
        "implementation": {"runner": {"sha256": runner_sha}},
        "shared_immutable_io": {"sha256": visual.SHARED_IO_SHA256},
        "sampling_contract": visual.SAMPLING_CONTRACT,
        "semantic_context": {
            "forbidden_cpu9_author_freeze": forbidden_freeze,
            "forbidden_cpu9_author_qa": forbidden_qa,
            "cpu9_a_hold": cpu9_a_hold,
            "current_s7_hold": hold_refs["S7"],
            "current_s8_hold": hold_refs["S8"],
        },
    }
    freeze_ref = _write_json(run_root / "CPU10_FREEZE.json", freeze)
    freeze_value, freeze_record = strict.read_json_nofollow(
        freeze_ref["path"],
        expected_bytes=freeze_ref["bytes"],
        expected_sha256=freeze_ref["sha256"],
        allowed_root=run_root,
    )
    return project, strict, {
        "session": session,
        "runner_sha": runner_sha,
        "freeze": freeze_value,
        "freeze_record": freeze_record,
        "run_ref": run_ref,
        "video_ref": video,
        "cpu9_qa_ref": cpu9_qa_ref,
        "cpu9_qa": cpu9_qa,
        "successor_freeze": successor_freeze,
        "forbidden_freeze": forbidden_freeze,
        "diagnostics_ref": diagnostics_ref,
        "diagnostics_qa_ref": diagnostics_qa_ref,
        "hold_refs": hold_refs,
    }


def test_shared_io_compiles_exact_same_fd_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"SENTINEL='verified-A'\n"
    opened = io.BytesIO(payload)
    monkeypatch.setattr(visual.os, "open", lambda *_args, **_kwargs: 19)
    monkeypatch.setattr(visual.os, "fstat", lambda _fd: type("S", (), {"st_mode": 0o100444})())
    monkeypatch.setattr(visual.stat, "S_ISREG", lambda _mode: True)
    monkeypatch.setattr(visual.os, "read", lambda _fd, _n: opened.read(_n))
    monkeypatch.setattr(visual.os, "close", lambda _fd: None)
    module = visual._same_fd_module(
        Path("/replacement-B.py"),
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        module_name="cpu10_same_fd_counterexample",
    )
    assert module.SENTINEL == "verified-A"


def test_preregistered_sampling_is_deterministic_and_keeps_transition_context() -> None:
    rows = _rows(24)
    result = visual.build_selection(rows)
    assert result["uniform"] == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23]
    assert result["status_flip"] == [11, 12]
    assert len(result["low_iou"]) <= 12
    assert all(frame in result["low_iou"] for frame in (22, 23))
    for axis in visual.STRESS_KEYS:
        assert len(result[axis]) == 12


def test_current_forbidden_cpu9_author_freeze_cannot_be_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, strict, data = _fixture_tree(tmp_path, monkeypatch)
    with pytest.raises(visual.VisualIndexError, match="current CPU9 author freeze is A-HOLD"):
        visual.execute(
            strict=strict,
            project_root=project,
            cpu10_freeze=data["freeze"],
            cpu10_freeze_record=data["freeze_record"],
            expected_runner_sha256=data["runner_sha"],
            run_ref=data["run_ref"],
            cpu9_qa_ref=data["cpu9_qa_ref"],
            cpu9_successor_freeze_ref=data["forbidden_freeze"],
            diagnostics_ref=data["diagnostics_ref"],
            diagnostics_qa_ref=data["diagnostics_qa_ref"],
            output_parent=project / "_run",
            run_name="must_not_exist",
        )


def test_cpu9_independent_qa_p0_or_missing_s7_s8_release_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, strict, data = _fixture_tree(tmp_path, monkeypatch)
    qa = dict(data["cpu9_qa"])
    qa["a_class_p0"] = True
    with pytest.raises(visual.VisualIndexError, match="a_class_p0"):
        visual._validate_cpu9_independent_qa(
            qa,
            run_ref=data["run_ref"],
            video_ref=data["video_ref"],
            session=data["session"],
            frame_count=24,
            cpu9_successor_freeze_ref=data["successor_freeze"],
            forbidden_cpu9_freeze_ref=data["forbidden_freeze"],
            forbidden_cpu9_qa_ref=data["freeze"]["semantic_context"]["forbidden_cpu9_author_qa"],
            cpu9_a_hold_ref=data["freeze"]["semantic_context"]["cpu9_a_hold"],
            strict=strict,
            project_root=project,
            hold_refs=data["hold_refs"],
        )
    qa = dict(data["cpu9_qa"])
    qa["upstream_hold_releases"] = {}
    with pytest.raises(visual.VisualIndexError, match="current S7/S8 A-HOLDs"):
        visual._validate_cpu9_independent_qa(
            qa,
            run_ref=data["run_ref"],
            video_ref=data["video_ref"],
            session=data["session"],
            frame_count=24,
            cpu9_successor_freeze_ref=data["successor_freeze"],
            forbidden_cpu9_freeze_ref=data["forbidden_freeze"],
            forbidden_cpu9_qa_ref=data["freeze"]["semantic_context"]["forbidden_cpu9_author_qa"],
            cpu9_a_hold_ref=data["freeze"]["semantic_context"]["cpu9_a_hold"],
            strict=strict,
            project_root=project,
            hold_refs=data["hold_refs"],
        )


def test_quad_run_video_exact_reference_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, strict, data = _fixture_tree(tmp_path, monkeypatch)
    del project, strict
    run = json.loads(Path(data["run_ref"]["path"]).read_text())
    wrong = dict(data["successor_freeze"])
    wrong["sha256"] = "0" * 64
    with pytest.raises(visual.VisualIndexError, match="successor CPU9 freeze"):
        visual._validate_quad_run(
            run, run_ref=data["run_ref"], cpu9_successor_freeze_ref=wrong
        )


def test_forbidden_partitions_fail_closed() -> None:
    for name in ("labels", "future_blind", "grap_a_cap_025", "processed"):
        with pytest.raises(visual.VisualIndexError, match="forbidden partition"):
            visual._evidence_ref(
                {"path": f"/mnt/workspace/{name}/x.json", "bytes": 1, "sha256": "a" * 64},
                name,
            )


def test_end_to_end_builds_read_only_pending_index_without_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project, strict, data = _fixture_tree(tmp_path, monkeypatch)
    result = visual.execute(
        strict=strict,
        project_root=project,
        cpu10_freeze=data["freeze"],
        cpu10_freeze_record=data["freeze_record"],
        expected_runner_sha256=data["runner_sha"],
        run_ref=data["run_ref"],
        cpu9_qa_ref=data["cpu9_qa_ref"],
        cpu9_successor_freeze_ref=data["successor_freeze"],
        diagnostics_ref=data["diagnostics_ref"],
        diagnostics_qa_ref=data["diagnostics_qa_ref"],
        output_parent=project / "_run",
        run_name="review_index",
    )
    assert result["status"] == "REVIEW_INDEX_COMPLETE_HUMAN_VERDICT_PENDING"
    assert result["automatic_pass"] is False
    assert result["pixels_produced"] == 0
    status_path = Path(result["human_review_status_template"]["path"])
    status = json.loads(status_path.read_text())
    assert status["status"] == "PENDING_HUMAN_REVIEW_NO_AUTOMATIC_PASS"
    assert all(row["review_status"] == "PENDING_HUMAN_REVIEW" for row in status["rows"])
    assert all(
        value == "UNREVIEWED"
        for row in status["rows"]
        for value in row["checks"].values()
    )
    for name in ("REVIEW_INDEX.json", "HUMAN_REVIEW_STATUS_TEMPLATE.json", "RUN_MANIFEST.json"):
        mode = (project / "_run" / "review_index" / name).stat().st_mode & 0o777
        assert mode == 0o440
