from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "tools/build_quad_visual_review_index_t1.py"
SPEC = importlib.util.spec_from_file_location("cpu10_independent_hold_replay", RUNNER)
assert SPEC and SPEC.loader
visual = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = visual
SPEC.loader.exec_module(visual)


def _ref(path: Path, payload: bytes) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _ref(path, payload)


def _write_json(path: Path, value: dict[str, object]) -> dict[str, object]:
    return _write(path, (json.dumps(value, sort_keys=True) + "\n").encode())


def _diagnostic_rows(frame_count: int) -> list[dict[str, object]]:
    return [
        {
            "frame_index": frame,
            "adjacent_mask_iou": {
                "left": None if frame == 0 else 0.9,
                "right": None if frame == 0 else 0.8,
            },
            "selector_status": {
                "left": "ADVANCEMENT_AUTHORIZED" if frame == 1 else "ACCEPT",
                "right": "ACCEPT",
            },
            "stress": {
                axis: {"left": 0.5, "right": 0.5} for axis in visual.STRESS_KEYS
            },
        }
        for frame in range(frame_count)
    ]


def test_hold_replay_forged_same_path_successors_and_media_alias_are_admitted(
    tmp_path: Path,
) -> None:
    """One end-to-end replay exercises the missing trust/identity joins.

    The current implementation accepts a caller-authored CPU10 freeze, reuses the
    exact frozen CPU9 author paths after replacing their bytes, treats one byte
    string as both the CPU9 freeze and the quad video, and accepts S7/S8 releases
    whose claimed successor manifests do not exist.
    """

    project = tmp_path / "project"
    run_root = project / "_run"
    audits = project / "audits"
    run_root.mkdir(parents=True)
    audits.mkdir()
    strict = visual.load_strict_io(PROJECT)
    runner_sha = hashlib.sha256(RUNNER.read_bytes()).hexdigest()

    # The old refs are frozen claims.  Their paths are then reused with different
    # bytes.  CPU10 compares full triples, not pathname identity, and never re-reads
    # the frozen bytes.
    shared_freeze_path = run_root / "CPU9_FREEZE_FINAL.json"
    forbidden_freeze_ref = _ref(shared_freeze_path, b"forbidden-freeze-A")
    successor_and_video_ref = _write(shared_freeze_path, b"not-a-freeze-or-mp4-B")
    shared_qa_path = audits / "CPU9_QA_FINAL.json"
    forbidden_qa_ref = _ref(shared_qa_path, b"forbidden-author-qa-A")

    cpu9_hold_ref = _ref(audits / "CPU9_A_HOLD.json", b"unread-hold")
    hold_refs = {
        component: _ref(audits / f"{component}_HOLD.json", b"unread-hold")
        for component in ("S7", "S8")
    }
    fake_cpu10_freeze = {
        "schema_version": "cpu10-quad-visual-index-cpu-freeze-v1",
        "status": "CPU_READY_ADMISSION_CLOSED_UNTIL_S7_S8_A_HOLDS_SUPERSEDED",
        "implementation": {"runner": {"sha256": runner_sha}},
        "shared_immutable_io": {"sha256": visual.SHARED_IO_SHA256},
        "sampling_contract": visual.SAMPLING_CONTRACT,
        "semantic_context": {
            "forbidden_cpu9_author_freeze": forbidden_freeze_ref,
            "forbidden_cpu9_author_qa": forbidden_qa_ref,
            "cpu9_a_hold": cpu9_hold_ref,
            "current_s7_hold": hold_refs["S7"],
            "current_s8_hold": hold_refs["S8"],
        },
    }
    cpu10_freeze_ref = _write_json(run_root / "CALLER_AUTHORED_CPU10_FREEZE.json", fake_cpu10_freeze)
    cpu10_freeze, cpu10_freeze_record = strict.read_json_nofollow(
        cpu10_freeze_ref["path"],
        expected_bytes=cpu10_freeze_ref["bytes"],
        expected_sha256=cpu10_freeze_ref["sha256"],
        allowed_root=run_root,
    )

    session = "grap_a_cap_002"
    frame_count = visual.SESSIONS[session]["frame_count"]
    run = {
        "schema_version": "quad-review-video-run-manifest-v1",
        "status": "CANDIDATE_REVIEW_VIDEO_COMPLETE_NOT_BASELINE",
        "session_id": session,
        "frame_count": frame_count,
        "fps": 30,
        "layout": list(visual.LAYOUT),
        "source_resolution": [1, 1],
        "display_contract": {"unverified": True},
        "frame_policy": "EXACT_LOCKSTEP_NO_FILL_NO_REPEAT_NO_INTERPOLATION",
        "pixel_claim": "DISPLAY_ASSEMBLY_ONLY_NO_MASK_CLEAN_ROBOT_SOURCE_PIXELS_CREATED",
        # These forbidden nested refs are never validated or consumed.
        "input_manifest": {"path": "/labels/forbidden.json"},
        "independent_input_qa": {"path": "/processed/forbidden.json"},
        "cpu_freeze": successor_and_video_ref,
        "panels": [
            {
                "role": role,
                "admitted_status": "CANDIDATE_ACCEPTABLE",
                "source": {"path": "/blind/unjoined.json"},
            }
            for role in visual.LAYOUT
        ],
        # The same exact non-media bytes are accepted in the freeze and video roles.
        "video": successor_and_video_ref,
        "video_probe": {"unverified": True},
        "encoder_terminal": {},
        "decoder_terminals": [],
        "wall_seconds": 0.0,
        "mount_provenance": "PROVISIONAL_MOUNT_VISUAL_ONLY",
        "contact_infeasible": "UNMEASURED",
        **visual.GOVERNANCE,
    }
    run_ref = _write_json(run_root / "quad" / "RUN_MANIFEST.json", run)

    releases: dict[str, dict[str, object]] = {}
    for component in ("S7", "S8"):
        missing_successor_ref = _ref(
            run_root / f"MISSING_{component}_SUCCESSOR.json", b"never-created"
        )
        release = {
            "schema_version": "quad-upstream-a-hold-release-v1",
            "producer": "INDEPENDENT_CPU_QA",
            "status": "PASS",
            "a_class_p0": False,
            "component": component,
            "supersedes_hold": hold_refs[component],
            "successor_producer_manifest": missing_successor_ref,
            "quad_consumer_allowed": True,
        }
        releases[component] = _write_json(audits / f"{component}_RELEASE.json", release)

    cpu9_qa = {
        "schema_version": "cpu9-quad-run-independent-qa-v1",
        "producer": "INDEPENDENT_CPU_QA",
        "status": "PASS",
        "a_class_p0": False,
        "quad_run_manifest": run_ref,
        "quad_video": successor_and_video_ref,
        "session_id": session,
        "frame_count": frame_count,
        "fps": 30,
        "layout": list(visual.LAYOUT),
        "cpu9_successor_freeze": successor_and_video_ref,
        "supersedes_forbidden_cpu9_author_freeze": forbidden_freeze_ref,
        "supersedes_forbidden_cpu9_author_qa": forbidden_qa_ref,
        "supersedes_cpu9_a_hold": cpu9_hold_ref,
        "executable_lineage_recomputed": True,
        "upstream_hold_releases": releases,
        **visual.GOVERNANCE,
    }
    # This is the exact old author-QA pathname with replacement bytes.
    cpu9_qa_ref = _write_json(shared_qa_path, cpu9_qa)

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
        "frames": _diagnostic_rows(frame_count),
    }
    diagnostics_ref = _write_json(audits / "DIAGNOSTICS.json", diagnostics)
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
    diagnostics_qa_ref = _write_json(audits / "DIAGNOSTICS_QA.json", diagnostics_qa)

    result = visual.execute(
        strict=strict,
        project_root=project,
        cpu10_freeze=cpu10_freeze,
        cpu10_freeze_record=cpu10_freeze_record,
        expected_runner_sha256=runner_sha,
        run_ref=run_ref,
        cpu9_qa_ref=cpu9_qa_ref,
        cpu9_successor_freeze_ref=successor_and_video_ref,
        diagnostics_ref=diagnostics_ref,
        diagnostics_qa_ref=diagnostics_qa_ref,
        output_parent=run_root,
        run_name="counterexample_admitted",
    )

    assert result["status"] == "REVIEW_INDEX_COMPLETE_HUMAN_VERDICT_PENDING"
    assert result["quad_video"] == successor_and_video_ref
    assert result["automatic_pass"] is False
    assert result["pixels_produced"] == 0
    assert not (run_root / "MISSING_S7_SUCCESSOR.json").exists()
    assert not (run_root / "MISSING_S8_SUCCESSOR.json").exists()

    review = json.loads(
        (run_root / "counterexample_admitted" / "HUMAN_REVIEW_STATUS_TEMPLATE.json").read_text()
    )
    assert review["status"] == "PENDING_HUMAN_REVIEW_NO_AUTOMATIC_PASS"
    assert all(row["review_status"] == "PENDING_HUMAN_REVIEW" for row in review["rows"])


def test_independent_sampling_ties_are_deterministic() -> None:
    rows = _diagnostic_rows(24)
    first = visual.build_selection(rows)
    second = visual.build_selection(rows)
    assert first == second
    assert first["low_iou"] == list(range(7))
    assert first["wrist_wearable"] == list(range(12))
    assert first["contact_boundary"] == list(range(12))
    assert first["thumb_embedding"] == list(range(12))
