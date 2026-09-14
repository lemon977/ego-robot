import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "task27_temporal_baseline", PROJECT / "tools/run_004_full_session_temporal_baseline_t1.py"
)
assert SPEC is not None and SPEC.loader is not None
baseline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = baseline
SPEC.loader.exec_module(baseline)


def _row(left: bool, right: bool, *, left_iou=None, right_iou=None):
    return {
        "frame_complete": left and right,
        "sides": {
            "left": {
                "pass": left,
                "area_relative_change_from_previous": None,
                "raw_iou_from_previous": left_iou,
                "raw_iou_to_next": left_iou,
            },
            "right": {
                "pass": right,
                "area_relative_change_from_previous": None,
                "raw_iou_from_previous": right_iou,
                "raw_iou_to_next": right_iou,
            },
        },
    }


def test_exact_old_v3_identities_and_thresholds_are_constants():
    assert baseline.MODULE_SHA == "694a9bc3b24bae3e36eda8648befc22db690ac4c515c8251985c2d57e81f2e5f"
    assert baseline.OLD_RUNNER_SHA == "68ac4a1ceff7a7db7f2ec13b436105f59a6a8613ea5b971093f8229ac83bbd84"
    assert baseline.CONFIG_SHA == "6b64a62cb5b439d64ba09187a6801b82762b9766bdd8949aacaa063a87a73628"
    assert baseline.TASK26_RUN_REL == Path("_run/lr_distributed_side_candidate_t1_20260828_v4")
    assert baseline.TAXONOMY == (
        "AUTHORITY_POINT_OUT_OF_IMAGE",
        "SINGLE_JOINT_VETO",
        "NO_INSTANCE_SUPPORTS_SIDE",
        "AUTHORITY_SIDE_ALIAS",
        "WEAK_SUPPORT",
        "MISSING_OR_UNVERIFIED_IDENTITY",
    )


def test_temporal_summary_counts_flips_longest_failure_and_isolated_drop():
    rows = [
        _row(True, True),
        _row(False, True),
        _row(True, False),
        _row(False, False),
        _row(False, True),
    ]
    summary = baseline.temporal_summary(rows, "left")
    assert summary["pass_count"] == 2
    assert summary["status_flip_count"] == 3
    assert summary["longest_failure_run"] == 2
    assert summary["isolated_drop_count"] == 1


def test_area_change_uses_current_area_denominator_and_flags_zero_current():
    assert baseline.relative_area_change(20, 10) == (0.5, False)
    assert baseline.relative_area_change(10, 20) == (1.0, False)
    assert baseline.relative_area_change(0, 20) == (None, True)
    assert baseline.relative_area_change(0, None) == (None, True)


def test_adjacent_iou_includes_formal_empty_masks():
    empty = baseline.np.zeros((4, 4), dtype=bool)
    one = empty.copy()
    one[0, 0] = True
    assert baseline.iou(empty, empty) == 1.0
    assert baseline.iou(one, empty) == 0.0


def test_failed_side_formal_mask_is_always_empty_and_accept_is_exact_raw_mask():
    raw = baseline.np.zeros((960, 1280), dtype=bool)
    raw[10:20, 30:40] = True
    accepted = baseline.formal_mask_for_decision(SimpleNamespace(status="ACCEPT", mask=raw))
    rejected = baseline.formal_mask_for_decision(SimpleNamespace(status="HOLD", mask=raw))
    assert baseline.np.array_equal(accepted, raw)
    assert not rejected.any()
    with pytest.raises(baseline.BaselineError, match="missing exact raw mask"):
        baseline.formal_mask_for_decision(SimpleNamespace(status="ACCEPT", mask=None))


@pytest.mark.parametrize(
    ("state", "pool", "audits", "expected"),
    [
        ("OUTSIDE_IMAGE", [], [], "AUTHORITY_POINT_OUT_OF_IMAGE"),
        ("MISSING", [], [], "MISSING_OR_UNVERIFIED_IDENTITY"),
        ("AVAILABLE", [], [], "NO_INSTANCE_SUPPORTS_SIDE"),
        (
            "AVAILABLE",
            [object()],
            [SimpleNamespace(rejection_reasons=("EGO_WRIST_NOT_SUPPORTED",), joint_support_ratio=0.20)],
            "SINGLE_JOINT_VETO",
        ),
        ("AVAILABLE", [object()], [], "WEAK_SUPPORT"),
    ],
)
def test_six_class_taxonomy_is_posthoc_and_total_for_failures(state, pool, audits, expected):
    decision = SimpleNamespace(status="HOLD", failure_reason="X", candidate_audit=audits)
    assert baseline.classify(decision, {"state": state}, pool) == expected


def test_admission_fails_before_run_creation_below_disk_gate(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(baseline, "disk_free", lambda _path: baseline.ADMISSION_REQUIRED_FREE - 1)
    with pytest.raises(baseline.BaselineError, match="8 GiB reserve"):
        baseline.admit(tmp_path, "004_full_session_temporal_baseline_test", tmp_path, tmp_path)
    assert not (tmp_path / "_run").exists()


def test_run_growth_and_free_space_gates_have_preregistered_margin():
    assert baseline.MAX_RUN_BYTES == int(2.5 * 1024**3)
    assert baseline.PREDICTED_RETAINED_BYTES + baseline.SAFETY_MARGIN_BYTES == baseline.MAX_RUN_BYTES
    assert baseline.ABSOLUTE_RESERVE_BYTES == 8 * 1024**3
    assert baseline.ADMISSION_REQUIRED_FREE == baseline.ABSOLUTE_RESERVE_BYTES + baseline.MAX_RUN_BYTES


def test_task26_preemption_requires_exact_qa_and_task_digest_binding(tmp_path: Path):
    audits = tmp_path / "audits"
    task = tmp_path / "task"
    run = tmp_path / baseline.TASK26_RUN_REL
    audits.mkdir()
    task.mkdir()
    run.mkdir(parents=True)
    qa_path = audits / "INDEPENDENT_QA_LR_DISTRIBUTED_SIDE_EVIDENCE_T1_V1.json"
    task_path = task / "26_APRIME_DISTRIBUTED_SIDE_EVIDENCE_T1.md"
    task_path.write_text("task26 exact bytes")
    qa_path.write_text(
        json.dumps(
            {
                "status": "PASS_CPU_ADMISSION_EXACT",
                "p0_findings": 0,
                "frozen_refs": {"task26": baseline.identity(task_path)},
            }
        )
    )
    admission = {
        "status": "PASS_EXACT_FREEZE_O_EXCL_GPU_READY",
        "frozen_refs": {
            "qa": baseline.identity(qa_path),
            "task26": baseline.identity(task_path),
        },
    }
    (run / "GPU_ADMISSION.json").write_text(json.dumps(admission))
    ready = baseline.priority_task26_ready(tmp_path)
    assert ready is not None
    assert ready["independent_qa"] == baseline.identity(qa_path)
    assert ready["task26"] == baseline.identity(task_path)

    terminal = tmp_path / baseline.TASK26_TERMINAL_REL
    terminal.parent.mkdir()
    terminal.write_text(
        json.dumps(
            {
                "status": "COMPLETED_UNREVIEWED",
                "strict_model_build": {"checkpoint_loaded": True},
                "gpu_metrics": {"peak_allocated_bytes": 1},
                "frame_records": [{} for _ in range(39)],
                "sessions": [
                    {"session_id": "grap_a_cap_004", "frame_count": 15, "prompt_execution": {"seconds": 1}},
                    {"session_id": "grap_a_cap_002", "frame_count": 8, "prompt_execution": {"seconds": 1}},
                    {"session_id": "grap_a_cap_005", "frame_count": 8, "prompt_execution": {"seconds": 1}},
                    {"session_id": "grap_a_cap_012", "frame_count": 8, "prompt_execution": {"seconds": 1}},
                ],
            }
        )
    )
    release = {
        "status": "GPU_RELEASED",
        "gpu_processes_remaining": 0,
        "partial_artifacts_preserved": True,
        "GPU_ADMISSION": baseline.identity(run / "GPU_ADMISSION.json"),
        "terminal_manifest": baseline.identity(terminal),
        "task26_qa": baseline.identity(qa_path),
    }
    (run / "GPU_RELEASE.json").write_text(json.dumps(release))
    assert baseline.priority_task26_ready(tmp_path) is None


def test_task26_preemption_rejects_filename_only_admission(tmp_path: Path):
    audits = tmp_path / "audits"
    task = tmp_path / "task"
    run = tmp_path / baseline.TASK26_RUN_REL
    audits.mkdir()
    task.mkdir()
    run.mkdir(parents=True)
    qa_path = audits / "INDEPENDENT_QA_LR_DISTRIBUTED_SIDE_EVIDENCE_T1_V1.json"
    task_path = task / "26_APRIME_DISTRIBUTED_SIDE_EVIDENCE_T1.md"
    task_path.write_text("task26 exact bytes")
    qa_path.write_text(
        json.dumps(
            {
                "status": "PASS_CPU_ADMISSION_EXACT",
                "p0_findings": 0,
                "frozen_refs": {"task26": baseline.identity(task_path)},
            }
        )
    )
    (run / "GPU_ADMISSION.json").write_text(json.dumps({"status": "PASS_FAKE"}))
    with pytest.raises(baseline.BaselineError, match="does not bind canonical QA"):
        baseline.priority_task26_ready(tmp_path)


def test_task27_release_gate_distinguishes_never_admitted_from_released(tmp_path: Path):
    with pytest.raises(baseline.BaselineError, match="admission/release handoff is missing"):
        baseline.task26_handoff(tmp_path, require_release=True)
    assert baseline.task26_handoff(tmp_path, require_release=False) == {
        "state": "TASK26_NOT_ADMITTED"
    }


def test_live_canonical_task26_v4_real_handoff_exact_bytes():
    terminal = PROJECT / baseline.TASK26_TERMINAL_REL
    release = PROJECT / baseline.TASK26_RELEASE_REL
    admission = PROJECT / baseline.TASK26_ADMISSION_REL
    assert baseline.identity(terminal)["sha256"] == "7ecee41b95716c81997830923a616b91f0bbc6a626aaa3f4786598907408c040"
    assert baseline.identity(release)["sha256"] == "bf920e990378dde234a3f1548bc558554399082a9ae650c73e427dbeb8ac13db"
    assert baseline.identity(admission)["sha256"] == "f2006e599a9ac5bd7071742b1969555f3a49e0a6d458f22f38263a615f10a42c"
    handoff = baseline.task26_handoff(PROJECT, require_release=True)
    assert handoff["state"] == "TASK26_CANONICALLY_COMPLETED_AND_GPU_RELEASED"


def test_identity_rejects_intermediate_symlink(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    payload = real / "evidence.json"
    payload.write_text("{}")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(baseline.BaselineError, match="symlink component"):
        baseline.identity(link / "evidence.json")


def test_terminal_manifest_is_written_only_after_decode_and_artifact_validation():
    source = (PROJECT / "tools/run_004_full_session_temporal_baseline_t1.py").read_text()
    terminal_write = source.index("exclusive_json(manifest_path, result)")
    mask_decode = source.index("mask_png_full_decode_count")
    artifact_write = source.index("exclusive_json(\n            artifact_path")
    assert terminal_write > mask_decode
    assert terminal_write > artifact_write
    assert 'failure_path = component / "FAILURE_MANIFEST.json"' in source
