from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import jsonschema
import numpy as np
import pytest

from pipeline.auditor.mask_auditor_a1 import (
    AuditInputError,
    audit_candidates,
    compute_tolerance_diagnostics,
    write_new_report,
)


CREATED_AT = "2026-08-27T00:00:00Z"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> tuple[Path, str]:
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return path, _sha(payload)


def _ref(path: Path, producer: str, schema_version: str) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": _sha(payload),
        "bytes": len(payload),
        "producer": producer,
        "schema_version": schema_version,
    }


def _write_png(path: Path, value: np.ndarray) -> dict[str, Any]:
    ok, encoded = cv2.imencode(".png", value)
    assert ok
    path.write_bytes(encoded.tobytes())
    return _ref(path, "synthetic_fixture", "binary-mask-v1")


def _fixture(tmp_path: Path, state: str = "SYNTHETIC_TEST_ONLY") -> dict[str, Any]:
    h = np.zeros((12, 12), np.uint8)
    o = np.zeros_like(h)
    u = np.zeros_like(h)
    h[2:6, 2:6] = 255
    o[2:6, 6:9] = 255
    u[6, 2:9] = 255
    b = np.where((h | o | u) == 0, 255, 0).astype(np.uint8)
    good = (h | u).astype(np.uint8)
    bad = good.copy()
    bad[2:4, 2:4] = 0
    bad[2:5, 6] = 255
    bad[9, 9:11] = 255

    frames: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    good_rows: list[dict[str, Any]] = []
    bad_rows: list[dict[str, Any]] = []
    splits = (
        ("development", "same_session_blind", "cross_session_blind")
        if state == "SYNTHETIC_TEST_ONLY"
        else tuple(
            ["development"] * 15 + ["same_session_blind"] * 15 + ["cross_session_blind"] * 15
        )
    )
    for index, split in enumerate(splits):
        source = np.tile(np.arange(12, dtype=np.uint8), (12, 1))
        source_path = tmp_path / f"source_{index}.png"
        source_ref = _write_png(source_path, source)
        source_ref["producer"] = "source_resolver"
        source_ref["schema_version"] = "raw-frame-v1"
        sample_id = f"sample-{index}"
        frames.append({
            "sample_id": sample_id,
            "session_id": "session-a" if split != "cross_session_blind" else "session-blind",
            "frame_index": index,
            "split": split,
            "source_image": source_ref,
        })
        label_rows.append({
            "sample_id": sample_id,
            "source_image_sha256": source_ref["sha256"],
            "h_mask": _write_png(tmp_path / f"h_{index}.png", h),
            "o_mask": _write_png(tmp_path / f"o_{index}.png", o),
            "u_mask": _write_png(tmp_path / f"u_{index}.png", u),
            "b_mask": _write_png(tmp_path / f"b_{index}.png", b),
        })
        good_rows.append({
            "sample_id": sample_id,
            "source_image_sha256": source_ref["sha256"],
            "predicted_human_mask": _write_png(tmp_path / f"good_{index}.png", good),
        })
        bad_rows.append({
            "sample_id": sample_id,
            "source_image_sha256": source_ref["sha256"],
            "predicted_human_mask": _write_png(tmp_path / f"bad_{index}.png", bad),
        })

    benchmark = {
        "schema_version": "mask-benchmark-v1",
        "artifact_state": state,
        "producer": "mask_benchmark_curator",
        "benchmark_id": "synthetic-a1",
        "uncertainty_policy": {
            "labels": ["H", "O", "U", "B"],
            "u_excluded_from_metrics": True,
            "manual_labels_required": True,
        },
        "frames": frames,
        "temporal_pairs": [{
            "pair_id": "pair-0-1",
            "from_sample_id": "sample-0",
            "to_sample_id": "sample-1",
        }],
        "created_at": CREATED_AT,
    }
    benchmark_path, benchmark_sha = _write_json(tmp_path / "benchmark.json", benchmark)
    benchmark_ref = _ref(benchmark_path, "mask_benchmark_curator", "mask-benchmark-v1")
    approval_path, _ = _write_json(tmp_path / "human_approval.json", {"decision": "APPROVED"})
    labels = {
        "schema_version": "mask-houb-labels-v1",
        "artifact_state": state,
        "producer": "human_label_curator",
        "benchmark_manifest_ref": benchmark_ref,
        "completed_by_human": state == "FROZEN",
        "frozen": state == "FROZEN",
        "human_approval_ref": _ref(approval_path, "human_reviewer", "human-review-v1"),
        "frames": label_rows,
        "created_at": CREATED_AT,
    }
    labels_path, labels_sha = _write_json(tmp_path / "labels.json", labels)

    def candidate(candidate_id: str, rows: list[dict[str, Any]]) -> tuple[Path, str]:
        return _write_json(tmp_path / f"{candidate_id}.json", {
            "schema_version": "mask-candidate-eval-v1",
            "artifact_state": state,
            "producer": "mask_candidate_manifest_adapter",
            "candidate_id": candidate_id,
            "producer_version": candidate_id,
            "benchmark_manifest_ref": benchmark_ref,
            "frames": rows,
            "created_at": CREATED_AT,
        })

    return {
        "benchmark": (benchmark_path, benchmark_sha),
        "labels": (labels_path, labels_sha),
        "good": candidate("producer_good", good_rows),
        "bad": candidate("producer_bad", bad_rows),
        "good_rows": good_rows,
        "benchmark_ref": benchmark_ref,
    }


def test_formal_run_without_human_labels_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, state="FROZEN")
    report = audit_candidates(
        benchmark_path=fixture["benchmark"][0],
        benchmark_sha256=fixture["benchmark"][1],
        candidates=[fixture["good"]],
        mode="FORMAL",
    )
    assert report["status"] == "AWAITING_HUMAN_LABELS"
    assert report["formal_comparison_authorized"] is False
    assert report["candidate_results"] == []
    assert report["ranking"] == []
    assert report["upstream_mutation_permitted"] is False


def test_synthetic_metrics_exclude_u_and_use_one_fixed_ranking(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report = audit_candidates(
        benchmark_path=fixture["benchmark"][0],
        benchmark_sha256=fixture["benchmark"][1],
        candidates=[fixture["bad"], fixture["good"]],
        human_labels_path=fixture["labels"][0],
        human_labels_sha256=fixture["labels"][1],
        mode="SYNTHETIC_TEST_ONLY",
    )
    assert report["status"] == "EVALUATED"
    assert report["ranking"] == ["producer_good", "producer_bad"]
    good = next(row for row in report["candidate_results"] if row["candidate_id"] == "producer_good")
    assert good["aggregate"]["human_miss"] == 0
    assert good["aggregate"]["object_corruption"] == 0
    assert good["aggregate"]["background_overmask"] == 0
    assert good["temporal"]["temporal_flip"] == 0
    assert report["auditor"]["cross_auditor_comparison_allowed"] is False
    assert len(report["auditor"]["implementation_sha256"]) == 64


def test_incomplete_candidate_is_forensic_not_comparable_without_intersection(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    candidate = json.loads(fixture["good"][0].read_text())
    candidate["frames"] = candidate["frames"][:-1]
    short_path, short_sha = _write_json(tmp_path / "short.json", candidate)
    report = audit_candidates(
        benchmark_path=fixture["benchmark"][0],
        benchmark_sha256=fixture["benchmark"][1],
        candidates=[(short_path, short_sha), fixture["bad"]],
        mode="SYNTHETIC_TEST_ONLY",
    )
    assert report["status"] == "FORENSIC_NOT_COMPARABLE"
    assert report["candidate_results"] == []
    assert report["ranking"] == []
    assert report["formal_comparison_authorized"] is False


def test_candidate_source_sha_mismatch_refuses_ranking(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    candidate = json.loads(fixture["good"][0].read_text())
    candidate["frames"][0]["source_image_sha256"] = "0" * 64
    path, digest = _write_json(tmp_path / "source_mismatch.json", candidate)
    report = audit_candidates(
        benchmark_path=fixture["benchmark"][0],
        benchmark_sha256=fixture["benchmark"][1],
        candidates=[(path, digest)],
        mode="SYNTHETIC_TEST_ONLY",
    )
    assert report["status"] == "FORENSIC_NOT_COMPARABLE"
    assert report["coverage"]["candidates"][0]["source_sha_match"] is False


def test_overlapping_houb_labels_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    labels = json.loads(fixture["labels"][0].read_text())
    labels["frames"][0]["o_mask"] = labels["frames"][0]["h_mask"]
    path, digest = _write_json(tmp_path / "overlap_labels.json", labels)
    with pytest.raises(AuditInputError, match="mutually exclusive and exhaustive"):
        audit_candidates(
            benchmark_path=fixture["benchmark"][0],
            benchmark_sha256=fixture["benchmark"][1],
            candidates=[fixture["good"]],
            human_labels_path=path,
            human_labels_sha256=digest,
            mode="SYNTHETIC_TEST_ONLY",
        )


def test_manifest_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(AuditInputError, match="manifest SHA mismatch"):
        audit_candidates(
            benchmark_path=fixture["benchmark"][0],
            benchmark_sha256="0" * 64,
            candidates=[fixture["good"]],
            mode="SYNTHETIC_TEST_ONLY",
        )


def test_symlink_manifest_and_mask_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    candidate_link = tmp_path / "candidate_link.json"
    candidate_link.symlink_to(fixture["good"][0])
    with pytest.raises(AuditInputError, match="not an ordinary file"):
        audit_candidates(
            benchmark_path=fixture["benchmark"][0],
            benchmark_sha256=fixture["benchmark"][1],
            candidates=[(candidate_link, fixture["good"][1])],
            mode="SYNTHETIC_TEST_ONLY",
        )

    candidate = json.loads(fixture["good"][0].read_text())
    original = Path(candidate["frames"][0]["predicted_human_mask"]["path"])
    linked = tmp_path / "linked_mask.png"
    linked.symlink_to(original)
    candidate["frames"][0]["predicted_human_mask"]["path"] = str(linked)
    linked_candidate, linked_sha = _write_json(tmp_path / "linked_candidate.json", candidate)
    with pytest.raises(AuditInputError, match="not an ordinary file"):
        audit_candidates(
            benchmark_path=fixture["benchmark"][0],
            benchmark_sha256=fixture["benchmark"][1],
            candidates=[(linked_candidate, linked_sha)],
            human_labels_path=fixture["labels"][0],
            human_labels_sha256=fixture["labels"][1],
            mode="SYNTHETIC_TEST_ONLY",
        )


def test_report_is_schema_valid_and_cannot_overwrite(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, state="FROZEN")
    report = audit_candidates(
        benchmark_path=fixture["benchmark"][0],
        benchmark_sha256=fixture["benchmark"][1],
        candidates=[fixture["good"]],
        mode="FORMAL",
    )
    schema = json.loads(
        (Path(__file__).parents[1] / "contracts" / "mask_auditor_a1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(report)
    output = tmp_path / "qa.json"
    write_new_report(output, report)
    before = output.read_bytes()
    with pytest.raises(AuditInputError, match="refusing to overwrite"):
        write_new_report(output, report)
    assert output.read_bytes() == before


def _tolerance_labels(shape: tuple[int, int] = (24, 24)) -> dict[str, np.ndarray]:
    h = np.zeros(shape, dtype=bool)
    o = np.zeros(shape, dtype=bool)
    u = np.zeros(shape, dtype=bool)
    h[4:16, 4:12] = True
    o[7:14, 12:17] = True
    u[4:16, 17] = True
    b = ~(h | o | u)
    return {"h": h, "o": o, "u": u, "b": b}


def test_b2_tolerance_diagnostics_are_separate_and_perfect() -> None:
    labels = _tolerance_labels()
    metrics = compute_tolerance_diagnostics(
        labels, labels["h"], wrist_width_px=8.0, core_erosion_ratio=0.25
    )
    assert metrics["decision_use"] == "DIAGNOSTIC_ONLY_NOT_PASS_FAIL"
    assert metrics["core_erosion_px"] == 2
    assert metrics["human_core_miss"]["value"] == 0.0
    assert metrics["background_core_overmask"]["value"] == 0.0
    for tolerance in ("1", "3", "5"):
        assert metrics["boundary_iou_excluding_u"][tolerance]["value"] == 1.0


def test_b2_boundary_tolerances_and_u_statistics_are_explicit() -> None:
    labels = _tolerance_labels()
    shifted = np.zeros_like(labels["h"])
    shifted[4:16, 5:13] = True
    metrics = compute_tolerance_diagnostics(
        labels, shifted, wrist_width_px=8.0, core_erosion_ratio=0.0
    )
    values = [metrics["boundary_iou_excluding_u"][str(px)]["value"] for px in (1, 3, 5)]
    assert all(value is not None for value in values)
    assert values[0] < values[1] < values[2]
    assert metrics["u_statistics"]["pixels"] == int(np.count_nonzero(labels["u"]))
    assert 0.0 < metrics["u_statistics"]["ratio"] < 1.0


def test_b2_zero_denominators_are_null_not_fabricated() -> None:
    shape = (12, 12)
    labels = {
        "h": np.zeros(shape, dtype=bool),
        "o": np.zeros(shape, dtype=bool),
        "u": np.zeros(shape, dtype=bool),
        "b": np.ones(shape, dtype=bool),
    }
    predicted = np.zeros(shape, dtype=bool)
    metrics = compute_tolerance_diagnostics(
        labels, predicted, wrist_width_px=20.0, core_erosion_ratio=1.0
    )
    assert metrics["human_core_miss"] == {
        "value": None, "defined": False, "numerator": 0, "denominator": 0
    }
    assert metrics["background_core_overmask"]["defined"] is False
    for tolerance in ("1", "3", "5"):
        assert metrics["boundary_iou_excluding_u"][tolerance]["defined"] is False


@pytest.mark.parametrize(
    ("wrist_width_px", "core_erosion_ratio"),
    [(0.0, 0.2), (float("nan"), 0.2), (8.0, -0.1), (8.0, float("inf"))],
)
def test_b2_invalid_scale_inputs_fail_closed(
    wrist_width_px: float, core_erosion_ratio: float
) -> None:
    labels = _tolerance_labels()
    with pytest.raises(AuditInputError):
        compute_tolerance_diagnostics(
            labels,
            labels["h"],
            wrist_width_px=wrist_width_px,
            core_erosion_ratio=core_erosion_ratio,
        )
