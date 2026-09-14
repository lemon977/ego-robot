"""Independent, read-only H/O/U/B benchmark auditor for MASK candidates.

This module never imports or invokes a MASK producer.  It reads immutable,
SHA-bound manifests and pixels, computes fixed auditor-a1 metrics, and returns
a QA document.  The only filesystem write exposed here creates a new QA report
with O_EXCL; upstream artifacts and human labels are never changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import jsonschema
import numpy as np


AUDITOR_ID = "auditor_a1"
AUDITOR_FORMULA_ID = "houb-a1-lexicographic-v1"
FORMAL_MODE = "FORMAL"
SYNTHETIC_MODE = "SYNTHETIC_TEST_ONLY"
IMPLEMENTATION_PATH = Path(__file__)
CONTRACT_ROOT = IMPLEMENTATION_PATH.parents[2] / "contracts"
AUDITOR_SCHEMA_NAMES = (
    "mask_benchmark_v1.schema.json",
    "mask_houb_labels_v1.schema.json",
    "mask_candidate_eval_v1.schema.json",
    "mask_auditor_a1.schema.json",
)

FORMULAS = {
    "human_miss": "sum(H and not P_H) / sum(H); U is excluded",
    "object_corruption": "sum(O and P_H) / sum(O); U is excluded",
    "background_overmask": "sum(B and P_H) / sum(B); U is excluded",
    "boundary_quality": (
        "F1 of one-pixel 4-neighbour H/P_H boundaries with one-pixel matching; "
        "U and its one-pixel 4-neighbour band are excluded"
    ),
    "temporal_flip": (
        "symmetric P_H XOR after deterministic Farneback alignment, divided by "
        "the aligned P_H union; out-of-frame pixels and U are excluded in both directions"
    ),
    "aggregation": "micro-aggregate pixel counts across the complete frozen benchmark",
    "ranking": (
        "ascending lexicographic: max(human_miss,object_corruption), "
        "human_miss+object_corruption, background_overmask, "
        "1-boundary_f1, temporal_flip; candidate_id is the deterministic exact-tie break"
    ),
}

# B2 diagnostics are deliberately separate from FORMULAS and the frozen
# auditor-a1 ranking key.  They may quantify annotation tolerance, but they do
# not change any of the five existing metrics or authorize a pass/fail result.
TOLERANCE_DIAGNOSTIC_FORMULAS = {
    "core_radius": "floor(core_erosion_ratio * wrist_width_px + 0.5) pixels",
    "human_core_miss": "sum(erode(H,r) and not P_H) / sum(erode(H,r))",
    "background_core_overmask": "sum(erode(B,r) and P_H) / sum(erode(B,r))",
    "boundary_iou": (
        "Cheng et al. Boundary IoU: IoU of the inner H and P_H boundary regions "
        "at distance d; pixels labelled U are removed; d is separately 1, 3 and 5 px"
    ),
    "zero_denominator": "value=null, defined=false, numerator=0, denominator=0",
    "decision_use": "diagnostic only; excluded from pass/fail and auditor-a1 ranking",
}


class AuditInputError(RuntimeError):
    """An input cannot be safely or unambiguously audited."""


def _normal_absolute(path: Path) -> Path:
    if not path.is_absolute():
        raise AuditInputError(f"path must be absolute (no relative fallback): {path}")
    if os.path.normpath(str(path)) != str(path):
        raise AuditInputError(f"path must be lexically normalized: {path}")
    for parent in path.parents:
        if parent == Path("/"):
            break
        try:
            if parent.is_symlink():
                raise AuditInputError(f"symlink path component is forbidden: {parent}")
        except OSError as exc:
            raise AuditInputError(f"cannot inspect path component: {parent}") from exc
    return path


def _read_regular_nofollow(path: Path) -> bytes:
    """Read one ordinary file without following a final symlink."""

    path = _normal_absolute(path)
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise AuditInputError(f"required file is missing: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise AuditInputError(f"required path is not an ordinary file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AuditInputError(f"secure open failed: {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise AuditInputError(f"opened input is not an ordinary file: {path}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise AuditInputError(f"input changed during secure open: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
    finally:
        os.close(descriptor)
    if not payload:
        raise AuditInputError(f"empty input is forbidden: {path}")
    return payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_bound_json(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise AuditInputError("expected manifest SHA256 must be 64 lowercase hex characters")
    payload = _read_regular_nofollow(path)
    actual = _sha256(payload)
    if actual != expected_sha256:
        raise AuditInputError(f"manifest SHA mismatch: {path}")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditInputError(f"invalid JSON manifest: {path}") from exc
    if not isinstance(value, dict):
        raise AuditInputError(f"manifest must be a JSON object: {path}")
    return value, {
        "path": str(path),
        "sha256": actual,
        "bytes": len(payload),
        "producer": str(value.get("producer", "unknown")),
        "schema_version": str(value.get("schema_version", "unknown")),
    }


def _load_schema(name: str) -> dict[str, Any]:
    payload = _read_regular_nofollow(CONTRACT_ROOT / name)
    return json.loads(payload)


def _auditor_contract_refs() -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for name in AUDITOR_SCHEMA_NAMES:
        path = CONTRACT_ROOT / name
        payload = _read_regular_nofollow(path)
        refs.append({
            "path": str(path),
            "sha256": _sha256(payload),
            "bytes": len(payload),
            "producer": "contract_authority",
            "schema_version": name.removesuffix(".schema.json"),
        })
    return refs


def _validate(document: dict[str, Any], schema_name: str) -> None:
    try:
        jsonschema.Draft202012Validator(
            _load_schema(schema_name), format_checker=jsonschema.FormatChecker()
        ).validate(document)
    except jsonschema.ValidationError as exc:
        raise AuditInputError(f"{schema_name} validation failed: {exc.message}") from exc


def _read_ref(ref: dict[str, Any]) -> bytes:
    payload = _read_regular_nofollow(Path(ref["path"]))
    if len(payload) != int(ref["bytes"]):
        raise AuditInputError(f"referenced byte count mismatch: {ref['path']}")
    if _sha256(payload) != ref["sha256"]:
        raise AuditInputError(f"referenced SHA mismatch: {ref['path']}")
    return payload


def _decode_mask(ref: dict[str, Any]) -> np.ndarray:
    payload = _read_ref(ref)
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise AuditInputError(f"cannot decode mask PNG: {ref['path']}")
    values = np.unique(image)
    if np.any((values != 0) & (values != 255)):
        raise AuditInputError(f"mask must be lossless binary 0/255: {ref['path']}")
    return image == 255


def _decode_gray(ref: dict[str, Any]) -> np.ndarray:
    payload = _read_ref(ref)
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2:
        raise AuditInputError(f"cannot decode source image: {ref['path']}")
    return image


def _implementation_sha256() -> str:
    return _sha256(_read_regular_nofollow(IMPLEMENTATION_PATH))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coverage_reason(
    benchmark: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, list[str]]:
    expected = benchmark["frames"]
    observed = candidate["frames"]
    reasons: list[str] = []
    expected_ids = [row["sample_id"] for row in expected]
    observed_ids = [row["sample_id"] for row in observed]
    if observed_ids != expected_ids:
        reasons.append("candidate sample IDs/order do not exactly equal the complete benchmark")
    by_id = {row["sample_id"]: row for row in expected}
    for row in observed:
        sample_id = row["sample_id"]
        reference = by_id.get(sample_id)
        if reference is not None and row["source_image_sha256"] != reference["source_image"]["sha256"]:
            reasons.append(f"source SHA mismatch for {sample_id}")
    return not reasons, reasons


def _validate_benchmark_identity(benchmark: dict[str, Any]) -> None:
    rows = benchmark["frames"]
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise AuditInputError("benchmark sample_id values must be unique")
    sample_by_id = {row["sample_id"]: row for row in rows}
    pair_ids: set[str] = set()
    for pair in benchmark["temporal_pairs"]:
        if pair["pair_id"] in pair_ids:
            raise AuditInputError(f"duplicate temporal pair_id: {pair['pair_id']}")
        pair_ids.add(pair["pair_id"])
        first = sample_by_id.get(pair["from_sample_id"])
        second = sample_by_id.get(pair["to_sample_id"])
        if first is None or second is None:
            raise AuditInputError(f"temporal pair references an unknown sample: {pair['pair_id']}")
        if first["sample_id"] == second["sample_id"]:
            raise AuditInputError(f"temporal pair endpoints must differ: {pair['pair_id']}")
        if first["session_id"] != second["session_id"]:
            raise AuditInputError(f"temporal pair cannot cross sessions: {pair['pair_id']}")


def _labels_coverage_reason(
    benchmark: dict[str, Any], labels: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    expected = benchmark["frames"]
    observed = labels["frames"]
    expected_ids = [row["sample_id"] for row in expected]
    observed_ids = [row["sample_id"] for row in observed]
    if observed_ids != expected_ids:
        reasons.append("human labels do not exactly cover the complete benchmark in frozen order")
    by_id = {row["sample_id"]: row for row in expected}
    for row in observed:
        reference = by_id.get(row["sample_id"])
        if reference is not None and row["source_image_sha256"] != reference["source_image"]["sha256"]:
            reasons.append(f"human-label source SHA mismatch for {row['sample_id']}")
    return not reasons, reasons


def _strict_ref_matches(ref: dict[str, Any], target: dict[str, Any]) -> bool:
    return (
        ref["path"] == target["path"]
        and ref["sha256"] == target["sha256"]
        and int(ref["bytes"]) == int(target["bytes"])
    )


def _boundary(mask: np.ndarray) -> np.ndarray:
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(
        mask.astype(np.uint8), cross, borderType=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(bool)
    return mask & ~eroded


def _erode_by_pixel_radius(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Erode a binary mask by an Euclidean pixel radius, including image edges.

    Padding makes the out-of-frame region explicit background.  This avoids
    OpenCV's otherwise surprising behaviour for all-foreground fixtures and
    makes an over-large radius produce an empty, representable core.
    """

    if mask.ndim != 2:
        raise AuditInputError("tolerance metrics require two-dimensional masks")
    if radius_px < 0:
        raise AuditInputError("erosion radius cannot be negative")
    binary = mask.astype(bool)
    if radius_px == 0:
        return binary.copy()
    padded = cv2.copyMakeBorder(
        binary.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    return binary & (distance > float(radius_px))


def _nullable_ratio(numerator: int, denominator: int) -> dict[str, Any]:
    """Represent a ratio without inventing a value for an empty denominator."""

    return {
        "value": numerator / denominator if denominator else None,
        "defined": denominator > 0,
        "numerator": numerator,
        "denominator": denominator,
    }


def compute_tolerance_diagnostics(
    labels: dict[str, np.ndarray],
    predicted_h: np.ndarray,
    *,
    wrist_width_px: float,
    core_erosion_ratio: float,
) -> dict[str, Any]:
    """Compute B2 annotation-tolerance diagnostics for one frame.

    ``core_erosion_ratio`` is a required, external input because B1 has not yet
    established a noise floor from which a project value may be frozen.  This
    function therefore supplies measurements only: it never selects a
    threshold and its output is not consumed by the auditor-a1 ranking.
    """

    required = {"h", "o", "u", "b"}
    if set(labels) != required:
        raise AuditInputError("tolerance labels must contain exactly H/O/U/B")
    h, o, u, b = (np.asarray(labels[key], dtype=bool) for key in ("h", "o", "u", "b"))
    predicted_h = np.asarray(predicted_h, dtype=bool)
    shapes = {array.shape for array in (h, o, u, b, predicted_h)}
    if len(shapes) != 1 or h.ndim != 2:
        raise AuditInputError("H/O/U/B/candidate mask dimensions disagree")
    stacked = h.astype(np.uint8) + o + u + b
    if not np.all(stacked == 1):
        raise AuditInputError("H/O/U/B labels must be mutually exclusive and exhaustive")
    if not math.isfinite(wrist_width_px) or wrist_width_px <= 0:
        raise AuditInputError("wrist_width_px must be finite and positive")
    if not math.isfinite(core_erosion_ratio) or core_erosion_ratio < 0:
        raise AuditInputError("core_erosion_ratio must be finite and non-negative")

    core_radius_px = int(math.floor(core_erosion_ratio * wrist_width_px + 0.5))
    h_core = _erode_by_pixel_radius(h, core_radius_px)
    b_core = _erode_by_pixel_radius(b, core_radius_px)
    human_core = _nullable_ratio(
        int(np.count_nonzero(h_core & ~predicted_h)), int(np.count_nonzero(h_core))
    )
    background_core = _nullable_ratio(
        int(np.count_nonzero(b_core & predicted_h)), int(np.count_nonzero(b_core))
    )

    boundary_iou: dict[str, dict[str, Any]] = {}
    for tolerance_px in (1, 3, 5):
        true_region = (h & ~_erode_by_pixel_radius(h, tolerance_px)) & ~u
        pred_region = (predicted_h & ~_erode_by_pixel_radius(predicted_h, tolerance_px)) & ~u
        intersection = int(np.count_nonzero(true_region & pred_region))
        union = int(np.count_nonzero(true_region | pred_region))
        boundary_iou[str(tolerance_px)] = _nullable_ratio(intersection, union)

    return {
        "metric_set": "annotation-tolerance-b2-v1",
        "decision_use": "DIAGNOSTIC_ONLY_NOT_PASS_FAIL",
        "formulas": TOLERANCE_DIAGNOSTIC_FORMULAS,
        "wrist_width_px": float(wrist_width_px),
        "core_erosion_ratio": float(core_erosion_ratio),
        "core_erosion_px": core_radius_px,
        "human_core_miss": human_core,
        "background_core_overmask": background_core,
        "boundary_iou_excluding_u": boundary_iou,
        "u_statistics": {
            "pixels": int(np.count_nonzero(u)),
            "ratio": float(np.count_nonzero(u) / u.size),
        },
    }


@dataclass
class Counts:
    human_fn: int = 0
    human_total: int = 0
    object_pred_h: int = 0
    object_total: int = 0
    background_pred_h: int = 0
    background_total: int = 0
    boundary_pred_matched: int = 0
    boundary_pred_total: int = 0
    boundary_true_matched: int = 0
    boundary_true_total: int = 0
    frame_count: int = 0

    def add(self, labels: dict[str, np.ndarray], predicted_h: np.ndarray) -> None:
        h, o, u, b = (labels[key] for key in ("h", "o", "u", "b"))
        shapes = {array.shape for array in (h, o, u, b, predicted_h)}
        if len(shapes) != 1:
            raise AuditInputError("H/O/U/B/candidate mask dimensions disagree")
        stacked = h.astype(np.uint8) + o + u + b
        if not np.all(stacked == 1):
            raise AuditInputError("frozen H/O/U/B labels must be mutually exclusive and exhaustive")

        self.human_fn += int(np.count_nonzero(h & ~predicted_h))
        self.human_total += int(np.count_nonzero(h))
        self.object_pred_h += int(np.count_nonzero(o & predicted_h))
        self.object_total += int(np.count_nonzero(o))
        self.background_pred_h += int(np.count_nonzero(b & predicted_h))
        self.background_total += int(np.count_nonzero(b))

        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        excluded = cv2.dilate(u.astype(np.uint8), cross).astype(bool)
        true_boundary = _boundary(h) & ~excluded
        pred_boundary = _boundary(predicted_h) & ~excluded
        pred_match_zone = cv2.dilate(true_boundary.astype(np.uint8), cross).astype(bool)
        true_match_zone = cv2.dilate(pred_boundary.astype(np.uint8), cross).astype(bool)
        self.boundary_pred_matched += int(np.count_nonzero(pred_boundary & pred_match_zone))
        self.boundary_pred_total += int(np.count_nonzero(pred_boundary))
        self.boundary_true_matched += int(np.count_nonzero(true_boundary & true_match_zone))
        self.boundary_true_total += int(np.count_nonzero(true_boundary))
        self.frame_count += 1

    def metrics(self) -> dict[str, Any]:
        if self.human_total <= 0 or self.object_total <= 0 or self.background_total <= 0:
            raise AuditInputError("each audited aggregation needs non-empty H, O and B denominators")
        precision = (
            self.boundary_pred_matched / self.boundary_pred_total
            if self.boundary_pred_total
            else (1.0 if self.boundary_true_total == 0 else 0.0)
        )
        recall = (
            self.boundary_true_matched / self.boundary_true_total
            if self.boundary_true_total
            else (1.0 if self.boundary_pred_total == 0 else 0.0)
        )
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "frame_count": self.frame_count,
            "human_miss": self.human_fn / self.human_total,
            "object_corruption": self.object_pred_h / self.object_total,
            "background_overmask": self.background_pred_h / self.background_total,
            "boundary_precision_excluding_u": precision,
            "boundary_recall_excluding_u": recall,
            "boundary_f1_excluding_u": f1,
            "denominators": {
                "human_pixels": self.human_total,
                "object_pixels": self.object_total,
                "background_pixels": self.background_total,
                "boundary_pred_pixels": self.boundary_pred_total,
                "boundary_true_pixels": self.boundary_true_total,
            },
        }


@dataclass
class TemporalCounts:
    flips: int = 0
    valid: int = 0
    pair_count: int = 0
    pair_metrics: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        pair_id: str,
        gray_from: np.ndarray,
        gray_to: np.ndarray,
        pred_from: np.ndarray,
        pred_to: np.ndarray,
        u_from: np.ndarray,
        u_to: np.ndarray,
    ) -> None:
        shapes = {array.shape for array in (gray_from, gray_to, pred_from, pred_to, u_from, u_to)}
        if len(shapes) != 1:
            raise AuditInputError(f"temporal pair dimensions disagree: {pair_id}")
        forward = cv2.calcOpticalFlowFarneback(
            gray_from, gray_to, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        backward = cv2.calcOpticalFlowFarneback(
            gray_to, gray_from, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        height, width = gray_from.shape
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

        def compare(
            base: np.ndarray,
            sample: np.ndarray,
            base_u: np.ndarray,
            sample_u: np.ndarray,
            flow: np.ndarray,
        ) -> tuple[int, int]:
            map_x = grid_x + flow[..., 0]
            map_y = grid_y + flow[..., 1]
            in_bounds = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
            aligned = cv2.remap(
                sample.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            ).astype(bool)
            aligned_u = cv2.remap(
                sample_u.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=1,
            ).astype(bool)
            valid = in_bounds & ~base_u & ~aligned_u & (base | aligned)
            return int(np.count_nonzero((base ^ aligned) & valid)), int(np.count_nonzero(valid))

        f_flips, f_valid = compare(pred_from, pred_to, u_from, u_to, forward)
        b_flips, b_valid = compare(pred_to, pred_from, u_to, u_from, backward)
        flips, valid = f_flips + b_flips, f_valid + b_valid
        if valid <= 0:
            raise AuditInputError(f"temporal pair has no non-U aligned support: {pair_id}")
        self.flips += flips
        self.valid += valid
        self.pair_count += 1
        self.pair_metrics.append({"pair_id": pair_id, "flip": flips / valid, "valid_pixels": valid})

    def metrics(self) -> dict[str, Any]:
        if self.pair_count <= 0 or self.valid <= 0:
            raise AuditInputError("frozen benchmark must contain at least one valid temporal pair")
        return {
            "temporal_flip": self.flips / self.valid,
            "pair_count": self.pair_count,
            "valid_aligned_pixels": self.valid,
            "pairs": self.pair_metrics,
        }


def _empty_report(
    *,
    mode: str,
    status: str,
    benchmark_ref: dict[str, Any],
    label_ref: dict[str, Any] | None,
    candidate_refs: list[dict[str, Any]],
    coverage: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "mask-auditor-a1-report-v1",
        "artifact_state": "SYNTHETIC_TEST_ONLY" if mode == SYNTHETIC_MODE else "MASK_AUDITOR_A1_QA",
        "producer": "independent_mask_qa",
        "auditor": {
            "auditor_id": AUDITOR_ID,
            "implementation_ref": str(IMPLEMENTATION_PATH),
            "implementation_sha256": _implementation_sha256(),
            "formula_id": AUDITOR_FORMULA_ID,
            "formulas": FORMULAS,
            "contract_refs": _auditor_contract_refs(),
            "runtime": {
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "cross_auditor_comparison_allowed": False,
        },
        "mode": mode,
        "status": status,
        "formal_comparison_authorized": False,
        "benchmark_manifest_ref": benchmark_ref,
        "human_labels_manifest_ref": label_ref,
        "candidate_manifest_refs": candidate_refs,
        "coverage": coverage,
        "candidate_results": [],
        "ranking": [],
        "reasons": reasons,
        "claim_limit": (
            "FORENSIC_NOT_COMPARABLE_NO_RANKING"
            if status == "FORENSIC_NOT_COMPARABLE"
            else (
                "SYNTHETIC_TEST_ONLY"
                if mode == SYNTHETIC_MODE
                else "AWAITING_HUMAN_LABELS_NO_PIXEL_QUALITY_CLAIM"
            )
        ),
        "upstream_mutation_permitted": False,
        "created_at": _utc_now(),
    }


def audit_candidates(
    *,
    benchmark_path: Path,
    benchmark_sha256: str,
    candidates: Sequence[tuple[Path, str]],
    human_labels_path: Path | None = None,
    human_labels_sha256: str | None = None,
    mode: str = FORMAL_MODE,
) -> dict[str, Any]:
    """Audit complete candidates or fail closed without intersecting coverage."""

    if mode not in {FORMAL_MODE, SYNTHETIC_MODE}:
        raise AuditInputError(f"unsupported audit mode: {mode}")
    if not candidates:
        raise AuditInputError("at least one candidate manifest is required")

    benchmark, benchmark_ref = _load_bound_json(benchmark_path, benchmark_sha256)
    _validate(benchmark, "mask_benchmark_v1.schema.json")
    _validate_benchmark_identity(benchmark)
    expected_state = "SYNTHETIC_TEST_ONLY" if mode == SYNTHETIC_MODE else "FROZEN"
    if benchmark["artifact_state"] != expected_state:
        raise AuditInputError(f"benchmark artifact_state must be {expected_state} in {mode} mode")

    candidate_documents: list[tuple[dict[str, Any], dict[str, Any]]] = []
    candidate_ids: set[str] = set()
    coverage: dict[str, Any] = {"benchmark_frame_count": len(benchmark["frames"]), "candidates": []}
    coverage_reasons: list[str] = []
    for path, digest in candidates:
        candidate, ref = _load_bound_json(path, digest)
        _validate(candidate, "mask_candidate_eval_v1.schema.json")
        if candidate["artifact_state"] != expected_state:
            raise AuditInputError(f"candidate artifact_state must be {expected_state}: {candidate['candidate_id']}")
        if candidate["candidate_id"] in candidate_ids:
            raise AuditInputError(f"duplicate candidate_id: {candidate['candidate_id']}")
        candidate_ids.add(candidate["candidate_id"])
        if not _strict_ref_matches(candidate["benchmark_manifest_ref"], benchmark_ref):
            raise AuditInputError(f"candidate does not bind the exact benchmark: {candidate['candidate_id']}")
        exact, reasons = _coverage_reason(benchmark, candidate)
        coverage["candidates"].append({
            "candidate_id": candidate["candidate_id"],
            "frame_count": len(candidate["frames"]),
            "exact_complete_coverage": exact,
            "source_sha_match": not any("source SHA mismatch" in reason for reason in reasons),
        })
        coverage_reasons.extend(f"{candidate['candidate_id']}: {reason}" for reason in reasons)
        candidate_documents.append((candidate, ref))

    candidate_refs = [ref for _, ref in candidate_documents]
    if coverage_reasons:
        return _empty_report(
            mode=mode,
            status="FORENSIC_NOT_COMPARABLE",
            benchmark_ref=benchmark_ref,
            label_ref=None,
            candidate_refs=candidate_refs,
            coverage=coverage,
            reasons=coverage_reasons,
        )

    if human_labels_path is None or human_labels_sha256 is None:
        if mode == SYNTHETIC_MODE:
            raise AuditInputError("synthetic evaluation requires an explicit synthetic labels manifest")
        return _empty_report(
            mode=mode,
            status="AWAITING_HUMAN_LABELS",
            benchmark_ref=benchmark_ref,
            label_ref=None,
            candidate_refs=candidate_refs,
            coverage=coverage,
            reasons=["completed, frozen, SHA-bound H/O/U/B human labels manifest is absent"],
        )

    labels, labels_ref = _load_bound_json(human_labels_path, human_labels_sha256)
    _validate(labels, "mask_houb_labels_v1.schema.json")
    if labels["artifact_state"] != expected_state:
        raise AuditInputError(f"human labels artifact_state must be {expected_state} in {mode} mode")
    if not _strict_ref_matches(labels["benchmark_manifest_ref"], benchmark_ref):
        raise AuditInputError("human labels do not bind the exact benchmark manifest")
    if mode == FORMAL_MODE:
        if not (labels["completed_by_human"] and labels["frozen"]):
            return _empty_report(
                mode=mode,
                status="AWAITING_HUMAN_LABELS",
                benchmark_ref=benchmark_ref,
                label_ref=labels_ref,
                candidate_refs=candidate_refs,
                coverage=coverage,
                reasons=["H/O/U/B labels are not both human-completed and frozen"],
            )
        _read_ref(labels["human_approval_ref"])

    labels_exact, labels_reasons = _labels_coverage_reason(benchmark, labels)
    coverage["human_labels_exact_complete_coverage"] = labels_exact
    if labels_reasons:
        return _empty_report(
            mode=mode,
            status="FORENSIC_NOT_COMPARABLE",
            benchmark_ref=benchmark_ref,
            label_ref=labels_ref,
            candidate_refs=candidate_refs,
            coverage=coverage,
            reasons=labels_reasons,
        )

    label_by_id = {row["sample_id"]: row for row in labels["frames"]}
    benchmark_by_id = {row["sample_id"]: row for row in benchmark["frames"]}
    label_arrays: dict[str, dict[str, np.ndarray]] = {}
    gray: dict[str, np.ndarray] = {}
    for sample_id, row in benchmark_by_id.items():
        label_row = label_by_id[sample_id]
        label_arrays[sample_id] = {
            key: _decode_mask(label_row[f"{key}_mask"]) for key in ("h", "o", "u", "b")
        }
        gray[sample_id] = _decode_gray(row["source_image"])
        if gray[sample_id].shape != label_arrays[sample_id]["h"].shape:
            raise AuditInputError(f"source/label dimensions disagree: {sample_id}")

    results: list[dict[str, Any]] = []
    for candidate, manifest_ref in candidate_documents:
        predicted = {
            row["sample_id"]: _decode_mask(row["predicted_human_mask"])
            for row in candidate["frames"]
        }
        aggregate = Counts()
        by_split: dict[str, Counts] = {
            split: Counts() for split in ("development", "same_session_blind", "cross_session_blind")
        }
        for row in benchmark["frames"]:
            sample_id = row["sample_id"]
            aggregate.add(label_arrays[sample_id], predicted[sample_id])
            by_split[row["split"]].add(label_arrays[sample_id], predicted[sample_id])

        temporal = TemporalCounts()
        for pair in benchmark["temporal_pairs"]:
            first, second = pair["from_sample_id"], pair["to_sample_id"]
            temporal.add(
                pair["pair_id"], gray[first], gray[second], predicted[first], predicted[second],
                label_arrays[first]["u"], label_arrays[second]["u"],
            )
        aggregate_metrics = aggregate.metrics()
        temporal_metrics = temporal.metrics()
        ranking_key = [
            max(aggregate_metrics["human_miss"], aggregate_metrics["object_corruption"]),
            aggregate_metrics["human_miss"] + aggregate_metrics["object_corruption"],
            aggregate_metrics["background_overmask"],
            1.0 - aggregate_metrics["boundary_f1_excluding_u"],
            temporal_metrics["temporal_flip"],
        ]
        results.append({
            "candidate_id": candidate["candidate_id"],
            "producer_version": candidate["producer_version"],
            "manifest_ref": manifest_ref,
            "aggregate": aggregate_metrics,
            "by_split": {name: counts.metrics() for name, counts in by_split.items()},
            "temporal": temporal_metrics,
            "ranking_key": ranking_key,
        })

    ranked = sorted(results, key=lambda row: (row["ranking_key"], row["candidate_id"]))
    report = _empty_report(
        mode=mode,
        status="EVALUATED",
        benchmark_ref=benchmark_ref,
        label_ref=labels_ref,
        candidate_refs=candidate_refs,
        coverage=coverage,
        reasons=[],
    )
    report["formal_comparison_authorized"] = mode == FORMAL_MODE
    report["candidate_results"] = results
    report["ranking"] = [row["candidate_id"] for row in ranked]
    report["claim_limit"] = (
        "BENCHMARK_COMPARISON_ONLY_NOT_PRODUCTION_PASS"
        if mode == FORMAL_MODE
        else "SYNTHETIC_TEST_ONLY"
    )
    _validate(report, "mask_auditor_a1.schema.json")
    return report


def write_new_report(path: Path, report: dict[str, Any]) -> None:
    """Create one new QA report; never overwrite or follow a symlink."""

    _validate(report, "mask_auditor_a1.schema.json")
    path = _normal_absolute(path)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise AuditInputError("QA output parent must be an existing ordinary directory")
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as exc:
        raise AuditInputError(f"refusing to overwrite/follow QA output: {path}: {exc}") from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _candidate_arg(value: str) -> tuple[Path, str]:
    try:
        path_text, digest = value.rsplit("@", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate must be ABSOLUTE_PATH@SHA256") from exc
    return Path(path_text), digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--benchmark-sha256", required=True)
    parser.add_argument("--candidate", action="append", type=_candidate_arg, required=True)
    parser.add_argument("--human-labels", type=Path)
    parser.add_argument("--human-labels-sha256")
    parser.add_argument("--mode", choices=(FORMAL_MODE, SYNTHETIC_MODE), default=FORMAL_MODE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.human_labels is None) != (args.human_labels_sha256 is None):
        parser.error("--human-labels and --human-labels-sha256 must be provided together")
    report = audit_candidates(
        benchmark_path=args.benchmark,
        benchmark_sha256=args.benchmark_sha256,
        candidates=args.candidate,
        human_labels_path=args.human_labels,
        human_labels_sha256=args.human_labels_sha256,
        mode=args.mode,
    )
    write_new_report(args.output, report)


if __name__ == "__main__":
    main()
