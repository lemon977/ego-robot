#!/usr/bin/env python3
"""Freeze the exact writer semantics of the legacy 031 HaWoR source flags.

This is a provenance sidecar only.  It never rewrites the sealed HaWoR NPZ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path("/mnt/workspace/code/chaoyang")
TASK = "human_to_robot_evidence_unlock_s2_20260923"
V3 = REPO / "_run/current/four_stream_full_pipeline_v3/attempts/attempt_0001"
DEFAULT_OUTPUT = (
    REPO
    / f"_run/current/{TASK}/attempts/attempt_0001/lanes/motion_product"
    / "source_provenance_031/attempt_0001"
)


def artifact(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"FRESH_OUTPUT_REQUIRED:{output}")
    if not output.is_relative_to(REPO.resolve()):
        raise RuntimeError("OUTPUT_MUST_BE_INSIDE_REPOSITORY")

    producer = (
        REPO
        / "_run/current/worktrees/full-pipeline-v3-integration/src/chaoyang/ops"
        / "run_hawor_roi_full_pipeline_v3.py"
    )
    spec = V3 / "lanes/ai2/FULL031_COMPONENT_EPOCH7_SPEC.json"
    result_path = V3 / "lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/RESULT.json"
    npz_path = V3 / "lanes/ai2/hawor_full031_epoch7_c3/play_cards_0915_031/HAWOR_CAMERA_SOURCE_V3.npz"
    receipt = V3 / "lanes/ai2/HAWOR031_EPOCH7_GPU_RECEIPT.json"
    for path in (producer, spec, result_path, npz_path, receipt):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = producer.read_text(encoding="utf-8")
    required_logic = (
        "observed = predicted & direct.T",
        "inferred = predicted & ~direct.T",
        "direct = np.asarray(rois['det_direct'], bool)",
    )
    missing = [token for token in required_logic if token not in source]
    if missing:
        raise RuntimeError(f"PRODUCER_LOGIC_NOT_FOUND:{missing}")

    result = json.loads(result_path.read_text(encoding="utf-8"))
    spec_value = json.loads(spec.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as arrays:
        observed = np.asarray(arrays["observed"], dtype=bool)
        inferred = np.asarray(arrays["inferred"], dtype=bool)
        predicted = np.asarray(arrays["predicted_valid"], dtype=bool)
        frame_ids = np.asarray(arrays["original_frame_indices"], dtype=np.int64)
        side_names = [str(x) for x in arrays["anatomical_side_names"]]

    if observed.shape != inferred.shape or observed.shape != predicted.shape:
        raise RuntimeError("SOURCE_FLAG_SHAPE_MISMATCH")
    if np.any(observed & inferred):
        raise RuntimeError("OBSERVED_AND_INFERRED_OVERLAP")
    if not np.array_equal(observed | inferred, predicted):
        raise RuntimeError("SOURCE_FLAGS_DO_NOT_PARTITION_PREDICTIONS")
    counts_observed = observed.sum(axis=1).astype(int).tolist()
    counts_inferred = inferred.sum(axis=1).astype(int).tolist()
    counts_predicted = predicted.sum(axis=1).astype(int).tolist()
    if counts_observed != result["direct_roi_by_side"]:
        raise RuntimeError("OBSERVED_COUNT_MISMATCH")
    if counts_inferred != result["inferred_roi_by_side"]:
        raise RuntimeError("INFERRED_COUNT_MISMATCH")
    if counts_predicted != result["predicted_by_side"]:
        raise RuntimeError("PREDICTED_COUNT_MISMATCH")

    identity = spec_value["sessions"][0]["identity_evidence"]
    side_rows = []
    for side_index, side_name in enumerate(side_names):
        side_rows.append(
            {
                "side_index": side_index,
                "anatomical_side_name": side_name,
                "predicted_count": counts_predicted[side_index],
                "observed_count": counts_observed[side_index],
                "inferred_count": counts_inferred[side_index],
                "first_predicted_frame": (
                    int(frame_ids[np.flatnonzero(predicted[side_index])[0]])
                    if counts_predicted[side_index]
                    else None
                ),
                "last_predicted_frame": (
                    int(frame_ids[np.flatnonzero(predicted[side_index])[-1]])
                    if counts_predicted[side_index]
                    else None
                ),
            }
        )

    payload = {
        "schema_version": "HUMAN_TO_ROBOT_S2_031_SOURCE_PROVENANCE_V1",
        "task_id": TASK,
        "session_id": "play_cards_0915_031",
        "created_at": utc_now(),
        "execution": "EXECUTED",
        "structure": "PASS",
        "quality": "INCONCLUSIVE_SOURCE_AUTHORITY",
        "adoption": "CANDIDATE_ONLY",
        "writer_semantics": {
            "observed": "predicted_valid AND ROI det_direct",
            "inferred": "predicted_valid AND NOT ROI det_direct",
            "critical_limit": (
                "inferred means a HaWoR model prediction was produced from a non-direct ROI; "
                "it is not proof that HaWoR motion infiller produced the row"
            ),
            "three_dimensional_claim": "MODEL_PREDICTION_NOT_DIRECT_3D_OBSERVATION",
        },
        "side_rows": side_rows,
        "identity_authority": identity,
        "temporal_authority": "OFFLINE_NONCAUSAL",
        "left_status": "NO_ROI_NO_MODEL_OUTPUT_NOT_FILLED",
        "right_status": (
            "102_MODEL_PREDICTIONS_FROM_SAM_TEMPORAL_ROI; "
            "AI_VISUAL_SIDE_ANCHOR_NOT_INDEPENDENT_GOLD"
        ),
        "old_npz_modified": False,
        "inputs": {
            "producer_source": artifact(producer),
            "producer_execution_receipt": artifact(receipt),
            "component_spec": artifact(spec),
            "producer_result": artifact(result_path),
            "sealed_npz": artifact(npz_path),
        },
        "claim_limit": (
            "Source-field lineage only. It does not upgrade side identity, anatomical wrist accuracy, "
            "motion quality, causal availability, Robot control, or deployment authority."
        ),
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    output.mkdir(parents=True)
    path = output / "RESULT.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "result": artifact(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
