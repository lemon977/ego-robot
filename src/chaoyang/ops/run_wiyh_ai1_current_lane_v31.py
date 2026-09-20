#!/usr/bin/env python3
"""Run the CPU-only, current-source WIYH AI1 lane V3.1.

The runner preserves partial, truthful results.  It fits only on 097/098,
evaluates 101 as regression, and records independent blockers for 102/103
rather than fabricating anatomical-wrist observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np

from chaoyang.ops import build_wiyh_wrist_dual_input_v1 as input_builder
from chaoyang.ops import build_wiyh_wrist_dual_representation_v1 as producer
from chaoyang.research.world_in_your_hands.wrist_dual_representation_v1 import (
    compose_camera_wrist,
    evaluate_static_calibration,
)


SCHEMA_VERSION = "chaoyang-wiyh-ai1-lane-result-v31"
LEDGER_VERSION = "chaoyang-wiyh-ai1-lane-ledger-v31"
FIT_IDS = ("play_cards_0916_097", "play_cards_0916_098")
REGRESSION_ID = "play_cards_0916_101"
ADOPTION_IDS = ("play_cards_0916_102", "play_cards_0916_103")
MODEL_FIELDS = {
    "M0_LEGACY": "T_controller_wrist_M0",
    "M1_CONTROLLER_LOCAL_TRANSLATION": "T_controller_wrist_M1",
    "M2_STATIC_SE3": "T_controller_wrist_M2",
}


class WiyhAi1LaneError(RuntimeError):
    """Raised when the current-only lane cannot be safely published."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    value = path.resolve(strict=True)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": _sha256(value)}


def _write_json_no_clobber(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise WiyhAi1LaneError(f"refusing to clobber JSON output: {path}")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)


def _write_npz_no_clobber(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise WiyhAi1LaneError(f"refusing to clobber NPZ output: {path}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)


def _producer_config() -> dict[str, Any]:
    return {
        "schema_version": "chaoyang-wrist-dual-producer-config-v1",
        "fit_recording_ids": list(FIT_IDS),
        "forbidden_fit_recording_ids": [REGRESSION_ID],
        "selected_model": {
            "left": "M1_CONTROLLER_LOCAL_TRANSLATION",
            "right": "M1_CONTROLLER_LOCAL_TRANSLATION",
        },
        "minimum_m2_orientation_samples": 8,
        "minimum_m2_controller_rotation_span_deg": 10.0,
        "fusion_heuristic_weight": 0.0,
        "development_numeric_correction_bound_mm": 30.0,
        "fps": 30.0,
        "review_image_domain": {
            "name": "PRECOMPUTED_HAWOR_SOURCE_UV",
            "width": 1280,
            "height": 960,
            "projection": "PRECOMPUTED_SOURCE_UV",
        },
        "temporal_authority": {
            "T_camera_controller_raw": "UNKNOWN_TEMPORAL_AUTHORITY",
            "observed_T_camera_wrist": "OFFLINE_NONCAUSAL",
            "visible_wrist_surface": "UNKNOWN_TEMPORAL_AUTHORITY",
        },
    }


def _m0_baseline(input_path: Path, output_path: Path) -> dict[str, Any]:
    with np.load(input_path, allow_pickle=False) as archive:
        controller = np.asarray(archive["T_camera_controller_raw"], dtype=np.float64)
        m0 = np.asarray(archive["T_controller_wrist_M0"], dtype=np.float64)
        arrays = {
            "frame_id": np.asarray(archive["frame_id"]),
            "timestamp_s": np.asarray(archive["timestamp_s"]),
            "recording_id": np.asarray(archive["recording_id"]),
            "T_camera_controller_raw": controller,
            "T_controller_wrist_M0": m0,
            "M0_T_camera_wrist": compose_camera_wrist(controller, m0[None]),
            "observed_T_camera_wrist": np.asarray(archive["observed_T_camera_wrist"]),
            "observed_valid": np.asarray(archive["observed_valid"], dtype=bool),
            "orientation_valid": np.asarray(archive["orientation_valid"], dtype=bool),
        }
    _write_npz_no_clobber(output_path, arrays)
    return _artifact(output_path)


def _empty_model_rows(reason: str) -> dict[str, Any]:
    return {
        "M0_LEGACY": {"status": "NOT_AVAILABLE", "reason": reason},
        "M1_CONTROLLER_LOCAL_TRANSLATION": {"status": "BLOCKED", "reason": reason},
        "M2_STATIC_SE3": {"status": "BLOCKED", "reason": reason},
    }


def _evaluate_sessions(
    input_path: Path, representation_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    with (
        np.load(input_path, allow_pickle=False) as source,
        np.load(representation_path, allow_pickle=False) as representation,
    ):
        recording = np.asarray(source["recording_id"]).astype(str)
        controller = np.asarray(source["T_camera_controller_raw"], dtype=np.float64)
        observed = np.asarray(source["observed_T_camera_wrist"], dtype=np.float64)
        observed_valid = np.asarray(source["observed_valid"], dtype=bool)
        orientation_valid = np.asarray(source["orientation_valid"], dtype=bool)
        models = {
            name: np.asarray(representation[field], dtype=np.float64)
            for name, field in MODEL_FIELDS.items()
        }
    sessions: dict[str, Any] = {}
    for session_id in (*FIT_IDS, REGRESSION_ID):
        role = "DEVELOPMENT" if session_id in FIT_IDS else "REGRESSION_FORBIDDEN_FIT"
        session_mask = recording == session_id
        model_rows: dict[str, Any] = {}
        for model_name, transforms in models.items():
            side_rows: dict[str, Any] = {}
            available = np.isfinite(transforms).all(axis=(1, 2))
            for side_index, side in enumerate(("left", "right")):
                if not available[side_index]:
                    side_rows[side] = {
                        "status": "BLOCKED_ORIENTATION_EVIDENCE"
                        if model_name == "M2_STATIC_SE3"
                        else "BLOCKED_MODEL_UNAVAILABLE"
                    }
                    continue
                valid = session_mask & observed_valid[:, side_index]
                metrics = evaluate_static_calibration(
                    controller[:, side_index],
                    transforms[side_index],
                    observed[:, side_index],
                    valid,
                    orientation_valid[:, side_index],
                )
                side_rows[side] = {"status": "AVAILABLE_POSITION_ONLY", "metrics": metrics}
            model_rows[model_name] = {"status": "EVALUATED", "sides": side_rows}
        sessions[session_id] = {
            "role": role,
            "status": "POSITION_ONLY_EVALUATED",
            "models": model_rows,
        }
    for session_id in ADOPTION_IDS:
        sessions[session_id] = {
            "role": "ADOPTION",
            "status": "BLOCKED_MISSING_PINNED_OBSERVATION",
            "blocker": "No current pinned HaWoR anatomical-wrist observation; filler is forbidden.",
        }
    model_summary = {
        "M0_LEGACY": {
            "status": "AVAILABLE_LEGACY_PRIOR",
            "selection_authority": "REFERENCE_ONLY_NOT_INDEPENDENT_TRUTH",
        },
        "M1_CONTROLLER_LOCAL_TRANSLATION": {
            "status": "AVAILABLE_DEVELOPMENT_CANDIDATE",
            "fit_recording_ids": list(FIT_IDS),
            "regression_recording_id": REGRESSION_ID,
            "adopted": False,
        },
        "M2_STATIC_SE3": {
            "status": "BLOCKED_ORIENTATION_EVIDENCE",
            "orientation_valid_count": 0,
            "adopted": False,
        },
    }
    return sessions, model_summary


def _validate_and_write(*, output_root: Path, ledger: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    ledger_schema = json.loads(
        (repo_root / "contracts/wiyh_ai1_lane_ledger_v31.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(ledger, ledger_schema)
    ledger_path = output_root / "LANE_LEDGER.json"
    _write_json_no_clobber(ledger_path, ledger)
    artifacts = [value for value in ledger["outputs"].values() if isinstance(value, dict)]
    artifacts.append(_artifact(ledger_path))
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": ledger["status"],
        "lane": "ai1",
        "ledger": _artifact(ledger_path),
        "latest_artifacts": artifacts,
        "blocker": ledger["blockers"][0] if ledger["blockers"] else None,
        "claims": {
            "PIPELINE_COMPLETE": False,
            "NUMERIC_QUALITY_PASS": False,
            "VISUAL_REVIEW_STATUS": "NOT_REVIEWED",
            "TRAINING_COMPLETE": False,
            "TRAINING_ELIGIBLE": False,
            "CONTROL_GROUND_TRUTH": False,
            "PHYSICAL_DEPLOYABLE": False,
        },
    }
    result_schema = json.loads(
        (repo_root / "contracts/wiyh_ai1_lane_result_v31.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(result, result_schema)
    _write_json_no_clobber(output_root / "RESULT.json", result)
    return result


def run(*, processed_root: Path, experiment_root: Path, output_root: Path) -> dict[str, Any]:
    """Run the bounded AI1 lane and always publish a machine-readable terminal result."""

    output = output_root.absolute()
    if output.exists() or output.is_symlink():
        raise WiyhAi1LaneError(f"fresh output root required: {output}")
    output.mkdir(parents=True)
    stages: dict[str, str] = {
        "input": "NOT_STARTED",
        "m0": "NOT_STARTED",
        "producer": "NOT_STARTED",
    }
    outputs: dict[str, dict[str, Any] | None] = {
        "input_provenance": None,
        "m0_baseline": None,
        "producer_result": None,
    }
    models = _empty_model_rows("INPUT_NOT_AVAILABLE")
    sessions = {session_id: {"role": "DEVELOPMENT", "status": "NOT_RUN"} for session_id in FIT_IDS}
    sessions[REGRESSION_ID] = {"role": "REGRESSION_FORBIDDEN_FIT", "status": "NOT_RUN"}
    sessions.update(
        {
            session_id: {
                "role": "ADOPTION",
                "status": "BLOCKED_MISSING_PINNED_OBSERVATION",
                "blocker": "No current pinned HaWoR anatomical-wrist observation; filler is forbidden.",
            }
            for session_id in ADOPTION_IDS
        }
    )
    blockers = [f"{session_id}:BLOCKED_MISSING_PINNED_OBSERVATION" for session_id in ADOPTION_IDS]
    status = "BLOCKED_ADOPTION_OBSERVATIONS"
    try:
        input_builder.build(
            processed_root=processed_root,
            experiment_root=experiment_root,
            output_root=output / "input_bundle",
        )
        input_path = output / "input_bundle" / "WRIST_DUAL_INPUT.npz"
        outputs["input_provenance"] = _artifact(output / "input_bundle" / "PROVENANCE.json")
        stages["input"] = "PASS_FIXED_CURRENT_INPUT"
        outputs["m0_baseline"] = _m0_baseline(input_path, output / "M0_BASELINE.npz")
        stages["m0"] = "PASS_M0_PRESERVED"

        config_path = output / "PRODUCER_CONFIG.json"
        _write_json_no_clobber(config_path, _producer_config())
        try:
            producer.produce(
                input_npz=input_path,
                config_path=config_path,
                output_root=output / "dual_representation",
                raw_video=None,
            )
            producer_result_path = output / "dual_representation" / "RESULT.json"
            outputs["producer_result"] = _artifact(producer_result_path)
            stages["producer"] = "PASS_DEVELOPMENT_POSITION_ONLY"
            sessions, models = _evaluate_sessions(
                input_path,
                output / "dual_representation" / "WRIST_DUAL_REPRESENTATION_V1.npz",
            )
        except (producer.WristDualProducerError, ValueError) as error:
            stages["producer"] = "BLOCKED_STATIC_CALIBRATION"
            status = "BLOCKED_STATIC_CALIBRATION"
            blockers.insert(0, f"PRODUCER:{type(error).__name__}:{error}")
            models = _empty_model_rows(str(error))
            models["M0_LEGACY"] = {
                "status": "AVAILABLE_LEGACY_PRIOR",
                "selection_authority": "REFERENCE_ONLY_NOT_INDEPENDENT_TRUTH",
            }
    except (input_builder.WiyhWristDualInputError, FileNotFoundError, KeyError) as error:
        stages["input"] = "BLOCKED_CURRENT_SOURCE_VALIDATION"
        status = "BLOCKED_CURRENT_SOURCE_VALIDATION"
        blockers.insert(0, f"INPUT:{type(error).__name__}:{error}")
    except Exception as error:  # noqa: BLE001 - terminal ledger must survive unexpected runtime failure.
        status = "FAILED_RUNTIME"
        blockers.insert(0, f"RUNTIME:{type(error).__name__}:{error}")

    ledger = {
        "schema_version": LEDGER_VERSION,
        "status": status,
        "lane": "ai1",
        "stages": stages,
        "models": models,
        "sessions": sessions,
        "outputs": outputs,
        "blockers": blockers,
        "authority": {
            "external_wrist_truth": False,
            "external_metric_authority": False,
            "control_ground_truth": False,
            "physical_deployable": False,
        },
        "execution": {
            "cpu_only": True,
            "gpu_used": False,
            "source_mutated": False,
            "old_artifact_mutated": False,
        },
    }
    return _validate_and_write(output_root=output, ledger=ledger)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                processed_root=args.processed_root,
                experiment_root=args.experiment_root,
                output_root=args.output_root,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
