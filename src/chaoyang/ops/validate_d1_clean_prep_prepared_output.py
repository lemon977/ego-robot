#!/usr/bin/env python3
"""Independent CPU validator for a D1_CLEAN_PREP zero-write preparation output."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "d1_clean_prep_cpu", HERE / "run_d1_clean_prep_cpu_orchestrator.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load D1_CLEAN_PREP producer")
d1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d1)


def load_mask(reference, label: str) -> np.ndarray:
    path = d1.checked_ref(reference, label)
    with Image.open(path) as image:
        value = np.asarray(image.convert("L"))
    if not set(np.unique(value).tolist()).issubset({0, 255}):
        raise d1.ContractError(f"{label}: non-binary mask")
    return value > 0


def validate(output_root: Path) -> dict:
    output_root = output_root.resolve(strict=True)
    result_path = output_root / "RESULT.json"
    result = d1.load_json(result_path)
    if result.get("schema_version") != f"{d1.SCHEMA}-result-v1":
        raise d1.ContractError("D1_CLEAN_PREP result schema drift")
    if tuple(result.get("fixed_sessions", ())) != d1.FIXED_SESSIONS:
        raise d1.ContractError("D1_CLEAN_PREP result scope drift")
    if result.get("gpu_used") is not False or result.get("model_execution_performed") is not False:
        raise d1.ContractError("zero-write validator received a model-executed result")
    summaries = result.get("sessions")
    if not isinstance(summaries, list) or tuple(row.get("session_id") for row in summaries if isinstance(row, dict)) != d1.FIXED_SESSIONS:
        raise d1.ContractError("D1_CLEAN_PREP result session rows drift")
    validated_frames = 0
    aggregate = {
        "m_remove_pixels": 0,
        "m_flow_pixels": 0,
        "m_write_pixels": 0,
        "unknown_pixels": 0,
        "protected_pixels": 0,
        "changed_pixels": 0,
        "changed_outside_m_write_pixels": 0,
        "changed_protected_pixels": 0,
    }
    session_receipts = []
    allowed_codes = {0, 1, 2, 10, 11, 12, 250}
    for summary in summaries:
        session_id = summary["session_id"]
        task = summary["task"]
        session_result_path = d1.checked_ref(summary["result"], f"{session_id} session result")
        session_result = d1.load_json(session_result_path)
        manifest_path = d1.checked_ref(summary["frame_manifest"], f"{session_id} frame manifest")
        manifest = d1.load_json(manifest_path)
        frames = d1.ordered_rows(manifest.get("frames"), manifest.get("frame_count"), f"{session_id} frame manifest")
        frame_count = len(frames)
        pins = session_result.get("input_pins", {})
        _, _, raw_rows = d1.validate_raw_manifest(
            pins.get("raw_frame_manifest", {}), session_id, task, frame_count
        )
        object_manifest, _, object_rows, _ = d1.validate_object_manifest(
            pins.get("task_object_manifest", {}), session_id, task, frame_count
        )
        hand_rows = {}
        unknown_sides = set(session_result.get("unknown_hand_sides", []))
        for side in d1.SIDES:
            side_ref = pins.get("hand_side_manifests", {}).get(side)
            if side in unknown_sides:
                if side_ref is not None:
                    raise d1.ContractError(f"{session_id}/{side}: UNKNOWN side has a manifest pin")
                hand_rows[side] = None
            else:
                _, _, rows = d1.validate_hand_manifest(side_ref, session_id, side, frame_count)
                hand_rows[side] = rows
        session_totals = {key: 0 for key in aggregate}
        for frame_id, row in enumerate(frames):
            raw = d1.read_rgb(row["raw_rgb"], f"{session_id} frame {frame_id} Raw")
            if row["raw_rgb"] != raw_rows[frame_id]["rgb"]:
                raise d1.ContractError(f"{session_id} frame {frame_id}: Raw manifest ref drift")
            candidate = d1.read_rgb(row["candidate_rgb"], f"{session_id} frame {frame_id} candidate")
            m_remove = load_mask(row["M_remove"], f"{session_id} frame {frame_id} M_remove")
            m_flow = load_mask(row["M_flow"], f"{session_id} frame {frame_id} M_flow")
            m_write = load_mask(row["M_write"], f"{session_id} frame {frame_id} M_write")
            unknown = load_mask(row["UNKNOWN"], f"{session_id} frame {frame_id} UNKNOWN")
            source_path = d1.checked_ref(row["source_map"], f"{session_id} frame {frame_id} source map")
            with Image.open(source_path) as image:
                source = np.asarray(image.convert("L"))
            if source.shape != raw.shape[:2] or not set(np.unique(source).tolist()).issubset(allowed_codes):
                raise d1.ContractError(f"{session_id} frame {frame_id}: source-map domain drift")
            recomputed_remove = np.zeros(raw.shape[:2], dtype=bool)
            for side in d1.SIDES:
                if hand_rows[side] is not None:
                    recomputed_remove |= d1.read_mask(
                        hand_rows[side][frame_id]["mask"], raw.shape[:2],
                        f"{session_id} frame {frame_id} {side} Hand recheck",
                    )
            protected = np.zeros(raw.shape[:2], dtype=bool)
            expected_source = np.full(raw.shape[:2], int(d1.SOURCE_TARGET_RAW), dtype=np.uint8)
            for instance_index, item in enumerate(object_rows[frame_id]["instances"]):
                mask = d1.read_mask(
                    item["mask"], raw.shape[:2],
                    f"{session_id} frame {frame_id} object {item['instance_id']} recheck",
                )
                if np.any(protected & mask):
                    raise d1.ContractError(f"{session_id} frame {frame_id}: object masks overlap")
                protected |= mask
                if task == "potato_chips":
                    code = (10, 11, 12)[instance_index]
                elif object_manifest.get("identity_authority") == "PASSED_INDEPENDENT_PHYSICAL_CARD_ID":
                    code = 1
                else:
                    code = 2
                expected_source[mask] = code
            recomputed_remove &= ~protected
            if not np.array_equal(recomputed_remove, m_remove):
                raise d1.ContractError(f"{session_id} frame {frame_id}: M_remove is not the object-vetoed admitted Hand union")
            expected_source[unknown] = int(d1.SOURCE_UNKNOWN_UNWRITTEN)
            if not np.array_equal(source, expected_source):
                raise d1.ContractError(f"{session_id} frame {frame_id}: source map does not match separate object masks/UNKNOWN")
            if not np.array_equal(source == int(d1.SOURCE_UNKNOWN_UNWRITTEN), unknown):
                raise d1.ContractError(f"{session_id} frame {frame_id}: UNKNOWN/source-map mismatch")
            metrics = d1.audit_domains(
                raw, candidate, m_remove, m_flow, m_write, unknown, protected
            )
            if row.get("raw_decoded_sha256") != d1.decoded_sha256(raw):
                raise d1.ContractError(f"{session_id} frame {frame_id}: Raw decoded SHA drift")
            if row.get("candidate_decoded_sha256") != d1.decoded_sha256(candidate):
                raise d1.ContractError(f"{session_id} frame {frame_id}: candidate decoded SHA drift")
            for key in session_totals:
                if int(row.get(key, -1)) != int(metrics[key]):
                    raise d1.ContractError(f"{session_id} frame {frame_id}: metric drift for {key}")
                session_totals[key] += int(metrics[key])
                aggregate[key] += int(metrics[key])
            validated_frames += 1
        if session_totals != summary.get("totals"):
            raise d1.ContractError(f"{session_id}: session totals drift")
        session_receipts.append({
            "session_id": session_id,
            "frames_validated": len(frames),
            "totals": session_totals,
            "frame_manifest": d1.file_ref(manifest_path),
        })
    if aggregate["changed_pixels"] != 0:
        raise d1.ContractError("CPU preparation must not materialize candidate writes")
    return {
        "schema_version": f"{d1.SCHEMA}-independent-validation-v1",
        "created_at": d1.now(),
        "status": "PASSED_D1_CLEAN_PREP_CPU_PREFILL_DEEP_VALIDATION",
        "fixed_sessions": list(d1.FIXED_SESSIONS),
        "frames_validated": validated_frames,
        "aggregate": aggregate,
        "hard_gates": {
            "all_artifact_refs_sha_exact": "PASS",
            "M_remove_subset_M_write_subset_M_flow": "PASS",
            "M_write_equals_object_vetoed_M_remove_in_fail_closed_mode": "PASS",
            "M_remove_recomputed_from_B1_admitted_sides": "PASS",
            "Task_Object_recomputed_from_separate_instance_masks": "PASS",
            "UNKNOWN_equals_M_write_until_fresh_materialization": "PASS",
            "source_map_full_frame_and_code_domain": "PASS",
            "M_write_outside_raw_pixel_byte_exact": "PASS",
            "protected_object_raw_pixel_byte_exact": "PASS",
            "zero_model_materialization": "PASS",
        },
        "sessions": session_receipts,
        "pins": {
            "producer_result": d1.file_ref(result_path),
            "producer": d1.file_ref(HERE / "run_d1_clean_prep_cpu_orchestrator.py"),
            "validator": d1.file_ref(Path(__file__)),
        },
        "clean_terminal": False,
        "training_eligible": False,
        "claim_limit": "Validates only the CPU prefill preparation and provenance closure; M_write is not yet materialized and there is no inpainting quality or Clean authority.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = validate(args.output_root)
        receipt = args.receipt.resolve()
        if receipt.exists() or receipt.is_symlink():
            raise d1.ContractError(f"fresh validation receipt required: {receipt}")
        d1.write_json(receipt, payload)
    except (d1.ContractError, OSError, ValueError) as error:
        print(json.dumps({
            "status": "BLOCKED_FAIL_CLOSED",
            "error": f"{type(error).__name__}: {error}",
        }, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
