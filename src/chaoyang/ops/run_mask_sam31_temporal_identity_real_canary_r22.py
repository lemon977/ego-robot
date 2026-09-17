#!/usr/bin/env python3
"""Run the two frozen SAM3.1 temporal identity canaries.

This is the executable binding layer.  It binds immutable selection and seed
receipts to the tested causal integration module, applies fail-closed gates,
and seals a six-file terminal result.  It never updates current governance.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SAM_ROOT = ROOT / "vendor/SAM3"
if str(SAM_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM_ROOT))

from chaoyang.pipeline import sam31_compat_adapter_v1
from chaoyang.pipeline.sam31_temporal_fullsession_integration_r22 import (
    FrozenSeedReplay,
    persist_compact_evidence,
    replay_detector_then_track_stable_point_id,
    replay_seed_then_collect,
)
from chaoyang.pipeline.sam31_causal_seed_propagation_adapter_r22 import gate_single_instance
from chaoyang.ops import run_clean_layered_sam31_t0_visual_v1 as sam_helpers


SELECTION_SHA = "4af887bf9502ae0561f0838e30939b034d358658cdc0bee2522a18a296df25e1"
CHECKPOINT_SHA = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
DEFAULT_SELECTION = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_next_cpu_freeze_v1/attempts/attempt_0001/MASK_NEXT_FROZEN_SELECTION.json"
DEFAULT_ANNOTATIONS = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_prompt_bootstrap_canary_v1/PROMPT_ANNOTATIONS_FROZEN.json"
DEFAULT_BOOTSTRAP = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260916_r22_mask_prompt_bootstrap_canary_v1/attempts/attempt_0002/PROMPT_CANDIDATES.json"
DEFAULT_CHECKPOINT = ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"


class RealCanaryContractError(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def reference(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _session_count(selection: dict[str, Any], key: str) -> int:
    return int(selection["tasks"][key]["failed_canary"]["baseline_terminal"]["verified_frame_count"])


def bind_frozen_targets(
    selection_path: Path,
    annotations_path: Path,
    bootstrap_path: Path,
) -> list[dict[str, Any]]:
    """Bind CLI inputs to the exact frozen 0901 identities and accepted IDs."""
    if digest(selection_path) != SELECTION_SHA:
        raise RealCanaryContractError("MASK_NEXT_FROZEN_SELECTION_SHA_MISMATCH")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))["annotations"]
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))["tasks"]
    expected = {
        "chips010": selection["tasks"]["chips_role"]["failed_canary"]["baseline_terminal"]["session_id"],
        "poker015": selection["tasks"]["poker_object"]["failed_canary"]["baseline_terminal"]["session_id"],
    }
    if expected != {"chips010": "get_potato_chips_0901_010", "poker015": "play_cards_0901_015"}:
        raise RealCanaryContractError(f"unexpected frozen identities: {expected}")
    if annotations["chips010"]["session"] != expected["chips010"] or annotations["poker015"]["session"] != expected["poker015"]:
        raise RealCanaryContractError("annotation session identity mismatch")
    targets: list[dict[str, Any]] = []
    chips = annotations["chips010"]
    chips_total = _session_count(selection, "chips_role")
    for role, prompt in chips["roles"].items():
        accepted = int(bootstrap["chips010"]["roles"][role]["selected_object_id"])
        targets.append({
            "name": f"chips010_{role}", "session_id": expected["chips010"],
            "source_rgb": chips["source_rgb"]["path"], "seed_frame": int(chips["source_frame"]),
            "frame_count": chips_total, "point_xy": prompt["positive_point_xy"],
            "box_xyxy": prompt["box_xyxy"], "accepted_object_id": accepted,
            "text": "a person's hand and forearm", "quality_role": "human_role",
        })
    poker = annotations["poker015"]
    targets.append({
        "name": "poker015_task_object", "session_id": expected["poker015"],
        "source_rgb": poker["source_rgb"]["path"], "seed_frame": int(poker["source_frame"]),
        "frame_count": _session_count(selection, "poker_object"),
        "point_xy": poker["positive_point_xy"], "box_xyxy": poker["box_xyxy"],
        "accepted_object_id": int(bootstrap["poker015"]["object"]["selected_object_id"]),
        "text": "a purple-backed playing card", "quality_role": "task_object_instance",
    })
    for target in targets:
        if not 0 <= target["seed_frame"] < target["frame_count"]:
            raise RealCanaryContractError(f"seed outside session: {target['name']}")
    return targets


def prepare_lossless_sequence(target: dict[str, Any], destination: Path) -> Path:
    source = Path(target["source_rgb"]).resolve(strict=True)
    all_data = source.parents[1]
    destination.mkdir(parents=True, exist_ok=False)
    for frame in range(target["frame_count"]):
        rgb = all_data / f"{frame:05d}" / "rgb.png"
        if not rgb.is_file():
            raise RealCanaryContractError(f"missing lossless frame {frame}: {target['name']}")
        os.symlink(rgb, destination / f"{frame:05d}.png")
    return destination


def evaluate_target(target: dict[str, Any], rows: list[dict[str, Any]], seed_area: int) -> dict[str, Any]:
    expected = target["frame_count"] - target["seed_frame"]
    structural = {
        "exact_row_count": len(rows) == expected,
        "exact_source_range": bool(rows) and rows[0]["source_frame"] == target["seed_frame"] and rows[-1]["source_frame"] == target["frame_count"] - 1,
        "monotonic_unique": len({row["source_frame"] for row in rows}) == len(rows),
        "accepted_id_present_at_seed_source": bool(rows) and bool(rows[0]["present"]),
    }
    single = gate_single_instance(rows, seed_area=seed_area, maximum_components=4, maximum_area_multiple=4.0)
    present = sum(bool(row["present"]) for row in rows)
    presence_fraction = present / len(rows) if rows else 0.0
    # This is only an API/functionality smoke gate, not the contract's
    # visibility-conditioned 95% quality metric.  Without an independent
    # visible-frame reference that metric is NOT_MEASURABLE.  A stream that
    # emits only its seed frame must nevertheless fail rather than being
    # misreported as a successful temporal integration.
    minimum_functional_presence_fraction = 0.05
    development = {
        "accepted_id_presence_fraction": presence_fraction,
        "minimum_functional_presence_fraction": minimum_functional_presence_fraction,
        "functional_stream_presence_pass": presence_fraction >= minimum_functional_presence_fraction,
        "visibility_conditioned_presence_95pct": "NOT_MEASURABLE_NO_INDEPENDENT_VISIBLE_FRAME_REFERENCE",
        "missing_stream_rows": sum(row["reason"] == "NO_STREAM_OUTPUT" for row in rows),
        "single_instance_gate": single,
    }
    passed = all(structural.values()) and single["pass"] and development["functional_stream_presence_pass"]
    return {"pass": passed, "structural_gates": structural, "development_quality_gates": development}


def seal_terminal(output: Path, status: str, targets: list[dict[str, Any]], target_results: dict[str, Any], inputs: dict[str, Any], error: str | None, started: float) -> None:
    metrics = {
        "schema_version": "mask-sam31-temporal-real-canary-metrics-r22-v1",
        "status": status, "target_count": len(targets), "target_results": target_results,
        "wall_seconds": time.perf_counter() - started, "external_accuracy": "UNKNOWN",
        "authority_promoted": False,
    }
    write_json(output / "METRICS.json", metrics)
    receipt = {
        "schema_version": "mask-sam31-temporal-real-canary-run-r22-v1", "created_at": now(),
        "task_id": "mask_sam31_temporal_identity_real_canary_r22", "attempt_id": output.name,
        "status": status, "host": socket.gethostname(), "pid": os.getpid(), "inputs": inputs,
        "code": reference(Path(__file__)), "error": error, "current_ledger_modified": False,
    }
    write_json(output / "RUN_RECEIPT.json", receipt)
    write_text(output / "DECISION.md", f"# SAM3.1 真实因果时序 canary\n\n状态：`{status}`。\n\n只处理冻结的 0901 Chips010/Poker015；结果是开发质量证据，不是像素真值或 Mask authority。\n")
    next_action = {
        "schema_version": "chaoyang-next-action-v1", "status": status,
        "next_task_id": None if status == "FAILED_QUALITY_C" else "mask_sam31_temporal_identity_regression_r22",
        "regression_authorized": status == "PASSED_DEVELOPMENT",
        "claim_limit": "Regressions require both frozen failed canaries to pass all gates.",
    }
    write_json(output / "NEXT_ACTION.json", next_action)
    artifact_paths = [output / name for name in ("METRICS.json", "RUN_RECEIPT.json", "DECISION.md", "NEXT_ACTION.json")]
    write_json(output / "ARTIFACT_MANIFEST.json", {
        "schema_version": "chaoyang-artifact-manifest-v1", "status": status,
        "artifacts": [reference(path) for path in artifact_paths], "authority_promoted": False,
    })
    write_json(output / "RESULT.json", {
        "schema_version": "mask-sam31-temporal-real-canary-result-r22-v1", "created_at": now(),
        "task_id": "mask_sam31_temporal_identity_real_canary_r22", "attempt_id": output.name,
        "status": status, "metrics": reference(output / "METRICS.json"),
        "manifest": reference(output / "ARTIFACT_MANIFEST.json"), "error": error,
        "authority_promoted": False,
        "claim_limit": "Frozen SAM3.1 causal development canary only; external accuracy unknown.",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--bootstrap", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--stable-point-tracking",
        action="store_true",
        help="Round-2 bounded strategy: detector ID is frozen evidence; an explicit primed point/box ID is tracked.",
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status, error, adapter = "FAILED_RUNTIME_FINAL", None, None
    target_results: dict[str, Any] = {}
    paths = {"selection": args.selection.resolve(), "annotations": args.annotations.resolve(), "bootstrap": args.bootstrap.resolve(), "checkpoint": args.checkpoint.resolve()}
    inputs: dict[str, Any] = {}
    targets: list[dict[str, Any]] = []
    try:
        targets = bind_frozen_targets(paths["selection"], paths["annotations"], paths["bootstrap"])
        if digest(paths["checkpoint"]) != CHECKPOINT_SHA:
            raise RealCanaryContractError("SAM31_CHECKPOINT_SHA_MISMATCH")
        inputs = {name: reference(path) for name, path in paths.items()}
        adapter, _ = sam31_compat_adapter_v1.build_pinned_adapter(official_code_root=SAM_ROOT, checkpoint_path=paths["checkpoint"])
        for target in targets:
            frames = prepare_lossless_sequence(target, output / "input_frames" / target["name"])
            image = cv2.imread(target["source_rgb"], cv2.IMREAD_COLOR)
            if image is None:
                raise RealCanaryContractError(f"source decode failed: {target['name']}")
            height, width = image.shape[:2]
            spec = FrozenSeedReplay(
                session_id=f"{target['name']}-{output.name}", accepted_object_id=target["accepted_object_id"],
                point_xy=tuple(target["point_xy"]), box_xyxy=tuple(target["box_xyxy"]), text=target["text"],
                start_frame_index=target["seed_frame"], frame_count=target["frame_count"] - target["seed_frame"],
                source_frame_offset=target["seed_frame"], height=height, width=width,
            )
            if args.stable_point_tracking:
                seed_rows, rows, masks = replay_detector_then_track_stable_point_id(
                    adapter, frames, spec, sam_helpers.normalize,
                    tracking_object_id=10001 + int(target["accepted_object_id"]),
                )
            else:
                seed_rows, rows, masks = replay_seed_then_collect(adapter, frames, spec, sam_helpers.normalize)
            evidence = persist_compact_evidence(
                output / "evidence" / target["name"], seed_rows, rows, masks,
                lambda frame, root=Path(target["source_rgb"]).parents[1]: cv2.imread(str(root / f"{frame:05d}" / "rgb.png")),
            )
            evaluated_id = 10001 + int(target["accepted_object_id"]) if args.stable_point_tracking else target["accepted_object_id"]
            seed_area = max(row["area_pixels"] for row in seed_rows if row["object_id"] == evaluated_id)
            target_results[target["name"]] = {
                "spec": asdict(spec),
                "strategy": "STABLE_PRIMED_POINT_BOX_TRACKING" if args.stable_point_tracking else "LEGACY_DETECTOR_ID_TRACKING",
                "tracking_object_id": evaluated_id,
                "gates": evaluate_target(target, rows, seed_area),
                "evidence": evidence,
            }
        status = "PASSED_DEVELOPMENT" if all(row["gates"]["pass"] for row in target_results.values()) else "FAILED_QUALITY_C"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        status = "FAILED_RUNTIME_FINAL"
    finally:
        if adapter is not None:
            adapter.predictor.shutdown()
    seal_terminal(output, status, targets, target_results, inputs, error, started)
    print(json.dumps({"status": status, "result": reference(output / "RESULT.json")}, ensure_ascii=False))
    return 0 if status in {"PASSED_DEVELOPMENT", "FAILED_QUALITY_C"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
