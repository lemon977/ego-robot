#!/usr/bin/env python3
"""Finite post-Robot causal Visual Aux bundle and training automation.

The two task pairs close independently.  Only frozen train/validation splits
are consumed; future donors and ProPainter pixels are excluded by the bundle
builder.  Missing data closes as BLOCKED_DATA_VOLUME and never creates a fake
checkpoint.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chaoyang.human_ego.tools.validate_visual_aux_bundle_v52 import validate_bundle  # noqa: E402
from chaoyang.human_ego.tools.train_visual_aux_future2d_v53 import (  # noqa: E402
    pair_dataset_signature,
    validate_paired_ledgers,
    validate_training_result,
)

ROBOT_STATE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/visual_tier_robot_R7_2/AUTOMATION_STATE.json"
MATRIX = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/visual_tier_robot_R7_2/combined_eligibility/CURRENT_DRAFT_EXACT78_CROSS_STAGE_MATRIX.json"
ELIGIBILITY = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/visual_tier_robot_R7_2/combined_eligibility/ELIGIBILITY_INDEX.json"
DEFAULT_OUTPUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/visual_aux/causal_training_R7_2"
STATUS_RECEIPT = ROOT / "docs/governance/CURRENT_STATUS_RECEIPT.json"
MINIMUM = {"train": {"sessions": 16, "windows": 256}, "validation": {"sessions": 3, "windows": 48}}
BUILDER = ROOT / "src/chaoyang/human_ego/tools/build_visual_aux_session_bundle_v54.py"
MIN_VALID_PIXEL_FRACTION = 0.70


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def exact(item: dict[str, Any], label: str) -> Path:
    path = Path(str(item["path"])).resolve(strict=True)
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        raise RuntimeError(f"{label}: bytes/SHA mismatch")
    return path


def exact_from(root: Path, item: dict[str, Any], label: str) -> Path:
    raw = Path(str(item.get("path", "")))
    path = raw if raw.is_absolute() else root / raw
    return exact({**item, "path": str(path)}, label)


def same_ref(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("path", "bytes", "sha256"))


def expected_bundle_signature(
    *,
    robot_ref: dict[str, Any],
    clean_ref: dict[str, Any],
    silver_ref: dict[str, Any],
    compositor_ref: dict[str, Any],
    split: str,
    h50_eligibility_mode: str,
    minimum_valid_pixel_fraction: float,
) -> dict[str, Any]:
    payload = {
        "schema_version": "VISUAL_AUX_BUNDLE_PRODUCER_SIGNATURE_V1",
        "builder_code": ref(BUILDER),
        "robot_result": robot_ref,
        "clean_result": clean_ref,
        "occlusion_silver_result": silver_ref,
        "compositor_result": compositor_ref,
        "split": split,
        "seed": 7,
        "h50_eligibility_mode": h50_eligibility_mode,
        "minimum_valid_pixel_fraction": minimum_valid_pixel_fraction,
    }
    return {"payload": payload, "sha256": canonical_sha256(payload)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_logged(command: list[str], log: Path, timeout: int) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            handle.write(f"\nAUTOMATION_TIMEOUT_SECONDS={timeout}\n")
            return 124


def manifest_from_bundle(
    bundle: Path,
    *,
    expected_signature: dict[str, Any],
    expected_inputs: dict[str, dict[str, Any]],
) -> tuple[Path, int] | None:
    result_path = bundle / "RESULT.json"
    manifest = bundle / "VISUAL_AUX_SESSION_MANIFEST.json"
    if not result_path.is_file() or not manifest.is_file():
        return None
    result = load(result_path)
    if result.get("status") != "PASS_VISUAL_AUX_BUNDLE_SILVER_BOUND":
        return None
    if result.get("producer_signature") != expected_signature:
        raise RuntimeError(f"stale bundle producer signature: {bundle}")
    manifest_ref = result.get("manifest")
    if not isinstance(manifest_ref, dict) or exact_from(bundle, manifest_ref, "bundle manifest") != manifest:
        raise RuntimeError(f"bundle result does not bind its exact manifest: {bundle}")
    payload = load(manifest)
    if payload.get("input_mode") != "CAUSAL_TRAINING_INPUT" or payload.get("causal_proof", {}).get("enabled") is not True:
        raise RuntimeError(f"non-causal bundle found: {bundle}")
    if payload.get("producer_signature") != expected_signature:
        raise RuntimeError(f"bundle manifest producer signature mismatch: {bundle}")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError(f"bundle input bindings missing: {bundle}")
    for key, expected in expected_inputs.items():
        actual = inputs.get(key)
        if not isinstance(actual, dict) or not same_ref(actual, expected):
            raise RuntimeError(f"bundle exact input binding mismatch: {key}")
        exact(actual, f"bundle.{key}")
    report = validate_bundle(bundle)
    gate = payload.get("valid_pixel_gate", {})
    if (
        gate.get("eligible_frames_meet_minimum") is not True
        or float(gate.get("minimum_fraction", -1)) < MIN_VALID_PIXEL_FRACTION
        or float(report["rgb_valid_pixel_fraction"]) < float(gate["minimum_fraction"])
    ):
        raise RuntimeError(f"bundle valid-pixel gate failed: {bundle}")
    if int(result.get("eligible_h50_windows", -1)) != int(report["eligible_h50_window_count"]):
        raise RuntimeError(f"bundle window count mismatch: {bundle}")
    return manifest, int(report["eligible_h50_window_count"])


def build_bundles(output: Path, bundle_timeout: int) -> dict[str, Any]:
    matrix = load(MATRIX)
    index = load(ELIGIBILITY)
    matrix_rows = {row["session_id"]: row for row in matrix["rows"]}
    candidates = sorted(
        (
            row for row in index["rows"]
            if row.get("status") == "READY" and row.get("split") in MINIMUM
        ),
        key=lambda row: (row["task"], row["split"], -int(row["eligible_h50_window_count"]), row["session"]),
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        session = candidate["session"]
        source = matrix_rows.get(session, {})
        clean_ref, robot_ref = source.get("clean_result"), source.get("robot_candidate_result")
        silver_ref = source.get("occlusion_silver_result")
        compositor_ref = source.get("causal_compositor_result")
        if (
            source.get("clean_state") != "PASSED_GRADE_B"
            or not all(
                isinstance(item, dict)
                for item in (clean_ref, robot_ref, silver_ref, compositor_ref)
            )
        ):
            rows.append(
                {
                    **candidate,
                    "bundle_status": "BLOCKED_PREREQ_MISSING_CLEAN_ROBOT_SILVER_OR_COMPOSITOR",
                }
            )
            continue
        clean_path = exact(clean_ref, f"{session}.clean")
        robot_path = exact(robot_ref, f"{session}.robot")
        silver_path = exact(silver_ref, f"{session}.occlusion_silver")
        compositor_path = exact(compositor_ref, f"{session}.causal_compositor")
        exact_inputs = {
            "robot_result": ref(robot_path),
            "clean_result": ref(clean_path),
            "occlusion_silver_result": ref(silver_path),
            "compositor_result": ref(compositor_path),
        }
        signature = expected_bundle_signature(
            robot_ref=exact_inputs["robot_result"],
            clean_ref=exact_inputs["clean_result"],
            silver_ref=exact_inputs["occlusion_silver_result"],
            compositor_ref=exact_inputs["compositor_result"],
            split=candidate["split"],
            h50_eligibility_mode="BOTH_ENDPOINTS_40_OF_50",
            minimum_valid_pixel_fraction=MIN_VALID_PIXEL_FRACTION,
        )
        bundle = output / "bundles" / session
        try:
            adopted = manifest_from_bundle(
                bundle, expected_signature=signature, expected_inputs=exact_inputs
            )
        except (RuntimeError, OSError, ValueError, KeyError) as error:
            rows.append(
                {
                    **candidate,
                    "bundle_status": "BLOCKED_PREREQ_STALE_OR_INVALID_BUNDLE",
                    "reason": str(error),
                }
            )
            continue
        if adopted is None and not bundle.exists():
            log = output / "logs" / f"bundle_{session}.log"
            rc = run_logged(
                [
                    sys.executable, str(BUILDER),
                    "--robot-review-result", str(robot_path), "--clean-result", str(clean_path),
                    "--occlusion-silver-result", str(silver_path),
                    "--compositor-result", str(compositor_path),
                    "--split", candidate["split"], "--output-root", str(output / "bundles"),
                    "--causal-training",
                    "--h50-eligibility-mode", "BOTH_ENDPOINTS_40_OF_50",
                    "--min-valid-pixel-fraction", str(MIN_VALID_PIXEL_FRACTION),
                ],
                log, bundle_timeout,
            )
            adopted = manifest_from_bundle(
                bundle, expected_signature=signature, expected_inputs=exact_inputs
            )
        else:
            rc = 0
        if adopted is None:
            rows.append({**candidate, "bundle_status": "FAILED_OR_ZERO_WINDOWS", "return_code": rc})
            continue
        manifest, windows = adopted
        rows.append({
            **candidate, "bundle_status": "READY", "eligible_h50_window_count": windows,
            "manifest": ref(manifest), "return_code": rc,
        })
    return {"rows": rows, "source_matrix": ref(MATRIX), "source_eligibility": ref(ELIGIBILITY)}


def materialize_hardsets(source_path: Path, output: Path) -> dict[str, Path]:
    source_path = source_path.resolve(strict=True)
    source = load(source_path)
    if (
        source.get("schema_version") != "VISUAL_AUX_OCCLUSION_HARDSET_INDEX_V1"
        or source.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise RuntimeError("frozen Visual Aux occlusion hardset index required")
    tasks = source.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != {"chips", "poker"}:
        raise RuntimeError("hardset index must contain exactly chips and poker")
    outputs: dict[str, Path] = {}
    for task in ("chips", "poker"):
        task_row = tasks[task]
        if not isinstance(task_row, dict) or not isinstance(task_row.get("sessions"), list):
            raise RuntimeError(f"invalid frozen hardset task row: {task}")
        payload = {
            "schema_version": "VISUAL_AUX_OCCLUSION_HARDSET_V1",
            "status": "FROZEN_BEFORE_TRAINING",
            "task": task,
            "split": "validation",
            "sessions": task_row["sessions"],
            "source_index": ref(source_path),
            "claim_limit": "Frozen occlusion-difficult validation windows for paired Visual Aux engineering comparison.",
        }
        path = output / "hardsets" / f"{task}_OCCLUSION_HARDSET.json"
        if path.exists():
            if load(path) != payload:
                raise RuntimeError(f"stale frozen hardset output: {path}")
        else:
            atomic_json(path, payload)
        outputs[task] = path
    return outputs


def make_ledgers(
    output: Path, built: dict[str, Any], hardsets: dict[str, Path]
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for task in ("chips", "poker"):
        groups: dict[str, list[dict[str, Any]]] = {}
        counts = {}
        for split, requirement in MINIMUM.items():
            selected = [
                row for row in built["rows"]
                if row["task"] == task and row["split"] == split and row.get("bundle_status") == "READY"
            ]
            groups[split] = [row["manifest"] for row in selected]
            counts[split] = {
                "sessions": len(selected),
                "windows": sum(int(row["eligible_h50_window_count"]) for row in selected),
                "required": requirement,
            }
        ready = all(
            counts[split]["sessions"] >= requirement["sessions"]
            and counts[split]["windows"] >= requirement["windows"]
            for split, requirement in MINIMUM.items()
        )
        ledgers = {}
        if ready:
            common = {
                "schema_version": "exact78-visual-aux-dataset-ledger-v53-v1",
                "status": "PASS_VISUAL_AUX_DATASET_LEDGER",
                "task": task,
                "seed": 7,
                "train": groups["train"],
                "validation": groups["validation"],
                "control_ground_truth": False,
                "input_mode": "CAUSAL_TRAINING_INPUT",
                "occlusion_hardset": ref(hardsets[task]),
                "claim_limit": "Future-2D Visual Aux only; no real Robot action, policy or deployment authority.",
            }
            common["pair_dataset_signature"] = pair_dataset_signature(common)
            ledger_paths: dict[str, Path] = {}
            for branch in ("HUMAN_RAW_RGB", "ROBOTIZED_RGB"):
                ledger_path = output / "ledgers" / task / f"{branch}_DATASET_LEDGER.json"
                payload = {**common, "branch": branch}
                if ledger_path.exists():
                    if load(ledger_path) != payload:
                        raise RuntimeError(f"existing Visual Aux ledger signature/content mismatch: {ledger_path}")
                else:
                    atomic_json(ledger_path, payload)
                ledgers[branch] = ref(ledger_path)
                ledger_paths[branch] = ledger_path
            try:
                validate_paired_ledgers(
                    ledger_paths["HUMAN_RAW_RGB"], ledger_paths["ROBOTIZED_RGB"]
                )
            except (RuntimeError, OSError, ValueError, KeyError) as error:
                summary[task] = {
                    "status": "BLOCKED_PREREQ_OCCLUSION_HARDSET_OR_PAIR_CONTRACT",
                    "counts": counts,
                    "ledgers": {},
                    "reason": str(error),
                }
                continue
        summary[task] = {
            "status": "READY_FOR_EPOCH0" if ready else "BLOCKED_DATA_VOLUME",
            "counts": counts, "ledgers": ledgers,
        }
    return summary


def gpu_stage(
    task: str,
    branch: str,
    stage: str,
    ledger: Path,
    paired_ledger: Path,
    output: Path,
    epoch0: bool,
) -> tuple[str, Path | None]:
    stage_root = output / "training" / task / branch / stage
    for attempt in range(1, 4):
        attempt_root = stage_root / f"attempt_{attempt:04d}"
        result_name = "EPOCH0_RESULT.json" if epoch0 else "RESULT.json"
        result_path = attempt_root / result_name
        if result_path.is_file():
            try:
                validate_training_result(
                    result_path,
                    ledger_path=ledger,
                    paired_ledger_path=paired_ledger,
                    epoch0=epoch0,
                )
            except (RuntimeError, OSError, ValueError, KeyError):
                continue
            return "PASSED", result_path
        if attempt_root.exists():
            continue
        receipt = output / "gpu_receipts" / f"{task}_{branch}_{stage}_attempt_{attempt:04d}.json"
        command = [
            sys.executable, "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
            "--task-id", f"visual_aux_{task}_pair_v1",
            "--attempt-id", f"{branch.lower()}_{stage}_{attempt:04d}",
            "--priority", "CHECKPOINT_TRAINING", "--wait-seconds", "1800",
            "--wall-seconds", "43200", "--receipt", str(receipt),
            "--claim-limit", "Single-seed future-2D Visual Aux; not policy/action truth.",
            "--", sys.executable, "src/chaoyang/human_ego/tools/train_visual_aux_future2d_v53.py",
            "--ledger", str(ledger), "--paired-ledger", str(paired_ledger),
            "--output-root", str(attempt_root),
            "--device", "cuda", "--epochs", "180", "--validation-frequency", "5",
            "--patience-validations", "12", "--batch-size", "16", "--workers", "2",
        ]
        if epoch0:
            command.append("--epoch0")
        rc = run_logged(command, output / "logs" / f"train_{task}_{branch}_{stage}_{attempt:04d}.log", 45000)
        if result_path.is_file() and rc == 0:
            try:
                validate_training_result(
                    result_path,
                    ledger_path=ledger,
                    paired_ledger_path=paired_ledger,
                    epoch0=epoch0,
                )
            except (RuntimeError, OSError, ValueError, KeyError):
                continue
            return "PASSED", result_path
        gpu_receipt = load(receipt) if receipt.is_file() else {}
        if gpu_receipt.get("status") == "BLOCKED_RESOURCE":
            return "BLOCKED_RESOURCE", receipt
    return "FAILED_RUNTIME_FINAL", None


def visual_aux_value_gate(raw: dict[str, Any], robot: dict[str, Any]) -> dict[str, Any]:
    """Evaluate only the pre-frozen hardset and full-validation pair contract."""

    if raw.get("occlusion_hardset") != robot.get("occlusion_hardset"):
        raise RuntimeError("Raw/Robotized results do not bind the same frozen hardset")
    pair_signature = raw.get("pair_dataset_signature")
    if (
        not isinstance(pair_signature, str)
        or len(pair_signature) != 64
        or robot.get("pair_dataset_signature") != pair_signature
    ):
        raise RuntimeError("Raw/Robotized results do not bind the same paired dataset")
    raw_full_ade = float(raw["best_validation_metrics"]["ADE_2D_px"])
    robot_full_ade = float(robot["best_validation_metrics"]["ADE_2D_px"])
    raw_hard_ade = float(raw["best_occlusion_hardset_metrics"]["ADE_2D_px"])
    robot_hard_ade = float(robot["best_occlusion_hardset_metrics"]["ADE_2D_px"])
    raw_hard_fde = float(raw["best_occlusion_hardset_metrics"]["FDE_2D_px"])
    robot_hard_fde = float(robot["best_occlusion_hardset_metrics"]["FDE_2D_px"])
    values = (
        raw_full_ade,
        robot_full_ade,
        raw_hard_ade,
        robot_hard_ade,
        raw_hard_fde,
        robot_hard_fde,
    )
    if not all(value >= 0 and value < float("inf") for value in values):
        raise RuntimeError("Value Gate metrics must be finite and non-negative")
    hard_ade_relative = (raw_hard_ade - robot_hard_ade) / max(raw_hard_ade, 1e-12)
    hard_fde_relative = (raw_hard_fde - robot_hard_fde) / max(raw_hard_fde, 1e-12)
    full_ade_degradation = (robot_full_ade - raw_full_ade) / max(raw_full_ade, 1e-12)
    value_pass = (
        raw_full_ade > 0
        and raw_hard_ade > 0
        and raw_hard_fde > 0
        and hard_ade_relative >= 0.05
        and hard_fde_relative >= 0.05
        and full_ade_degradation <= 0.02
    )
    return {
        "status": "PASS_FROZEN_HARDSET_AND_FULL_VALIDATION" if value_pass else "NO_GO_DIAGNOSTIC_ONLY",
        "frozen_occlusion_hardset": raw["occlusion_hardset"],
        "pair_dataset_signature": pair_signature,
        "hardset_raw_ADE_2D_px": raw_hard_ade,
        "hardset_robotized_ADE_2D_px": robot_hard_ade,
        "hardset_ADE_relative_improvement": hard_ade_relative,
        "hardset_raw_FDE_2D_px": raw_hard_fde,
        "hardset_robotized_FDE_2D_px": robot_hard_fde,
        "hardset_FDE_relative_improvement": hard_fde_relative,
        "full_validation_raw_ADE_2D_px": raw_full_ade,
        "full_validation_robotized_ADE_2D_px": robot_full_ade,
        "full_validation_ADE_relative_degradation": full_ade_degradation,
        "gates": {
            "hardset_ADE_improvement_ge_0p05": hard_ade_relative >= 0.05,
            "hardset_FDE_improvement_ge_0p05": hard_fde_relative >= 0.05,
            "full_validation_ADE_degradation_le_0p02": full_ade_degradation <= 0.02,
        },
        "claim_limit": "Single-seed engineering comparison; not statistical significance.",
    }


def train_pair(task: str, eligibility: dict[str, Any], output: Path) -> dict[str, Any]:
    if eligibility["status"] != "READY_FOR_EPOCH0":
        return {"task": task, "status": "BLOCKED_DATA_VOLUME", "eligibility": eligibility}
    results: dict[str, Any] = {}
    for branch, ledger_ref in eligibility["ledgers"].items():
        ledger = exact(ledger_ref, f"{task}.{branch}.ledger")
        paired_branch = "ROBOTIZED_RGB" if branch == "HUMAN_RAW_RGB" else "HUMAN_RAW_RGB"
        paired_ledger = exact(
            eligibility["ledgers"][paired_branch], f"{task}.{paired_branch}.ledger"
        )
        epoch_status, epoch_result = gpu_stage(
            task, branch, "epoch0", ledger, paired_ledger, output, True
        )
        results[branch] = {"epoch0_status": epoch_status, "epoch0_result": ref(epoch_result) if epoch_result and epoch_result.is_file() else None}
        if epoch_status != "PASSED":
            return {"task": task, "status": epoch_status, "branches": results, "eligibility": eligibility}
    for branch, ledger_ref in eligibility["ledgers"].items():
        ledger = exact(ledger_ref, f"{task}.{branch}.ledger")
        paired_branch = "ROBOTIZED_RGB" if branch == "HUMAN_RAW_RGB" else "HUMAN_RAW_RGB"
        paired_ledger = exact(
            eligibility["ledgers"][paired_branch], f"{task}.{paired_branch}.ledger"
        )
        train_status, train_result = gpu_stage(
            task, branch, "full", ledger, paired_ledger, output, False
        )
        results[branch].update(full_status=train_status, full_result=ref(train_result) if train_result and train_result.is_file() else None)
        if train_status != "PASSED":
            return {"task": task, "status": train_status, "branches": results, "eligibility": eligibility}
    raw = load(Path(results["HUMAN_RAW_RGB"]["full_result"]["path"]))
    robot = load(Path(results["ROBOTIZED_RGB"]["full_result"]["path"]))
    return {
        "task": task, "status": "PASS_CHECKPOINT_PAIR",
        "branches": results, "eligibility": eligibility,
        "value_gate": visual_aux_value_gate(raw, robot),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--robot-state", type=Path, default=ROBOT_STATE)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=604800)
    parser.add_argument("--bundle-timeout-seconds", type=int, default=3600)
    parser.add_argument(
        "--occlusion-hardset-index",
        type=Path,
        help="Frozen pre-training Chips/Poker occlusion-hard validation window index.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output_root.resolve()
    state_path = output / "AUTOMATION_STATE.json"
    state = {
        "schema_version": "v71-post-robot-visual-aux-automation-v1",
        "status": "DRY_RUN_READY" if args.dry_run else "WAITING_ROBOT_EXPANSION",
        "pid": os.getpid(), "robot_state": str(args.robot_state.resolve()),
        "claim_limit": "Causal future-2D Visual Aux only; no real action, policy, Contact, physical or deployment authority.",
    }
    atomic_json(state_path, state)
    if args.dry_run:
        print(json.dumps(state, ensure_ascii=False))
        return 0
    if args.occlusion_hardset_index is None or not args.occlusion_hardset_index.is_file():
        state.update(
            status="BLOCKED_PREREQ",
            reason="FROZEN_OCCLUSION_HARDSET_INDEX_REQUIRED_BEFORE_TRAINING",
        )
        atomic_json(state_path, state)
        return 3
    hardsets = materialize_hardsets(args.occlusion_hardset_index, output)
    started = time.monotonic()
    while True:
        upstream = load(args.robot_state) if args.robot_state.is_file() else {}
        if upstream.get("status") == "TERMINAL":
            break
        if time.monotonic() - started > args.max_wait_seconds:
            state.update(status="BLOCKED_RESOURCE", reason="ROBOT_EXPANSION_WAIT_BUDGET_EXHAUSTED")
            atomic_json(state_path, state)
            return 3
        state.update(updated_at=datetime.now().astimezone().isoformat(timespec="seconds"), upstream_status=upstream.get("status", "MISSING"))
        atomic_json(state_path, state)
        time.sleep(max(5, args.poll_seconds))

    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < 10 * (1 << 30):
        state.update(status="BLOCKED_RESOURCE", reason="LESS_THAN_10_GIB_FREE", free_bytes=free_bytes)
        atomic_json(state_path, state)
        return 3
    built = build_bundles(output, args.bundle_timeout_seconds)
    preflight_path = output / "CAUSAL_DATASET_PREFLIGHT.json"
    eligibility = make_ledgers(output, built, hardsets)
    atomic_json(preflight_path, {
        "schema_version": "v71-causal-visual-aux-dataset-preflight-v1",
        "status": "PASSED_WITH_PER_TASK_TERMINALS", "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **built, "task_eligibility": eligibility, "control_ground_truth": False,
        "claim_limit": state["claim_limit"],
    })
    pair_results = [train_pair(task, eligibility[task], output) for task in ("chips", "poker")]
    pair_paths = []
    aggregation_requests = []
    for pair in pair_results:
        path = output / pair["task"] / "CHECKPOINT_PAIR_RESULT.json"
        atomic_json(path, {
            "schema_version": "v71-visual-aux-checkpoint-pair-result-v1",
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            **pair, "control_ground_truth": False, "policy_checkpoint": False,
            "claim_limit": state["claim_limit"],
        })
        pair_paths.append(path)
        status = "PASSED" if pair["status"] == "PASS_CHECKPOINT_PAIR" else (
            "FAILED_RUNTIME_FINAL" if pair["status"] == "FAILED_RUNTIME_FINAL" else
            "BLOCKED_RESOURCE" if pair["status"] == "BLOCKED_RESOURCE" else "BLOCKED_PREREQ"
        )
        aggregation_requests.append(
            {
                "task_id": f"visual_aux_{pair['task']}_pair_v1",
                "requested_status": status,
                "phase": pair["status"].lower(),
                "result": ref(path),
                "worker_updated_current_governance": False,
            }
        )
    aggregation_path = output / "AGGREGATOR_UPDATE_REQUESTS.json"
    atomic_json(
        aggregation_path,
        {
            "schema_version": "VISUAL_AUX_AGGREGATOR_UPDATE_REQUESTS_V1",
            "status": "PENDING_SINGLE_GOVERNANCE_AGGREGATOR",
            "requests": aggregation_requests,
            "claim_limit": "Worker receipts only; this file does not mutate current governance.",
        },
    )
    index_path = output / "VISUAL_AUX_CHECKPOINT_INDEX.json"
    atomic_json(index_path, {
        "schema_version": "v71-visual-aux-checkpoint-index-v1",
        "status": "FULL_PASS" if all(row["status"] == "PASS_CHECKPOINT_PAIR" for row in pair_results) else (
            "PARTIAL_PASS" if any(row["status"] == "PASS_CHECKPOINT_PAIR" for row in pair_results) else "FULL_BLOCKED"
        ),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pairs": [ref(path) for path in pair_paths], "dataset_preflight": ref(preflight_path),
        "aggregator_update_requests": ref(aggregation_path),
        "control_ground_truth": False, "policy_checkpoint": False, "claim_limit": state["claim_limit"],
    })
    state.update(status="TERMINAL", result=ref(index_path))
    atomic_json(state_path, state)
    print(json.dumps(load(index_path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
