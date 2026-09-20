#!/usr/bin/env python3
"""Run the fixed 044/097 W1-DIAG pair with one persistent HaWoR load.

This is an isolated-package operation for the registered V2.1 parent.  It does
not route through the predecessor W0 result and does not upgrade diagnostic or
structural windows to Kai22 R0 quality admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from chaoyang.ops import run_0915_robot15h_hawor_wave0_v1 as base


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_robot_quality_recovery_15h_v2"
WINDOW_RUN_ID = "0915-robot-quality-recovery-v21-20260919T234412+0800"
EXPECTED_LEDGER = {
    "path": str(ROOT / "_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/COHORT_ACCESS_LEDGER_V1.json"),
    "bytes": 23834,
    "sha256": "f08c34638b22fbfa4a66a3413915c90e77d667310e5341e71cbc860aa640f17f",
}
EXPECTED_INPUTS = {
    "play_cards_0915_044": {
        "frame_count": 166,
        "source_snapshot_sha256": "a0f5a6c672ed809facf65b16b4362875c7b188ce47a7dd131c404097ed161108",
        "source_stereo": {
            "bytes": 14556213,
            "sha256": "b48b40bb52a6c469e72e2c542658fe889e4baa8a909de5c52fd285a0b1990a23",
        },
        "camera_params": {
            "bytes": 3454,
            "sha256": "7f19d5ce92b5188a0013bd7ae85e3c031bc81303e54b5343605151bbdb777b0e",
        },
    },
    "get_potato_chips_0915_097": {
        "frame_count": 394,
        "source_snapshot_sha256": "891786766dd447206ecb38f035a83f6bf3ae4cfe85c09374992220ed173eb94c",
        "source_stereo": {
            "bytes": 31347559,
            "sha256": "642d0b6450d8eca29971a8724fd1382c9042ff0f1a6d7e310134aad11084b314",
        },
        "camera_params": {
            "bytes": 3454,
            "sha256": "a40d4d0a56c229ac24be9558e88ad3c0f47dc146fec7068b89cf1fb24aad4f41",
        },
    },
}
EXPECTED_ORDER = tuple(EXPECTED_INPUTS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def manifest_for_files(root: Path, paths: list[Path], destination: Path) -> dict[str, Any]:
    rows = []
    for path in paths:
        item = ref(path)
        item["relative_path"] = str(path.relative_to(root))
        rows.append(item)
    value = {
        "schema_version": "0915-ordered-file-manifest-v1",
        "root": str(root.resolve(strict=True)),
        "count": len(rows),
        "files": rows,
        "ordered_manifest_sha256": canonical_sha(rows),
    }
    atomic_json(destination, value)
    return ref(destination)


def verify_ref(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
        raise RuntimeError(f"{label} exact reference drift")


def maximal_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    ids = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not len(ids):
        return []
    chunks = np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1)
    return [(int(chunk[0]), int(chunk[-1])) for chunk in chunks if len(chunk)]


def structural_windows(npz_path: Path) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as archive:
        observed = np.asarray(archive["observed"], dtype=bool)
        joints_2d = np.asarray(archive["joints_2d"], dtype=np.float64)
        joints_3d = np.asarray(archive["joints_3d_camera"], dtype=np.float64)
        roots = np.asarray(archive["root_orient_camera"], dtype=np.float64)
        provenance = np.asarray(archive["provenance"]).astype(str)
    finite_2d = np.isfinite(joints_2d).all(axis=(2, 3))
    in_frame = (
        (joints_2d[..., 0] >= 0).all(axis=2)
        & (joints_2d[..., 0] < 1280).all(axis=2)
        & (joints_2d[..., 1] >= 0).all(axis=2)
        & (joints_2d[..., 1] < 960).all(axis=2)
    )
    finite_3d = np.isfinite(joints_3d).all(axis=(2, 3))
    positive_depth = (joints_3d[..., 2] > 0).all(axis=2)
    finite_root = np.isfinite(roots).all(axis=(2, 3))
    determinant = np.linalg.det(np.where(finite_root[..., None, None], roots, np.eye(3)))
    orthogonality = np.max(
        np.abs(np.transpose(roots, (0, 1, 3, 2)) @ roots - np.eye(3)), axis=(2, 3)
    )
    root_valid = finite_root & (determinant > 0) & (orthogonality <= 0.0001)
    provenance_valid = provenance == "OBSERVED"
    structural = observed & finite_2d & in_frame & finite_3d & positive_depth & root_valid & provenance_valid
    sides = {}
    for side, name in enumerate(("left", "right")):
        windows = [
            {
                "start_frame_inclusive": start,
                "end_frame_inclusive": end,
                "frame_count": end - start + 1,
                "hawor_camera_prior_admitted": True,
                "kai22_r0_evaluation_allowed": True,
                "kai22_r0_quality_admitted": "PENDING_R0_OWN_GATES",
                "minimum_window_gate": "PENDING_ORIGINAL_CONSUMER_CONTRACT_NO_NEW_THRESHOLD",
            }
            for start, end in maximal_runs(structural[side])
        ]
        sides[name] = {
            "full_timeline_frames": int(structural.shape[1]),
            "structural_admitted_frames": int(structural[side].sum()),
            "structural_rejected_frames": int((~structural[side]).sum()),
            "windows": windows,
        }
    return {"sides": sides, "structural_mask": structural}


def write_terminal(output: Path, status: str, **fields: Any) -> int:
    value = {
        "schema_version": "0915-robot-recovery-a1-w1-diag-result-v1",
        "status": status,
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "package_id": "A1",
        "claim_limit": "W1-DIAG HaWoR evidence and per-side structural consumer eligibility only; no R0 quality, adoption, control, training, or deployment authority.",
        **fields,
    }
    atomic_json(output / "RESULT.json", value)
    return 0 if status == "COMPLETED_DIAGNOSTIC_ALL_TERMINAL" else 3 if status == "BLOCKED_RESOURCE" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=61440)
    parser.add_argument("--gpu-wait-seconds", type=int, default=1800)
    parser.add_argument("--gpu-wall-seconds", type=int, default=3600)
    parser.add_argument("--executor-epoch", type=int, default=1)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh A1 output required: {output}")
    output.mkdir(parents=True)

    ledger_path = Path(EXPECTED_LEDGER["path"])
    verify_ref(ref(ledger_path), EXPECTED_LEDGER, "cohort access ledger")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    rows = [row for row in ledger["sessions"] if row.get("cohort_role") == "W1_DIAG"]
    by_id = {row["session_id"]: row for row in rows}
    if set(by_id) != set(EXPECTED_ORDER):
        return write_terminal(output, "BLOCKED_INPUT_FINAL", first_blocker="W1_DIAG_FIXED_IDS_DRIFT")
    rows = [by_id[session_id] for session_id in EXPECTED_ORDER]
    if any(row.get("access_state") != "OPEN_AT_T0" or row.get("replacement_allowed") is not False for row in rows):
        return write_terminal(output, "BLOCKED_INPUT_FINAL", first_blocker="W1_DIAG_ACCESS_NOT_OPEN_AND_FIXED")

    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != "0915-robot-recovery-a1-hawor-w1-diag-config-v1"
        or tuple(config.get("sessions", [])) != EXPECTED_ORDER
    ):
        return write_terminal(output, "BLOCKED_INPUT_FINAL", first_blocker="CONFIG_IDENTITY_DRIFT")
    for weight in config["weights"].values():
        verify_ref(ref(Path(weight["path"])), weight, "pinned HaWoR weight")

    prepared_rows = []
    for row in rows:
        expected = EXPECTED_INPUTS[row["session_id"]]
        if row["frame_count"] != expected["frame_count"] or row["source_snapshot_sha256"] != expected["source_snapshot_sha256"]:
            return write_terminal(output, "BLOCKED_INPUT_FINAL", first_blocker=f"{row['session_id']}:FROZEN_METADATA_DRIFT")
        verify_ref(ref(Path(row["source_stereo"]["path"])), expected["source_stereo"], f"{row['session_id']} source stereo")
        source_root = Path(row["processed_root"])
        verify_ref(ref(source_root / "camera_params.json"), expected["camera_params"], f"{row['session_id']} camera params")
        source_frames = sorted((source_root / "preprocess/all_data").glob("*/training_data.json"))
        if len(source_frames) != row["frame_count"]:
            return write_terminal(output, "BLOCKED_INPUT_FINAL", first_blocker=f"{row['session_id']}:SOURCE_FRAME_AXIS_DRIFT")
        source_manifest = manifest_for_files(
            source_root,
            source_frames,
            output / "input_manifests" / f"{row['session_id']}_SOURCE_FRAME_METADATA_MANIFEST.json",
        )
        prepared = base.prepare_session(row, output / "prepared" / row["session_id"])
        adapter_root = Path(prepared["adapter_session"])
        adapter_frames = sorted((adapter_root / "preprocess/all_data").glob("*/training_data.json"))
        adapter_manifest = manifest_for_files(
            adapter_root,
            adapter_frames,
            output / "input_manifests" / f"{row['session_id']}_ADAPTER_FRAME_METADATA_MANIFEST.json",
        )
        prepared.update(
            cohort_role="W1_DIAG",
            access_state=row["access_state"],
            source_snapshot_sha256=row["source_snapshot_sha256"],
            source_frame_metadata_manifest=source_manifest,
            adapter_frame_metadata_manifest=adapter_manifest,
        )
        prepared_rows.append(prepared)

    code_paths = [
        Path(__file__),
        ROOT / "src/chaoyang/ops/run_0915_robot15h_hawor_wave0_v1.py",
        ROOT / "src/chaoyang/ops/run_0915_hawor_w1_diag_persistent_worker_v1.py",
        ROOT / "src/chaoyang/ops/run_0915_hawor_resize_only_persistent_worker_v2.py",
        ROOT / "src/chaoyang/ops/run_0915_hawor_persistent_worker_v1.py",
        ROOT / "src/chaoyang/ops/run_play_cards_0910_001_hawor_raw.py",
        ROOT / "src/chaoyang/ops/run_0915_hawor_resize_only_canary_v1.py",
        ROOT / "src/chaoyang/ops/hawor_python.sh",
        ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py",
    ]
    candidate = {
        "candidate_id": config["candidate_id"],
        "cohort_role": "W1_DIAG",
        "session_inputs": [
            {
                "session_id": row["session_id"],
                "frame_count": row["frame_count"],
                "source_snapshot_sha256": row["source_snapshot_sha256"],
                "source_stereo": row["source_stereo"],
            }
            for row in rows
        ],
        "code": [ref(path) for path in code_paths],
        "config": ref(config_path),
        "weights": config["weights"],
        "input_domain": config["input_domain"],
        "numeric_thresholds_changed": False,
        "bounded_successor_policy": config["bounded_successor_policy"],
    }
    candidate_signature = canonical_sha(candidate)
    prepared_manifest = {
        "schema_version": "0915-hawor-resize-only-prepared-manifest-v2",
        "status": "PASS",
        "window_run_id": WINDOW_RUN_ID,
        "cohort_role": "W1_DIAG",
        "session_count": 2,
        "frame_count": 560,
        "source_access_ledger": ref(ledger_path),
        "candidate_signature_sha256": candidate_signature,
        "config": ref(config_path),
        "code": candidate["code"],
        "weights": config["weights"],
        "results": prepared_rows,
        "source_mutated": False,
        "processed_mutated": False,
    }
    prepared_path = output / "PREPARED_MANIFEST.json"
    atomic_json(prepared_path, prepared_manifest)
    atomic_json(output / "CANDIDATE_SIGNATURE.json", {**candidate, "candidate_signature_sha256": candidate_signature})

    worker_output = output / "hawor"
    gpu_receipt = output / "GPU_COMMAND_RECEIPT.json"
    worker_command = [
        str(ROOT / "src/chaoyang/ops/hawor_python.sh"),
        str(ROOT / "src/chaoyang/ops/run_0915_hawor_w1_diag_persistent_worker_v1.py"),
        "--prepared-manifest", str(prepared_path),
        "--output-root", str(worker_output),
    ]
    lease_command = [
        sys.executable,
        str(ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"),
        "--task-id", TASK_ID,
        "--attempt-id", "attempt_0001:A1:W1_DIAG",
        "--executor-epoch", str(args.executor_epoch),
        "--priority", "CANARY",
        "--gpu-id", str(args.gpu_id),
        "--min-free-mib", str(args.min_free_mib),
        "--wait-seconds", str(args.gpu_wait_seconds),
        "--wall-seconds", str(args.gpu_wall_seconds),
        "--receipt", str(gpu_receipt),
        "--claim-limit", "Fixed 044/097 W1-DIAG only; no automatic R0/adoption authority.",
        "--",
        *worker_command,
    ]
    atomic_json(output / "COMMAND.json", {
        "schema_version": "0915-robot-recovery-a1-w1-diag-command-v1",
        "candidate_signature_sha256": candidate_signature,
        "worker_command": worker_command,
        "lease_command": lease_command,
    })
    completed = subprocess.run(lease_command, cwd=ROOT, check=False)
    gpu = json.loads(gpu_receipt.read_text(encoding="utf-8")) if gpu_receipt.is_file() else {}
    if completed.returncode or gpu.get("status") != "PASSED":
        terminal = "BLOCKED_RESOURCE" if gpu.get("status") == "BLOCKED_RESOURCE" else "FAILED_RUNTIME_FINAL"
        return write_terminal(
            output,
            terminal,
            first_blocker=gpu.get("reason", "HAWOR_W1_DIAG_RUNTIME"),
            candidate_signature_sha256=candidate_signature,
            gpu_command_receipt=ref(gpu_receipt) if gpu_receipt.is_file() else None,
        )

    batch_path = worker_output / "BATCH_RESULT.json"
    if not batch_path.is_file():
        return write_terminal(output, "FAILED_EVIDENCE_FINAL", first_blocker="MISSING_WORKER_BATCH")
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    if batch.get("session_count") != 2 or batch.get("failed_runtime") != 0:
        return write_terminal(output, "FAILED_RUNTIME_FINAL", first_blocker="WORKER_BATCH_NOT_ALL_RUNTIME_TERMINAL")

    admissions = []
    window_rows = []
    for item in batch["results"]:
        npz_path = Path(item["npz"]["path"])
        windows = structural_windows(npz_path)
        side_rows = []
        for side in ("left", "right"):
            strict = item["per_side_strict"][side]
            side_row = {
                "side": side,
                "hawor_side_strict": strict,
                "hawor_session_strict": item["session_strict_status"],
                "diagnostic_review_admitted": True,
                "r0_evaluation_window_count": len(windows["sides"][side]["windows"]),
                "r0_structural_eligible_frames": windows["sides"][side]["structural_admitted_frames"],
                "r0_quality_admitted": "PENDING_R0_OWN_GATES",
                "world_wrist_admitted": False,
                "world_wrist_blocker": "TRUSTED_C2W_OR_SLAM_CONTRACT_NOT_BOUND",
            }
            side_rows.append(side_row)
            window_rows.append({"session_id": item["session_id"], "side": side, **windows["sides"][side]})
        admissions.append({
            "session_id": item["session_id"],
            "runtime_terminal": item["status"],
            "npz": item["npz"],
            "sides": side_rows,
        })

    admission_doc = {
        "schema_version": "0915-robot-recovery-consumer-admission-v1",
        "status": "SHADOW_COMPLETE_NO_R0_QUALITY_UPGRADE",
        "candidate_signature_sha256": candidate_signature,
        "whole_w0_strict_pass_required": False,
        "numeric_thresholds_changed": False,
        "sessions": admissions,
    }
    windows_doc = {
        "schema_version": "0915-robot-recovery-window-validity-v1",
        "status": "STRUCTURAL_WINDOWS_COMPLETE_R0_QUALITY_PENDING",
        "candidate_signature_sha256": candidate_signature,
        "minimum_window_policy": "NO_NEW_THRESHOLD; FINAL_QUALITY_REQUIRES_ORIGINAL_CONSUMER_CONTRACT_AND_R0_OWN_GATES",
        "rows": window_rows,
    }
    admission_path = output / "CONSUMER_ADMISSION_V1.json"
    windows_path = output / "WINDOW_VALIDITY_V1.json"
    atomic_json(admission_path, admission_doc)
    atomic_json(windows_path, windows_doc)

    adoption_rows = [row for row in ledger["sessions"] if row.get("cohort_role") == "W1_ADOPTION"]
    adoption_still_sealed = len(adoption_rows) == 2 and all(row.get("access_state") != "OPEN_AT_T0" for row in adoption_rows)
    freeze_path = output / "CANDIDATE_FREEZE_V1.json"
    atomic_json(freeze_path, {
        "schema_version": "0915-robot-recovery-candidate-freeze-v1",
        "status": "FROZEN_BEFORE_W1_ADOPTION_OPEN" if adoption_still_sealed else "BLOCKED_ADOPTION_ALREADY_OPEN",
        "candidate_id": config["candidate_id"],
        "candidate_signature_sha256": candidate_signature,
        "code": candidate["code"],
        "config": candidate["config"],
        "weights": candidate["weights"],
        "input_domain": candidate["input_domain"],
        "numeric_thresholds_changed": False,
        "bounded_successor_policy": candidate["bounded_successor_policy"],
        "w1_diag_evidence": ref(batch_path),
        "w1_adoption_access_state_at_freeze": [
            {"session_id": row["session_id"], "access_state": row["access_state"]}
            for row in adoption_rows
        ],
        "result_based_candidate_change_allowed": False,
        "same_signature_runtime_retry_allowed": True,
        "runtime_attempts_per_signature": 1,
    })
    if not adoption_still_sealed:
        return write_terminal(output, "FAILED_EVIDENCE_FINAL", first_blocker="CANDIDATE_NOT_FROZEN_BEFORE_ADOPTION_OPEN")
    return write_terminal(
        output,
        "COMPLETED_DIAGNOSTIC_ALL_TERMINAL",
        candidate_signature_sha256=candidate_signature,
        counts={
            "attempted": 2,
            "runtime_terminal": 2,
            "session_strict_pass": batch["passed"],
            "session_strict_rejected": batch["quality_c"],
            "failed_runtime": 0,
        },
        prepared_manifest=ref(prepared_path),
        gpu_command_receipt=ref(gpu_receipt),
        worker_batch=ref(batch_path),
        consumer_admission=ref(admission_path),
        window_validity=ref(windows_path),
        candidate_freeze=ref(freeze_path),
        source_mutated=False,
        processed_mutated=False,
        governance_mutated=False,
        training_eligible=False,
        physical_deployment_authorized=False,
    )


if __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException as exc:  # noqa: BLE001 - immutable terminal on every failure
        try:
            index = sys.argv.index("--output-root")
            failed_output = Path(sys.argv[index + 1]).resolve()
            if failed_output.is_dir() and not (failed_output / "RESULT.json").exists():
                message = repr(exc)
                input_failure = any(token in message.lower() for token in (
                    "drift", "identity", "access", "manifest", "frame axis", "config", "weight",
                ))
                write_terminal(
                    failed_output,
                    "BLOCKED_INPUT_FINAL" if input_failure else "FAILED_RUNTIME_FINAL",
                    first_blocker="UNCAUGHT_PREFLIGHT_OR_RUNTIME_EXCEPTION",
                    error=message,
                )
        finally:
            exit_code = 2
    raise SystemExit(exit_code)
