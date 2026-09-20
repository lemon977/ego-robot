#!/usr/bin/env python3
"""Run frozen A1 candidate on fixed 106/029 W1-ADOPTION evidence.

The candidate remains the exact A1 freeze.  This operation first emits a
role-only amendment and a local access-open event, then reads the sealed
W1-ADOPTION inputs.  It never promotes HaWoR evidence to Kai22 R0 quality.
"""

from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath

if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from chaoyang.ops import run_0915_robot_recovery_hawor_w1_diag_v1 as a1


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "0915_robot_quality_recovery_15h_v2"
WINDOW_RUN_ID = "0915-robot-quality-recovery-v21-20260919T234412+0800"
PACKAGE_ID = "A2"
PARENT_CANDIDATE_SIGNATURE = "9ccf7ff5eef43158848f73206a772ae756c0387cd84de9ecc72a2ee678db781b"
EXPECTED_LEDGER = {
    "path": str(ROOT / "_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/COHORT_ACCESS_LEDGER_V1.json"),
    "bytes": 23834,
    "sha256": "f08c34638b22fbfa4a66a3413915c90e77d667310e5341e71cbc860aa640f17f",
}
A1_FREEZE = {
    "path": str(ROOT / "_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/A1/CANDIDATE_FREEZE_V1.json"),
    "bytes": 4569,
    "sha256": "a0b2c232aa72a477e02452f74be7c3aa94eaa891b915579c9fb2d7d05220718f",
}
A1_RESULT = {
    "path": str(ROOT / "_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/A1/RESULT.json"),
    "bytes": 2522,
    "sha256": "a93b31f865ae00d7bcc8503b5dcd5da38b34377ddcf8da742021d83b8a18d0e1",
}
A1_CONFIG = {
    "path": str(ROOT / "_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/A1_HAWOR_W1_DIAG_CONFIG_V1.json"),
    "bytes": 4103,
    "sha256": "625e0add5d6580e35e44318fd41bde83520834527a460b0649ebf56da3b9ee6d",
}
A1_ORCHESTRATOR = {
    "path": str(ROOT / "src/chaoyang/ops/run_0915_robot_recovery_hawor_w1_diag_v1.py"),
    "bytes": 20898,
    "sha256": "534416707138c42d65d844f321b579a3b351462d749a48c43573182bc0784dc0",
}
A1_WORKER = {
    "path": str(ROOT / "src/chaoyang/ops/run_0915_hawor_w1_diag_persistent_worker_v1.py"),
    "bytes": 9306,
    "sha256": "1288883c661d0fafb100f31345c7697724a8589328611e9fdfcadff7af9c2fe4",
}
LEGACY_V2_WORKER = {
    "path": str(ROOT / "src/chaoyang/ops/run_0915_hawor_resize_only_persistent_worker_v2.py"),
    "bytes": 9231,
    "sha256": "1288bf26bb335564552fbb2fc73475fbc460fe68154f8486d1d6edfe45445240",
}
EXPECTED_WEIGHTS = {
    "bundle": {
        "bytes": 887,
        "path": str(ROOT / "assets/models/vendor/hawor/0915_HAWOR_INFERENCE_BUNDLE_V1.json"),
        "sha256": "45b130f19369d7a7184f21fac27cfaa53c5eea3716ab2fa8b69d86a56ca87d79",
    },
    "model": {
        "bytes": 3267481572,
        "path": str(ROOT / "assets/models/vendor/hawor/hawor/checkpoints/hawor.ckpt"),
        "sha256": "4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2",
    },
    "detector": {
        "bytes": 53582271,
        "path": str(ROOT / "assets/models/vendor/hawor/external/detector.pt"),
        "sha256": "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033",
    },
}
EXPECTED_DOMAIN = {
    "physical_left_source_index": 1,
    "physical_right_source_index": 0,
    "operation": "CROP_THEN_RESIZE_ONLY",
    "output_size": [1280, 960],
    "lens_undistortion": False,
    "remap_applied": False,
}
EXPECTED_INPUTS = {
    "play_cards_0915_106": {
        "frame_count": 170,
        "source_group": "0915:cards_120_0915:106:e5da06b908bd467399c6189439b1c17f",
        "source_snapshot_sha256": "bd0a9ba6b68fe69e76b846e1f0df1768bdb764d24a6637412d20a8b289322164",
        "source_stereo": {
            "bytes": 13741878,
            "sha256": "48dbdf78a1fb21fd784c61772aa224b2208b6474ccc3bdfd78141c2744e5e26b",
        },
        "camera_params": {
            "bytes": 3454,
            "sha256": "8d014351f222dada644215922fb669ae8a56c238a6a52d73094b6358cc392837",
        },
    },
    "get_potato_chips_0915_029": {
        "frame_count": 234,
        "source_group": "0915:chips_100_0915:029:f0dc829c93c34dff8f952a5886daa25e",
        "source_snapshot_sha256": "041bdd60525f0a08d66b2a59dfddfc4ba3ff0d8ee48985117862b3b7939a83d8",
        "source_stereo": {
            "bytes": 19464152,
            "sha256": "67cbc2f75d48913245107ed40761f2a913f03ab4176a5834df1cd894c902c594",
        },
        "camera_params": {
            "bytes": 3454,
            "sha256": "7cd2fff2bb30e149eeae4d93f1c049f2dd4fe474a581da808c51dd50b169a3c3",
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


def verify_ref(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
        raise RuntimeError(f"{label} exact reference drift")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def allowed_worker_diff(a1_path: Path, a2_path: Path) -> dict[str, Any]:
    """Prove execution AST equality after role/access/lineage normalization."""
    source_a1 = a1_path.read_text(encoding="utf-8")
    source_a2 = a2_path.read_text(encoding="utf-8")
    replacements = [
        ("Run fixed 044/097 W1-DIAG", "Run fixed 106/029 W1-ADOPTION"),
        ("play_cards_0915_044", "play_cards_0915_106"),
        ("get_potato_chips_0915_097", "get_potato_chips_0915_029"),
        ("fresh HaWoR W1-DIAG worker", "fresh HaWoR W1-ADOPTION worker"),
        ("0915-hawor-resize-only-prepared-manifest-v2", "0915-hawor-resize-only-w1-adoption-prepared-manifest-v1"),
        ('manifest.get("cohort_role") != "W1_DIAG"', 'manifest.get("cohort_role") != "W1_ADOPTION"'),
        ('manifest.get("frame_count") != 560', 'manifest.get("frame_count") != 404'),
        ("signed fixed W1-DIAG pair", "signed fixed W1-ADOPTION pair"),
        ('row.get("cohort_role") != "W1_DIAG"', 'row.get("cohort_role") != "W1_ADOPTION"'),
        ('row.get("access_state") != "OPEN_AT_T0"', 'row.get("access_state") != "OPENED_AFTER_CANDIDATE_AMENDMENT_V2"'),
        ("wrong W1-DIAG input contract", "wrong W1-ADOPTION input contract"),
        ("0915-hawor-w1-diag", "0915-hawor-w1-adoption"),
        ('"cohort_role": "W1_DIAG"', '"cohort_role": "W1_ADOPTION"'),
        ("W1-DIAG HaWoR prediction only; no automatic R0, adoption, control, training or deployment authority.",
         "W1-ADOPTION HaWoR evidence only; no automatic R0, adoption decision, control, training or deployment authority."),
        ("COMPLETED_DIAGNOSTIC_ALL_TERMINAL", "COMPLETED_ADOPTION_ALL_TERMINAL"),
    ]
    transformed = source_a1
    counts = []
    for before, after in replacements:
        count = transformed.count(before)
        if count < 1:
            raise RuntimeError(f"allowed worker diff source token missing: {before}")
        transformed = transformed.replace(before, after)
        counts.append({"from": before, "to": after, "occurrences": count})
    lineage_keys = {
        "candidate_signature_sha256",
        "parent_candidate_signature_sha256",
        "adoption_adapter_or_run_signature_sha256",
    }

    class StripLineage(ast.NodeTransformer):
        def visit_Dict(self, node: ast.Dict) -> ast.AST:
            node = self.generic_visit(node)
            kept = [
                (key, value)
                for key, value in zip(node.keys, node.values)
                if not (isinstance(key, ast.Constant) and key.value in lineage_keys)
            ]
            node.keys = [item[0] for item in kept]
            node.values = [item[1] for item in kept]
            return node

        def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
            node = self.generic_visit(node)
            node.values = [
                value for value in node.values
                if not any(key in ast.dump(value) for key in lineage_keys)
            ]
            return node

    normalized_a1 = StripLineage().visit(ast.parse(transformed))
    normalized_a2 = StripLineage().visit(ast.parse(source_a2))
    ast.fix_missing_locations(normalized_a1)
    ast.fix_missing_locations(normalized_a2)
    ast_a1 = ast.dump(normalized_a1, annotate_fields=True, include_attributes=False)
    ast_a2 = ast.dump(normalized_a2, annotate_fields=True, include_attributes=False)
    if ast_a1 != ast_a2:
        raise RuntimeError("A1/A2 worker execution AST drift after role/access/lineage normalization")
    raw_diff = list(difflib.unified_diff(
        source_a1.splitlines(), source_a2.splitlines(),
        fromfile=str(a1_path), tofile=str(a2_path), lineterm="",
    ))
    return {
        "schema_version": "0915-hawor-a1-a2-worker-semantic-diff-v1",
        "status": "PASS_ENUMERATED_ROLE_ACCESS_ONLY",
        "a1_worker": ref(a1_path),
        "a2_worker": ref(a2_path),
        "normalized_ast_sha256": hashlib.sha256(ast_a2.encode("utf-8")).hexdigest(),
        "algorithm_body_ast_identical_after_role_access_and_lineage_normalization": True,
        "allowed_literal_substitutions": counts,
        "unified_diff": raw_diff,
        "prohibited_change_assertions": {
            "model_load_changed": False,
            "weights_changed": False,
            "preprocessing_changed": False,
            "numeric_thresholds_changed": False,
            "quality_function_changed": False,
            "session_execution_body_changed": False,
            "lineage_identity_conflated": False,
        },
    }


def write_terminal(output: Path, status: str, **fields: Any) -> int:
    value = {
        "schema_version": "0915-robot-recovery-a2-w1-adoption-result-v1",
        "status": status,
        "task_id": TASK_ID,
        "window_run_id": WINDOW_RUN_ID,
        "package_id": PACKAGE_ID,
        "parent_candidate_signature_sha256": PARENT_CANDIDATE_SIGNATURE,
        "claim_limit": "Fixed 106/029 W1-ADOPTION HaWoR and structural evidence only; no R0 quality upgrade, automatic adoption decision, control, training, or deployment authority.",
        **fields,
    }
    atomic_json(output / "RESULT.json", value)
    return 0 if status == "COMPLETED_ADOPTION_ALL_TERMINAL" else 3 if status == "BLOCKED_RESOURCE" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(A1_CONFIG["path"]))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--min-free-mib", type=int, default=61440)
    parser.add_argument("--gpu-wait-seconds", type=int, default=1800)
    parser.add_argument("--gpu-wall-seconds", type=int, default=3600)
    parser.add_argument("--executor-epoch", type=int, default=1)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh A2 output required: {output}")
    output.mkdir(parents=True)

    # Fence phase 1: verify the frozen candidate without parsing A1 numeric results.
    freeze_path = Path(A1_FREEZE["path"])
    result_path = Path(A1_RESULT["path"])
    verify_ref(ref(freeze_path), A1_FREEZE, "A1 candidate freeze")
    verify_ref(ref(result_path), A1_RESULT, "A1 terminal result")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("status") != "FROZEN_BEFORE_W1_ADOPTION_OPEN"
        or freeze.get("candidate_signature_sha256") != PARENT_CANDIDATE_SIGNATURE
        or freeze.get("candidate_id") != "A1_HAWOR_W1_DIAG_DEPENDENCY_SPLIT_V1"
        or freeze.get("config") != A1_CONFIG
        or freeze.get("weights") != EXPECTED_WEIGHTS
        or freeze.get("input_domain") != EXPECTED_DOMAIN
        or freeze.get("numeric_thresholds_changed") is not False
        or freeze.get("result_based_candidate_change_allowed") is not False
    ):
        return write_terminal(output, "BLOCKED_INPUT_FINAL", first_blocker="A1_FREEZE_INVARIANT_DRIFT")
    expected_sealed = [
        {"session_id": session_id, "access_state": "SEALED_UNTIL_CANDIDATE_FREEZE"}
        for session_id in EXPECTED_ORDER
    ]
    if freeze.get("w1_adoption_access_state_at_freeze") != expected_sealed:
        return write_terminal(output, "BLOCKED_INPUT_FINAL", first_blocker="A1_FREEZE_ADOPTION_FENCE_DRIFT")

    verify_ref(ref(Path(A1_ORCHESTRATOR["path"])), A1_ORCHESTRATOR, "A1 orchestrator")
    verify_ref(ref(Path(A1_WORKER["path"])), A1_WORKER, "A1 worker")
    verify_ref(ref(Path(LEGACY_V2_WORKER["path"])), LEGACY_V2_WORKER, "legacy v2 worker")
    a2_worker = ROOT / "src/chaoyang/ops/run_0915_hawor_w1_adoption_persistent_worker_v1.py"
    diff_doc = allowed_worker_diff(Path(A1_WORKER["path"]), a2_worker)
    diff_path = output / "A1_A2_WORKER_SEMANTIC_DIFF_V1.json"
    atomic_json(diff_path, diff_doc)

    config_path = args.config.resolve(strict=True)
    verify_ref(ref(config_path), A1_CONFIG, "frozen A1 config")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("weights") != EXPECTED_WEIGHTS or config.get("input_domain") != EXPECTED_DOMAIN:
        return write_terminal(output, "BLOCKED_INPUT_FINAL", first_blocker="FROZEN_CONFIG_INVARIANT_DRIFT")
    for weight in EXPECTED_WEIGHTS.values():
        verify_ref(ref(Path(weight["path"])), weight, "pinned HaWoR weight")

    adapter_code = [
        ref(Path(__file__)),
        ref(a2_worker),
    ]
    adoption_signature_payload = {
        "schema_version": "0915-robot-recovery-a2-adoption-run-signature-payload-v1",
        "parent_candidate_signature_sha256": PARENT_CANDIDATE_SIGNATURE,
        "cohort_role": "W1_ADOPTION",
        "fixed_inputs": EXPECTED_INPUTS,
        "frozen_config": ref(config_path),
        "frozen_weights": EXPECTED_WEIGHTS,
        "frozen_input_domain": EXPECTED_DOMAIN,
        "adoption_adapter_code": adapter_code,
        "semantic_diff": ref(diff_path),
    }
    adoption_run_signature = a1.canonical_sha(adoption_signature_payload)

    # Fence phase 2: amendment lands before any W1-ADOPTION ledger row or source is read.
    amendment_path = output / "CANDIDATE_AMENDMENT_V2.json"
    amendment = {
        "schema_version": "0915-robot-recovery-candidate-amendment-v2",
        "status": "ROLE_ADAPTER_ONLY_LANDED_BEFORE_W1_ADOPTION_OPEN",
        "candidate_id": freeze["candidate_id"],
        "parent_candidate_signature_sha256": PARENT_CANDIDATE_SIGNATURE,
        "adoption_adapter_or_run_signature_sha256": adoption_run_signature,
        "lineage_identity": "A2_RUN_IS_CHILD_ADAPTER_OF_FROZEN_A1_CANDIDATE_NOT_THE_SAME_SIGNATURE",
        "a1_candidate_freeze": ref(freeze_path),
        "a1_terminal_result_exact_reference_only": ref(result_path),
        "a1_terminal_result_json_parsed": False,
        "w1_diag_numeric_outputs_read_for_candidate_revision": False,
        "w1_adoption_inputs_read_before_amendment": False,
        "semantic_diff": ref(diff_path),
        "adoption_run_signature_payload": adoption_signature_payload,
        "frozen_config": ref(config_path),
        "frozen_weights": EXPECTED_WEIGHTS,
        "frozen_input_domain": EXPECTED_DOMAIN,
        "candidate_invariants": {
            "model_changed": False,
            "weights_changed": False,
            "preprocessing_changed": False,
            "numeric_thresholds_changed": False,
            "quality_function_changed": False,
            "legacy_quality_binding_changed": False,
        },
        "permitted_adapter_changes": [
            "cohort access fence",
            "expected session ids and fixed frame total",
            "manifest and result schema role names",
            "output claim role",
        ],
        "result_based_candidate_change_allowed": False,
        "automatic_adoption_decision_allowed": False,
    }
    atomic_json(amendment_path, amendment)
    amendment_ref = ref(amendment_path)
    access_path = output / "W1_ADOPTION_ACCESS_OPEN_V1.json"
    atomic_json(access_path, {
        "schema_version": "0915-robot-recovery-w1-adoption-access-open-v1",
        "status": "OPENED_AFTER_CANDIDATE_AMENDMENT_V2",
        "parent_candidate_signature_sha256": PARENT_CANDIDATE_SIGNATURE,
        "adoption_adapter_or_run_signature_sha256": adoption_run_signature,
        "candidate_amendment": amendment_ref,
        "sessions": list(EXPECTED_ORDER),
        "source_ledger_mutated": False,
        "opened_scope": "A2_FIXED_106_029_READ_ONLY_EVIDENCE_RUN",
    })
    access_ref = ref(access_path)

    # Only after the two durable fence records exist may adoption metadata/data be read.
    ledger_path = Path(EXPECTED_LEDGER["path"])
    verify_ref(ref(ledger_path), EXPECTED_LEDGER, "cohort access ledger")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    rows = [row for row in ledger["sessions"] if row.get("cohort_role") == "W1_ADOPTION"]
    by_id = {row["session_id"]: row for row in rows}
    if set(by_id) != set(EXPECTED_ORDER):
        return write_terminal(output, "BLOCKED_INPUT_FINAL", first_blocker="W1_ADOPTION_FIXED_IDS_DRIFT", adoption_adapter_or_run_signature_sha256=adoption_run_signature)
    rows = [by_id[session_id] for session_id in EXPECTED_ORDER]
    if any(
        row.get("access_state") != "SEALED_UNTIL_CANDIDATE_FREEZE"
        or row.get("open_after") != "CANDIDATE_FREEZE_V1"
        or row.get("replacement_allowed") is not False
        for row in rows
    ):
        return write_terminal(output, "BLOCKED_INPUT_FINAL", first_blocker="W1_ADOPTION_ORIGINAL_FENCE_DRIFT", adoption_adapter_or_run_signature_sha256=adoption_run_signature)

    prepared_rows = []
    for row in rows:
        expected = EXPECTED_INPUTS[row["session_id"]]
        if any((
            row.get("frame_count") != expected["frame_count"],
            row.get("source_group") != expected["source_group"],
            row.get("source_snapshot_sha256") != expected["source_snapshot_sha256"],
        )):
            return write_terminal(output, "BLOCKED_INPUT_FINAL", first_blocker=f"{row['session_id']}:FROZEN_METADATA_DRIFT", adoption_adapter_or_run_signature_sha256=adoption_run_signature)
        verify_ref(ref(Path(row["source_stereo"]["path"])), expected["source_stereo"], f"{row['session_id']} source stereo")
        source_root = Path(row["processed_root"])
        verify_ref(ref(source_root / "camera_params.json"), expected["camera_params"], f"{row['session_id']} camera params")
        source_frames = sorted((source_root / "preprocess/all_data").glob("*/training_data.json"))
        if len(source_frames) != row["frame_count"]:
            return write_terminal(output, "BLOCKED_INPUT_FINAL", first_blocker=f"{row['session_id']}:SOURCE_FRAME_AXIS_DRIFT", adoption_adapter_or_run_signature_sha256=adoption_run_signature)
        source_manifest = a1.manifest_for_files(
            source_root,
            source_frames,
            output / "input_manifests" / f"{row['session_id']}_SOURCE_FRAME_METADATA_MANIFEST.json",
        )
        prepared = a1.base.prepare_session(row, output / "prepared" / row["session_id"])
        adapter_root = Path(prepared["adapter_session"])
        adapter_frames = sorted((adapter_root / "preprocess/all_data").glob("*/training_data.json"))
        adapter_manifest = a1.manifest_for_files(
            adapter_root,
            adapter_frames,
            output / "input_manifests" / f"{row['session_id']}_ADAPTER_FRAME_METADATA_MANIFEST.json",
        )
        prepared.update(
            cohort_role="W1_ADOPTION",
            access_state="OPENED_AFTER_CANDIDATE_AMENDMENT_V2",
            original_ledger_access_state=row["access_state"],
            source_snapshot_sha256=row["source_snapshot_sha256"],
            source_frame_metadata_manifest=source_manifest,
            adapter_frame_metadata_manifest=adapter_manifest,
        )
        prepared_rows.append(prepared)

    prepared_manifest = {
        "schema_version": "0915-hawor-resize-only-w1-adoption-prepared-manifest-v1",
        "status": "PASS",
        "window_run_id": WINDOW_RUN_ID,
        "cohort_role": "W1_ADOPTION",
        "session_count": 2,
        "frame_count": 404,
        "source_access_ledger": ref(ledger_path),
        "candidate_freeze": ref(freeze_path),
        "candidate_amendment": amendment_ref,
        "w1_adoption_access_open": access_ref,
        "parent_candidate_signature_sha256": PARENT_CANDIDATE_SIGNATURE,
        "adoption_adapter_or_run_signature_sha256": adoption_run_signature,
        "frozen_candidate_code": freeze["code"],
        "adoption_adapter_code": adapter_code,
        "config": ref(config_path),
        "weights": EXPECTED_WEIGHTS,
        "input_domain": EXPECTED_DOMAIN,
        "results": prepared_rows,
        "source_mutated": False,
        "processed_mutated": False,
    }
    prepared_path = output / "PREPARED_MANIFEST.json"
    atomic_json(prepared_path, prepared_manifest)

    worker_output = output / "hawor"
    gpu_receipt = output / "GPU_COMMAND_RECEIPT.json"
    worker_command = [
        str(ROOT / "src/chaoyang/ops/hawor_python.sh"),
        str(a2_worker),
        "--prepared-manifest", str(prepared_path),
        "--output-root", str(worker_output),
    ]
    lease_command = [
        sys.executable,
        str(ROOT / "src/chaoyang/ops/run_gpu_command_with_v71_lease.py"),
        "--task-id", TASK_ID,
        "--attempt-id", "attempt_0001:A2:W1_ADOPTION",
        "--executor-epoch", str(args.executor_epoch),
        "--priority", "CANARY",
        "--gpu-id", str(args.gpu_id),
        "--min-free-mib", str(args.min_free_mib),
        "--wait-seconds", str(args.gpu_wait_seconds),
        "--wall-seconds", str(args.gpu_wall_seconds),
        "--receipt", str(gpu_receipt),
        "--claim-limit", "Fixed 106/029 W1-ADOPTION evidence only; no automatic adoption/R0 authority.",
        "--",
        *worker_command,
    ]
    atomic_json(output / "COMMAND.json", {
        "schema_version": "0915-robot-recovery-a2-w1-adoption-command-v1",
        "parent_candidate_signature_sha256": PARENT_CANDIDATE_SIGNATURE,
        "adoption_adapter_or_run_signature_sha256": adoption_run_signature,
        "candidate_amendment": amendment_ref,
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
            first_blocker=gpu.get("reason", "HAWOR_W1_ADOPTION_RUNTIME"),
            adoption_adapter_or_run_signature_sha256=adoption_run_signature,
            candidate_amendment=amendment_ref,
            gpu_command_receipt=ref(gpu_receipt) if gpu_receipt.is_file() else None,
        )

    batch_path = worker_output / "BATCH_RESULT.json"
    if not batch_path.is_file():
        return write_terminal(output, "FAILED_EVIDENCE_FINAL", first_blocker="MISSING_WORKER_BATCH", adoption_adapter_or_run_signature_sha256=adoption_run_signature)
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    if (
        batch.get("parent_candidate_signature_sha256") != PARENT_CANDIDATE_SIGNATURE
        or batch.get("adoption_adapter_or_run_signature_sha256") != adoption_run_signature
        or batch.get("session_count") != 2
        or batch.get("frame_count") != 404
        or batch.get("failed_runtime") != 0
    ):
        return write_terminal(output, "FAILED_RUNTIME_FINAL", first_blocker="WORKER_BATCH_NOT_ALL_RUNTIME_TERMINAL", adoption_adapter_or_run_signature_sha256=adoption_run_signature)

    admissions = []
    window_rows = []
    for item in batch["results"]:
        npz_path = Path(item["npz"]["path"])
        windows = a1.structural_windows(npz_path)
        side_rows = []
        for side in ("left", "right"):
            strict = item["per_side_strict"][side]
            side_row = {
                "side": side,
                "hawor_side_strict": strict,
                "hawor_session_strict": item["session_strict_status"],
                "adoption_evidence_admitted": True,
                "automatic_adoption_decision": "NOT_AUTHORIZED",
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

    admission_path = output / "CONSUMER_ADMISSION_V1.json"
    windows_path = output / "WINDOW_VALIDITY_V1.json"
    atomic_json(admission_path, {
        "schema_version": "0915-robot-recovery-a2-consumer-admission-v1",
        "status": "ADOPTION_EVIDENCE_COMPLETE_NO_R0_QUALITY_UPGRADE",
        "parent_candidate_signature_sha256": PARENT_CANDIDATE_SIGNATURE,
        "adoption_adapter_or_run_signature_sha256": adoption_run_signature,
        "whole_w0_strict_pass_required": False,
        "numeric_thresholds_changed": False,
        "automatic_adoption_decision": "NOT_AUTHORIZED",
        "sessions": admissions,
    })
    atomic_json(windows_path, {
        "schema_version": "0915-robot-recovery-a2-window-validity-v1",
        "status": "STRUCTURAL_WINDOWS_COMPLETE_R0_QUALITY_PENDING",
        "parent_candidate_signature_sha256": PARENT_CANDIDATE_SIGNATURE,
        "adoption_adapter_or_run_signature_sha256": adoption_run_signature,
        "minimum_window_policy": "NO_NEW_THRESHOLD; FINAL_QUALITY_REQUIRES_ORIGINAL_CONSUMER_CONTRACT_AND_R0_OWN_GATES",
        "rows": window_rows,
    })
    return write_terminal(
        output,
        "COMPLETED_ADOPTION_ALL_TERMINAL",
        adoption_adapter_or_run_signature_sha256=adoption_run_signature,
        lineage_identity="A2_RUN_IS_CHILD_ADAPTER_OF_FROZEN_A1_CANDIDATE_NOT_THE_SAME_SIGNATURE",
        adoption_decision="NOT_AUTOMATED",
        counts={
            "attempted": 2,
            "runtime_terminal": 2,
            "session_strict_pass": batch["passed"],
            "session_strict_rejected": batch["quality_c"],
            "failed_runtime": 0,
        },
        candidate_amendment=amendment_ref,
        access_open=access_ref,
        prepared_manifest=ref(prepared_path),
        gpu_command_receipt=ref(gpu_receipt),
        worker_batch=ref(batch_path),
        consumer_admission=ref(admission_path),
        window_validity=ref(windows_path),
        source_mutated=False,
        processed_mutated=False,
        governance_mutated=False,
        r0_quality_upgraded=False,
        training_eligible=False,
        physical_deployment_authorized=False,
    )


if __name__ == "__main__":
    try:
        exit_code = main()
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - immutable terminal on every failure
        try:
            index = sys.argv.index("--output-root")
            failed_output = Path(sys.argv[index + 1]).resolve()
            if failed_output.is_dir() and not (failed_output / "RESULT.json").exists():
                message = repr(exc)
                terminal_fields: dict[str, Any] = {}
                amendment_file = failed_output / "CANDIDATE_AMENDMENT_V2.json"
                if amendment_file.is_file():
                    saved_amendment = json.loads(amendment_file.read_text(encoding="utf-8"))
                    terminal_fields["adoption_adapter_or_run_signature_sha256"] = saved_amendment.get(
                        "adoption_adapter_or_run_signature_sha256"
                    )
                input_failure = any(token in message.lower() for token in (
                    "drift", "invariant", "access", "manifest", "frame axis", "config", "weight", "diff",
                ))
                write_terminal(
                    failed_output,
                    "BLOCKED_INPUT_FINAL" if input_failure else "FAILED_RUNTIME_FINAL",
                    first_blocker="UNCAUGHT_PREFLIGHT_OR_RUNTIME_EXCEPTION",
                    error=message,
                    **terminal_fields,
                )
        finally:
            exit_code = 2
    raise SystemExit(exit_code)
