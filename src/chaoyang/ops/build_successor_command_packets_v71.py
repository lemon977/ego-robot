#!/usr/bin/env python3
"""Build immutable, evidence-backed V7.1 successor child packets.

This builder is intentionally CPU-only.  It audits the currently registered
producer closures and materialises commands, signatures and explicit local
blockers.  It never claims a GPU lease and never edits governance CURRENT_*.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71"
SELECTION = RUN / "successors/SUCCESSOR_CANARY_SELECTION_V71.json"
OUT = RUN / "successors/child_packets"
REGISTRY = ROOT / "docs/governance/CURRENT_BASELINE_REGISTRY_V2.json"
PLAN = ROOT / "docs/governance/PLAN_REVISION.json"
HAWOR_INDEX = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/HAWOR_TERMINAL_INDEX.json"
ROLE_INDEX = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/MASK_ROLE_BOUNDED_V2_1_TERMINAL_INDEX.json"
OBJECT_INDEX = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/MASK_TASK_OBJECT_TERMINAL_INDEX.json"
ROLE_CONFIG_ROOT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/mask_role_bounded_v2_1_configs"
HAWOR_REFERENCE_CONTRACT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/FRESH_CANARY_HAWOR_CONTRACT.json"

HAWOR_RUNNER = ROOT / "src/chaoyang/ops/run_hawor_bounded_parameter_successor.py"
ROLE_RUNNER = ROOT / "src/chaoyang/ops/run_exact78_fullsession_role_mask.py"
OBJECT_RUNNER = ROOT / "src/chaoyang/ops/run_exact78_task_object_identity_guardian.py"
OBJECT_MISSING_IMPORT = ROOT / "src/chaoyang/ops/run_exact78_task_object_identity_canary.py"
LEGACY_GPU_LAUNCHER = ROOT / "src/chaoyang/ops/run_gpu_command_with_lease.py"
V71_GPU_LEASE = ROOT / "src/chaoyang/governance/gpu_lease_v71.py"
SAM_WEIGHT = ROOT / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"

ROLE_HELPERS = [
    ROOT / "src/chaoyang/ops/run_configurable_bilateral_mask_canary.py",
    ROOT / "src/chaoyang/ops/run_newtask_baseline_sam31_mask_probe.py",
    ROOT / "src/chaoyang/ops/run_chips001_pico_mask_temporal.py",
    ROOT / "src/chaoyang/ops/run_assisted_bilateral_flow_refresh.py",
    ROOT / "src/chaoyang/ops/run_exact78_tracker_reentry_bounded_v2_canary.py",
    ROOT / "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
    ROOT / "configs/systems/mask/configs/sam31_runtime_source_v1.json",
]
HAWOR_HELPERS = [
    ROOT / "vendor/HaWoR/hawor/utils/process.py",
    ROOT / "vendor/HaWoR/hawor/utils/rotation.py",
    ROOT / "vendor/HaWoR/lib/models/mano_wrapper.py",
]
MANO_ASSETS = [
    ROOT / "vendor/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl",
    ROOT / "vendor/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl",
]

ACCEPTANCE_THRESHOLDS = {
    "per_side_reprojection_p95_px_max": 12.0,
    "per_side_observed_fraction_drop_max": 0.001,
    "identity_switch_count_max": 0,
    "root_and_pose_rotation_orthogonality_max": 0.0001,
    "root_and_pose_rotation_determinant_min_exclusive": 0.0,
    "bone_length_cv_must_not_regress": True,
    "wrist_step_p95_must_improve_over_raw": True,
    "all_joint_acceleration_p95_must_improve_over_raw": True,
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path, *, hash_content: bool = True, recorded_sha: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    value = {"path": str(resolved), "bytes": resolved.stat().st_size}
    value["sha256"] = sha256(resolved) if hash_content else recorded_sha
    if not value["sha256"]:
        raise RuntimeError(f"missing recorded SHA: {resolved}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def stable_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def terminals(path: Path) -> dict[str, dict[str, Any]]:
    return {item["session_id"]: item for item in load(path)["terminals"]}


def registry_stage(name: str) -> dict[str, Any]:
    rows = [row for row in load(REGISTRY)["entries"] if row["stage"] == name]
    if len(rows) != 1:
        raise RuntimeError(f"expected one registry stage {name}, got {len(rows)}")
    return rows[0]


def group_id(group: dict[str, Any]) -> str:
    return f"{group['stage']}_{group['task']}_{group['acquisition_contract'].removeprefix('chips_cards_')}"


def result_ref(session: str, index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = index[session]
    result = row["result"]
    path = Path(result["path"])
    if not path.is_file() or path.stat().st_size != result["bytes"] or sha256(path) != result["sha256"]:
        raise RuntimeError(f"terminal result closure mismatch: {session}")
    return {"session": session, "grade": row["grade"], "downstream_authorized": row["downstream_authorized"], "result": result}


def hawor_contract(group: dict[str, Any], hawor: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for session in [group["canary_session"], *group["regression_sessions"]]:
        terminal = hawor[session]
        result = load(Path(terminal["result"]["path"]))
        raw = result["inputs"]["raw_hawor"]
        video = result["inputs"]["source_video"]
        with __import__("numpy").load(raw["path"], allow_pickle=False) as archive:
            frame_count = int(archive["observed"].shape[1])
        rows.append({
            "task": terminal["task"],
            "session_id": session,
            "frame_count": frame_count,
            "fps": 30,
            "raw_hawor_npz": raw["path"],
            "raw_hawor_sha256": raw["sha256"],
            "raw_hawor_source_grade": "INPUT_TRACK_ONLY_NOT_CURRENT_TERMINAL_GRADE",
            "source_video": video["path"],
            "source_video_sha256": video["sha256"],
        })
    return {
        "schema_version": "hawor-bounded-parameter-three-session-diagnostic-contract-v71",
        "plan_revision": "chaoyang-v7.1",
        "artifact_revision": "R7_1",
        "method": "SINGLE_TRACK_CONFIDENCE_BOUNDED_PARAMETER_FIT_NO_WINDOW_GAUGE",
        "execution_scope": "DIAGNOSTIC_REPRODUCTION_ONLY_NOT_SUCCESSOR_AUTHORITY",
        "canaries": rows,
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
        "claim_limit": "CPU reproduction of current bounded-v2 only; it is not a distinct R7_1 successor and cannot promote authority.",
    }


def build_hawor(group: dict[str, Any], directory: Path, hawor: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = hawor_contract(group, hawor)
    contract_path = directory / "HAWOR_DIAGNOSTIC_RUN_CONTRACT.json"
    atomic_json(contract_path, contract)
    code = [ref(HAWOR_RUNNER), *[ref(path) for path in HAWOR_HELPERS]]
    model_assets = [ref(path) for path in MANO_ASSETS]
    inputs = [result_ref(s, hawor) for s in [group["canary_session"], *group["regression_sessions"]]]
    signature_inputs = {
        "input_manifest_sha": sha256(contract_path),
        "code_sha": stable_sha(code),
        "config_sha": sha256(contract_path),
        "model_weights_sha": stable_sha(model_assets),
        "calibration_sha": "ABSENT_NOT_CONSUMED",
        "schema_version": contract["schema_version"],
        "vendor_commit": "0aa69e9a26b73345e9cd7c43fa8a8fb2fa5d8fc2",
    }
    output = directory / "attempts/attempt_0001/run_output"
    preflight = ["python", str(HAWOR_RUNNER), "--contract", str(contract_path), "--output-root", str(output), "--preflight-only"]
    diagnostic = ["python", str(HAWOR_RUNNER), "--contract", str(contract_path), "--output-root", str(output)]
    blockers = [{
        "code": "NO_DISTINCT_R7_1_SUCCESSOR_IMPLEMENTATION",
        "detail": "The only verified producer is the same current bounded-v2 CPU postprocessor that produced the C terminal; rerunning it is diagnostic reproduction, not an optimization successor.",
    }]
    command = {
        "schema_version": "successor-command-manifest-v71",
        "stage": "hawor",
        "execution_authorized": False,
        "safe_cpu_preflight_authorized": True,
        "preflight_command": preflight,
        "diagnostic_reproduction_command": diagnostic,
        "authority_promotion_command": None,
        "code_closure": code,
        "model_assets": model_assets,
        "network_inference_weights": "ABSENT_NOT_USED_BY_THIS_CPU_POSTPROCESSOR",
        "run_signature_inputs": signature_inputs,
        "run_signature": stable_sha(signature_inputs),
        "blockers": blockers,
    }
    packet = base_packet(group, directory, command, inputs, blockers)
    receipt = blocked_receipt(group, command, blockers)
    return packet, command, receipt


def build_role(group: dict[str, Any], directory: Path, role: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sessions = [group["canary_session"], *group["regression_sessions"]]
    configs = []
    inputs = []
    for session in sessions:
        path = ROLE_CONFIG_ROOT / f"{session}.json"
        configs.append(ref(path))
        inputs.append(result_ref(session, role))
    code = [ref(ROLE_RUNNER), *[ref(path) for path in ROLE_HELPERS]]
    weight = ref(SAM_WEIGHT, hash_content=False, recorded_sha="0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6")
    signature_inputs = {
        "input_manifest_sha": stable_sha(inputs),
        "code_sha": stable_sha(code),
        "config_sha": stable_sha(configs),
        "model_weights_sha": weight["sha256"],
        "calibration_sha": "ABSENT_NOT_CONSUMED",
        "schema_version": "exact78-fullsession-role-mask-config-v1",
        "sam31_commit": "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7",
    }
    holder = f"v71-successor-{group_id(group)}-attempt-0001"
    commands = []
    for session, config in zip(sessions, configs, strict=True):
        commands.append([
            "python", str(ROLE_RUNNER), "--config", config["path"],
            "--confirm-config-sha", config["sha256"],
            "--output-root", str(directory / f"attempts/attempt_0001/{session}"),
            "--lease-holder", holder,
        ])
    blockers = [
        {
            "code": "GPU_LEASE_INTERFACE_NOT_V71_COMPATIBLE",
            "detail": "The verified role runner requires legacy lease keys status/holder, while the V7.1 lease requires executor_epoch/fencing_token/TTL. The only launcher writes gpu-lease-v1, so no authorized V7.1 launch command exists.",
        },
    ]
    bad_regressions = [row for row in inputs[1:] if row["grade"] not in {"A", "B"} or not row["downstream_authorized"]]
    if bad_regressions:
        blockers.append({
            "code": "FROZEN_REGRESSION_NOT_AB",
            "detail": "The frozen selection violates its two-old-A/B regression contract.",
            "sessions": [row["session"] for row in bad_regressions],
        })
    command = {
        "schema_version": "successor-command-manifest-v71",
        "stage": "role_mask",
        "execution_authorized": False,
        "safe_cpu_preflight_authorized": False,
        "commands_after_v71_lease_adapter_exists": commands,
        "code_closure": code,
        "weight": weight,
        "configs": configs,
        "run_signature_inputs": signature_inputs,
        "run_signature": stable_sha(signature_inputs),
        "blockers": blockers,
    }
    packet = base_packet(group, directory, command, inputs, blockers)
    receipt = blocked_receipt(group, command, blockers)
    return packet, command, receipt


def build_object(group: dict[str, Any], directory: Path, obj: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sessions = [group["canary_session"], *group["regression_sessions"]]
    inputs = [result_ref(session, obj) for session in sessions]
    code = [ref(OBJECT_RUNNER)]
    weight = ref(SAM_WEIGHT, hash_content=False, recorded_sha="0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6")
    signature_inputs = {
        "input_manifest_sha": stable_sha(inputs),
        "code_sha": stable_sha(code),
        "config_sha": "ABSENT_HARDCODED_LEGACY_GUARDIAN",
        "model_weights_sha": weight["sha256"],
        "calibration_sha": "ABSENT_NOT_CONSUMED",
        "schema_version": "exact78-task-object-identity-batch-result-v1",
    }
    blockers = [
        {
            "code": "MISSING_TRANSITIVE_PRODUCER_MODULE",
            "detail": f"Current guardian imports a module that is absent: {OBJECT_MISSING_IMPORT}",
        },
        {
            "code": "LEGACY_HARDCODED_BATCH_OUTPUT_NO_IMMUTABLE_CHILD_CLI",
            "detail": "The guardian has no per-session/config/output CLI and writes the retired fixed batch root, so it cannot satisfy R7_1 no-clobber attempts.",
        },
    ]
    command = {
        "schema_version": "successor-command-manifest-v71",
        "stage": "object_identity",
        "execution_authorized": False,
        "safe_cpu_preflight_authorized": False,
        "command": None,
        "code_closure": code,
        "missing_code": str(OBJECT_MISSING_IMPORT),
        "weight": weight,
        "run_signature_inputs": signature_inputs,
        "run_signature": stable_sha(signature_inputs),
        "blockers": blockers,
    }
    packet = base_packet(group, directory, command, inputs, blockers)
    receipt = blocked_receipt(group, command, blockers)
    return packet, command, receipt


def base_packet(group: dict[str, Any], directory: Path, command: dict[str, Any], inputs: list[dict[str, Any]], blockers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "exact78-successor-child-task-packet-v71",
        "task_id": f"successor_child_{group_id(group)}_v71",
        "plan_revision": "chaoyang-v7.1",
        "artifact_revision": "R7_1",
        "status": "BLOCKED_REFERENCE_PROOF",
        "objective": f"Run one frozen {group['stage']} canary and exactly two regressions without overwriting R7_0.",
        "scope": group,
        "read_set": [
            str(PLAN), str(REGISTRY), str(SELECTION),
            *[row["result"]["path"] for row in inputs],
            str(directory / "COMMAND_MANIFEST.json"),
        ][:8],
        "write_set": [str(directory / "attempts"), str(directory / "final")],
        "run_signature": command["run_signature"],
        "budgets": {"max_rounds": 2, "gpu_seconds_per_round": 7200, "wall_seconds_per_round": 14400},
        "go_no_go": {
            "go": ["canary terminal A/B", "both frozen regressions remain A/B", "no R7_0 overwrite", "signature and evidence closure pass"],
            "no_go": ["any blocker remains", "canary C/runtime failure", "either regression C or degradation", "budget exhausted", "signature drift"],
        },
        "stop_condition": "Remain BLOCKED_REFERENCE_PROOF until all listed blockers are resolved by a new immutable packet; never launch this packet as-is.",
        "claim_limit": "Task packet audit only; it grants no successor, mask, HaWoR, Object6D or Robot authority.",
        "blockers": blockers,
    }


def blocked_receipt(group: dict[str, Any], command: dict[str, Any], blockers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "successor-child-blocked-reference-proof-v71",
        "task_id": f"successor_child_{group_id(group)}_v71",
        "status": "BLOCKED_REFERENCE_PROOF",
        "stage": group["stage"],
        "canary_session": group["canary_session"],
        "regression_sessions": group["regression_sessions"],
        "run_signature": command["run_signature"],
        "blockers": blockers,
        "gpu_started": False,
        "authority_promoted": False,
        "claim_limit": "Closure audit result only; negative terminal closes this child packet without claiming algorithm failure or success.",
    }


def verify_existing() -> int:
    index = load(OUT / "SUCCESSOR_CHILD_PACKET_INDEX_V71.json")
    rows = []
    for child in index["children"]:
        directory = Path(child["packet"]["path"]).parent
        command = load(directory / "COMMAND_MANIFEST.json")
        receipt_path = directory / "PREFLIGHT_RESULT.json"
        if receipt_path.exists():
            raise RuntimeError(f"no-clobber preflight receipt exists: {receipt_path}")
        if command["safe_cpu_preflight_authorized"]:
            completed = subprocess.run(command["preflight_command"], cwd=ROOT, text=True, capture_output=True, check=False)
            status = "PASSED" if completed.returncode == 0 else "FAILED_RUNTIME_FINAL"
            receipt = {
                "schema_version": "successor-child-preflight-result-v71",
                "task_id": child["task_id"],
                "status": status,
                "command": command["preflight_command"],
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
                "gpu_calls": 0,
                "authority_promoted": False,
                "claim_limit": "CPU input/SHA preflight only; the R7_1 successor remains blocked.",
            }
        else:
            receipt = {
                "schema_version": "successor-child-preflight-result-v71",
                "task_id": child["task_id"],
                "status": "SKIPPED_BLOCKED_REFERENCE_PROOF",
                "command": None,
                "returncode": None,
                "gpu_calls": 0,
                "authority_promoted": False,
                "claim_limit": "No command was run because its producer/lease closure is incomplete.",
            }
        atomic_json(receipt_path, receipt)
        rows.append({"task_id": child["task_id"], "status": receipt["status"], "receipt": ref(receipt_path)})
    if any(row["status"] == "FAILED_RUNTIME_FINAL" for row in rows):
        overall = "FAILED_RUNTIME_FINAL"
    else:
        overall = "PASSED_TWO_CPU_PREFLIGHTS_THREE_EXPLICIT_SKIPS"
    result = {
        "schema_version": "successor-command-packet-verification-result-v71",
        "status": overall,
        "gpu_calls": 0,
        "rows": rows,
        "claim_limit": "Packet verification only; all child successor executions remain blocked.",
    }
    atomic_json(OUT / "VERIFICATION_RESULT.json", result)
    print(json.dumps({"status": overall, "rows": len(rows)}, ensure_ascii=False))
    return 0 if overall.startswith("PASSED") else 2


def build() -> int:
    selection = load(SELECTION)
    if selection["plan_revision"] != "chaoyang-v7.1" or selection["counts"]["groups"] != 5:
        raise RuntimeError("selection is not the frozen V7.1 five-group contract")
    if load(PLAN)["plan_revision"] != "chaoyang-v7.1":
        raise RuntimeError("plan revision drift")
    # Registry checks deliberately precede historical artifact traversal.
    expected = {
        "HaWoR": ("hawor_bounded_v2", HAWOR_RUNNER),
        "Role Mask": ("sam31_role_successor_v3", ROLE_RUNNER),
        "Object Mask": ("sam31_task_object_identity_v1", OBJECT_RUNNER),
    }
    for stage, (algorithm, runner) in expected.items():
        entry = registry_stage(stage)
        if entry["algorithm_id"] != algorithm or entry["code_closure"][0]["path"] != str(runner):
            raise RuntimeError(f"registry drift for {stage}")
        if sha256(runner) != entry["code_closure"][0]["sha256"]:
            raise RuntimeError(f"runner SHA drift for {stage}")
    if not HAWOR_REFERENCE_CONTRACT.is_file():
        raise RuntimeError("missing acceptance-threshold reference contract")
    if load(HAWOR_REFERENCE_CONTRACT)["acceptance_thresholds"] != ACCEPTANCE_THRESHOLDS:
        raise RuntimeError("HaWoR acceptance thresholds drift")
    hawor, role, obj = terminals(HAWOR_INDEX), terminals(ROLE_INDEX), terminals(OBJECT_INDEX)
    rows = []
    for group in selection["groups"]:
        directory = OUT / group_id(group)
        if directory.exists():
            raise RuntimeError(f"no-clobber child packet already exists: {directory}")
        directory.mkdir(parents=True)
        if group["stage"] == "hawor":
            packet, command, receipt = build_hawor(group, directory, hawor)
        elif group["stage"] == "role_mask":
            packet, command, receipt = build_role(group, directory, role)
        elif group["stage"] == "object_identity":
            packet, command, receipt = build_object(group, directory, obj)
        else:
            raise RuntimeError(f"unexpected stage: {group['stage']}")
        atomic_json(directory / "COMMAND_MANIFEST.json", command)
        atomic_json(directory / "TASK_PACKET.json", packet)
        atomic_json(directory / "BLOCKED_REFERENCE_PROOF.json", receipt)
        rows.append({
            "task_id": packet["task_id"], "stage": group["stage"], "task": group["task"],
            "status": packet["status"], "canary_session": group["canary_session"],
            "regression_sessions": group["regression_sessions"], "run_signature": command["run_signature"],
            "packet": ref(directory / "TASK_PACKET.json"),
            "command_manifest": ref(directory / "COMMAND_MANIFEST.json"),
            "receipt": ref(directory / "BLOCKED_REFERENCE_PROOF.json"),
        })
    index = {
        "schema_version": "successor-child-packet-index-v71",
        "plan_revision": "chaoyang-v7.1",
        "artifact_revision": "R7_1",
        "status": "PASS_AUDIT_FIVE_CHILDREN_ALL_EXPLICITLY_BLOCKED",
        "counts": {"total": len(rows), "execution_ready": 0, "blocked_reference_proof": len(rows)},
        "children": rows,
        "source_selection": ref(SELECTION),
        "claim_limit": "Command/closure audit only. All five child packets are blocked and no GPU job was launched.",
    }
    atomic_json(OUT / "SUCCESSOR_CHILD_PACKET_INDEX_V71.json", index)
    atomic_json(OUT / "RESULT.json", {
        "schema_version": "successor-command-packet-build-result-v71",
        "status": "PASSED",
        "children_total": 5,
        "execution_ready": 0,
        "blocked_reference_proof": 5,
        "gpu_started": False,
        "index": ref(OUT / "SUCCESSOR_CHILD_PACKET_INDEX_V71.json"),
        "claim_limit": index["claim_limit"],
    })
    print(json.dumps(index["counts"], ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    return verify_existing() if args.verify_existing else build()


if __name__ == "__main__":
    raise SystemExit(main())
