#!/usr/bin/env python3
"""CAS-register exactly one finite 0915 campaign task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from chaoyang.governance.campaign_0915_task_specs_v1 import (
    PLAN_REVISION,
    TASK_ORDER,
    build_packet,
    predecessor_task,
)
from chaoyang.governance.common import (
    AUTHORITY_PATH,
    RECEIPT_PATH,
    REPO_ROOT,
    TASK_STATE_PATH,
    V71_TASK_PACKET_INDEX_PATH,
    artifact_ref,
    atomic_json,
    load_json,
    now_iso,
    publish_bundle,
    validate_artifact_ref,
)
from chaoyang.governance.register_single_task_packet import _validate_packet


CURRENT_INDEX = REPO_ROOT / "tasks/current/INDEX.json"
LIVE = {"PENDING", "READY", "CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}


def _validate_materialized_read_closure(
    packet: dict[str, Any], *, repo_root: Path = REPO_ROOT,
) -> list[str]:
    """Reject a sole executable route whose declared inputs are not present."""
    errors: list[str] = []
    for raw in packet.get("read_set", []):
        path = Path(str(raw))
        resolved = path if path.is_absolute() else repo_root / path
        if not resolved.exists():
            errors.append(f"read_set path is not materialized: {raw}")
    weights = packet.get("weights")
    if isinstance(weights, list):
        for raw in weights:
            path = Path(str(raw))
            resolved = path if path.is_absolute() else repo_root / path
            if not resolved.is_file():
                errors.append(f"weight is not materialized: {raw}")
    return errors


def _validate_predecessor(state: dict[str, Any], task_id: str) -> dict[str, Any]:
    if state.get("next_task") is not None:
        raise RuntimeError("registration requires no current next_task")
    live = [row for row in state.get("tasks", []) if row.get("status") in LIVE]
    if live:
        raise RuntimeError("registration requires no active or pending task")
    if any(row.get("task_id") == task_id for row in state.get("tasks", [])):
        raise RuntimeError(f"task already registered: {task_id}")
    predecessor_id = predecessor_task(task_id)
    predecessor = next(
        (row for row in state.get("tasks", []) if row.get("task_id") == predecessor_id),
        None,
    )
    if predecessor is None:
        raise RuntimeError(f"predecessor task is absent: {predecessor_id}")
    if task_id == TASK_ORDER[0]:
        if predecessor.get("status") not in {"PASSED", "FAILED_RUNTIME_FINAL"}:
            raise RuntimeError("input preparation predecessor is not terminal")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("input preparation predecessor result is not bound")
        result = load_json(Path(result_ref["path"]))
        counts = result.get("counts", {})
        if counts.get("0916_sessions") != 240 or counts.get("0916_failed") != 0:
            raise RuntimeError("0916 cleaning did not close 240 sessions with zero runtime failure")
    elif task_id == "0915_input_prepare_cad_v2":
        if predecessor.get("status") != "FAILED_RUNTIME_FINAL":
            raise RuntimeError("corrective v2 requires terminal v1 CLI routing failure")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("corrective v2 predecessor result is not bound")
        result_path = Path(result_ref["path"])
        result = load_json(result_path)
        audit_log = result_path.parent / "AUDIT.log"
        expected_error = (
            "operation is not maintained by the current algorithm contract: "
            "audit_0915_processed_self_containment_v2"
        )
        if (
            result.get("returncodes") != {"audit": 1}
            or result.get("self_containment") is not None
            or result.get("prepared_manifest") is not None
            or not audit_log.is_file()
            or audit_log.read_text(encoding="utf-8").strip() != expected_error
        ):
            raise RuntimeError("v1 failure is not the exact pre-execution CLI route blocker")
    elif task_id == "0915_vst_image_domain_ab_v1":
        if predecessor.get("status") != "CANCELLED":
            raise RuntimeError("VST image-domain research requires the HaWoR task to be cancelled")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("cancelled HaWoR result is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("task_id") != "0915_hawor_full_v1"
            or result.get("status") != "CANCELLED"
            or result.get("first_blocker")
            != "VST_IMAGE_DOMAIN_UNRESOLVED_POSSIBLE_REDUNDANT_UNDISTORTION"
        ):
            raise RuntimeError("VST research predecessor is not the exact user image-domain hold")
    elif task_id == "0915_hawor_resize_only_canary_v1":
        if predecessor.get("status") != "BLOCKED_EXTERNAL":
            raise RuntimeError("HaWoR resize-only canary requires the closed A/B review")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("A/B result is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("conclusion", {}).get("candidate_mono_domain")
            != "PHYSICAL_LEFT_SOURCEINDEX1_PASSTHROUGH_RESIZE_ONLY"
            or result.get("conclusion", {}).get("current_remap_baseline_admissible") is not False
        ):
            raise RuntimeError("A/B result does not bind the confirmed candidate")
        confirmation_path = REPO_ROOT / "tasks/receipts/0915_VST_IMAGE_DOMAIN_USER_CONFIRMATION.json"
        confirmation = load_json(confirmation_path)
        if (
            confirmation.get("status") != "CONFIRMED"
            or confirmation.get("confirmed_candidate")
            != "A_PHYSICAL_LEFT_SOURCEINDEX1_PASSTHROUGH_RESIZE_ONLY"
            or confirmation.get("authorized_next_scope") != "ONE_SESSION_HAWOR_CANARY_ONLY"
        ):
            raise RuntimeError("user confirmation does not authorize this bounded canary")
    elif task_id == "0915_sam31_mask_full_v1":
        raise RuntimeError(
            "SAM3.1 full batch remains blocked pending separate batch authorization"
        )
    elif task_id == "0915_sam31_strict_role_canary_v2":
        if predecessor.get("status") != "CANCELLED":
            raise RuntimeError("strict-role v2 requires the terminal read-set correction")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("strict-role v1 cancellation result is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("execution_started") is not False
            or result.get("gpu_started") is not False
            or result.get("first_blocker")
            != "TASK_PACKET_READ_SET_OMITS_ACCEPTED_BOUNDED_HAWOR_SUCCESSOR"
        ):
            raise RuntimeError("strict-role v1 was not the exact pre-execution read-set correction")
    elif task_id == "0915_sam31_strict_role_canary_v3":
        if predecessor.get("status") != "FAILED_RUNTIME_FINAL":
            raise RuntimeError("strict-role v3 requires the terminal v2 import-path failure")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("strict-role v2 failure result is not bound")
        result = load_json(Path(result_ref["path"]))
        gpu_ref = result.get("gpu_command_receipt")
        if not isinstance(gpu_ref, dict) or validate_artifact_ref(gpu_ref):
            raise RuntimeError("strict-role v2 GPU failure receipt is not bound")
        gpu = load_json(Path(gpu_ref["path"]))
        if (
            result.get("session_admission") != "NOT_PRODUCED"
            or gpu.get("returncode") != 1
            or "ModuleNotFoundError: No module named 'sam3'"
            not in str(gpu.get("stderr_tail"))
        ):
            raise RuntimeError("strict-role v2 was not the exact pre-model vendor import failure")
    elif task_id == "0915_sam31_strict_role_canary_v4":
        if predecessor.get("status") != "CANCELLED":
            raise RuntimeError("strict-role v4 requires the terminal v3 early diagnostic")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("strict-role v3 cancellation result is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("completed_instances") != 2
            or result.get("session_admission") != "NOT_PRODUCED"
            or result.get("first_blocker")
            != "POINT_REFINEMENT_PROPAGATION_ID_OR_ROUTE_NOT_CONTINUOUS"
        ):
            raise RuntimeError("strict-role v3 was not the exact two-hand early diagnostic")
    elif task_id == "0915_sam31_strict_role_canary_v5":
        if predecessor.get("status") != "FAILED_RUNTIME_FINAL":
            raise RuntimeError("strict-role v5 requires the terminal v4 direction failure")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("strict-role v4 failure result is not bound")
        result = load_json(Path(result_ref["path"]))
        gpu_ref = result.get("gpu_command_receipt")
        if not isinstance(gpu_ref, dict) or validate_artifact_ref(gpu_ref):
            raise RuntimeError("strict-role v4 GPU failure receipt is not bound")
        gpu = load_json(Path(gpu_ref["path"]))
        if (
            result.get("session_admission") != "NOT_PRODUCED"
            or "RuntimeError: No points are provided; please add points first"
            not in str(gpu.get("stderr_tail"))
            or '"completed_instances": 2' not in str(gpu.get("stdout_tail"))
        ):
            raise RuntimeError("strict-role v4 was not the exact direction-level exception")
    elif task_id == "0915_stereo_interaction_cpu_canary_v1":
        if predecessor.get("status") != "PASSED":
            raise RuntimeError("CPU evidence canary requires the completed strict-role v5")
        authorization = load_json(
            REPO_ROOT / "tasks/receipts/0915_PARALLEL_CANARIES_USER_AUTHORIZATION.json"
        )
        if (
            authorization.get("status") != "CONFIRMED"
            or authorization.get("authorized_session") != "play_cards_0915_001"
            or "STEREO_DOMAIN_CPU_PREFLIGHT"
            not in authorization.get("authorized_scope", [])
            or "INTERACTION_V0A_IMAGE_2D_ONLY"
            not in authorization.get("authorized_scope", [])
        ):
            raise RuntimeError("bounded CPU evidence lacks exact user authorization")
    elif task_id == "0915_sam31_weak_role_canary_v1":
        if predecessor.get("status") != "PASSED":
            raise RuntimeError("weak-role canary requires terminal CPU evidence routing")
        result_ref = predecessor.get("result")
        if not isinstance(result_ref, dict) or validate_artifact_ref(result_ref):
            raise RuntimeError("CPU evidence predecessor result is not bound")
        result = load_json(Path(result_ref["path"]))
        if (
            result.get("lane_fences_valid") is not True
            or result.get("gpu_used") is not False
            or result.get("session_id") != "play_cards_0915_001"
        ):
            raise RuntimeError("CPU evidence predecessor did not close both fenced lanes")
    elif task_id == "0915_foundationstereo_single_session_canary_v1":
        if predecessor.get("status") != "PASSED":
            raise RuntimeError("Depth canary requires terminal weak-role routing predecessor")
        cpu_task = next(
            (
                row for row in state.get("tasks", [])
                if row.get("task_id") == "0915_stereo_interaction_cpu_canary_v1"
            ),
            None,
        )
        if cpu_task is None:
            raise RuntimeError("Depth canary lacks the CPU stereo preflight task")
        cpu_ref = cpu_task.get("result")
        if not isinstance(cpu_ref, dict) or validate_artifact_ref(cpu_ref):
            raise RuntimeError("Depth canary CPU predecessor result is not bound")
        cpu_result = load_json(Path(cpu_ref["path"]))
        if cpu_result.get("stereo_gpu_depth_allowed") is not True:
            raise RuntimeError("Depth canary is blocked by the inner stereo admission")
    elif task_id == "0915_removal_envelope_single_session_canary_v1":
        if predecessor.get("status") != "PASSED":
            raise RuntimeError("Removal Envelope canary requires terminal weak-role execution")
        weak_ref = predecessor.get("result")
        if not isinstance(weak_ref, dict) or validate_artifact_ref(weak_ref):
            raise RuntimeError("Removal Envelope weak-role predecessor is not bound")
        weak_result = load_json(Path(weak_ref["path"]))
        if (
            weak_result.get("session_id") != "play_cards_0915_001"
            or weak_result.get("session_admission") != "AWAITING_USER_VISUAL_REVIEW"
        ):
            raise RuntimeError("Removal Envelope predecessor is not the bounded weak-role run")
        review = load_json(
            REPO_ROOT / "tasks/receipts/0915_SAM31_WEAK_ROLE_CANARY_V1_USER_VISUAL_REVIEW.json"
        )
        if review.get("status") != "REJECTED_QUALITY_AS_CLEAN_BASELINE":
            raise RuntimeError("Removal Envelope requires the recorded weak-role visual rejection")
        authorization = load_json(
            REPO_ROOT / "tasks/receipts/0915_REMOVAL_ENVELOPE_V1_USER_AUTHORIZATION.json"
        )
        if (
            authorization.get("status") != "CONFIRMED"
            or authorization.get("authorized_task") != task_id
            or authorization.get("authorized_session") != "play_cards_0915_001"
            or authorization.get("weights") != "ABSENT"
        ):
            raise RuntimeError("Removal Envelope lacks exact user authorization")
    elif task_id == "0915_planar_object6d_single_session_canary_v1":
        if predecessor.get("status") != "PASSED":
            raise RuntimeError("Object6D canary requires the Depth execution terminal")
        depth_ref = predecessor.get("result")
        if not isinstance(depth_ref, dict) or validate_artifact_ref(depth_ref):
            raise RuntimeError("Depth canary result is not bound")
        depth_result = load_json(Path(depth_ref["path"]))
        if (
            depth_result.get("depth_admission") != "PASS"
            or depth_result.get("external_accuracy") != "UNVERIFIED"
        ):
            raise RuntimeError("Object6D canary requires admitted unverified optical-Z")
    elif predecessor.get("status") != "PASSED":
        raise RuntimeError(f"predecessor did not pass: {predecessor_id}")
    return predecessor


def _write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    if path.exists() or path.is_symlink():
        if path.is_file() and path.read_bytes() == encoded:
            return
        raise RuntimeError(f"immutable artifact conflict: {path}")
    atomic_json(path, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, choices=TASK_ORDER)
    parser.add_argument("--expected-revision", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh output required: {output}")

    receipt = load_json(RECEIPT_PATH)
    if receipt.get("governance_revision") != args.expected_revision:
        raise RuntimeError("CAS revision mismatch before registration")
    for reference in receipt.get("files", {}).values():
        errors = validate_artifact_ref(reference)
        if errors:
            raise RuntimeError("current receipt conflict: " + "; ".join(errors))

    state = load_json(TASK_STATE_PATH)
    predecessor = _validate_predecessor(state, args.task_id)
    pointer = load_json(V71_TASK_PACKET_INDEX_PATH)
    predecessor_index = REPO_ROOT / pointer["index_path"]
    if predecessor_index.resolve() != CURRENT_INDEX.resolve():
        raise RuntimeError("current pointer does not resolve to tasks/current/INDEX.json")
    current_ref = artifact_ref(CURRENT_INDEX)
    if (
        pointer.get("index_bytes") != current_ref["bytes"]
        or pointer.get("index_sha256") != current_ref["sha256"]
    ):
        raise RuntimeError("current task index pointer mismatch")

    packet = build_packet(args.task_id)
    errors = _validate_packet(packet)
    errors.extend(_validate_materialized_read_closure(packet))
    if errors:
        raise RuntimeError("invalid campaign packet: " + "; ".join(errors))
    packet_path = REPO_ROOT / "tasks/current" / args.task_id / "TASK_PACKET.json"
    if packet_path.parent.exists() or packet_path.parent.is_symlink():
        raise RuntimeError(f"fresh task packet directory required: {packet_path.parent}")

    output.mkdir(parents=True)
    frozen = output / "PREDECESSOR_TASK_PACKET_INDEX.json"
    shutil.copyfile(CURRENT_INDEX, frozen)
    _write_once(packet_path, packet)
    packet_ref = artifact_ref(packet_path)
    successor_index = {
        "schema_version": "chaoyang-v71-task-packet-index-v3",
        "packet_revision": f"{args.task_id.upper()}_ROUTABLE",
        "plan_revision": PLAN_REVISION,
        "execution_revision": "0915_FULL_FUNNEL_V1",
        "status": "PASS",
        "supersedes_index": artifact_ref(frozen),
        "task_packets": [{
            "task_id": args.task_id,
            "packet_path": str(packet_path.relative_to(REPO_ROOT)),
            "packet_sha256": packet_ref["sha256"],
            "execution_class": "CURRENT_LEDGER_ROUTABLE",
            "execution_allowed": True,
            "weights": packet["weights"],
        }],
        "claim_limit": "Exactly one finite 0915 campaign task is routable; algorithm tasks bind exactly one logical weight.",
    }
    created = now_iso()
    state["tasks"].append({
        "task_id": args.task_id,
        "phase": packet["phase"],
        "plan_execution_revision": "0915_FULL_FUNNEL_V1",
        "attempt": 0,
        "status": "PENDING",
        "updated_at": created,
        "heartbeat_at": None,
        "session": None,
        "pid": None,
        "proc_start_ticks": None,
        "gpu_id": None,
        "task_packet": packet_ref,
    })
    state["next_task"] = {
        "task_id": args.task_id,
        "session": None,
        "prerequisites": packet["prerequisites"],
        "expected_resource": packet["expected_resource"],
        "stop_condition": "All fixed-denominator inputs reach a recorded terminal without changing quality gates.",
    }
    state["recent_events"] = (state.get("recent_events", []) + [{
        "task_id": args.task_id,
        "session": None,
        "attempt": 0,
        "status": "PENDING",
        "created_at": created,
        "message": "Registered one finite 0915 campaign task; execution not started.",
        "task_packet": packet_ref,
        "predecessor": predecessor.get("task_id"),
    }])[-100:]

    result_path = output / "RESULT.json"
    _write_once(result_path, {
        "schema_version": "register-0915-campaign-task-result-v1",
        "task_id": args.task_id,
        "status": "PASSED",
        "registered_at": created,
        "task_packet": packet_ref,
        "weights": packet["weights"],
        "execution_started": False,
        "mask_policy": "SAM3.1_ONLY_USER_LOCKED",
        "claim_limit": "Governance registration only; no algorithm or dataset result.",
    })
    published = publish_bundle(
        load_json(AUTHORITY_PATH),
        state,
        event_type=f"{args.task_id.upper()}_REGISTERED",
        expected_revision=args.expected_revision,
        generator_path=Path(__file__),
        task_packet_index_path=CURRENT_INDEX,
        task_packet_index_value=successor_index,
    )
    _write_once(output / "RUN_RECEIPT.json", {
        "schema_version": "register-0915-campaign-task-receipt-v1",
        "task_id": args.task_id,
        "status": "PASSED",
        "result": artifact_ref(result_path),
        "governance_revision": published["governance_revision"],
        "generation_id": published["generation_id"],
        "execution_started": False,
    })
    print(json.dumps({
        "status": "PASSED", "task_id": args.task_id,
        "weights": packet["weights"],
        "governance_revision": published["governance_revision"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
