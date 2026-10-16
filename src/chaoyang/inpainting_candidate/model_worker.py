"""Bounded Mask/Clean worker; called only by the registered operation.

This worker cannot adopt a baseline or train. Runtime and device authority are
checked before model imports and again around calls by the model-stage bridge.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso
from .contracts import canonical_sha, verify_ref, verify_sealed_stage
from .resources import CANDIDATE_CARD_LOCK_ROOT, proc_start_ticks

MODEL_CALLS_UNIT = "SAM_ADAPTER_INFERENCE_API_INVOCATION"
SAM_CHECKPOINT = "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"


class ModelCallBudget:
    """Count actual adapter inference API attempts, not gates or neural forwards.

    track_text counts one complete text-track invocation; point add_prompt and
    each directional propagate_in_video invocation count once.  init_state,
    constructor, per-frame yields and authority checks are not these units.
    Calls that start and fail remain counted.  Refusal occurs before provider
    invocation, so the frozen cap cannot silently be exceeded.
    """
    def __init__(self, maximum: int, unit: str, *, allow_zero: bool = False):
        if type(maximum) is not int or maximum < (0 if allow_zero else 1) or unit != MODEL_CALLS_UNIT:
            raise ValueError("MODEL_CALL_BUDGET_UNIT_OR_MAX_INVALID")
        self.maximum = maximum
        self.counts = {"track_text": 0, "point_add_prompt": 0, "point_propagate_in_video": 0}

    @property
    def total(self):
        return sum(self.counts.values())

    def consume(self, name):
        if name not in self.counts:
            raise ValueError("MODEL_CALL_KIND_UNSUPPORTED")
        if self.total >= self.maximum:
            raise ValueError("MODEL_CALL_BUDGET_EXHAUSTED")
        self.counts[name] += 1

    def receipt(self):
        return {"unit": MODEL_CALLS_UNIT, "maximum": self.maximum,
                "attempted_api_calls": self.total, "by_api": dict(self.counts),
                "counts_neural_forward_passes": False, "counts_gate_checks": False,
                "counts_constructor_or_init_state": False}


class _CountedPointModel:
    def __init__(self, provider, budget):
        self.provider, self.budget = provider, budget

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def add_prompt(self, *args, **kwargs):
        self.budget.consume("point_add_prompt")
        return self.provider.add_prompt(*args, **kwargs)

    def propagate_in_video(self, *args, **kwargs):
        self.budget.consume("point_propagate_in_video")
        yield from self.provider.propagate_in_video(*args, **kwargs)


class CountedSam:
    def __init__(self, provider, budget):
        self.provider, self.budget = provider, budget
        self.model = _CountedPointModel(provider.model, budget)
        # Keep stage CUDA timing synchronization working through the proxy.
        self.torch = getattr(provider, "torch", sys.modules.get("torch"))

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def track_text(self, *args, **kwargs):
        self.budget.consume("track_text")
        return self.provider.track_text(*args, **kwargs)


def sam_weight_reference(weights, *, repo_root=REPO_ROOT):
    expected = str(Path(repo_root) / SAM_CHECKPOINT)
    matches = [reference for reference in weights if reference.get("path") == expected]
    if len(matches) != 1:
        raise ValueError("UNIQUE_FROZEN_SAM_CHECKPOINT_REFERENCE_REQUIRED")
    verify_ref(matches[0])
    return matches[0]


def model_authority(packet: dict) -> dict:
    """Observe the existing root fence and its card lease; never self-authorize."""
    with (REPO_ROOT / "_run/current/locks/governance.lock").open("rb") as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_SH)
        index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
        state = load_json(REPO_ROOT / "docs/governance/LONG_HORIZON_TASK_STATE.json")
        fcntl.flock(guard.fileno(), fcntl.LOCK_UN)
    pointer = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    packet_path = Path(packet["packet_path"])
    ref = artifact_ref(packet_path)
    if load_json(packet_path) != packet:
        raise ValueError("MODEL_IN_MEMORY_PACKET_DRIFT")
    if len(pointer) != 1 or pointer[0].get("task_id") != packet["task_id"] or pointer[0].get("packet_sha256") != ref["sha256"]:
        raise ValueError("MODEL_ACTIVE_PACKET_BINDING_REQUIRED")
    if packet.get("algorithm_execution_allowed") is not True or packet.get("operation") != "run_inpainting_candidate_visual_stages_v1":
        raise ValueError("MODEL_FINITE_OPERATION_REQUIRED")
    mask_bounds = (packet.get("model_stages") == ["Mask"] and packet["budgets"].get("gpu_count") == 1
            and type(packet["budgets"].get("model_calls_max")) is int and packet["budgets"]["model_calls_max"] > 0
            and packet["budgets"].get("model_calls_unit") == MODEL_CALLS_UNIT)
    from chaoyang.governance.inpainting_candidate_registration import (CLEAN_SPEC_SCHEMA, CLEAN_MODEL_CALLS_UNIT,
                                                                       CLEAN_STAGE_SCOPE, G3_SCOPE)
    g3_all_blocked = (packet.get("experiment_scope") == G3_SCOPE and packet.get("model_stages") == ["Mask"]
        and packet["budgets"].get("gpu_count") == 1 and packet["budgets"].get("model_calls_max") == 0
        and packet["budgets"].get("model_calls_unit") == MODEL_CALLS_UNIT
        and bool(packet.get("sessions")) and all(row.get("source_status") == "BLOCKED" for row in packet["sessions"]))
    clean_bounds = (packet.get("model_stages") == ["Clean"] and packet["budgets"].get("gpu_count") == 1
            and packet.get("task_id", "").startswith("inpainting_generalization_g2_")
            and packet.get("spec_schema") == CLEAN_SPEC_SCHEMA and packet.get("actual_stage_scope") == CLEAN_STAGE_SCOPE
            and type(packet["budgets"].get("model_calls_max")) is int
            and 1 <= packet["budgets"]["model_calls_max"] <= 2
            and packet["budgets"].get("model_calls_unit") == CLEAN_MODEL_CALLS_UNIT)
    native_clean_bounds = (packet.get("experiment_scope") == "INPAINTING_NATIVE_BASELINE_CLEAN"
            and packet.get("model_stages") == ["Clean"] and packet["budgets"].get("gpu_count") == 1
            and packet.get("spec_schema") == "inpainting-finite-native-clean-spec-v1"
            and packet.get("actual_stage_scope") == "INPAINTING_NATIVE_BASELINE_CLEAN"
            and type(packet["budgets"].get("model_calls_max")) is int
            and packet["budgets"]["model_calls_max"] == sum(
                row.get("upstream_status") == "READY" for row in packet.get("sessions", []))
            and packet["budgets"].get("model_calls_unit") == CLEAN_MODEL_CALLS_UNIT)
    if not (mask_bounds or clean_bounds or native_clean_bounds or g3_all_blocked):
        raise ValueError("MODEL_G2_CLEAN_BUDGET_REQUIRED" if packet.get("model_stages") == ["Clean"]
                         else "MODEL_G1_MASK_ONLY_BUDGET_REQUIRED")
    writer = packet["writer"]
    rows = [row for row in state["tasks"] if row.get("task_id") == packet["task_id"]]
    if (len(rows) != 1 or rows[0].get("status") != "RUNNING"
            or (state.get("next_task") or {}).get("task_id") != packet["task_id"]
            or rows[0].get("pid") != writer["pid"] or rows[0].get("proc_start_ticks") != writer["proc_start_ticks"]
            or rows[0].get("run_signature") != packet["run_signature"] or rows[0].get("task_packet") != ref):
        raise ValueError("MODEL_RUNNING_LEDGER_REQUIRED")
    if load_json(Path(packet["writer_file"])) != writer:
        raise ValueError("MODEL_WRITER_FILE_DRIFT")
    if proc_start_ticks(writer["pid"]) != writer["proc_start_ticks"]:
        raise ValueError("MODEL_WRITER_FENCE_CHANGED")
    from chaoyang.governance.exact78_operation_gate import _caller_is_writer_descendant
    import os
    if not _caller_is_writer_descendant(writer["pid"], writer["proc_start_ticks"], os.getpid()):
        raise ValueError("MODEL_CALLER_NOT_FENCED_WRITER_DESCENDANT")
    from datetime import datetime, timezone
    if datetime.now(timezone.utc) >= datetime.fromisoformat(packet["deadline_at"]):
        raise ValueError("MODEL_FINITE_DEADLINE")
    lease = load_json(CANDIDATE_CARD_LOCK_ROOT / (packet["gpu_mapping"]["uuid"] + ".json"))
    required = {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
                "fencing_token": writer["fencing_token"], "pid": writer["pid"],
                "proc_start_ticks": writer["proc_start_ticks"], "uuid": packet["gpu_mapping"]["uuid"],
                "physical_index": packet["gpu_mapping"]["physical_index"], "worker_logical_index": 0}
    if any(lease.get(key) != value for key, value in required.items()):
        raise ValueError("MODEL_CARD_LEASE_BINDING_REQUIRED")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != required["uuid"]:
        raise ValueError("MODEL_DEVICE_ENVIRONMENT_BINDING_REQUIRED")
    return {**required, "execution_allowed": True, "algorithm_execution_allowed": True,
            "gpu_uuid": required["uuid"]}


def mask_review(rgb: np.ndarray, arrays: dict) -> np.ndarray:
    """Raw and role overlay only; never described as a final Robot video."""
    overlay = rgb.copy()
    colors = {"effective_removal": (255, 50, 50), "protected_visible": (30, 220, 30),
              "unknown": (220, 180, 20)}
    for name, color in colors.items():
        mask = arrays.get(name)
        if isinstance(mask, np.ndarray) and mask.shape == rgb.shape[:3]:
            overlay[mask] = ((overlay[mask].astype(np.uint16) + np.array(color, dtype=np.uint16)) // 2).astype(np.uint8)
    return np.concatenate([rgb, overlay], axis=2)


def run(packet: dict) -> dict:
    from chaoyang.governance.inpainting_candidate_registration import (
        authorize_candidate_operation, VISUAL_OPERATION, G3_SCOPE, G3_RESULT_SCHEMA)
    frozen = authorize_candidate_operation(VISUAL_OPERATION, ["--packet", packet["packet_path"]], repo_root=REPO_ROOT)
    if packet != frozen:
        raise ValueError("MODEL_PACKET_NOT_FROZEN_CURRENT_PACKET")
    authority = lambda: model_authority(packet)
    authority()  # No CUDA/model module may be imported before this gate.
    for ref in packet["runtime_code"]:
        verify_ref(ref)
    if packet.get("model_stages") == ["Clean"]:
        # The existing gate/lease is shared, but Clean never constructs SAM or
        # silently spends G1's adapter API budget.
        from .clean_worker import run_clean_batch
        return run_clean_batch(packet, authority=authority)
    from .model_stages import load_frozen_models, run_automatic_mask_stage
    from .automatic_masks import (resolve_controller_text_prompt, resolve_text_wearer_retention,
                                  resolve_source_aware_retention_veto, resolve_glove_text_prompt)
    from .source_frames import write_review_video
    controller_text_prompt = packet.get("config", {}).get("controller_text_prompt")
    resolve_controller_text_prompt(controller_text_prompt)  # Refuse before model construction.
    prompt_options = ({} if controller_text_prompt is None else
                      {"controller_text_prompt": controller_text_prompt})
    retention = packet.get("config", {}).get("text_wearer_retention")
    resolve_text_wearer_retention(retention)
    if retention is not None:
        prompt_options["text_wearer_retention"] = retention
    source_aware_veto = packet.get("config", {}).get("source_aware_retention_veto")
    resolve_source_aware_retention_veto(source_aware_veto)
    if source_aware_veto is not None:
        prompt_options["source_aware_retention_veto"] = source_aware_veto
    glove_text_prompt = packet.get("config", {}).get("glove_text_prompt")
    resolve_glove_text_prompt(glove_text_prompt)
    if glove_text_prompt is not None:
        prompt_options["glove_text_prompt"] = glove_text_prompt
    root = Path(packet["output_dir"])
    g3 = packet.get("experiment_scope") == G3_SCOPE
    budget = ModelCallBudget(packet["budgets"]["model_calls_max"], packet["budgets"].get("model_calls_unit"), allow_zero=g3)
    model_ref = sam_weight_reference(packet["model_weights"])
    has_ready_source = any(row.get("source_status") == "READY" for row in packet["sessions"]) if g3 else True
    models = load_frozen_models(REPO_ROOT, root / "model_load", authorization=authority,
                                frozen_inputs=[*packet["model_weights"], *packet["runtime_code"]],
                                need_sam=True, need_clean=False) if has_ready_source else {}
    atomic_json(root / "MODEL_LOAD.json", {
        "schema": "inpainting-g1-actual-model-load-evidence-v1",
        "task_id": packet["task_id"], "run_signature": packet["run_signature"],
        "load_receipt": models.get("load_receipt"),
        "sam_adapter_evidence": getattr(models.get("sam"), "evidence", None),
        "inference_api_calls": 0, "visual_quality": "NOT_EVALUATED",
    })
    sam = CountedSam(models["sam"], budget) if has_ready_source else None
    results = []
    for row in packet["sessions"]:
        authority()
        if g3 and row.get("source_status") == "BLOCKED":
            verify_ref(row["diagnostic_ref"])
            status = {"session_id": row["session_id"], "session_key": row["session_key"],
                "execution": "BLOCKED_SOURCE", "source_status": "BLOCKED", "failure_code": row["failure_code"],
                "diagnostic_ref": row["diagnostic_ref"], "native_frames": row["native_frames"],
                "model_api_calls_attempted": 0, "model_calls_unit": MODEL_CALLS_UNIT,
                "manual_seed_reads": 0, "partial_consumable": False, "visual_quality": "NOT_REVIEWED",
                "control_ground_truth": False, "training_eligible": False, "experiment_scope": G3_SCOPE,
                "completed_at": now_iso()}
            out = root / "sessions" / row["session_key"]
            out.mkdir(parents=True, exist_ok=True)
            atomic_json(out / "SESSION_RESULT.json", status)
            results.append(status)
            atomic_json(root / "MODEL_PROGRESS.json", {"completed": len(results), "planned": len(packet["sessions"]),
                "sessions": results, "model_call_accounting": budget.receipt(), "updated_at": now_iso()})
            continue
        source_ref = row["source_receipt"]
        verify_ref(source_ref)
        source = load_json(Path(source_ref["path"]))
        folder = Path(source_ref["path"]).parent
        verify_sealed_stage(source, source["component_signature"], output_root=folder)
        for source_input in source.get("actual_inputs", []):
            verify_ref(source_input)
        report = load_json(folder / "SOURCE_REPORT.json")
        for raw_ref in report["actual_inputs"].values():
            verify_ref(raw_ref)
        with np.load(folder / "RAW_RGB.npz", allow_pickle=False) as archive:
            rgb = archive["rgb"]
        # Source seals an ordered JSON list; governance.load_json intentionally
        # accepts objects only. Preserve this native list for stage validation.
        mapping = json.loads((folder / "FRAME_MAP.json").read_text(encoding="utf-8"))
        out = root / "sessions" / row["session_key"]
        before_calls = budget.total
        try:
            receipt = run_automatic_mask_stage(rgb, sam=sam, output_dir=out / "Mask",
                raw_source_ref=report["actual_inputs"]["video"], frame_map=mapping,
                model_ref=model_ref, component_signature=row["mask_signature"],
                authorization=authority, projection_hints=None, **prompt_options)
            for raw_ref in report["actual_inputs"].values():
                verify_ref(raw_ref)
            for source_input in source.get("actual_inputs", []):
                verify_ref(source_input)
            with np.load(out / "Mask/MASK_ARRAYS.npz", allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            review = write_review_video(out / "raw_mask_review.mp4", mask_review(rgb, arrays), mapping)
            status = {"session_id": report["session_id"], "session_key": row["session_key"],
                "execution": "COMPLETE_MASK_ONLY", "native_frames": len(rgb),
                "source": source_ref, "mask": artifact_ref(out / "Mask/RECEIPT.json"), "review": review,
                "manual_seed_reads": 0, "motion": "NOT_EXECUTED", "clean": "NOT_EXECUTED",
                "render": "NOT_EXECUTED", "composite": "NOT_EXECUTED", "visual_quality": "NOT_REVIEWED",
                "control_ground_truth": False, "training_eligible": False, "completed_at": now_iso()}
        except Exception as error:
            status = {"session_id": report["session_id"], "session_key": row["session_key"],
                      "execution": "FAILED_RUNTIME", "failure": type(error).__name__ + ":" + str(error),
                      "partial_consumable": False, "visual_quality": "NOT_REVIEWED", "completed_at": now_iso()}
            out.mkdir(parents=True, exist_ok=True)
        status.update(model_api_calls_attempted=budget.total - before_calls, model_calls_unit=MODEL_CALLS_UNIT)
        if "experiment_scope" in packet:
            status["experiment_scope"] = packet["experiment_scope"]
        atomic_json(out / "SESSION_RESULT.json", status)
        results.append(status)
        atomic_json(root / "MODEL_PROGRESS.json", {"completed": len(results), "planned": len(packet["sessions"]),
                                                   "sessions": results, "model_call_accounting": budget.receipt(), "updated_at": now_iso()})
        # Identity/SHA/lease failures stop this entire frozen model task; genuine
        # session data/model failures remain isolated, not an infinite retry.
        if status["execution"] != "COMPLETE_MASK_ONLY" and any(code in status["failure"] for code in ("SHA_DRIFT", "FENCE", "LEASE", "DEADLINE", "ACTIVE_PACKET", "MODEL_CALL_BUDGET")):
            break
    result = {"schema": G3_RESULT_SCHEMA if g3 else "inpainting-g1-mask-batch-result-v1", "task_id": packet["task_id"],
              "run_signature": packet["run_signature"], "sessions": results,
              "completed_mask_sessions": sum(row["execution"] == "COMPLETE_MASK_ONLY" for row in results),
              "planned_sessions": len(packet["sessions"]), "stage_scope": "G1_AUTOMATIC_MASK_ONLY",
              "model_call_accounting": budget.receipt(), "sam_weight": model_ref,
              "underlying_sam_provider": (type(models["sam"]).__module__ + "." + type(models["sam"]).__qualname__
                                          if has_ready_source else "NOT_CONSTRUCTED_ALL_SOURCES_BLOCKED"),
              "candidate_adopted": False, "visual_quality": "NOT_REVIEWED",
              "files_deleted": 0, "training_steps": 0, "completed_at": now_iso()}
    if "experiment_scope" in packet:
        result["experiment_scope"] = packet["experiment_scope"]
    if g3:
        result.update(blocked_source_sessions=sum(row["execution"] == "BLOCKED_SOURCE" for row in results),
            ready_source_sessions=sum(row.get("source_status") == "READY" for row in packet["sessions"]))
        # Recording all blocked identities is not successful model execution.
        result["execution_diagnostic_complete"] = (result["completed_mask_sessions"] > 0 and
            result["completed_mask_sessions"] + result["blocked_source_sessions"] == result["planned_sessions"])
    atomic_json(root / "MODEL_RESULT.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args(argv)
    packet = load_json(args.packet.resolve(strict=True))
    value = run(packet)
    if packet.get("model_stages") == ["Clean"]:
        print(json.dumps({key: value[key] for key in ("task_id", "completed_clean_sessions", "planned_sessions")}), flush=True)
        if packet.get("experiment_scope") == "INPAINTING_NATIVE_BASELINE_CLEAN":
            return 0 if value["execution_diagnostic_complete"] else 2
        return 0 if value["completed_clean_sessions"] == value["planned_sessions"] else 2
    from chaoyang.governance.inpainting_candidate_registration import G3_SCOPE
    if packet.get("experiment_scope") == G3_SCOPE:
        print(json.dumps({key: value[key] for key in ("task_id", "completed_mask_sessions", "blocked_source_sessions", "planned_sessions")}), flush=True)
        return 0 if value["execution_diagnostic_complete"] else 2
    print(json.dumps({key: value[key] for key in ("task_id", "completed_mask_sessions", "planned_sessions")}), flush=True)
    return 0 if value["completed_mask_sessions"] == value["planned_sessions"] else 2
