"""Read-only planning of explicit sealed stages for a new finite Task.

Operational candidates are not algorithm inputs or a mutable ``latest`` index.
This module never scans output directories, runs builders, writes files, or
changes the parent Task's outcome. The existing executor retains its own fence.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

from .automation import (DEPENDENCIES, PROJECT_MOUNT_ALIASES, component_signatures,
                         project_local, read_reference, verify_component,
                         verify_reference, verify_v1_request)
from .contracts import STAGES

MANIFEST_SCHEMA = "inpainting-v1-resume-candidates-v1"
PLAN_SCHEMA = "inpainting-v1-sealed-resume-plan-v1"


class ResumePlanError(ValueError):
    """An explicit candidate cannot substantiate its claimed stage provenance."""


def _require(value, code):
    if not value:
        raise ResumePlanError(code)


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _reference_shape(ref):
    return (isinstance(ref, Mapping) and set(ref) == {"path", "bytes", "sha256"}
            and isinstance(ref["path"], str) and Path(ref["path"]).is_absolute()
            and ".." not in Path(ref["path"]).parts
            and type(ref["bytes"]) is int and ref["bytes"] >= 0 and _sha(ref["sha256"]))


def _local_path(path, repo):
    resolved = Path(path).resolve(strict=True)
    _require(project_local(resolved, repo), "RESUME_REFERENCE_OUTSIDE_PROJECT")
    if resolved.is_relative_to(repo):
        return resolved
    for alias in PROJECT_MOUNT_ALIASES:
        if alias.exists() and alias.samefile(repo) and resolved.is_relative_to(alias.resolve()):
            return (repo / resolved.relative_to(alias.resolve())).resolve(strict=True)
    raise ResumePlanError("RESUME_PROJECT_ALIAS_NOT_PROVED")


def _read(ref, repo):
    _require(_reference_shape(ref), "RESUME_EXACT_FILE_REFERENCE_REQUIRED")
    try:
        _local_path(verify_reference(ref), repo)
        return read_reference(ref)
    except (ValueError, OSError) as error:
        raise ResumePlanError("RESUME_REFERENCE_REJECTED:" + str(error)) from error


def _key(ref, repo):
    _require(_reference_shape(ref), "RESUME_EXACT_FILE_REFERENCE_REQUIRED")
    return (str(_local_path(ref["path"], repo)), ref["bytes"], ref["sha256"])


def _terminal(status):
    return (isinstance(status, str)
            and (status == "PASSED" or status == "FAILED" or status.startswith("FAILED_")
                 or status in {"TIMEOUT", "TIMED_OUT", "DEADLINE_EXCEEDED"}))


def _request_reference(packet):
    ref = packet.get("inputs", {}).get("request")
    _require(_reference_shape(ref), "RESUME_PARENT_SIGNED_REQUEST_REQUIRED")
    return ref


def _catalog(candidate, packet, result, repo, rows):
    """Only follow declared file refs; a failed prefix must be explicitly listed."""
    catalog = {}
    model_ref = candidate.get("model_result")
    if model_ref is None and result.get("status") == "PASSED":
        model_ref = result.get("model_result")
    if model_ref is not None:
        declared = result.get("model_result")
        if declared is not None:
            _require(_key(declared, repo) == _key(model_ref, repo), "RESUME_PARENT_MODEL_REFERENCE_DRIFT")
        else:
            expected = _local_path(packet["output_dir"], repo) / "MODEL_RESULT.json"
            _require(_local_path(model_ref["path"], repo) == expected, "RESUME_PARENT_MODEL_PATH_DRIFT")
        model = _read(model_ref, repo)
        _require(model.get("status") == "SEALED_COMPLETE"
                 and model.get("task_id") == packet["task_id"]
                 and model.get("run_signature") == packet["run_signature"], "RESUME_PARENT_MODEL_NOT_SEALED_OR_BOUND")
        sessions = model.get("sessions")
        _require(isinstance(sessions, list) and sessions, "RESUME_PARENT_MODEL_SESSIONS_REQUIRED")
        for row in sessions:
            sid = row.get("session_id", row.get("session_key"))
            _require(isinstance(sid, str) and sid in rows and sid not in catalog
                     and row.get("native_frames") == rows[sid]["native_frames"], "RESUME_PARENT_MODEL_SESSION_AMBIGUOUS")
            stages = row.get("stage_receipts", {})
            _require(isinstance(stages, Mapping) and set(stages) == set(STAGES), "RESUME_PARENT_MODEL_STAGE_SET_INCOMPLETE")
            catalog[sid] = deepcopy(stages)
        _require(set(catalog) == set(rows), "RESUME_PARENT_MODEL_SESSION_SET_INCOMPLETE")
    elif result.get("status") == "PASSED":
        raise ResumePlanError("RESUME_PASSED_PARENT_MODEL_REQUIRED")
    explicit = candidate.get("stage_receipts", {})
    _require(isinstance(explicit, Mapping), "RESUME_EXPLICIT_STAGE_MAP_REQUIRED")
    for sid, stages in explicit.items():
        _require(isinstance(stages, Mapping), "RESUME_EXPLICIT_STAGE_MAP_REQUIRED")
        for stage, ref in stages.items():
            if model_ref is not None:
                _require(stage in catalog.get(sid, {})
                         and _key(ref, repo) == _key(catalog[sid][stage], repo), "RESUME_STAGE_NOT_IN_PARENT_MODEL")
            catalog.setdefault(sid, {})[stage] = deepcopy(ref)
    return catalog, model_ref


def _parent(candidate, index, repo):
    _require(isinstance(candidate, Mapping) and {"packet", "result"} <= candidate.keys()
             and set(candidate) <= {"packet", "result", "model_result", "stage_receipts"}, "RESUME_PARENT_FIELDS_REQUIRED")
    packet, result = _read(candidate["packet"], repo), _read(candidate["result"], repo)
    _require(isinstance(packet.get("task_id"), str) and packet["task_id"]
             and result.get("task_id") == packet["task_id"] and _sha(packet.get("run_signature"))
             and result.get("run_signature") == packet["run_signature"], "RESUME_PARENT_TASK_IDENTITY_DRIFT")
    _require(_terminal(result.get("status")), "RESUME_PARENT_NOT_TERMINAL")
    _require(packet.get("mode") == "Automation", "RESUME_PARENT_COMPONENT_PACKET_REQUIRED")
    runtime = packet.get("runtime_code")
    _require(isinstance(runtime, list) and runtime and all(_reference_shape(ref) for ref in runtime),
             "RESUME_PARENT_RUNTIME_SIGNATURE_INCOMPLETE")
    code, weights = packet.get("component_code"), packet.get("component_weights", {})
    _require(isinstance(code, Mapping) and set(code) == set(STAGES) and isinstance(weights, Mapping)
             and set(weights) <= {"Mask", "Clean"}
             and all(isinstance(refs, list) and refs and all(_reference_shape(ref) for ref in refs) for refs in code.values())
             and all(isinstance(refs, list) and all(_reference_shape(ref) for ref in refs) for refs in weights.values()),
             "RESUME_PARENT_ALGORITHM_SIGNATURE_INCOMPLETE")
    parent_request_ref = _request_reference(packet)
    parent_request = _read(parent_request_ref, repo)
    try:
        verify_v1_request(parent_request, project_root=repo)
        parent_signatures = component_signatures(parent_request, packet["component_code"],
            packet.get("component_weights"), packet.get("component_config"))
    except (ValueError, KeyError) as error:
        raise ResumePlanError("RESUME_PARENT_INPUT_SIGNATURE_REJECTED:" + str(error)) from error
    _require(parent_signatures == packet.get("component_signatures"), "RESUME_PARENT_COMPONENT_SIGNATURE_DRIFT")
    rows = {row["session_id"]: row for row in parent_request["sessions"]}
    catalog, model_ref = _catalog(candidate, packet, result, repo, rows)
    own_root = _local_path(packet["output_dir"], repo)
    _require(own_root != repo, "RESUME_PARENT_OUTPUT_ROOT_INVALID")
    choices = []
    for sid, stages in catalog.items():
        _require(sid in rows and isinstance(stages, Mapping) and set(stages) <= set(STAGES), "RESUME_PARENT_STAGE_SESSION_DRIFT")
        for stage, ref in stages.items():
            path = _local_path(ref["path"], repo)
            own_path = own_root / "sessions" / sid / stage / "RECEIPT.json"
            reused = packet.get("reuse_stages", {}).get(sid, {}).get(stage)
            _require(path == own_path or (reused is not None and _key(ref, repo) == _key(reused, repo)),
                     "RESUME_STAGE_NOT_BOUND_TO_PARENT_OUTPUT_OR_REUSE")
            try:
                receipt = verify_component(ref, parent_signatures[sid][stage], rows[sid], stage=stage, repo_root=repo)
            except (ValueError, OSError) as error:
                raise ResumePlanError("RESUME_STAGE_REJECTED:" + str(error)) from error
            upstream = receipt.get("upstream", {})
            _require(isinstance(upstream, Mapping) and set(upstream) == set(DEPENDENCIES[stage]), "RESUME_STAGE_UPSTREAM_SET_DRIFT")
            for dependency, dependency_ref in upstream.items():
                bound = _read(dependency_ref, repo)
                _require(bound.get("stage") == dependency and bound.get("session_id") == sid
                         and bound.get("native_frames") == rows[sid]["native_frames"]
                         and bound.get("status") == "SEALED_COMPLETE"
                         and bound.get("component_signature") == parent_signatures[sid][dependency],
                         "RESUME_STAGE_UPSTREAM_IDENTITY_DRIFT")
            choices.append({"session_id": sid, "stage": stage, "reference": deepcopy(ref),
                "signature": parent_signatures[sid][stage], "upstream": dict(upstream), "candidate_index": index,
                "parent_task_id": packet["task_id"], "parent_status": result["status"]})
    proof = {"candidate_index": index, "packet": deepcopy(candidate["packet"]), "result": deepcopy(candidate["result"]),
             "request": parent_request_ref, "model_result": model_ref, "task_id": packet["task_id"],
             "run_signature": packet["run_signature"], "parent_status": result["status"],
             "runtime_code_refs_bound_by_parent_packet": runtime,
             "historical_wrapper_bytes_required_to_equal_current_wrapper": False,
             "sealed_stages_verified": len(choices)}
    return choices, proof


def _select(choices, signatures, repo):
    """Find a largest consistent six-node DAG; explicit order breaks equal ties."""
    best = {}

    def visit(position, selected):
        nonlocal best
        if len(best) == len(STAGES) or len(selected) + len(STAGES) - position <= len(best):
            return
        if position == len(STAGES):
            best = dict(selected)
            return
        stage = STAGES[position]
        for choice in choices.get(stage, []):
            if choice["signature"] != signatures[stage]:
                continue
            if any(dependency not in selected or _key(ref, repo) != _key(selected[dependency]["reference"], repo)
                   for dependency, ref in choice["upstream"].items()):
                continue
            selected[stage] = choice
            visit(position + 1, selected)
            del selected[stage]
        visit(position + 1, selected)

    visit(0, {})
    return best


def plan_resume(*, project_root, request, component_code, component_weights=None,
                component_config=None, candidates=None):
    """Return executor-ready ``reuse_stages`` without executing or writing.

    ``candidates`` is an explicit list of parent records, or a manifest with
    ``schema=MANIFEST_SCHEMA`` and ``parents``. Each parent supplies immutable
    ``packet``/``result`` refs; optional ``model_result`` must be sealed. Failed
    parents without a sealed ModelResult require explicit
    ``stage_receipts[session_id][stage]``. No candidate means no filesystem I/O.

    The old packet attests its historical outer runtime closure. Current outer
    wrappers need not equal that historical run: stage eligibility instead
    requires the exact algorithm/input/config/weight signature and verified
    stage bytes. The new executor still verifies its own entire live closure.
    """
    signatures = component_signatures(request, component_code, component_weights, component_config)
    if isinstance(candidates, Mapping):
        _require(candidates.get("schema") == MANIFEST_SCHEMA and set(candidates) == {"schema", "parents"},
                 "RESUME_MANIFEST_SCHEMA_REQUIRED")
        candidates = candidates["parents"]
    candidates = [] if candidates is None else candidates
    _require(isinstance(candidates, (list, tuple)), "RESUME_EXPLICIT_PARENT_LIST_REQUIRED")
    all_choices, parents = [], []
    if candidates:
        repo = Path(project_root).resolve(strict=True)
        try:
            verify_v1_request(request, project_root=repo)
        except ValueError as error:
            raise ResumePlanError("RESUME_TARGET_INPUT_REJECTED:" + str(error)) from error
        for index, candidate in enumerate(candidates):
            choices, proof = _parent(candidate, index, repo)
            all_choices.extend(choices)
            parents.append(proof)
    reuse, decisions = {}, []
    for sid, target in signatures.items():
        by_stage = {stage: [] for stage in STAGES}
        seen = {stage: set() for stage in STAGES}
        for choice in all_choices:
            if choice["session_id"] == sid:
                stage = choice["stage"]
                key = _key(choice["reference"], repo)
                if key not in seen[stage]:
                    by_stage[stage].append(choice)
                    seen[stage].add(key)
        selected = _select(by_stage, target, repo) if candidates else {}
        for stage in STAGES:
            chosen = selected.get(stage)
            if chosen:
                reuse.setdefault(sid, {})[stage] = chosen["reference"]
                decision = {"decision": "REUSE_SEALED_STAGE", **{key: chosen[key] for key in
                            ("reference", "candidate_index", "parent_task_id", "parent_status")}}
            else:
                compatible = [choice for choice in by_stage[stage] if choice["signature"] == target[stage]]
                reason = ("NO_EXPLICIT_CANDIDATE" if not by_stage[stage] else "COMPONENT_SIGNATURE_MISMATCH"
                          if not compatible else "UPSTREAM_STAGE_NOT_REUSED"
                          if any(dependency not in selected for dependency in DEPENDENCIES[stage])
                          else "UPSTREAM_BYTE_REFERENCE_MISMATCH")
                decision = {"decision": "BUILD_IN_NEW_TASK", "reason": reason}
            decisions.append({"session_id": sid, "stage": stage, **decision})
    return {"schema": PLAN_SCHEMA, "status": "READ_ONLY_PLAN_COMPLETE", "reuse_stages": reuse,
            "component_signatures": signatures, "decisions": decisions, "parents": parents,
            "session_execution_status": "NOT_EXECUTED_BY_PLANNER", "parent_outcomes_changed": False,
            "files_written": 0, "model_calls": 0, "render_calls": 0,
            "human_review": "PENDING", "training_eligible": False, "control_ground_truth": False,
            "candidate_adopted": False}


__all__ = ["MANIFEST_SCHEMA", "PLAN_SCHEMA", "ResumePlanError", "plan_resume"]
