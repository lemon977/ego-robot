"""V1-compatible submission and serial component consumer, with no model imports.

The existing Task Packet publisher owns registration, writer identity and leases.
Only the explicitly adopted V1 Raw/config/seed references enter this producer.
Historical outputs are comparison evidence, never substitutes for these inputs.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Callable, Mapping

from .contracts import SCHEMA, STAGES, canonical_sha, file_ref, verify_sealed_stage

REQUEST_SCHEMA = "inpainting-v1-compatible-submission-v1"
POLICY = "V1_PARENT_RAW_AND_EXPLICIT_SEEDS_ONLY"
DEPENDENCIES = {"Source": (), "Motion": ("Source",), "Mask": ("Source",),
                "Clean": ("Source", "Mask"), "Render": ("Source", "Motion"),
                "Composite": ("Source", "Mask", "Clean", "Render")}
RAW_NAMES = {"camera_meta": "raw/camera_params.meta.json", "camera_params": "raw/camera_params.json",
    "hdf5_prior_only": "dataset.hdf5", "manus": "raw/manus.jsonl", "manus_meta": "raw/manus.meta.json",
    "pico": "raw/pico.jsonl", "raw_vst": "raw/vst.h264", "selection": "raw/capture.selection.json",
    "source_manifest": "manifest.json", "video_qpc": "raw/vst.qpc.ts.jsonl",
    "video_wall": "raw/vst.ts.jsonl", "workflow": "raw/capture.workflow.json"}
ROW_FIELDS = {"session_id", "raw_session_dir", "native_frames", "input_status", "raw_inputs",
              "seed", "time_boundary_resolution"}
DERIVED_COMPONENTS = frozenset({"_run", "review", "processed", "attempts", "archive"})
PROJECT_MOUNT_ALIASES = (Path("/path/to/chaoyang"), Path("/path/to/private_workspace/code/chaoyang"))


class AutomationError(ValueError):
    pass


def require(value, code):
    if not value:
        raise AutomationError(code)


def project_local(path, project):
    """Accept a project mount alias only after proving the roots are samefile."""
    path, project = Path(path).resolve(), Path(project).resolve(strict=True)
    if path.is_relative_to(project):
        return True
    for alias in PROJECT_MOUNT_ALIASES:
        if alias.exists() and alias.samefile(project) and path.is_relative_to(alias.resolve()):
            return True
    return False


def verify_reference(reference):
    """Compare immutable bytes; MNT/CPFS mount aliases retain the signed path."""
    require(isinstance(reference, Mapping) and set(reference) == {"path", "bytes", "sha256"},
            "AUTOMATION_EXACT_REFERENCE_REQUIRED")
    path = Path(reference["path"])
    require(path.is_absolute() and ".." not in path.parts, "AUTOMATION_ABSOLUTE_REFERENCE_REQUIRED")
    actual = file_ref(path)
    require(all(actual[k] == reference[k] for k in ("bytes", "sha256")), "AUTOMATION_INPUT_SHA_DRIFT:" + str(path))
    return path.resolve(strict=True)


def read_reference(reference):
    verify_reference(reference)
    value = json.loads(Path(reference["path"]).read_text(encoding="utf-8-sig"))
    return value


def _selection(project, session, manifest):
    require((session is None) != (manifest is None), "EXACTLY_ONE_SESSION_OR_MANIFEST_REQUIRED")
    if session is not None:
        return [Path(session).absolute()], None
    path = Path(manifest).absolute()
    require(project_local(path.resolve(strict=True), project), "AUTOMATION_SELECTION_MUST_BE_PROJECT_LOCAL")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict) and set(value) == {"schema", "sessions"}
        and value["schema"] == "inpainting-raw-session-manifest-v1"
        and isinstance(value["sessions"], list) and value["sessions"]
        and all(isinstance(p, str) and Path(p).is_absolute() for p in value["sessions"]),
        "AUTOMATION_RAW_SELECTION_ONLY_NO_TASK_CONFIG")
    return [Path(p) for p in value["sessions"]], file_ref(path)


def _row(raw_row, source_root, seed):
    return {"session_id": raw_row["session_id"], "raw_session_dir": str(source_root / raw_row["session_id"]),
        "native_frames": raw_row["raw_frame_count"], "input_status": raw_row["input_status"],
        "raw_inputs": deepcopy(raw_row["source_refs"]), "seed": deepcopy(seed),
        "time_boundary_resolution": deepcopy(raw_row.get("time_boundary_resolution"))}


def build_v1_request(*, project_root, session=None, manifest=None, repair_rounds=1):
    """Read-only concrete full-flow request; this does not authorize execution."""
    require(type(repair_rounds) is int and repair_rounds in (1, 2, 3), "AUTOMATION_REPAIR_ROUNDS_MUST_BE_ONE_TO_THREE")
    project = Path(project_root).resolve(strict=True)
    paths, selection = _selection(project, session, manifest)
    parent_ref = file_ref(project / "inpainting_v1/MANIFEST.json")
    parent = read_reference(parent_ref)
    require(parent.get("baseline_id") == "INPAINTING_RAW_TO_ROBOT_V1", "AUTOMATION_ADOPTED_V1_REQUIRED")
    inputs = {k: deepcopy(parent["inputs"][k]) for k in ("raw_manifest", "camera", "mask_seeds")}
    raw = read_reference(inputs["raw_manifest"])
    read_reference(inputs["camera"])
    seeds = read_reference(inputs["mask_seeds"])
    leaves = {r["session_id"]: r["seed"] for r in parent["inputs"]["per_session_seed_leaves"]}
    index = {r["session_id"]: r["seed"] for r in seeds["records"]}
    source_root = Path(raw["source_root"])
    require(source_root.is_absolute(), "AUTOMATION_FROZEN_RAW_ROOT_REQUIRED")
    available = {r["session_id"]: r for r in raw["sessions"]}
    rows, seen = [], set()
    for path in paths:
        resolved = path.resolve(strict=True)
        sid = path.name
        require(sid in available and sid not in seen and resolved == (source_root / sid).resolve(strict=True),
                "AUTOMATION_ONLY_EXACT_PARENT_RAW_SESSIONS_ALLOWED")
        require(leaves.get(sid) == index.get(sid), "AUTOMATION_PARENT_SEED_INDEX_DRIFT")
        rows.append(_row(available[sid], source_root, leaves[sid]))
        seen.add(sid)
    payload = {"schema": REQUEST_SCHEMA, "recipe": "inpainting-raw-to-robot", "mode": "v1-compatible",
        "parent_v1": parent_ref, "parent_inputs": inputs, "selection_manifest": selection, "sessions": rows,
        "motion_source": "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS", "producer_policy": POLICY,
        "manual_session_edits": 0}
    if repair_rounds != 1:
        payload["repair_rounds"] = repair_rounds
    request = {**payload, "submission_input_sha256": canonical_sha(payload),
        "status": "FULL_FLOW_REQUEST_NOT_EXECUTED", "robot_video_produced": False,
        "training_eligible": False, "control_ground_truth": False, "candidate_adopted": False}
    verify_v1_request(request, project_root=project)
    return request


def verify_v1_request(request, *, project_root):
    project = Path(project_root).resolve(strict=True)
    fields = {"schema", "recipe", "mode", "parent_v1", "parent_inputs", "selection_manifest", "sessions",
        "motion_source", "producer_policy", "manual_session_edits"}
    metadata = {"submission_input_sha256", "status", "robot_video_produced", "training_eligible",
                "control_ground_truth", "candidate_adopted"}
    if isinstance(request, dict) and "repair_rounds" in request:
        fields.add("repair_rounds")
    require(isinstance(request, dict) and set(request) == fields | metadata, "AUTOMATION_REQUEST_FIELDS_REQUIRED")
    require(type(request.get("repair_rounds", 1)) is int and request.get("repair_rounds", 1) in (1, 2, 3),
            "AUTOMATION_REPAIR_ROUNDS_MUST_BE_ONE_TO_THREE")
    require(request["schema"] == REQUEST_SCHEMA and request["mode"] == "v1-compatible"
        and request["recipe"] == "inpainting-raw-to-robot" and request["producer_policy"] == POLICY
        and request["motion_source"] == "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS" and request["manual_session_edits"] == 0
        and all(request[k] is False for k in ("robot_video_produced", "training_eligible", "control_ground_truth", "candidate_adopted"))
        and request["submission_input_sha256"] == canonical_sha({k: request[k] for k in fields}),
        "AUTOMATION_REQUEST_SIGNATURE_OR_POLICY_DRIFT")
    require(verify_reference(request["parent_v1"]).samefile(project / "inpainting_v1/MANIFEST.json"),
            "AUTOMATION_ONLY_PINNED_PARENT_V1_ALLOWED")
    parent = read_reference(request["parent_v1"])
    require(parent.get("baseline_id") == "INPAINTING_RAW_TO_ROBOT_V1" and request["parent_inputs"]
        == {k: parent["inputs"][k] for k in ("raw_manifest", "camera", "mask_seeds")}, "AUTOMATION_PARENT_CONFIG_DRIFT")
    raw = read_reference(request["parent_inputs"]["raw_manifest"])
    read_reference(request["parent_inputs"]["camera"])
    index = read_reference(request["parent_inputs"]["mask_seeds"])
    available = {r["session_id"]: r for r in raw["sessions"]}
    leaves = {r["session_id"]: r["seed"] for r in parent["inputs"]["per_session_seed_leaves"]}
    seeds = {r["session_id"]: r["seed"] for r in index["records"]}
    rows = request["sessions"]
    require(isinstance(rows, list) and rows and len({r.get("session_id") for r in rows}) == len(rows),
            "AUTOMATION_SESSIONS_EMPTY_OR_DUPLICATED")
    if request["selection_manifest"] is not None:
        selected = read_reference(request["selection_manifest"])
        require(selected.get("schema") == "inpainting-raw-session-manifest-v1" and set(selected) == {"schema", "sessions"}
            and [Path(p).resolve(strict=True) for p in selected["sessions"]]
            == [Path(r["raw_session_dir"]).resolve(strict=True) for r in rows], "AUTOMATION_SELECTION_IDENTITY_DRIFT")
    for row in rows:
        sid = row.get("session_id")
        require(set(row) == ROW_FIELDS and sid in available and leaves.get(sid) == seeds.get(sid)
            and row == _row(available[sid], Path(raw["source_root"]), leaves[sid]),
            "AUTOMATION_PARENT_SESSION_BINDING_OR_FORBIDDEN_INPUT")
        require(type(row["native_frames"]) is int and row["native_frames"] > 0
            and row["input_status"] in {"READY_RAW_AUDIT", "READY_RAW_WITH_INVALID_SHORT_GAP"},
            "AUTOMATION_PARENT_RAW_ADMISSION_REQUIRED")
        source = Path(row["raw_session_dir"])
        require(not {p.lower() for p in source.parts} & DERIVED_COMPONENTS,
                "AUTOMATION_DERIVED_SOURCE_FORBIDDEN")
        require(set(row["raw_inputs"]) <= RAW_NAMES.keys()
            and {"raw_vst", "video_qpc", "video_wall", "source_manifest", "pico", "manus"} <= row["raw_inputs"].keys(),
            "AUTOMATION_RAW_ROLE_OR_FILE_FORBIDDEN")
        for role, ref in row["raw_inputs"].items():
            require(Path(ref["path"]) == source / RAW_NAMES[role], "AUTOMATION_RAW_INPUT_ESCAPES_SESSION")
            verify_reference(ref)
        verify_reference(row["seed"])
        if row["time_boundary_resolution"] is not None:
            verify_reference(row["time_boundary_resolution"])
        qpc = [json.loads(line) for line in Path(row["raw_inputs"]["video_qpc"]["path"]).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        require(len(qpc) == row["native_frames"] and all(type(t) is int for t in qpc)
            and all(a < b for a, b in zip(qpc, qpc[1:])), "AUTOMATION_ACTUAL_RAW_FRAME_COUNT_OR_QPC_DRIFT")


def component_signatures(request, code_closure, model_weights=None, stage_config=None):
    """Actual component DAG: Mask edits do not invalidate Motion or Render."""
    require(set(code_closure) == set(STAGES) and all(isinstance(v, list) and v for v in code_closure.values()),
            "AUTOMATION_COMPONENT_CODE_CLOSURE_REQUIRED")
    weights, config = model_weights or {}, stage_config or {}
    require(set(weights) <= {"Mask", "Clean"} and set(config) <= set(STAGES), "AUTOMATION_COMPONENT_DOMAIN_REQUIRED")
    repair_rounds = request.get("repair_rounds", 1)
    require(type(repair_rounds) is int and repair_rounds in (1, 2, 3), "AUTOMATION_REPAIR_ROUNDS_MUST_BE_ONE_TO_THREE")
    mask_config = {"repair_rounds": repair_rounds, **config.get("Mask", {})}
    require(type(mask_config["repair_rounds"]) is int and mask_config["repair_rounds"] == repair_rounds,
            "AUTOMATION_REPAIR_ROUNDS_CONFIG_DRIFT")
    result = {}
    for row in request["sessions"]:
        signatures = {}
        for stage in STAGES:
            inputs = ({"raw_inputs": row["raw_inputs"], "native_frames": row["native_frames"],
                "camera": request["parent_inputs"]["camera"], "time_boundary_resolution": row["time_boundary_resolution"]}
                if stage == "Source" else {"seed": row["seed"], "seed_index": request["parent_inputs"]["mask_seeds"]}
                if stage == "Mask" else {})
            payload = {"schema": SCHEMA, "mode": "v1-compatible", "stage": stage, "session_id": row["session_id"],
                "inputs": inputs, "code": code_closure[stage], "weights": weights.get(stage, []),
                "config": mask_config if stage == "Mask" else config.get(stage, {}),
                "upstream": {p: signatures[p] for p in DEPENDENCIES[stage]},
                "calibration": request["parent_inputs"]["camera"] if stage in {"Source", "Motion", "Render"} else "NOT_CONSUMED",
                "result_schema": SCHEMA}
            signatures[stage] = canonical_sha(payload)
        result[row["session_id"]] = signatures
    return result


def write_once_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return file_ref(path)


def verify_component(reference, signature, row, *, stage, repo_root):
    path = verify_reference(reference)
    require(project_local(path, repo_root), "AUTOMATION_REUSE_MUST_BE_PROJECT_LOCAL")
    receipt = read_reference(reference)
    verify_sealed_stage(receipt, signature, output_root=path.parent)
    require(receipt.get("stage") == stage and receipt.get("session_id") == row["session_id"]
        and receipt.get("native_frames") == row["native_frames"], "AUTOMATION_COMPONENT_NATIVE_IDENTITY_DRIFT")
    names = {Path(r["path"]).name: r for r in receipt["outputs"] if Path(r["path"]).parent == path.parent}
    if stage == "Source":
        import hashlib
        import numpy as np
        require({"FRAME_MAP.json", "RAW_RGB.npz", "SOURCE_REPORT.json"} <= names.keys(), "AUTOMATION_SOURCE_MAP_REQUIRED")
        report = read_reference(names["SOURCE_REPORT.json"])
        require(receipt.get("actual_inputs") == row["raw_inputs"] and report.get("actual_inputs") == row["raw_inputs"]
            and report.get("session_id") == row["session_id"] and report.get("native_sidecar_frames") == row["native_frames"],
            "AUTOMATION_SOURCE_RAW_BINDING_DRIFT")
        with np.load(names["RAW_RGB.npz"]["path"], allow_pickle=False) as archive:
            rgb = archive["rgb"]
        require(rgb.dtype == np.uint8 and rgb.shape == (row["native_frames"], 480, 640, 3),
                "AUTOMATION_SOURCE_RAW_PIXEL_DOMAIN_DRIFT")
        mapping = read_reference(names["FRAME_MAP.json"])
        qpc = [json.loads(line) for line in Path(row["raw_inputs"]["video_qpc"]["path"]).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        require(isinstance(mapping, list) and len(mapping) == row["native_frames"]
            and all(isinstance(m, dict) and m.get("session_id") == row["session_id"]
                and all(type(m.get(k)) is int and m[k] == i for k in
                    ("native_frame_index", "decoded_display_index", "model_input_frame", "output_frame"))
                and m.get("qpc_declared_ns") == qpc[i]
                and m.get("pixel_sha256") == hashlib.sha256(rgb[i].tobytes()).hexdigest() for i, m in enumerate(mapping)),
                "AUTOMATION_SOURCE_MAP_IDENTITY_OR_QPC_DRIFT")
    return receipt


def execute_automation(packet, *, authorization: Callable, builders=None):
    """One existing fenced operation consumes all six stages serially."""
    from chaoyang.paths import get_paths
    repo = Path(packet.get("project_root", get_paths().repo_root)).resolve(strict=True)
    root = Path(packet["output_dir"]).resolve(strict=True)
    require(not root.samefile(repo) and project_local(root, repo), "AUTOMATION_PROJECT_OUTPUT_REQUIRED")
    request = json.loads(Path(packet["request_path"]).read_text(encoding="utf-8-sig"))
    verify_v1_request(request, project_root=repo)
    signatures = component_signatures(request, packet["component_code"], packet.get("component_weights"), packet.get("component_config"))
    require(packet.get("component_signatures") == signatures, "AUTOMATION_FROZEN_COMPONENT_SIGNATURE_DRIFT")
    for references in [*packet["component_code"].values(), *packet.get("component_weights", {}).values()]:
        for reference in references:
            verify_reference(reference)
    owner = None
    def gate():
        nonlocal owner
        current = dict(authorization())
        require(current.get("execution_allowed") is True and current.get("algorithm_execution_allowed") is True,
                "AUTOMATION_REGISTERED_EXECUTION_REQUIRED")
        identity = {k: current.get(k) for k in ("task_id", "run_signature", "fencing_token")}
        require(identity["task_id"] == packet["task_id"] and identity["run_signature"] == packet["run_signature"]
            and identity["fencing_token"] == packet["writer"]["fencing_token"], "AUTOMATION_WRITER_FENCE_DRIFT")
        require(owner is None or owner == identity, "AUTOMATION_SECOND_WRITER_REJECTED")
        owner = identity
        return current
    gate()
    if builders is None:
        from .automation_worker import V1Builders
        builders = V1Builders(packet, gate).builders()
    require(set(builders) == set(STAGES), "AUTOMATION_SIX_ACTUAL_CONSUMERS_REQUIRED")
    result_rows = []
    robot_video_produced = True
    model_calls, render_calls, artifacts = 0, 0, []
    reuse = packet.get("reuse_stages", {})
    require(isinstance(reuse, dict) and set(reuse) <= signatures.keys(), "AUTOMATION_REUSE_SESSION_IDENTITY_REQUIRED")
    for row in request["sessions"]:
        upstream, reused = {}, []
        session_reuse = reuse.get(row["session_id"], {})
        require(set(session_reuse) <= set(STAGES), "AUTOMATION_REUSE_STAGE_INVALID")
        for stage in STAGES:
            gate()
            destination = root / "sessions" / row["session_id"] / stage
            signature = signatures[row["session_id"]][stage]
            reference = session_reuse.get(stage)
            if reference is not None:
                verify_component(reference, signature, row, stage=stage, repo_root=repo)
                reused.append(stage)
            else:
                require(not destination.exists(), "AUTOMATION_PARTIAL_OR_SECOND_WRITER_OUTPUT_CONFLICT:" + stage)
                builders[stage](row=row, output_dir=destination, signature=signature, upstream=dict(upstream), request=request)
                reference = file_ref(destination / "RECEIPT.json")
                verify_component(reference, signature, row, stage=stage, repo_root=repo)
            upstream[stage] = reference
            sealed = read_reference(reference)
            if stage not in reused:
                artifacts.extend([reference, *sealed["outputs"]])
            if stage not in reused:
                model_calls += int(sealed.get("sam_api_calls", 0)) + int(sealed.get("propainter_calls", 0))
                render_calls += int(sealed.get("render_calls", 0))
            gate()
        result_rows.append({"session_id": row["session_id"], "native_frames": row["native_frames"],
            "stage_receipts": upstream, "reused_stages": reused, "execution": "SEALED_COMPLETE_ALL_SIX_STAGES",
            "visual_quality": "NOT_REVIEWED", "human_review": "PENDING", "training_eligible": False,
            "control_ground_truth": False, "candidate_adopted": False})
        composite = read_reference(upstream["Composite"])
        robot_video_produced = robot_video_produced and composite.get("robot_video_produced") is True
    gate()
    result = {"schema": "inpainting-v1-compatible-automation-result-v1", "task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "mode": "v1-compatible", "status": "SEALED_COMPLETE",
        "sessions": result_rows, "robot_video_produced": robot_video_produced, "training_eligible": False,
        "model_calls": model_calls, "render_calls": render_calls, "artifacts": artifacts,
        "control_ground_truth": False, "candidate_adopted": False,
        "next_action": "Review the sealed native-frame Mask, Clean and final Robot arrays/videos; adoption remains separate."}
    write_once_json(root / "AUTOMATION_STAGE.json", result)
    return result
