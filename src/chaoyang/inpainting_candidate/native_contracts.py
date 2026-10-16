"""Finite native-frame candidate contracts, before any algorithm import.

These byte/provenance contracts preserve existing numerical quality settings.
They do not authorize execution, turn model pixels into truth, or adopt V1.
"""
from __future__ import annotations

import json
import ast
from pathlib import Path
import re

from .contracts import ContractError, verify_ref, verify_sealed_stage

NATIVE_GEOMETRY_SCOPE = "INPAINTING_NATIVE_MOTION_RENDER_CPU"
NATIVE_GEOMETRY_SCHEMA = "inpainting-finite-native-geometry-spec-v1"
NATIVE_CLEAN_SCOPE = "INPAINTING_NATIVE_BASELINE_CLEAN"
NATIVE_CLEAN_SCHEMA = "inpainting-finite-native-clean-spec-v1"
NATIVE_COMPOSITE_SCOPE = "INPAINTING_NATIVE_COMPOSITE_CPU"
NATIVE_COMPOSITE_SCHEMA = "inpainting-finite-native-composite-spec-v1"


def require(condition, code):
    if not condition:
        raise ContractError(code)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def signed_ref(reference, root, *, verify=True, local=True):
    require(isinstance(reference, dict) and {"path", "bytes", "sha256"} <= set(reference),
            "NATIVE_SIGNED_FILE_REF_REQUIRED")
    path = Path(reference["path"])
    require(path.is_absolute() and (not local or path.is_relative_to(root)),
            "NATIVE_LOCAL_REF_OUTSIDE_PROJECT")
    if verify:
        require(path.resolve(strict=True) == path, "NATIVE_INPUT_ALIAS_FORBIDDEN")
        verify_ref(reference)
    return path


def direct_outputs(receipt):
    refs = receipt.get("outputs", [])
    names = [Path(ref["path"]).name for ref in refs]
    require(len(names) == len(set(names)), "NATIVE_DUPLICATE_DIRECT_OUTPUT_ROLE")
    return dict(zip(names, refs))


def source_binding(row, root, *, verify=True):
    """Bind original Source/report/QPC/map identity, not only equal counts."""
    count = row.get("native_frames")
    require(type(count) is int and count > 0, "NATIVE_FULL_FRAME_COUNT_REQUIRED")
    require(all(isinstance(row.get(k), str) and bool(row[k]) for k in ("session_key", "session_id")),
            "NATIVE_SOURCE_IDENTITY_REQUIRED")
    require(re.fullmatch(r"[a-zA-Z0-9_]+", row["session_key"]) is not None,
            "NATIVE_SAFE_SESSION_KEY_REQUIRED")
    signature = row.get("source_signature")
    require(re.fullmatch(r"[0-9a-f]{64}", str(signature)) is not None, "NATIVE_SOURCE_SIGNATURE_REQUIRED")
    path = signed_ref(row.get("source_receipt"), root, verify=verify)
    signed_ref(row.get("frame_map_ref"), root, verify=verify)
    if not verify:
        return None, None, None
    receipt = load(path)
    verify_sealed_stage(receipt, signature, output_root=path.parent)
    require(receipt.get("stage") == "Source" and receipt.get("native_frames") == count
            and receipt.get("session_id") == row["session_id"], "NATIVE_SOURCE_IDENTITY_OR_COUNT_DRIFT")
    refs = direct_outputs(receipt)
    require({"SOURCE_REPORT.json", "FRAME_MAP.json", "RAW_RGB.npz"} <= set(refs)
            and refs["FRAME_MAP.json"] == row["frame_map_ref"], "NATIVE_SOURCE_MAP_BINDING_DRIFT")
    report, mapping = load(refs["SOURCE_REPORT.json"]["path"]), load(row["frame_map_ref"]["path"])
    actual = report.get("actual_inputs", {})
    require(report.get("session_id") == row["session_id"] and report.get("native_sidecar_frames") == count
            and isinstance(actual, dict) and {"video", "video_qpc", "video_wall", "manifest"} <= set(actual),
            "NATIVE_ORIGINAL_SOURCE_REPORT_REQUIRED")
    for reference in actual.values():
        raw = signed_ref(reference, root, verify=True, local=False)
        require(not {p.lower() for p in raw.parts} & {"_run", "review", "processed", "attempts"},
                "NATIVE_HISTORICAL_RAW_INPUT_FORBIDDEN")
    qpc = [json.loads(line) for line in Path(actual["video_qpc"]["path"]).read_text(
        encoding="utf-8-sig").splitlines() if line.strip()]
    require(len(qpc) == count and all(type(v) is int for v in qpc)
            and all(a < b for a, b in zip(qpc, qpc[1:])), "NATIVE_ORIGINAL_QPC_ROWS_REQUIRED")
    require(isinstance(mapping, list) and len(mapping) == count and all(isinstance(m, dict)
        and all(type(m.get(k)) is int and m[k] == i for k in
            ("native_frame_index", "decoded_display_index", "model_input_frame", "output_frame"))
        and m.get("qpc_declared_ns") == qpc[i] for i, m in enumerate(mapping)),
        "NATIVE_ONE_TO_ONE_SOURCE_FRAME_MAP_REQUIRED")
    return receipt, report, mapping


def _base(spec, root, *, scope, schema, prefix, operation, gpu, verify):
    require(spec.get("spec_schema") == schema and spec.get("experiment_scope") == scope
            and spec.get("operation") == operation, "NATIVE_EXPLICIT_FINITE_SCOPE_REQUIRED")
    task = spec.get("task_id", "")
    require(isinstance(task, str) and task.startswith(prefix) and task.replace("_", "").isalnum(),
            "NATIVE_FRESH_TASK_ID_REQUIRED")
    require(not any(k in spec for k in ("resume_context", "original_g1_deadline_at")),
            "NATIVE_NO_HISTORICAL_TASK_REVIVAL")
    require(all(k in spec for k in ("inputs", "runtime_code", "calibration", "schema", "config")),
            "NATIVE_SIGNATURE_DOMAINS_REQUIRED")
    bounds = spec.get("budgets", {})
    require(all(type(bounds.get(k)) is int and bounds[k] == v for k, v in
        {"gpu_count": gpu, "training_steps_max": 0, "cpu_threads_per_worker": 2, "candidates_max": 1}.items())
        and type(bounds.get("wall_seconds_max")) is int and 0 < bounds["wall_seconds_max"] <= 28800,
        "NATIVE_FINITE_RESOURCE_BOUNDS_REQUIRED")
    if gpu == 0:
        require(type(bounds.get("model_calls_max")) is int and bounds["model_calls_max"] == 0
                and not spec.get("model_weights"), "NATIVE_CPU_CANNOT_RUN_MODEL")
    output = Path(spec.get("output_dir", ""))
    require(output.is_absolute() and output.is_relative_to(root) and output != root,
            "NATIVE_PROJECT_OUTPUT_REQUIRED")
    rows = spec.get("sessions")
    require(isinstance(rows, list) and 0 < len(rows) <= 11 and all(isinstance(row, dict) for row in rows)
        and len({row.get("session_key") for row in rows}) == len(rows), "NATIVE_FINITE_UNIQUE_SESSIONS_REQUIRED")
    for row in rows:
        require(not any(k in row for k in ("manual_seed", "seed", "manual_points", "frame_limit", "placement",
            "manual_placement", "robot_q", "q", "historical_motion", "historical_mask", "historical_clean",
            "evaluation_annotation", "task_config", "session_config")), "NATIVE_FORBIDDEN_PRODUCER_INPUT")
        if scope in {NATIVE_CLEAN_SCOPE, NATIVE_COMPOSITE_SCOPE}:
            require(row.get("upstream_status") in {"READY", "BLOCKED"}, "NATIVE_UPSTREAM_STATUS_REQUIRED")
        else:
            require(row.get("upstream_status") in {None, "READY"}, "NATIVE_GEOMETRY_SOURCE_READY_REQUIRED")
        if row.get("upstream_status") == "BLOCKED":
            require(type(row.get("native_frames")) is int and row["native_frames"] > 0
                and all(isinstance(row.get(k), str) and bool(row[k]) for k in ("session_key", "session_id", "failure_code"))
                and re.fullmatch(r"[a-zA-Z0-9_]+", row["session_key"]) is not None,
                "NATIVE_EXPLICIT_LOCAL_BLOCKED_IDENTITY_REQUIRED")
            diagnostic = signed_ref(row.get("diagnostic_ref"), root, verify=verify)
            if verify:
                value = load(diagnostic)
                require(value.get("status") in {"BLOCKED", "FAILED_RUNTIME", "FAILED_RUNTIME_FINAL", "FAILED_SOURCE"}
                    and value.get("failure_code") == row["failure_code"] and value.get("native_frames") == row["native_frames"]
                    and any(value.get(k) == row[k] for k in ("session_key", "session_id"))
                    and all(k not in value or value[k] == row[k] for k in ("session_key", "session_id")),
                    "NATIVE_BLOCKED_CANNOT_FAKE_SEALED_OR_WRONG_IDENTITY")
        else:
            source_binding(row, root, verify=verify)
    return rows, bounds


def _signature(row, key):
    require(re.fullmatch(r"[0-9a-f]{64}", str(row.get(key))) is not None,
            "NATIVE_STAGE_SIGNATURE_REQUIRED:" + key)


def validate_native_geometry_spec(spec, *, repo_root, verify_inputs=True):
    root = Path(repo_root).resolve(strict=True)
    rows, bounds = _base(spec, root, scope=NATIVE_GEOMETRY_SCOPE, schema=NATIVE_GEOMETRY_SCHEMA,
        prefix="inpainting_native_motion_render_", operation="run_inpainting_candidate_geometry_stages_v1",
        gpu=0, verify=verify_inputs)
    stages = spec.get("stages")
    require(stages in (["Native"], ["Motion", "Render"]), "NATIVE_TOP_LEVEL_STAGES_REQUIRED")
    native_only = stages == ["Native"]
    total = sum(row["native_frames"] for row in rows)
    require(bounds.get("render_calls_max") == (0 if native_only else 2 * total)
            and bounds.get("zbuffer_passes_max") == (0 if native_only else 4 * total),
            "NATIVE_TWO_VIEW_FOUR_ZBUFFER_BUDGET_REQUIRED")
    expected = ({"native_rule": "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS", "camera_binding": "NOT_REQUESTED"}
        if native_only else {"complete_robot": True, "image_size": [640, 480],
            "motion_rule": "FROZEN_V1_RETARGET_AUTOMATIC_BASE", "scene_depth": "ABSENT"})
    require(spec["config"] == expected, "NATIVE_GEOMETRY_FROZEN_RECIPE_REQUIRED")
    if native_only:
        require(not spec.get("robot_assets") and "asset_closure" not in spec["inputs"],
                "NATIVE_ONLY_NO_ASSET_OR_ROBOT_DEPENDENCY")
    else:
        assets = spec.get("robot_assets")
        require(isinstance(assets, list) and len(assets) == 69 and len({a.get("path") for a in assets}) == 69,
                "NATIVE_SAME_69_ROBOT_ASSETS_REQUIRED")
        closure_path = signed_ref(spec["inputs"].get("asset_closure"), root, verify=verify_inputs)
        for asset in assets:
            signed_ref(asset, root, verify=verify_inputs)
        if verify_inputs:
            closure = load(closure_path)
            actual = [{k: a[k] for k in ("path", "bytes", "sha256")} for a in closure.get("assets", [])]
            require(closure.get("asset_count") == 69 and sorted(actual, key=lambda a: a["path"])
                == sorted(assets, key=lambda a: a["path"]), "NATIVE_ASSET_CLOSURE_DRIFT")
    for row in rows:
        raw = row.get("raw_inputs")
        require(isinstance(raw, dict) and bool(raw), "NATIVE_CURRENT_RAW_INPUTS_REQUIRED")
        for ref in raw.values():
            raw_path = signed_ref(ref, root, verify=verify_inputs, local=False)
            require(not {p.lower() for p in raw_path.parts} & {"_run", "review", "processed", "attempts"},
                    "NATIVE_HISTORICAL_RAW_INPUT_FORBIDDEN")
        for key in (["native_signature"] if native_only else ["motion_signature", "render_signature"]):
            _signature(row, key)
        require(not any(k in row for k in ("stages", "mask_receipt", "clean_receipt")),
                "NATIVE_GEOMETRY_NO_MASK_CLEAN_OR_PER_SESSION_STAGE_CONFIG")
        if native_only:
            require("camera_contract" not in row, "NATIVE_ONLY_NO_CAMERA_FIT")
        else:
            camera = row.get("camera_contract")
            require(isinstance(camera, dict) and set(camera) == {"v1_manifest", "camera_config"},
                    "NATIVE_EXPLICIT_CAMERA_CONSUMER_CONTRACT_REQUIRED")
            for ref in camera.values():
                signed_ref(ref, root, verify=verify_inputs)
        if verify_inputs:
            _, report, _ = source_binding(row, root)
            require(report["actual_inputs"] == raw, "NATIVE_RAW_INPUTS_NOT_BOUND_SOURCE")


def stage_binding(row, name, root, *, verify=True):
    key = name.lower()
    _signature(row, key + "_signature")
    path = signed_ref(row.get(key + "_receipt"), root, verify=verify)
    if not verify:
        return None
    receipt = load(path)
    verify_sealed_stage(receipt, row[key + "_signature"], output_root=path.parent)
    n = row["native_frames"]
    require(receipt.get("stage") == name and (receipt.get("native_frames") == n
        or receipt.get("native_input_frames") == receipt.get("native_output_frames") == n),
        "NATIVE_UPSTREAM_STAGE_OR_FRAME_DRIFT:" + name)
    return receipt


def validate_native_clean_spec(spec, *, repo_root, verify_inputs=True):
    root = Path(repo_root).resolve(strict=True)
    rows, bounds = _base(spec, root, scope=NATIVE_CLEAN_SCOPE, schema=NATIVE_CLEAN_SCHEMA,
        prefix="inpainting_native_clean_", operation="run_inpainting_candidate_visual_stages_v1",
        gpu=1, verify=verify_inputs)
    clean_config = spec.get("clean_config")
    reference_experiment = (isinstance(clean_config, dict) and clean_config.get("reference_policy")
                            == "RAW_VISIBLE_FB_RANKED_REFERENCE_EXPERIMENT")
    require(spec.get("model_stages") == ["Clean"] and spec.get("actual_stage_scope") == NATIVE_CLEAN_SCOPE
            and spec["config"].get("stage") == ("NATIVE_REFERENCE_CLEAN" if reference_experiment
                                                 else "NATIVE_BASELINE_CLEAN"),
            "NATIVE_BASELINE_CLEAN_ONLY_REQUIRED")
    ready = [row for row in rows if row["upstream_status"] == "READY"]
    require(bounds.get("model_calls_unit") == "PROPAINTER_INPAINT_API_INVOCATION"
            and type(bounds.get("model_calls_max")) is int and bounds["model_calls_max"] == len(ready),
            "NATIVE_CLEAN_OUTER_CALL_BUDGET_REQUIRED")
    require(isinstance(clean_config, dict) and set(clean_config) == {"frozen_settings", "reference_policy", "candidates"}
        and clean_config["reference_policy"] in {"FROZEN_BASELINE", "RAW_VISIBLE_FB_RANKED_REFERENCE_EXPERIMENT"}
        and clean_config["candidates"] == (["B"] if reference_experiment else ["A"]),
        "NATIVE_CLEAN_BASELINE_REFERENCE_ONLY_REQUIRED")
    settings_file = root / "src/chaoyang/exact78_baseline/clean.py"
    if verify_inputs:
        # Read the existing literal, without importing the algorithm/CV module
        # before authority. This is equality to V1, not a new numerical gate.
        tree = ast.parse(settings_file.read_text(encoding="utf-8"))
        assignments = [n for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "SETTINGS" for t in n.targets)]
        require(len(assignments) == 1 and clean_config["frozen_settings"] == ast.literal_eval(assignments[0].value),
                "NATIVE_EXISTING_PROPAINTER_SETTINGS_DRIFT")
    require(isinstance(spec.get("model_weights"), list) and len(spec["model_weights"]) == 3,
            "NATIVE_FROZEN_THREE_PROPAINTER_WEIGHTS_REQUIRED")
    for ref in spec["model_weights"]:
        signed_ref(ref, root, verify=verify_inputs)
    for row in ready:
        mask = stage_binding(row, "Mask", root, verify=verify_inputs)
        _signature(row, "clean_signature")
        raw_path = signed_ref(row.get("raw_source_ref"), root, verify=verify_inputs, local=False)
        require(not {p.lower() for p in raw_path.parts} & {"_run", "review", "processed", "attempts"},
                "NATIVE_CLEAN_RAW_VIDEO_REQUIRED")
        if reference_experiment:
            capture_path = signed_ref(row.get("capture_config_ref"), root, verify=verify_inputs)
            if verify_inputs:
                capture = load(capture_path)
                source, _, _ = source_binding(row, root)
                outputs = direct_outputs(source)
                require(capture.get("schema") == "inpainting-raw-raft-capture-config-v1"
                    and capture.get("session_id") == row["session_id"]
                    and capture.get("source_receipt") == row["source_receipt"]
                    and capture.get("raw_source_ref") == row["raw_source_ref"]
                    and capture.get("raw_rgb_ref") == outputs["RAW_RGB.npz"]
                    and capture.get("frame_map_ref") == row["frame_map_ref"]
                    and capture.get("runtime_code") == spec["runtime_code"]
                    and capture.get("raft_weight_ref") in spec["model_weights"]
                    and capture.get("raft_iter") == clean_config["frozen_settings"]["raft_iter"]
                    and capture.get("chunk_length") == 12
                    and capture.get("preprocessing") == "RGB_UINT8_FLOAT32_B1_T3HW_DIV127_5_MINUS1",
                    "NATIVE_REFERENCE_CAPTURE_SOURCE_OR_RECIPE_DRIFT")
        if verify_inputs:
            _, report, mapping = source_binding(row, root)
            require(report["actual_inputs"]["video"] == row["raw_source_ref"]
                and mask.get("actual_inputs", {}).get("raw_source") == row["raw_source_ref"],
                "NATIVE_CLEAN_MASK_RAW_SOURCE_DRIFT")
            refs = direct_outputs(mask)
            require("FRAME_MAP.json" in refs and load(refs["FRAME_MAP.json"]["path"]) == mapping,
                    "NATIVE_CLEAN_MASK_SOURCE_MAP_DRIFT")


def validate_native_composite_spec(spec, *, repo_root, verify_inputs=True):
    root = Path(repo_root).resolve(strict=True)
    rows, bounds = _base(spec, root, scope=NATIVE_COMPOSITE_SCOPE, schema=NATIVE_COMPOSITE_SCHEMA,
        prefix="inpainting_native_composite_", operation="run_inpainting_candidate_composite_stage_v1",
        gpu=0, verify=verify_inputs)
    require(spec.get("stages") == ["Composite"] and bounds.get("render_calls_max") == 0,
            "NATIVE_COMPOSITE_NO_NEW_RENDER_REQUIRED")
    require(spec["config"] == {"scene_depth": "ABSENT", "display_policy": "RAW_MODEL_OBJECT_FIRST_DIAGNOSTIC",
        "image_size": [640, 480], "complete_robot": True}, "NATIVE_COMPOSITE_FROZEN_DISPLAY_REQUIRED")
    for row in rows:
        if row["upstream_status"] == "BLOCKED":
            continue
        _signature(row, "composite_signature")
        values = {name: stage_binding(row, name, root, verify=verify_inputs) for name in ("Mask", "Clean", "Render")}
        quality = row.get("upstream_quality_evidence")
        require(isinstance(quality, dict) and set(quality) <= {"Mask", "Clean"},
                "NATIVE_COMPACT_UPSTREAM_QUALITY_REQUIRED")
        for name, ref in quality.items():
            path = signed_ref(ref, root, verify=verify_inputs)
            if verify_inputs:
                value = load(path)
                require(value.get("schema") == "inpainting-upstream-quality-status-v1" and value.get("stage") == name
                    and value.get("upstream_receipt") == row[name.lower() + "_receipt"]
                    and value.get("visual_quality") == "FAILED"
                    and value.get("human_review") == "PENDING" and value.get("hidden_regions") == "UNKNOWN"
                    and not any(k in value for k in ("points", "annotations", "pixel_points", "manual_seed")),
                    "NATIVE_COMPACT_QUALITY_PROVENANCE_REQUIRED")
                signed_ref(value.get("review_reference"), root, verify=True)
        if verify_inputs:
            _, report, mapping = source_binding(row, root)
            raw = report["actual_inputs"]["video"]
            require(values["Mask"].get("actual_inputs", {}).get("raw_source") == raw
                and values["Clean"].get("actual_inputs", {}).get("raw_source") == raw
                and values["Clean"].get("actual_inputs", {}).get("mask_receipt") == row["mask_receipt"],
                "NATIVE_COMPOSITE_EXACT_NEW_MASK_CLEAN_REQUIRED")
            require(values["Render"].get("source_receipt") == row["source_receipt"]
                and values["Render"].get("source_signature") == row["source_signature"]
                and values["Render"].get("complete_robot") is True, "NATIVE_COMPOSITE_COMPLETE_ROBOT_SOURCE_REQUIRED")
            for name, receipt in values.items():
                refs = direct_outputs(receipt)
                require("FRAME_MAP.json" in refs and load(refs["FRAME_MAP.json"]["path"]) == mapping,
                        "NATIVE_COMPOSITE_FULL_UPSTREAM_MAP_REQUIRED:" + name)
