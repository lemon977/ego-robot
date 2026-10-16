"""Registered original-V1 M/C/O consumers; never an implicit CLI or adoption."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json
from .contracts import verify_ref

RESULT_SCHEMA = "inpainting-v1-mco-stage-result-v1"


def _runner():
    path = REPO_ROOT / "scripts/maintenance/inpainting_v1_mco_task.py"
    definition = importlib.util.spec_from_file_location("_v1_mco_worker_fence", path)
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def _authority(packet):
    runner = _runner()
    infra, helpers = runner._infra(), runner._helpers()

    def check():
        from datetime import datetime, timezone
        _, state, index = infra._snapshot()
        row, _ = helpers.own_row(packet, state, index)
        if (row["status"] != "RUNNING" or packet.get("algorithm_execution_allowed") is not True
                or datetime.now(timezone.utc) >= runner._deadline(packet["deadline_at"])):
            raise ValueError("MCO_RUNNING_FENCE_OR_DEADLINE")
        writer = packet["writer"]
        result = {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
                  "fencing_token": writer["fencing_token"], "execution_allowed": True,
                  "algorithm_execution_allowed": True}
        cpu_clean = (packet.get("config", {}).get("cpu_raw_donor", False)
                     or packet.get("config", {}).get("cpu_spatial_clean", False)
                     or packet.get("config", {}).get("cpu_context_seam", False))
        cpu_equipment = packet.get("config", {}).get("cpu_equipment_rgb", False)
        if cpu_equipment and (packet["mode"] != "Mask" or any(packet['budgets'].get(k) != 0 for k in
                ('gpu_count','model_calls_max','render_calls_max','training_steps_max'))):
            raise ValueError('MCO_CPU_EQUIPMENT_RESOURCE_CONTRACT')
        if cpu_clean and (packet["mode"] != "Clean" or packet["budgets"].get("gpu_count") != 0
                or packet["budgets"].get("model_calls_max") != 0
                or packet["budgets"].get("render_calls_max") != 0):
            raise ValueError("MCO_CPU_DONOR_RESOURCE_CONTRACT")
        if packet["mode"] != "Boundary" and not cpu_clean and not cpu_equipment:
            from .resources import CANDIDATE_CARD_LOCK_ROOT
            mapping = packet["gpu_mapping"]
            lease = json.loads((CANDIDATE_CARD_LOCK_ROOT / (mapping["uuid"] + ".json")).read_text())
            expected = {"task_id": packet["task_id"], "run_signature": packet["run_signature"],
                "fencing_token": writer["fencing_token"], "pid": writer["pid"],
                "proc_start_ticks": writer["proc_start_ticks"], "uuid": mapping["uuid"],
                "physical_index": mapping["physical_index"], "worker_logical_index": 0}
            if (any(lease.get(k) != v for k, v in expected.items())
                    or os.environ.get("CUDA_VISIBLE_DEVICES") != mapping["uuid"]):
                raise ValueError("MCO_CARD_LEASE_DRIFT")
            result.update(expected, gpu_uuid=mapping["uuid"])
        return result
    return check


def _original(reader, row):
    from chaoyang.pick_place_1008.visual import effective_clean_masks
    original = row["original_row"]
    if original["session_id"] != row["session_key"] or original["original_native_frames"] != row["native_frames"]:
        raise ValueError("MCO_ORIGINAL_SESSION_IDENTITY")
    refs = original["stage_files"]["gpu"]
    raw = reader.arrays(refs["raw_rgb.npz"])["rgb"]
    clean = reader.arrays(refs["clean_rgb.npz"])["rgb"]
    n, h, w = raw.shape[:3]
    if n != row["native_frames"] or (h, w) != (480, 640) or raw.dtype != np.uint8 or clean.shape != raw.shape:
        raise ValueError("MCO_ORIGINAL_NATIVE_RGB_DOMAIN")
    packed = reader.arrays(refs["masks.npz"])
    masks = {name: np.unpackbits(value, axis=1, count=h * w).reshape(n, h, w).astype(bool)
             for name, value in packed.items()}
    effective, _ = effective_clean_masks(masks["removal"], masks["protected_visible"])
    if not np.array_equal(effective, masks["effective_removal"]):
        raise ValueError("MCO_ORIGINAL_EFFECTIVE_MASK_DRIFT")
    meta = reader.read_json(refs["masks.json"])
    session = reader.load_prepared(Path(original["source_session"]) / "source")
    if session.frame_count != n or len(session.qpc_ns) != n:
        raise ValueError("MCO_ORIGINAL_NATIVE_TIME_DOMAIN")
    mapping = [{"native_frame_index": i, "output_frame": i,
                "qpc_declared_ns": int(session.qpc_ns[i]),
                "timebase_status": "TIMEBASE_UNVERIFIED_HOST_ARRIVAL_QPC_DECLARED_NS"}
               for i in range(n)]
    return raw, clean, masks, meta, session, mapping


def _label(image, text):
    from PIL import Image, ImageDraw, ImageFont
    font_path = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    if not font_path.is_file():
        raise ValueError("MCO_CHINESE_REVIEW_FONT_ABSENT")
    output = Image.fromarray(image)
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, image.shape[1], 31), fill=(0, 0, 0))
    draw.text((6, 3), text, font=ImageFont.truetype(str(font_path), 21), fill=(255, 255, 255))
    return np.asarray(output)


def _comparison(left, right, indices, sid, left_name, right_name):
    return np.stack([np.concatenate((
        _label(a, f"{sid} 第{frame}帧｜{left_name}"),
        _label(b, f"{sid} 第{frame}帧｜{right_name}")), axis=1)
        for frame, a, b in zip(indices, left, right)])


def _require_sealed_input(array_ref, result_ref, sid, mode):
    from chaoyang.humanego_ict.project_mount_paths import resolve_input_path
    if not isinstance(result_ref, dict):
        raise ValueError("MCO_SEALED_SUCCESSOR_RESULT_REQUIRED")
    for ref in (array_ref, result_ref):
        resolve_input_path(ref["path"], REPO_ROOT)
        verify_ref(ref)
    receipt = json.loads(Path(result_ref["path"]).read_text())
    if (receipt.get("schema") != RESULT_SCHEMA or receipt.get("status") != "SEALED_COMPLETE"
            or receipt.get("mode") != mode or receipt.get("candidate_adopted") is not False
            or receipt.get("control_ground_truth") is not False
            or array_ref not in receipt.get("artifacts", [])):
        raise ValueError("MCO_SEALED_SUCCESSOR_SIGNATURE_OR_ARTIFACT")
    rows = [row for row in receipt["sessions"] if row["session_key"] == sid]
    if (len(rows) != 1 or rows[0].get("execution_complete") is not True
            or rows[0]["completed_frame_indices"] != list(range(rows[0]["native_frames"]))):
        raise ValueError("MCO_SUCCESSOR_PARTIAL_INPUT_FORBIDDEN")


def _check_successor_masks(original, candidate, shape):
    from chaoyang.pick_place_1008.visual import effective_clean_masks
    required = {k for k, v in original.items() if isinstance(v, np.ndarray)}
    if not required.issubset(candidate) or any(candidate[k].shape != shape or candidate[k].dtype != bool for k in required):
        raise ValueError("MCO_SUCCESSOR_MASK_NATIVE_BOOL_DOMAIN")
    for role in ("equipment", "protected_visible", "unknown", "cup_observed", "semantic_conflict"):
        if not np.array_equal(candidate[role], original[role]):
            raise ValueError("MCO_SUCCESSOR_ORIGINAL_MASK_SUPPORT_DRIFT:" + role)
    for role in ("core", "sam_core", "removal"):
        if np.any(original[role] & ~candidate[role]):
            raise ValueError("MCO_SUCCESSOR_ORIGINAL_SUPPORT_REMOVED:" + role)
    effective, _ = effective_clean_masks(candidate["removal"], candidate["protected_visible"])
    if not np.array_equal(effective, candidate["effective_removal"]):
        raise ValueError("MCO_SUCCESSOR_EFFECTIVE_MASK_SIGNATURE")


def _verify_frozen_tree(value):
    if isinstance(value, dict):
        if set(value) == {"path", "bytes", "sha256"}:
            verify_ref(value)
        else:
            for item in value.values():
                _verify_frozen_tree(item)
    elif isinstance(value, list):
        for item in value:
            _verify_frozen_tree(item)


def _require_pair_source_inputs(packet, raw, session):
    from chaoyang.humanego_ict.project_mount_paths import resolve_input_path
    for name in ("frozen_source_raw", "frozen_frame_map"):
        if name not in packet["inputs"]:
            raise ValueError("MCO_PAIR_FROZEN_SOURCE_AND_FRAME_MAP_REQUIRED")
    source_ref, frame_ref = packet["inputs"]["frozen_source_raw"], packet["inputs"]["frozen_frame_map"]
    source_path = resolve_input_path(source_ref["path"], REPO_ROOT)
    frame_path = resolve_input_path(frame_ref["path"], REPO_ROOT)
    verify_ref(source_ref)
    verify_ref(frame_ref)
    with np.load(source_path, allow_pickle=False) as archive:
        if archive.files != ["rgb"] or not np.array_equal(archive["rgb"], raw):
            raise ValueError("MCO_PAIR_FROZEN_SOURCE_RGB_MISMATCH")
    source_mapping = json.loads(frame_path.read_text())
    if (not isinstance(source_mapping, list) or len(source_mapping) != len(raw)
            or any(m.get("native_frame_index") != i
                   or m.get("output_frame") != i
                   or m.get("qpc_declared_ns") != int(session.qpc_ns[i])
                   for i, m in enumerate(source_mapping))):
        raise ValueError("MCO_PAIR_FROZEN_SOURCE_FRAME_MAP_MISMATCH")
    return source_ref, frame_ref


def _require_mask_causal_inputs(packet, original):
    """Validate both sealed Mask inputs against original V1 before model load."""
    from chaoyang.humanego_ict.project_mount_paths import resolve_input_path

    raw, _, original_masks, _, session, _ = original
    row = packet["sessions"][0]
    if len(raw) != row["native_frames"] or len(session.qpc_ns) != len(raw):
        raise ValueError("MCO_MASK_CAUSAL_NATIVE_FRAME_DOMAIN")
    source_ref, frame_ref = _require_pair_source_inputs(packet, raw, session)
    names = ("frozen_base_masks", "frozen_repaired_masks", "frozen_mask_receipt",
             "frozen_source_receipt", "original_v1_masks", "original_gpu_receipt")
    if any(not isinstance(packet["inputs"].get(name), dict) for name in names):
        raise ValueError("MCO_MASK_CAUSAL_FROZEN_MASK_INPUTS_REQUIRED")
    references = {name: packet["inputs"][name] for name in names}
    paths = {}
    for name, ref in references.items():
        paths[name] = resolve_input_path(ref["path"], REPO_ROOT)
        verify_ref(ref)

    def same_ref(left, right):
        return (isinstance(left, dict) and isinstance(right, dict)
                and isinstance(left.get("path"), str) and isinstance(right.get("path"), str)
                and left.get("sha256") == right.get("sha256") and left.get("bytes") == right.get("bytes")
                and resolve_input_path(left["path"], REPO_ROOT) == resolve_input_path(right["path"], REPO_ROOT))

    if (not same_ref(references["original_v1_masks"], row["original_row"]["stage_files"]["gpu"]["masks.npz"])
            or not same_ref(references["original_gpu_receipt"], row["original_row"]["stage_receipts"]["gpu"])):
        raise ValueError("MCO_MASK_CAUSAL_ORIGINAL_PRODUCER_BINDING")
    original_receipt = json.loads(paths["original_gpu_receipt"].read_text())
    if (original_receipt.get("schema") != "pick_place_1008_sealed_stage_v1"
            or original_receipt.get("status") != "COMPLETE"
            or original_receipt.get("binding", {}).get("session_id") != row["session_key"]
            or original_receipt.get("binding", {}).get("stage") != "gpu"
            or not same_ref(original_receipt.get("files", {}).get("masks.npz", {}), references["original_v1_masks"])):
        raise ValueError("MCO_MASK_CAUSAL_ORIGINAL_GPU_RECEIPT_REQUIRED")
    source_receipt = json.loads(paths["frozen_source_receipt"].read_text())
    if (source_receipt.get("schema") != "inpainting-candidate-component-v1"
            or source_receipt.get("status") != "SEALED_COMPLETE" or source_receipt.get("stage") != "Source"
            or source_receipt.get("session_id") != row["session_key"]
            or source_receipt.get("native_frames") != len(raw)
            or source_receipt.get("control_ground_truth") is not False
            or source_receipt.get("candidate_adopted") is not False
            or any(not any(same_ref(ref, output) for output in source_receipt.get("outputs", []))
                   for ref in (source_ref, frame_ref))):
        raise ValueError("MCO_MASK_CAUSAL_SEALED_SOURCE_RECEIPT_REQUIRED")
    receipt = json.loads(paths["frozen_mask_receipt"].read_text())
    if (receipt.get("schema") != "inpainting-candidate-component-v1"
            or receipt.get("status") != "SEALED_COMPLETE" or receipt.get("stage") != "Mask"
            or receipt.get("session_id") != row["session_key"]
            or receipt.get("native_frames") != len(raw)
            or not same_ref(receipt.get("upstream", {}).get("Source", {}), references["frozen_source_receipt"])
            or receipt.get("candidate_adopted") is not False
            or receipt.get("control_ground_truth") is not False):
        raise ValueError("MCO_MASK_CAUSAL_SEALED_MASK_RECEIPT_REQUIRED")
    for name in names[:2]:
        ref = references[name]
        if not any(output.get("sha256") == ref["sha256"] and output.get("bytes") == ref["bytes"]
                   and resolve_input_path(output["path"], REPO_ROOT) == paths[name]
                   for output in receipt.get("outputs", [])):
            raise ValueError("MCO_MASK_CAUSAL_MASK_NOT_IN_SEALED_RECEIPT:" + name)
    with np.load(paths["frozen_base_masks"], allow_pickle=False) as archive:
        base = {name: archive[name] for name in archive.files}
    with np.load(paths["frozen_repaired_masks"], allow_pickle=False) as archive:
        repaired = {name: archive[name] for name in archive.files}
    required = {name for name, value in original_masks.items() if isinstance(value, np.ndarray)}
    base_kind = packet["config"].get("base_mask_kind", "EXACT_ORIGINAL")
    if base_kind == "SEALED_V1_MASK_SUCCESSOR":
        _check_successor_masks(original_masks, base, raw.shape[:3])
    elif base_kind == "EXACT_ORIGINAL":
        if (not required.issubset(base) or any(base[name].dtype != bool
                or base[name].shape != raw.shape[:3]
                or not np.array_equal(base[name], original_masks[name]) for name in required)):
            raise ValueError("MCO_MASK_CAUSAL_BASE_NOT_EXACT_ORIGINAL_V1")
    else:
        raise ValueError("MCO_MASK_CAUSAL_BASE_KIND_UNSUPPORTED")
    _check_successor_masks(base, repaired, raw.shape[:3])
    return base, repaired, {**references, "frozen_source_raw": source_ref,
                            "frozen_frame_map": frame_ref}


def execute_v1_mco_stage(packet: dict) -> int:
    """Called only after the registered operation's before-import fence."""
    if packet.get('config',{}).get('cpu_context_seam') is True:
        from .v1_context_seam import execute_candidate
        return execute_candidate(packet,authority=_authority(packet))
    if packet.get('config', {}).get('context_exclusion_pair') is True:
        from .v1_context_exclusion import execute
        return execute(packet,authority=_authority(packet))
    if packet.get('config', {}).get('cpu_equipment_rgb') is True:
        from .v1_equipment_rgb import execute
        return execute(packet,authority=_authority(packet))
    if packet.get("config", {}).get("generic_object_probe") is True:
        from .v1_device_proposals import execute_generic_object_probe
        return execute_generic_object_probe(packet,authority=_authority(packet))
    if packet.get("config", {}).get("cpu_spatial_clean") is True:
        from .v1_spatial_clean import execute_spatial_clean
        return execute_spatial_clean(packet, authority=_authority(packet))
    if packet.get("config", {}).get("cpu_raw_donor") is True:
        from .v1_scene_donor_worker import execute_raw_donor
        return execute_raw_donor(packet, authority=_authority(packet))
    if packet.get("mode") == "Automation":
        _verify_frozen_tree(packet.get("component_config", {}))
        from .automation import execute_automation
        builders=None
        if packet.get('component_config',{}).get('Clean',{}).get('write_boundary_policy') is not None:
            from .v1_boundary_builder import BoundaryContextBuilders
            builders=BoundaryContextBuilders(packet,_authority(packet)).builders()
        elif packet.get('component_config',{}).get('Clean',{}).get('reference_policy') == 'EXCLUDE_UNCERTAIN_EQUIPMENT_RGB':
            from .v1_context_builder import ContextExclusionBuilders
            builders=ContextExclusionBuilders(packet,_authority(packet)).builders()
        result = execute_automation(packet, authorization=_authority(packet), **({'builders':builders}if builders is not None else {}))
        if any(type(result.get(name)) is not int for name in ("model_calls", "render_calls")):
            raise ValueError("MCO_AUTOMATION_ACTUAL_CALL_COUNTS_REQUIRED")
        rows = [{**row, "session_key": row["session_id"],
            "native_output_frames": row["native_frames"],
            "completed_frame_indices": list(range(row["native_frames"])),
            "execution_complete": row.get("execution") == "SEALED_COMPLETE_ALL_SIX_STAGES"}
            for row in result["sessions"]]
        # A full six-stage resume legitimately produces no new pixel arrays.
        # Its own sealed graph is a real artifact in this Task's write set;
        # reused receipts remain inputs, not falsely claimed new output files.
        graph_path=Path(packet['output_dir'])/'AUTOMATION_STAGE.json'
        if not graph_path.is_file() or json.loads(graph_path.read_text())!=result:
            raise ValueError('MCO_AUTOMATION_SEALED_GRAPH_REQUIRED')
        artifacts = [*result.get("artifacts", []),artifact_ref(graph_path)]
        sealed = {**result, "schema": RESULT_SCHEMA, "mode": "Automation", "sessions": rows,
                  "automation_result_schema": result["schema"],"artifacts":artifacts,
                  "reused_pixels_are_new_outputs":False}
        path = Path(packet["output_dir"]) / "MODEL_RESULT.json"
        if path.exists():
            raise ValueError("MCO_FRESH_MODEL_RESULT_REQUIRED")
        atomic_json(path, sealed)
        return 0
    from .model_stages import configure_project_caches, load_frozen_models
    from .source_frames import write_review_video
    from chaoyang.storage_relocation.pick_place_original199_v1 import Original199Reader
    from chaoyang.pick_place_1008 import visual
    from .v1_clean_ablation import make_pixel_propagation_ablation
    from .v1_boundary_alpha import compose_support_clipped

    mode, root = packet["mode"], Path(packet["output_dir"])
    if mode not in {"Clean", "Boundary", "Mask"}:
        raise ValueError("MCO_MODE_REQUIRED")
    paired_clean = packet["config"].get("paired_clean", False)
    if type(paired_clean) is not bool:
        raise ValueError("MCO_PAIRED_CLEAN_BOOL_REQUIRED")
    if paired_clean and (mode != "Clean" or len(packet["sessions"]) != 1
            or "output_frames" in packet["sessions"][0]
            or packet["sessions"][0].get("mask_input") is None
            or packet["sessions"][0].get("mask_result") is None
            or packet["budgets"]["model_calls_max"] != 2):
        raise ValueError("MCO_PAIRED_CLEAN_FROZEN_FULL_NATIVE_TWO_CALLS_REQUIRED")
    mask_causal_pair = packet["config"].get("mask_causal_pair", False)
    if type(mask_causal_pair) is not bool:
        raise ValueError("MCO_MASK_CAUSAL_PAIR_BOOL_REQUIRED")
    context_clean_pair = packet["config"].get("context_clean_pair", False)
    if type(context_clean_pair) is not bool:
        raise ValueError("MCO_CONTEXT_PAIR_BOOL_REQUIRED")
    if context_clean_pair and (paired_clean or mask_causal_pair or mode != "Clean"
            or len(packet["sessions"]) != 1 or "output_frames" in packet["sessions"][0]
            or packet["config"].get("pixel_propagation") != "FROZEN"
            or packet["config"].get("context_experiment") not in {"FULL_SESSION_EQUAL_COUNT", "REVERSE_NATIVE_TIME_ORDER", "TRACE_UNCHANGED_ORIGINAL_PATH", "FULL_NATIVE_IMAGE_PROPAGATION_CONTEXT"}
            or packet["budgets"].get("model_calls_max") != 3
            or packet["budgets"].get("render_calls_max") != 0):
        raise ValueError("MCO_CONTEXT_PAIR_FIXED_MASK_FULL_NATIVE_THREE_CALLS_REQUIRED")
    if mask_causal_pair and (paired_clean or mode != "Clean" or len(packet["sessions"]) != 1
            or "output_frames" in packet["sessions"][0]
            or packet["config"].get("pixel_propagation") != "FROZEN"
            or packet["budgets"].get("model_calls_max") != 3
            or packet["budgets"].get("render_calls_max") != 0
            or packet["budgets"].get("training_steps_max") != 0):
        raise ValueError("MCO_MASK_CAUSAL_FULL_NATIVE_ORIGINAL_PP_THREE_CALLS_REQUIRED")
    configure_project_caches(root)
    authority = _authority(packet)
    authority()
    reader = Original199Reader(packet["inputs"]["storage_binding"])
    manifest = reader.read_json(reader.binding["source_manifest"])
    original_rows = {row["session_id"]: row for row in manifest["sessions"]}
    for row in packet["sessions"]:
        if row["original_row"] != original_rows.get(row["session_key"]):
            raise ValueError("MCO_ORIGINAL_ROW_NOT_FROZEN_MANIFEST")
    for ref in [*packet["runtime_code"], *packet["model_weights"]]:
        verify_ref(ref)
    for pin in packet["inputs"].values():
        verify_ref(pin)
    if "frozen_inputs" in packet["inputs"]:
        _verify_frozen_tree(json.loads(Path(packet["inputs"]["frozen_inputs"]["path"]).read_text()))
    _label(np.zeros((32, 64, 3), np.uint8), "原V1")
    review = Path(packet["review_dir"])
    review.mkdir(parents=True, exist_ok=True)
    artifacts, rows, model_calls, render_calls = [], [], 0, 0
    models = None

    def save_json(path, value):
        if path.exists():
            raise ValueError("MCO_FRESH_ARTIFACT_REQUIRED:" + str(path))
        atomic_json(path, value)
        artifacts.append(artifact_ref(path))

    def save_npz(path, **arrays):
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        artifacts.append(artifact_ref(path))

    def video(path, pixels, mapping):
        info = write_review_video(path, pixels, mapping)
        artifacts.append(info["video"])
        return info

    sam_budget, pair_backend_before_load, pair_original = None, None, None
    causal_inputs = None
    if mode in {"Clean", "Mask"}:
        if paired_clean or mask_causal_pair or context_clean_pair:
            pair_original = _original(reader, packet["sessions"][0])
            if mask_causal_pair or context_clean_pair:
                causal_inputs = _require_mask_causal_inputs(packet, pair_original)
                if context_clean_pair and (causal_inputs[0].keys() != causal_inputs[1].keys()
                        or any(not np.array_equal(causal_inputs[0][k],causal_inputs[1][k])
                               for k in causal_inputs[0])):
                    raise ValueError("MCO_CONTEXT_PAIR_MASK_CHANGE_FORBIDDEN")
            else:
                source_ref, frame_ref = _require_pair_source_inputs(packet, pair_original[0], pair_original[4])
            import torch
            from .v1_clean_pair import runtime_backend, require_same_backend
            pair_backend_before_load = runtime_backend(torch, "cuda")
            save_json(root / "BACKEND_BEFORE_MODEL_LOAD.json", pair_backend_before_load)
        models = load_frozen_models(REPO_ROOT, root, authorization=authority,
            frozen_inputs=[*packet["runtime_code"], *packet["model_weights"]],
            need_sam=mode == "Mask", need_clean=mode == "Clean")
        if paired_clean or mask_causal_pair or context_clean_pair:
            after_load = runtime_backend(models["propainter"].torch, models["propainter"].device)
            require_same_backend(pair_backend_before_load, after_load)
            if "sam" in models:
                raise ValueError("MCO_PAIRED_CLEAN_SAM_LOAD_FORBIDDEN")
            save_json(root / "BACKEND_AFTER_MODEL_LOAD.json", after_load)
        save_json(root / "MODEL_LOAD.json", models["load_receipt"])
        if mode == "Mask":
            from .model_worker import CountedSam, ModelCallBudget, MODEL_CALLS_UNIT
            sam_budget = ModelCallBudget(packet["budgets"]["model_calls_max"], MODEL_CALLS_UNIT)
            models["sam"] = CountedSam(models["sam"], sam_budget)
    else:
        from chaoyang.exact78_baseline import render as rd
        from chaoyang.exact78_baseline.pipeline import _robot
        renderer = rd.RobotRenderer(_robot(REPO_ROOT))

    for row in packet["sessions"]:
        authority()
        started = time.monotonic()
        sid = row["session_key"]
        out = root / "sessions" / sid / mode
        out.mkdir(parents=True, exist_ok=False)
        raw, old_clean, masks, mask_meta, session, mapping = (pair_original
            if paired_clean or mask_causal_pair or context_clean_pair else _original(reader, row))
        save_json(out / "ACTUAL_INPUTS.json", {"storage_binding": packet["inputs"]["storage_binding"],
            "original_row": row["original_row"], "runtime_code": packet["runtime_code"],
            "model_weights": packet["model_weights"], "config": packet["config"],
            "mask_input": row.get("mask_input"), "mask_result": row.get("mask_result"),
            "paired_clean": paired_clean,
            "old_automatic_candidate_consumed": False, "manual_points_added": 0,
            "candidate_adopted": False, "control_ground_truth": False})
        all_indices = list(range(row["native_frames"]))
        anchors = row.get("review_anchors", row.get("anchors", [0, len(raw) // 2, len(raw) - 1]))
        masks["metadata"] = mask_meta
        if row.get("mask_input") is not None and not mask_causal_pair and not context_clean_pair:
            mask_result = row.get("mask_result")
            _require_sealed_input(row["mask_input"], mask_result, sid, "Mask")
            with np.load(row["mask_input"]["path"], allow_pickle=False) as archive:
                new_masks = {name: archive[name] for name in archive.files}
            _check_successor_masks(masks, new_masks, raw.shape[:3])
            masks = {**new_masks, "metadata": mask_meta}
        if mode == "Mask":
            from .v1_mask_repair import (make_repair_prompts, merge_repair_support,
                                         qualify_identity_support, REPAIR_AUTHORITY)
            from .automatic_masks import RGBProposal, frames_sha256
            from .contracts import canonical_sha
            from chaoyang.pick_place_1008.sam_point_adapter import track_visible_points
            ready_ref = row["mask_ready_input"]
            verify_ref(ready_ref)
            ready = json.loads(Path(ready_ref["path"]).read_text())
            if (ready["session_key"] != sid or ready["native_frames"] != len(raw)
                    or ready["storage_binding"] != packet["inputs"]["storage_binding"]
                    or ready["original_masks"]["masks.npz"] != row["original_row"]["stage_files"]["gpu"]["masks.npz"]):
                raise ValueError("MCO_MASK_READY_IDENTITY_DRIFT")
            for ref in ready["file_refs_verified"]:
                reader.verify_ref(ref)
            with np.load(ready["source_rgb"]["path"], allow_pickle=False) as archive:
                if not np.array_equal(raw, archive["rgb"]):
                    raise ValueError("MCO_READY_RGB_ORIGINAL_V1_MISMATCH")
            original_mapping = json.loads(Path(ready["frame_map"]["path"]).read_text())
            binding = ready["input_bindings"]
            from .model_worker import sam_weight_reference
            source_input = reader.read_json(row["original_row"]["source_input"])
            if (binding["frames_sha256"] != frames_sha256(raw)
                    or binding["frame_map_sha256"] != canonical_sha(original_mapping)
                    or binding["raw_source_sha256"] != source_input["source_refs"]["raw/vst.h264"]["sha256"]
                    or binding["model_sha256"] != sam_weight_reference(packet["model_weights"])["sha256"]
                    or [r["qpc_declared_ns"] for r in original_mapping] != session.qpc_ns.tolist()):
                raise ValueError("MCO_READY_CURRENT_RGB_TIME_SIGNATURE_MISMATCH")
            text_meta = json.loads(Path(ready["text_proposals"]["path"]).read_text())
            if text_meta.get("arrays") != ready["text_arrays"]:
                raise ValueError("MCO_READY_TEXT_ARRAY_BINDING_MISMATCH")
            proposals = []
            with np.load(ready["text_arrays"]["path"], allow_pickle=False) as archive:
                for item in text_meta["proposals"]:
                    shape = tuple(item["native_shape"])
                    if shape != raw.shape[:3]:
                        raise ValueError("MCO_READY_TEXT_NATIVE_DOMAIN")
                    track = np.unpackbits(archive[item["array_key"]], axis=1,
                        count=shape[1] * shape[2]).reshape(shape).astype(bool)
                    unsupported_key = item.get("unsupported_frames_array_key")
                    unsupported = archive[unsupported_key].astype(bool) if unsupported_key else np.zeros(len(raw), bool)
                    proposals.append(RGBProposal(role=item["role"], track_id=item["track_id"], masks=track,
                        source=item["source"], unsupported_frames=unsupported,
                        **{key: item[key] for key in binding}))
            prompt_policy = packet["config"].get("repair_prompt_policy", "SINGLE_MAX_GAP")
            if ready.get("repair_prompt_policy", "SINGLE_MAX_GAP") != prompt_policy:
                raise ValueError("MCO_READY_PROMPT_POLICY_MISMATCH")
            unqualified_prompts, _ = make_repair_prompts(masks, proposals, ready["assignments"],
                input_bindings=binding, prompt_policy=prompt_policy)
            if unqualified_prompts != ready["unqualified_prompts"]:
                raise ValueError("MCO_READY_UNQUALIFIED_PROMPT_REPLAY_MISMATCH")
            verify_ref(ready["identity_hints"])
            with np.load(ready["identity_hints"]["path"], allow_pickle=False) as archive:
                hints = archive["xy"]
            proposals, qualification = qualify_identity_support(
                proposals, ready["assignments"], hints, binding)
            qualification_ref = ready["identity_qualification"]["report"]
            verify_ref(qualification_ref)
            sealed_qualification = json.loads(Path(qualification_ref["path"]).read_text())
            if canonical_sha(qualification) != canonical_sha(sealed_qualification):
                raise ValueError("MCO_READY_FRAME_COMPONENT_IDENTITY_REPLAY_MISMATCH")
            save_json(out / "IDENTITY_QUALIFICATION.json", qualification)
            prompts, prompt_meta = make_repair_prompts(masks, proposals, ready["assignments"],
                input_bindings=binding, prompt_policy=prompt_policy)
            if prompts != ready["prompts"]:
                raise ValueError("MCO_READY_AUTOMATIC_PROMPT_REPLAY_MISMATCH")
            authority()
            observer = None
            point_sam = models["sam"]
            if packet["config"].get("observe_point_outputs") is True:
                from .v1_point_observer import PointOutputObserver
                observer = PointOutputObserver(point_sam, tuple(raw.shape[:3]))
                point_sam = observer.sam
            tracks, point_meta = track_visible_points(point_sam, raw, prompts,
                min((p["frame_idx"] for p in prompts), default=0))
            authority()
            point_meta.update(seed_authority=REPAIR_AUTHORITY, human_visible_confirmed=False,
                manual_points_added=0, automatic_points=sum(len(p["points_px"]) for p in prompts),
                hardware_pixels_used_as_evidence=False)
            save_json(out / "AUTOMATIC_PROMPTS.json", {"prompts": prompts, "metadata": prompt_meta,
                "unqualified_prompts_before_identity_check": unqualified_prompts,
                "unqualified_point_count": sum(len(p["points_px"]) for p in unqualified_prompts),
                "qualified_point_count": sum(len(p["points_px"]) for p in prompts),
                "point_changes_are_identity_rejections_not_manual_edits": True})
            save_json(out / "POINT_RESULT.json", point_meta)
            if observer is not None:
                adapter_arrays, adapter_objects = observer.packed_adapter_tracks(tracks)
                save_npz(out / "PUBLIC_POINT_OUTPUTS.npz", **observer.arrays)
                save_npz(out / "ADAPTER_POINT_TRACKS.npz", **adapter_arrays)
                save_json(out / "POINT_OBSERVER.json", observer.receipt(adapter_objects))
            if point_meta.get("errors") or point_meta.get("aborted"):
                raise ValueError("MCO_POINT_PROPAGATION_FAILED")
            new_masks, merge = merge_repair_support(masks, tracks, proposals, prompts,
                ready["assignments"], input_bindings=binding, prompt_policy=prompt_policy)
            save_npz(out / "MASK_ARRAYS.npz", **{k: v for k, v in new_masks.items()
                if isinstance(v, np.ndarray) and v.dtype == np.bool_ and v.shape == raw.shape[:3]})
            save_json(out / "MASK_MERGE.json", merge)
            def overlay(selected):
                pixels = raw.copy()
                support, protect = selected["removal"], selected["protected_visible"]
                pixels[support] = (.4 * pixels[support] + .6 * np.array([255, 0, 255])).astype(np.uint8)
                pixels[protect] = (.4 * pixels[protect] + .6 * np.array([255, 160, 0])).astype(np.uint8)
                return pixels
            left, right, indices = overlay(masks), overlay(new_masks), all_indices
            completed = video(out / "mask_review.mp4", right, mapping)
            frame_states = [{**m, "mask_state": "RGB_POINT_SUPPORTED_V1_REPAIR_CANDIDATE",
                "automatic_point_authority": REPAIR_AUTHORITY, "hidden_regions": "UNKNOWN",
                "manual_points_added": 0, "human_review": "PENDING", "training_eligible": False,
                "control_ground_truth": False} for m in mapping]
            title = "本轮候选Mask：紫色删除／橙色保护"
            model_calls = sam_budget.total
        elif mode == "Clean" and context_clean_pair:
            from .v1_clean_context import run_context_clean_pair
            from .v1_clean_pair import comparison_outside_titles
            fixed_masks, _, context_refs = causal_inputs
            def context_call(variant):
                nonlocal model_calls
                authority()
                if model_calls >= packet["budgets"]["model_calls_max"]:
                    raise ValueError("MCO_MODEL_CALL_BUDGET_EXHAUSTED")
                model_calls += 1
            save_npz(out / "FROZEN_INPUT_ARRAYS.npz", raw_rgb=raw,
                removal=fixed_masks["removal"], protected_visible=fixed_masks["protected_visible"],
                effective_removal=fixed_masks["effective_removal"])
            experiment = packet["config"]["context_experiment"]
            pair = run_context_clean_pair(models["propainter"],raw,fixed_masks["removal"],
                fixed_masks["protected_visible"], experiment=experiment, authority=authority,
                on_call=context_call, backend_before_load=pair_backend_before_load)
            output_names = {"A":"CLEAN_A_RGB.npz","B":"CLEAN_RGB.npz","A_repeat":"CLEAN_A_REPEAT_RGB.npz"}
            for variant, filename in output_names.items():
                save_npz(out / filename,rgb=pair[variant])
                save_json(out / ("CLEAN_"+variant.upper()+"_VALIDATION.json"),pair["validations"][variant])
            save_npz(out / "RANDOM_STATE_ARRAYS.npz",**pair["rng_arrays"])
            if "observation" in pair:
                save_npz(out/"PP_OBSERVATION_ARRAYS.npz",**pair["observation_arrays"])
                pair["observation"].update(arrays=artifact_ref(out/"PP_OBSERVATION_ARRAYS.npz"))
                save_json(out/"PP_OBSERVATION.json",pair["observation"])
            for variant,observation in pair.get("stage_observations",{}).items():
                save_npz(out/("PP_STAGE_"+variant+"_ARRAYS.npz"),**observation["arrays"])
                observation["metadata"].update(arrays=artifact_ref(out/("PP_STAGE_"+variant+"_ARRAYS.npz")))
                save_json(out/("PP_STAGE_"+variant+".json"),observation["metadata"])
            pair["receipt"].update(task_id=packet["task_id"],run_signature=packet["run_signature"],
                **context_refs,frozen_input_arrays=artifact_ref(out/"FROZEN_INPUT_ARRAYS.npz"),
                output_arrays={v:artifact_ref(out/n) for v,n in output_names.items()})
            save_json(out / "CLEAN_CONTEXT_PAIR_RECEIPT.json",pair["receipt"])
            indices = all_indices
            video(out/"clean_A.mp4",pair["A"],mapping)
            completed = video(out/"clean.mp4",pair["B"],mapping)
            video(out/"clean_A_repeat.mp4",pair["A_repeat"],mapping)
            b_title = ("B：完整原生像素传播" if experiment == "FULL_NATIVE_IMAGE_PROPAGATION_CONTEXT" else
                       "B：只读内部观测／原配方" if experiment == "TRACE_UNCHANGED_ORIGINAL_PATH" else
                       "B：全片等数量参考" if experiment == "FULL_SESSION_EQUAL_COUNT" else "B：反向遍历恢复原序")
            comparison = comparison_outside_titles([pair["A"],pair["B"],pair["A_repeat"]],
                ["A：当前Mask＋原补图",b_title,"A复测：原补图"],indices,sid,_label)
            frame_states = [{**m,"clean_state":"MATCHED_CONTEXT_ONLY_ORIGINAL_PP",
                "context_experiment":experiment,"fixed_mask":True,"pixel_propagation":"FROZEN_ORIGINAL_V1",
                "protected_changed_pixels":{v:pair["validations"][v]["protected_changed_pixels_per_frame"][i] for v in output_names},
                "outside_effective_changed_pixels":{v:pair["validations"][v]["outside_delete_changed_pixels_per_frame"][i] for v in output_names},
                "hidden_regions":"UNKNOWN","visual_quality":"NOT_REVIEWED","human_review":"PENDING",
                "control_ground_truth":False,"training_eligible":False} for i,m in enumerate(mapping)]
        elif mode == "Clean" and mask_causal_pair:
            from .v1_mask_clean_causal import run_mask_clean_causal
            from .v1_clean_pair import comparison_outside_titles

            base_masks, repaired_masks, causal_refs = causal_inputs

            def causal_call(variant):
                nonlocal model_calls
                authority()
                if model_calls >= packet["budgets"]["model_calls_max"]:
                    raise ValueError("MCO_MODEL_CALL_BUDGET_EXHAUSTED")
                model_calls += 1

            save_npz(out / "FROZEN_INPUT_ARRAYS.npz", raw_rgb=raw,
                base_removal=base_masks["removal"], repaired_removal=repaired_masks["removal"],
                protected_visible=base_masks["protected_visible"],
                base_effective_removal=base_masks["effective_removal"],
                repaired_effective_removal=repaired_masks["effective_removal"])
            pair = run_mask_clean_causal(models["propainter"], raw, base_masks["removal"],
                repaired_masks["removal"], base_masks["protected_visible"], authority=authority,
                on_call=causal_call, backend_before_load=pair_backend_before_load)
            output_names = {"A": "CLEAN_A_RGB.npz", "B": "CLEAN_RGB.npz",
                            "A_repeat": "CLEAN_A_REPEAT_RGB.npz"}
            for variant, filename in output_names.items():
                save_npz(out / filename, rgb=pair[variant])
                save_json(out / ("CLEAN_" + variant.upper() + "_VALIDATION.json"), pair["validations"][variant])
            save_npz(out / "RANDOM_STATE_ARRAYS.npz", **pair["rng_arrays"])
            pair["receipt"].update(task_id=packet["task_id"], run_signature=packet["run_signature"],
                **causal_refs, original_mask=row["original_row"]["stage_files"]["gpu"]["masks.npz"],
                frozen_input_arrays=artifact_ref(out / "FROZEN_INPUT_ARRAYS.npz"),
                output_arrays={variant: artifact_ref(out / filename) for variant, filename in output_names.items()},
                random_state_arrays=artifact_ref(out / "RANDOM_STATE_ARRAYS.npz"))
            save_json(out / "MASK_CLEAN_CAUSAL_RECEIPT.json", pair["receipt"])
            indices = all_indices
            video(out / "clean_A.mp4", pair["A"], mapping)
            completed = video(out / "clean.mp4", pair["B"], mapping)
            video(out / "clean_A_repeat.mp4", pair["A_repeat"], mapping)
            comparison = comparison_outside_titles([pair["A"], pair["B"], pair["A_repeat"]],
                ["A：原V1 Mask＋原补图", "B：三点Mask＋原补图", "A重复：原V1 Mask＋原补图"],
                indices, sid, _label)
            frame_states = [{**m, "clean_state": "MATCHED_MASK_CAUSAL_ORIGINAL_PP",
                "mask_causal_pair": True, "pixel_propagation": "FROZEN_ORIGINAL_V1",
                "protected_changed_pixels": {variant: pair["validations"][variant]["protected_changed_pixels_per_frame"][i]
                    for variant in output_names},
                "outside_effective_changed_pixels": {variant: pair["validations"][variant]["outside_delete_changed_pixels_per_frame"][i]
                    for variant in output_names},
                "hidden_regions": "UNKNOWN", "visual_quality": "NOT_REVIEWED", "human_review": "PENDING",
                "control_ground_truth": False, "training_eligible": False} for i, m in enumerate(mapping)]
        elif mode == "Clean" and paired_clean:
            from .v1_clean_pair import run_fresh_clean_pair, comparison_outside_titles

            def pair_call(variant):
                nonlocal model_calls
                authority()
                if model_calls >= packet["budgets"]["model_calls_max"]:
                    raise ValueError("MCO_MODEL_CALL_BUDGET_EXHAUSTED")
                model_calls += 1

            save_npz(out / "FROZEN_INPUT_ARRAYS.npz", raw_rgb=raw, **{key: value
                for key, value in masks.items() if isinstance(value, np.ndarray)
                and value.shape == raw.shape[:3] and value.dtype == bool})
            pair = run_fresh_clean_pair(models["propainter"], raw, masks["removal"],
                masks["protected_visible"], authority=authority, on_call=pair_call,
                backend_before_load=pair_backend_before_load)
            save_npz(out / "CLEAN_A_RGB.npz", rgb=pair["A"])
            # Preserve the existing downstream Clean filename; its role is B.
            save_npz(out / "CLEAN_RGB.npz", rgb=pair["B"])
            save_npz(out / "RANDOM_STATE_ARRAYS.npz", **pair["rng_arrays"])
            save_json(out / "CLEAN_A_VALIDATION.json", pair["validations"]["A"])
            save_json(out / "CLEAN_VALIDATION.json", pair["validations"]["B"])
            save_json(out / "PIXEL_PROPAGATION_ABLATION.json", pair["receipt"]["B_ablation"])
            indices = all_indices
            video(out / "clean_A.mp4", pair["A"], mapping)
            completed = video(out / "clean.mp4", pair["B"], mapping)
            pair_columns = [pair["A"], pair["B"]]
            pair_titles = ["A：本轮原像素传播", "B：仅跳过图像像素传播"]
            history_ref, history_result = row.get("evaluation_clean_input"), row.get("evaluation_clean_result")
            if (history_ref is None) != (history_result is None):
                raise ValueError("MCO_PAIR_HISTORY_EVALUATION_REFS_REQUIRED_TOGETHER")
            if history_ref is not None:
                _require_sealed_input(history_ref, history_result, sid, "Clean")
                with np.load(history_ref["path"], allow_pickle=False) as archive:
                    history_clean = archive["rgb"]
                if history_clean.shape != raw.shape or history_clean.dtype != np.uint8:
                    raise ValueError("MCO_PAIR_HISTORY_EVALUATION_NATIVE_DOMAIN")
                pair_columns.append(history_clean)
                pair_titles.append("旧三点Clean：仅历史评价")
            pair["receipt"].update(task_id=packet["task_id"], run_signature=packet["run_signature"],
                mask_input=row["mask_input"], mask_result=row["mask_result"],
                raw_source=row["original_row"]["stage_files"]["gpu"]["raw_rgb.npz"],
                frozen_source_raw=source_ref, frozen_frame_map=frame_ref,
                frozen_input_arrays=artifact_ref(out / "FROZEN_INPUT_ARRAYS.npz"),
                output_arrays={"A": artifact_ref(out / "CLEAN_A_RGB.npz"),
                               "B": artifact_ref(out / "CLEAN_RGB.npz")},
                random_state_arrays=artifact_ref(out / "RANDOM_STATE_ARRAYS.npz"),
                evaluation_clean_input=history_ref, evaluation_clean_result=history_result,
                historical_evaluation_column_present=history_ref is not None)
            save_json(out / "CLEAN_PAIR_RECEIPT.json", pair["receipt"])
            frame_states = [{**m, "clean_state": "FRESH_PAIRED_OFFLINE_PROPAINTER_AB",
                "A_pixel_propagation": "FROZEN_ORIGINAL", "B_pixel_propagation": "BYPASS_ONLY",
                "mask_source": "SEALED_THREE_POINT_MASK", "paired_clean": True,
                "protected_changed_pixels_A": pair["validations"]["A"]["protected_changed_pixels_per_frame"][i],
                "protected_changed_pixels_B": pair["validations"]["B"]["protected_changed_pixels_per_frame"][i],
                "outside_effective_changed_pixels_A": pair["validations"]["A"]["outside_delete_changed_pixels_per_frame"][i],
                "outside_effective_changed_pixels_B": pair["validations"]["B"]["outside_delete_changed_pixels_per_frame"][i],
                "hidden_regions": "UNKNOWN", "visual_quality": "NOT_REVIEWED", "human_review": "PENDING",
                "control_ground_truth": False, "training_eligible": False} for i, m in enumerate(mapping)]
            comparison = comparison_outside_titles(pair_columns, pair_titles, indices, sid, _label)
        elif mode == "Clean":
            if model_calls >= packet["budgets"]["model_calls_max"]:
                raise ValueError("MCO_MODEL_CALL_BUDGET_EXHAUSTED")
            propagation = packet["config"].get("pixel_propagation", "BYPASS")
            if propagation not in {"BYPASS", "FROZEN"}:
                raise ValueError("MCO_EXPLICIT_PIXEL_PROPAGATION_MODE_REQUIRED")
            parent = (make_pixel_propagation_ablation(models["propainter"])[0]
                      if propagation == "BYPASS" else models["propainter"])
            painter = visual.ProtectedProPainter(parent)
            authority()
            model_calls += 1
            clean = painter.inpaint(raw, masks["removal"], masks["protected_visible"])
            authority()
            validation = painter.last_validation
            if (validation["status"] != "PASS"
                    or any(validation["protected_changed_pixels_per_frame"])
                    or any(validation["outside_delete_changed_pixels_per_frame"])):
                raise ValueError("MCO_CLEAN_ORIGINAL_PIXEL_PROTECTION_FAILED")
            save_npz(out / "CLEAN_RGB.npz", rgb=clean)
            save_json(out / "CLEAN_VALIDATION.json", validation)
            save_json(out / "PIXEL_PROPAGATION_ABLATION.json", parent.model.metadata() if propagation == "BYPASS"
                      else {"pixel_propagation": "FROZEN_ORIGINAL_V1", "settings_unchanged": True})
            left, right, indices = old_clean, clean, all_indices
            completed = video(out / "clean.mp4", clean, mapping)
            frame_states = [{**m, "clean_state": "OFFLINE_BIDIRECTIONAL_PROPAINTER_CANDIDATE",
                "mask_source": "SEALED_MASK_REPAIR" if row.get("mask_input") else "FROZEN_ORIGINAL_V1", "hidden_regions": "UNKNOWN",
                "visual_quality": "NOT_REVIEWED", "human_review": "PENDING",
                "control_ground_truth": False, "training_eligible": False} for m in mapping]
            title = "本轮候选Clean：不做像素传播" if propagation == "BYPASS" else "本轮候选Clean：原补图＋新Mask"
        else:
            from chaoyang.exact78_baseline import render as rd
            indices = row.get("output_frames", all_indices)
            if row.get("clean_input") is not None:
                _require_sealed_input(row["clean_input"], row.get("clean_result"), sid, "Clean")
                with np.load(row["clean_input"]["path"], allow_pickle=False) as archive:
                    old_clean = archive["rgb"]
                if old_clean.shape != raw.shape or old_clean.dtype != np.uint8:
                    raise ValueError("MCO_SEALED_CLEAN_NATIVE_DOMAIN")
                changed = np.any(old_clean != raw, axis=-1)
                if np.any(changed & (~masks["effective_removal"] | masks["protected_visible"])):
                    raise ValueError("MCO_SEALED_CLEAN_PROTECTION_FAILED")
            motion = reader.arrays(row["original_row"]["stage_files"]["motion"]["actions.npz"])
            K = rd.scaled_intrinsics(session.K, 640, 480, (session.width, session.height))
            left, right, evidence, robot_pixels = [], [], [], []
            for frame in indices:
                authority()
                if render_calls >= packet["budgets"]["render_calls_max"]:
                    raise ValueError("MCO_RENDER_CALL_BUDGET_EXHAUSTED")
                camera = np.linalg.inv(session.c2w[frame]) @ motion["T_world_base"]
                render_calls += 1
                bgr, alpha, _ = renderer.render(motion["q_arm"][frame], motion["q_hand"][frame],
                                               camera, K, complete_robot=False)
                one = {key: masks[key][frame] for key in ("core", "protected_visible", "unknown")}
                raw_bgr, clean_bgr = raw[frame][..., ::-1], old_clean[frame][..., ::-1]
                composed = visual.compose_frame(raw_bgr, clean_bgr, bgr, alpha, one,
                    robot_observed=True, masks_available=True)
                baseline = rd.over(clean_bgr, bgr, composed["draw"])
                baseline[composed["object_front"]] = clean_bgr[composed["object_front"]]
                baseline[one["protected_visible"]] = clean_bgr[one["protected_visible"]]
                candidate, info = compose_support_clipped(clean_bgr, bgr, composed["draw"], one["protected_visible"])
                candidate[composed["object_front"]] = clean_bgr[composed["object_front"]]
                if info["protected_changed_pixels"] or info["outside_draw_changed_pixels"]:
                    raise ValueError("MCO_BOUNDARY_PROTECTION_FAILED")
                difference = np.any(baseline != candidate, axis=-1)
                info.update(native_frame_index=frame, raw_qpc_ns=int(session.qpc_ns[frame]),
                    baseline_candidate_difference_pixels=int(difference.sum()),
                    changed_outside_draw_pixels=int((difference & ~composed["draw"]).sum()),
                    changed_inside_draw_pixels=int((difference & composed["draw"]).sum()))
                evidence.append(info)
                left.append(baseline[..., ::-1].copy())
                right.append(candidate[..., ::-1].copy())
                robot_pixels.append(int(alpha.sum()))
            left, right = np.stack(left), np.stack(right)
            frame_states = evidence
            save_json(out / "BOUNDARY_EVIDENCE.json", evidence)
            save_npz(out / "CANDIDATE_WINDOW_RGB.npz", rgb=right, native_frame_indices=np.asarray(indices))
            completed = video(out / "candidate_window.mp4", right, [mapping[i] for i in indices])
            title = "本轮候选Robot：轮廓外不羽化"
        selected_mapping = [{**mapping[frame], "output_frame": output} for output, frame in enumerate(indices)]
        if not paired_clean and not mask_causal_pair and not context_clean_pair:
            comparison = _comparison(left, right, indices, sid, "原V1", title)
        comparison_info = video(review / f"{sid}_{mode}_comparison.mp4", comparison, selected_mapping)
        from PIL import Image
        for frame in anchors:
            if frame in indices:
                path = review / f"{sid}_{mode}_{frame:04d}.png"
                if path.exists():
                    raise ValueError("MCO_FRESH_REVIEW_ANCHOR_REQUIRED")
                Image.fromarray(comparison[indices.index(frame)]).save(path)
                artifacts.append(artifact_ref(path))
        save_json(out / "FRAME_MAP.json", selected_mapping)
        save_json(out / "FRAME_STATUS.json", frame_states)
        save_json(out / "REVIEW_SCOPE.json", {"actual_AI_appearance_review": "NOT_PERFORMED_BY_WORKER",
            "manual_review": "PENDING", "review_anchors_saved": [f for f in anchors if f in indices],
            "native_frames": len(raw), "rendered_frame_indices": indices,
            "unrendered_frames": [i for i in all_indices if i not in indices],
            "geometry_decision": "UNKNOWN", "visual_quality_pass": False})
        rows.append({"session_key": sid, "native_frames": row["native_frames"],
            "native_output_frames": len(indices), "completed_frame_indices": indices,
            "execution_complete": True, "wall_seconds": time.monotonic() - started,
            "video": completed, "comparison": comparison_info,
            "paired_clean": paired_clean,
            **({"mask_causal_pair": True} if mask_causal_pair else {}),
            **({"context_clean_pair": True} if context_clean_pair else {}),
            "visual_quality_pass": False, "human_review": "PENDING", "candidate_adopted": False})
    authority()
    size = sum(ref["bytes"] for ref in artifacts)
    if size > packet["budgets"]["output_bytes_estimate"] or size > 512 * 1024 * 1024:
        raise ValueError("MCO_OUTPUT_BYTES_BUDGET_EXCEEDED")
    result = {"schema": RESULT_SCHEMA, "status": "SEALED_COMPLETE", "task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "mode": mode, "model_calls": model_calls,
        "render_calls": render_calls, "sessions": rows, "artifacts": artifacts,
        "artifact_bytes": size, "candidate_adopted": False, "control_ground_truth": False,
        "paired_clean": paired_clean,
        **({"mask_causal_pair": True} if mask_causal_pair else {}),
        **({"context_clean_pair": True} if context_clean_pair else {}),
        "training_eligible": False, "visual_quality_pass": False, "human_review": "PENDING"}
    if (root / "MODEL_RESULT.json").exists():
        raise ValueError("MCO_FRESH_MODEL_RESULT_REQUIRED")
    atomic_json(root / "MODEL_RESULT.json", result)
    return 0


execute_mco_experiment = execute_v1_mco_stage
execute = execute_v1_mco_stage
