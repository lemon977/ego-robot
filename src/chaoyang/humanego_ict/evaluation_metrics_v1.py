"""Finite CPU descriptive evaluation of sealed predictions, never model input.

Four future target members are decoded only by this separately authorized CPU
consumer. No Torch, model, optimizer, SDK, NAS replay, or 2D projection exists.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import sys
import zipfile

from chaoyang.governance.common import REPO_ROOT
from chaoyang.humanego_ict.runtime_gate import _is_descendant

OPERATION = "evaluate_pick_place_humanego_ict_revo2_offline_v1"
MODE = "HUMANEGO_ICT_OFFLINE_METRICS_VISUALIZATION_CPU"
ROUTE = "humanego_ict_offline_metrics_v1"
POLICY = "FROZEN_ADMITTED_PROJECT_PAYLOAD_SNAPSHOT_RESEARCH_ONLY"
REQUEST_SCHEMA = "humanego-ict-offline-metrics-request-v1"
TARGET_MEMBERS = {"y_action", "action_valid_mask", "target_qpc_ns", "target_source_frame_ids"}
HAND_CHANNELS = ("thumb_bend", "thumb_rotation", "index", "middle", "ring", "pinky")


def _need(condition, code):
    if not condition:
        raise ValueError(code)


def _ref_shape(ref):
    _need(isinstance(ref, dict) and set(ref) == {"path", "bytes", "sha256"}, "EXACT_FILE_REF_REQUIRED")
    _need(isinstance(ref["path"], str) and Path(ref["path"]).is_absolute()
          and ".." not in Path(ref["path"]).parts and type(ref["bytes"]) is int and ref["bytes"] >= 0
          and isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]), "REF_DECLARATION_INVALID")


def _path(value, root, existing=False):
    root = Path(root).resolve(strict=True)
    path = Path(value)
    _need(path.is_absolute() and ".." not in path.parts and path != root and path.is_relative_to(root), "PROJECT_ONLY_PATH_REQUIRED")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        _need(not current.is_symlink(), "SYMLINK_FORBIDDEN")
    return path.resolve(strict=existing)


def file_ref(path):
    path = Path(path)
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            h.update(chunk); size += len(chunk)
    return {"path": str(path), "bytes": size, "sha256": h.hexdigest()}


def _verify(ref, root, seen=None):
    _ref_shape(ref)
    path = _path(ref["path"], root, True)
    key = (ref["path"], ref["bytes"], ref["sha256"])
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if seen is not None and seen.get(key) == identity:
        return path
    _need(file_ref(path) == ref, "LOCAL_REF_SHA_OR_BYTES_DRIFT:" + ref["path"])
    after = path.stat()
    _need((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) == identity,
          "LOCAL_REF_CHANGED_DURING_HASH")
    if seen is not None:
        seen[key] = identity
    return path


def _read(ref, root, seen=None):
    _need(ref["bytes"] <= 8 * 1024 * 1024, "JSON_OR_NPZ_READ_LIMIT_EXCEEDED")
    path = _verify(ref, root, seen)
    blob = path.read_bytes()
    _need(len(blob) == ref["bytes"] and hashlib.sha256(blob).hexdigest() == ref["sha256"], "SHA_DRIFT_DURING_PARSE")
    return blob


def _json_blob(blob):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result
    obj = json.loads(blob, object_pairs_hook=unique,
                     parse_constant=lambda v: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))
    _need(isinstance(obj, dict), "JSON_OBJECT_REQUIRED")
    return obj


def _json(ref, root, seen=None):
    return _json_blob(_read(ref, root, seen))


def verify_tree(value, root, seen=None, declarations=False):
    if isinstance(value, dict):
        if set(value) == {"path", "bytes", "sha256"}:
            _ref_shape(value)
            if Path(value["path"]).is_relative_to(Path(root)):
                _verify(value, root, seen)
            else:
                _need(declarations, "UNAUTHORIZED_EXTERNAL_REF")
                # Declaration only: no external stat/read/resolve.
        else:
            for key, child in value.items():
                verify_tree(child, root, seen, key in {"unreplayed_external_source_refs", "evidence_refs"})
    elif isinstance(value, list):
        for child in value:
            verify_tree(child, root, seen, declarations)


def authorize(root, packet_path, argv, request_path, output):
    root = Path(root).resolve(strict=True)
    canonical = [root / "src/chaoyang/humanego_ict/evaluation_metrics_v1.py",
                 root / "src/chaoyang/humanego_ict/evaluation_metrics_contract_v1.py",
                 root / ("src/chaoyang/ops/" + OPERATION + ".py")]
    _need(Path(__file__).resolve(strict=True) == canonical[0], "CANONICAL_ADOPTED_METRICS_SOURCE_REQUIRED")
    packet_path = _path(packet_path, root, True)
    request_path = _path(request_path, root, True)
    output = _path(output, root)
    _need(packet_path.is_relative_to(root / "tasks/current"), "REGISTERED_CPU_PACKET_REQUIRED")
    receipt = _json_blob((root / "docs/governance/CURRENT_STATUS_RECEIPT.json").read_bytes())
    route = _json(receipt["files"]["algorithm_contract"], root)["special_status_contracts"].get(ROUTE, {})
    _need(route.get("candidate_operation") == OPERATION and route.get("task_packet_required") is True
          and route.get("physical_deployment_authorized") is False and route.get("control_ground_truth") is False,
          "REGISTERED_METRICS_ROUTE_REQUIRED")
    packet_ref = file_ref(packet_path)
    packet = _json(packet_ref, root)
    deadline = datetime.fromisoformat(packet["deadline_at"])
    _need(packet.get("execution_mode") == MODE and deadline.tzinfo is not None
          and deadline > datetime.now(timezone.utc) and packet.get("source_policy") == POLICY
          and packet.get("control_ground_truth") is False and packet.get("physical_deployment_authorized") is False,
          "FINITE_RESEARCH_CPU_TASK_REQUIRED")
    budgets = packet["budgets"]
    _need(type(budgets.get("gpu_count")) is int and budgets["gpu_count"] == 0
          and type(budgets.get("model_calls_max")) is int and budgets["model_calls_max"] == 0
          and type(budgets.get("optimizer_updates")) is int and budgets["optimizer_updates"] == 0
          and type(budgets.get("hardware_or_network_calls")) is int and budgets["hardware_or_network_calls"] == 0
          and type(budgets.get("wall_seconds_max")) is int and 1 <= budgets["wall_seconds_max"] <= 7200
          and type(budgets.get("cpu_threads_per_worker")) is int and 1 <= budgets["cpu_threads_per_worker"] <= 2,
          "CPU_ONLY_FINITE_BUDGET_REQUIRED")
    index = _json(receipt["files"]["task_packet_index_current"], root)
    allowed = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    _need(len(allowed) == 1 and allowed[0]["task_id"] == packet["task_id"]
          and allowed[0]["packet_sha256"] == packet_ref["sha256"]
          and (root / allowed[0]["packet_path"]).resolve() == packet_path, "UNIQUE_REGISTERED_CPU_TASK_REQUIRED")
    state = _json(receipt["files"]["task_state"], root)
    active = [row for row in state["tasks"] if row.get("status") in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}]
    writer = packet["writer"]
    _need(len(active) == 1 and active[0]["task_id"] == packet["task_id"] and active[0]["status"] == "RUNNING"
          and active[0].get("run_signature") == packet.get("run_signature")
          and active[0].get("pid") == writer.get("pid") and active[0].get("proc_start_ticks") == writer.get("proc_start_ticks")
          and writer.get("scope") == packet["task_id"] and _is_descendant(writer), "LIVE_CPU_OWNER_ANCESTRY_REQUIRED")
    _need(_json_blob(_path(packet["writer_file"], root, True).read_bytes()) == writer, "CPU_WRITER_FENCE_DRIFT")
    matched = False
    for command in packet["commands"]:
        tokens = shlex.split(command)
        for start in range(len(tokens) - 2):
            if tokens[start:start + 3] == ["-m", "chaoyang.cli", "run"]:
                matched |= tokens[start + 3:] == [OPERATION, *argv]
    _need(matched and packet.get("inputs", {}).get("bundle") == file_ref(request_path)
          and _path(packet["output_dir"], root) == output, "EXACT_CPU_COMMAND_INPUT_OUTPUT_REQUIRED")
    closure = route.get("code_closure", [])
    _need(closure and all(ref in packet.get("runtime_code", []) for ref in closure), "SIGNED_CPU_CODE_CLOSURE_REQUIRED")
    _need(all(file_ref(_path(path, root, True)) in closure for path in canonical), "CANONICAL_THREE_FILE_CLOSURE_REQUIRED")
    for ref in packet["runtime_code"]:
        _verify(ref, root)
    return packet


def read_npz_members(ref, members, root, seen=None, exact=False):
    """Only named .npy members are decoded; unknown future members stay unread."""
    import numpy as np
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(_read(ref, root, seen))) as archive:
        names = archive.namelist()
        _need(len(names) == len(set(names)), "DUPLICATE_NPZ_MEMBER")
        required = {name + ".npy" for name in members}
        _need(required.issubset(names) and (not exact or set(names) == required), "NPZ_MEMBERS_MISMATCH")
        for name in sorted(members):
            entry = archive.getinfo(name + ".npy")
            _need(entry.file_size <= 2 * 1024 * 1024, "NPZ_MEMBER_SIZE_LIMIT")
            array = np.load(io.BytesIO(archive.read(entry)), allow_pickle=False)
            _need(isinstance(array, np.ndarray) and array.dtype.kind != "O", "NUMERIC_ARRAY_ONLY")
            arrays[name] = array
    return arrays


def validate_targets(arrays, *, anchor_ns, source_frame_id):
    import numpy as np
    _need(set(arrays) == TARGET_MEMBERS, "FOUR_TARGET_MEMBERS_ONLY")
    y, mask = arrays["y_action"], arrays["action_valid_mask"]
    _need(y.shape == (50, 30) and y.dtype.kind == "f" and mask.shape == (50, 30) and mask.dtype.kind == "b"
          and np.isfinite(y[mask]).all(), "MASKED_FINITE_H50_ACTION30_REQUIRED")
    times, frames = arrays["target_qpc_ns"], arrays["target_source_frame_ids"]
    _need(times.shape == (50,) and frames.shape == (50,) and times.dtype.kind in "iu" and frames.dtype.kind in "iu"
          and np.all(times > anchor_ns) and np.all(np.diff(times.astype(object)) > 0)
          and np.array_equal(frames, np.arange(source_frame_id + 1, source_frame_id + 51)), "NATIVE_FUTURE_H50_MAPPING_REQUIRED")
    return arrays


def error_vectors(prediction, targets):
    """Raw-value metrics with component masks; no range clipping or repair."""
    import numpy as np
    from chaoyang.humanego_ict.inference import rot6_to_matrix
    p = np.asarray(prediction)
    _need(p.shape == (50, 30) and p.dtype.kind == "f" and np.isfinite(p).all(), "FINITE_RAW_PREDICTION_REQUIRED")
    p = p.astype(np.float64)
    y, mask = targets["y_action"], targets["action_valid_mask"]
    vectors, invalid = {}, {"left": 0, "right": 0}
    for side, xyz, rotation, hand in (("left", slice(0, 3), slice(6, 12), slice(18, 24)),
                                      ("right", slice(3, 6), slice(12, 18), slice(24, 30))):
        valid = mask[:, xyz].all(axis=1)
        vectors[side + "_xyz_cm"] = (100 * np.linalg.norm(p[valid, xyz] - y[valid, xyz], axis=1)).tolist()
        angles = []
        for frame in np.flatnonzero(mask[:, rotation].all(axis=1)):
            # A valid GT degenerate rotation is evidence corruption, not a
            # missing prediction that can be hidden in aggregate statistics.
            truth = rot6_to_matrix(y[frame, rotation])
            try:
                predicted = rot6_to_matrix(p[frame, rotation])
            except ValueError:
                invalid[side] += 1
                continue
            cosine = (np.trace(truth.T @ predicted) - 1) / 2
            angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))))
        vectors[side + "_so3_deg"] = angles
        all_hand_errors = []
        for channel, name in enumerate(HAND_CHANNELS):
            column = hand.start + channel
            valid = mask[:, column]
            errors = np.abs(p[valid, column] - y[valid, column]).tolist()
            vectors[side + "_hand_" + name + "_absolute_error"] = errors
            all_hand_errors.extend(errors)
        vectors[side + "_hand6_absolute_error"] = all_hand_errors
    vectors["bilateral_xyz_cm"] = vectors["left_xyz_cm"] + vectors["right_xyz_cm"]
    vectors["bilateral_so3_deg"] = vectors["left_so3_deg"] + vectors["right_so3_deg"]
    return vectors, invalid


def statistics(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(values.size), "mean": float(values.mean()) if values.size else None,
            "p50": float(np.percentile(values, 50)) if values.size else None,
            "p95": float(np.percentile(values, 95)) if values.size else None}


def aggregate(samples):
    """Pooled quantiles and equal-independent-group macro means are explicit."""
    import numpy as np
    groups = sorted({s["source_group_id"] for s in samples})
    names = sorted({name for s in samples for name in s["error_vectors"]})
    result = {}
    for name in names:
        per_group = {group: statistics([v for sample in samples if sample["source_group_id"] == group
                                       for v in sample["error_vectors"][name]]) for group in groups}
        pooled = statistics([v for sample in samples for v in sample["error_vectors"][name]])
        contributing = [stats for stats in per_group.values() if stats["count"]]
        result[name] = {"pooled": pooled, "independent_group_statistics": per_group,
            "independent_groups_total": len(groups), "independent_groups_contributing": len(contributing),
            "group_macro_mean_of_means": float(np.mean([v["mean"] for v in contributing])) if contributing else None,
            "group_macro_mean_of_p50": float(np.mean([v["p50"] for v in contributing])) if contributing else None,
            "group_macro_mean_of_p95": float(np.mean([v["p95"] for v in contributing])) if contributing else None}
    return result


def _write(path, blob, root):
    path = _path(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(blob); stream.flush(); os.fsync(stream.fileno())
    ref = file_ref(path)
    _need(ref["bytes"] == len(blob) and ref["sha256"] == hashlib.sha256(blob).hexdigest(), "OUTPUT_READBACK_FAILED")
    return ref


def _write_json(path, obj, root):
    return _write(path, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(), root)


def validate_sealed_branch(bundle, branch, root, seen):
    _need(set(bundle) == {"task_packet_ref", "outer_result_ref", "inner_result_ref", "post_validation_ref",
                         "current_inputs_outer_result_ref"}, "EXACT_SEALED_BRANCH_REFS_REQUIRED")
    task = _json(bundle["task_packet_ref"], root, seen)
    outer = _json(bundle["outer_result_ref"], root, seen)
    inner = _json(bundle["inner_result_ref"], root, seen)
    post = _json(bundle["post_validation_ref"], root, seen)
    producer_outer = _json(bundle["current_inputs_outer_result_ref"], root, seen)
    _need(producer_outer.get("schema") == "humanego-finite-offline-publisher-result-v1"
          and producer_outer.get("task_terminal_status") == "PASSED" and producer_outer.get("errors") == []
          and type(producer_outer.get("child_exit_code")) is int and producer_outer["child_exit_code"] == 0
          and producer_outer.get("control_ground_truth") is False and producer_outer.get("physical_deployment_authorized") is False,
          "NORMAL_SEALED_CURRENT_INPUT_PRODUCER_REQUIRED")
    _need(task["inputs"].get("current_inputs_sealed_result") == bundle["current_inputs_outer_result_ref"],
          "GPU_TASK_CURRENT_PRODUCER_EXACT_REF_REQUIRED")
    producer_stage = _json(producer_outer["stage_result_ref"], root, seen)
    _need(producer_stage.get("schema") == "humanego-ict-current-input-production-result-v1"
          and producer_stage.get("status") == "CURRENT_INPUTS_PRODUCED_NOT_MODEL_EVALUATED"
          and producer_stage.get("branch") == branch and producer_stage.get("sample_count") == 57
          and producer_stage.get("source_group_count") == 20 and producer_stage.get("source_policy") == POLICY
          and producer_stage.get("future_target_members_decoded") == [] and producer_stage.get("fixture_only") is False
          and producer_stage.get("task_id") == producer_outer["task_id"]
          and producer_stage.get("run_signature") == producer_outer["run_signature"]
          and producer_stage.get("manifest_ref") == task["inputs"].get("admitted_current_inputs_manifest"),
          "ACTUAL_SEALED_CURRENT_PRODUCER_STAGE_BINDING_REQUIRED")
    current_manifest = _json(producer_stage["manifest_ref"], root, seen)
    _need(current_manifest.get("schema") == "humanego-ict-revo2-batch-current-inputs-v1"
          and current_manifest.get("branch") == branch and current_manifest.get("fixture_only") is False,
          "CURRENT_INPUT_MANIFEST_BRANCH_REQUIRED")
    current_samples = {s["sample_id"]: s for s in current_manifest["samples"]}
    verify_tree(producer_outer, root, seen)
    verify_tree(producer_stage, root, seen); verify_tree(current_manifest, root, seen)
    _need(outer.get("schema") == "humanego-finite-offline-publisher-result-v1"
          and outer.get("task_terminal_status") == "PASSED" and outer.get("errors") == []
          and type(outer.get("child_exit_code")) is int and outer["child_exit_code"] == 0
          and outer.get("stage_result_ref") == bundle["inner_result_ref"], "ACTUAL_NORMAL_SEALED_GPU_OUTER_REQUIRED")
    _need(type(post.get("exit_code")) is int and post["exit_code"] == 0
          and _json_blob(post["stdout"]).get("status") == "PASS", "ACTUAL_GPU_POST_GOVERNANCE_REQUIRED")
    _need(task.get("execution_mode") == "HUMANEGO_ICT_REVO2_OFFLINE_EVALUATION" and task.get("branch") == branch
          and task.get("source_policy") == POLICY and task.get("control_ground_truth") is False
          and task.get("physical_deployment_authorized") is False
          and all(obj.get("task_id") == task["task_id"] and obj.get("run_signature") == task["run_signature"]
                  for obj in (outer, inner)), "GPU_TASK_BRANCH_AND_SIGNATURE_BINDING_REQUIRED")
    _need(inner.get("schema") == "humanego-ict-revo2-offline-evaluation-result-v1"
          and inner.get("status") == "COMPLETED_REAL_MODEL_OFFLINE_DIAGNOSTICS_NOT_DEPLOYMENT"
          and inner.get("branch") == branch and inner.get("source_policy") == POLICY
          and inner.get("samples_completed") == 57 and inner.get("groups_completed") == 20 and inner.get("model_calls") == 1140
          and inner.get("sampling") == {"method": "euler", "steps": 20, "seed": 7}
          and inner.get("request") == task["inputs"]["request"] and inner.get("clipping_applied") is False
          and inner.get("future_labels_provided_to_model") is False
          and inner.get("physical_deployment_authorized") is False and inner.get("control_ground_truth") is False,
          "ACTUAL57_GROUP20_EULER20_UNCLIPPED_GPU_STAGE_REQUIRED")
    for obj in (task, outer, inner):
        verify_tree(obj, root, seen)
    request = _json(inner["request"], root, seen)
    _need(request.get("inference_contract") == inner["inference_contract"] and request.get("branch") == branch, "GPU_REQUEST_MODEL_BINDING_REQUIRED")
    contract = _json(inner["inference_contract"], root, seen)
    verify_tree(contract, root, seen)
    provenance = contract["training_provenance"]
    _need(provenance.get("branch") == branch and provenance.get("optimizer_updates") == 20000
          and provenance.get("source_policy") == POLICY and contract.get("state_dict_key") == "model", "ACTUAL20K_MODEL_PROVENANCE_REQUIRED")
    samples = inner["samples"]
    _need(len(samples) == 57 and len({s["sample_id"] for s in samples}) == 57
          and len({s["group_id"] for s in samples}) == 20, "EXACT57_UNIQUE20_GROUPS_REQUIRED")
    _need(len(current_samples) == len(current_manifest["samples"]) == 57
          and set(current_samples) == {s["sample_id"] for s in samples}, "CURRENT_PRODUCER_AND_GPU_SAMPLE_SET_REQUIRED")
    request_samples = {s["sample_id"]: s for s in request["samples"]}
    _need(len(request_samples) == 57 and set(request_samples) == {s["sample_id"] for s in samples}, "SEALED_GPU_REQUEST_SAMPLE_SET_REQUIRED")
    for sample in samples:
        original = request_samples[sample["sample_id"]]
        produced = current_samples[sample["sample_id"]]
        _need(all(sample[k] == produced[k] for k in ("current_input_npz", "input_receipt", "source_frame_id", "anchor_qpc_ns"))
              and sample["group_id"] == produced["source_group_id"], "GPU_SAMPLE_AND_PRODUCED_CURRENT_INPUTS_BINDING_REQUIRED")
        _need(all(sample[key] == original[key] for key in ("group_id", "current_input_npz", "input_receipt"))
              and sample.get("model_calls") == 20 and sample.get("command_policy") == "REJECT"
              and sample.get("clipping_applied") is False, "SEALED_GPU_SAMPLE_EXACT_BINDINGS_REQUIRED")
        receipt = _json(sample["input_receipt"], root, seen)
        verify_tree(receipt, root, seen)
    return {s["sample_id"]: s for s in samples}, inner


def render_example(sample_id, target, predictions, current_rgbs, output, root, *, make_mp4=True, fps=15, frame_indices=None):
    """Camera-frame 3D only; RGB is explicitly static, not future recording."""
    import numpy as np
    output = _path(output, root)
    cache = output / "runtime_cache/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.animation import FFMpegWriter
    output.mkdir(parents=True, exist_ok=True)
    for path in (output / "comparison.png", output / "trajectory.mp4"):
        _need(not path.exists(), "VISUALIZATION_OUTPUT_CONFLICT")
    y, mask = target["y_action"], target["action_valid_mask"]
    series = {"GT derived": y, "Raw prediction": predictions["raw"], "Robotized prediction": predictions["robotized"]}
    colors = {"GT derived": "black", "Raw prediction": "tab:orange", "Robotized prediction": "tab:purple"}
    fig = plt.figure(figsize=(12.8, 9.6), dpi=100)
    grid = fig.add_gridspec(4, 6)
    for i, branch in enumerate(("raw", "robotized")):
        ax = fig.add_subplot(grid[0, i * 3:(i + 1) * 3])
        rgb = current_rgbs[branch][0].transpose(1, 2, 0)
        _need(rgb.shape == (240, 320, 3) and np.isfinite(rgb).all(), "CURRENT_RGB_REQUIRED_FOR_VISUALIZATION")
        ax.imshow(rgb); ax.set_title(branch + ": STATIC current RGB, not future recording"); ax.axis("off")
    axes3d, moving = [], []
    for side, xyz in (("left", slice(0, 3)), ("right", slice(3, 6))):
        ax = fig.add_subplot(grid[1, 0:3] if side == "left" else grid[1, 3:6], projection="3d")
        axes3d.append(ax)
        valid = mask[:, xyz].all(axis=1)
        for name, a in series.items():
            points = a[:, xyz].copy()
            if name == "GT derived": points[~valid] = np.nan
            line, = ax.plot(points[:, 0], points[:, 1], points[:, 2], color=colors[name], label=name)
            dot, = ax.plot([], [], [], "o", color=colors[name], markersize=4)
            moving.append((line, dot, points))
        ax.set(title=side + " wrist: REFERENCE_CAMERA 3D", xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
        ax.legend(fontsize=6); ax.set_box_aspect((1, 1, 1))
    times = (target["target_qpc_ns"].astype(np.float64) - float(target["anchor_ns"])) / 1e9
    for side_index, side in enumerate(("left", "right")):
        for channel, name in enumerate(HAND_CHANNELS):
            ax = fig.add_subplot(grid[2 + side_index, channel])
            column = 18 + side_index * 6 + channel
            for title, a in series.items():
                values = a[:, column].copy()
                if title == "GT derived": values[~mask[:, column]] = np.nan
                ax.plot(times, values, color=colors[title], linewidth=1)
            ax.axhline(0, color="grey", linewidth=.5); ax.axhline(1, color="grey", linewidth=.5)
            ax.set_title(side + " " + name, fontsize=7); ax.set_xlabel("native label delta seconds", fontsize=6)
            ax.tick_params(labelsize=6)
    title = fig.suptitle(sample_id + " | DERIVED GT vs UNCLIPPED predictions\nOffline visualization only; no intrinsics/2D projection, calibrated control, or physical recording", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .94))
    png = output / "comparison.png"
    fig.savefig(png)
    result = {"sample_id": sample_id, "png_ref": file_ref(png), "static_reference_RGB": True,
              "prediction_not_recording": True, "camera_intrinsics": "ABSENT_NO_2D_PROJECTION",
              "clipping_applied": False, "control_ground_truth": False, "physical_deployment_authorized": False}
    try:
        if make_mp4:
            frames = list(range(50)) if frame_indices is None else list(frame_indices)
            _need(1 <= len(frames) <= 50 and all(type(i) is int and 0 <= i < 50 for i in frames), "BOUNDED_ANIMATION_FRAMES_REQUIRED")
            mp4 = output / "trajectory.mp4"
            writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-threads", "2", "-pix_fmt", "yuv420p", "-y"])
            with writer.saving(fig, str(mp4), 100):
                for frame in frames:
                    for line, dot, points in moving:
                        shown = points[:frame + 1]
                        line.set_data_3d(shown[:, 0], shown[:, 1], shown[:, 2])
                        point = points[frame]
                        dot.set_data_3d([point[0]], [point[1]], [point[2]])
                    title.set_text(sample_id + f" | prediction trajectory animation, native label frame {frame + 1}/50, delta {times[frame]:.6f}s\nSTATIC reference RGB; animation is NOT a future recording or physical execution")
                    writer.grab_frame()
            result.update(mp4_ref=file_ref(mp4), rendered_frames=len(frames), playback_fps=fps,
                          playback_fps_is_acquisition_rate=False)
    finally:
        plt.close(fig)
    return result


def evaluate_request(request, packet, root, *, recheck):
    import numpy as np
    from chaoyang.humanego_ict.inference import decode_actions
    root = Path(root).resolve(strict=True)
    _need(set(request) == {"schema", "raw", "robotized", "targets_ref", "input_spec_ref", "visualization"}
          and request["schema"] == REQUEST_SCHEMA, "EXACT_CPU_METRICS_REQUEST_REQUIRED")
    visual = request["visualization"]
    _need(set(visual) == {"samples_max", "make_mp4", "playback_fps"}
          and type(visual["samples_max"]) is int and 0 <= visual["samples_max"] <= 3
          and type(visual["make_mp4"]) is bool and type(visual["playback_fps"]) is int
          and 1 <= visual["playback_fps"] <= 30, "BOUNDED_EXPLICIT_VISUALIZATION_REQUIRED")
    seen = {}
    spec = _json(request["input_spec_ref"], root, seen)
    refs = _json(request["targets_ref"], root, seen)
    _need(refs.get("schema") == "humanego-ict-frozen-evaluator-only-target-references-v1"
          and refs.get("evaluation_input_spec_ref") == request["input_spec_ref"]
          and refs.get("production_manifest_ref") == spec.get("manifest_ref")
          and refs.get("model_input_allowed") is False and refs.get("control_ground_truth") is False
          and refs.get("physical_deployment_authorized") is False and spec.get("source_policy") == POLICY
          and spec.get("normalization") == {"method": "IDENTITY_NO_FIT", "position_mean": [0, 0, 0], "position_std": [1, 1, 1]},
          "FROZEN_EVALUATOR_ONLY_TARGET_SPEC_REQUIRED")
    verify_tree(refs, root, seen); verify_tree(spec, root, seen)
    rows = {s["sample_id"]: s for s in refs["samples"]}
    common = {s["sample_id"]: s for s in spec["common_validation_rows"]}
    _need(len(rows) == len(refs["samples"]) == len(common) == 57
          and set(rows) == set(common) and len({s["source_group_id"] for s in rows.values()}) == 20, "EXACT57_COMMON20_GROUP_TARGETS_REQUIRED")
    for sid, row in rows.items():
        _need(set(row["allowed_evaluator_members"]) == TARGET_MEMBERS
              and all(row[key] == common[sid][key] for key in ("sample_id", "session_id", "source_frame_id", "source_group_id")),
              "EXACT_CURRENT_SAMPLE_TARGET_GROUP_AND_MEMBER_BINDING_REQUIRED")
    branches, stages = {}, {}
    for branch in ("raw", "robotized"):
        branches[branch], stages[branch] = validate_sealed_branch(request[branch], branch, root, seen)
        _need(set(branches[branch]) == set(rows), "PREDICTION_AND_TARGET_SAMPLE_SET_MISMATCH")
    output = _path(packet["output_dir"], root)
    recheck()
    reservation = _write_json(output / "CPU_METRICS_RESERVED.json", {"task_id": packet["task_id"],
        "run_signature": packet["run_signature"], "writer": packet["writer"], "evaluation_bundle_ref": packet["inputs"]["bundle"],
        "status": "RESERVED_NOT_COMPLETE", "failure_requires_new_attempt": True}, root)
    branch_samples = {branch: [] for branch in branches}
    rendered = []
    chosen_groups = set()
    for sid in sorted(rows):
        recheck()
        row = rows[sid]
        sample0 = branches["raw"][sid]
        target = read_npz_members(row["source_labels_ref"], TARGET_MEMBERS, root, seen)
        validate_targets(target, anchor_ns=sample0["anchor_qpc_ns"], source_frame_id=row["source_frame_id"])
        target["anchor_ns"] = sample0["anchor_qpc_ns"]
        predictions, rgbs = {}, {}
        for branch in ("raw", "robotized"):
            sample = branches[branch][sid]
            _need(sample["group_id"] == row["source_group_id"] and sample["source_frame_id"] == row["source_frame_id"]
                  and sample["anchor_qpc_ns"] == sample0["anchor_qpc_ns"], "SAME_NATIVE_ANCHOR_GROUP_ACROSS_BRANCHES_REQUIRED")
            predictions[branch] = read_npz_members(sample["raw_actions_ref"], {"actions_50x30"}, root, seen, exact=True)["actions_50x30"]
            vectors, invalid = error_vectors(predictions[branch], target)
            current = read_npz_members(sample["current_input_npz"], {"x_rgb", "reference_camera_to_world"}, root, seen)
            rgbs[branch] = current["x_rgb"]
            failure = None
            try:
                decode_actions(predictions[branch], current["reference_camera_to_world"], normalization=spec["normalization"], command_policy="REJECT")
            except ValueError as error:
                failure = str(error)
            expected_pass = failure is None
            _need(sample.get("qualified_for_assumed_bridge") is expected_pass
                  and sample.get("strict_decode_status") == ("PASSED_STRICT_DECODE_ONLY" if expected_pass else "FAILED_STRICT_REJECT"),
                  "SEALED_STRICT_DECODE_STATUS_DRIFT")
            _need(sample.get("failure_codes") == ([] if expected_pass else [failure]), "SEALED_STRICT_FAILURE_CODE_DRIFT")
            hand = predictions[branch][:, 18:]
            branch_samples[branch].append({"sample_id": sid, "source_group_id": row["source_group_id"],
                "error_vectors": vectors, "invalid_predicted_rot6_frames": invalid,
                "strict_decode_failed": not expected_pass, "strict_decode_failure": failure,
                "raw_hand_out_of_range_values": int(((hand < 0) | (hand > 1)).sum()),
                "raw_actions_ref": sample["raw_actions_ref"], "target_labels_ref": row["source_labels_ref"]})
        if len(rendered) < visual["samples_max"] and row["source_group_id"] not in chosen_groups:
            recheck()
            chosen_groups.add(row["source_group_id"])
            example = render_example(sid, target, predictions, rgbs, output / ("example_" + str(len(rendered)).zfill(2)), root,
                make_mp4=visual["make_mp4"], fps=visual["playback_fps"])
            example["source_group_id"] = row["source_group_id"]
            rendered.append(example)
    summary = {}
    for branch, samples in branch_samples.items():
        summary[branch] = {"samples": len(samples), "independent_source_groups": len({s["source_group_id"] for s in samples}),
            "metrics": aggregate(samples), "strict_decode_failed_samples": sum(s["strict_decode_failed"] for s in samples),
            "raw_hand_out_of_range_values": sum(s["raw_hand_out_of_range_values"] for s in samples),
            "invalid_predicted_rot6_frames": {side: sum(s["invalid_predicted_rot6_frames"][side] for s in samples) for side in ("left", "right")}}
    # Revalidate every local source ref after work; a changed immutable input
    # aborts publication rather than leaving a successful partial result.
    verify_tree(request, root, {})
    for stage in stages.values(): verify_tree(stage, root, {})
    recheck()
    details = _write_json(output / "PER_SAMPLE_METRICS.json", {"samples": branch_samples,
        "control_ground_truth": False, "physical_deployment_authorized": False}, root)
    return {"schema": "humanego-ict-offline-metrics-result-v1", "status": "DESCRIPTIVE_OFFLINE_EVALUATION_NOT_QUALIFICATION",
        "task_id": packet["task_id"], "run_signature": packet["run_signature"], "evaluation_bundle_ref": packet["inputs"]["bundle"],
        "source_policy": POLICY, "sealed_branch_refs": {b: request[b] for b in branches},
        "targets_ref": request["targets_ref"], "input_spec_ref": request["input_spec_ref"], "summary": summary,
        "per_sample_metrics_ref": details, "visualizations": rendered, "reservation_ref": reservation,
        "target_members_decoded": sorted(TARGET_MEMBERS), "future_targets_provided_to_model": False,
        "normalization": spec["normalization"], "position_metric": "euclidean_camera_frame_cm_complete_xyz_mask",
        "so3_metric": "degrees_acos_trace_same_row_interleaved_rot6_GramSchmidt", "hand_metric": "unclipped_normalized_0_1_absolute_error_per_channel",
        "macro_definition": "Equal weight for each contributing original source_group_id; coverage reported, never zero-impute missing groups",
        "quality_thresholds": "ABSENT_NO_PASS_CLAIM", "source_replay_status": "SOURCE_REPLAY_NOT_PERFORMED",
        "control_ground_truth": False, "physical_deployment_authorized": False, "model_calls": 0,
        "gpu_calls": 0, "optimizer_updates": 0, "robot_commands_sent": 0, "hardware_or_network_calls": 0}


def request_from_bundle(bundle, root):
    _need(bundle.get("schema") == "humanego-ict-real20k-offline-sealed-evaluation-bundle-v1"
          and bundle.get("source_policy") == POLICY and bundle.get("control_ground_truth") is False
          and bundle.get("physical_deployment_authorized") is False
          and set(bundle.get("branches", {})) == {"raw", "robotized"}, "EXACT_OFFLINE_SEALED_BUNDLE_REQUIRED")
    refs = _json(bundle["evaluation_targets_ref"], root)
    return {"schema": REQUEST_SCHEMA, "raw": bundle["branches"]["raw"], "robotized": bundle["branches"]["robotized"],
        "targets_ref": bundle["evaluation_targets_ref"], "input_spec_ref": refs["evaluation_input_spec_ref"],
        "visualization": {"samples_max": 3, "make_mp4": True, "playback_fps": 15}}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-packet", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        packet = authorize(REPO_ROOT, args.task_packet, argv, args.bundle, args.output_dir)
        def recheck():
            current = authorize(REPO_ROOT, args.task_packet, argv, args.bundle, args.output_dir)
            _need(current == packet, "CPU_AUTHORIZATION_CHANGED")
        request = request_from_bundle(_json(file_ref(args.bundle), REPO_ROOT), REPO_ROOT)
        result = evaluate_request(request, packet, REPO_ROOT, recheck=recheck)
        recheck()
        _write_json(args.output_dir / "RESULT.json", result, REPO_ROOT)
        print(json.dumps({"status": result["status"], "model_calls": 0, "physical_deployment_authorized": False}))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "BLOCKED", "failure_code": str(error), "physical_deployment_authorized": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
