"""Thin offline candidate Composite over sealed native components (zero models).

The pixel-local helper is reused with MODEL-PREDICTED visible raw support only.
Its internal historical 'trusted' field name is NOT a semantic qualification:
no scene depth is consumed, all geometry remains UNKNOWN, trusted coverage=0.
No review points, historical pixels, dilation, FK or rerender enter production.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Mapping

import numpy as np

from .contracts import SCHEMA, canonical_sha, file_ref, verify_ref, verify_sealed_stage
from .occlusion import DisplayPolicy, GeometryDecision, compose_with_occlusion, resolve_pixel_occlusion

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TIMEBASE = "TIMEBASE_UNVERIFIED_HOST_ARRIVAL_QPC_DECLARED_NS"
SUPPORT_QUALIFICATION = "MODEL_PREDICTED_RAW_SUPPORT_UNVERIFIED"
GEOMETRY = "UNKNOWN_SCENE_DEPTH_ABSENT"
CODE_CLOSURE = ("src/chaoyang/inpainting_candidate/contracts.py",
                "src/chaoyang/inpainting_candidate/occlusion.py",
                "src/chaoyang/inpainting_candidate/clean_references.py",
                "src/chaoyang/inpainting_candidate/source_frames.py",
                "src/chaoyang/exact78_baseline/clean.py")


class CompositeStageError(ValueError):
    pass


def _read(ref):
    verify_ref(ref)
    return json.loads(Path(ref["path"]).read_text(encoding="utf-8-sig"))


def _signature(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise CompositeStageError(f"COMPOSITE_{name}_SIGNATURE_REQUIRED")


def _gate(authorization: Callable, previous=None):
    if not callable(authorization):
        raise CompositeStageError("COMPOSITE_REGISTERED_AUTHORIZATION_REQUIRED")
    value = authorization()
    if (not isinstance(value, Mapping) or value.get("execution_allowed") is not True
            or value.get("algorithm_execution_allowed") is not True
            or not value.get("task_id") or not value.get("fencing_token")):
        raise CompositeStageError("COMPOSITE_EXECUTION_NOT_AUTHORIZED")
    _signature(value.get("run_signature"), "RUN")
    identity = {key: value[key] for key in ("task_id", "run_signature", "fencing_token")}
    if previous is not None and identity != previous:
        raise CompositeStageError("COMPOSITE_WRITER_IDENTITY_CHANGED")
    return identity


def _sealed(ref, signature, stage, repo):
    path = Path(ref["path"]).resolve(strict=True)
    if not path.is_relative_to(repo):
        raise CompositeStageError("COMPOSITE_DERIVED_INPUT_OUTSIDE_PROJECT")
    value = _read(ref)
    if value.get("stage") != stage:
        raise CompositeStageError("COMPOSITE_UPSTREAM_STAGE_MISMATCH:" + stage)
    outputs = verify_sealed_stage(value, signature, output_root=path.parent)
    direct = {Path(row["path"]).name: row for row in outputs
              if Path(row["path"]).resolve(strict=True).parent == path.parent}
    return value, direct


def _npz(ref):
    verify_ref(ref)
    with np.load(ref["path"], allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _rgb_sha(value):
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(value).data)
    return digest.hexdigest()


def _json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return file_ref(path)


def _write_review_video(path, frames, mapping):
    from .source_frames import write_review_video
    return write_review_video(path, frames, mapping)


def _quality(stage, upstream, evidence):
    if evidence is None:
        return upstream.get("visual_quality", "NOT_EVALUATED"), []
    value = _read(evidence)
    if (value.get("schema") != "inpainting-upstream-quality-status-v1"
            or value.get("stage") != stage or value.get("upstream_receipt") != upstream["_reference"]
            or value.get("visual_quality") != "FAILED" or value.get("human_review") != "PENDING"
            or value.get("hidden_regions") != "UNKNOWN"):
        raise CompositeStageError("COMPOSITE_QUALITY_EVIDENCE_BINDING_MISMATCH:" + stage)
    review_ref = value.get("review_reference")
    if not isinstance(review_ref, Mapping):
        raise CompositeStageError("COMPOSITE_ACTUAL_REVIEW_REFERENCE_REQUIRED:" + stage)
    # Verify evidence bytes only. Visible point annotations are never read and
    # can neither alter support pixels nor the production Composite algorithm.
    verify_ref(review_ref)
    return "FAILED", [dict(evidence), dict(review_ref)]


def _dense_owner(labels, identities):
    if labels.dtype != np.int32 or labels.ndim != 3:
        raise CompositeStageError("COMPOSITE_SOURCE_LINK_LABEL_DOMAIN")
    actual = set(np.unique(labels[labels >= 0]).tolist())
    if not actual <= {int(key) for key in identities} or np.any(labels < -1):
        raise CompositeStageError("COMPOSITE_SPARSE_LABEL_NOT_IN_RENDER_LOOKUP")
    entries, names = [], []
    for dense, sparse in enumerate(sorted(int(key) for key in identities)):
        row = identities[str(sparse)]
        if (not isinstance(row, Mapping) or set(row) != {"instance_id", "part_id", "link_id"}
                or any(not isinstance(v, str) or not v for v in row.values())):
            raise CompositeStageError("COMPOSITE_RENDER_LINK_IDENTITY_INVALID")
        entries.append({"dense_owner": dense, "source_link_label": sparse, **dict(row)})
        names.append(f"{sparse}:{row['instance_id']}/{row['part_id']}/{row['link_id']}")
    owner = np.full(labels.shape, -1, np.int32)
    for row in entries:
        owner[labels == row["source_link_label"]] = row["dense_owner"]
    return owner, entries, tuple(names)


def build_composite_stage(*, source_receipt: Mapping, source_signature: str,
                          mask_receipt: Mapping, mask_signature: str,
                          clean_receipt: Mapping, clean_signature: str,
                          render_receipt: Mapping, render_signature: str,
                          output_dir: str | Path, component_signature: str,
                          repo_root: str | Path, authorization: Callable,
                          upstream_quality_evidence: Mapping | None = None) -> dict:
    """Seal pixel-local display; never upgrade Mask/Clean/geometry qualification."""
    identity = _gate(authorization)
    for value, name in ((source_signature, "SOURCE"), (mask_signature, "MASK"),
                        (clean_signature, "CLEAN"), (render_signature, "RENDER"),
                        (component_signature, "COMPONENT")):
        _signature(value, name)
    repo, root = Path(repo_root).resolve(strict=True), Path(output_dir).resolve()
    if repo != PROJECT_ROOT.resolve() or root == repo or not root.is_relative_to(repo):
        raise CompositeStageError("COMPOSITE_OUTPUT_OR_REPO_OUTSIDE_PROJECT")
    if root.exists():
        raise CompositeStageError("COMPOSITE_FRESH_OUTPUT_REQUIRED")
    specs = {"Source": (source_receipt, source_signature), "Mask": (mask_receipt, mask_signature),
             "Clean": (clean_receipt, clean_signature), "Render": (render_receipt, render_signature)}
    stages, outputs = {}, {}
    for stage, (ref, signature) in specs.items():
        stages[stage], outputs[stage] = _sealed(ref, signature, stage, repo)
        stages[stage]["_reference"] = dict(ref)
    count = stages["Source"].get("native_frames")
    if type(count) is not int or count < 1:
        raise CompositeStageError("COMPOSITE_SOURCE_NATIVE_COUNT_REQUIRED")
    for name in ("Mask", "Clean"):
        if (stages[name].get("native_input_frames") != count
                or stages[name].get("native_output_frames") != count):
            raise CompositeStageError("COMPOSITE_UPSTREAM_NATIVE_COUNT_CHANGED:" + name)
    render = stages["Render"]
    if (render.get("native_frames") != count or render.get("source_receipt") != dict(source_receipt)
            or render.get("source_signature") != source_signature
            or render.get("session_id") != stages["Source"].get("session_id")
            or render.get("complete_robot") is not True):
        raise CompositeStageError("COMPOSITE_RENDER_SOURCE_OR_FULL_ROBOT_MISMATCH")
    required = {"Source": {"RAW_RGB.npz", "FRAME_MAP.json", "SOURCE_REPORT.json"},
        "Mask": {"MASK_ARRAYS.npz", "FRAME_MAP.json", "FRAME_STATUS.json"},
        "Clean": {"CLEAN_ARRAYS.npz", "FRAME_MAP.json", "FRAME_STATUS.json"},
        "Render": {"ROBOT_BUFFERS.npz", "FRAME_MAP.json", "FRAME_STATUS.json", "LABEL_IDENTITY.json"}}
    if any(not names <= outputs[stage].keys() for stage, names in required.items()):
        raise CompositeStageError("COMPOSITE_NATIVE_COMPONENT_OUTPUTS_INCOMPLETE")
    mapping = _read(outputs["Source"]["FRAME_MAP.json"])
    if not isinstance(mapping, list) or len(mapping) != count:
        raise CompositeStageError("COMPOSITE_NATIVE_FRAME_MAP_REQUIRED")
    raw = _npz(outputs["Source"]["RAW_RGB.npz"])["rgb"]
    if (raw.dtype != np.uint8 or raw.ndim != 4 or raw.shape[0] != count
            or raw.shape[-1] != 3 or min(raw.shape[1:3]) < 1):
        raise CompositeStageError("COMPOSITE_SOURCE_RGB_DOMAIN")
    for index, row in enumerate(mapping):
        if (not isinstance(row, Mapping)
                or any(type(row.get(key)) is not int or row[key] != index for key in
                       ("native_frame_index", "decoded_display_index", "model_input_frame", "output_frame"))
                or type(row.get("qpc_declared_ns")) is not int
                or row.get("pixel_sha256") != hashlib.sha256(np.ascontiguousarray(raw[index]).data).hexdigest()):
            raise CompositeStageError("COMPOSITE_NATIVE_FRAME_OR_PIXEL_SHA_CHANGED")
    qpc = np.asarray([row["qpc_declared_ns"] for row in mapping], np.int64)
    if np.any(np.diff(qpc) <= 0):
        raise CompositeStageError("COMPOSITE_QPC_DUPLICATE_OR_BACKWARD")
    source_report = _read(outputs["Source"]["SOURCE_REPORT.json"])
    raw_video = source_report.get("actual_inputs", {}).get("video")
    if not isinstance(raw_video, Mapping):
        raise CompositeStageError("COMPOSITE_ACTUAL_RAW_VIDEO_REFERENCE_REQUIRED")
    verify_ref(raw_video)
    frame_states = {}
    for stage in ("Mask", "Clean", "Render"):
        if _read(outputs[stage]["FRAME_MAP.json"]) != mapping:
            raise CompositeStageError("COMPOSITE_UPSTREAM_NATIVE_MAP_CHANGED:" + stage)
        frame_states[stage] = _read(outputs[stage]["FRAME_STATUS.json"])
        if (not isinstance(frame_states[stage], list) or len(frame_states[stage]) != count
                or any(row.get("native_frame_index") != i for i, row in enumerate(frame_states[stage]))):
            raise CompositeStageError("COMPOSITE_FRAME_STATUS_NATIVE_IDENTITY_CHANGED:" + stage)
    array_sha, map_sha = _rgb_sha(raw), canonical_sha(mapping)
    for stage in ("Mask", "Clean"):
        actual = stages[stage].get("actual_inputs", {})
        if (actual.get("frames_sha256") != array_sha or actual.get("frame_map_sha256") != map_sha
                or actual.get("raw_source") != raw_video):
            raise CompositeStageError("COMPOSITE_RAW_PIXEL_OR_TIME_BINDING_CHANGED:" + stage)
    if stages["Clean"]["actual_inputs"].get("mask_receipt") != dict(mask_receipt):
        raise CompositeStageError("COMPOSITE_CLEAN_MASK_RECEIPT_CHANGED")
    mask, clean = _npz(outputs["Mask"]["MASK_ARRAYS.npz"]), _npz(outputs["Clean"]["CLEAN_ARRAYS.npz"])
    keys = {"effective_removal", "protected_visible", "object_observed", "unknown", "human", "equipment"}
    if not keys <= mask.keys() or not {"clean_rgb", "effective_removal", "protected_visible"} <= clean.keys():
        raise CompositeStageError("COMPOSITE_MASK_CLEAN_ARRAY_CONTRACT_INCOMPLETE")
    shape = raw.shape[:3]
    if any(mask[key].shape != shape or mask[key].dtype != np.bool_ for key in keys):
        raise CompositeStageError("COMPOSITE_MASK_BOOL_NATIVE_DOMAIN")
    if (clean["clean_rgb"].shape != raw.shape or clean["clean_rgb"].dtype != np.uint8
            or not np.array_equal(clean["effective_removal"], mask["effective_removal"])
            or not np.array_equal(clean["protected_visible"], mask["protected_visible"])):
        raise CompositeStageError("COMPOSITE_ACTUAL_CLEAN_EP_DIFFER_FROM_FROZEN_MASK")
    buffers = _npz(outputs["Render"]["ROBOT_BUFFERS.npz"])
    if not {"rgba", "depth_m", "link_id", "qpc_ns"} <= buffers.keys():
        raise CompositeStageError("COMPOSITE_RENDER_BUFFER_CONTRACT_INCOMPLETE")
    rgba, depth, labels = buffers["rgba"], buffers["depth_m"], buffers["link_id"]
    if (rgba.shape != (*shape, 4) or rgba.dtype != np.uint8 or depth.shape != shape
            or depth.dtype.kind != "f" or labels.shape != shape or labels.dtype != np.int32
            or not np.array_equal(buffers["qpc_ns"], qpc)):
        raise CompositeStageError("COMPOSITE_RENDER_BUFFER_DOMAIN_OR_TIME_CHANGED")
    visible = rgba[..., 3] > 0
    if (not np.array_equal(visible, labels >= 0)
            or np.any(visible & (~np.isfinite(depth) | (depth <= 0)))
            or np.any(~visible & ~np.isposinf(depth))):
        raise CompositeStageError("COMPOSITE_RENDER_ALPHA_DEPTH_LINK_DISAGREE")
    owner, lookup, names = _dense_owner(labels, _read(outputs["Render"]["LABEL_IDENTITY.json"]))
    evidence = {} if upstream_quality_evidence is None else dict(upstream_quality_evidence)
    if set(evidence) - {"Mask", "Clean"}:
        raise CompositeStageError("COMPOSITE_QUALITY_EVIDENCE_STAGE_UNSUPPORTED")
    mask_quality, mask_quality_refs = _quality("Mask", stages["Mask"], evidence.get("Mask"))
    clean_quality, clean_quality_refs = _quality("Clean", stages["Clean"], evidence.get("Clean"))
    actual_inputs = [dict(ref) for ref, _ in specs.values()] + [dict(raw_video)]
    for stage in specs:
        actual_inputs.extend(stages[stage]["outputs"])
    actual_inputs += [*mask_quality_refs, *clean_quality_refs, file_ref(Path(__file__)),
                     *[file_ref(repo / item) for item in CODE_CLOSURE]]
    actual_inputs = list({ref["path"]: dict(ref) for ref in actual_inputs}.values())
    root.mkdir(parents=True, exist_ok=False)
    final = np.empty_like(raw)
    geometry = np.zeros(shape, np.uint8)
    policy = np.zeros(shape, np.uint8)
    support = (mask["protected_visible"] & mask["object_observed"] & ~mask["unknown"]
        & ~mask["human"] & ~mask["equipment"] & ~mask["effective_removal"])
    statuses, total_fallback, link_totals = [], 0, {}
    review = np.empty((count, raw.shape[1], raw.shape[2] * 4, 3), np.uint8)
    for frame in range(count):
        _gate(authorization, identity)
        # The Render has already merged the complete frozen Robot z-chain. Do
        # not reconstruct per-link depth images or rerender a finger separately.
        chain = {"robot_depth": depth[frame], "robot_link_index": owner[frame],
            "link_names": names, "complete_required_link_closure": True}
        alpha = rgba[frame, ..., 3].astype(np.float32) / 255.0
        resolved = resolve_pixel_occlusion(alpha, chain, support[frame], mask["human"][frame],
            mask["equipment"][frame], mask["effective_removal"][frame],
            object_support_source="RAW_VISIBLE_OBJECT_SUPPORT")
        result = compose_with_occlusion(raw[frame], clean["clean_rgb"][frame], rgba[frame, ..., :3],
            alpha, resolved, clean["effective_removal"][frame], clean["protected_visible"][frame])
        final[frame], geometry[frame], policy[frame] = result["image"], result["geometry_decision"], result["display_policy"]
        if np.any(geometry[frame] != GeometryDecision.UNKNOWN):
            raise CompositeStageError("COMPOSITE_ABSENT_DEPTH_CANNOT_BECOME_QUALIFIED_GEOMETRY")
        fallback = int((policy[frame] == DisplayPolicy.OBJECT_FIRST_FALLBACK).sum())
        total_fallback += fallback
        for name, row in resolved["link_summary"].items():
            current = link_totals.setdefault(name, {"robot_pixels": 0, "model_support_overlap_pixels": 0,
                "object_first_fallback_pixels": 0, "qualified_trusted_occlusion_pixels": 0})
            current["robot_pixels"] += row["visible_robot_pixels"]
            current["model_support_overlap_pixels"] += row["trusted_object_overlap_pixels"]
            current["object_first_fallback_pixels"] += row["object_first_fallback_pixels"]
        marked = raw[frame].copy()
        marked[mask["effective_removal"][frame]] = [255, 0, 0]
        marked[mask["protected_visible"][frame]] = [0, 255, 0]
        marked[mask["unknown"][frame]] = [255, 255, 0]
        review[frame] = np.concatenate((raw[frame], marked, clean["clean_rgb"][frame], final[frame]), axis=1)
        statuses.append({"native_frame_index": frame, "source_frame_map": deepcopy(mapping[frame]),
            "upstream_status": {key: deepcopy(frame_states[key][frame]) for key in frame_states},
            "component_mask_quality": mask_quality, "component_clean_quality": clean_quality,
            "upstream_quality_scope": "COMPONENT_VERDICT_NOT_PER_FRAME_MANUAL_REVIEW",
            "known_upstream_failure_preserved": mask_quality == "FAILED" or clean_quality == "FAILED",
            "mask_unknown_pixels": int(mask["unknown"][frame].sum()),
            "geometry": GEOMETRY, "geometry_unknown_pixels": int(np.prod(shape[1:])),
            "object_first_fallback_pixels": fallback, "object_support_qualification": SUPPORT_QUALIFICATION,
            "qualified_trusted_occlusion_pixels": 0, "hidden_regions": "UNKNOWN",
            "stage_execution": "COMPLETE_PIXEL_LOCAL_CANDIDATE_COMPOSITE",
            "timebase_status": TIMEBASE, "visual_quality": "NOT_REVIEWED", "human_review": "PENDING",
            "training_eligible": False, "control_ground_truth": False})
        _gate(authorization, identity)
    path = root / "COMPOSITE_ARRAYS.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, final_rgb=final, geometry_decision=geometry, display_policy=policy,
            dense_robot_owner=owner, source_link_labels=labels, model_predicted_object_support=support,
            mask_unknown=mask["unknown"], qpc_ns=qpc)
    final_video = _write_review_video(root / "FINAL_CANDIDATE.mp4", final, mapping)
    review_video = _write_review_video(root / "RAW_MASK_CLEAN_ROBOT_REVIEW.mp4", review, mapping)
    if final_video.get("decoded_frames") != count or review_video.get("decoded_frames") != count:
        raise CompositeStageError("COMPOSITE_FULL_NATIVE_VIDEO_DECODE_REQUIRED")
    verify_ref(final_video["video"])
    verify_ref(review_video["video"])
    report = {"schema": "inpainting-candidate-composite-v1", "session_id": stages["Source"]["session_id"],
        "native_frames": count, "actual_inputs": actual_inputs, "complete_robot": True,
        "geometry": GEOMETRY, "scene_depth": "ABSENT", "geometry_unknown_pixels": int(np.prod(shape)),
        "object_support_source": "MODEL_PREDICTED_RAW_VISIBLE_SUPPORT",
        "object_support_qualification": SUPPORT_QUALIFICATION,
        "qualified_trusted_occlusion_pixels": 0, "object_first_fallback_pixels": total_fallback,
        "model_support_pixels": int(support.sum()), "mask_unknown_pixels": int(mask["unknown"].sum()),
        "support_rule": "P_AND_OBJECT_OBSERVED_AND_NOT_UNKNOWN_MINUS_HUMAN_EQUIPMENT_EFFECTIVE_E",
        "sparse_labels_to_dense_lookup": "LABEL_LOOKUP.json", "link_statistics": link_totals,
        "mask_quality": mask_quality, "clean_quality": clean_quality,
        "upstream_quality_evidence": evidence, "known_upstream_failure_preserved":
            mask_quality == "FAILED" or clean_quality == "FAILED",
        "clean_score_improved_by_robot_coverage": False, "clean_pixels_as_geometry": False,
        "pixel_channels": "RGB_UINT8_WITH_ALPHA_DIV255", "review": {"final": final_video, "four_panel": review_video},
        "timebase_status": TIMEBASE, "time_mode": "OFFLINE_VISUAL_NONCAUSAL_CLEAN_AND_MOTION",
        "geometry_labels": {m.name: int(m) for m in GeometryDecision},
        "display_labels": {m.name: int(m) for m in DisplayPolicy},
        "visual_quality": "NOT_REVIEWED", "human_review": "PENDING", "hidden_regions": "UNKNOWN",
        "control_ground_truth": False, "training_eligible": False, "model_calls": 0, "render_calls": 0}
    outputs_final = [file_ref(path), _json(root / "FRAME_MAP.json", mapping),
        _json(root / "FRAME_STATUS.json", statuses),
        _json(root / "LABEL_LOOKUP.json", {"schema": "inpainting-render-sparse-dense-label-lookup-v1", "entries": lookup}),
        _json(root / "COMPOSITE_REPORT.json", report), final_video["video"], review_video["video"]]
    for ref in actual_inputs:
        verify_ref(ref)
    _gate(authorization, identity)
    receipt = {"schema": SCHEMA, "stage": "Composite", "status": "SEALED_COMPLETE",
        "component_signature": component_signature, "source_receipt": dict(source_receipt),
        "source_signature": source_signature, "mask_receipt": dict(mask_receipt), "mask_signature": mask_signature,
        "clean_receipt": dict(clean_receipt), "clean_signature": clean_signature,
        "render_receipt": dict(render_receipt), "render_signature": render_signature,
        "session_id": stages["Source"]["session_id"], "native_frames": count,
        "outputs": outputs_final, "actual_inputs": actual_inputs, "writer_identity": identity,
        "geometry": GEOMETRY, "object_support_qualification": SUPPORT_QUALIFICATION,
        "qualified_trusted_occlusion_pixels": 0, "mask_quality": mask_quality, "clean_quality": clean_quality,
        "upstream_quality_evidence": evidence, "known_upstream_failure_preserved": report["known_upstream_failure_preserved"],
        "complete_robot": True, "timebase_status": TIMEBASE, "visual_quality": "NOT_REVIEWED",
        "human_review": "PENDING", "hidden_regions": "UNKNOWN", "training_eligible": False,
        "control_ground_truth": False, "model_calls": 0, "render_calls": 0}
    _json(root / "RECEIPT.json", receipt)
    return receipt
