"""Finite V5 Scene producer for original-RGB offline visual compositing.

Usage: --stage prepare with domain/mask bindings, then --stage inpaint with
the resulting SCENE_PREP_MANIFEST.json. Each invocation writes a new child of
the registered V5 Scene lane; immutable predecessors are only read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from chaoyang.pipeline.v5_scene import (
    DEVICE_ROLES, HUMAN_ROLES, OBJECT_ROLES, build_scene_mask_window,
    composite_clean, merge_roles,
)


TASK = "four_stream_visual_delivery_v5"
LANE = Path("_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene")
IMAGE_SIZE = (960, 720)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def ref(path: Path) -> dict:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def checked_ref(value: dict) -> Path:
    actual = ref(Path(value["path"]))
    if actual["sha256"] != value["sha256"] or actual["bytes"] != value["bytes"]:
        raise ValueError(f"INPUT_SHA_MISMATCH:{value['path']}")
    return Path(actual["path"])


def image(path: Path, value: np.ndarray) -> None:
    if not cv2.imwrite(str(path), value):
        raise RuntimeError(f"IMAGE_WRITE:{path}")


def output_root(project: Path, destination: Path) -> Path:
    lane = (project / LANE).resolve(strict=True)
    destination = destination.resolve()
    if destination.parent != lane:
        raise ValueError("SCENE_OUTPUT_SCOPE")
    destination.mkdir(exist_ok=False)
    return destination


def _role_group(role: str) -> str:
    name = role.lower().replace("-", "_")
    if name in HUMAN_ROLES:
        return "human"
    if name in DEVICE_ROLES:
        return "device"
    if name in OBJECT_ROLES:
        return "object"
    raise ValueError(f"UNKNOWN_ROLE:{role}")


def _mask_files(mask_docs: list[dict], frame: int, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    masks = {}
    for mask_doc in mask_docs:
        root = Path(mask_doc["mask_root"])
        for row in mask_doc["roles"]:
            role = row["role"]
            path = root / role / f"{frame:06d}.png"
            value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if value is None or value.shape != shape:
                raise ValueError(f"MASK_READ_OR_DOMAIN:{role}:{frame}:{path}")
            if role in masks:
                masks[role] = np.logical_or(masks[role] > 0, value > 0)
            else:
                masks[role] = value
    return masks


def preflight(spec_path: Path, output: Path) -> Path:
    """Read-only input binding gate with a terminal receipt in the Scene lane."""
    spec = load(spec_path)
    project = Path(spec["project_root"]).resolve(strict=True)
    domain_path = Path(spec["domain_manifest"])
    domain = load(domain_path)
    if domain.get("session_id") != spec["session_id"]:
        raise ValueError("SCENE_SESSION_MISMATCH")
    if len(domain.get("frames", [])) != domain["frame_count"]:
        raise ValueError("SCENE_DOMAIN_ROW_COUNT")
    paths = [Path(value) for value in spec.get("mask_manifests", [spec.get("mask_manifest")])
             if value is not None]
    docs = [load(path) for path in paths]
    mismatch = any(doc.get("session_id") != spec["session_id"] or
                   doc.get("frame_count") != domain["frame_count"] or
                   doc.get("domain") != domain["image_domain"] for doc in docs)
    role_names = {_role_group(role["role"]) for doc in docs for role in doc["roles"]}
    missing = sorted({"human", "device", "object"} - role_names)
    if not paths:
        missing = ["human", "device", "object"]
    trusted = spec.get("trusted_object_frames", [])
    trusted_count = domain["frame_count"] if trusted == "all" else len(trusted)
    trust_ref = spec.get("object_trust_evidence")
    if trust_ref:
        checked_ref(trust_ref)
    reasons = []
    if mismatch:
        reasons.append("MASK_DOMAIN_MISMATCH")
    if missing:
        reasons.append("MISSING_SEMANTIC_ROLES")
    if any(not (Path(doc["mask_root"]) / role["role"] / f"{i:06d}.png").is_file()
           for doc in docs for role in doc["roles"] for i in range(domain["frame_count"])):
        reasons.append("MASK_FRAME_MISSING")
    if trusted_count != domain["frame_count"] or not trust_ref:
        reasons.append("VISIBLE_OBJECT_PROTECTION_NOT_VERIFIED")
    if domain["frame_count"] != domain.get("full_source_frame_count", domain["frame_count"]):
        reasons.append("PARTIAL_SESSION")
    destination = output_root(project, output)
    receipt = {
        "schema_version": "v5-scene-input-gate-v1",
        "status": "READY" if not reasons else "BLOCKED_INPUT",
        "session_id": spec["session_id"], "frame_count": domain["frame_count"],
        "domain": domain["image_domain"], "source_domain": ref(domain_path),
        "source_masks": [ref(path) for path in paths],
        "roles_present": sorted(role_names), "missing_semantic_roles": missing,
        "trusted_visible_object_frames": trusted_count,
        "reasons": reasons, "config": ref(spec_path), "code": ref(Path(__file__)),
        "algorithm_executed": False, "control_ground_truth": False,
    }
    path = destination / "RESULT.json"
    save(path, receipt)
    return path


def prepare(spec_path: Path, output: Path) -> Path:
    spec = load(spec_path)
    project = Path(spec["project_root"]).resolve(strict=True)
    session = spec["session_id"]
    domain_path = Path(spec["domain_manifest"])
    mask_paths = [Path(value) for value in spec.get("mask_manifests", [spec.get("mask_manifest")])]
    if not mask_paths or any(value is None for value in mask_paths):
        raise ValueError("MASK_MANIFEST_REQUIRED")
    domain, mask_docs = load(domain_path), [load(path) for path in mask_paths]
    if domain["session_id"] != session or any(mask["session_id"] != session for mask in mask_docs):
        raise ValueError("SCENE_SESSION_MISMATCH")
    if any(domain["frame_count"] != mask["frame_count"] or domain["image_domain"] != mask["domain"]
           for mask in mask_docs):
        raise ValueError("SCENE_SOURCE_DOMAIN_MISMATCH")
    if domain["frame_count"] != domain.get("full_source_frame_count", domain["frame_count"]):
        raise ValueError("SCENE_NOT_FULL_SESSION")
    if len(domain["frames"]) != domain["frame_count"]:
        raise ValueError("SCENE_DOMAIN_ROW_COUNT")
    shape = (int(domain["height"]), int(domain["width"]))
    names = {_role_group(row["role"]) for mask in mask_docs for row in mask["roles"]}
    missing = sorted({"human", "device", "object"} - names)
    if missing:
        raise ValueError(f"BLOCKED_INPUT_MISSING_ROLES:{','.join(missing)}")
    trust = spec.get("trusted_object_frames", [])
    if trust == "all":
        if not spec.get("object_trust_evidence"):
            raise ValueError("OBJECT_TRUST_EVIDENCE_REQUIRED")
        checked_ref(spec["object_trust_evidence"])
        trust = range(domain["frame_count"])
    if trust and not spec.get("object_trust_evidence"):
        raise ValueError("OBJECT_TRUST_EVIDENCE_REQUIRED")
    if spec.get("object_trust_evidence"):
        checked_ref(spec["object_trust_evidence"])
    if trust and "object" not in names:
        raise ValueError("OBJECT_ROLE_ABSENT")
    if len(set(trust)) != domain["frame_count"]:
        raise ValueError("BLOCKED_INPUT_OBJECT_PROTECTION_UNVERIFIED")
    frame_count = domain["frame_count"]
    trusted = set(trust)
    dst = output_root(project, output)
    for sub in ("frames", "write", "protect", "context_exclude", "unknown", "model_masks"):
        (dst / sub).mkdir()
    rows = []
    role_cache = {}
    for i, raw_row in enumerate(domain["frames"]):
        start, stop = max(0, i - 2), min(frame_count, i + 3)
        for j in range(start, stop):
            if j not in role_cache:
                role_cache[j] = merge_roles(_mask_files(mask_docs, j, shape), shape)
        for j in list(role_cache):
            if j < start:
                del role_cache[j]
        frame_masks = build_scene_mask_window([role_cache[j] for j in range(start, stop)],
                                              i - start, i, i in trusted)
        if raw_row["frame_id"] != i:
            raise ValueError("SCENE_FRAME_ID")
        raw_path = Path(raw_row["rgb"])
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        if raw is None or raw.shape[:2] != shape:
            raise ValueError(f"SCENE_RAW_DECODE:{i}")
        for name in ("write", "protect", "context_exclude", "unknown"):
            image(dst / name / f"{i:06d}.png", frame_masks[name].astype(np.uint8) * 255)
        image(dst / "frames" / f"{i:06d}.png", cv2.resize(raw, IMAGE_SIZE, interpolation=cv2.INTER_AREA))
        # Nearest-neighbor downsampling can lose a thin exclusion mask. This
        # one-pixel support is model context only; M_write remains in raw pixels.
        small = cv2.resize(frame_masks["context_exclude"].astype(np.uint8),
                           IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        small = cv2.dilate(small, np.ones((3, 3), np.uint8))
        coverage = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        if np.any(frame_masks["context_exclude"] & ~coverage):
            raise ValueError(f"MODEL_MASK_RESIZE_LOST_CONTEXT:{i}")
        image(dst / "model_masks" / f"{i:06d}.png", small * 255)
        rows.append({
            "frame_id": i, "raw": str(raw_path), "capture_time": raw_row.get("capture_time"),
            "write": str(dst / "write" / f"{i:06d}.png"),
            "protect": str(dst / "protect" / f"{i:06d}.png"),
            "context_exclude": str(dst / "context_exclude" / f"{i:06d}.png"),
            "unknown": str(dst / "unknown" / f"{i:06d}.png"),
            "stats": frame_masks["stats"],
        })
    complete = len(trusted) == frame_count
    manifest = {
        "schema_version": "v5-scene-prep-v1", "session_id": session,
        "frame_count": len(rows), "image_domain": domain["image_domain"],
        "output_size": [shape[1], shape[0]], "inpaint_size": list(IMAGE_SIZE),
        "source_domain": ref(domain_path), "source_masks": [ref(path) for path in mask_paths],
        "config": ref(spec_path), "code": ref(Path(__file__)),
        "rows": rows, "missing_semantic_roles": missing,
        "trusted_visible_object_frames": len(trusted),
        "semantic_ready_for_product": complete,
        "model_context_mask": "M_context_exclude resized nearest plus one model pixel support",
        "write_policy": "COMPONENT_HULL_CURRENT_AND_PM2_MARGIN8_NO_TEMPORAL_OBJECT_PROTECTION",
        "source_kind": "ORIGINAL_RGB_AND_OBSERVED_MASK_CANDIDATES",
        "synthetic_pixels_used_as_geometry": False,
        "control_ground_truth": False, "training_eligible": False,
    }
    path = dst / "SCENE_PREP_MANIFEST.json"
    save(path, manifest)
    save(dst / "RESULT.json", {"status": "PREPARED" if complete else "PREPARED_WITH_SEMANTIC_GAPS",
                               "session_id": session, "manifest": ref(path),
                               "missing_semantic_roles": missing})
    return path


def _gpu_guard(project: Path) -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise RuntimeError("GPU_LEASE_REQUIRED")
    lease_path = project / "_run/current/GPU_LEASE.json"
    lease = load(lease_path)
    if lease.get("status") != "ACQUIRED" or lease.get("task_id") != f"{TASK}:scene":
        raise RuntimeError("GPU_LEASE_OWNER")
    if lease.get("gpu_process_pid") != os.getpid() or str(lease.get("gpu_id")) != visible:
        raise RuntimeError("GPU_LEASE_PROCESS")
    if not lease.get("fencing_token"):
        raise RuntimeError("GPU_LEASE_FENCE")
    return {"lease": ref(lease_path), "token": lease["fencing_token"]}


def _sam_outputs(output: dict, shape: tuple[int, int]):
    def array(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)
    masks = array(output.get("out_binary_masks", np.zeros((0, *shape), bool)))
    while masks.ndim > 3 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    if masks.ndim == 2:
        masks = masks[None]
    ids = array(output.get("out_obj_ids", [])).reshape(-1)
    scores = array(output.get("out_probs", [])).reshape(-1)
    if masks.shape[1:] != shape or len(masks) != len(ids) or len(ids) != len(scores):
        raise ValueError("SAM_OUTPUT_DOMAIN")
    return masks.astype(bool), ids, scores


def role_masks(spec_path: Path, output: Path, *, dry_run: bool = False) -> Path | None:
    """Thin V5 adapter for the frozen SAM31 role producer.

    It preserves original SAM instance labels and confidence. Forward and
    reverse track IDs are direction-local, never treated as object identity.
    """
    spec = load(spec_path)
    project = Path(spec["project_root"]).resolve(strict=True)
    domain_path = Path(spec["domain_manifest"])
    domain = load(domain_path)
    if domain["session_id"] != spec["session_id"]:
        raise ValueError("SCENE_SESSION_MISMATCH")
    if len(domain["frames"]) != domain["frame_count"] or \
       domain["frame_count"] != domain.get("full_source_frame_count", domain["frame_count"]):
        raise ValueError("SCENE_NOT_FULL_SESSION")
    roles = spec["roles"]
    if not roles or len({row["role"] for row in roles}) != len(roles):
        raise ValueError("SAM_ROLE_SPEC")
    for role in roles:
        _role_group(role["role"])
        anchor = int(role.get("anchor_frame", 0))
        if not 0 <= anchor < domain["frame_count"]:
            raise ValueError("SAM_ANCHOR_RANGE")
        if not role.get("text"):
            raise ValueError("SAM_PROMPT_EMPTY")
    weight = project / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt"
    adapter_file = project / "src/chaoyang/pipeline/sam31_compat_adapter_v1.py"
    dependency_refs = {"weight": ref(weight), "adapter": ref(adapter_file),
                       "vendor_license": ref(project / "vendor/SAM3/LICENSE")}
    if dry_run:
        print(json.dumps({"stage": "role_masks", "session_id": domain["session_id"],
                          "frame_count": domain["frame_count"], "image_domain": domain["image_domain"],
                          "roles": roles, "dependencies": dependency_refs,
                          "resource": "SINGLE_GOVERNED_GPU_LEASE", "algorithm_executed": False},
                         ensure_ascii=False))
        return None
    lease = _gpu_guard(project)
    sys.path.insert(0, str(project / "vendor/SAM3"))
    from chaoyang.pipeline import sam31_compat_adapter_v1 as sam_adapter
    import torch
    sam_adapter.PROJECT_ROOT = project
    adapter, model_evidence = sam_adapter.build_pinned_adapter(
        official_code_root=project / "vendor/SAM3", checkpoint_path=weight)
    model = adapter.model
    destination = output_root(project, output)
    count = domain["frame_count"]
    shape = (int(domain["height"]), int(domain["width"]))
    rows = [[] for _ in range(count)]
    for prompt in roles:
        name = prompt["role"]
        role_dir = destination / name
        role_dir.mkdir()
        anchor = int(prompt.get("anchor_frame", 0))
        kwargs = {}
        if prompt.get("boxes_xyxy"):
            boxes = np.asarray(prompt["boxes_xyxy"], dtype=float).copy()
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError("SAM_SEED_BOX_SHAPE")
            boxes[:, 2:] -= boxes[:, :2]
            boxes /= np.array([shape[1], shape[0], shape[1], shape[0]])
            if not np.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any():
                raise ValueError("SAM_SEED_BOX_RANGE")
            kwargs = {"boxes_xywh": boxes.tolist(), "box_labels": [1] * len(boxes),
                      "clear_old_boxes": True}
        # Separate states avoid carrying one direction's tracking history into
        # the other. IDs remain direction-local and are never used as truth.
        for direction in ("forward", "reverse"):
            if direction == "reverse" and anchor == 0:
                continue
            state = model.init_state(resource_path=str(Path(domain["frames"][0]["rgb"]).parent),
                                     offload_video_to_cpu=True, async_loading_frames=False)
            try:
                _, seed = model.add_prompt(inference_state=state, frame_idx=anchor,
                                           text_str=prompt["text"], output_prob_thresh=0.5,
                                           **kwargs)
                def save_frame(frame_id: int, result: dict) -> None:
                    masks, ids, scores = _sam_outputs(result, shape)
                    labels = np.zeros(shape, np.uint16)
                    evidence = []
                    for mask_value, instance_id, score in zip(masks, ids, scores):
                        instance_id = int(instance_id)
                        if not 0 <= instance_id < 65534:
                            raise ValueError("SAM_INSTANCE_ID_RANGE")
                        yy, xx = np.where(mask_value)
                        box = None if not len(xx) else [int(xx.min()), int(yy.min()),
                                                        int(xx.max() + 1), int(yy.max() + 1)]
                        evidence.append({"role": name, "raw_instance_id": instance_id,
                                         "score": float(score), "area_px": int(len(xx)),
                                         "box_xyxy": box, "direction": direction,
                                         "anatomical_side": "UNKNOWN",
                                         "identity_authority": "SAM_DIRECTION_LOCAL_NOT_GOLD"})
                        labels[mask_value] = instance_id + 1
                    image(role_dir / f"{frame_id:06d}.png", labels)
                    rows[frame_id] = [r for r in rows[frame_id] if r["role"] != name] + evidence
                if direction == "forward":
                    save_frame(anchor, seed)
                start = anchor
                maximum = count - anchor if direction == "forward" else anchor
                for idx, result in model.propagate_in_video(
                        inference_state=state, start_frame_idx=start,
                        max_frame_num_to_track=maximum, reverse=direction == "reverse",
                        output_prob_thresh=0.5):
                    idx = int(idx)
                    if not 0 <= idx < count:
                        raise ValueError("SAM_PROPAGATION_FRAME")
                    if direction == "forward" and idx == anchor:
                        # A provisional empty seed is replaced by actual tracking.
                        masks, _, _ = _sam_outputs(result, shape)
                        if len(masks):
                            save_frame(idx, result)
                    else:
                        save_frame(idx, result)
            finally:
                state.clear()
                torch.cuda.empty_cache()
        # Explicit zero observation for missed frames; no fabricated masks.
        for i in range(count):
            path = role_dir / f"{i:06d}.png"
            if not path.exists():
                image(path, np.zeros(shape, np.uint16))
    manifest = {
        "schema_version": "v5-scene-role-masks-v1", "session_id": domain["session_id"],
        "frame_count": count, "domain": domain["image_domain"], "roles": roles,
        "rows": rows, "mask_root": str(destination), "source_manifest": ref(domain_path),
        "model": model_evidence, "dependencies": dependency_refs, "gpu_lease": lease,
        "code": ref(Path(__file__)), "config": ref(spec_path),
        "quality_adopted": False, "independent_accuracy": "UNVERIFIED",
        "anatomical_side_verified": False, "object_identity_verified": False,
        "control_ground_truth": False, "training_eligible": False,
    }
    path = destination / "MASK_MANIFEST.json"
    save(path, manifest)
    save(destination / "RESULT.json", {"status": "EXECUTED_AWAITING_VISUAL_QA",
                                       "session_id": domain["session_id"], "manifest": ref(path)})
    return path


def inpaint(spec_path: Path, output: Path) -> Path:
    spec = load(spec_path)
    project = Path(spec["project_root"]).resolve(strict=True)
    prep_path = Path(spec["scene_prep_manifest"])
    prep = load(prep_path)
    if prep["session_id"] != spec["session_id"]:
        raise ValueError("SCENE_SESSION_MISMATCH")
    if not prep.get("semantic_ready_for_product"):
        raise ValueError("BLOCKED_INPUT_SCENE_SEMANTICS")
    checked_ref(prep["source_domain"])
    for source in prep["source_masks"]:
        checked_ref(source)
    lease = _gpu_guard(project)
    vendor = project / "vendor/ProPainter"
    weights = [vendor / "weights" / name for name in
               ("ProPainter.pth", "raft-things.pth", "recurrent_flow_completion.pth")]
    pins = [ref(path) for path in weights]
    dst = output_root(project, output)
    command = [sys.executable, "-B", str(vendor / "inference_propainter.py"),
               "--video", str(prep_path.parent / "frames"),
               "--mask", str(prep_path.parent / "model_masks"),
               "--output", str(dst / "upstream"), "--width", "960", "--height", "720",
               "--mask_dilation", "0", "--ref_stride", "10", "--neighbor_length", "10",
               "--subvideo_length", "80", "--raft_iter", "20", "--save_fps", "30",
               "--save_frames", "--fp16"]
    save(dst / "INVOCATION.json", {"command": command, "weights": pins,
                                   "vendor_license": ref(vendor / "LICENSE"),
                                   "gpu_lease": lease, "network_download_allowed": False})
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    with (dst / "PROPAINTER.log").open("xb") as log:
        completed = subprocess.run(command, cwd=vendor, env=env, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"PROPAINTER_EXIT:{completed.returncode}:{dst}")
    generated = sorted((dst / "upstream/frames/frames").glob("*.png"))
    if len(generated) != prep["frame_count"]:
        raise ValueError(f"PROPAINTER_FRAME_COUNT:{len(generated)}")
    (dst / "clean").mkdir()
    rows = []
    for i, (source, generated_path) in enumerate(zip(prep["rows"], generated)):
        raw = cv2.imread(source["raw"], cv2.IMREAD_COLOR)
        fake = cv2.imread(str(generated_path), cv2.IMREAD_COLOR)
        if raw is None or fake is None:
            raise ValueError(f"CLEAN_DECODE:{i}")
        fake = cv2.resize(fake, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_LINEAR)
        write = cv2.imread(source["write"], cv2.IMREAD_GRAYSCALE) > 0
        protect = cv2.imread(source["protect"], cv2.IMREAD_GRAYSCALE) > 0
        clean = composite_clean(raw, fake, write, protect)
        target = dst / "clean" / f"{i:06d}.png"
        image(target, clean)
        rows.append({"frame_id": i, "raw": source["raw"], "clean": str(target),
                     "write": source["write"], "protect": source["protect"],
                     "context_exclude": source["context_exclude"], "unknown": source["unknown"],
                     "outside_write_changed_px": int(np.any(clean != raw, axis=2)[~write].sum()),
                     "protected_changed_px": int(np.any(clean != raw, axis=2)[protect].sum()),
                     "pixel_source_inside_write": "PROPAINTER_SYNTHETIC_OFFLINE_VISUAL"})
    manifest = {
        "schema_version": "v5-scene-clean-v1", "session_id": prep["session_id"],
        "frame_count": len(rows), "image_domain": prep["image_domain"], "frames": rows,
        "scene_prep": ref(prep_path), "config": ref(spec_path), "code": ref(Path(__file__)),
        "weights": pins, "new_inpainting_performed": True,
        "future_frames_used": True, "geometry_input_allowed": False,
        "hidden_object_truth": False, "visual_quality_adopted": False,
        "semantic_ready_for_product": prep["semantic_ready_for_product"],
        "control_ground_truth": False, "training_eligible": False,
    }
    path = dst / "SCENE_CLEAN_MANIFEST.json"
    save(path, manifest)
    save(dst / "RESULT.json", {"status": "EXECUTED_AWAITING_VISUAL_QA", "session_id": prep["session_id"],
                               "returncode": completed.returncode, "manifest": ref(path)})
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("preflight", "role_masks", "prepare", "inpaint"), required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run and args.stage != "role_masks":
        raise ValueError("DRY_RUN_ROLE_MASKS_ONLY")
    result = (role_masks(args.spec, args.output, dry_run=args.dry_run) if args.stage == "role_masks"
              else {"preflight": preflight, "prepare": prepare, "inpaint": inpaint}[args.stage](args.spec, args.output))
    print(result)


if __name__ == "__main__":
    main()
