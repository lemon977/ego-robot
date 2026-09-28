#!/usr/bin/env python3
"""Bind R2 product caches and exercise the occlusion contract synthetically.

This wave never upgrades real-session occlusion: no suitable registered scene
depth exists for the four product sessions.  It proves the compositor fails
closed and records exact content identities for subsequent resume decisions.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from chaoyang.ops.run_human_to_robot_root_cause_gated_r2_wave0 import ATTEMPT, REPO, TASK, ref, write_json
from chaoyang.pipeline.occlusion_compositor_v1 import (
    DepthQualityEvidence,
    ObjectPixelSource,
    Ownership,
    exclude_removed_foreground_depth,
    resolve_ownership,
)
from chaoyang.pipeline.r2_dependency_signature import affected_stages, build_product_binding


CASES = {
    "031": "play_cards_0915_031",
    "007": "get_potato_chips_0915_007",
    "103": "get_potato_chips_0902_103",
    "042": "play_cards_0902_042",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def synthetic_occlusion_receipt() -> dict:
    shape = (2, 3)
    human_equipment = np.asarray([[True, False, False], [False, False, True]], dtype=np.bool_)
    raw_scene_depth = np.asarray([[0.35, 0.8, 0.8], [0.8, 0.8, 0.4]], dtype=np.float64)
    cleaned_depth, cleaned_valid = exclude_removed_foreground_depth(
        scene_depth_m=raw_scene_depth,
        scene_depth_valid=np.ones(shape, dtype=np.bool_),
        human_equipment_mask=human_equipment,
    )
    if np.any(cleaned_valid[human_equipment]) or not np.isnan(cleaned_depth[human_equipment]).all():
        raise AssertionError("removed foreground depth survived sanitization")

    # A visible object is ordered against Robot only where legal object depth
    # exists.  The deleted-human locations are deliberately absent, rather
    # than being reused as a nearer scene surface.
    object_mask = np.asarray([[False, True, True], [False, True, False]], dtype=np.bool_)
    robot_mask = np.asarray([[True, True, True], [False, True, True]], dtype=np.bool_)
    quality = DepthQualityEvidence(*(np.ones(shape, dtype=np.bool_) for _ in range(6)))
    provenance = np.where(object_mask, int(ObjectPixelSource.RAW_VISIBLE),
                          int(ObjectPixelSource.NONE_UNKNOWN)).astype(np.uint8)
    result = resolve_ownership(
        human_mask=human_equipment,
        object_amodal_mask=object_mask,
        object_depth_m=np.where(object_mask, 0.8, np.nan),
        object_depth_valid=object_mask.copy(),
        robot_alpha_mask=robot_mask,
        robot_depth_m=np.where(robot_mask, 1.0, np.nan),
        robot_depth_valid=robot_mask.copy(),
        stereo_depth_valid=object_mask.copy(),
        depth_quality_evidence=quality,
        object_rgb=np.zeros((*shape, 3), dtype=np.uint8),
        object_pixel_source=provenance,
        contact_decision_mask=object_mask & robot_mask,
    )
    if not np.all(result.ownership[object_mask & robot_mask] == int(Ownership.OBJECT_FRONT)):
        raise AssertionError("known object/Robot ordering did not use metric depth")
    return {
        "schema_version": "HUMAN_TO_ROBOT_R2_OCCLUSION_CONTRACT_EVIDENCE_V1",
        "execution": "EXECUTED",
        "structure": "PASS",
        "quality": "INCONCLUSIVE_REAL_SESSION",
        "adoption": "NOT_ADOPTED",
        "synthetic_same_domain_metric_ordering": "PASS",
        "removed_human_equipment_depth_excluded": True,
        "removed_depth_pixels": int(human_equipment.sum()),
        "unknown_depth_policy": "UNKNOWN_NO_SYNTHESIZED_SURFACE",
        "real_session_depth_available": False,
        "claim_limit": "Synthetic contract evidence only; no R2 product session has admitted registered scene depth, so real-session occlusion remains UNKNOWN.",
    }


def main() -> int:
    binding_rows = []
    for short, session_id in CASES.items():
        root = ATTEMPT / f"lanes/lane2_motion/product_candidate_{short}_wave6"
        result_path = root / "RESULT.json"
        result = load(result_path)
        if result.get("session_id") != session_id:
            raise ValueError(f"SESSION_BINDING_MISMATCH:{short}")
        files = {
            "scene_clean": Path(result["scene"]["path"]),
            "motion_r0": Path(result["motion"]["path"]),
            "camera_domain": Path(result["domain"]["path"]),
            "robot_asset": REPO / "assets/robot/ROBOT_ASSET_PIN.json",
            "renderer_code": REPO / "src/chaoyang/pipeline/v5_product.py",
            "product_entry_code": REPO / "src/chaoyang/ops/run_human_to_robot_r2_candidate_product.py",
        }
        binding = build_product_binding(
            files=files,
            render_config={
                "renderer": "PYBULLET_TINY_RENDERER",
                "invalid_side_policy": "NOT_RENDERED_NO_FILL_NO_HOLD",
                "overlay": "ROBOT_OVER_REJECTED_CLEAN",
                "fps": 30,
            },
            occlusion_status="UNKNOWN_NO_REGISTERED_SCENE_DEPTH",
        )
        binding.update({
            "task_id": TASK,
            "session_id": session_id,
            "created_at": now(),
            "product_result": ref(result_path),
            "product_video": result["video"],
            "resume_policy": "REUSE_ONLY_WHEN_SIGNATURE_SHA256_MATCHES",
        })
        binding_path = root / "CACHE_BINDING.json"
        if binding_path.is_file():
            previous = load(binding_path)
            if previous.get("signature_sha256") != binding["signature_sha256"]:
                raise ValueError(f"CACHE_BINDING_DRIFT:{short}")
        else:
            write_json(binding_path, binding)
        binding_rows.append({
            "session_id": session_id,
            "signature_sha256": binding["signature_sha256"],
            "binding": ref(binding_path),
        })

    occlusion = synthetic_occlusion_receipt()
    occlusion.update({"task_id": TASK, "created_at": now()})
    occlusion_path = ATTEMPT / "lanes/lane1_scene/OCCLUSION_CONTRACT_WAVE8.json"
    write_json(occlusion_path, occlusion)
    invalidation = {
        key: sorted(affected_stages({key}))
        for key in ("scene_masks", "roi", "render_config", "scene_depth")
    }
    receipt = {
        "schema_version": "HUMAN_TO_ROBOT_R2_DEPENDENCY_AND_OCCLUSION_WAVE8_V1",
        "task_id": TASK,
        "created_at": now(),
        "product_cache_bindings": binding_rows,
        "invalidation_examples": invalidation,
        "occlusion_contract": ref(occlusion_path),
        "real_session_occlusion": "UNKNOWN",
        "training_eligible": False,
        "control_ground_truth": False,
        "physical_deployable": False,
        "external_metric_authority": False,
    }
    receipt_path = ATTEMPT / "PROGRESS_WAVE8.json"
    write_json(receipt_path, receipt)

    for lane, action in (
        ("lane1_scene", "Synthetic occlusion contract passed; all real-session occlusion remains UNKNOWN because registered scene depth is absent."),
        ("lane2_motion", "Four product candidates are content-bound; resume is allowed only for an identical dependency signature."),
    ):
        state_path = ATTEMPT / f"lanes/{lane}/STATE.json"
        state = load(state_path)
        artifacts = state.get("latest_artifacts", [])
        existing = {item.get("path") for item in artifacts if isinstance(item, dict)}
        for path in ((occlusion_path,) if lane == "lane1_scene" else tuple(Path(row["binding"]["path"]) for row in binding_rows)):
            item = ref(path)
            if item["path"] not in existing:
                artifacts.append(item); existing.add(item["path"])
        state.update(current_action=action, latest_artifacts=artifacts,
                     writer={"pid": os.getpid(), "proc_start_ticks": None}, updated_at=now())
        write_json(state_path, state)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
