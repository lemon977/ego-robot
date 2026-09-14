"""v9 diagnostic-only SAM boundary-reprompt short-window runner."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .forearm_boundary_prompt import (
    final_forearm_prompt_support,
    local_support_from_proposal,
    propose_boundary_reprompt,
    select_boundary_reprompt_candidate,
)
from .mask_producer_sam2 import (
    MaskConfig,
    _InferencePredictor,
    _bidirectional_mask_consistency,
    _clamp_px,
    _load_json_nofollow,
    _load_rgb,
    _named_point,
    _project_cylinder_mask,
    _read_regular_nofollow,
    _segment_hand_side,
    _select_wrist_connected_forearm,
    _verify_regular_sha256_nofollow,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_mask(path: Path, value: np.ndarray) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), value.astype(np.uint8) * 255):
        raise RuntimeError(f"mask write failed: {path}")
    return {"path": str(path), "sha256": _sha(path), "producer": "v9_short_window"}


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    layer = np.empty_like(image)
    layer[:] = color
    result[mask] = cv2.addWeighted(result[mask], 0.45, layer[mask], 0.55, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    return result


def _panel(image: np.ndarray, text: str, width: int = 320) -> np.ndarray:
    height = int(round(image.shape[0] * width / image.shape[1]))
    result = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(result, (0, 0), (width, 30), (0, 0, 0), -1)
    cv2.putText(result, text, (7, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def _boundary_connected(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def _validate_output(path: Path, project_root: Path, task_id: str, session_id: str) -> Path:
    expected = (
        project_root / "_run" / task_id / "sessions" / session_id / "diagnostic" / "short_window"
    ).resolve()
    resolved = path.resolve()
    if resolved != expected:
        raise RuntimeError(f"v9 output must equal task-scoped short_window root: {expected}")
    return resolved


def _task_ref_path(project_root: Path, ref: dict[str, str]) -> Path:
    path = project_root / ref["path"]
    resolved_parent = path.parent.resolve()
    if project_root != resolved_parent and project_root not in resolved_parent.parents:
        raise RuntimeError("task implementation ref escapes project root")
    _verify_regular_sha256_nofollow(path, ref["sha256"])
    return path


def _validate_task_frozen_cli_bindings(
    frozen: dict[str, Any], args: argparse.Namespace
) -> None:
    expected_bindings = (
        (
            frozen.get("source_manifest", {}),
            Path(args.source_manifest),
            args.source_manifest_sha256,
        ),
        (
            {
                "path": frozen.get("sam2", {}).get("checkpoint"),
                "sha256": frozen.get("sam2", {}).get("checkpoint_sha256"),
            },
            Path(args.sam2_checkpoint),
            args.sam2_checkpoint_sha256,
        ),
        (
            {
                "path": frozen.get("sam2", {}).get("config_file"),
                "sha256": frozen.get("sam2", {}).get("config_sha256"),
            },
            Path(args.sam2_config_file),
            args.sam2_config_sha256,
        ),
        (
            {
                "path": frozen.get("object6d_read_only_evidence", {}).get(
                    "pose_path"
                ),
                "sha256": frozen.get("object6d_read_only_evidence", {}).get(
                    "pose_sha256"
                ),
            },
            Path(args.object_pose_npz),
            args.object_pose_sha256,
        ),
    )
    for ref, actual_path, actual_sha in expected_bindings:
        if not (
            ref.get("sha256") == actual_sha
            and Path(str(ref.get("path"))).absolute() == actual_path.absolute()
        ):
            raise RuntimeError("CLI frozen input does not match task-card identity")
    expected_sam2_pythonpath = Path(
        frozen.get("sam2", {}).get("implementation_root", "")
    ).parent.absolute()
    if Path(args.sam2_root).absolute() != expected_sam2_pythonpath:
        raise RuntimeError("SAM2 PythonPath root does not match task-card implementation")


def run(args: argparse.Namespace) -> None:
    project_root = Path(__file__).absolute().parents[1]
    source_path = Path(args.source_manifest).absolute()
    context_path = Path(args.session_context).absolute()
    task_path = Path(args.task_card).absolute()
    source = _load_json_nofollow(source_path, args.source_manifest_sha256)
    context = _load_json_nofollow(context_path, args.session_context_sha256)
    task_payload = _read_regular_nofollow(task_path, args.task_card_sha256)
    task = yaml.safe_load(task_payload.decode("utf-8"))
    if not isinstance(task, dict) or not (
        task.get("status") == "AUTHORIZED_G2_CALIBRATION"
        and task.get("authorization_scope") == "V9_SHORT_WINDOW_DIAGNOSTIC_ONLY"
        and task.get("full_h20_authorized") is False
        and task.get("session_id") == args.session_id
        and task.get("product_line") == "004_CONTACT_GOLD"
    ):
        raise RuntimeError("task card does not authorize v9 short-window diagnostic")
    expected_task_path = project_root / "_run" / str(task["task_id"]) / "TASK_CARD.yaml"
    if task_path != expected_task_path:
        raise RuntimeError("task card must be the exact task-scoped ordinary file")
    frozen = task.get("frozen_inputs", {})
    _validate_task_frozen_cli_bindings(frozen, args)
    implementation = frozen.get("implementation", {})
    required_implementation = {
        "mask_producer": Path(__file__).with_name("mask_producer_sam2.py"),
        "boundary_prompt": Path(__file__).with_name("forearm_boundary_prompt.py"),
        "short_window_diagnostic": Path(__file__),
        "mask_tests": project_root / "tests" / "test_mask_producer_sam2.py",
        "boundary_tests": project_root / "tests" / "test_forearm_boundary_prompt.py",
    }
    for key, actual in required_implementation.items():
        ref = implementation.get(key)
        if not isinstance(ref, dict) or _task_ref_path(project_root, ref).absolute() != actual.absolute():
            raise RuntimeError(f"task implementation pin mismatch: {key}")
    output_root = _validate_output(
        Path(args.output_root), project_root, str(task["task_id"]), args.session_id
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty v9 output: {output_root}")
    if not (
        context.get("execution_allowed") is True
        and context.get("execution_blockers") == []
        and context.get("session_id") == args.session_id
        and context.get("product_line") == "004_CONTACT_GOLD"
        and context.get("task_card_ref", {}).get("sha256") == args.task_card_sha256
        and context.get("source", {}).get("source_manifest_sha256") == args.source_manifest_sha256
    ):
        raise RuntimeError("context does not authorize exact v9 inputs")
    requested = list(args.frames)
    if requested != list(range(requested[0], requested[-1] + 1)) or len(requested) > 24:
        raise RuntimeError("v9 selector must be one ascending contiguous window of at most 24 frames")
    if requested != list(task.get("diagnostic_gate", {}).get("window_frames", [])):
        raise RuntimeError("CLI window does not equal task-card diagnostic window")

    sessions = [item for item in source["sessions"] if item["session_id"] == args.session_id]
    if len(sessions) != 1:
        raise RuntimeError("source session identity is not unique")
    session = sessions[0]
    frames = session["frames"]
    if any(index < 0 or index >= len(frames) for index in requested):
        raise RuntimeError("v9 window outside source manifest")
    overlay = task["dimensionless_calibration_overlay"]
    route = task["v9_reprompt_gates"]
    config = MaskConfig.from_overlay(overlay)
    _verify_regular_sha256_nofollow(Path(args.sam2_checkpoint), args.sam2_checkpoint_sha256)
    _verify_regular_sha256_nofollow(Path(args.sam2_config_file), args.sam2_config_sha256)
    _verify_regular_sha256_nofollow(Path(args.object_pose_npz), args.object_pose_sha256)

    import sys
    os.environ.setdefault("PYTHONPATH", args.sam2_root)
    if args.sam2_root not in sys.path:
        sys.path.insert(0, args.sam2_root)
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not torch.cuda.is_available():
        raise RuntimeError("v9 diagnostic requires task-pinned CUDA")
    predictor = _InferencePredictor(
        SAM2ImagePredictor(build_sam2(args.sam2_config, args.sam2_checkpoint, device="cuda")), torch
    )
    object_pose_bytes = _read_regular_nofollow(
        Path(args.object_pose_npz), args.object_pose_sha256
    )
    with np.load(io.BytesIO(object_pose_bytes), allow_pickle=False) as object_npz:
        transforms = np.asarray(object_npz["T_object_to_camera"], np.float64)
        radius = float(object_npz["cylinder_radius_m"])
        cylinder_height = float(object_npz["cylinder_height_m"])
    if transforms.shape != (len(frames), 4, 4):
        raise RuntimeError("Object6D frame shape mismatch")

    output_root.mkdir(parents=True, exist_ok=False)
    rows: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    temporal_grays: list[np.ndarray] = []
    temporal_masks: list[np.ndarray] = []
    side_temporal_masks: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    critical = set(task["diagnostic_gate"]["critical_frames"])
    for frame_index in requested:
        frame = frames[frame_index]
        rgb = _load_rgb(frame)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = rgb.shape[:2]
        metadata = _load_json_nofollow(
            Path(frame["metadata"]["path"]), frame["metadata"]["sha256"]
        )
        intrinsics = np.asarray(metadata["metadata"]["k"], np.float64)
        analytic_object, _ = _project_cylinder_mask(
            transforms[frame_index], intrinsics, radius, cylinder_height, width, height
        )
        object_scale = math.sqrt(float(np.count_nonzero(analytic_object)))
        contact = _clamp_px(
            config.contact_ratio * object_scale,
            min(width, height),
            config.contact_min,
            config.contact_max,
        )
        contact_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * contact + 1, 2 * contact + 1)
        )
        prompt_exclusion = cv2.dilate(
            analytic_object.astype(np.uint8), contact_kernel
        ).astype(bool)
        predictor.set_image(rgb)

        side_final: dict[str, np.ndarray] = {}
        side_metrics: dict[str, Any] = {}
        output_refs: dict[str, Any] = {}
        raw_panel = bgr.copy()
        initial_panel = bgr.copy()
        final_panel = bgr.copy()
        for side, color in (("left", (255, 255, 0)), ("right", (255, 0, 255))):
            hand = metadata["entities"]["hands"][side]
            debug: dict[str, np.ndarray] = {}
            combined, _, _, base_metrics = _segment_hand_side(
                predictor, hand, width, height, config, prompt_exclusion, debug_masks=debug
            )
            wrist = _named_point(hand, "wrist")
            palm = _named_point(hand, "palm_center")
            wrist_width = float(
                np.linalg.norm(
                    _named_point(hand, "index_proximal")
                    - _named_point(hand, "pinky_proximal")
                )
            )
            initial_forearm, initial_metrics = _select_wrist_connected_forearm(
                debug["regularized_forearm_sam"],
                debug["wrist_candidate"],
                debug["wrist_region"],
                debug["prior_corridor"],
                wrist,
                palm,
                wrist_width,
            )
            proposal = propose_boundary_reprompt(
                initial_forearm,
                wrist_width,
                outward_prior=wrist - palm,
                max_gap_wrist_ratio=float(route["max_gap_wrist_ratio"]),
                min_edge_ambiguity_ratio=float(route["min_edge_ambiguity_ratio"]),
                min_edge_direction_cosine=float(route["min_edge_direction_cosine"]),
            )
            reprompt_metrics: dict[str, Any] = {"held": 0.0, "route": proposal["status"]}
            final_forearm = initial_forearm
            sleeve_points = np.asarray(
                base_metrics["forearm_prompt_points_xy"], np.float32
            )
            sleeve_xy = np.rint(sleeve_points).astype(int)
            sleeve_xy[:, 0] = np.clip(sleeve_xy[:, 0], 0, width - 1)
            sleeve_xy[:, 1] = np.clip(sleeve_xy[:, 1], 0, height - 1)
            usable_sleeve_points = sleeve_points[
                ~prompt_exclusion[sleeve_xy[:, 1], sleeve_xy[:, 0]]
            ]
            sleeve_evidence_availability = float(
                len(usable_sleeve_points) / max(len(sleeve_points), 1)
            )
            if proposal["status"] == "REPROMPT_ELIGIBLE":
                local_support = local_support_from_proposal(
                    initial_forearm.shape, proposal
                )
                positive = np.stack(
                    [proposal["terminal_center_xy"], proposal["target_positive_xy"]]
                )
                positive = np.concatenate(
                    [usable_sleeve_points, wrist[None], positive], axis=0
                ).astype(np.float32)
                negative = proposal["table_negative_xy"]
                y, x = np.nonzero(initial_forearm)
                prompt_union = np.concatenate([positive, negative], axis=0)
                pad = float(route["box_padding_wrist_ratio"]) * wrist_width
                lo = np.minimum([x.min(), y.min()], prompt_union.min(axis=0)) - pad
                hi = np.maximum([x.max(), y.max()], prompt_union.max(axis=0)) + pad
                box = np.asarray(
                    [
                        np.clip(lo[0], 0, width - 1),
                        np.clip(lo[1], 0, height - 1),
                        np.clip(hi[0], 0, width - 1),
                        np.clip(hi[1], 0, height - 1),
                    ],
                    np.float32,
                )
                masks, scores, _ = predictor.predict(
                    point_coords=np.concatenate([positive, negative], axis=0),
                    point_labels=np.concatenate(
                        [np.ones(len(positive), np.int32), np.zeros(len(negative), np.int32)]
                    ),
                    box=box,
                    multimask_output=True,
                )
                repaired, selected_metrics = select_boundary_reprompt_candidate(
                    masks,
                    scores,
                    positive_points=positive,
                    negative_points=negative,
                    initial_component=initial_forearm,
                    target_edge=str(proposal["edge"]),
                    object_core_exclusion=analytic_object,
                    uncertain_contact_exclusion=prompt_exclusion & ~analytic_object,
                    local_added_support=local_support,
                    observed_terminal_width_px=float(
                        proposal["observed_terminal_width_px"]
                    ),
                    min_initial_recall=float(route["min_initial_recall"]),
                    max_expansion_ratio=float(route["max_expansion_ratio"]),
                    min_boundary_width_ratio=float(
                        route["min_boundary_width_ratio"]
                    ),
                    max_boundary_width_ratio=float(
                        route["max_boundary_width_ratio"]
                    ),
                )
                reprompt_metrics = {**selected_metrics, "route": "SECOND_SAM_BOUNDARY_REPROMPT"}
                reprompt_dir = (
                    output_root / "frames" / f"{frame_index:05d}" / side
                )
                output_refs.setdefault(side, {})["reprompt_raw_sam_candidates"] = [
                    _write_mask(
                        reprompt_dir / f"reprompt_raw_sam_candidate_{index}.png",
                        candidate,
                    )
                    for index, candidate in enumerate(masks.astype(bool))
                ]
                if selected_metrics["held"] == 0.0:
                    final_forearm = repaired
                output_refs.setdefault(side, {})["local_added_support"] = _write_mask(
                    output_root
                    / "frames"
                    / f"{frame_index:05d}"
                    / side
                    / "local_added_support.png",
                    local_support,
                )
            side_final[side] = combined | final_forearm
            final_support = final_forearm_prompt_support(
                final_forearm,
                usable_sleeve_points,
                min_prompt_recall=config.forearm_min_recall,
            )
            final_boundary = final_support["final_boundary_connected"]
            side_hold = float(
                base_metrics["hand_candidate"]["held"] == 1.0
                or base_metrics["wrist_candidate"]["held"] == 1.0
                or sleeve_evidence_availability < config.forearm_min_recall
                or reprompt_metrics["held"] == 1.0
                or final_support["final_forearm_supported"] == 0.0
            )
            proposal_json = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in proposal.items()
            }
            side_metrics[side] = {
                "base": base_metrics,
                "original_sleeve_prompt_count": len(sleeve_points),
                "usable_sleeve_prompt_count": len(usable_sleeve_points),
                "excluded_sleeve_prompt_count": len(sleeve_points)
                - len(usable_sleeve_points),
                "usable_sleeve_prompt_ratio": sleeve_evidence_availability,
                "usable_sleeve_prompt_points_xy": usable_sleeve_points.tolist(),
                "initial_forearm": initial_metrics,
                "proposal": proposal_json,
                "reprompt": reprompt_metrics,
                **final_support,
                "final_boundary_connected": final_boundary,
                "diagnostic_hold": side_hold,
                "geometric_bridge_pixels": 0.0,
            }
            side_dir = output_root / "frames" / f"{frame_index:05d}" / side
            output_refs.setdefault(side, {}).update({
                "raw_selected_forearm_sam": _write_mask(
                    side_dir / "raw_selected_forearm_sam.png", debug["raw_selected_forearm_sam"]
                ),
                "regularized_forearm_sam": _write_mask(
                    side_dir / "regularized_forearm_sam.png", debug["regularized_forearm_sam"]
                ),
                "initial_wrist_component": _write_mask(
                    side_dir / "initial_wrist_component.png", initial_forearm
                ),
                "final_sam_forearm": _write_mask(side_dir / "final_sam_forearm.png", final_forearm),
            })
            raw_panel = _overlay(raw_panel, debug["raw_selected_forearm_sam"], color)
            initial_panel = _overlay(initial_panel, initial_forearm, color)
            final_panel = _overlay(final_panel, final_forearm, color)

            analysis_size = (
                max(1, int(round(width * config.temporal_analysis_ratio))),
                max(1, int(round(height * config.temporal_analysis_ratio))),
            )
            side_temporal_masks[side].append(
                cv2.resize(
                    final_forearm.astype(np.uint8),
                    analysis_size,
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            )

        combined_pre = side_final["left"] | side_final["right"]
        pre_overlap = int(np.count_nonzero(combined_pre & analytic_object))
        combined_post = combined_pre & ~analytic_object
        post_overlap = int(np.count_nonzero(combined_post & analytic_object))
        output_refs["combined_pre_object_protection"] = _write_mask(
            output_root / "frames" / f"{frame_index:05d}" / "combined_pre_object_protection.png",
            combined_pre,
        )
        output_refs["combined_post_object_protection"] = _write_mask(
            output_root / "frames" / f"{frame_index:05d}" / "combined_post_object_protection.png",
            combined_post,
        )
        records.append(
            {
                "frame_index": frame_index,
                "source_image_sha256": frame["image"]["sha256"],
                "sides": side_metrics,
                "pre_object_overlap_pixels": pre_overlap,
                "post_object_overlap_pixels": post_overlap,
                "outputs": output_refs,
            }
        )
        analysis_size = (
            max(1, int(round(width * config.temporal_analysis_ratio))),
            max(1, int(round(height * config.temporal_analysis_ratio))),
        )
        temporal_grays.append(
            cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), analysis_size, interpolation=cv2.INTER_AREA)
        )
        temporal_masks.append(
            cv2.resize(combined_post.astype(np.uint8), analysis_size, interpolation=cv2.INTER_NEAREST).astype(bool)
        )
        if frame_index in critical:
            rows.append(
                np.concatenate(
                    [
                        _panel(bgr, f"RAW {frame_index}"),
                        _panel(raw_panel, "raw SAM L/R"),
                        _panel(initial_panel, "initial wrist CC"),
                        _panel(final_panel, "v9 second-SAM final"),
                    ],
                    axis=1,
                )
            )

    temporal_scores = _bidirectional_mask_consistency(temporal_grays, temporal_masks)
    side_temporal_scores = {
        side: _bidirectional_mask_consistency(temporal_grays, masks)
        for side, masks in side_temporal_masks.items()
    }
    for record, score in zip(records, temporal_scores, strict=True):
        record["forward_backward_consistency"] = float(score)
    for record_index, record in enumerate(records):
        for side in ("left", "right"):
            record["sides"][side]["forearm_forward_backward_consistency"] = float(
                side_temporal_scores[side][record_index]
            )
    sheet = output_root / "V9_CRITICAL_4FRAME.png"
    if not cv2.imwrite(str(sheet), np.concatenate(rows, axis=0)):
        raise RuntimeError("v9 critical sheet write failed")
    temporal_gate = float(overlay["mask_forward_backward_iou_ratio"])
    machine_gate_pass = bool(
        all(
            record["post_object_overlap_pixels"] == 0
            and record["forward_backward_consistency"] >= temporal_gate
            and all(
                record["sides"][side]["diagnostic_hold"] == 0.0
                and record["sides"][side]["forearm_forward_backward_consistency"]
                >= temporal_gate
                for side in ("left", "right")
            )
            for record in records
        )
    )
    manifest = {
        "schema_version": "forearm-v9-short-window-v1",
        "artifact_state": "G2_DIAGNOSTIC_ONLY_NOT_MASK_CANDIDATE",
        "session_id": args.session_id,
        "window_frames": requested,
        "critical_frames": sorted(critical),
        "full_h20_authorized": False,
        "clean_allowed": False,
        "diagnostic_status": (
            "PENDING_INDEPENDENT_VISUAL_QA"
            if machine_gate_pass
            else "DIAGNOSTIC_HOLD"
        ),
        "machine_gate_pass": machine_gate_pass,
        "sam3_used": False,
        "source_manifest_ref": {"path": str(source_path), "sha256": args.source_manifest_sha256},
        "task_card_ref": {"path": str(task_path), "sha256": args.task_card_sha256},
        "session_context_ref": {"path": str(context_path), "sha256": args.session_context_sha256},
        "model": {
            "identity": "SAM2_1_HIERA_LARGE",
            "config_sha256": args.sam2_config_sha256,
            "checkpoint_sha256": args.sam2_checkpoint_sha256,
        },
        "implementation_refs": implementation,
        "temporal_gate": temporal_gate,
        "records": records,
        "critical_sheet": {"path": str(sheet), "sha256": _sha(sheet)},
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    temporary = output_root / "WINDOW_MANIFEST.json.tmp"
    temporary.write_bytes(encoded)
    os.replace(temporary, output_root / "WINDOW_MANIFEST.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for name in (
        "source-manifest", "source-manifest-sha256", "session-context",
        "session-context-sha256", "task-card", "task-card-sha256", "session-id",
        "output-root", "sam2-root", "sam2-config", "sam2-config-file",
        "sam2-config-sha256", "sam2-checkpoint", "sam2-checkpoint-sha256",
        "object-pose-npz", "object-pose-sha256",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
