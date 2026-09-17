"""V7.1-R3 Depth、Mask 与 Clean 的 fail-closed 开发合同。

本模块只实现可重复的合同检查和轻量数值基线，不发布 authority，也不把
内部一致性包装成外部物理精度。正式 Mask 基线固定为 SAM3.1；Stereo 的
quality evidence 不是 FoundationStereo 原生 confidence。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


class PipelineContractError(ValueError):
    """输入违反 R3 合同。"""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PipelineContractError(message)


def _finite_matrix(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.shape == shape, f"{name} must have shape {shape}")
    _require(bool(np.isfinite(result).all()), f"{name} must be finite")
    return result


def validate_depth_source_audit(record: Mapping[str, Any]) -> None:
    """验证 DEPTH-00 的坐标、尺度、置信度和外部精度边界。"""

    _require(record.get("schema_version") == "DEPTH_SOURCE_AUDIT_R3", "wrong schema")
    _require(record.get("stage") == "DEPTH-00", "wrong stage")
    _require(record.get("depth_convention") == "OPTICAL_Z", "depth must be optical-Z")
    _require(record.get("depth_unit") == "m", "depth unit must be metres")
    _require(record.get("formula") == "Z=fB/d", "metric formula must be Z=fB/d")
    _require(record.get("depth_confidence_present") is False, "native confidence must be absent")
    _require(record.get("external_metric_accuracy") == "UNKNOWN", "external accuracy is unknown")

    calibration = record.get("calibration")
    _require(isinstance(calibration, Mapping), "calibration must be an object")
    intrinsic = _finite_matrix(calibration.get("K"), (3, 3), "K")
    _require(intrinsic[0, 0] > 0 and intrinsic[1, 1] > 0, "focal length must be positive")
    baseline = calibration.get("baseline_m")
    _require(isinstance(baseline, (int, float)) and baseline > 0, "baseline_m must be positive")
    _require(
        calibration.get("identity_status") in {"VERIFIED_SAME_SESSION", "VERIFIED_SAME_DEVICE_CONTRACT"},
        "calibration identity must be verified; nearby-session borrowing is forbidden",
    )

    domains = record.get("image_domains")
    _require(isinstance(domains, Mapping), "image_domains must be an object")
    required_domains = {"raw_fisheye", "rectified_left", "selected_left", "visual_mp4"}
    _require(required_domains.issubset(domains), "all four image domains must be explicit")
    adapters = record.get("coordinate_adapters")
    _require(isinstance(adapters, list) and adapters, "coordinate adapters are required")
    for adapter in adapters:
        _require(isinstance(adapter, Mapping), "adapter must be an object")
        _require(adapter.get("source_domain") in domains, "adapter source domain is unknown")
        _require(adapter.get("target_domain") in domains, "adapter target domain is unknown")
        sha = adapter.get("implementation_sha256")
        _require(isinstance(sha, str) and len(sha) == 64, "adapter SHA256 is required")


@dataclass(frozen=True)
class DepthQualitySummary:
    valid_coverage: float
    formula_mae_m: float
    formula_p95_m: float
    lr_consistency_mae_px: float | None
    registration_p90_px: float | None
    temporal_static_std_m: float | None
    temporal_static_std_p95_m: float | None
    temporal_frame_median_std_m: float | None
    boundary_uncertain_ratio: float
    low_texture_ratio: float
    reflective_ratio: float
    motion_blur_ratio: float
    depth_confidence_present: bool = False
    external_metric_accuracy: str = "UNKNOWN"


def evaluate_depth_quality(
    *,
    disparity_px: np.ndarray,
    depth_m: np.ndarray,
    valid: np.ndarray,
    focal_px: float,
    baseline_m: float,
    lr_residual_px: np.ndarray | None = None,
    registration_residual_px: np.ndarray | None = None,
    static_region_mask: np.ndarray | None = None,
    boundary_mask: np.ndarray | None = None,
    low_texture_mask: np.ndarray | None = None,
    reflective_mask: np.ndarray | None = None,
    motion_blur_mask: np.ndarray | None = None,
) -> DepthQualitySummary:
    """计算 Stereo 内部 QA；不估计或伪造模型 confidence。"""

    disparity = np.asarray(disparity_px, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float64)
    valid_mask = np.asarray(valid, dtype=np.bool_)
    _require(disparity.shape == depth.shape == valid_mask.shape, "depth arrays must share shape")
    _require(disparity.ndim >= 2, "depth arrays must include image dimensions")
    _require(np.isfinite(focal_px) and focal_px > 0, "focal_px must be positive")
    _require(np.isfinite(baseline_m) and baseline_m > 0, "baseline_m must be positive")
    valid_mask &= np.isfinite(disparity) & (disparity > 0) & np.isfinite(depth) & (depth > 0)
    _require(bool(valid_mask.any()), "no valid depth samples")
    recomputed = focal_px * baseline_m / disparity[valid_mask]
    residual = np.abs(recomputed - depth[valid_mask])

    def optional_abs_mean(values: np.ndarray | None, *, depth_aligned: bool = False) -> float | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float64)
        if depth_aligned:
            _require(array.shape == depth.shape, "LR residual shape mismatch")
            finite = np.isfinite(array) & valid_mask
        else:
            finite = np.isfinite(array)
        return float(np.mean(np.abs(array[finite]))) if finite.any() else None

    registration_p90 = None
    if registration_residual_px is not None:
        registration = np.asarray(registration_residual_px, dtype=np.float64)
        finite = np.isfinite(registration)
        if finite.any():
            registration_p90 = float(np.percentile(np.abs(registration[finite]), 90))

    temporal_std = None
    temporal_std_p95 = None
    temporal_frame_median_std = None
    if static_region_mask is not None:
        static = np.asarray(static_region_mask, dtype=np.bool_)
        if depth.ndim >= 3 and static.shape == depth.shape[-2:]:
            static = np.broadcast_to(static, depth.shape)
        _require(static.shape == depth.shape, "static_region_mask shape mismatch")
        selected = valid_mask & static
        # 时间漂移必须沿 T 计算，不能把倾斜平面/空间深度差算成抖动。
        if depth.ndim >= 3 and selected.any():
            selected_depth = np.where(selected, depth, np.nan)
            flat_depth = selected_depth.reshape(depth.shape[0], -1)
            flat_selected = selected.reshape(depth.shape[0], -1)
            usable_pixels = np.sum(flat_selected, axis=0) >= 2
            usable_std = (
                np.nanstd(flat_depth[:, usable_pixels], axis=0)
                if np.any(usable_pixels)
                else np.asarray([], dtype=np.float64)
            )
            usable_std = usable_std[np.isfinite(usable_std)]
            if usable_std.size:
                temporal_std = float(np.percentile(usable_std, 50))
                temporal_std_p95 = float(np.percentile(usable_std, 95))
            per_frame_median = np.full(depth.shape[0], np.nan, dtype=np.float64)
            for frame_index in range(depth.shape[0]):
                frame_values = flat_depth[frame_index, flat_selected[frame_index]]
                if frame_values.size:
                    per_frame_median[frame_index] = float(np.median(frame_values))
            finite_frames = np.isfinite(per_frame_median)
            if np.count_nonzero(finite_frames) >= 2:
                temporal_frame_median_std = float(np.std(per_frame_median[finite_frames]))

    def ratio(mask: np.ndarray | None) -> float:
        if mask is None:
            return 0.0
        array = np.asarray(mask, dtype=np.bool_)
        _require(array.shape == depth.shape, "quality mask shape mismatch")
        return float(np.count_nonzero(array & valid_mask) / np.count_nonzero(valid_mask))

    return DepthQualitySummary(
        valid_coverage=float(np.count_nonzero(valid_mask) / valid_mask.size),
        formula_mae_m=float(np.mean(residual)),
        formula_p95_m=float(np.percentile(residual, 95)),
        lr_consistency_mae_px=optional_abs_mean(lr_residual_px, depth_aligned=True),
        registration_p90_px=registration_p90,
        temporal_static_std_m=temporal_std,
        temporal_static_std_p95_m=temporal_std_p95,
        temporal_frame_median_std_m=temporal_frame_median_std,
        boundary_uncertain_ratio=ratio(boundary_mask),
        low_texture_ratio=ratio(low_texture_mask),
        reflective_ratio=ratio(reflective_mask),
        motion_blur_ratio=ratio(motion_blur_mask),
    )


@dataclass(frozen=True)
class WristFusionResult:
    fused_wrist_m: np.ndarray
    correction_m: np.ndarray
    accepted_stereo: np.ndarray
    source_flags: tuple[str, ...]


def fuse_controller_wrist_depth(
    controller_wrist_m: np.ndarray,
    stereo_surface_z_m: np.ndarray,
    stereo_valid: np.ndarray,
    *,
    window: int = 15,
    innovation_limit_m: float = 0.030,
    stereo_weight: float = 0.15,
    ema_alpha: float = 0.15,
    correction_cap_m: float = 0.005,
) -> WristFusionResult:
    """以 Controller 为主锚点，对 optical-Z 做因果、有界修正。"""

    controller = np.asarray(controller_wrist_m, dtype=np.float64)
    stereo_z = np.asarray(stereo_surface_z_m, dtype=np.float64)
    stereo_ok = np.asarray(stereo_valid, dtype=np.bool_)
    _require(controller.ndim == 2 and controller.shape[1] == 3, "controller wrist must be [T,3]")
    _require(stereo_z.shape == stereo_ok.shape == (controller.shape[0],), "stereo arrays must be [T]")
    _require(bool(np.isfinite(controller).all()), "controller wrist must be finite")
    _require(window > 0, "window must be positive")
    _require(0 < stereo_weight <= 1 and 0 < ema_alpha <= 1, "fusion weights must be in (0,1]")

    fused = controller.copy()
    corrections = np.zeros(controller.shape[0], dtype=np.float64)
    accepted = np.zeros(controller.shape[0], dtype=np.bool_)
    flags: list[str] = []
    history: list[float] = []
    previous_correction = 0.0
    for index in range(controller.shape[0]):
        if not stereo_ok[index] or not np.isfinite(stereo_z[index]):
            flags.append("CONTROLLER_ONLY_STEREO_INVALID")
            continue
        innovation = float(stereo_z[index] - controller[index, 2])
        if abs(innovation) > innovation_limit_m:
            flags.append("CONTROLLER_ONLY_INNOVATION_REJECTED")
            continue
        history.append(innovation)
        robust = float(np.median(history[-window:]))
        target = stereo_weight * robust
        current = (1.0 - ema_alpha) * previous_correction + ema_alpha * target
        current = float(np.clip(current, -correction_cap_m, correction_cap_m))
        previous_correction = current
        corrections[index] = current
        fused[index, 2] += current
        accepted[index] = True
        flags.append("CONTROLLER_PLUS_STEREO_OPTICAL_Z_BOUNDED")
    return WristFusionResult(fused, corrections, accepted, tuple(flags))


ROLE_BASELINES = {"SAM3.1", "RENDER_PART_ID"}


def validate_mask_contract(record: Mapping[str, Any]) -> None:
    """验证 Role/Object Mask 的算法身份和新传感器 H4 状态语义。"""

    _require(record.get("schema_version") == "MASK_ROLE_OBJECT_R3", "wrong mask schema")
    kind = record.get("mask_kind")
    _require(kind in {"ROLE", "OBJECT"}, "mask_kind must be ROLE or OBJECT")
    baseline = record.get("algorithm_baseline")
    _require(baseline in ROLE_BASELINES, "SAM3.1 is the visual Mask baseline")
    if baseline == "RENDER_PART_ID":
        _require(record.get("input_domain") == "ROBOT_RENDER", "part IDs only belong to Robot render")
    elif record.get("input_domain") == "ROBOT_RENDER":
        raise PipelineContractError("Robot links must use renderer part-ID, not SAM3.1")

    if record.get("scope") == "SENSOR_H4_FORMAL":
        expected = {
            "execution_status": "BLOCKED_RESOURCE",
            "qa_status": "NOT_EVALUATED",
            "policy_status": "POLICY_DEFERRED",
            "pixel_mask_authority": False,
        }
        _require(all(record.get(key) == value for key, value in expected.items()), "formal H4 status mismatch")
    if record.get("scope") == "SENSOR_H4_DEVELOPMENT_CANARY":
        _require(record.get("authority_promoted") is False, "development canary cannot promote H4")

    instances = record.get("instances")
    _require(isinstance(instances, list) and instances, "mask instances are required")
    identifiers = [item.get("instance_id") for item in instances if isinstance(item, Mapping)]
    _require(len(identifiers) == len(instances) == len(set(identifiers)), "instance IDs must be unique")
    if kind == "OBJECT" and record.get("task") == "chips":
        _require(len(instances) == 3, "Chips requires exactly three independent task-object instances")
        _require(not any("union" in str(item).lower() for item in identifiers), "Chips union is forbidden")


def audit_mask_visibility(
    masks_by_instance: Mapping[str, np.ndarray], visible_by_instance: Mapping[str, Sequence[bool]]
) -> dict[str, int]:
    """离屏必须为空；该检查不把稳定性等同于人工准确率。"""

    _require(set(masks_by_instance) == set(visible_by_instance), "mask/visibility identities differ")
    offscreen_nonempty = 0
    total_frames = None
    for instance_id, raw_mask in masks_by_instance.items():
        mask = np.asarray(raw_mask, dtype=np.bool_)
        visible = np.asarray(visible_by_instance[instance_id], dtype=np.bool_)
        _require(mask.ndim == 3 and visible.shape == (mask.shape[0],), "mask must be [T,H,W]")
        total_frames = mask.shape[0] if total_frames is None else total_frames
        _require(mask.shape[0] == total_frames, "all instance masks must share T")
        offscreen_nonempty += int(np.count_nonzero(mask[~visible].reshape((-1, mask.shape[1] * mask.shape[2])).any(axis=1)))
    _require(offscreen_nonempty == 0, "offscreen masks must be empty; frozen stale masks are forbidden")
    return {"instances": len(masks_by_instance), "frames": int(total_frames or 0), "offscreen_nonempty": 0}


@dataclass(frozen=True)
class CleanDomainAudit:
    remove_pixels: int
    flow_pixels: int
    write_pixels: int
    protected_overlap_pixels: int
    changed_outside_write_pixels: int


def audit_clean_domains(
    *,
    m_remove: np.ndarray,
    m_flow: np.ndarray,
    m_write: np.ndarray,
    protected_object_mask: np.ndarray,
    input_rgb: np.ndarray | None = None,
    output_rgb: np.ndarray | None = None,
) -> CleanDomainAudit:
    """强制 M_remove/M_flow/M_write 分离和保护域外 byte-exact。"""

    remove = np.asarray(m_remove, dtype=np.bool_)
    flow = np.asarray(m_flow, dtype=np.bool_)
    write = np.asarray(m_write, dtype=np.bool_)
    protected = np.asarray(protected_object_mask, dtype=np.bool_)
    _require(remove.shape == flow.shape == write.shape == protected.shape, "Clean masks must share shape")
    _require(bool(np.all(~remove | flow)), "M_remove must be contained in M_flow")
    _require(bool(np.all(~write | remove)), "M_write must be contained in M_remove")
    overlap = int(np.count_nonzero(write & protected))
    _require(overlap == 0, "M_write must not modify protected visible object pixels")
    changed_outside = 0
    if input_rgb is not None or output_rgb is not None:
        _require(input_rgb is not None and output_rgb is not None, "both input and output RGB are required")
        source = np.asarray(input_rgb)
        output = np.asarray(output_rgb)
        _require(source.shape == output.shape, "input/output RGB shape mismatch")
        _require(source.shape[: remove.ndim] == remove.shape, "RGB/mask shape mismatch")
        changed = np.any(source != output, axis=-1) if source.ndim == remove.ndim + 1 else source != output
        changed_outside = int(np.count_nonzero(changed & ~write))
        _require(changed_outside == 0, "pixels outside M_write must be byte-exact")
    return CleanDomainAudit(
        remove_pixels=int(np.count_nonzero(remove)),
        flow_pixels=int(np.count_nonzero(flow)),
        write_pixels=int(np.count_nonzero(write)),
        protected_overlap_pixels=overlap,
        changed_outside_write_pixels=changed_outside,
    )


def validate_causal_donor(target_frame_id: int, donor_frame_ids: Sequence[int]) -> None:
    """训练用 donor 只能来自当前或过去帧。"""

    _require(type(target_frame_id) is int and target_frame_id >= 0, "invalid target frame")
    _require(bool(donor_frame_ids), "at least one donor is required")
    _require(all(type(item) is int and 0 <= item <= target_frame_id for item in donor_frame_ids), "future donor is forbidden")


def validate_donor_atlas_record(record: Mapping[str, Any]) -> None:
    """验证独立 donor/atlas 记录的因果性和对象身份来源。"""

    _require(record.get("schema_version") == "CAUSAL_DONOR_ATLAS_R3", "wrong donor schema")
    kind = record.get("artifact_kind")
    _require(kind in {"BACKGROUND_DONOR", "OBJECT_ATLAS"}, "invalid artifact_kind")
    target = record.get("target_frame_id")
    sources = record.get("source_frame_ids")
    _require(type(target) is int and isinstance(sources, list), "frame IDs are required")
    if record.get("causal_training_eligible") is True:
        validate_causal_donor(target, sources)
    if kind == "OBJECT_ATLAS":
        _require(bool(record.get("object_instance_id")), "object atlas needs an instance identity")
        _require(record.get("identity_verified") is True, "object atlas identity must be verified")
        _require(record.get("pose_verified") is True, "object atlas pose must be verified")


def evaluate_clean_successor_gate(
    *,
    baseline_contact_added_removal: float,
    candidate_contact_added_removal: float,
    baseline_human_residual: float,
    candidate_human_residual: float,
    baseline_nonhuman_leakage: float,
    candidate_nonhuman_leakage: float,
    visible_object_retention: float,
) -> dict[str, bool]:
    """R3 Clean 相对门；防止靠少删人伪造接触保护改进。"""

    values = (
        baseline_contact_added_removal,
        candidate_contact_added_removal,
        baseline_human_residual,
        candidate_human_residual,
        baseline_nonhuman_leakage,
        candidate_nonhuman_leakage,
        visible_object_retention,
    )
    _require(all(np.isfinite(value) and value >= 0 for value in values), "Clean metrics must be non-negative")
    contact_pass = (
        candidate_contact_added_removal == 0
        if baseline_contact_added_removal == 0
        else candidate_contact_added_removal <= 0.5 * baseline_contact_added_removal
    )
    result = {
        "contact_band_reduction_pass": bool(contact_pass),
        "human_residual_no_regression": candidate_human_residual <= baseline_human_residual,
        "nonhuman_leakage_no_regression": candidate_nonhuman_leakage <= baseline_nonhuman_leakage,
        "visible_object_retention_pass": visible_object_retention >= 0.999,
    }
    result["passed"] = all(result.values())
    return result
