#!/usr/bin/env python3
"""Audit two bounded Clean-20/21 semantic canaries without changing Clean authority.

The audit consumes the frozen Poker245 and Chips039 R7_0 inputs together with
the existing contact-protected development videos.  It deliberately does not
run ProPainter.  Instead it answers the prerequisite questions which must be
true before a fresh model run is useful:

* M_remove, M_flow and M_write are distinct and nested as contracted;
* visible object pixels are outside M_write;
* temporal donor provenance is causal (source_frame <= target_frame);
* the published video preserves pixels outside M_write after decoding;
* hidden object appearance has a legal atlas rather than an invented texture.

The existing donor is bidirectional, so future donor pixels are rejected and
counted as UNKNOWN.  Existing lossy MP4s are evaluated as artifacts, not as a
substitute for the required lossless frame/source-map publication.
"""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import csv
import hashlib
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops.audit_clean_contact_protection_successor_v1 import dilate, mask, proposed, unions


CASES = {
    "Poker245": {
        "task": "poker",
        "session": "play_cards_0903_245",
        "manifest": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/sessions/play_cards_0903_245/expanded_role_handoff/FRAME_MANIFEST.json",
        "donor": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/real_donor_v1/play_cards_0903_245/SOURCE_MAP_MANIFEST.json",
        "clean_result": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260909_exact78_current_baseline_batch_v1/clean_expanded_role_v3_prepare_v1/propainter_v1/play_cards_0903_245/RESULT.json",
        "candidate": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_clean_contact_fullsession_candidate_v1/Poker245/Poker245_Clean接触保护候选_MASTER.mp4",
        "atlas_required": True,
    },
    "Chips039": {
        "task": "chips",
        "session": "get_potato_chips_0902_039",
        "manifest": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/sessions/get_potato_chips_0902_039/expanded_role_handoff/FRAME_MANIFEST.json",
        "donor": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/real_donor_v1/get_potato_chips_0902_039/SOURCE_MAP_MANIFEST.json",
        "clean_result": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260913_exact78_v3_wave_clean_v1/propainter_v1/get_potato_chips_0902_039/RESULT.json",
        "candidate": ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260915_clean_contact_fullsession_candidate_v1/Chips039/Chips039_Clean接触保护候选_MASTER.mp4",
        "atlas_required": False,
    },
}

SIX_NAMES = (
    "RESULT.json",
    "ARTIFACT_MANIFEST.json",
    "METRICS.json",
    "RUN_RECEIPT.json",
    "DECISION.md",
    "NEXT_ACTION.json",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    value = path.resolve(strict=True)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": sha256(value)}


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def labels(image: np.ndarray, rows: list[tuple[int, int, str, tuple[int, int, int], int]]) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    drawer = ImageDraw.Draw(pil)
    cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
    for x, y, value, bgr, size in rows:
        cache.setdefault(size, font(size))
        drawer.text((x, y), value, font=cache[size], fill=(bgr[2], bgr[1], bgr[0]))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def edge(value: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(value.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return value & ~eroded


def tint(image: np.ndarray, value: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> None:
    if np.any(value):
        image[value] = ((1.0 - alpha) * image[value] + alpha * np.asarray(color)).astype(np.uint8)


def video(path: Path, frame_count: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    observed = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if observed != frame_count:
        raise RuntimeError(f"video frame mismatch {path}: {observed} != {frame_count}")
    return capture


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def percentile_from_histogram(histogram: np.ndarray, quantile: float) -> int | None:
    count = int(histogram.sum())
    if count == 0:
        return None
    target = max(1, int(np.ceil(count * quantile)))
    return int(np.searchsorted(np.cumsum(histogram), target))


def run_case(case: str, spec: dict[str, Any], output_root: Path) -> dict[str, Any]:
    case_root = output_root / case
    case_root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(spec["manifest"]).read_text(encoding="utf-8"))
    donor = json.loads(Path(spec["donor"]).read_text(encoding="utf-8"))
    clean_result = json.loads(Path(spec["clean_result"]).read_text(encoding="utf-8"))
    rows = manifest["frames"]
    donor_rows = donor["frames"]
    if len(rows) != len(donor_rows):
        raise RuntimeError(f"{case}: manifest/donor frame mismatch")
    clean_path = Path(clean_result["artifacts"]["clean_synthetic_master"]["path"])
    candidate_path = Path(spec["candidate"])
    clean_capture = video(clean_path, len(rows))
    candidate_capture = video(candidate_path, len(rows))

    first = cv2.imread(rows[0]["source_rgb"]["path"])
    if first is None:
        raise RuntimeError(f"{case}: cannot decode first Raw frame")
    height, width = first.shape[:2]
    review_path = case_root / f"{case}_Clean20_21语义前置审计_全片.mp4"
    writer = cv2.VideoWriter(str(review_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(f"{case}: cannot open review writer")

    metric_rows: list[dict[str, Any]] = []
    total: dict[str, int] = {
        "m_remove": 0,
        "m_write": 0,
        "m_flow": 0,
        "old_write": 0,
        "contact_old_extra": 0,
        "contact_new_extra": 0,
        "visible_object": 0,
        "visible_object_in_write": 0,
        "temporal_donor_in_write": 0,
        "causal_temporal_donor_in_write": 0,
        "future_temporal_donor_in_write": 0,
        "causal_unknown_in_write": 0,
        "decoded_changed_outside_write": 0,
        "decoded_pixels_outside_write": 0,
        "decoded_visible_object_changed": 0,
        "raw_like_human_pixels": 0,
    }
    object_frames = 0
    selected_scores: list[tuple[int, int]] = []
    selected_images: dict[int, np.ndarray] = {}
    outside_delta_hist = np.zeros(256, np.int64)
    object_delta_hist = np.zeros(256, np.int64)

    try:
        for frame_id, (row, donor_row) in enumerate(zip(rows, donor_rows)):
            raw = cv2.imread(row["source_rgb"]["path"])
            ok_clean, current_clean = clean_capture.read()
            ok_candidate, candidate = candidate_capture.read()
            if raw is None or not ok_clean or not ok_candidate:
                raise RuntimeError(f"{case}: decode failed at frame {frame_id}")
            human, tracker, obj = unions(row)
            m_remove = (human | tracker) & ~obj
            m_write = proposed(human, tracker, obj)
            m_flow = dilate(m_write, 16)
            old_write = mask(row["clean_removal_object_protected"])
            object_frames += int(np.any(obj))
            contact_band = dilate(obj, 20) if np.any(obj) else np.zeros_like(obj)

            with np.load(donor_row["pixel_source_map"]["path"]) as source_map:
                source_kind = np.asarray(source_map["source_kind"])
                source_frame = np.asarray(source_map["source_frame"])
            temporal = m_write & (source_kind == 1)
            causal_temporal = temporal & (source_frame <= frame_id)
            future_temporal = temporal & (source_frame > frame_id)
            protected = m_write & (source_kind == 2)
            causal_supported = causal_temporal | protected
            causal_unknown = m_write & ~causal_supported

            decoded_delta = np.max(np.abs(candidate.astype(np.int16) - raw.astype(np.int16)), axis=2)
            outside_write = ~m_write
            decoded_changed_outside = outside_write & (decoded_delta != 0)
            decoded_visible_changed = obj & (decoded_delta != 0)
            raw_like_human = m_remove & (decoded_delta <= 8)
            outside_delta_hist += np.bincount(decoded_delta[outside_write], minlength=256)
            if np.any(obj):
                object_delta_hist += np.bincount(decoded_delta[obj], minlength=256)

            old_extra = (old_write & contact_band) & ~m_remove
            new_extra = (m_write & contact_band) & ~m_remove
            values = {
                "frame": frame_id,
                "object_observed": int(np.any(obj)),
                "m_remove_pixels": int(m_remove.sum()),
                "m_write_pixels": int(m_write.sum()),
                "m_flow_pixels": int(m_flow.sum()),
                "m_remove_outside_m_write": int((m_remove & ~m_write).sum()),
                "m_write_outside_m_flow": int((m_write & ~m_flow).sum()),
                "visible_object_pixels": int(obj.sum()),
                "visible_object_in_m_write": int((obj & m_write).sum()),
                "contact_old_extra_pixels": int(old_extra.sum()),
                "contact_new_extra_pixels": int(new_extra.sum()),
                "temporal_donor_in_m_write": int(temporal.sum()),
                "causal_temporal_donor_in_m_write": int(causal_temporal.sum()),
                "future_temporal_donor_in_m_write": int(future_temporal.sum()),
                "causal_unknown_in_m_write": int(causal_unknown.sum()),
                "causal_unknown_ratio": ratio(int(causal_unknown.sum()), int(m_write.sum())),
                "decoded_changed_outside_m_write": int(decoded_changed_outside.sum()),
                "decoded_visible_object_changed": int(decoded_visible_changed.sum()),
                "raw_like_human_pixels_proxy": int(raw_like_human.sum()),
            }
            metric_rows.append(values)
            for key, value in (
                ("m_remove", m_remove.sum()),
                ("m_write", m_write.sum()),
                ("m_flow", m_flow.sum()),
                ("old_write", old_write.sum()),
                ("contact_old_extra", old_extra.sum()),
                ("contact_new_extra", new_extra.sum()),
                ("visible_object", obj.sum()),
                ("visible_object_in_write", (obj & m_write).sum()),
                ("temporal_donor_in_write", temporal.sum()),
                ("causal_temporal_donor_in_write", causal_temporal.sum()),
                ("future_temporal_donor_in_write", future_temporal.sum()),
                ("causal_unknown_in_write", causal_unknown.sum()),
                ("decoded_changed_outside_write", decoded_changed_outside.sum()),
                ("decoded_pixels_outside_write", outside_write.sum()),
                ("decoded_visible_object_changed", decoded_visible_changed.sum()),
                ("raw_like_human_pixels", raw_like_human.sum()),
            ):
                total[key] += int(value)

            domain = raw.copy()
            tint(domain, m_flow & ~m_write, (255, 128, 0), 0.30)
            tint(domain, m_write, (0, 255, 255), 0.42)
            tint(domain, m_remove, (255, 0, 0), 0.44)
            tint(domain, obj, (0, 255, 0), 0.65)
            tint(domain, future_temporal, (255, 0, 255), 0.85)
            tint(domain, causal_unknown, (0, 0, 255), 0.65)
            small = [cv2.resize(x, (640, 360), interpolation=cv2.INTER_AREA) for x in (raw, current_clean, candidate, domain)]
            canvas = np.vstack((np.hstack((small[0], small[1])), np.hstack((small[2], small[3]))))
            future_ratio = ratio(int(future_temporal.sum()), int(temporal.sum())) or 0.0
            unknown_ratio = values["causal_unknown_ratio"] or 0.0
            canvas = labels(
                canvas,
                [
                    (8, 8, f"Raw｜{case}｜帧 {frame_id:04d}", (255, 255, 255), 20),
                    (648, 8, "现行 R7_0 Clean（只作对照）", (255, 255, 255), 20),
                    (8, 368, "既有接触保护候选（非新模型输出）", (255, 255, 255), 19),
                    (648, 368, "蓝=remove 黄=write 青=flow 绿=物体", (255, 255, 255), 18),
                    (648, 393, "紫=未来donor(泄漏) 红=因果UNKNOWN", (255, 255, 255), 18),
                    (8, 682, f"未来donor {future_ratio:.1%}｜因果UNKNOWN {unknown_ratio:.1%}｜物体Mask {'有效' if np.any(obj) else '无效'}", (0, 220, 255), 18),
                ],
            )
            writer.write(canvas)
            score = int(old_extra.sum()) + 2 * int(future_temporal.sum()) + int(causal_unknown.sum())
            selected_scores.append((score, frame_id))
            if frame_id % max(1, len(rows) // 10) == 0 or frame_id in (0, len(rows) - 1):
                selected_images[frame_id] = canvas.copy()
    finally:
        clean_capture.release()
        candidate_capture.release()
        writer.release()

    # Include the hardest well-separated frames in addition to uniform samples.
    picked = set(selected_images)
    for _, frame_id in sorted(selected_scores, reverse=True):
        if all(abs(frame_id - existing) >= 5 for existing in picked):
            picked.add(frame_id)
        if len(picked) >= 12:
            break
    # Reuse frames captured uniformly; the full video contains every hard frame.
    chosen = sorted(selected_images)[:12]
    sheet_rows = [cv2.resize(selected_images[idx], (640, 360), interpolation=cv2.INTER_AREA) for idx in chosen]
    while len(sheet_rows) < 12:
        sheet_rows.append(np.zeros((360, 640, 3), np.uint8))
    sheet = np.vstack([np.hstack(sheet_rows[i : i + 3]) for i in range(0, 12, 3)])
    sheet_path = case_root / f"{case}_Clean20_21语义前置审计_12帧总览.jpg"
    if not cv2.imwrite(str(sheet_path), sheet):
        raise RuntimeError(f"{case}: cannot write contact sheet")

    csv_path = case_root / "FRAME_METRICS.csv"
    with csv_path.open("x", newline="", encoding="utf-8-sig") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(metric_rows)
        handle.flush()
        os.fsync(handle.fileno())

    gates = {
        "domain_remove_subset_write": all(row["m_remove_outside_m_write"] == 0 for row in metric_rows),
        "domain_write_subset_flow": all(row["m_write_outside_m_flow"] == 0 for row in metric_rows),
        "visible_object_excluded_from_write": total["visible_object_in_write"] == 0,
        "contact_band_extra_deletion_reduced_50pct": (
            total["contact_old_extra"] > 0
            and total["contact_new_extra"] <= 0.5 * total["contact_old_extra"]
        ),
        "causal_donor_only": total["future_temporal_donor_in_write"] == 0,
        "semantic_support_surface_rejection_evidenced": False,
        "decoded_m_write_outside_byte_exact": total["decoded_changed_outside_write"] == 0,
        "fresh_propainter_successor": False,
        "lossless_frames_and_source_map_published": False,
        "poker_verified_object_atlas": False if spec["atlas_required"] else None,
    }
    metrics = {
        "schema_version": "clean-20-21-semantic-canary-metrics-r3-v1",
        "case": case,
        "session": spec["session"],
        "task": spec["task"],
        "frame_count": len(rows),
        "fps": 30,
        "object_observed_frame_coverage": ratio(object_frames, len(rows)),
        "domain_pixels": {
            "m_remove": total["m_remove"],
            "m_write": total["m_write"],
            "m_flow": total["m_flow"],
            "old_write": total["old_write"],
        },
        "contact_band": {
            "old_extra_deletion_pixels": total["contact_old_extra"],
            "new_extra_deletion_pixels": total["contact_new_extra"],
            "extra_deletion_reduction_ratio": (
                1.0 - total["contact_new_extra"] / total["contact_old_extra"]
                if total["contact_old_extra"]
                else None
            ),
        },
        "visible_object": {
            "pixels": total["visible_object"],
            "pixels_in_m_write": total["visible_object_in_write"],
            "decoded_changed_pixels_in_existing_lossy_candidate": total["decoded_visible_object_changed"],
            "decoded_retention_exact_ratio": (
                1.0 - total["decoded_visible_object_changed"] / total["visible_object"]
                if total["visible_object"]
                else None
            ),
            "decoded_linf_mean": (
                float(np.dot(np.arange(256), object_delta_hist) / object_delta_hist.sum())
                if object_delta_hist.sum()
                else None
            ),
            "decoded_linf_p95": percentile_from_histogram(object_delta_hist, 0.95),
            "interpretation": "Exact-byte ratio is expected to be low after lossy MP4 encoding; Linf summaries describe codec-scale visual deviation, not semantic object accuracy.",
        },
        "donor_causality": {
            "declared_method": donor.get("method"),
            "temporal_pixels_in_m_write": total["temporal_donor_in_write"],
            "causal_temporal_pixels_in_m_write": total["causal_temporal_donor_in_write"],
            "future_temporal_pixels_in_m_write": total["future_temporal_donor_in_write"],
            "future_fraction_of_temporal": ratio(
                total["future_temporal_donor_in_write"], total["temporal_donor_in_write"]
            ),
            "causal_unknown_pixels_in_m_write": total["causal_unknown_in_write"],
            "causal_unknown_ratio_in_m_write": ratio(total["causal_unknown_in_write"], total["m_write"]),
        },
        "artifact_pixel_gate": {
            "decoded_changed_pixels_outside_m_write": total["decoded_changed_outside_write"],
            "decoded_pixels_outside_m_write": total["decoded_pixels_outside_write"],
            "decoded_byte_exact_ratio_outside_m_write": (
                1.0 - total["decoded_changed_outside_write"] / total["decoded_pixels_outside_write"]
            ),
            "decoded_linf_mean_outside_m_write": float(
                np.dot(np.arange(256), outside_delta_hist) / outside_delta_hist.sum()
            ),
            "decoded_linf_p95_outside_m_write": percentile_from_histogram(outside_delta_hist, 0.95),
            "reason": "The existing candidate is a lossy MP4 and has no published lossless frame/source-map closure.",
        },
        "hand_residual_proxy": {
            "raw_like_pixels_within_m_remove_linf_le_8": total["raw_like_human_pixels"],
            "raw_like_ratio_within_m_remove": ratio(total["raw_like_human_pixels"], total["m_remove"]),
            "claim_limit": "Image-difference proxy only; there is no independent clean-background or human-residual truth.",
        },
        "gates": gates,
        "claim_limit": "CPU prerequisite and artifact audit only; no semantic Clean, hidden-object, contact, Robotized RGB or physical authority.",
    }
    write_json(case_root / "METRICS.json", metrics)

    blockers = [name for name, passed in gates.items() if passed is False]
    status = "BLOCKED_PREREQ"
    decision = {
        "schema_version": "clean-20-21-semantic-canary-decision-r3-v1",
        "status": status,
        "clean_20": "BLOCKED_PREREQ_CAUSAL_AND_SEMANTIC_DONOR",
        "clean_21": "BLOCKED_PREREQ_VERIFIED_POKER_ATLAS" if spec["atlas_required"] else "NOT_APPLICABLE_CHIPS",
        "blocked_gates": blockers,
        "gpu_execution": "NOT_ATTEMPTED_FAIL_CLOSED_BEFORE_GPU",
        "authority_promoted": False,
        "r7_0_modified": False,
    }
    result = {
        "schema_version": "clean-20-21-semantic-canary-result-r3-v1",
        "task_id": f"clean_20_21_{spec['session']}_r3",
        "status": status,
        "artifact_revision": "R7_CLEAN_SEMANTIC_20_21_CANARY_1",
        "validity": "VALID_FOR_PINNED_REVISION",
        "session": spec["session"],
        "task": spec["task"],
        "decision": decision,
        "metrics": metrics,
        "inputs": {
            "frame_manifest": ref(Path(spec["manifest"])),
            "bidirectional_donor_manifest": ref(Path(spec["donor"])),
            "r7_0_clean_result": ref(Path(spec["clean_result"])),
            "existing_contact_candidate": ref(candidate_path),
        },
        "visualizations": {"full_review": ref(review_path), "contact_sheet": ref(sheet_path)},
        "producer_code": ref(Path(__file__)),
        "claim_limit": metrics["claim_limit"],
    }
    signature_payload = {
        "artifact_revision": result["artifact_revision"],
        "inputs": result["inputs"],
        "producer_code": result["producer_code"],
        "method": "CPU_SEMANTIC_PREREQUISITE_AUDIT_NO_MODEL_EXECUTION",
    }
    result["producer_signature"] = hashlib.sha256(
        json.dumps(signature_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    write_json(case_root / "RESULT.json", result)
    next_action = {
        "schema_version": "clean-20-21-next-action-r3-v1",
        "status": status,
        "next": [
            "Build a causal donor map by rejecting every temporal source_frame > target_frame.",
            "Add task/support-surface semantic labels and reject plate/table/other-instance donors.",
            "Publish lossless successor frames plus a per-pixel source map before testing byte-exact gates.",
            "For Poker only, build and independently pose/identity-verify a real visible-surface atlas.",
            "Only after these CPU gates pass, acquire the central GPU lease and run fresh ProPainter on the remaining UNKNOWN holes.",
        ],
        "do_not": ["Do not reuse the bidirectional candidate as causal training input.", "Do not modify or supersede R7_0 authority."],
    }
    write_json(case_root / "NEXT_ACTION.json", next_action)
    receipt = {
        "schema_version": "clean-20-21-semantic-canary-run-receipt-r3-v1",
        "task_id": result["task_id"],
        "attempt_id": output_root.name,
        "created_at": now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "status": status,
        "cpu_preflight": "COMPLETED",
        "gpu_lease": "NOT_ACQUIRED",
        "gpu_reason": "Prerequisite audit failed before GPU execution; no GPU work was justified.",
        "authority_promoted": False,
    }
    write_json(case_root / "RUN_RECEIPT.json", receipt)
    decision_md = f"""# {case} Clean-20/21 有界语义 canary 决定

状态：`{status}`。

- 三个域已独立计算：`M_remove` 是原始人体/Tracker；`M_write` 是允许修改的接触保护写域；`M_flow` 仅是光流上下文。
- 当前可见物体与 `M_write` 的重叠为 {total['visible_object_in_write']} 像素。
- 旧 donor 含 {total['future_temporal_donor_in_write']} 个写域内未来帧像素，不能作为因果训练输入。
- 因果过滤后写域 UNKNOWN 比例为 {metrics['donor_causality']['causal_unknown_ratio_in_m_write']:.2%}。
- 现有候选只有有损 MP4，没有 lossless frame/source-map，因此不能通过 `M_write` 外 byte-exact 发布门。
- {'Poker 的已验证真实牌面 Atlas 尚不存在，Clean-21 不运行。' if spec['atlas_required'] else 'Chips 不适用 Poker Atlas；Clean-20 仍被因果 donor 与语义 donor 门阻塞。'}

本结果不修改 R7_0，不晋升 Clean、Contact、Occlusion 或 Robotized RGB authority。
"""
    write_text(case_root / "DECISION.md", decision_md)

    output_refs = {
        "review_video": ref(review_path),
        "contact_sheet": ref(sheet_path),
        "frame_metrics_csv": ref(csv_path),
    }
    output_refs.update({name: ref(case_root / name) for name in SIX_NAMES if (case_root / name).is_file()})
    artifact_manifest = {
        "schema_version": "clean-20-21-semantic-canary-artifact-manifest-r3-v1",
        "task_id": result["task_id"],
        "status": status,
        "artifacts": output_refs,
        "authority_promoted": False,
    }
    write_json(case_root / "ARTIFACT_MANIFEST.json", artifact_manifest)
    return {"case": case, "status": status, "case_result": ref(case_root / "RESULT.json"), "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    reports = [run_case(case, spec, output_root) for case, spec in CASES.items()]
    summary = {
        "schema_version": "clean-20-21-semantic-canary-batch-result-r3-v1",
        "task_id": "clean_20_21_semantic_canary_r3",
        "status": "BLOCKED_PREREQ",
        "artifact_revision": "R7_CLEAN_SEMANTIC_20_21_CANARY_1",
        "reports": reports,
        "gpu_execution": "NOT_ATTEMPTED_FAIL_CLOSED_BEFORE_GPU",
        "r7_0_modified": False,
        "authority_promoted": False,
        "claim_limit": "Two-session CPU semantic prerequisite audit; not a fresh Clean-20/21 model result or authority.",
    }
    write_json(output_root / "RESULT.json", summary)
    write_json(output_root / "METRICS.json", {"schema_version": "clean-20-21-semantic-canary-batch-metrics-r3-v1", "reports": [{"case": row["case"], "status": row["status"], "metrics": row["metrics"]} for row in reports]})
    write_json(output_root / "NEXT_ACTION.json", {"schema_version": "clean-20-21-next-action-r3-v1", "status": "BLOCKED_PREREQ", "next": "Close causal donor, semantic donor rejection, lossless source-map and Poker atlas prerequisites before any GPU attempt."})
    write_json(output_root / "RUN_RECEIPT.json", {"schema_version": "clean-20-21-semantic-canary-run-receipt-r3-v1", "task_id": summary["task_id"], "created_at": now(), "status": summary["status"], "host": socket.gethostname(), "pid": os.getpid(), "gpu_lease": "NOT_ACQUIRED", "authority_promoted": False})
    write_text(output_root / "DECISION.md", "# Clean-20/21 两会话决定\n\n两条均为 `BLOCKED_PREREQ`。旧双向 donor 含未来信息，语义 donor rejection、lossless source-map 与 Poker Atlas 尚未闭合；因此未占用 GPU、未运行 ProPainter、未修改 R7_0。\n")
    manifest = {"schema_version": "clean-20-21-semantic-canary-artifact-manifest-r3-v1", "task_id": summary["task_id"], "status": summary["status"], "case_results": [row["case_result"] for row in reports], "authority_promoted": False}
    # The aggregate manifest is written last and deliberately does not self-hash.
    for name in SIX_NAMES:
        candidate = output_root / name
        if candidate.is_file() and name != "ARTIFACT_MANIFEST.json":
            manifest.setdefault("artifacts", {})[name] = ref(candidate)
    write_json(output_root / "ARTIFACT_MANIFEST.json", manifest)
    print(json.dumps({"result": ref(output_root / "RESULT.json"), "reports": [{"case": r["case"], "status": r["status"]} for r in reports]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
