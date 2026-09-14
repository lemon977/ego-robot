#!/usr/bin/env python3
"""Build an explicitly synthetic Clean baseline with ProPainter.

The runner consumes a current Mask frame manifest and a previous real-donor
Clean source-map manifest.  Real temporal/stereo donor pixels are kept first;
only deletion pixels which have no real donor are sent to ProPainter.  The
published source map distinguishes target raw, real donor, protected object,
and synthetic pixels for every output pixel.

This is deliberately an adapter around an unmodified, commit-pinned upstream
ProPainter checkout.  No generated pixel is described as observed background.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SOURCE_TARGET_RAW = np.uint8(0)
SOURCE_REAL_DONOR = np.uint8(1)
SOURCE_PROTECTED_OBJECT_RAW = np.uint8(2)
SOURCE_SYNTHETIC_PROPAINTER = np.uint8(3)

SOURCE_KIND_LABELS = {
    "0": "TARGET_RAW_UNCHANGED",
    "1": "REAL_DONOR_SAME_SESSION_TEMPORAL_OR_STEREO",
    "2": "PROTECTED_OBJECT_TARGET_RAW",
    "3": "SYNTHETIC_PROPAINTER_NOT_OBSERVED_BACKGROUND",
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(Image.open(path).convert("L")) > 0
    if value.shape != shape:
        raise RuntimeError(f"mask shape {value.shape} does not match RGB {shape}: {path}")
    return value


def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def label_panel(rgb: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, image.width, 70), fill=(0, 0, 0, 180))
    draw.text((16, 8), title, font=find_font(25), fill=(255, 255, 255, 255))
    if subtitle:
        draw.text((16, 39), subtitle, font=find_font(17), fill=(235, 235, 235, 255))
    return np.asarray(image)


def fit_panel(rgb: np.ndarray, width: int = 800, height: int = 450) -> np.ndarray:
    resized = cv2.resize(rgb, (600, 450), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    canvas[:, 100:700] = resized
    return canvas


def overlay_mask(raw: np.ndarray, removal: np.ndarray, protected: np.ndarray) -> np.ndarray:
    value = raw.astype(np.float32)
    red = np.zeros_like(value)
    red[..., 0] = 255
    cyan = np.zeros_like(value)
    cyan[..., 1:] = 255
    value[removal] = 0.42 * value[removal] + 0.58 * red[removal]
    value[protected] = 0.30 * value[protected] + 0.70 * cyan[protected]
    return np.clip(value, 0, 255).astype(np.uint8)


def provenance_view(raw: np.ndarray, source_kind: np.ndarray) -> np.ndarray:
    value = (raw.astype(np.float32) * 0.28).astype(np.uint8)
    colors = {
        int(SOURCE_REAL_DONOR): np.array([40, 220, 70], np.uint8),
        int(SOURCE_PROTECTED_OBJECT_RAW): np.array([30, 220, 235], np.uint8),
        int(SOURCE_SYNTHETIC_PROPAINTER): np.array([235, 45, 220], np.uint8),
    }
    for code, color in colors.items():
        selected = source_kind == code
        value[selected] = color
    return value


def video_decode_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _frame = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def encode_png_sequence(frame_root: Path, output: Path, fps: float) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", f"{fps:.8f}", "-i", str(frame_root / "%06d.png"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "17",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    subprocess.run(command, check=True)


def physical_object_union(frame: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    protected = np.zeros(shape, dtype=bool)
    for key, value in frame.items():
        if not key.startswith("physical_object_") or not isinstance(value, dict):
            continue
        path = value.get("path")
        if isinstance(path, str):
            protected |= read_mask(Path(path), shape)
    return protected


def validate_vendor(propainter_root: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    commit = subprocess.check_output(
        ["git", "-C", str(propainter_root), "rev-parse", "HEAD"], text=True
    ).strip()
    expected_commit = "e870e79321c31b733e2031af5aa2fb1fe3ac7eec"
    if commit != expected_commit:
        raise RuntimeError(f"ProPainter commit mismatch: {commit} != {expected_commit}")
    weights = {}
    for name in ("raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth"):
        path = propainter_root / "weights" / name
        if not path.is_file() or path.stat().st_size < 1_000_000:
            raise RuntimeError(f"missing or incomplete ProPainter weight: {path}")
        weights[name] = ref(path)
    return commit, weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true", help="Prepare inputs only; alias for --prepare-only")
    args = parser.parse_args()

    spec_path = args.spec.resolve()
    spec = load_json(spec_path)
    output_root = Path(spec["output_root"]).resolve()
    if output_root.exists():
        raise RuntimeError(f"no-clobber output already exists: {output_root}")
    output_root.mkdir(parents=True)

    mask_manifest_path = Path(spec["mask_frame_manifest"]).resolve()
    donor_manifest_path = Path(spec["real_donor_source_manifest"]).resolve()
    mask_manifest = load_json(mask_manifest_path)
    donor_manifest = load_json(donor_manifest_path)
    mask_frames = mask_manifest["frames"]
    donor_frames = donor_manifest["frames"]
    if len(mask_frames) != len(donor_frames):
        raise RuntimeError("Mask and real-donor manifests have different frame counts")

    start = int(spec.get("start_frame", 0))
    stop = int(spec.get("stop_frame_exclusive", len(mask_frames)))
    if not 0 <= start < stop <= len(mask_frames):
        raise RuntimeError(f"invalid frame interval [{start}, {stop})")
    selected_mask_frames = mask_frames[start:stop]
    selected_donor_frames = donor_frames[start:stop]

    propainter_root = Path(spec["propainter_root"]).resolve()
    process_width = int(spec.get("process_width", 960))
    process_height = int(spec.get("process_height", 720))
    fps = float(spec.get("fps", 30.0))
    if process_width % 8 or process_height % 8:
        raise RuntimeError("ProPainter processing dimensions must be divisible by 8")

    input_frames_root = output_root / "propainter_input" / "frames"
    input_masks_root = output_root / "propainter_input" / "masks"
    input_frames_root.mkdir(parents=True)
    input_masks_root.mkdir(parents=True)
    prepared_root = output_root / "prepared_full_resolution"
    prepared_root.mkdir()

    prepared: list[dict[str, Any]] = []
    totals = {"removal": 0, "real_donor": 0, "synthetic": 0, "protected": 0}
    original_shape: tuple[int, int] | None = None

    for local_index, (mask_frame, donor_frame) in enumerate(
        zip(selected_mask_frames, selected_donor_frames, strict=True)
    ):
        source_frame = int(mask_frame["source_frame"])
        if source_frame != int(donor_frame["frame_id"]):
            raise RuntimeError(f"frame mismatch at local {local_index}")
        raw_path = Path(mask_frame["source_rgb"]["path"])
        raw = read_rgb(raw_path)
        if original_shape is None:
            original_shape = raw.shape[:2]
        if raw.shape[:2] != original_shape:
            raise RuntimeError("source RGB dimensions change within session")
        shape = raw.shape[:2]
        removal = read_mask(Path(mask_frame["clean_removal_object_protected"]["path"]), shape)
        protected = physical_object_union(mask_frame, shape)
        if np.any(removal & protected):
            raise RuntimeError(f"removal overlaps protected object at frame {source_frame}")

        clean_real = read_rgb(Path(donor_frame["clean_rgb"]["path"]))
        source_npz_path = Path(donor_frame["pixel_source_map"]["path"])
        with np.load(source_npz_path) as source_npz:
            legacy_kind = source_npz["source_kind"]
        if legacy_kind.shape != shape or clean_real.shape[:2] != shape:
            raise RuntimeError(f"real donor dimensions mismatch at frame {source_frame}")
        real_donor = removal & np.isin(legacy_kind, (1, 4))
        synthetic = removal & ~real_donor & ~protected

        base = raw.copy()
        base[real_donor] = clean_real[real_donor]
        base[protected] = raw[protected]

        source_kind = np.full(shape, SOURCE_TARGET_RAW, dtype=np.uint8)
        source_kind[real_donor] = SOURCE_REAL_DONOR
        source_kind[protected] = SOURCE_PROTECTED_OBJECT_RAW
        source_kind[synthetic] = SOURCE_SYNTHETIC_PROPAINTER

        local_name = f"{local_index:06d}"
        base_path = prepared_root / f"{local_name}_base.png"
        source_kind_path = prepared_root / f"{local_name}_source_kind.npz"
        removal_path = prepared_root / f"{local_name}_removal.png"
        protected_path = prepared_root / f"{local_name}_protected.png"
        Image.fromarray(base).save(base_path)
        np.savez_compressed(source_kind_path, source_kind=source_kind, source_frame=np.int32(source_frame))
        Image.fromarray(removal.astype(np.uint8) * 255).save(removal_path)
        Image.fromarray(protected.astype(np.uint8) * 255).save(protected_path)

        process_base = cv2.resize(base, (process_width, process_height), interpolation=cv2.INTER_AREA)
        process_mask = cv2.resize(
            synthetic.astype(np.uint8) * 255,
            (process_width, process_height),
            interpolation=cv2.INTER_NEAREST,
        )
        Image.fromarray(process_base).save(input_frames_root / f"{local_name}.png")
        Image.fromarray(process_mask).save(input_masks_root / f"{local_name}.png")

        counts = {
            "removal_pixels": int(removal.sum()),
            "real_donor_pixels": int(real_donor.sum()),
            "synthetic_pixels": int(synthetic.sum()),
            "protected_object_pixels": int(protected.sum()),
        }
        totals["removal"] += counts["removal_pixels"]
        totals["real_donor"] += counts["real_donor_pixels"]
        totals["synthetic"] += counts["synthetic_pixels"]
        totals["protected"] += counts["protected_object_pixels"]
        prepared.append(
            {
                "local_index": local_index,
                "source_frame": source_frame,
                "raw_path": str(raw_path),
                "base_path": str(base_path),
                "source_kind_path": str(source_kind_path),
                "removal_path": str(removal_path),
                "protected_path": str(protected_path),
                **counts,
            }
        )

    prepare_result = {
        "schema_version": "clean-synthetic-propainter-preparation-v1",
        "created_at": now(),
        "status": "PASS_PREPARED_SYNTHETIC_HOLES_NOT_YET_INPAINTED",
        "spec": ref(spec_path),
        "mask_frame_manifest": ref(mask_manifest_path),
        "real_donor_source_manifest": ref(donor_manifest_path),
        "frame_interval": [start, stop],
        "frame_count": len(prepared),
        "original_hw": list(original_shape or ()),
        "process_hw": [process_height, process_width],
        "totals": totals,
        "source_kind_labels": SOURCE_KIND_LABELS,
        "frames": prepared,
        "claim_limit": "Only REAL_DONOR pixels are observed; SYNTHETIC_PROPAINTER pixels are generated and must never be claimed as true background.",
    }
    prepare_result_path = output_root / "PREPARE_RESULT.json"
    write_json(prepare_result_path, prepare_result)
    if args.prepare_only or args.skip_gpu:
        print(json.dumps({"status": prepare_result["status"], "output": str(output_root)}, ensure_ascii=False))
        return 0

    commit, weight_refs = validate_vendor(propainter_root)
    upstream_output = output_root / "propainter_workspace"
    log_path = output_root / "PROPAINTER_RUN.log"
    command = [
        sys.executable,
        "inference_propainter.py",
        "--video", str(input_frames_root),
        "--mask", str(input_masks_root),
        "--output", str(upstream_output),
        "--width", str(process_width),
        "--height", str(process_height),
        "--mask_dilation", str(int(spec.get("mask_dilation", 4))),
        "--ref_stride", str(int(spec.get("ref_stride", 10))),
        "--neighbor_length", str(int(spec.get("neighbor_length", 10))),
        "--subvideo_length", str(int(spec.get("subvideo_length", 80))),
        "--raft_iter", str(int(spec.get("raft_iter", 20))),
        "--save_fps", str(int(round(fps))),
        "--save_frames",
        "--fp16",
    ]
    with log_path.open("wb") as log:
        completed = subprocess.run(command, cwd=propainter_root, stdout=log, stderr=subprocess.STDOUT)
    upstream_frames = upstream_output / "frames" / "frames"
    upstream_frame_count = len(list(upstream_frames.glob("*.png"))) if upstream_frames.is_dir() else 0
    # Commit e870e79 writes all requested PNGs before two optional preview MP4s.
    # New ImageIO/PyAV rejects its legacy ``quality=`` argument.  This adapter
    # uses the lossless PNGs and encodes both published videos with FFmpeg, so
    # that output-only compatibility error is safe to bypass when (and only
    # when) the complete PNG sequence exists.
    known_imageio_video_writer_only_failure = False
    if completed.returncode != 0:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        known_imageio_video_writer_only_failure = (
            upstream_frame_count == len(prepared)
            and "PyAVPlugin.write() got an unexpected keyword argument 'quality'" in log_text
        )
        if not known_imageio_video_writer_only_failure:
            raise RuntimeError(f"ProPainter failed rc={completed.returncode}; see {log_path}")
    if not upstream_frames.is_dir():
        raise RuntimeError(f"ProPainter frame output missing: {upstream_frames}")

    clean_frames_root = output_root / "clean_frames"
    source_maps_root = output_root / "pixel_sources"
    review_frames_root = output_root / "review_frames"
    clean_frames_root.mkdir()
    source_maps_root.mkdir()
    review_frames_root.mkdir()
    frame_manifest_rows: list[dict[str, Any]] = []
    changed_outside_total = 0
    changed_protected_total = 0

    assert original_shape is not None
    original_height, original_width = original_shape
    for row in prepared:
        local_index = int(row["local_index"])
        local_name = f"{local_index:06d}"
        raw = read_rgb(Path(row["raw_path"]))
        base = read_rgb(Path(row["base_path"]))
        removal = read_mask(Path(row["removal_path"]), original_shape)
        protected = read_mask(Path(row["protected_path"]), original_shape)
        with np.load(Path(row["source_kind_path"])) as source_npz:
            source_kind = source_npz["source_kind"]
        synthetic = source_kind == SOURCE_SYNTHETIC_PROPAINTER
        generated_small = read_rgb(upstream_frames / f"{local_index:04d}.png")
        generated = cv2.resize(
            generated_small, (original_width, original_height), interpolation=cv2.INTER_CUBIC
        )
        final = base.copy()
        final[synthetic] = generated[synthetic]
        final[protected] = raw[protected]
        changed = np.any(final != raw, axis=2)
        changed_outside = int((changed & ~removal).sum())
        changed_protected = int((changed & protected).sum())
        if changed_outside or changed_protected:
            raise RuntimeError(
                f"pixel protection failure frame={row['source_frame']} outside={changed_outside} protected={changed_protected}"
            )
        changed_outside_total += changed_outside
        changed_protected_total += changed_protected

        clean_path = clean_frames_root / f"{local_name}.png"
        source_map_path = source_maps_root / f"{local_name}.npz"
        Image.fromarray(final).save(clean_path)
        np.savez_compressed(
            source_map_path,
            source_kind=source_kind,
            source_frame=np.int32(row["source_frame"]),
            real_donor=(source_kind == SOURCE_REAL_DONOR),
            synthetic=synthetic,
            protected_object=(source_kind == SOURCE_PROTECTED_OBJECT_RAW),
        )

        mask_panel = overlay_mask(raw, removal, protected)
        provenance_panel = provenance_view(raw, source_kind)
        real_pixels = int((source_kind == SOURCE_REAL_DONOR).sum())
        synthetic_pixels = int(synthetic.sum())
        panels = [
            label_panel(fit_panel(raw), "原始 RGB", f"源帧 {row['source_frame']}"),
            label_panel(fit_panel(mask_panel), "Mask 删除区", "红=删除；青=任务物体保护"),
            label_panel(fit_panel(final), "CLEAN_SYNTHETIC", "真实 donor 优先，其余由 ProPainter 生成"),
            label_panel(
                fit_panel(provenance_panel),
                "逐像素来源",
                f"绿=REAL_DONOR {real_pixels}；紫=SYNTHETIC {synthetic_pixels}；青=保护物体",
            ),
        ]
        review = np.vstack((np.hstack((panels[0], panels[1])), np.hstack((panels[2], panels[3]))))
        Image.fromarray(review).save(review_frames_root / f"{local_name}.jpg", quality=91)
        frame_manifest_rows.append(
            {
                "local_index": local_index,
                "source_frame": int(row["source_frame"]),
                "clean_rgb": ref(clean_path),
                "pixel_source_map": ref(source_map_path),
                "removal_pixels": int(removal.sum()),
                "real_donor_pixels": real_pixels,
                "synthetic_pixels": synthetic_pixels,
                "protected_object_pixels": int(protected.sum()),
                "changed_outside_removal_pixels": changed_outside,
                "changed_protected_object_pixels": changed_protected,
            }
        )

    master_video = output_root / "CLEAN_SYNTHETIC_MASTER.mp4"
    review_video = output_root / "CLEAN_SYNTHETIC_中文全片复核.mp4"
    encode_png_sequence(clean_frames_root, master_video, fps)
    command_review = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", f"{fps:.8f}", "-i", str(review_frames_root / "%06d.jpg"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(review_video),
    ]
    subprocess.run(command_review, check=True)
    expected_frames = len(prepared)
    master_decoded = video_decode_count(master_video)
    review_decoded = video_decode_count(review_video)
    if master_decoded != expected_frames or review_decoded != expected_frames:
        raise RuntimeError(
            f"video decode mismatch expected={expected_frames} master={master_decoded} review={review_decoded}"
        )

    source_manifest = {
        "schema_version": "clean-synthetic-pixel-source-manifest-v1",
        "created_at": now(),
        "task": spec["task"],
        "session": spec["session"],
        "frame_interval": [start, stop],
        "frame_count": expected_frames,
        "source_kind_codes": SOURCE_KIND_LABELS,
        "frames": frame_manifest_rows,
        "claim_limit": "Code 3 is SYNTHETIC ProPainter output and is not observed background. Code 1 alone is real same-session donor.",
    }
    source_manifest_path = output_root / "SOURCE_MAP_MANIFEST.json"
    write_json(source_manifest_path, source_manifest)

    synthetic_fraction = totals["synthetic"] / max(totals["removal"], 1)
    real_fraction = totals["real_donor"] / max(totals["removal"], 1)
    hard_gates = {
        "frame_count_and_order": "PASS",
        "finite_uint8_output": "PASS",
        "changed_outside_removal_zero": "PASS" if changed_outside_total == 0 else "FAIL",
        "protected_object_byte_exact": "PASS" if changed_protected_total == 0 else "FAIL",
        "per_pixel_real_vs_synthetic_provenance": "PASS",
        "master_video_full_decode": "PASS" if master_decoded == expected_frames else "FAIL",
        "review_video_full_decode": "PASS" if review_decoded == expected_frames else "FAIL",
        "synthetic_label_explicit": "PASS",
    }
    downstream = all(value == "PASS" for value in hard_gates.values())
    result = {
        "schema_version": "clean-synthetic-propainter-baseline-result-v1",
        "created_at": now(),
        "status": "PASS_SYNTHETIC_CLEAN_BASELINE_GRADE_B" if downstream else "GRADE_C",
        "task": spec["task"],
        "session": spec["session"],
        "grade": "B" if downstream else "C",
        "downstream_authorized": downstream,
        "frame_interval": [start, stop],
        "frame_count": expected_frames,
        "method": "REAL_TEMPORAL_STEREO_DONOR_THEN_PROPAINTER_FOR_REMAINDER",
        "pixel_fractions_within_removal": {
            "real_donor": real_fraction,
            "synthetic_propainter": synthetic_fraction,
        },
        "hard_gates": hard_gates,
        "vendor": {
            "name": "ProPainter",
            "path": str(propainter_root),
            "commit": commit,
            "weights": weight_refs,
            "upstream_returncode": completed.returncode,
            "upstream_png_frame_count": upstream_frame_count,
            "known_imageio_preview_writer_only_failure_bypassed": known_imageio_video_writer_only_failure,
            "published_video_encoder": "PROJECT_FFMPEG_FROM_COMPLETE_UPSTREAM_PNG_SEQUENCE",
        },
        "inputs": {
            "spec": ref(spec_path),
            "mask_frame_manifest": ref(mask_manifest_path),
            "real_donor_source_manifest": ref(donor_manifest_path),
            "prepare_result": ref(prepare_result_path),
        },
        "artifacts": {
            "clean_synthetic_master": ref(master_video),
            "chinese_full_review": ref(review_video),
            "source_map_manifest": ref(source_manifest_path),
            "propainter_log": ref(log_path),
        },
        "claim_limit": "Usable synthetic Clean baseline. REAL_DONOR pixels are observed same-session pixels; SYNTHETIC_PROPAINTER pixels are generated and must not be interpreted as physical background truth.",
    }
    result_path = output_root / "RESULT.json"
    write_json(result_path, result)
    review = {
        "schema_version": "clean-synthetic-agent-review-v1",
        "created_at": now(),
        "stage": "CLEAN_SYNTHETIC",
        "task": spec["task"],
        "session": spec["session"],
        "grade": result["grade"],
        "downstream_authorized": downstream,
        "hard_gates": hard_gates,
        "result": ref(result_path),
        "review_video": ref(review_video),
        "claim_limit": result["claim_limit"],
    }
    write_json(output_root / "AGENT_REVIEW.json", review)
    print(json.dumps({"status": result["status"], "output": str(output_root)}, ensure_ascii=False))
    return 0 if downstream else 2


if __name__ == "__main__":
    raise SystemExit(main())
