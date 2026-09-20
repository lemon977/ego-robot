#!/usr/bin/env python3
"""Build one bounded D2 fresh ProPainter *offline visual* candidate.

This successor consumes only the lossless Raw frames and the frozen
``M_remove``, ``M_write``, ``M_flow``, ``UNKNOWN`` and object-protection
source map published by D1_CLEAN_PREP/PREPARED_V1.  It never reads an older
Clean image, a donor frame, Object6D, Contact, Robot, or hidden ground truth.

The frozen policy has two deliberately different session actions:

* ``play_cards_0915_044`` has ``M_write == 0`` and is copied from Raw.  No
  model call is made for Poker.
* ``get_potato_chips_0915_097`` sends Raw plus ``M_flow`` to the unchanged,
  SHA-pinned ProPainter implementation.  Generated RGB is composited only in
  ``M_write``.  ``UNKNOWN`` remains exactly ``M_write`` because generated
  pixels are visual hypotheses, not observed scene truth.

ProPainter is temporally bidirectional.  Every output therefore remains
``OFFLINE_BIDIRECTIONAL_VISUAL_ONLY`` and is ineligible for training, contact,
control or deployment.  There is no parameter sweep or model fallback.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


RAW = np.uint8(0)
PROTECTED_VISIBLE_OBJECT = np.uint8(2)
SYNTHETIC_PROPAINTER = np.uint8(3)
UNKNOWN = np.uint8(250)
INSTANCE_CODES = (10, 11, 12)


class ContractError(RuntimeError):
    """A frozen D2 input or output contract was violated."""


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_ref(value: dict[str, Any], label: str) -> Path:
    path = Path(value["path"]).resolve(strict=True)
    actual = file_ref(path)
    for key in ("bytes", "sha256"):
        if actual[key] != value[key]:
            raise ContractError(f"{label} {key} mismatch: {path}")
    return path


def verify_lexical_resolved_ref(
    value: dict[str, Any], expected_lexical: Path, label: str
) -> dict[str, Any]:
    """Verify both a configured symlink-facing name and its canonical target."""
    lexical = Path(value["lexical_path"])
    if not lexical.is_absolute():
        raise ContractError(f"{label} lexical_path must be absolute")
    # absolute() normalizes spelling without following symlinks; resolve() here
    # would collapse the exact vendor-facing provenance that this gate protects.
    lexical_absolute = lexical.absolute()
    expected_absolute = expected_lexical.absolute()
    if lexical_absolute != expected_absolute:
        raise ContractError(
            f"{label} lexical path mismatch: {lexical_absolute} != {expected_absolute}"
        )
    resolved = lexical_absolute.resolve(strict=True)
    configured_resolved = Path(value["resolved_path"])
    if not configured_resolved.is_absolute() or str(configured_resolved) != str(resolved):
        raise ContractError(
            f"{label} resolved target mismatch: {resolved} != {configured_resolved}"
        )
    actual_bytes = resolved.stat().st_size
    actual_sha = sha256(resolved)
    if actual_bytes != value["bytes"] or actual_sha != value["sha256"]:
        raise ContractError(f"{label} target bytes/SHA mismatch: {resolved}")
    return {
        "lexical_path": str(lexical_absolute),
        "resolved_path": str(resolved),
        "bytes": actual_bytes,
        "sha256": actual_sha,
        "lexical_parent_is_symlink": lexical_absolute.parent.is_symlink(),
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def projected_ref(actual_path: Path, staging_root: Path, final_root: Path) -> dict[str, Any]:
    """Hash a staging artifact while publishing only its future final path."""
    actual_path = actual_path.resolve(strict=True)
    staging_root = staging_root.resolve(strict=True)
    try:
        relative = actual_path.relative_to(staging_root)
    except ValueError as exception:
        raise ContractError(f"artifact is outside staging root: {actual_path}") from exception
    value = file_ref(actual_path)
    value["path"] = str(final_root.resolve() / relative)
    if ".staging." in value["path"]:
        raise ContractError("projected artifact reference leaked staging path")
    return value


def publish_directory_atomically(final_root: Path, builder: Any) -> Any:
    """Build in a same-filesystem sibling and expose it with one rename."""
    final_root = final_root.resolve()
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists() or final_root.is_symlink():
        raise ContractError(f"no-clobber final output exists: {final_root}")
    staging_root = Path(tempfile.mkdtemp(
        prefix=f".{final_root.name}.staging.", dir=final_root.parent
    )).resolve()
    try:
        result = builder(staging_root, final_root)
        staging_token = str(staging_root)
        for manifest in staging_root.rglob("*.json"):
            if staging_token in manifest.read_text(encoding="utf-8"):
                raise ContractError(f"staging path leaked into JSON artifact: {manifest}")
        if final_root.exists() or final_root.is_symlink():
            raise ContractError(f"final output appeared during build: {final_root}")
        # staging_root and final_root are siblings, so this is one same-filesystem
        # visibility transition.  No file is copied after this point.
        os.replace(staging_root, final_root)
        return result
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(Image.open(path).convert("L")) > 0
    if value.shape != shape:
        raise ContractError(f"mask shape {value.shape} != RGB shape {shape}: {path}")
    return value


def read_u8(path: Path, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if value.shape != shape:
        raise ContractError(f"map shape {value.shape} != RGB shape {shape}: {path}")
    return value


def decoded_sha256(rgb: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def panel(rgb: np.ndarray, title: str, subtitle: str = "", size: tuple[int, int] = (640, 480)) -> np.ndarray:
    width, height = size
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(resized)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, width, 70), fill=(0, 0, 0, 190))
    draw.text((14, 7), title, font=font(24), fill=(255, 255, 255, 255))
    if subtitle:
        draw.text((14, 39), subtitle, font=font(16), fill=(235, 235, 235, 255))
    return np.asarray(image)


def mask_overlay(raw: np.ndarray, m_write: np.ndarray, m_flow: np.ndarray, protected: np.ndarray) -> np.ndarray:
    value = raw.astype(np.float32)
    flow_only = m_flow & ~m_write
    value[flow_only] = 0.55 * value[flow_only] + 0.45 * np.array([255, 190, 0], np.float32)
    value[m_write] = 0.35 * value[m_write] + 0.65 * np.array([255, 0, 255], np.float32)
    value[protected] = 0.25 * value[protected] + 0.75 * np.array([0, 255, 255], np.float32)
    return np.clip(value, 0, 255).astype(np.uint8)


def difference_overlay(raw: np.ndarray, candidate: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    diff = np.max(np.abs(candidate.astype(np.int16) - raw.astype(np.int16)), axis=2)
    base = (raw.astype(np.float32) * 0.25).astype(np.uint8)
    changed = diff > 0
    base[changed] = np.array([255, 60, 30], np.uint8)
    # UNKNOWN remains visible even when a synthetic pixel happens to equal Raw.
    base[unknown & ~changed] = np.array([190, 0, 255], np.uint8)
    return base


def composite_visual_candidate(
    raw: np.ndarray,
    generated: np.ndarray,
    m_remove: np.ndarray,
    m_write: np.ndarray,
    m_flow: np.ndarray,
    upstream_unknown: np.ndarray,
    upstream_source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Apply generated pixels only in M_write and retain semantic UNKNOWN."""
    if raw.shape != generated.shape or raw.ndim != 3 or raw.shape[2] != 3:
        raise ContractError("Raw/generated RGB shapes differ")
    shape = raw.shape[:2]
    for name, mask in (
        ("M_remove", m_remove), ("M_write", m_write), ("M_flow", m_flow),
        ("UNKNOWN", upstream_unknown),
    ):
        if mask.shape != shape:
            raise ContractError(f"{name} shape mismatch")
    if upstream_source.shape != shape:
        raise ContractError("source map shape mismatch")
    if np.any(m_remove & ~m_write):
        raise ContractError("M_remove must be a subset of M_write")
    if np.any(m_write & ~m_flow):
        raise ContractError("M_write must be a subset of M_flow")
    if not np.array_equal(upstream_unknown, m_write):
        raise ContractError("D1 UNKNOWN must equal M_write")
    protected = np.isin(upstream_source, (int(PROTECTED_VISIBLE_OBJECT), *INSTANCE_CODES))
    if np.any(protected & m_write):
        raise ContractError("protected object overlaps M_write")

    candidate = raw.copy()
    candidate[m_write] = generated[m_write]
    # A defensive assignment makes object preservation explicit even though the
    # domains are required to be disjoint.
    candidate[protected] = raw[protected]
    unknown = m_write.copy()
    visual_source = np.full(shape, RAW, dtype=np.uint8)
    visual_source[protected] = upstream_source[protected]
    visual_source[m_write] = SYNTHETIC_PROPAINTER
    changed = np.any(candidate != raw, axis=2)
    stats = {
        "m_remove_pixels": int(m_remove.sum()),
        "m_write_pixels": int(m_write.sum()),
        "m_flow_pixels": int(m_flow.sum()),
        "unknown_pixels": int(unknown.sum()),
        "protected_pixels": int(protected.sum()),
        "changed_pixels": int(changed.sum()),
        "changed_outside_m_write_pixels": int((changed & ~m_write).sum()),
        "changed_protected_pixels": int((changed & protected).sum()),
    }
    if stats["changed_outside_m_write_pixels"] or stats["changed_protected_pixels"]:
        raise ContractError("candidate changed Raw outside M_write or in protected object")
    return candidate, visual_source, unknown, stats


def validate_policy(config: dict[str, Any]) -> None:
    required = {
        "algorithm_id": "robot_quality_recovery_v21_d2_fresh_propainter_offline_v1",
        "execution_semantics": "OFFLINE_BIDIRECTIONAL_VISUAL_ONLY",
        "training_eligible": False,
        "clean_terminal": False,
        "control_ground_truth": False,
        "model_fallback_allowed": False,
        "parameter_sweep_allowed": False,
        "old_clean_allowed": False,
        "future_donor_allowed": False,
        "hidden_ground_truth_allowed": False,
        "object6d_contact_robot_allowed": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ContractError(f"frozen policy mismatch: {key}")
    if config.get("fixed_sessions") != ["play_cards_0915_044", "get_potato_chips_0915_097"]:
        raise ContractError("fixed session list changed")
    actions = config["session_actions"]
    if actions.get("play_cards_0915_044") != "RAW_PASS_THROUGH_REQUIRE_ZERO_M_WRITE_NO_MODEL":
        raise ContractError("Poker action changed")
    if actions.get("get_potato_chips_0915_097") != "PROPAINTER_M_FLOW_CONTEXT_COMPOSITE_M_WRITE_ONLY":
        raise ContractError("Chips action changed")
    p = config["inference_parameters"]
    if p != {
        "process_width": 960,
        "process_height": 720,
        "mask_dilation": 4,
        "ref_stride": 10,
        "neighbor_length": 10,
        "subvideo_length": 80,
        "raft_iter": 20,
        "fp16": True,
        "fps": 30,
    }:
        raise ContractError("inference parameters changed or expanded")
    write_contract = config["write_contract"]
    if write_contract.get("input_mask_domain") != "M_flow":
        raise ContractError("model input mask domain must remain M_flow")
    if write_contract.get("effective_internal_model_mask") != "DILATE(RESIZED_M_flow,4)_INSIDE_PROPAINTER":
        raise ContractError("effective ProPainter mask semantics changed")
    if write_contract.get("publish_write_domain") != "M_write":
        raise ContractError("publish write domain must remain M_write")


def validate_vendor(config: dict[str, Any]) -> dict[str, Any]:
    vendor = config["vendor"]
    root = Path(vendor["root"]).resolve(strict=True)
    entry = verify_ref(vendor["inference_entry"], "ProPainter inference entry")
    if entry != root / "inference_propainter.py":
        raise ContractError("inference entry is outside frozen vendor root")
    weights: dict[str, Any] = {}
    for name in ("raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth"):
        weights[name] = verify_lexical_resolved_ref(
            vendor["weights"][name], root / "weights" / name, f"weight {name}"
        )
    tree_digest = hashlib.sha256()
    tree_files = 0
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root)
        if "weights" in relative.parts or "assets" in relative.parts:
            continue
        tree_digest.update(relative.as_posix().encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(sha256(path).encode("ascii"))
        tree_digest.update(b"\n")
        tree_files += 1
    if tree_files != vendor["current_source_tree_manifest_file_count"]:
        raise ContractError("ProPainter source-tree file count mismatch")
    if tree_digest.hexdigest() != vendor["current_source_tree_manifest_sha256"]:
        raise ContractError("ProPainter source-tree manifest SHA mismatch")
    return {
        "root": str(root),
        "historical_upstream_commit": vendor["historical_upstream_commit"],
        "current_vendor_has_nested_git_metadata": (root / ".git").exists(),
        "source_tree_manifest_file_count": tree_files,
        "source_tree_manifest_sha256": tree_digest.hexdigest(),
        "inference_entry": file_ref(entry),
        "weights": weights,
    }


def validate_d1(config: dict[str, Any], full_sha: bool) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    d1 = config["d1"]
    terminal_path = verify_ref(d1["terminal_index"], "D1 terminal index")
    prepared_path = verify_ref(d1["prepared_result"], "D1 prepared result")
    terminal = read_json(terminal_path)
    prepared = read_json(prepared_path)
    if terminal.get("status") != "BLOCKED_PREREQ_FRESH_INPAINTING_OR_UNKNOWN":
        raise ContractError("unexpected D1 terminal status")
    if terminal.get("clean_terminal") is not False:
        raise ContractError("D1 must not be a Clean terminal")
    if prepared.get("old_clean_consumed") or prepared.get("future_frame_donor_consumed") or prepared.get("hidden_ground_truth_consumed"):
        raise ContractError("D1 provenance prohibition violated")

    reports: dict[str, dict[str, Any]] = {}
    for sid in config["fixed_sessions"]:
        manifest_path = verify_ref(d1["frame_manifests"][sid], f"D1 manifest {sid}")
        manifest = read_json(manifest_path)
        frames = manifest.get("frames", [])
        expected = 166 if sid == "play_cards_0915_044" else 394
        if len(frames) != expected or manifest.get("frame_count") != expected:
            raise ContractError(f"{sid}: frame count mismatch")
        if [int(row["frame_id"]) for row in frames] != list(range(expected)):
            raise ContractError(f"{sid}: frame order mismatch")
        totals = {"m_write_pixels": 0, "m_flow_pixels": 0, "protected_pixels": 0}
        refs_verified = 0
        for row in frames:
            totals["m_write_pixels"] += int(row["m_write_pixels"])
            totals["m_flow_pixels"] += int(row["m_flow_pixels"])
            totals["protected_pixels"] += int(row["protected_pixels"])
            if full_sha:
                for key in ("raw_rgb", "M_remove", "M_write", "M_flow", "UNKNOWN", "source_map"):
                    verify_ref(row[key], f"{sid} frame {row['frame_id']} {key}")
                    refs_verified += 1
        if sid == "play_cards_0915_044" and totals["m_write_pixels"] != 0:
            raise ContractError("Poker M_write must remain zero")
        if sid == "get_potato_chips_0915_097":
            if manifest.get("chips_instance_mode") != "THREE_SEPARATE_NO_UNION":
                raise ContractError("Chips three-instance/no-union contract changed")
            if manifest.get("unknown_hand_sides") != ["left"]:
                raise ContractError("Chips left-hand UNKNOWN contract changed")
        reports[sid] = {
            "manifest": file_ref(manifest_path),
            "frame_count": expected,
            "totals": totals,
            "per_frame_refs_sha_verified": refs_verified,
        }
    return {"terminal_index": file_ref(terminal_path), "prepared_result": file_ref(prepared_path)}, reports


def preflight(config_path: Path, full_sha: bool = True) -> dict[str, Any]:
    config = read_json(config_path)
    validate_policy(config)
    vendor = validate_vendor(config)
    d1, sessions = validate_d1(config, full_sha=full_sha)
    return {
        "schema_version": "0915-robot-recovery-v21-d2-fresh-propainter-offline-preflight-v1",
        "created_at": now(),
        "status": "PASSED_PREPARED_NOT_EXECUTED",
        "config": file_ref(config_path),
        "vendor": vendor,
        "d1": d1,
        "sessions": sessions,
        "gpu_used": False,
        "model_execution_performed": False,
        "claim_limit": "Code/config/input preflight only; no generated pixels or Clean authority.",
    }


def encode_video(frame_root: Path, output: Path, fps: int) -> None:
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(fps), "-i", str(frame_root / "%06d.png"),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], check=True)


def decode_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def require_exact_png_sequence(root: Path, expected: int, digits: int) -> list[Path]:
    expected_names = [f"{index:0{digits}d}.png" for index in range(expected)]
    actual_names = sorted(path.name for path in root.glob("*.png"))
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))[:5]
        extra = sorted(set(actual_names) - set(expected_names))[:5]
        raise ContractError(
            f"generated PNG filename sequence mismatch digits={digits} "
            f"missing={missing} extra={extra}"
        )
    return [root / name for name in expected_names]


def normalize_generated_frames(model_root: Path, output_root: Path, expected: int) -> list[Path]:
    """Normalize pinned vendor four-digit files to the D2 six-digit contract."""
    upstream_root = model_root / "frames" / "frames"
    upstream = require_exact_png_sequence(upstream_root, expected, digits=4)
    output_root.mkdir(parents=True, exist_ok=False)
    for index, source in enumerate(upstream):
        target = output_root / f"{index:06d}.png"
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise ContractError(f"generated frame normalization SHA mismatch: {index}")
    return require_exact_png_sequence(output_root, expected, digits=6)


def _execute_into_staging(
    config_path: Path, output_root: Path, final_root: Path
) -> dict[str, Any]:
    """Build the full result under staging; never expose staging paths."""
    config = read_json(config_path)
    preflight_result = preflight(config_path, full_sha=True)
    write_json(output_root / "PREFLIGHT.json", preflight_result)

    params = config["inference_parameters"]
    vendor_root = Path(config["vendor"]["root"])
    session_manifests = {
        sid: read_json(Path(config["d1"]["frame_manifests"][sid]["path"]))
        for sid in config["fixed_sessions"]
    }
    chips_sid = "get_potato_chips_0915_097"
    chips_rows = session_manifests[chips_sid]["frames"]
    input_root = output_root / "propainter_input_chips097"
    input_frames = input_root / "frames"
    input_masks = input_root / "masks"
    input_frames.mkdir(parents=True)
    input_masks.mkdir()
    for row in chips_rows:
        idx = int(row["frame_id"])
        raw = read_rgb(Path(row["raw_rgb"]["path"]))
        m_flow = read_mask(Path(row["M_flow"]["path"]), raw.shape[:2])
        resized = cv2.resize(raw, (params["process_width"], params["process_height"]), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(
            m_flow.astype(np.uint8) * 255,
            (params["process_width"], params["process_height"]),
            interpolation=cv2.INTER_NEAREST,
        )
        Image.fromarray(resized).save(input_frames / f"{idx:06d}.png")
        Image.fromarray(resized_mask).save(input_masks / f"{idx:06d}.png")

    model_out = output_root / "propainter_workspace_chips097"
    log_path = output_root / "PROPAINTER_RUN.log"
    command = [
        sys.executable, "inference_propainter.py",
        "--video", str(input_frames), "--mask", str(input_masks),
        "--output", str(model_out),
        "--width", str(params["process_width"]), "--height", str(params["process_height"]),
        "--mask_dilation", str(params["mask_dilation"]),
        "--ref_stride", str(params["ref_stride"]),
        "--neighbor_length", str(params["neighbor_length"]),
        "--subvideo_length", str(params["subvideo_length"]),
        "--raft_iter", str(params["raft_iter"]),
        "--save_fps", str(params["fps"]), "--save_frames", "--fp16",
    ]
    with log_path.open("xb") as log:
        completed = subprocess.run(command, cwd=vendor_root, stdout=log, stderr=subprocess.STDOUT)
    # The pinned vendor writes 0000.png..0393.png.  D2 immediately validates
    # that exact upstream sequence and byte-exactly normalizes it to the frozen
    # six-digit publication/consumption sequence 000000.png..000393.png.
    generated = normalize_generated_frames(
        model_out, output_root / "generated_frames_chips097", len(chips_rows)
    )
    writer_only_bypass = False
    if completed.returncode:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        writer_only_bypass = "PyAVPlugin.write() got an unexpected keyword argument 'quality'" in text
        if not writer_only_bypass:
            raise ContractError(f"ProPainter failed rc={completed.returncode}")

    session_results: list[dict[str, Any]] = []
    for sid in config["fixed_sessions"]:
        rows = session_manifests[sid]["frames"]
        session_root = output_root / "sessions" / sid
        candidate_root = session_root / "candidate_frames"
        map_root = session_root / "source_maps"
        unknown_root = session_root / "unknown_masks"
        review_root = session_root / "review_frames"
        for p in (candidate_root, map_root, unknown_root, review_root):
            p.mkdir(parents=True, exist_ok=True)
        output_rows: list[dict[str, Any]] = []
        totals: dict[str, int] = {}
        for row in rows:
            idx = int(row["frame_id"])
            raw = read_rgb(Path(row["raw_rgb"]["path"]))
            shape = raw.shape[:2]
            m_remove = read_mask(Path(row["M_remove"]["path"]), shape)
            m_write = read_mask(Path(row["M_write"]["path"]), shape)
            m_flow = read_mask(Path(row["M_flow"]["path"]), shape)
            prior_unknown = read_mask(Path(row["UNKNOWN"]["path"]), shape)
            prior_source = read_u8(Path(row["source_map"]["path"]), shape)
            if sid == "play_cards_0915_044":
                if m_write.any():
                    raise ContractError(f"Poker frame {idx} has nonzero M_write")
                generated_rgb = raw
            else:
                generated_small = read_rgb(generated[idx])
                generated_rgb = cv2.resize(generated_small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
            candidate, source, unknown, stats = composite_visual_candidate(
                raw, generated_rgb, m_remove, m_write, m_flow, prior_unknown, prior_source
            )
            candidate_path = candidate_root / f"{idx:06d}.png"
            source_path = map_root / f"{idx:06d}.npz"
            unknown_path = unknown_root / f"{idx:06d}.png"
            Image.fromarray(candidate).save(candidate_path)
            Image.fromarray(unknown.astype(np.uint8) * 255).save(unknown_path)
            np.savez_compressed(
                source_path,
                visual_source_kind=source,
                unknown=unknown,
                truth_valid=~unknown,
                frame_id=np.int32(idx),
                semantics=np.array("SYNTHETIC_VISUAL_DOES_NOT_CLEAR_UNKNOWN"),
            )
            protected = np.isin(prior_source, (int(PROTECTED_VISIBLE_OBJECT), *INSTANCE_CODES))
            overlay = mask_overlay(raw, m_write, m_flow, protected)
            diff = difference_overlay(raw, candidate, unknown)
            subtitle = "OFFLINE_VISUAL / NOT_FOR_TRAINING / UNKNOWN retained"
            review = np.hstack([
                panel(raw, "Raw", f"frame={idx}"),
                panel(overlay, "D1 domains", "magenta=M_write; amber=M_flow; cyan=object"),
                panel(candidate, "D2 fresh visual", subtitle),
                panel(diff, "change / UNKNOWN", "orange=changed; magenta=UNKNOWN"),
            ])
            Image.fromarray(review).save(review_root / f"{idx:06d}.png")
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + int(value)
            output_rows.append({
                "frame_id": idx,
                "raw_rgb": row["raw_rgb"],
                "candidate_rgb": projected_ref(candidate_path, output_root, final_root),
                "source_map": projected_ref(source_path, output_root, final_root),
                "unknown_mask": projected_ref(unknown_path, output_root, final_root),
                "raw_decoded_sha256": decoded_sha256(raw),
                "candidate_decoded_sha256": decoded_sha256(candidate),
                **stats,
            })
        review_video = session_root / f"{sid}_D2_FRESH_PROPAINTER_OFFLINE_FULL_REVIEW.mp4"
        encode_video(review_root, review_video, int(params["fps"]))
        decoded = decode_count(review_video)
        if decoded != len(rows):
            raise ContractError(f"{sid}: review decode count {decoded} != {len(rows)}")
        session_manifest = {
            "schema_version": "0915-robot-recovery-v21-d2-fresh-propainter-offline-session-v1",
            "session_id": sid,
            "status": "PASS_THROUGH_ZERO_WRITE" if sid.startswith("play_cards") else "COMPLETED_OFFLINE_VISUAL_UNKNOWN_RETAINED",
            "execution_semantics": "OFFLINE_BIDIRECTIONAL_VISUAL_ONLY",
            "frame_count": len(rows),
            "fps": params["fps"],
            "totals": totals,
            "frames": output_rows,
            "review_video": projected_ref(review_video, output_root, final_root),
            "unknown_semantics": "UNKNOWN remains exactly M_write after synthetic visual fill",
            "training_eligible": False,
            "clean_terminal": False,
            "claim_limit": "Structural offline visual candidate only; generated pixels are not scene truth.",
        }
        session_manifest_path = session_root / "SESSION_RESULT.json"
        write_json(session_manifest_path, session_manifest)
        session_results.append(projected_ref(session_manifest_path, output_root, final_root))

    result = {
        "schema_version": "0915-robot-recovery-v21-d2-fresh-propainter-offline-result-v1",
        "created_at": now(),
        "status": "COMPLETED_OFFLINE_VISUAL_CANDIDATE_UNKNOWN_RETAINED",
        "algorithm_id": config["algorithm_id"],
        "execution_semantics": config["execution_semantics"],
        "config": file_ref(config_path),
        "preflight": projected_ref(output_root / "PREFLIGHT.json", output_root, final_root),
        "model": {
            "executed_for_sessions": [chips_sid],
            "not_executed_for_sessions": ["play_cards_0915_044"],
            "returncode": completed.returncode,
            "known_preview_writer_only_failure_bypassed": writer_only_bypass,
            "input_mask_domain": "M_flow",
            "effective_internal_model_mask": "DILATE(RESIZED_M_flow,4)_INSIDE_PROPAINTER",
            "publish_write_domain": "M_write",
            "vendor_generated_sequence": "0000.png..0393.png_EXACT",
            "consumer_generated_sequence": "000000.png..000393.png_EXACT",
            "log": projected_ref(log_path, output_root, final_root),
        },
        "sessions": session_results,
        "old_clean_consumed": False,
        "future_donor_consumed": False,
        "hidden_ground_truth_consumed": False,
        "object6d_contact_robot_consumed": False,
        "training_eligible": False,
        "clean_terminal": False,
        "control_ground_truth": False,
        "claim_limit": "One bounded fresh offline ProPainter visual candidate. No Clean quality, causal training, contact, control or deployment authority.",
    }
    write_json(output_root / "RESULT.json", result)
    return result


def execute(config_path: Path, output_root: Path) -> dict[str, Any]:
    """Execute through sibling staging and one atomic final-directory rename.

    The caller must own the shared V7.1 GPU lease.  If preparation, inference,
    validation, video encoding or manifest writing fails, the staging sibling is
    recursively removed and the final path remains absent.
    """
    return publish_directory_atomically(
        output_root,
        lambda staging, final: _execute_into_staging(config_path, staging, final),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--full-sha", action="store_true", help="Hash every D1 per-frame reference in preflight")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    if args.preflight_only:
        result = preflight(config_path, full_sha=args.full_sha)
        if args.receipt:
            write_json(args.receipt.resolve(), result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.receipt:
        raise ContractError("--receipt is only valid with --preflight-only")
    if args.output_root is None:
        raise ContractError("--output-root is required for model execution")
    result = execute(config_path, args.output_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
