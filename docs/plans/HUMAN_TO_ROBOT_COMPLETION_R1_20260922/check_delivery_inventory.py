#!/usr/bin/env python3
"""Read-only delivery inventory lint. NOT an algorithm/visual quality certificate.

Python >=3.10; standard library only. --media additionally needs ffprobe/ffmpeg.
All references are resolved relative to the delivery JSON unless absolute.
No project files, receipts, or authority are changed. JSON is emitted to stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def checked_ref(ref: Any, root: Path, *, media: bool = False) -> Path:
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise ValueError("missing path/SHA file reference")
    sha = ref.get("sha256")
    if not isinstance(sha, str) or not HEX64.fullmatch(sha):
        raise ValueError("invalid or missing SHA256")
    path = Path(ref["path"]).expanduser()
    path = path if path.is_absolute() else root / path
    if media and any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"delivery media must not use symlinks: {path}")
    if not path.is_file():
        raise ValueError(f"file missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"empty file: {path}")
    if digest(path) != sha:
        raise ValueError(f"SHA mismatch: {path}")
    return path.resolve()


def timeline(data: dict[str, Any], expected: int) -> None:
    if type(data.get("frame_count")) is not int or data["frame_count"] != expected:
        raise ValueError("source/receipt frame_count differs from contract")
    ids = data.get("source_frame_ids")
    ts = data.get("timestamps_ns")
    for name, values in (("source_frame_ids", ids), ("timestamps_ns", ts)):
        if not isinstance(values, list) or len(values) != expected:
            raise ValueError(f"{name} length mismatch")
        if any(type(v) is not int for v in values):
            raise ValueError(f"{name} must be integer values, not null/bool")
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError(f"{name} must be strictly increasing")
    if any(i < 0 for i in ids):
        raise ValueError("source_frame_ids must be nonnegative")


def decode_video(path: Path, expected: int, timeout: int) -> dict[str, int]:
    for tool in ("ffprobe", "ffmpeg"):
        if not shutil.which(tool):
            raise ValueError(f"missing {tool}; media NOT_EVALUATED, cannot pass")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames,width,height",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=timeout, check=True, env=env,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("expected one selected video stream")
    s = streams[0]
    n = int(s.get("nb_read_frames", -1))
    if n != expected:
        raise ValueError(f"decoded frame count {n}, expected {expected}")
    if int(s.get("width", 0)) <= 0 or int(s.get("height", 0)) <= 0:
        raise ValueError("invalid video dimensions")
    decode = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1",
         "-i", str(path), "-map", "0:v:0", "-threads", "1", "-f", "null", "-"],
        capture_output=True, text=True, timeout=timeout, check=True, env=env,
    )
    if decode.stderr.strip():
        raise ValueError("full decoder reported errors: " + decode.stderr[-1000:])
    return {"decoded_frames": n, "width": int(s["width"]), "height": int(s["height"])}


def verify(contract_path: Path, delivery_path: Path, pinned_sha: str,
           *, media: bool = False, timeout: int = 120) -> dict[str, Any]:
    """Validate inventory only. Caller must separately run frozen project QA."""
    result: dict[str, Any] = {
        "inventory_status": "FAIL", "quality_status": "NOT_ASSESSED",
        "media_requested": media, "errors": [], "checked": [],
        "warning": "Inventory PASS is not product, contact, or algorithm quality PASS.",
    }
    errors: list[str] = result["errors"]
    try:
        if not HEX64.fullmatch(pinned_sha) or digest(contract_path) != pinned_sha:
            raise ValueError("frozen delivery contract SHA mismatch")
        contract = load_json(contract_path)
        delivery = load_json(delivery_path)
        if contract.get("schema_version") != 1 or delivery.get("schema_version") != 1:
            raise ValueError("unsupported schema_version")
        if delivery.get("route") != contract.get("route"):
            raise ValueError("route mismatch")
        if not isinstance(delivery.get("run_id"), str) or not delivery["run_id"].strip():
            raise ValueError("actual run_id missing")
        for key, expected in (("scope", "OFFLINE_VISUAL"),
                              ("training_eligible", False),
                              ("control_ground_truth", False)):
            if delivery.get(key) != expected or type(delivery.get(key)) is not type(expected):
                raise ValueError(f"invalid {key}")
        specs_list = contract["expected_artifacts"]
        if not isinstance(specs_list, list) or not specs_list:
            raise ValueError("empty expected artifact contract")
        specs = {s["artifact_id"]: s for s in specs_list}
        if len(specs) != len(specs_list):
            raise ValueError("duplicate contract artifact_id")
        rows = delivery.get("artifacts")
        if not isinstance(rows, list):
            raise ValueError("artifacts must be a list")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
        return result

    seen: set[str] = set()
    media_paths: set[Path] = set()
    root = delivery_path.resolve().parent
    for row in rows:
        aid = row.get("artifact_id") if isinstance(row, dict) else None
        if not isinstance(aid, str) or aid not in specs:
            errors.append(f"unknown/invalid artifact_id: {aid}")
            continue
        if aid in seen:
            errors.append(f"duplicate artifact_id: {aid}")
            continue
        seen.add(aid)
        spec = specs[aid]
        try:
            for key in ("session_id", "kind"):
                if row.get(key) != spec[key]:
                    raise ValueError(f"{key} mismatch")
            n = spec["frames"]
            if type(n) is not int or n <= 0:
                raise ValueError("invalid contract frame count")
            path = checked_ref(row.get("file"), root, media=True)
            if path.suffix.lower() != ".mp4":
                raise ValueError("expected actual .mp4 media")
            if path in media_paths:
                raise ValueError("same media file reused for multiple deliveries")
            media_paths.add(path)
            receipt_path = checked_ref(row.get("receipt"), root)
            receipt = load_json(receipt_path)
            for key in ("artifact_id", "session_id", "kind"):
                if receipt.get(key) != row[key]:
                    raise ValueError(f"receipt {key} mismatch")
            if receipt.get("media_sha256") != row["file"]["sha256"]:
                raise ValueError("receipt not bound to actual media SHA")
            timeline(receipt, n)
            if receipt.get("output_frame_ids") != list(range(n)):
                raise ValueError("output frame IDs incomplete or reordered")
            source_path = checked_ref(receipt.get("source_ref"), root)
            source = load_json(source_path)
            if source.get("session_id") != spec["session_id"]:
                raise ValueError("source session mismatch; relabelled video not allowed")
            timeline(source, n)
            for key in ("source_frame_ids", "timestamps_ns"):
                if receipt[key] != source[key]:
                    raise ValueError(f"source {key} mismatch")
            if spec["kind"] == "robot_product":
                background = receipt.get("background")
                if not isinstance(background, dict) or background.get("kind") != "CLEAN_SYNTHETIC":
                    raise ValueError("robot product did not declare actual Clean consumption")
                clean_path = checked_ref(background.get("result_ref"), root)
                clean = load_json(clean_path)
                if clean.get("session_id") != spec["session_id"] or clean.get("frame_count") != n:
                    raise ValueError("Clean session/coverage mismatch")
                if clean.get("execution_state") != "COMPLETED":
                    raise ValueError("Clean was not completed")
                if type(clean.get("model_invocations")) is not int or clean["model_invocations"] < 1:
                    raise ValueError("no actual model invocation in bound Clean producer receipt")
            row_result: dict[str, Any] = {"artifact_id": aid, "frames": n,
                                          "media_status": "NOT_EVALUATED"}
            if media:
                row_result.update(decode_video(path, n, timeout))
                row_result["media_status"] = "PASS"
            result["checked"].append(row_result)
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            errors.append(f"{aid}: {str(exc)[-1500:]}")
    for aid in sorted(set(specs) - seen):
        errors.append(f"missing artifact: {aid}")
    result["expected_count"] = len(specs)
    result["checked_count"] = len(result["checked"])
    result["inventory_status"] = "PASS" if not errors else "FAIL"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True,
                        help="SHA frozen before execution, not recomputed to bless edits")
    parser.add_argument("--media", action="store_true", help="fully decode each video")
    parser.add_argument("--timeout", type=int, default=120, help="per decoder process timeout")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    result = verify(args.contract, args.delivery, args.contract_sha256,
                    media=args.media, timeout=args.timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["inventory_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
