#!/usr/bin/env python3
"""Atomically publish B3 review videos into the shallow visual tree.

This renderer/publisher is intentionally independent from the GPU worker.  It
only accepts an already committed deep B3 terminal, fully decodes every source
and copied video, then exposes a fresh shallow directory with one os.replace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from typing import Any


ALLOWED_DEEP_STATUSES = {
    "COMPLETED_DIAGNOSTIC_SEPARATION_REJECTED",
    "COMPLETED_DIAGNOSTIC_SEPARATION_UNKNOWN",
    "COMPLETED_DIAGNOSTIC_SEPARATION_GEOMETRY_PASS",
}
ALLOWED_SESSION_STATUSES = {
    "COMPLETED_SEPARATION_REJECTED",
    "COMPLETED_SEPARATION_UNKNOWN",
    "COMPLETED_SEPARATION_GEOMETRY_PASS",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def decode_gate(path: Path, expected_frames: int) -> dict[str, Any]:
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("review must contain exactly one video stream")
    stream = streams[0]
    if (
        int(stream["width"]) != 1920
        or int(stream["height"]) != 480
        or stream["r_frame_rate"] != "30/1"
        or int(stream["nb_read_frames"]) != expected_frames
    ):
        raise RuntimeError(f"review video contract mismatch: {stream}")
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-map", "0:v:0", "-f", "null", "-",
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return {"full_decode": True, "frames": expected_frames, "fps": "30/1",
            "width": 1920, "height": 480}


def under(root: Path, child: Path) -> bool:
    try:
        child.relative_to(root)
        return True
    except ValueError:
        return False


def publish(deep_root: Path, final_visual_root: Path) -> dict[str, Any]:
    deep_root = deep_root.resolve(strict=True)
    result_path = deep_root / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") not in ALLOWED_DEEP_STATUSES:
        raise RuntimeError("deep B3 terminal is not publishable")
    if result.get("consumer_allowed") is not False or result.get("mask_accuracy_claimed") is not False:
        raise RuntimeError("deep B3 authority flags drift")
    final_visual_root = final_visual_root.resolve()
    final_visual_root.parent.mkdir(parents=True, exist_ok=True)
    if final_visual_root.exists() or final_visual_root.is_symlink():
        raise RuntimeError(f"immutable shallow target exists: {final_visual_root}")
    staging = final_visual_root.parent / (
        f".{final_visual_root.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    try:
        if os.stat(staging).st_dev != os.stat(final_visual_root.parent).st_dev:
            raise RuntimeError("shallow staging must share final filesystem")
        sessions = []
        for row in result.get("sessions", []):
            if row.get("status") not in ALLOWED_SESSION_STATUSES:
                raise RuntimeError(f"unrecognized per-session separation terminal: {row.get('status')}")
            source_ref = row["review"]
            source = Path(source_ref["path"]).resolve(strict=True)
            if not under(deep_root, source):
                raise RuntimeError("review ref escapes committed deep output")
            if sha256(source) != source_ref["sha256"] or source.stat().st_size != int(source_ref["bytes"]):
                raise RuntimeError("deep review ref drift")
            source_decode = decode_gate(source, int(row["frame_count"]))
            destination = staging / source.name
            shutil.copy2(source, destination)
            copied_decode = decode_gate(destination, int(row["frame_count"]))
            if sha256(destination) != source_ref["sha256"]:
                raise RuntimeError("shallow copy SHA drift")
            sessions.append({
                "session_id": row["session_id"],
                "separation_status": row["status"],
                "mask_quality_status": "NOT_EVALUATED_NO_GROUND_TRUTH",
                "consumer_allowed": False,
                "source": {"path": str(source), "bytes": source.stat().st_size,
                           "sha256": sha256(source), "decode": source_decode},
                "published": {"path": str(final_visual_root / destination.name),
                              "bytes": destination.stat().st_size,
                              "sha256": sha256(destination), "decode": copied_decode},
            })
        receipt = {
            "schema_version": "0915-robot-recovery-v21-b3-shallow-publication-v1",
            "status": "COMMITTED",
            "deep_terminal_status": result["status"],
            "mask_quality_status": "NOT_EVALUATED_NO_GROUND_TRUTH",
            "consumer_allowed": False,
            "mask_accuracy_claimed": False,
            "deep_result": {"path": str(result_path), "bytes": result_path.stat().st_size,
                            "sha256": sha256(result_path)},
            "sessions": sessions,
        }
        atomic_json(staging / "SHALLOW_PUBLICATION_RECEIPT.json", receipt)
        os.replace(staging, final_visual_root)
        return receipt
    except BaseException:
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = publish(args.deep_root, args.visual_root)
    print(json.dumps({"status": receipt["status"],
                      "receipt": str(args.visual_root / "SHALLOW_PUBLICATION_RECEIPT.json")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
