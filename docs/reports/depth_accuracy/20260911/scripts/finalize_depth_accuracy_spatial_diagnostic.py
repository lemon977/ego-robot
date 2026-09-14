#!/usr/bin/env python3
"""Validate and receipt the 2026-09-11 depth accuracy diagnostic package."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "tasks/control/runs/20260911_depth_accuracy_spatial_diagnostic_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty artifact: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream: {path}")
    stream = streams[0]
    expected = {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": 1920,
        "height": 720,
        "r_frame_rate": "30/1",
        "nb_read_frames": "293",
    }
    for key, value in expected.items():
        if stream.get(key) != value:
            raise RuntimeError(
                f"video contract mismatch for {path}: {key}={stream.get(key)!r}, "
                f"expected {value!r}"
            )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        check=True,
    )
    return {**stream, "full_decode": "PASS"}


def validate_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    if width < 640 or height < 360:
        raise RuntimeError(f"unexpectedly small meeting image: {path} {width}x{height}")
    return {"width": width, "height": height, "mode": mode, "decode": "PASS"}


def validate_markdown_links(path: Path) -> list[str]:
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8"))
    checked: list[str] = []
    for raw in links:
        target = raw.split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.exists():
            raise RuntimeError(f"broken markdown link in {path}: {raw} -> {resolved}")
        checked.append(str(resolved))
    return checked


def write_new(path: Path, data: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"refusing overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def main() -> None:
    media = ROOT / "media_v1"
    tables = ROOT / "tables_v1"
    diagrams = ROOT / "diagrams_v1"
    videos = [
        media / "Chips034_Right_3D_Depth_Comparison.mp4",
        media / "Chips034_Right_Before_After_Z_Alignment.mp4",
        media / "Chips034_Right_Proxy_Skeleton.mp4",
    ]
    images = [
        media / "Chips034_Right_Proxy_Skeleton.png",
        tables / "HAWOR_STEREO_DIFFERENCE_TABLE.png",
        diagrams / "DEPTH_TYPES_OVERVIEW_ZH.png",
        diagrams / "INTERNAL_VS_EXTERNAL_ACCURACY_ZH.png",
    ]
    documents = [
        PROJECT / "docs/pipeline/DEPTH_ACCURACY_CURRENT_STATUS_ZH.md",
        PROJECT / "docs/pipeline/EXTERNAL_DEPTH_VALIDATION_PLAN_ZH.md",
        PROJECT / "docs/pipeline/C2W_ROBOT_COORDINATE_AUDIT.md",
        PROJECT / "docs/pipeline/MEETING_DEPTH_SUMMARY_ZH.md",
    ]
    other = [
        media / "MEDIA_DELIVERY_RECEIPT.json",
        tables / "HAWOR_STEREO_DIFFERENCE_TABLE.csv",
        tables / "CHIPS034_RIGHT_THREE_WINDOW_REGIONAL_METRICS.json",
        tables / "RESULT.json",
        diagrams / "RESULT.json",
        PROJECT / "tools/run_hawor_stereo_spatial_relation_video.py",
        PROJECT / "tools/build_hawor_stereo_difference_report.py",
        PROJECT / "tools/build_depth_accuracy_meeting_diagrams.py",
    ]

    video_checks = {str(path.resolve()): ffprobe(path) for path in videos}
    image_checks = {str(path.resolve()): validate_image(path) for path in images}

    csv_path = tables / "HAWOR_STEREO_DIFFERENCE_TABLE.csv"
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 5:
        raise RuntimeError(f"difference table is unexpectedly short: {len(rows)} rows")

    for path in (
        media / "MEDIA_DELIVERY_RECEIPT.json",
        tables / "CHIPS034_RIGHT_THREE_WINDOW_REGIONAL_METRICS.json",
        tables / "RESULT.json",
        diagrams / "RESULT.json",
    ):
        json.loads(path.read_text(encoding="utf-8"))

    markdown_checks = {
        str(path.resolve()): validate_markdown_links(path) for path in documents
    }
    all_artifacts = videos + images + documents + other
    artifacts = [artifact(path) for path in all_artifacts]
    if len({row["path"] for row in artifacts}) != len(artifacts):
        raise RuntimeError("duplicate artifact in final receipt")

    sums = "".join(
        f"{row['sha256']}  {row['path']}\n" for row in sorted(artifacts, key=lambda x: x["path"])
    )
    sums_path = ROOT / "SHA256SUMS.txt"
    write_new(sums_path, sums.encode("utf-8"))
    receipt = {
        "schema_version": "depth-accuracy-spatial-diagnostic-final-receipt-v1",
        "status": "COMPLETE_VALIDATED",
        "authority": False,
        "external_ground_truth": False,
        "claim_limit": (
            "HaWoR-versus-Stereo cross-system diagnostic only; FoundationStereo remains "
            "VISUAL_OBJECT6D_CANDIDATE_INPUT, not millimeter physical or Robot contact truth."
        ),
        "counts": {
            "artifacts": len(artifacts),
            "videos": len(videos),
            "images": len(images),
            "documents": len(documents),
            "csv_rows": len(rows),
        },
        "video_checks": video_checks,
        "image_checks": image_checks,
        "markdown_link_checks": {
            "status": "PASS",
            "documents": len(markdown_checks),
            "links": sum(len(value) for value in markdown_checks.values()),
            "resolved": markdown_checks,
        },
        "json_parse": "PASS",
        "csv_parse": "PASS",
        "artifacts": artifacts,
        "sha256sums": artifact(sums_path),
    }
    receipt_path = ROOT / "FINAL_RECEIPT.json"
    write_new(
        receipt_path,
        (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(receipt_path),
                "receipt_sha256": sha256(receipt_path),
                "sha256sums_sha256": sha256(sums_path),
                "counts": receipt["counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

