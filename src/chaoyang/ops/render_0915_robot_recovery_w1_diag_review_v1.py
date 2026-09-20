#!/usr/bin/env python3
"""Render the frozen A1 W1-DIAG HaWoR outputs without changing inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

import cv2
import numpy as np

from chaoyang.ops.run_0915_hawor_resize_only_canary_v1 import CHAINS, TARGET_SIZE, full_decode, open_encoder


ROOT = Path(__file__).resolve().parents[3]
A1 = ROOT / "_run/current/0915_robot_quality_recovery_15h_v2/attempts/attempt_0001/packages/A1"
EXPECTED_CANDIDATE = "9ccf7ff5eef43158848f73206a772ae756c0387cd84de9ecc72a2ee678db781b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def render(video: Path, npz_path: Path, output: Path, session_id: str, expected_frames: int,
           side_status: dict[str, str]) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as archive:
        joints = np.asarray(archive["joints_2d"], np.float64)
        observed = np.asarray(archive["observed"], bool)
    if joints.shape[:3] != (2, expected_frames, 21) or observed.shape != (2, expected_frames):
        raise RuntimeError(f"HaWoR frame axis drift: {session_id}")
    capture = cv2.VideoCapture(str(video))
    encoder = open_encoder(output, *TARGET_SIZE, 30.0)
    assert encoder.stdin is not None
    colors = ((255, 140, 20), (20, 40, 255))
    frame_index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            for side in (0, 1):
                if not observed[side, frame_index]:
                    continue
                uv = np.rint(joints[side, frame_index]).astype(np.int32)
                for chain in CHAINS:
                    cv2.polylines(image, [uv[np.asarray(chain)]], False, colors[side], 3, cv2.LINE_AA)
                cv2.circle(image, tuple(uv[0]), 7, colors[side], -1, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (1279, 92), (0, 0, 0), -1)
            cv2.putText(image, f"{session_id} | W1-DIAG raw HaWoR | {frame_index:04d}/{expected_frames - 1:04d}",
                        (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.61, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"left={'OBS' if observed[0, frame_index] else 'MISS'} [{side_status['left']}] | "
                                   f"right={'OBS' if observed[1, frame_index] else 'MISS'} [{side_status['right']}]",
                        (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, "OFFLINE_DIAGNOSTIC | NOT R0 QUALITY | NOT CONTROL | candidate 9ccf7ff5eef4",
                        (12, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (80, 210, 255), 2, cv2.LINE_AA)
            encoder.stdin.write(image.tobytes())
            frame_index += 1
    finally:
        capture.release()
        encoder.stdin.close()
    if encoder.wait() != 0 or frame_index != expected_frames:
        raise RuntimeError(f"review render failed: {session_id}")
    return {"video": ref(output), "decode": full_decode(output, expected_frames)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    args = parser.parse_args()
    output, visual = args.output_root.resolve(), args.visual_root.resolve()
    for path in (output, visual):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"fresh review output required: {path}")
    result = json.loads((A1 / "RESULT.json").read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETED_DIAGNOSTIC_ALL_TERMINAL" or result.get("candidate_signature_sha256") != EXPECTED_CANDIDATE:
        raise RuntimeError("A1 result/candidate drift")
    prepared = json.loads((A1 / "PREPARED_MANIFEST.json").read_text(encoding="utf-8"))
    batch = json.loads((A1 / "hawor/BATCH_RESULT.json").read_text(encoding="utf-8"))
    prepared_by_id = {row["session_id"]: row for row in prepared["results"]}
    output.mkdir(parents=True)
    visual.mkdir(parents=True)
    reviews = []
    for row in batch["results"]:
        session_id = row["session_id"]
        source = Path(prepared_by_id[session_id]["prepared_video"]["path"])
        npz_path = Path(row["npz"]["path"])
        destination = visual / f"{session_id}_W1_DIAG_HAWOR_RAW_FULL.mp4"
        item = render(source, npz_path, destination, session_id, int(row["frame_count"]), row["per_side_strict"])
        reviews.append({"session_id": session_id, "session_strict_status": row["session_strict_status"],
                        "per_side_strict": row["per_side_strict"], **item})
    manifest = {
        "schema_version": "0915-robot-recovery-w1-diag-review-manifest-v1",
        "status": "COMPLETE",
        "candidate_signature_sha256": EXPECTED_CANDIDATE,
        "source_a1_result": ref(A1 / "RESULT.json"),
        "renderer": ref(Path(__file__)),
        "reviews": reviews,
        "claim_limit": "Full-timeline diagnostic overlay only; no HaWoR accuracy, R0 quality, control or deployment authority.",
    }
    atomic_json(output / "REVIEW_MANIFEST.json", manifest)
    lines = [
        "# 0915 Robot Recovery V2.1 W1-DIAG HaWoR",
        "",
        "以下视频覆盖完整原始时间轴，只展示同域 raw HaWoR 观测。MISS 不补帧；严格状态保持原判定。",
        "所有视频均为 `OFFLINE_DIAGNOSTIC / NOT R0 QUALITY / NOT CONTROL`。",
        "",
    ]
    for item in reviews:
        name = Path(item["video"]["path"]).name
        lines.append(f"- [{item['session_id']}]({name})：session `{item['session_strict_status']}`；左右 `{item['per_side_strict']}`。")
    (visual / "README_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "review_count": len(reviews)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
