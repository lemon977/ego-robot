#!/usr/bin/env python3
"""Independently reconstruct lossless Clean research frames from sparse lineage."""

from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import artifact_ref, atomic_json, now_iso


def unpack(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.unpackbits(value, bitorder="little", count=shape[0] * shape[1]).reshape(shape).astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.result.resolve(strict=True)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema_version") != "rc1-clean-pixel-closure-shortfilm-v1":
        raise RuntimeError("unsupported source-map result")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    images = output / "lossless_frames"
    images.mkdir()
    manifest = json.loads(Path(result["inputs"]["manifest"]["path"]).read_text(encoding="utf-8"))
    if len(result["pixel_maps"]) > len(manifest["frames"]):
        raise RuntimeError("pixel map exceeds source manifest")
    frame_refs = []
    donor_pixels = 0
    for t, entry in enumerate(result["pixel_maps"]):
        if entry["frame_id"] != t or artifact_ref(Path(entry["map"]["path"])) != entry["map"]:
            raise RuntimeError("map identity mismatch")
        target_ref = manifest["frames"][t]["source_rgb"]
        target_path = Path(target_ref["path"])
        if artifact_ref(target_path) != target_ref:
            raise RuntimeError("target Raw SHA mismatch")
        raw = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if raw is None:
            raise RuntimeError("target Raw decode failed")
        with np.load(entry["map"]["path"], allow_pickle=False) as arr:
            shape = tuple(int(v) for v in arr["shape"])
            if shape != raw.shape[:2] or not np.array_equal(arr["transform"], [1, 0, 0, 0, 1, 0]):
                raise RuntimeError("shape/transform mismatch")
            m_write = unpack(arr["m_write"], shape)
            m_obj = unpack(arr["m_visible_object"], shape)
            yy = arr["candidate_y"].astype(np.intp)
            xx = arr["candidate_x"].astype(np.intp)
            sy = arr["source_y"].astype(np.intp)
            sx = arr["source_x"].astype(np.intp)
            sf = arr["source_frame"].astype(np.intp)
            source_class = arr["source_class"]
            reasons = arr["reject_reason"]
            if not (len(yy) == len(xx) == len(sy) == len(sx) == len(sf)):
                raise RuntimeError("sparse source lengths differ")
            if len(yy) and ((yy >= shape[0]).any() or (xx >= shape[1]).any()
                            or (sy >= shape[0]).any() or (sx >= shape[1]).any()
                            or (sf < 0).any() or (sf >= t).any()):
                raise RuntimeError("source coordinate or causal frame violation")
            if len(yy) and (not np.all(m_write[yy, xx]) or np.any(m_obj[yy, xx])
                            or not np.all(source_class[yy, xx] == 3)):
                raise RuntimeError("source-class/Mask violation")
            frame = raw.copy()
            unknown = m_write & (source_class != 3)
            frame[unknown] = (127, 127, 127)
            for source_id in np.unique(sf):
                select = sf == source_id
                donor_ref = manifest["frames"][int(source_id)]["source_rgb"]
                donor_path = Path(donor_ref["path"])
                if artifact_ref(donor_path) != donor_ref:
                    raise RuntimeError("donor Raw SHA mismatch")
                donor = cv2.imread(str(donor_path), cv2.IMREAD_COLOR)
                if donor is None or donor.shape != raw.shape:
                    raise RuntimeError("donor Raw decode/shape mismatch")
                frame[yy[select], xx[select]] = donor[sy[select], sx[select]]
            if np.any(frame[~m_write] != raw[~m_write]) or np.any(frame[m_obj] != raw[m_obj]):
                raise RuntimeError("byte-exact region violation")
            if np.any((reasons != 0) & (source_class == 3)):
                raise RuntimeError("accepted donor has a rejection reason")
        png = images / f"{t:06d}.png"
        if not cv2.imwrite(str(png), frame):
            raise RuntimeError("lossless frame write failed")
        donor_pixels += len(yy)
        frame_refs.append({"frame_id": t, "png": artifact_ref(png), "map": entry["map"]})
    receipt = {
        "schema_version": "rc1-clean-pixel-closure-audit-v1", "created_at": now_iso(),
        "status": "PASS_LOSSLESS_LINEAGE_ONLY", "session_id": result["session_id"],
        "frames": len(frame_refs), "candidate_pixels": donor_pixels,
        "m_write_outside_changed_pixels": 0, "visible_object_changed_pixels": 0,
        "input_mode": "OFFLINE_VISUAL", "training_eligible": False,
        "claim_limit": "This proves source coordinates and lossless byte-exact boundaries, not hidden-region semantics or causal upstream Mask.",
        "input_result": artifact_ref(result_path), "frame_refs": frame_refs,
        "code": artifact_ref(Path(__file__)),
    }
    atomic_json(output / "CLOSURE_AUDIT.json", receipt)
    print(json.dumps({"session_id": result["session_id"], "frames": len(frame_refs),
                      "candidate_pixels": donor_pixels, "status": receipt["status"]}))


if __name__ == "__main__":
    main()
