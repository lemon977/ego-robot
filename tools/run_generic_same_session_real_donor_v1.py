#!/usr/bin/env python3
"""Conservative generic same-session exact-pixel donor producer for Clean.

For every target removal pixel, the producer searches symmetric temporal
offsets at the identical integer pixel coordinate.  A donor is accepted only
when its removal/object masks are clear and a second independently selected
frame agrees in RGB within the frozen L-infinity threshold.  The selected RGB
sample is copied byte-exactly; unsupported pixels remain target-raw byte-exact.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


OFFSETS = (-1, 1, -2, 2, -4, 4, -8, 8, -16, 16, -32, 32, -64, 64)
CONSENSUS_LINF = 12
SOURCE_TARGET = np.uint8(0)
SOURCE_DONOR = np.uint8(1)
SOURCE_PROTECTED = np.uint8(2)
SOURCE_UNSUPPORTED = np.uint8(3)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(value.dtype.str.encode() + b"\0")
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"regular non-symlink file required: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def checked(reference: dict[str, Any], label: str) -> Path:
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    actual = ref(path)
    if any(actual[key] != reference.get(key) for key in ("path", "bytes", "sha256")):
        raise RuntimeError(f"{label}: path/bytes/SHA drift")
    return path


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_rgb(reference: dict[str, Any], label: str) -> np.ndarray:
    path = checked(reference, label)
    value = np.asarray(Image.open(path).convert("RGB"))
    if value.shape != (960, 1280, 3):
        raise RuntimeError(f"{label}: expected 1280x960 RGB")
    return value


def read_mask(reference: dict[str, Any], label: str) -> np.ndarray:
    path = checked(reference, label)
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None or value.shape != (960, 1280) or not set(np.unique(value)).issubset({0, 255}):
        raise RuntimeError(f"{label}: binary 1280x960 mask required")
    return value > 0


def protected(row: dict[str, Any], label: str) -> np.ndarray:
    value = np.zeros((960, 1280), bool)
    for key, reference in row.items():
        if key.startswith("physical_object_") and isinstance(reference, dict):
            value |= read_mask(reference, f"{label}.{key}")
    return value


def build_frame(
    target: int,
    rgbs: list[np.ndarray],
    removals: list[np.ndarray],
    objects: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    shape = removals[target].shape
    target_removal = removals[target]
    first_rgb = np.zeros((*shape, 3), np.uint8)
    first_frame = np.full(shape, -1, np.int32)
    have_first = np.zeros(shape, bool)
    supported = np.zeros(shape, bool)
    support_count = np.zeros(shape, np.uint8)
    for offset in OFFSETS:
        donor = target + offset
        if donor < 0 or donor >= len(rgbs):
            continue
        eligible = target_removal & ~removals[donor] & ~objects[donor] & ~supported
        if not eligible.any():
            continue
        new_first = eligible & ~have_first
        first_rgb[new_first] = rgbs[donor][new_first]
        first_frame[new_first] = donor
        have_first[new_first] = True
        compare = eligible & have_first & (first_frame != donor)
        difference = np.max(
            np.abs(rgbs[donor].astype(np.int16) - first_rgb.astype(np.int16)), axis=2
        )
        agree = compare & (difference <= CONSENSUS_LINF)
        supported[agree] = True
        support_count[agree] = 2
    clean = rgbs[target].copy()
    clean[supported] = first_rgb[supported]
    unsupported = target_removal & ~supported
    source_kind = np.full(shape, SOURCE_TARGET, np.uint8)
    source_kind[supported] = SOURCE_DONOR
    source_kind[unsupported] = SOURCE_UNSUPPORTED
    source_kind[objects[target]] = SOURCE_PROTECTED
    yy, xx = np.indices(shape, dtype=np.int32)
    source_frame = np.full(shape, target, np.int32)
    source_frame[supported] = first_frame[supported]
    provenance = {
        "frame_id": np.int32(target),
        "source_kind": source_kind,
        "source_frame": source_frame,
        "source_x": xx,
        "source_y": yy,
        "source_depth_m": np.full(shape, np.nan, np.float32),
        "support_count": support_count,
        "component_id": np.full(shape, -1, np.int16),
        "source_eye": np.zeros(shape, np.uint8),
    }
    mismatch = int(np.count_nonzero(clean[supported] != first_rgb[supported]))
    changed = np.any(clean != rgbs[target], axis=2)
    metrics = {
        "removal_pixels": int(target_removal.sum()),
        "protected_object_pixels": int(objects[target].sum()),
        "supported_donor_pixels": int(supported.sum()),
        "unsupported_pixels": int(unsupported.sum()),
        "supported_fraction": float(supported.sum() / max(target_removal.sum(), 1)),
        "changed_pixels": int(changed.sum()),
        "changed_outside_removal_pixels": int((changed & ~target_removal).sum()),
        "changed_protected_object_pixels": int((changed & objects[target]).sum()),
        "exact_selected_donor_mismatch_values": mismatch,
    }
    return clean, provenance, metrics


def validate_spec(spec_path: Path, output_override: Path | None) -> dict[str, Any]:
    spec_path = spec_path.resolve(strict=True)
    spec = load(spec_path)
    if spec.get("schema_version") != "exact78-real-donor-clean-stage-input-v1":
        raise RuntimeError("unexpected real-donor spec schema")
    if spec.get("method_contract", {}).get("donor_scope") != "SAME_SESSION_SELECTED_RGB_ONLY":
        raise RuntimeError("same-session donor scope required")
    if spec.get("method_contract", {}).get("generative_fill") is not False:
        raise RuntimeError("real-donor producer cannot be generative")
    manifest_path = checked(spec["inputs"]["expanded_role_mask_manifest"], "expanded manifest")
    manifest = load(manifest_path)
    frames = manifest.get("frames", [])
    if len(frames) != spec.get("frame_count") or manifest.get("session") != spec.get("session"):
        raise RuntimeError("frame/session closure failed")
    checked(spec["inputs"]["raw_video"], "raw video lineage")
    checked(spec["inputs"]["role_mask_result"], "role result")
    checked(spec["inputs"]["task_object_result"], "task-object result")
    checked(spec["inputs"]["object6d_result"], "object6d result")
    output = (output_override or Path(spec["output_root"])).resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh output required: {output}")
    return {"spec": spec, "spec_path": spec_path, "manifest": manifest, "manifest_path": manifest_path, "frames": frames, "output": output}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    validated = validate_spec(args.spec, args.output_root)
    if args.validate_only:
        print(json.dumps({
            "status": "PASS_GENERIC_REAL_DONOR_VALIDATE_ONLY",
            "task": validated["spec"]["task"], "session": validated["spec"]["session"],
            "frame_count": len(validated["frames"]), "spec": ref(validated["spec_path"]),
            "output_absent": True,
        }, ensure_ascii=False, indent=2))
        return 0
    frames = validated["frames"]
    rgbs, removals, objects = [], [], []
    for index, row in enumerate(frames):
        if row.get("source_frame") != index or row.get("published_removal_object_overlap_pixels") != 0:
            raise RuntimeError(f"frame {index}: identity/object-overlap gate failed")
        rgbs.append(read_rgb(row["source_rgb"], f"frame {index}.source_rgb"))
        removals.append(read_mask(row["clean_removal_object_protected"], f"frame {index}.removal"))
        objects.append(protected(row, f"frame {index}"))
        if np.any(removals[-1] & objects[-1]):
            raise RuntimeError(f"frame {index}: removal/object overlap")
    output = validated["output"]
    output.mkdir(parents=True)
    clean_root, source_root = output / "clean_frames", output / "pixel_sources"
    clean_root.mkdir(); source_root.mkdir()
    rows, totals = [], {"removal": 0, "supported": 0, "unsupported": 0, "protected": 0}
    for target in range(len(frames)):
        clean, provenance, metrics = build_frame(target, rgbs, removals, objects)
        clean_path = clean_root / f"{target:06d}.png"
        Image.fromarray(clean).save(clean_path)
        source_path = source_root / f"{target:06d}.npz"
        np.savez_compressed(source_path, **provenance)
        for key, source_key in (("removal", "removal_pixels"), ("supported", "supported_donor_pixels"), ("unsupported", "unsupported_pixels"), ("protected", "protected_object_pixels")):
            totals[key] += metrics[source_key]
        rows.append({
            "frame_id": target,
            "clean_rgb": ref(clean_path),
            "pixel_source_map": ref(source_path),
            "source_rgb_decoded_sha256": array_sha256(rgbs[target]),
            "clean_rgb_decoded_sha256": array_sha256(clean),
            **metrics,
        })
    gates = {
        "same_session_only": True,
        "two_independent_clear_donors_agree_linf_le_12": True,
        "selected_donor_rgb_byte_exact": all(row["exact_selected_donor_mismatch_values"] == 0 for row in rows),
        "changed_only_inside_removal": all(row["changed_outside_removal_pixels"] == 0 for row in rows),
        "protected_object_byte_exact": all(row["changed_protected_object_pixels"] == 0 for row in rows),
        "unsupported_target_raw_byte_exact": True,
        "full_frame_provenance": len(rows) == len(frames),
    }
    manifest_path = output / "SOURCE_MAP_MANIFEST.json"
    write_new(manifest_path, {
        "schema_version": "generic-same-session-real-donor-source-manifest-v1",
        "created_at": now(), "task": validated["spec"]["task"], "session": validated["spec"]["session"],
        "frame_count": len(rows),
        "method": "IDENTICAL_INTEGER_COORDINATE_TWO_TEMPORAL_DONOR_RGB_CONSENSUS",
        "temporal_offsets": list(OFFSETS), "consensus_linf": CONSENSUS_LINF,
        "source_kind_codes": {"0": "TARGET_RAW", "1": "SAME_SESSION_TEMPORAL_RAW", "2": "PROTECTED_OBJECT_RAW", "3": "UNSUPPORTED_RAW"},
        "source_lineage": {"input_spec": ref(validated["spec_path"]), "expanded_manifest": ref(validated["manifest_path"])},
        "frames": rows,
    })
    grade = "B" if all(gates.values()) else "C"
    result_path = output / "RESULT.json"
    write_new(result_path, {
        "schema_version": "generic-same-session-real-donor-result-v1", "created_at": now(),
        "status": f"TERMINAL_GRADE_{grade}", "grade": grade, "terminal": True,
        "stage": "CLEAN_REAL_DONOR", "task": validated["spec"]["task"], "session": validated["spec"]["session"],
        "consumption_authorized": grade == "B", "frame_count": len(rows), "totals": totals,
        "hard_gates": gates, "artifacts": {"source_map_manifest": ref(manifest_path)},
        "pins": {"input_spec": ref(validated["spec_path"]), "runner": ref(Path(__file__))},
        "claim_limit": "Only source_kind=1 pixels are observed same-session donor RGB. Unsupported pixels remain raw and require explicit synthetic processing downstream.",
    })
    write_new(output / "AGENT_REVIEW.json", {
        "schema_version": "baseline-agent-stage-review-v1", "created_at": now(),
        "stage": "CLEAN_REAL_DONOR", "task": validated["spec"]["task"], "session": validated["spec"]["session"],
        "grade": grade, "downstream_authorized": grade == "B",
        "hard_gates": {key: "PASS" if value else "FAIL" for key, value in gates.items()}, "result": ref(result_path),
    })
    print(json.dumps(load(result_path), ensure_ascii=False, indent=2))
    return 0 if grade == "B" else 2


if __name__ == "__main__":
    raise SystemExit(main())
