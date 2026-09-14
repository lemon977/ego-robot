#!/usr/bin/env python3
"""Independent deep validator and authority freezer for generic real donors."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from tools import run_generic_same_session_real_donor_v1 as producer


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.result.resolve(strict=True)
    result = producer.load(result_path)
    if result.get("status") != "TERMINAL_GRADE_B" or result.get("consumption_authorized") is not True:
        raise RuntimeError("producer result is not Grade-B authorized")
    manifest_path = producer.checked(result["artifacts"]["source_map_manifest"], "source manifest")
    manifest = producer.load(manifest_path)
    spec_path = producer.checked(result["pins"]["input_spec"], "input spec")
    validated = producer.validate_spec(spec_path, Path("/tmp/nonexistent-validator-output"))
    frames = validated["frames"]
    if manifest.get("frame_count") != len(frames) or len(manifest.get("frames", [])) != len(frames):
        raise RuntimeError("manifest full-frame closure failed")
    rgbs, removals, objects = [], [], []
    for index, row in enumerate(frames):
        rgbs.append(producer.read_rgb(row["source_rgb"], f"frame {index}.rgb"))
        removals.append(producer.read_mask(row["clean_removal_object_protected"], f"frame {index}.removal"))
        objects.append(producer.protected(row, f"frame {index}"))
    totals = {"supported": 0, "unsupported": 0, "changed": 0}
    for target, row in enumerate(manifest["frames"]):
        if row.get("frame_id") != target:
            raise RuntimeError(f"frame order mismatch at {target}")
        clean_path = producer.checked(row["clean_rgb"], f"frame {target}.clean")
        source_path = producer.checked(row["pixel_source_map"], f"frame {target}.source")
        clean = np.asarray(Image.open(clean_path).convert("RGB"))
        with np.load(source_path, allow_pickle=False) as archive:
            required = {"frame_id", "source_kind", "source_frame", "source_x", "source_y", "source_depth_m", "support_count", "component_id", "source_eye"}
            if set(archive.files) != required or int(archive["frame_id"]) != target:
                raise RuntimeError(f"frame {target}: provenance schema/identity mismatch")
            kind = archive["source_kind"]
            source_frame = archive["source_frame"]
            source_x, source_y = archive["source_x"], archive["source_y"]
            support = archive["support_count"]
        if kind.shape != removals[target].shape or not set(np.unique(kind)).issubset({0, 1, 2, 3}):
            raise RuntimeError(f"frame {target}: source-kind domain mismatch")
        donor = kind == producer.SOURCE_DONOR
        unsupported = kind == producer.SOURCE_UNSUPPORTED
        protected = kind == producer.SOURCE_PROTECTED
        target_raw = kind == producer.SOURCE_TARGET
        if np.any(donor & ~removals[target]) or np.any(unsupported & ~removals[target]):
            raise RuntimeError(f"frame {target}: donor/unsupported outside removal")
        if np.any(objects[target] & ~protected):
            raise RuntimeError(f"frame {target}: protected object provenance missing")
        if not np.array_equal(clean[target_raw | unsupported | protected], rgbs[target][target_raw | unsupported | protected]):
            raise RuntimeError(f"frame {target}: raw-byte preservation failed")
        yy, xx = np.nonzero(donor)
        if len(xx):
            sf = source_frame[yy, xx]
            if np.any(sf < 0) or np.any(sf >= len(frames)) or np.any(sf == target):
                raise RuntimeError(f"frame {target}: donor frame domain failed")
            if np.any(source_x[yy, xx] != xx) or np.any(source_y[yy, xx] != yy) or np.any(support[yy, xx] < 2):
                raise RuntimeError(f"frame {target}: integer coordinate/support contract failed")
            for donor_frame in np.unique(sf):
                select = sf == donor_frame
                sy, sx = yy[select], xx[select]
                if np.any(removals[int(donor_frame)][sy, sx]) or np.any(objects[int(donor_frame)][sy, sx]):
                    raise RuntimeError(f"frame {target}: donor source is not clear")
                if not np.array_equal(clean[sy, sx], rgbs[int(donor_frame)][sy, sx]):
                    raise RuntimeError(f"frame {target}: selected donor RGB is not byte exact")
        changed = np.any(clean != rgbs[target], axis=2)
        if np.any(changed & ~removals[target]) or np.any(changed & objects[target]):
            raise RuntimeError(f"frame {target}: change-domain gate failed")
        totals["supported"] += int(donor.sum()); totals["unsupported"] += int(unsupported.sum()); totals["changed"] += int(changed.sum())
    gates = {
        "producer_result_and_manifest_sha_exact": "PASS",
        "input_spec_and_all_source_artifacts_sha_exact": "PASS",
        "full_frame_identity_and_provenance_schema": "PASS",
        "same_session_source_frame_domain": "PASS",
        "two_or_more_support_and_integer_coordinate": "PASS",
        "donor_source_removal_and_object_clear": "PASS",
        "selected_donor_rgb_byte_exact": "PASS",
        "unsupported_target_raw_byte_exact": "PASS",
        "protected_object_target_raw_byte_exact": "PASS",
        "changed_only_inside_target_removal": "PASS",
    }
    payload = {
        "schema_version": "generic-same-session-real-donor-authority-v1", "created_at": now(),
        "status": "PASS_INDEPENDENT_DEEP_VALIDATION", "grade": "B", "downstream_authorized": True,
        "task": result["task"], "session": result["session"], "frame_count": len(frames),
        "hard_gates": gates, "recomputed_totals": totals,
        "pins": {"producer_result": producer.ref(result_path), "source_map_manifest": producer.ref(manifest_path), "producer": producer.ref(Path(producer.__file__)), "validator": producer.ref(Path(__file__))},
        "authorized_scope": "SYNTHETIC_PROPAINTER_REAL_DONOR_INPUT_ONLY",
        "claim_limit": "Authority covers source_kind=1 exact observed donor pixels only; it does not make synthetic pixels observed truth.",
    }
    write_new(args.authority.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
