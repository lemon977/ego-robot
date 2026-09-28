#!/usr/bin/env python3
"""Close path/bytes/SHA references used by the S2 H6/H9 evidence set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso


TASK = "human_to_robot_evidence_unlock_s2_20260923"
ROOT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
OUTPUT = ROOT / "evidence_ref_audit/attempt_0001/RESULT.json"
SEEDS = (
    ROOT / "checkpoints/H3_RESULT.json",
    ROOT / "h6_readiness/attempt_0001/RESULT.json",
    ROOT / "formal_entry_resume/attempt_0002/RESULT.json",
    ROOT / "lanes/sensor/reuse_audit_v1/RESULT.json",
    ROOT / "lanes/compare/reuse_audit_v1/RESULT.json",
    ROOT / "lanes/scene/h3_occlusion_031/attempt_0003/RESULT.json",
    ROOT / "lanes/motion_product/formal_product_007/attempt_0002/PRODUCT_RESULT.json",
    ROOT / "lanes/motion_product/formal_product_031/attempt_0004/PRODUCT_RESULT.json",
    ROOT
    / "lanes/motion_product/formal_product_get_potato_chips_0902_103/attempt_0002/PRODUCT_RESULT.json",
    ROOT
    / "lanes/motion_product/formal_product_play_cards_0902_042/attempt_0002/PRODUCT_RESULT.json",
)


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError(f"IMMUTABLE_EVIDENCE_REF_AUDIT_EXISTS:{OUTPUT}")
    checked: dict[tuple[str, int, str], dict[str, object]] = {}
    errors: list[dict[str, object]] = []

    def walk(value: object, location: str) -> None:
        if isinstance(value, dict):
            required = {"path", "bytes", "sha256"}
            if required.issubset(value) and isinstance(value["path"], str):
                raw_path = str(value["path"])
                if Path(raw_path).is_absolute():
                    key = (raw_path, int(value["bytes"]), str(value["sha256"]))
                    if key not in checked:
                        path = Path(raw_path)
                        record: dict[str, object] = {
                            "path": raw_path,
                            "expected_bytes": key[1],
                            "expected_sha256": key[2],
                            "first_reference": location,
                        }
                        if not path.is_file():
                            record["status"] = "MISSING"
                            errors.append(record)
                        else:
                            data = path.read_bytes()
                            actual_sha = hashlib.sha256(data).hexdigest()
                            record.update(actual_bytes=len(data), actual_sha256=actual_sha)
                            record["status"] = (
                                "PASS"
                                if len(data) == key[1] and actual_sha == key[2]
                                else "DRIFT"
                            )
                            if record["status"] != "PASS":
                                errors.append(record)
                        checked[key] = record
            for key, child in value.items():
                walk(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{location}[{index}]")

    for seed in SEEDS:
        walk(load_json(seed), str(seed))

    result = {
        "schema_version": "HUMAN_TO_ROBOT_S2_EVIDENCE_REF_AUDIT_V1",
        "task_id": TASK,
        "created_at": now_iso(),
        "status": "PASS" if not errors else "FAIL",
        "seed_count": len(SEEDS),
        "seeds": [artifact_ref(path) for path in SEEDS],
        "unique_absolute_artifact_refs": len(checked),
        "passed_refs": sum(row["status"] == "PASS" for row in checked.values()),
        "error_count": len(errors),
        "errors": errors,
        "claim_limit": (
            "Path/bytes/SHA closure only. This does not promote algorithm quality, "
            "visual acceptance, Contact, control, training or deployment authority."
        ),
    }
    atomic_json(OUTPUT, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "seed_count": result["seed_count"],
                "unique_refs": result["unique_absolute_artifact_refs"],
                "errors": result["error_count"],
                "output": artifact_ref(OUTPUT),
            },
            ensure_ascii=False,
        )
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
