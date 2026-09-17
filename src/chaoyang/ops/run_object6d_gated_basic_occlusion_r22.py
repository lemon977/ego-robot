#!/usr/bin/env python3
"""Run the R2.2 basic compositor after a three-instance Object6D gate."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PAYLOAD_TOOL = ROOT / "src/chaoyang/ops/run_basic_occlusion_fullsession_r22.py"


def digest(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    sha = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha.hexdigest()}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def physical_instance_id(result_path: Path, value: dict[str, Any]) -> int:
    """Read an explicit id when present, otherwise bind it to physical_object_N.

    The current Chips Object6D receipt predates the explicit id field, but its
    immutable parent directory is the authority identity carrier.  Refuse any
    path that does not use that exact convention instead of guessing from list
    order.
    """
    explicit = value.get("physical_instance_id")
    if explicit is not None:
        return int(explicit)
    match = re.fullmatch(r"physical_object_([0-9]+)", result_path.parent.name)
    if match is None:
        raise ValueError(
            "Object6D receipt has no physical_instance_id and its parent is not physical_object_N"
        )
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--robot-zbuffer-result", type=Path, required=True)
    parser.add_argument("--depth-result", type=Path, required=True)
    parser.add_argument("--object-mask-manifest", type=Path, required=True)
    parser.add_argument("--object6d-result", type=Path, action="append", required=True)
    parser.add_argument("--raw-video", type=Path, required=True)
    parser.add_argument("--clean-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True)

    if len(args.object6d_result) != 3:
        raise ValueError("Chips requires exactly three independent Object6D results")
    object_refs: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    identities: set[int] = set()
    for path in args.object6d_result:
        resolved = path.resolve(strict=True)
        value = json.loads(resolved.read_text(encoding="utf-8"))
        if value.get("session_id") != args.session_id:
            raise ValueError("Object6D session identity mismatch")
        if value.get("unobserved_pose_policy") != "KEEP_INVALID" or int(value.get("propagated_frames", -1)) != 0:
            raise ValueError("Object6D must be direct observed-only KEEP_INVALID")
        instance_id = physical_instance_id(resolved, value)
        identities.add(instance_id)
        object_refs.append(digest(resolved))
        object_rows.append({
            "physical_instance_id": instance_id,
            "valid_frames": int(value["valid_frames"]),
            "observed_frames": int(value["observed_frames"]),
            "propagated_frames": int(value["propagated_frames"]),
        })
    if identities != {0, 1, 2}:
        raise ValueError(f"Object6D identities must be {{0,1,2}}, got {identities}")

    payload = output / "payload"
    command = [
        sys.executable, str(PAYLOAD_TOOL),
        "--session-id", args.session_id,
        "--robot-zbuffer-result", str(args.robot_zbuffer_result),
        "--depth-result", str(args.depth_result),
        "--object-mask-manifest", str(args.object_mask_manifest),
        "--raw-video", str(args.raw_video),
        "--clean-result", str(args.clean_result),
        "--output-dir", str(payload),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    (output / "PAYLOAD_STDOUT.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    payload_result_path = payload / "RESULT.json"
    if completed.returncode != 0 or not payload_result_path.is_file():
        raise RuntimeError(f"basic compositor payload failed with returncode {completed.returncode}")
    payload_result = json.loads(payload_result_path.read_text(encoding="utf-8"))
    if payload_result.get("terminal_status") != "PASSED":
        raise RuntimeError("basic compositor payload is not PASSED")

    metrics = {
        "schema_version": "OBJECT6D_GATED_BASIC_OCCLUSION_METRICS_R22",
        "session_id": args.session_id,
        "object6d_gate": "PASSED_DIRECT_OBSERVED_ONLY_THREE_INDEPENDENT_INSTANCES",
        "object6d_instances": sorted(object_rows, key=lambda row: row["physical_instance_id"]),
        "object6d_role": "eligibility and instance-identity gate; pixel ordering uses visible mask plus Stereo optical-Z",
        "payload_metrics": payload_result["metrics"],
        "contact_aware_refinement": "BLOCKED_PREREQ_SAME_SESSION_CONTACT10",
        "accuracy_reported": False,
        "training_eligible": False,
    }
    write_json(output / "METRICS.json", metrics)
    (output / "DECISION.md").write_text(
        "# Object6D-gated基础Occlusion决定\n\n"
        "三份独立Chips Object6D均通过同会话、`DIRECT_OBSERVED_ONLY`、`KEEP_INVALID`"
        "和零传播资格门。基础像素排序仍只使用直接可见Mask和Stereo optical-Z，不借Object6D"
        "隐藏pose补物体。CONTACT-10缺失，因此contact-aware refinement保持阻塞。\n",
        encoding="utf-8",
    )
    write_json(output / "NEXT_ACTION.json", {
        "schema_version": "OBJECT6D_GATED_BASIC_OCCLUSION_NEXT_ACTION_R22",
        "status": "PASSED",
        "next_task_id": "CONTACT10_CHIPS023_THEN_CONTACT_AWARE_REFINEMENT",
    })
    write_json(output / "RUN_RECEIPT.json", {
        "schema_version": "OBJECT6D_GATED_BASIC_OCCLUSION_RUN_RECEIPT_R22",
        "task_id": "OBJECT6D-GATED-BASIC-OCCLUSION-R22",
        "artifact_revision": "R7_2_OBJECT6D_GATED_BASIC_OCCLUSION",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "object6d_inputs": object_refs,
        "payload_result": digest(payload_result_path),
        "producer": digest(Path(__file__)),
        "payload_producer": digest(PAYLOAD_TOOL),
        "authority_promoted": False,
    })
    review = Path(payload_result["review_video"]["path"])
    write_json(output / "ARTIFACT_MANIFEST.json", {
        "schema_version": "OBJECT6D_GATED_BASIC_OCCLUSION_ARTIFACT_MANIFEST_R22",
        "artifacts": [
            digest(payload_result_path), digest(review), digest(output / "METRICS.json"),
            digest(output / "DECISION.md"), digest(output / "NEXT_ACTION.json"),
            digest(output / "RUN_RECEIPT.json"), digest(output / "PAYLOAD_STDOUT.log"),
        ],
        "authority_promoted": False,
    })
    result = {
        "schema_version": "OBJECT6D_GATED_BASIC_OCCLUSION_RESULT_R22",
        "task_id": "OBJECT6D-GATED-BASIC-OCCLUSION-R22",
        "session_id": args.session_id,
        "terminal_status": "PASSED",
        "status": "PASS_DEVELOPMENT_OBJECT6D_GATED_BASIC_OCCLUSION",
        "object6d_gate_pass": True,
        "basic_occlusion_executed": True,
        "contact_aware_refinement_executed": False,
        "review_video": digest(review),
        "payload_result": digest(payload_result_path),
        "authority_promoted": False,
        "control_ground_truth": False,
        "training_eligible": False,
        "claim_limit": "Object6D-gated full-session visible-surface diagnostic; no hidden pose completion, Contact, accuracy, training, control or physical authority.",
    }
    write_json(output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
