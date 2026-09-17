#!/usr/bin/env python3
from __future__ import annotations

"""Publish an immutable R7_2 canary-plan delta after S1 prerequisites closed."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import jsonschema


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/CONTACT_OCCLUSION_CANARY_PLAN_V71.json"
DEFAULT_OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/revisions/R7_2"
SCHEMA = ROOT / "contracts/contact_occlusion_canary_plan_v71.schema.json"
S1_EVIDENCE = [
    (ROOT / "assets/models/sam2_1_hiera_large/ASSET_PIN.json", "SAM2_1_PIN"),
    (ROOT / "assets/models/cutie/ASSET_PIN.json", "CUTIE_PIN"),
    (ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/s1_reentry_audit/RESULT.json", "FROZEN_REENTRY_AUDIT"),
    (ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/s1_input_preflight/RESULT.json", "S1_INPUT_PREFLIGHT"),
    (ROOT / "src/chaoyang/pipeline/causal_modal_mask_gpu_adapter_v71.py", "GPU_ADAPTER"),
    (ROOT / "src/chaoyang/ops/run_post_clean_s1_automation_v71.py", "POST_CLEAN_AUTOMATION"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path), "role": role}


def r72_schema() -> dict[str, Any]:
    """Narrow schema successor: only revision provenance and resource wait are added."""
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    schema["properties"]["artifact_revision"] = {"const": "R7_2"}
    schema["properties"]["supersedes_artifact_id"] = {"type": "string", "minLength": 1}
    schema["required"].append("supersedes_artifact_id")
    dependency_status = schema["$defs"]["dependency"]["properties"]["status"]["enum"]
    if "WAITING_RESOURCE" not in dependency_status:
        dependency_status.append("WAITING_RESOURCE")
    return schema


def build() -> dict[str, Any]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    value = json.loads(json.dumps(source))
    value["artifact_revision"] = "R7_2"
    value["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    value["supersedes_artifact_id"] = f'{source["artifact_id"]}@{source["artifact_revision"]}'
    s1 = next(item for item in value["canaries"] if item["canary_id"] == "S1")
    s1["algorithm_prerequisites"] = [
        {
            "dependency_id": "SAM2_1_LOCAL_PIN",
            "required": True,
            "status": "VERIFIED_LOCAL",
            "reason": "Pinned implementation/config/weights and import preflight are available.",
            "evidence_paths": [str(S1_EVIDENCE[0][0].resolve())],
        },
        {
            "dependency_id": "CUTIE_CODE_AND_WEIGHTS",
            "required": True,
            "status": "VERIFIED_LOCAL",
            "reason": "Pinned code/config/weights/dependencies/license and CPU load smoke are available.",
            "evidence_paths": [str(S1_EVIDENCE[1][0].resolve())],
        },
        {
            "dependency_id": "FROZEN_REENTRY_AUDIT_FRAMES",
            "required": True,
            "status": "VERIFIED_INPUT_ONLY",
            "reason": "Per-instance causal re-entry audit and six executable input manifests are frozen.",
            "evidence_paths": [str(S1_EVIDENCE[2][0].resolve()), str(S1_EVIDENCE[3][0].resolve())],
        },
        {
            "dependency_id": "GPU_EXECUTION_GATE",
            "required": True,
            "status": "WAITING_RESOURCE",
            "reason": "Execution is automatically gated on the four-hour sprint deadline and Clean terminal completion.",
            "evidence_paths": [str(S1_EVIDENCE[5][0].resolve())],
        },
    ]
    s1["input_evidence"].extend(ref(path, role) for path, role in S1_EVIDENCE)
    s1["resource_status"] = "READY_FOR_BOUNDED_RUN"
    s1["blockers"] = []
    value["counts"] = {
        "total": len(value["canaries"]),
        "standard_exact78": sum(c["line"] == "STANDARD_EXACT78" for c in value["canaries"]),
        "controller_manus_sensor": sum(c["line"] == "CONTROLLER_MANUS_SENSOR" for c in value["canaries"]),
        "ready": sum(c["resource_status"] == "READY_FOR_BOUNDED_RUN" for c in value["canaries"]),
        "blocked_prereq": sum(c["resource_status"] == "BLOCKED_PREREQ" for c in value["canaries"]),
    }
    value["claim_limit"] = (
        "R7_2 closes S1 code/weight/input prerequisites and schedules a bounded development run. "
        "No inference result, current Mask authority, Contact/Object6D truth, Gold accuracy or physical authority is claimed."
    )
    value["producer_signature"] = hashlib.sha256(
        (sha256(SOURCE) + sha256(Path(__file__)) + "".join(item["sha256"] for item in s1["input_evidence"])).encode()
    ).hexdigest()
    jsonschema.validate(value, r72_schema())
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    value = build()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "CONTACT_OCCLUSION_CANARY_PLAN_V71.json"
    if output.exists():
        raise FileExistsError(f"immutable R7_2 plan exists: {output}")
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "schema_version": "contact-occlusion-canary-plan-build-result-v1",
        "status": "PASSED",
        "artifact_revision": "R7_2",
        "outputs": [ref(output, "CANARY_PLAN_JSON")],
        "counts": value["counts"],
        "claim_limit": value["claim_limit"],
    }
    (args.output_dir / "RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(value["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
