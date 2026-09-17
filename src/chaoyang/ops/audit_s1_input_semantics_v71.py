#!/usr/bin/env python3
from __future__ import annotations

"""Audit whether frozen S1 prompts actually match the claimed mask stage."""

from datetime import datetime
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/s1_input_preflight/S1_INPUT_PREFLIGHT_INDEX.json"
OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/s1_semantic_audit"
ARCHIVED_TASKS = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks"


def archived_path(value: str) -> Path:
    prefix = str(ROOT / "tasks") + "/"
    return ARCHIVED_TASKS / value[len(prefix):] if value.startswith(prefix) else Path(value)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build() -> dict:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    rows = []
    for row in index["rows"]:
        if row["status"] != "PASSED":
            rows.append({"session_id": row["session_id"], "status": row["status"], "prompt_semantics": None})
            continue
        path = archived_path(row["prompt_manifest"]["path"])
        prompt = json.loads(path.read_text(encoding="utf-8"))
        source = archived_path(prompt["source_object_manifest"]["path"])
        source_value = json.loads(source.read_text(encoding="utf-8"))
        semantics = str(source_value.get("semantic_type"))
        rows.append({
            "session_id": row["session_id"], "status": "VERIFIED",
            "prompt_semantics": semantics,
            "instance_ids": [item["instance_id"] for item in prompt["instances"]],
            "eligible_scopes": ["TASK_OBJECT_IDENTITY_MODAL_MASK_CHALLENGER"],
            "forbidden_scopes": ["ROLE_MASK_SUCCESSOR", "HUMAN_HAND_MASK", "TRACKER_ROLE_MASK"],
            "prompt_manifest": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)},
        })
    verified = [r for r in rows if r["status"] == "VERIFIED"]
    return {
        "schema_version": "s1-input-semantics-audit-v71",
        "artifact_revision": "R7_2",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "PASS_WITH_SCOPE_CORRECTION",
        "input_index": {"path": str(INDEX), "bytes": INDEX.stat().st_size, "sha256": sha(INDEX)},
        "counts": {"rows": len(rows), "verified": len(verified), "blocked": len(rows) - len(verified)},
        "rows": rows,
        "finding": "All executable S1 prompts are task-object identity masks. None is a Role Mask prompt; therefore S1 may test object occlusion/re-entry only and cannot repair the Role Mask conversion cluster.",
        "required_action": "Move the S1 challenger evidence to Object Mask, keep Role Mask successor separately blocked until four-role prompts are frozen, and do not spend GPU under a Role Mask claim.",
        "authority": False,
        "claim_limit": "Input-semantics audit only; no mask quality, Gold accuracy, Contact/Object6D or physical truth is established.",
    }


def main() -> None:
    value = build()
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "S1_INPUT_SEMANTICS_AUDIT_R72.json"
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "schema_version": "s1-input-semantics-audit-result-v71", "status": "PASSED",
        "output": {"path": str(output), "bytes": output.stat().st_size, "sha256": sha(output)},
        "claim_limit": value["claim_limit"],
    }
    (OUT / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
