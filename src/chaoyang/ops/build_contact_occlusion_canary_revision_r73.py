#!/usr/bin/env python3
from __future__ import annotations

# Portable absolute-script bootstrap for the src layout.
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
if __package__ in {None, ""}:
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))


"""Correct the R7_2 S1 scope from Role Mask to task-object modal Mask."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chaoyang.ops.build_contact_occlusion_canary_revision_r72 import r72_schema, ref


SOURCE = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/revisions/R7_2/CONTACT_OCCLUSION_CANARY_PLAN_V71.json"
SEMANTIC_AUDIT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/s1_semantic_audit/RESULT.json"
OUT = ROOT / "archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/20260914_chaoyang_v71/contact_occlusion_canaries/revisions/R7_3"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build() -> dict[str, Any]:
    value = json.loads(SOURCE.read_text(encoding="utf-8"))
    value["artifact_revision"] = "R7_3"
    value["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    value["supersedes_artifact_id"] = "CONTACT_OCCLUSION_CANARY_PLAN_V71@R7_2"
    s1 = next(item for item in value["canaries"] if item["canary_id"] == "S1")
    s1["objective"] = "Compare causal SAM 2.1 and Cutie task-object identity memory on occlusion and re-entry."
    s1["scope"] = "Task-object modal masks only; Poker card and three Chips instances stay separate. This is not a human/tracker Role Mask successor."
    s1["input_evidence"].append(ref(SEMANTIC_AUDIT, "INPUT_SEMANTICS_SCOPE_CORRECTION"))
    s1["authority_limit"]["forbidden_claims"].append("ROLE_MASK_SUCCESSOR")
    value["claim_limit"] = (
        "R7_3 corrects S1 to task-object modal-mask scope and withdraws any Role Mask implication from R7_2. "
        "No inference result or Mask/Contact/Object6D/Gold/physical authority is claimed."
    )
    value["producer_signature"] = hashlib.sha256(
        (sha(SOURCE) + sha(SEMANTIC_AUDIT) + sha(Path(__file__))).encode()
    ).hexdigest()
    schema = r72_schema()
    schema["properties"]["artifact_revision"] = {"const": "R7_3"}
    import jsonschema
    jsonschema.validate(value, schema)
    return value


def main() -> None:
    value = build()
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "CONTACT_OCCLUSION_CANARY_PLAN_V71.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "schema_version": "contact-occlusion-canary-plan-build-result-v1", "status": "PASSED",
        "artifact_revision": "R7_3", "outputs": [ref(output, "CANARY_PLAN_JSON")],
        "counts": value["counts"], "claim_limit": value["claim_limit"],
    }
    (OUT / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
