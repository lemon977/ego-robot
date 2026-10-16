"""Explicit wrapper-only publication; immutable V1 algorithm manifest unchanged."""
from __future__ import annotations

from pathlib import Path

from chaoyang.governance.common import artifact_ref, load_json

WRAPPERS = frozenset({"src/chaoyang/cli/__init__.py",
                     "src/chaoyang/governance/current_r3_contracts.py",
                     "src/chaoyang/governance/inpainting_v1_selection.py"})
HELPER = "src/chaoyang/inpainting_candidate/publication.py"


def runtime_publication(root: Path, manifest: dict) -> tuple[list[dict], list[dict], dict]:
    """Allow only three explicitly signed wrappers, never model/asset/data edits."""
    path = root / "inpainting_v1/GOVERNANCE_PUBLICATION_R03.json"
    value = load_json(path)
    parent_path = root / "inpainting_v1/GOVERNANCE_PUBLICATION_R02.json"
    parent = load_json(parent_path)
    original = {ref["path"]: ref for ref in manifest["runtime_closure"]}
    if (value.get("schema") != "inpainting-v1-runtime-publication-r03"
            or value.get("baseline_id") != manifest["baseline_id"]
            or value.get("baseline_pointer") != artifact_ref(root / "inpainting_v1/MANIFEST.json")
            or value.get("parent_supplement") != artifact_ref(parent_path)
            or parent.get("baseline_pointer") != value["baseline_pointer"]
            or value.get("algorithm_changed") is not False
            or any(value.get(key) != 0 for key in ("files_deleted", "model_calls", "training_steps", "external_writes"))):
        raise RuntimeError("INPAINTING_R03_PUBLICATION_BINDING_DRIFT")
    refs = value.get("wrapper_overrides", [])
    if {row.get("relative_path") for row in refs} != WRAPPERS or len(refs) != len(WRAPPERS):
        raise RuntimeError("INPAINTING_R03_EXACT_WRAPPERS_REQUIRED")
    replacements = {}
    for row in refs:
        relative = row["relative_path"]
        expected_original = original.get(str(root / relative))
        previous = parent["current_wrapper"] if relative.endswith("inpainting_v1_selection.py") else expected_original
        if (expected_original is None or row.get("original") != expected_original
                or row.get("parent") != previous or row.get("current") != artifact_ref(root / relative)
                or row.get("role") != "GOVERNANCE_RUNTIME_ONLY"):
            raise RuntimeError("INPAINTING_R03_WRAPPER_SHA_DRIFT:" + relative)
        replacements[expected_original["path"]] = row["current"]
    if parent.get("original_wrapper") != original.get(str(root / "src/chaoyang/governance/inpainting_v1_selection.py")):
        raise RuntimeError("INPAINTING_R03_PARENT_WRAPPER_DRIFT")
    helper = artifact_ref(root / HELPER)
    if value.get("additional_runtime") != [helper]:
        raise RuntimeError("INPAINTING_R03_HELPER_CLOSURE_DRIFT")
    task_ref = value.get("task_packet")
    if not isinstance(task_ref, dict) or artifact_ref(Path(task_ref["path"])) != task_ref:
        raise RuntimeError("INPAINTING_R03_TASK_REF_REQUIRED")
    task = load_json(Path(task_ref["path"]))
    if (not str(task.get("task_id", "")).startswith("inpainting_generalization_g0_publication_")
            or task.get("deletion_allowed") is not False or task.get("algorithm_execution_allowed") is not False):
        raise RuntimeError("INPAINTING_R03_CPU_PUBLICATION_TASK_REQUIRED")
    for relative in (*WRAPPERS, "inpainting_v1/GOVERNANCE_PUBLICATION_R03.json"):
        if relative not in task.get("write_set", []):
            raise RuntimeError("INPAINTING_R03_TASK_WRITE_SCOPE_MISSING:" + relative)
    if any(ref["path"] in replacements for ref in manifest["algorithm_core"]):
        raise RuntimeError("INPAINTING_R03_ALGORITHM_OVERRIDE_FORBIDDEN")
    runtime = [replacements.get(ref["path"], ref) for ref in manifest["runtime_closure"]] + [helper]
    dependencies = [replacements.get(ref["path"], ref) for ref in manifest["dependencies"]]
    return runtime, dependencies, artifact_ref(path)
