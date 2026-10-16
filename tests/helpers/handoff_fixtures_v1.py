"""Small synthetic receipt-bound trees for portable tests, not server receipts.

No on-import I/O, models, chaoyang imports or real dataset dependencies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DOC_NAMES = {"doc_authority_map": "DOC_AUTHORITY_MAP.json", "algorithm_contract": "ALGORITHM_CONTRACT.json",
             "baseline_registry_v2": "CURRENT_BASELINE_REGISTRY_V2.json", "task_packet_index_current": "INDEX.json",
             "task_state": "LONG_HORIZON_TASK_STATE.json"}
WRAPPER = "src/chaoyang/governance/current_r3_contracts.py"
ALGORITHM = "src/chaoyang/pipeline/synthetic_algorithm.py"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return ref(path)


def ref(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def refresh(root, manifest_changed=False, phase_changed=False):
    root = Path(root)
    governance = root / "docs/governance"
    algorithm = load(governance / DOC_NAMES["algorithm_contract"])
    specials = algorithm["special_status_contracts"]
    if phase_changed:
        phase_ref = ref(root / "docs/current/DUAL_BRANCH_BASELINE_V1.json")
        specials["dual_branch_phase_baselines_v1"]["contract"] = phase_ref
        phase = load(phase_ref["path"])
        specials["dual_branch_phase_baselines_v1"]["aliases"] = {
            alias: {key: row[key] for key in ("operation", "parent_baseline_id", "parent_implementation", "pointer")}
            for alias, row in phase["aliases"].items()}
        manifest = load(root / "baseline_v1/MANIFEST.json")
        manifest["phase_contract"] = phase_ref
        write_json(root / "baseline_v1/MANIFEST.json", manifest)
        manifest_changed = True
    if manifest_changed:
        specials["baseline_v1_source_release"]["manifest"] = ref(root / "baseline_v1/MANIFEST.json")
    write_json(governance / DOC_NAMES["algorithm_contract"], algorithm)
    authority = load(governance / DOC_NAMES["doc_authority_map"])
    for row in authority["documents"]:
        actual = specials["dual_branch_phase_baselines_v1"]["contract"] if row["document_id"] == "dual_branch_phase_baselines_v1" else specials["baseline_v1_source_release"]["manifest"]
        row.update(actual)
    write_json(governance / DOC_NAMES["doc_authority_map"], authority)
    # Same generated snapshot, with INDEX allowed to be a prior bound revision.
    for key, name in DOC_NAMES.items():
        path = governance / name
        document = load(path)
        document.setdefault("governance_revision", 1)
        document.setdefault("generation_id", "gov-000001-synthetic")
        write_json(path, document)
    receipt = {"governance_revision": 1, "generation_id": "gov-000001-synthetic",
               "files": {key: ref(governance / name) for key, name in DOC_NAMES.items()}}
    write_json(governance / "CURRENT_STATUS_RECEIPT.json", receipt)


def build(root):
    root = Path(root).resolve()
    rows = []
    for relative, role in ((ALGORITHM, "VISUAL_IMPLEMENTATION"), (WRAPPER, "RUNTIME_PLATFORM_DEPENDENCY_ONLY")):
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# synthetic original\n", encoding="utf-8")
        bundle_path = "frozen/" + relative
        frozen = root / "baseline_v1" / bundle_path
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(source.read_bytes())
        rows.append({"source": ref(source), "bundle_path": bundle_path, "role": role})
    phase = {"phase_id": "SYNTHETIC_PHASE_V1", "selected_alias_count": 2, "aliases": {}}
    for alias, implementation, operation in (("EXACT78_PHASE_V1", "takeover_v13", "run_visual"),
                                              ("MANUS_PHASE_V1", "robot100_v16", "run_manus")):
        pointer = write_json(root / "docs/current" / (alias + ".json"), {"synthetic_only": True})
        phase["aliases"][alias] = {"operation": operation, "parent_baseline_id": alias + "_PARENT",
                                   "parent_implementation": implementation, "pointer": pointer,
                                   "actual_motion_source": "SYNTHETIC_NOT_RGB_GROUND_TRUTH",
                                   "validated_scope": {"synthetic_cases": 1},
                                   "training_eligible": False, "control_ground_truth": False}
    phase_ref = write_json(root / "docs/current/DUAL_BRANCH_BASELINE_V1.json", phase)
    manifest = {"sources": rows, "metadata": [], "counts": {"source_files": 2, "metadata_files": 0},
                "indexes": {}, "algorithm_changed": False, "production_loader_changed": False,
                "phase_contract": phase_ref,
                "selected_operations": {alias: row["operation"] for alias, row in phase["aliases"].items()}}
    manifest_ref = write_json(root / "baseline_v1/MANIFEST.json", manifest)
    governance = root / "docs/governance"
    write_json(governance / DOC_NAMES["algorithm_contract"], {
        "governance_revision": 1, "special_status_contracts": {
            "dual_branch_phase_baselines_v1": {"contract": phase_ref, "aliases": {
                alias: {key: row[key] for key in ("operation", "parent_baseline_id", "parent_implementation", "pointer")}
                for alias, row in phase["aliases"].items()}},
            "baseline_v1_source_release": {"manifest": manifest_ref},
            "exact78_visual_baseline_v1": {"code_closure": []}}})
    write_json(governance / DOC_NAMES["doc_authority_map"], {"governance_revision": 1, "documents": [
        {"document_id": "dual_branch_phase_baselines_v1", "status": "CURRENT", **phase_ref},
        {"document_id": "baseline_v1_source_release", "status": "CURRENT", **manifest_ref}]})
    write_json(governance / DOC_NAMES["baseline_registry_v2"], {"governance_revision": 1, "entries": []})
    write_json(governance / DOC_NAMES["task_packet_index_current"], {"task_packets": []})
    write_json(governance / DOC_NAMES["task_state"], {"tasks": []})
    refresh(root)
    return root


def set_runtime(root, paths, active=False, status="RUNNING"):
    root = Path(root)
    refs = [ref(root / name) for name in paths]
    governance = root / "docs/governance"
    if active:
        packet = {"task_id": "synthetic_candidate", "execution_mode": "CANDIDATE_NOT_BASELINE", "runtime_code": refs}
        packet_ref = write_json(root / "tasks/synthetic/TASK_PACKET.json", packet)
        write_json(governance / DOC_NAMES["task_packet_index_current"], {"task_packets": [{
            "task_id": packet["task_id"], "packet_path": packet_ref["path"], "packet_sha256": packet_ref["sha256"], "execution_allowed": True}]})
        write_json(governance / DOC_NAMES["task_state"], {"tasks": [{"task_id": packet["task_id"], "status": status}]})
    else:
        algorithm = load(governance / DOC_NAMES["algorithm_contract"])
        algorithm["special_status_contracts"]["exact78_visual_baseline_v1"]["code_closure"] = refs
        write_json(governance / DOC_NAMES["algorithm_contract"], algorithm)
    refresh(root)
