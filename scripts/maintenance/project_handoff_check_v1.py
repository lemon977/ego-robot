"""Read-only, receipt-bound V1 handoff view; never an algorithm entry point.

Only JSON, source hashes and package metadata are inspected. No chaoyang module,
model, task executor, subprocess or publisher is imported or started.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
ALIASES = frozenset({"EXACT78_PHASE_V1", "MANUS_PHASE_V1"})
WRAPPERS = frozenset({"src/chaoyang/governance/current_r3_contracts.py",
                      "src/chaoyang/governance/exact78_visual_selection_v1.py"})
INPAINTING_ID = "INPAINTING_RAW_TO_ROBOT_V1"
INPAINTING_OPERATION = "run_exact78_pick_place_raw_to_robot_candidate_v1"
INPAINTING_WRAPPER = "src/chaoyang/governance/inpainting_v1_selection.py"
READ_KEYS = ("doc_authority_map", "algorithm_contract", "baseline_registry_v2",
             "task_packet_index_current", "task_state")
GENERATED_KEYS = READ_KEYS[:3] + ("task_state",)
PORTABLE_TESTS = ("tests/maintenance/test_project_handoff_check_v1.py",
                  "tests/maintenance/test_project_cpu_profiles_v1.py")
SERVER_TESTS = ("tests/unit/test_baseline_v1_source_release.py",
                "tests/unit/test_project_slim_v1.py",
                "tests/unit/test_frozen_bundle_runtime_separation_v1.py",
                "tests/unit/test_project_health_readset_v1.py")
SERVER_PACKET = "tasks/current/exact78_pick_place_production_batch04_full_v9_reverse_boundary_revision2_20261009/TASK_PACKET.json"
SERVER_PACKET_SHA = "2931068f54f0a311718716f3d952d0280e82af64d154e6166cf0d38b52ce3420"
SERVER_EVIDENCE = (
    (SERVER_PACKET, SERVER_PACKET_SHA),
    ("_run/current/project_slim_c1_v2_20261008/attempts/attempt_0001/RESULT.json",
     "5153f353e6938ab0e05cbf10016a02b50963781e92ec836a689b8f900c198293"),
    ("_run/current/project_slim_c0_v2_20261008/attempts/attempt_0001/PROOF.json",
     "f025e1e17cc99222fa4c4510687984c722d7039c648cf233abc036b02b69dc5e"),
)


class CheckError(RuntimeError):
    def __init__(self, code, detail=""):
        self.code, self.detail = code, str(detail)
        super().__init__(code + (":" + self.detail if self.detail else ""))


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def project_path(root, value):
    root = Path(root).resolve()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if root not in path.parents or path.resolve() != path:
        raise CheckError("PATH_SCOPE_OR_SYMLINK", path)
    return path


def file_ref(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read_ref(root, ref, code="INPUT_SHA_MISMATCH"):
    try:
        path = project_path(root, ref["path"])
        if not path.is_file():
            raise CheckError("INPUT_ABSENT", path)
        data = path.read_bytes()
        if len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise CheckError(code, path)
        return data
    except (KeyError, TypeError) as error:
        raise CheckError("MALFORMED_ARTIFACT_REF", type(error).__name__) from error


def decode(data):
    try:
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except (ValueError, UnicodeError) as error:
        raise CheckError("MALFORMED_JSON", str(error)[:120]) from error


def current_snapshot(root, attempts=2):
    receipt_path = project_path(root, "docs/governance/CURRENT_STATUS_RECEIPT.json")
    for attempt in range(attempts):
        before = receipt_path.read_bytes()
        receipt = decode(before)
        try:
            docs = {key: decode(read_ref(root, receipt["files"][key])) for key in READ_KEYS}
            revision = receipt["governance_revision"]
            if any(docs[key].get("governance_revision") != revision for key in GENERATED_KEYS):
                raise CheckError("SNAPSHOT_REVISION_CONFLICT", revision)
            generation = receipt["generation_id"]
            if any(docs[key].get("generation_id") != generation for key in GENERATED_KEYS):
                raise CheckError("SNAPSHOT_GENERATION_CONFLICT", generation)
            # INDEX is the last packet publication, not necessarily this heartbeat.
            index = docs["task_packet_index_current"]
            index_revision = index.get("governance_revision")
            index_generation = index.get("generation_id")
            if (type(index_revision) is not int or not 0 < index_revision <= revision
                    or not isinstance(index_generation, str)
                    or not index_generation.startswith(f"gov-{index_revision:06d}-")):
                raise CheckError("INDEX_GENERATION_REVISION_CONFLICT")
        except (CheckError, KeyError) as error:
            if receipt_path.read_bytes() != before:
                if attempt + 1 < attempts:
                    continue
                raise CheckError("RECEIPT_SNAPSHOT_MOVED") from error
            if isinstance(error, KeyError):
                raise CheckError("RECEIPT_SCHEMA_INCOMPLETE", str(error)) from error
            raise
        if receipt_path.read_bytes() == before:
            return receipt, docs
    raise CheckError("RECEIPT_SNAPSHOT_MOVED")


def authority_ref(documents, document_id):
    matches = [row for row in documents["documents"] if row.get("document_id") == document_id
               and row.get("status") == "CURRENT"]
    if len(matches) != 1:
        raise CheckError("CURRENT_AUTHORITY_NOT_UNIQUE", document_id)
    return {key: matches[0][key] for key in ("path", "bytes", "sha256")}


def source_release(root, manifest_ref):
    manifest = decode(read_ref(root, manifest_ref, "SOURCE_MANIFEST_SHA_DRIFT"))
    bundle = project_path(root, manifest_ref["path"]).parent
    rows = manifest["sources"] + manifest["metadata"]
    if len({row["bundle_path"] for row in rows}) != len(rows):
        raise CheckError("SOURCE_RELEASE_DUPLICATE_PATH")
    if (len(manifest["sources"]) != manifest["counts"]["source_files"]
            or len(manifest["metadata"]) != manifest["counts"]["metadata_files"]):
        raise CheckError("SOURCE_RELEASE_COUNT_DRIFT")
    for row in rows:
        relative = Path(row["bundle_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise CheckError("SOURCE_RELEASE_PATH_ESCAPE")
        path = bundle / relative
        read_ref(root, {**row["source"], "path": str(path)}, "SOURCE_RELEASE_SHA_DRIFT")
    for name, ref in manifest.get("indexes", {}).items():
        if Path(name).name != name or project_path(root, ref["path"]) != bundle / name:
            raise CheckError("SOURCE_RELEASE_INDEX_SCOPE")
        read_ref(root, ref, "SOURCE_RELEASE_INDEX_DRIFT")
    if manifest["algorithm_changed"] or manifest["production_loader_changed"]:
        raise CheckError("SOURCE_RELEASE_ROLE_DRIFT")
    return manifest, len(rows)


def current_tasks(root, docs):
    state_rows = docs["task_state"]["tasks"]
    active = [row for row in docs["task_packet_index_current"]["task_packets"] if row.get("execution_allowed") is True]
    if len(active) > 1:
        raise CheckError("CURRENT_TASK_NOT_UNIQUE")
    state = {row["task_id"]: row for row in state_rows}
    packets, summary = [], []
    for row in docs["task_packet_index_current"]["task_packets"]:
        if row.get("execution_allowed") is not True:
            continue
        ref = {"path": str(project_path(root, row["packet_path"])),
               "bytes": project_path(root, row["packet_path"]).stat().st_size,
               "sha256": row["packet_sha256"]}
        packet = decode(read_ref(root, ref, "CURRENT_TASK_PACKET_SHA_DRIFT"))
        if packet["task_id"] != row["task_id"]:
            raise CheckError("CURRENT_TASK_ID_DRIFT")
        status = state.get(row["task_id"], {}).get("status")
        if sum(item["task_id"] == row["task_id"] for item in state_rows) != 1:
            raise CheckError("CURRENT_TASK_NOT_UNIQUE", row["task_id"])
        if status not in {"RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE"}:
            raise CheckError("CURRENT_TASK_STATE_CONFLICT", row["task_id"])
        state_ref = state[row["task_id"]].get("task_packet")
        if state_ref is not None and state_ref != ref:
            raise CheckError("CURRENT_TASK_STATE_PACKET_CONFLICT", row["task_id"])
        for runtime_ref in packet.get("runtime_code", []):
            read_ref(root, runtime_ref, "CURRENT_RUNTIME_SHA_DRIFT")
        packets.append(packet)
        summary.append({"task_id": packet["task_id"], "status": status,
                        "verified_packet": ref, "read_set": packet.get("read_set", "ABSENT"),
                        "commands": packet.get("commands", packet.get("command", "ABSENT")),
                        "writer_declared": packet.get("writer", "ABSENT"),
                        "next_action": state[row["task_id"]].get("next_action", packet.get("next_action", "ABSENT")),
                        "execution_mode_declared": packet.get("execution_mode"),
                        "execution_allowed_declared": True,
                        "writer_liveness": "NOT_CHECKED_BY_READONLY_VIEW",
                        "baseline_adoption": "NOT_INFERRED_FROM_TASK_OR_VIDEO"})
    return packets, summary


def live_sources(root, manifest, algorithm, packets):
    refs = []
    specials = algorithm["special_status_contracts"]
    for key in ("exact78_visual_baseline_v1", "pick_place_1008_candidate_v1",
                "humanego_ict_joint26_prepare_v1", "humanego_ict_joint26_v1"):
        refs.extend(specials.get(key, {}).get("code_closure", []))
    for packet in packets:
        refs.extend(packet.get("runtime_code", []))
    pins = {}
    for ref in refs:
        path = project_path(root, ref["path"])
        read_ref(root, ref, "CURRENT_RUNTIME_SHA_DRIFT")
        pins.setdefault(str(path), []).append(ref)
    differences = []
    for row in manifest["sources"]:
        path = project_path(root, row["source"]["path"])
        if not path.is_file():
            raise CheckError("LIVE_SOURCE_ABSENT", path)
        actual = file_ref(path)
        if all(actual[key] == row["source"][key] for key in ("bytes", "sha256")):
            continue
        relative = str(path.relative_to(Path(root).resolve()))
        if relative not in WRAPPERS or row["role"] != "RUNTIME_PLATFORM_DEPENDENCY_ONLY":
            raise CheckError("LIVE_ALGORITHM_DRIFT", relative)
        if actual not in pins.get(str(path), []):
            raise CheckError("LIVE_WRAPPER_UNBOUND", relative)
        differences.append(relative)
    return differences


def inpainting_r03_runtime(root, manifest, branch, pointer):
    """Check the receipt-selected R03 data without importing production code."""
    frozen, live = manifest["runtime_closure"], branch["runtime_closure"]
    original = {ref["path"]: ref for ref in frozen}
    actual = {ref["path"]: ref for ref in live}
    if len(frozen) != 94 or len(original) != 94 or len(live) != 95 or len(actual) != 95:
        raise CheckError("INPAINTING_RUNTIME_CLOSURE_DRIFT")
    value = decode(read_ref(root, branch["governance_publication_supplement"],
                            "INPAINTING_R03_SUPPLEMENT_SHA_DRIFT"))
    parent_ref = value.get("parent_supplement", {})
    if project_path(root, parent_ref.get("path", "")) != root / "inpainting_v1/GOVERNANCE_PUBLICATION_R02.json":
        raise CheckError("INPAINTING_R03_PARENT_SCOPE")
    parent = decode(read_ref(root, parent_ref, "INPAINTING_R03_PARENT_SHA_DRIFT"))
    wrapper = str(root / INPAINTING_WRAPPER)
    if (value.get("schema") != "inpainting-v1-runtime-publication-r03"
            or value.get("baseline_id") != INPAINTING_ID or value.get("baseline_pointer") != pointer
            or value.get("algorithm_changed") is not False
            or any(type(value.get(key)) is not int or value[key] != 0
                   for key in ("files_deleted", "model_calls", "training_steps", "external_writes"))
            or parent.get("schema") != "inpainting-v1-governance-publication-supplement"
            or parent.get("baseline_id") != INPAINTING_ID or parent.get("baseline_pointer") != pointer
            or parent.get("original_wrapper") != original.get(wrapper)
            or parent.get("algorithm_changed") is not False
            or any(type(parent.get(key)) is not int or parent[key] != 0
                   for key in ("files_deleted", "algorithm_calls"))):
        raise CheckError("INPAINTING_R03_PUBLICATION_BINDING_DRIFT")
    read_ref(root, parent["task_packet"], "INPAINTING_R03_PARENT_PACKET_DRIFT")
    allowed = {INPAINTING_WRAPPER, "src/chaoyang/cli/__init__.py",
               "src/chaoyang/governance/current_r3_contracts.py"}
    overrides = value.get("wrapper_overrides", [])
    if len(overrides) != 3 or {row.get("relative_path") for row in overrides} != allowed:
        raise CheckError("INPAINTING_R03_EXACT_WRAPPERS_REQUIRED")
    replacements = {}
    for row in overrides:
        relative = row["relative_path"]
        path = str(root / relative)
        previous = parent.get("current_wrapper") if relative == INPAINTING_WRAPPER else original.get(path)
        if (row.get("original") != original.get(path) or original.get(path) is None
                or row.get("parent") != previous or row.get("current") != actual.get(path)
                or row.get("role") != "GOVERNANCE_RUNTIME_ONLY"):
            raise CheckError("INPAINTING_R03_WRAPPER_BINDING_DRIFT", relative)
        replacements[path] = row["current"]
    if any(ref["path"] in replacements for ref in manifest["algorithm_core"]):
        raise CheckError("INPAINTING_R03_ALGORITHM_OVERRIDE_FORBIDDEN")
    helper_path = str(root / "src/chaoyang/inpainting_candidate/publication.py")
    helper_ref = actual.get(helper_path)
    if helper_ref is None or value.get("additional_runtime") != [helper_ref]:
        raise CheckError("INPAINTING_R03_HELPER_CLOSURE_DRIFT")
    expected = [replacements.get(ref["path"], ref) for ref in frozen] + [helper_ref]
    if live != expected or branch.get("dependencies") != [replacements.get(ref["path"], ref) for ref in manifest["dependencies"]]:
        raise CheckError("INPAINTING_R03_RUNTIME_BINDING_DRIFT")
    task = decode(read_ref(root, value["task_packet"], "INPAINTING_R03_TASK_SHA_DRIFT"))
    if (not str(task.get("task_id", "")).startswith("inpainting_generalization_g0_publication_")
            or task.get("deletion_allowed") is not False or task.get("algorithm_execution_allowed") is not False
            or not (allowed | {"inpainting_v1/GOVERNANCE_PUBLICATION_R03.json"}).issubset(task.get("write_set", []))):
        raise CheckError("INPAINTING_R03_CPU_PUBLICATION_TASK_REQUIRED")
    return [path for path in original if actual[path] != original[path]]


def inspect_inpainting(root, receipt, docs, alias):
    """Verify the single adopted pipeline without importing its implementation."""
    branch = docs["algorithm_contract"]["special_status_contracts"]["inpainting_raw_to_robot_v1"]
    if alias is not None and alias != INPAINTING_ID:
        raise CheckError("ALIAS_NOT_SELECTED", alias)
    pointer = branch["baseline_pointer"]
    if pointer != authority_ref(docs["doc_authority_map"], "inpainting_raw_to_robot_v1"):
        raise CheckError("INPAINTING_AUTHORITY_REF_CONFLICT")
    if project_path(root, pointer["path"]) != root / "inpainting_v1/MANIFEST.json":
        raise CheckError("INPAINTING_POINTER_SCOPE")
    manifest = decode(read_ref(root, pointer, "INPAINTING_MANIFEST_SHA_DRIFT"))
    adoption_ref = branch["publication_receipt"]
    adoption = decode(read_ref(root, adoption_ref, "INPAINTING_ADOPTION_SHA_DRIFT"))
    if project_path(root, adoption_ref["path"]) != root / "tasks/receipts/INPAINTING_RAW_TO_ROBOT_V1.json":
        raise CheckError("INPAINTING_ADOPTION_SCOPE")
    authority = next(row for row in docs["doc_authority_map"]["documents"]
                     if row.get("document_id") == "inpainting_raw_to_robot_v1"
                     and row.get("status") == "CURRENT")
    if adoption_ref not in authority.get("evidence_receipts", []):
        raise CheckError("INPAINTING_AUTHORITY_ADOPTION_CONFLICT")
    if (adoption.get("baseline_id") != INPAINTING_ID or adoption.get("status") != "PASSED"
            or adoption.get("baseline_pointer") != pointer
            or adoption.get("adoption_only") is not True
            or adoption.get("control_ground_truth") is not False
            or adoption.get("all199_visual_accepted") is not False):
        raise CheckError("INPAINTING_ADOPTION_BINDING_DRIFT")
    if (manifest.get("baseline_id") != INPAINTING_ID or branch.get("baseline_id") != INPAINTING_ID
            or manifest.get("selected_operation") != INPAINTING_OPERATION
            or branch.get("selected_operation") != INPAINTING_OPERATION
            or manifest.get("selected_baseline_count") != 1 or branch.get("selected_baseline_count") != 1
            or manifest.get("actual_motion_source") != "RAW_CONTROLLER_HEAD_PLUS_RAW_MANUS"
            or branch.get("actual_motion_source") != manifest.get("actual_motion_source")
            or manifest.get("scope", {}).get("completed_sessions") != 199
            or manifest.get("scope", {}).get("native_frames") != 29322
            or branch.get("validated_scope") != manifest.get("scope")
            or manifest.get("quality", {}).get("all199_visual_accepted") is not False
            or manifest.get("quality", {}).get("control_ground_truth") is not False
            or branch.get("quality") != manifest.get("quality")
            or branch.get("control_ground_truth") is not False
            or branch.get("training_eligible") is not False
            or branch.get("task_packet_required") is not True
            or branch.get("current_runnable_operation") is not None):
        raise CheckError("INPAINTING_SELECTION_DRIFT")
    for key in ("algorithm_calls", "training_steps", "files_deleted", "external_writes"):
        if (type(manifest.get("adoption_sideeffects", {}).get(key)) is not int
                or manifest["adoption_sideeffects"][key] != 0
                or type(adoption.get(key)) is not int or adoption[key] != 0):
            raise CheckError("INPAINTING_ADOPTION_SIDE_EFFECT", key)
    core = manifest["algorithm_core"]
    if (len(core) != 19 or len({ref["path"] for ref in core}) != 19
            or branch.get("code_closure") != core):
        raise CheckError("INPAINTING_CORE_CLOSURE_DRIFT")
    for ref in core:
        read_ref(root, ref, "INPAINTING_CORE_SHA_DRIFT")
    frozen, live = manifest["runtime_closure"], branch["runtime_closure"]
    supplement_ref = branch.get("governance_publication_supplement")
    if (supplement_ref is not None and project_path(root, supplement_ref["path"])
            == root / "inpainting_v1/GOVERNANCE_PUBLICATION_R03.json"):
        changed = inpainting_r03_runtime(root, manifest, branch, pointer)
        live_by_path = {ref["path"]: ref for ref in live}
    else:
        if (len(frozen) != 94 or len(live) != 94
                or len({ref["path"] for ref in frozen}) != 94
                or len({ref["path"] for ref in live}) != 94
                or {ref["path"] for ref in frozen} != {ref["path"] for ref in live}):
            raise CheckError("INPAINTING_RUNTIME_CLOSURE_DRIFT")
        frozen_by_path = {ref["path"]: ref for ref in frozen}
        live_by_path = {ref["path"]: ref for ref in live}
        changed = [path for path in live_by_path if live_by_path[path] != frozen_by_path[path]]
        wrapper_path = str(root / INPAINTING_WRAPPER)
        if any(path != wrapper_path for path in changed):
            raise CheckError("INPAINTING_RUNTIME_UNAUTHORIZED_REBIND")
        supplement_ref = branch.get("governance_publication_supplement")
        if changed and supplement_ref is None:
            raise CheckError("INPAINTING_WRAPPER_SUPPLEMENT_ABSENT")
        if supplement_ref is not None:
            if project_path(root, supplement_ref["path"]) != root / "inpainting_v1/GOVERNANCE_PUBLICATION_R02.json":
                raise CheckError("INPAINTING_WRAPPER_SUPPLEMENT_SCOPE")
            supplement = decode(read_ref(root, supplement_ref, "INPAINTING_WRAPPER_SUPPLEMENT_SHA_DRIFT"))
            if (supplement.get("schema") != "inpainting-v1-governance-publication-supplement"
                    or supplement.get("baseline_id") != INPAINTING_ID
                    or supplement.get("baseline_pointer") != pointer
                    or supplement.get("original_wrapper") != frozen_by_path.get(wrapper_path)
                    or supplement.get("current_wrapper") != live_by_path.get(wrapper_path)
                    or supplement.get("algorithm_changed") is not False
                    or type(supplement.get("files_deleted")) is not int or supplement["files_deleted"] != 0
                    or type(supplement.get("algorithm_calls")) is not int or supplement["algorithm_calls"] != 0):
                raise CheckError("INPAINTING_WRAPPER_SUPPLEMENT_BINDING_DRIFT")
            read_ref(root, supplement["task_packet"], "INPAINTING_WRAPPER_SUPPLEMENT_PACKET_DRIFT")
    for ref in live:
        if project_path(root, ref["path"]).suffix != ".py":
            raise CheckError("INPAINTING_RUNTIME_SOURCE_SCOPE")
        read_ref(root, ref, "INPAINTING_RUNTIME_SHA_DRIFT")
    if not {ref["path"] for ref in core}.issubset(live_by_path):
        raise CheckError("INPAINTING_CORE_RUNTIME_BINDING_DRIFT")
    packets, task_summary = current_tasks(root, docs)
    selected = [{"phase_alias": INPAINTING_ID, "baseline_id": INPAINTING_ID,
                 "implementation_revision": manifest.get("parent_implementation"),
                 "operation": INPAINTING_OPERATION, "actual_motion_source": manifest["actual_motion_source"],
                 "validated_scope": manifest["scope"], "pointer": pointer,
                 "training_eligible": False, "control_ground_truth": False}]
    return {"schema": "project-readonly-handoff-view-v1", "status": "PASSED_READONLY_BINDING",
            "governance_revision": receipt["governance_revision"],
            "packet_index_revision": docs["task_packet_index_current"]["governance_revision"],
            "packet_index_generation": docs["task_packet_index_current"]["generation_id"],
            "selected_baselines": selected, "selected_inpainting_baseline_count": 1,
            "other_branches": {"manus_0916": "OUT_OF_SCOPE_UNCHANGED_OTHER_BRANCH_READ_ONLY"},
            "current_tasks": task_summary, "frozen_release_files_sha_verified": len(live),
            "algorithm_core_files_sha_verified": len(core), "runtime_sources_sha_verified": len(live),
            "signed_runtime_wrapper_differences": [str(Path(path).relative_to(root)) for path in changed],
            "development_source": str(root / "src/chaoyang"),
            "release_role": "FROZEN_MANIFEST_NOT_PYTHONPATH_OR_SECOND_DEVELOPMENT_TREE",
            "execution_authorization": False, "models_run": 0, "files_written": 0,
            "environment": environment_report(False),
            "full_governance_validation": "USE_EXISTING_validate-governance_SEPARATELY",
            "quality_or_training_revalidation": "NOT_PERFORMED",
            "claim_limit": "Read/SHA/selection binding only; no model, visual quality, training or execution authority."}


def inspect_project(root=ROOT, alias=None):
    root = Path(root).resolve()
    receipt, docs = current_snapshot(root)
    specials = docs["algorithm_contract"]["special_status_contracts"]
    if "inpainting_raw_to_robot_v1" in specials:
        return inspect_inpainting(root, receipt, docs, alias)
    dual = specials["dual_branch_phase_baselines_v1"]
    manifest_ref = specials["baseline_v1_source_release"]["manifest"]
    authority = docs["doc_authority_map"]
    if dual["contract"] != authority_ref(authority, "dual_branch_phase_baselines_v1"):
        raise CheckError("PHASE_AUTHORITY_REF_CONFLICT")
    if manifest_ref != authority_ref(authority, "baseline_v1_source_release"):
        raise CheckError("SOURCE_AUTHORITY_REF_CONFLICT")
    phase = decode(read_ref(root, dual["contract"], "PHASE_SHA_DRIFT"))
    if set(phase["aliases"]) != ALIASES or phase["selected_alias_count"] != 2:
        raise CheckError("SELECTED_ALIAS_SET_DRIFT")
    if alias is not None and alias not in ALIASES:
        raise CheckError("ALIAS_NOT_SELECTED", alias)
    manifest, verified = source_release(root, manifest_ref)
    if manifest["phase_contract"] != dual["contract"]:
        raise CheckError("FROZEN_PHASE_BINDING_DRIFT")
    selected = []
    for name, row in phase["aliases"].items():
        projection = {key: row[key] for key in ("operation", "parent_baseline_id", "parent_implementation", "pointer")}
        if projection != dual["aliases"][name] or manifest["selected_operations"].get(name) != row["operation"]:
            raise CheckError("PHASE_OPERATION_DRIFT", name)
        read_ref(root, row["pointer"], "BASELINE_POINTER_DRIFT")
        if alias is None or alias == name:
            selected.append({"phase_alias": name, "phase_id": phase["phase_id"],
                             "implementation_revision": row["parent_implementation"],
                             "operation": row["operation"], "actual_motion_source": row["actual_motion_source"],
                             "validated_scope": row["validated_scope"],
                             "training_eligible": row["training_eligible"],
                             "control_ground_truth": row["control_ground_truth"],
                             "pointer": row["pointer"]})
    packets, task_summary = current_tasks(root, docs)
    differences = live_sources(root, manifest, docs["algorithm_contract"], packets)
    return {"schema": "project-readonly-handoff-view-v1", "status": "PASSED_READONLY_BINDING",
            "governance_revision": receipt["governance_revision"],
            "packet_index_revision": docs["task_packet_index_current"]["governance_revision"],
            "packet_index_generation": docs["task_packet_index_current"]["generation_id"],
            "selected_baselines": selected, "current_tasks": task_summary,
            "frozen_release_files_sha_verified": verified,
            "frozen_source_role_counts": manifest["counts"].get("roles", {}),
            "signed_runtime_wrapper_differences": differences,
            "development_source": str(root / "src/chaoyang"),
            "release_role": "FROZEN_REFERENCE_NOT_PYTHONPATH_OR_SECOND_DEVELOPMENT_TREE",
            "execution_authorization": False, "models_run": 0, "files_written": 0,
            "environment": environment_report(False),
            "full_governance_validation": "USE_EXISTING_validate-governance_SEPARATELY",
            "quality_or_training_revalidation": "NOT_PERFORMED",
            "claim_limit": "Read/SHA/selection binding only; no writer liveness, dispatch, model, visual, metric, training or adoption PASS."}


def profile_files(profile):
    if profile == "portable":
        return PORTABLE_TESTS
    if profile == "server":
        return SERVER_TESTS
    raise CheckError("TEST_PROFILE_UNKNOWN", profile)


def environment_report(require_pytest=False):
    supported = sys.version_info >= (3, 10)
    try:
        pytest_version = importlib.metadata.version("pytest")
    except importlib.metadata.PackageNotFoundError:
        pytest_version = None
    errors = ([] if supported else ["ENV_PYTHON_UNSUPPORTED"])
    if require_pytest and pytest_version is None:
        errors.append("ENV_PYTEST_ABSENT")
    try:
        jsonschema_version = importlib.metadata.version("jsonschema")
    except importlib.metadata.PackageNotFoundError:
        jsonschema_version = None
    return {"python": sys.version.split()[0], "interpreter": sys.executable,
            "pytest": pytest_version, "jsonschema": jsonschema_version,
            "errors": errors, "model_environment": "NOT_CHECKED_NO_MODEL_IMPORT"}


def require_server_evidence(root):
    for relative, sha in SERVER_EVIDENCE:
        path = project_path(root, relative)
        if not path.is_file():
            raise CheckError("SERVER_EVIDENCE_ABSENT", relative)
        if file_ref(path)["sha256"] != sha:
            raise CheckError("SERVER_EVIDENCE_SHA_DRIFT", relative)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--alias")
    parser.add_argument("--test-profile", choices=("portable", "server"))
    args = parser.parse_args(argv)
    try:
        if args.test_profile:
            result = {"status": "TEST_PROFILE_DECLARED_NOT_EXECUTED", "profile": args.test_profile,
                      "tests": list(profile_files(args.test_profile)),
                      "environment": environment_report(True),
                      "requires_server_evidence": args.test_profile == "server",
                      "command": [sys.executable, "-B", str(args.project_root / "scripts/maintenance/run_project_checks_v1.py"),
                                  "--profile", args.test_profile], "execution_authorization": False}
        else:
            result = inspect_project(args.project_root, args.alias)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2 if args.test_profile and result["environment"]["errors"] else 0
    except (CheckError, OSError, KeyError, TypeError) as error:
        code = error.code if isinstance(error, CheckError) else "HANDOFF_INPUT_UNREADABLE"
        print(json.dumps({"status": "FAILED_READONLY_CHECK", "failure_code": code,
                          "detail": str(error)[:300], "execution_authorization": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
