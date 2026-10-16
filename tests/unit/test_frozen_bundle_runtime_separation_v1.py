"""Frozen release integrity and a pinned runtime audit are different contracts.

The runtime packet below was captured by the bounded maintenance task. It is
an audit input, not execution permission, and must never be replaced by latest.
Current wrapper bindings are separate, receipt-bound evidence; historical
wrapper records do not assert that their old source bytes remain on disk.
No production source, release bytes, numerical gate or writer is modified here.
"""
import ast
import copy
import importlib.util
from pathlib import Path

import pytest

from chaoyang.governance import baseline_v1_source_release as release
from chaoyang.governance.common import artifact_ref, load_json

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PACKET_REF = {
    "path": str(ROOT / "tasks/current/exact78_pick_place_production_batch04_full_v9_reverse_boundary_revision2_20261009/TASK_PACKET.json"),
    "bytes": 27998,
    "sha256": "2931068f54f0a311718716f3d952d0280e82af64d154e6166cf0d38b52ce3420",
}
SOURCE_RECEIPT_SHA = "f1327e0c005989fdd8f8e1a8d0e16b0f131346f414848674ec6675180ab39111"
ALLOWED_RUNTIME_WRAPPERS = frozenset({
    "src/chaoyang/governance/current_r3_contracts.py",
    "src/chaoyang/governance/exact78_visual_selection_v1.py",
})
CURRENT_RUNTIME_ROUTE_KEYS = (
    "exact78_visual_baseline_v1", "pick_place_1008_candidate_v1",
    "humanego_ict_joint26_prepare_v1", "humanego_ict_joint26_v1",
)


def load_test_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HANDOFF = load_test_helper("runtime_audit_handoff", ROOT / "scripts/maintenance/project_handoff_check_v1.py")
SYNTHETIC = load_test_helper("runtime_audit_fixture", ROOT / "tests/helpers/handoff_fixtures_v1.py")


def checked_ref(path, expected, failure):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(failure + ":NOT_REGULAR")
    actual = artifact_ref(path)
    if any(actual[key] != expected[key] for key in ("bytes", "sha256")):
        raise RuntimeError(failure + ":" + str(path))
    return actual


def verify_frozen_rows(bundle, manifest):
    bundle = Path(bundle).resolve()
    rows = manifest["sources"] + manifest["metadata"]
    names = [row["bundle_path"] for row in rows]
    if len(names) != len(set(names)):
        raise RuntimeError("DUPLICATE_BUNDLE_PATH")
    for row in rows:
        relative = Path(row["bundle_path"])
        path = bundle / relative
        if relative.is_absolute() or ".." in relative.parts or bundle not in path.resolve().parents:
            raise RuntimeError("BUNDLE_SCOPE_ESCAPE")
        checked_ref(path, row["source"], "BUNDLE_SHA_DRIFT")
        if path.suffix == ".py":
            ast.parse(path.read_text())
    return len(rows)


def verify_frozen_release(root, bundle):
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    receipt_path = root / "tasks/receipts/BASELINE_V1_SOURCE_RELEASE_20261008.json"
    receipt = checked_ref(receipt_path, {"bytes": 581, "sha256": SOURCE_RECEIPT_SHA}, "RELEASE_RECEIPT_DRIFT")
    assert receipt["sha256"] == SOURCE_RECEIPT_SHA
    bound = load_json(receipt_path)["source_release"]
    if Path(bound["path"]).resolve() != bundle / "MANIFEST.json":
        raise RuntimeError("MANIFEST_BINDING_SCOPE")
    checked_ref(bundle / "MANIFEST.json", bound, "FROZEN_MANIFEST_DRIFT")
    manifest = load_json(bundle / "MANIFEST.json")
    count = verify_frozen_rows(bundle, manifest)
    if manifest["selected_operations"] != release.SELECTED_OPERATIONS:
        raise RuntimeError("SELECTED_OPERATION_DRIFT")
    if manifest["algorithm_changed"] or manifest["production_loader_changed"]:
        raise RuntimeError("SOURCE_RELEASE_SCOPE_DRIFT")
    if manifest["phase_contract"] != artifact_ref(root / "docs/current/DUAL_BRANCH_BASELINE_V1.json"):
        raise RuntimeError("PHASE_CONTRACT_DRIFT")
    for name, ref in manifest["indexes"].items():
        if name not in release.RELEASE_FILES or Path(ref["path"]).resolve() != bundle / name:
            raise RuntimeError("INDEX_SCOPE_DRIFT")
        checked_ref(bundle / name, ref, "INDEX_DRIFT")
    return {"counts": manifest["counts"], "copied_files_sha_verified": count,
            "algorithm_changed": False, "production_loader_changed": False}


def verify_live_boundary(root, manifest, runtime_refs):
    """Only the two named platform wrappers may differ, with exact signed pins."""
    root = Path(root).resolve()
    pins = {}
    for ref in runtime_refs:
        path = Path(ref["path"])
        if root not in path.resolve().parents:
            raise RuntimeError("RUNTIME_SCOPE_ESCAPE")
        relative = str(path.relative_to(root))
        if relative in pins:
            raise RuntimeError("DUPLICATE_RUNTIME_REF")
        checked_ref(path, ref, "RUNTIME_SHA_MISMATCH")
        pins[relative] = ref
    differences = []
    for row in manifest["sources"]:
        path = Path(row["source"]["path"])
        if root not in path.resolve().parents or path.is_symlink():
            raise RuntimeError("LIVE_SOURCE_SCOPE_ESCAPE")
        relative = str(path.relative_to(root))
        actual = artifact_ref(path)
        if all(actual[key] == row["source"][key] for key in ("bytes", "sha256")):
            continue
        if relative not in ALLOWED_RUNTIME_WRAPPERS or row["role"] != "RUNTIME_PLATFORM_DEPENDENCY_ONLY":
            raise RuntimeError("LIVE_ALGORITHM_DRIFT:" + relative)
        if relative not in pins or actual != pins[relative]:
            raise RuntimeError("UNBOUND_RUNTIME_WRAPPER:" + relative)
        differences.append(relative)
    return {"runtime_refs_verified": len(pins), "signed_wrapper_differences": differences}


def current_runtime_wrapper_refs(root):
    """Only current receipt-bound, explicitly named routes can bind wrappers."""
    root = Path(root).resolve()
    receipt, documents = HANDOFF.current_snapshot(root)
    specials = documents["algorithm_contract"]["special_status_contracts"]
    allowed_paths = {str(root / relative) for relative in ALLOWED_RUNTIME_WRAPPERS}
    pins = {}
    for key in CURRENT_RUNTIME_ROUTE_KEYS:
        for ref in specials.get(key, {}).get("code_closure", []):
            if ref.get("path") not in allowed_paths:
                continue
            HANDOFF.read_ref(root, ref, "CURRENT_RUNTIME_SHA_DRIFT")
            if ref["path"] in pins and pins[ref["path"]] != ref:
                raise RuntimeError("CURRENT_WRAPPER_PIN_CONFLICT")
            pins[ref["path"]] = ref
    return {"governance_revision": receipt["governance_revision"],
            "generation_id": receipt["generation_id"],
            "algorithm_contract": receipt["files"]["algorithm_contract"],
            "refs": [pins[path] for path in sorted(pins)]}


def verify_historical_and_current_runtime(root, manifest, historical_refs):
    """Keep old records intact; independently bind only evolved wrappers."""
    root = Path(root).resolve()
    current = current_runtime_wrapper_refs(root)
    current_pins = {ref["path"]: ref for ref in current["refs"]}
    effective, rebindings = [], []
    nonwrappers_verified = 0
    for ref in historical_refs:
        path = Path(ref["path"])
        if root not in path.resolve().parents or path.is_symlink():
            raise RuntimeError("RUNTIME_SCOPE_ESCAPE")
        relative = str(path.relative_to(root))
        if relative in ALLOWED_RUNTIME_WRAPPERS and artifact_ref(path) != ref:
            current_ref = current_pins.get(str(path))
            if current_ref is None:
                raise RuntimeError("UNBOUND_RUNTIME_WRAPPER:" + relative)
            effective.append(current_ref)
            rebindings.append({"historical_ref": ref, "current_ref": current_ref})
        else:
            checked_ref(path, ref, "RUNTIME_SHA_MISMATCH")
            effective.append(ref)
            if relative not in ALLOWED_RUNTIME_WRAPPERS:
                nonwrappers_verified += 1
    boundary = verify_live_boundary(root, manifest, effective)
    return {"historical_runtime_ref_count": len(historical_refs),
            "historical_nonwrapper_refs_sha_verified": nonwrappers_verified,
            "historical_wrapper_records": [ref for ref in historical_refs
                if str(Path(ref["path"]).relative_to(root)) in ALLOWED_RUNTIME_WRAPPERS],
            "current_wrapper_binding": current,
            "current_wrapper_refs": current["refs"],
            "wrapper_rebindings": rebindings,
            "effective_runtime_refs": effective,
            "effective_runtime_refs_verified": boundary["runtime_refs_verified"],
            "signed_wrapper_differences": boundary["signed_wrapper_differences"]}


def verify_pinned_runtime(root, manifest):
    checked_ref(RUNTIME_PACKET_REF["path"], RUNTIME_PACKET_REF, "RUNTIME_PACKET_DRIFT")
    packet = load_json(Path(RUNTIME_PACKET_REF["path"]))
    if packet["task_id"] != "exact78_pick_place_production_batch04_full_v9_reverse_boundary_revision2_20261009":
        raise RuntimeError("RUNTIME_PACKET_ID_DRIFT")
    if len(packet["runtime_code"]) != 93:
        raise RuntimeError("HISTORICAL_RUNTIME_COUNT_DRIFT")
    return {"historical_packet_ref": dict(RUNTIME_PACKET_REF),
            **verify_historical_and_current_runtime(root, manifest, packet["runtime_code"])}


def verify_frozen_imports(root, bundle, manifest):
    """Inspect sealed ASTs, not the evolving current governance import graph."""
    root, bundle = Path(root), Path(bundle)
    modules = {}
    for row in manifest["sources"]:
        relative = Path(row["relative_path"]).relative_to("src")
        name = ".".join(relative.with_suffix("").parts)
        if name.endswith(".__init__"):
            name = name[:-9]
        modules[name] = bundle / row["bundle_path"]
    for module, path in modules.items():
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parent = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
                    prefix = ".".join(parent[:len(parent) - node.level + 1])
                    base = prefix + ("." + base if base else "")
                names = [base] + [base + "." + alias.name for alias in node.names if alias.name != "*"]
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "import_module" and node.args
                  and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                names = [node.args[0].value]
            for name in names:
                if not name.startswith("chaoyang"):
                    continue
                source = root / "src" / name.replace(".", "/")
                if (source.with_suffix(".py").is_file() or (source / "__init__.py").is_file()) and name not in modules:
                    raise RuntimeError("FROZEN_IMPORT_CLOSURE_MISSING:" + name)
    return set(modules)


def audit_deleted_generations(root, deleted, proof, sealed_after_ns, runtime_refs):
    """Audit old deletion, never require regenerable paths to stay absent forever.

    A same-path current file is accepted only as a different filesystem
    generation, created after the receipt witness and backed by a valid source.
    This does not claim its compiled bytes are current or authorize deletion.
    """
    from chaoyang.governance.project_slim_cache_proof_v1 import original

    root = Path(root).resolve()
    eligible = {row["path"]: row for row in proof["rows"] if row["status"] == "ELIGIBLE_C1"}
    rows = deleted["deleted"]
    if len({row["path"] for row in rows}) != len(rows) or set(eligible) != {row["path"] for row in rows}:
        raise RuntimeError("DELETE_MANIFEST_COVERAGE_DRIFT")
    if (len(rows) != deleted["files_deleted"]
            or sum(row["bytes"] for row in rows) != deleted["bytes_deleted"]
            or sum(row["allocated_bytes"] for row in rows) != deleted["allocated_bytes_deleted"]):
        raise RuntimeError("DELETE_MANIFEST_TOTAL_DRIFT")
    pins = {ref["path"]: ref for ref in runtime_refs}
    regenerated = []
    for row in rows:
        old = eligible[row["path"]]
        if (row["sha256"] != old["file"]["sha256"] or row["bytes"] != old["bytes"]
                or row["allocated_bytes"] != old["allocated_bytes"]
                or row["regeneration"] != old["regeneration"]):
            raise RuntimeError("DELETE_MANIFEST_ROW_DRIFT")
        path = root / row["path"]
        if root not in path.resolve().parents or path.is_symlink():
            raise RuntimeError("DELETED_PATH_SCOPE_DRIFT")
        if not path.exists():
            continue
        if not path.is_file():
            raise RuntimeError("REGENERATED_CACHE_NOT_REGULAR")
        current = path.stat()
        if (current.st_dev, current.st_ino) == (old["dev"], old["inode"]):
            raise RuntimeError("ORIGINAL_DELETED_GENERATION_STILL_PRESENT")
        if current.st_ctime_ns <= sealed_after_ns:
            raise RuntimeError("REGENERATED_GENERATION_TIME_UNVERIFIED")
        regeneration = row["regeneration"]
        source_ref = regeneration.get("source")
        if not source_ref:
            raise RuntimeError("REGENERATED_SOURCE_ABSENT")
        source = Path(source_ref["path"])
        if root not in source.resolve().parents or source.is_symlink() or not source.is_file():
            raise RuntimeError("REGENERATED_SOURCE_SCOPE_DRIFT")
        kind = regeneration["kind"]
        if kind == "PYTHON_BYTECODE_CACHE":
            if path.suffix != ".pyc" or original(path, old["root"]) != source:
                raise RuntimeError("REGENERATED_CACHE_SOURCE_MISMATCH")
        elif kind == "NUMBA_JIT_CACHE":
            if path.suffix not in {".nbc", ".nbi"} or not path.name.startswith(source.stem + "."):
                raise RuntimeError("REGENERATED_CACHE_SOURCE_MISMATCH")
        else:
            raise RuntimeError("REGENERATED_CACHE_KIND_UNVERIFIED")
        actual_source = artifact_ref(source)
        if actual_source != source_ref:
            relative = str(source.relative_to(root))
            if relative not in ALLOWED_RUNTIME_WRAPPERS or pins.get(str(source)) != actual_source:
                raise RuntimeError("REGENERATED_SOURCE_BINDING_DRIFT")
        regenerated.append(row["path"])
    return {"historical_deleted": len(rows), "currently_absent": len(rows) - len(regenerated),
            "regenerated_generations": regenerated}


@pytest.fixture
def boundary_fixture(tmp_path):
    root, bundle = tmp_path / "repo", tmp_path / "repo/baseline_v1"
    rows = []
    for name, role in [
        ("src/chaoyang/exact78_baseline/render.py", "VISUAL_IMPLEMENTATION"),
        ("src/chaoyang/governance/current_r3_contracts.py", "RUNTIME_PLATFORM_DEPENDENCY_ONLY"),
    ]:
        source = root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("VALUE = 1\n")
        relative = "code/" + str(Path(name).relative_to("src"))
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        rows.append({"source": artifact_ref(source), "relative_path": name,
                     "bundle_path": relative, "role": role})
    return root, bundle, {"sources": rows, "metadata": []}


def test_frozen_file_tampering_is_rejected(boundary_fixture):
    _, bundle, manifest = boundary_fixture
    (bundle / manifest["sources"][0]["bundle_path"]).write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="BUNDLE_SHA_DRIFT"):
        verify_frozen_rows(bundle, manifest)


def test_bound_manifest_tampering_is_rejected(tmp_path):
    path = tmp_path / "MANIFEST.json"
    path.write_text("{}\n")
    expected = artifact_ref(path)
    path.write_text('{"changed":true}\n')
    with pytest.raises(RuntimeError, match="FROZEN_MANIFEST_DRIFT"):
        checked_ref(path, expected, "FROZEN_MANIFEST_DRIFT")


def test_frozen_path_escape_is_rejected(boundary_fixture):
    _, bundle, manifest = boundary_fixture
    manifest["sources"][0]["bundle_path"] = "../outside.py"
    with pytest.raises(RuntimeError, match="BUNDLE_SCOPE_ESCAPE"):
        verify_frozen_rows(bundle, manifest)


def test_unchanged_live_sources_pass(boundary_fixture):
    root, bundle, manifest = boundary_fixture
    assert verify_frozen_rows(bundle, manifest) == 2
    result = verify_live_boundary(root, manifest, [row["source"] for row in manifest["sources"]])
    assert result["signed_wrapper_differences"] == []


def test_signed_wrapper_change_does_not_change_frozen_release(boundary_fixture):
    root, bundle, manifest = boundary_fixture
    wrapper = Path(manifest["sources"][1]["source"]["path"])
    wrapper.write_text("VALUE = 2\n")
    result = verify_live_boundary(root, manifest, [artifact_ref(wrapper)])
    assert result["signed_wrapper_differences"] == ["src/chaoyang/governance/current_r3_contracts.py"]
    assert verify_frozen_rows(bundle, manifest) == 2


def test_signed_nonwrapper_algorithm_change_is_still_rejected(boundary_fixture):
    root, _, manifest = boundary_fixture
    algorithm = Path(manifest["sources"][0]["source"]["path"])
    algorithm.write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="LIVE_ALGORITHM_DRIFT"):
        verify_live_boundary(root, manifest, [artifact_ref(algorithm)])


def test_unbound_wrapper_change_is_rejected(boundary_fixture):
    root, _, manifest = boundary_fixture
    Path(manifest["sources"][1]["source"]["path"]).write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="UNBOUND_RUNTIME_WRAPPER"):
        verify_live_boundary(root, manifest, [])


def test_runtime_sha_mismatch_is_rejected(boundary_fixture):
    root, _, manifest = boundary_fixture
    pin = copy.deepcopy(manifest["sources"][1]["source"])
    pin["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="RUNTIME_SHA_MISMATCH"):
        verify_live_boundary(root, manifest, [pin])


def test_runtime_byte_count_mismatch_is_rejected(boundary_fixture):
    root, _, manifest = boundary_fixture
    pin = copy.deepcopy(manifest["sources"][1]["source"])
    pin["bytes"] += 1
    with pytest.raises(RuntimeError, match="RUNTIME_SHA_MISMATCH"):
        verify_live_boundary(root, manifest, [pin])


def test_wrapper_role_cannot_be_spoofed(boundary_fixture):
    root, _, manifest = boundary_fixture
    wrapper = Path(manifest["sources"][1]["source"]["path"])
    wrapper.write_text("VALUE = 2\n")
    manifest["sources"][1]["role"] = "VISUAL_IMPLEMENTATION"
    with pytest.raises(RuntimeError, match="LIVE_ALGORITHM_DRIFT"):
        verify_live_boundary(root, manifest, [artifact_ref(wrapper)])


def test_duplicate_runtime_pin_is_rejected(boundary_fixture):
    root, _, manifest = boundary_fixture
    pin = manifest["sources"][1]["source"]
    with pytest.raises(RuntimeError, match="DUPLICATE_RUNTIME_REF"):
        verify_live_boundary(root, manifest, [pin, pin])


@pytest.fixture
def receipt_boundary_fixture(tmp_path):
    root = SYNTHETIC.build(tmp_path / "current_receipt_boundary")
    return root, load_json(root / "baseline_v1/MANIFEST.json")


def bind_current_test_route(root, key, refs):
    path = root / "docs/governance/ALGORITHM_CONTRACT.json"
    algorithm = load_json(path)
    algorithm["special_status_contracts"][key] = {"code_closure": refs}
    SYNTHETIC.write_json(path, algorithm)
    SYNTHETIC.refresh(root)


@pytest.mark.parametrize("key", CURRENT_RUNTIME_ROUTE_KEYS[2:])
def test_current_wrapper_binding_preserves_historical_record(receipt_boundary_fixture, key):
    root, manifest = receipt_boundary_fixture
    historical = copy.deepcopy([row["source"] for row in manifest["sources"]])
    wrapper = root / SYNTHETIC.WRAPPER
    wrapper.write_text("# current platform wrapper\n")
    current_ref = artifact_ref(wrapper)
    bind_current_test_route(root, key, [current_ref])
    result = verify_historical_and_current_runtime(root, manifest, historical)
    assert result["historical_runtime_ref_count"] == 2
    assert result["historical_nonwrapper_refs_sha_verified"] == 1
    assert result["historical_wrapper_records"] == [manifest["sources"][1]["source"]]
    assert result["current_wrapper_refs"] == [current_ref]
    assert result["effective_runtime_refs_verified"] == 2
    assert result["wrapper_rebindings"] == [{"historical_ref": historical[1], "current_ref": current_ref}]
    assert historical == [row["source"] for row in manifest["sources"]]
    assert verify_frozen_rows(root / "baseline_v1", manifest) == 2


@pytest.mark.parametrize("key", [None, "humanego_ict_joint26_v0_historical"])
def test_unbound_or_historical_candidate_cannot_rebind_wrapper(receipt_boundary_fixture, key):
    root, manifest = receipt_boundary_fixture
    wrapper = root / SYNTHETIC.WRAPPER
    wrapper.write_text("# unbound current wrapper\n")
    if key is not None:
        bind_current_test_route(root, key, [artifact_ref(wrapper)])
    with pytest.raises(RuntimeError, match="UNBOUND_RUNTIME_WRAPPER"):
        verify_historical_and_current_runtime(root, manifest, [row["source"] for row in manifest["sources"]])


@pytest.mark.parametrize("key", CURRENT_RUNTIME_ROUTE_KEYS[2:])
def test_current_wrapper_drift_after_receipt_binding_is_rejected(receipt_boundary_fixture, key):
    root, manifest = receipt_boundary_fixture
    wrapper = root / SYNTHETIC.WRAPPER
    bind_current_test_route(root, key, [artifact_ref(wrapper)])
    wrapper.write_text("# changed after current signature\n")
    with pytest.raises(HANDOFF.CheckError, match="CURRENT_RUNTIME_SHA_DRIFT"):
        verify_historical_and_current_runtime(root, manifest, [row["source"] for row in manifest["sources"]])


@pytest.mark.parametrize("key", CURRENT_RUNTIME_ROUTE_KEYS[2:])
def test_current_candidate_cannot_rebind_algorithm_source(receipt_boundary_fixture, key):
    root, manifest = receipt_boundary_fixture
    algorithm = root / SYNTHETIC.ALGORITHM
    algorithm.write_text("# changed nonwrapper algorithm\n")
    bind_current_test_route(root, key, [artifact_ref(algorithm)])
    with pytest.raises(RuntimeError, match="RUNTIME_SHA_MISMATCH"):
        verify_historical_and_current_runtime(root, manifest, [row["source"] for row in manifest["sources"]])


def test_current_binding_cannot_excuse_wrapper_role_change(receipt_boundary_fixture):
    root, manifest = receipt_boundary_fixture
    historical = [row["source"] for row in manifest["sources"]]
    wrapper = root / SYNTHETIC.WRAPPER
    wrapper.write_text("# changed wrapper with wrong manifest role\n")
    bind_current_test_route(root, CURRENT_RUNTIME_ROUTE_KEYS[2], [artifact_ref(wrapper)])
    manifest["sources"][1]["role"] = "VISUAL_IMPLEMENTATION"
    with pytest.raises(RuntimeError, match="LIVE_ALGORITHM_DRIFT"):
        verify_historical_and_current_runtime(root, manifest, historical)


def test_current_receipt_contract_tampering_is_rejected(receipt_boundary_fixture):
    root, manifest = receipt_boundary_fixture
    path = root / "docs/governance/ALGORITHM_CONTRACT.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(HANDOFF.CheckError, match="INPUT_SHA_MISMATCH"):
        verify_historical_and_current_runtime(root, manifest, [row["source"] for row in manifest["sources"]])


def test_pytest_ignores_generated_fixture_trees_without_replacing_defaults():
    tree = ast.parse((ROOT / "tests/conftest.py").read_text())
    assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    assert assignments["collect_ignore_glob"] == ["pick_place_1008/_tmp_*"]


@pytest.fixture
def deletion_generation_fixture(boundary_fixture):
    # Synthetic old inode metadata models two generations without deleting any
    # actual file. Nothing here is published as a real cleanup receipt.
    root, _, manifest = boundary_fixture
    source = manifest["sources"][0]["source"]
    cache = root / "src/chaoyang/exact78_baseline/__pycache__/render.cpython-311.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"new-generation-cache")
    current = cache.stat()
    relative = str(cache.relative_to(root))
    regeneration = {"kind": "PYTHON_BYTECODE_CACHE", "source": source}
    row = {"path": relative, "sha256": "f" * 64, "bytes": 5,
           "allocated_bytes": 512, "regeneration": regeneration}
    old = {**row, "file": {"sha256": row["sha256"]}, "status": "ELIGIBLE_C1",
           "dev": current.st_dev, "inode": current.st_ino + 1,
           "root": "src/chaoyang/exact78_baseline/__pycache__"}
    deleted = {"deleted": [row], "files_deleted": 1, "bytes_deleted": 5,
               "allocated_bytes_deleted": 512}
    return root, cache, deleted, {"rows": [old]}, current.st_ctime_ns - 1


def test_new_cache_generation_does_not_undo_historical_deletion(deletion_generation_fixture):
    root, _, deleted, proof, sealed = deletion_generation_fixture
    result = audit_deleted_generations(root, deleted, proof, sealed, [])
    assert result["historical_deleted"] == 1 and result["currently_absent"] == 0
    assert result["regenerated_generations"] == [deleted["deleted"][0]["path"]]


def test_original_deleted_inode_cannot_be_passed_off_as_regeneration(deletion_generation_fixture):
    root, cache, deleted, proof, sealed = deletion_generation_fixture
    proof["rows"][0]["inode"] = cache.stat().st_ino
    with pytest.raises(RuntimeError, match="ORIGINAL_DELETED_GENERATION_STILL_PRESENT"):
        audit_deleted_generations(root, deleted, proof, sealed, [])


def test_cache_older_than_deletion_witness_is_rejected(deletion_generation_fixture):
    root, cache, deleted, proof, _ = deletion_generation_fixture
    with pytest.raises(RuntimeError, match="REGENERATED_GENERATION_TIME_UNVERIFIED"):
        audit_deleted_generations(root, deleted, proof, cache.stat().st_ctime_ns + 1, [])


def test_changed_source_without_signed_runtime_is_rejected(deletion_generation_fixture):
    root, _, deleted, proof, sealed = deletion_generation_fixture
    Path(deleted["deleted"][0]["regeneration"]["source"]["path"]).write_text("VALUE = 9\n")
    with pytest.raises(RuntimeError, match="REGENERATED_SOURCE_BINDING_DRIFT"):
        audit_deleted_generations(root, deleted, proof, sealed, [])


def test_deleted_manifest_cannot_drop_eligible_proof_rows(deletion_generation_fixture):
    root, _, deleted, proof, sealed = deletion_generation_fixture
    deleted["deleted"] = []
    with pytest.raises(RuntimeError, match="DELETE_MANIFEST_COVERAGE_DRIFT"):
        audit_deleted_generations(root, deleted, proof, sealed, [])


def test_unknown_recreated_payload_cannot_be_called_cache(deletion_generation_fixture):
    root, _, deleted, proof, sealed = deletion_generation_fixture
    regeneration = {**deleted["deleted"][0]["regeneration"], "kind": "UNKNOWN"}
    deleted["deleted"][0]["regeneration"] = regeneration
    proof["rows"][0]["regeneration"] = regeneration
    with pytest.raises(RuntimeError, match="REGENERATED_CACHE_KIND_UNVERIFIED"):
        audit_deleted_generations(root, deleted, proof, sealed, [])

