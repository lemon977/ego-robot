"""Byte-preserving source release, not a new algorithm or execution loader."""
from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import shutil

from chaoyang.governance.common import artifact_ref, atomic_json, load_json, validate_artifact_ref
from chaoyang.governance.source_snapshot_closure_v1 import local_import_closure

VISUAL_SEEDS = (
    "src/chaoyang/ops/run_exact78_takeover_baseline_v1.py",
    "src/chaoyang/ops/run_exact78_takeover_audit_v1.py",
    "src/chaoyang/ops/run_exact78_takeover_cleanup_v1.py",
    "src/chaoyang/governance/execute_exact78_takeover_v1.py",
    "src/chaoyang/pipeline/sam31_compat_adapter_v1.py",
)
MANUS_SEEDS = ("src/chaoyang/ops/run_robot100_v16_safe_replay.py",)
SELECTED_OPERATIONS = {
    "EXACT78_PHASE_V1": "run_exact78_takeover_baseline_v1",
    "MANUS_PHASE_V1": "run_robot100_v16_safe_replay",
}
RELEASE_FILES = ("MANIFEST.json", "VISUAL.json", "MANUS.json", "SHARED.json", "PARAMETERS.json", "EXTERNAL_DEPENDENCIES.json")


def source_plan(root: Path) -> dict:
    root = root.resolve()
    v_seeds = [root / name for name in VISUAL_SEEDS]
    v_seeds += list((root / "src/chaoyang/exact78_baseline").glob("*.py"))
    visual = set(local_import_closure(root, v_seeds))
    manus = set(local_import_closure(root, [root / name for name in MANUS_SEEDS]))
    sources = visual | manus
    rows = []
    for path in sorted(sources):
        relative = path.relative_to(root)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("SOURCE_NOT_REGULAR:" + str(path))
        if "governance" in relative.parts or "cli" in relative.parts or len(relative.parts) == 3:
            role = "RUNTIME_PLATFORM_DEPENDENCY_ONLY"
        elif path in visual and path in manus:
            role = "SHARED_DEPENDENCY_ONLY"
        elif path in visual:
            role = "VISUAL_IMPLEMENTATION"
        else:
            role = "MANUS_IMPLEMENTATION"
        owners = (["EXACT78_PHASE_V1"] if path in visual else []) + (["MANUS_PHASE_V1"] if path in manus else [])
        rows.append({"source": artifact_ref(path), "relative_path": str(relative),
                     "bundle_path": str(Path("code") / relative.relative_to("src")),
                     "role": role, "owners": owners,
                     "standalone_operation_selected": path.stem in SELECTED_OPERATIONS.values()})
    metadata = {"pyproject.toml", "THIRD_PARTY_NOTICES.md", "manifests/vendor.json",
                "docs/current/EXACT78_VISUAL_BASELINE_V1.json", "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json",
                "docs/current/DUAL_BRANCH_BASELINE_V1.json"}
    # These are module-level dependency/config paths, not permission to select
    # historical candidates. Session outputs and dataset payloads are not copied.
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and node.value.startswith(("configs/", "contracts/", "manifests/"))):
                candidate = root / node.value
                if candidate.is_file() and not candidate.is_symlink() and candidate.suffix in {".json", ".yaml", ".yml", ".toml"}:
                    if root not in candidate.resolve().parents:
                        raise RuntimeError("METADATA_SCOPE_ESCAPE")
                    metadata.add(node.value)
    meta = [{"source": artifact_ref(root / name), "relative_path": name, "bundle_path": "metadata/"+name,
             "role": "CONFIG_OR_INTERFACE_NOT_BASELINE_SELECTION"} for name in sorted(metadata)]
    return {"sources": rows, "metadata": meta,
            "counts": {"source_files": len(rows), "shared_owned_once": len(visual & manus),
                       "metadata_files": len(meta), "roles": dict(Counter(row["role"] for row in rows))}}


def parameters(plan: dict) -> dict:
    rows = []
    for row in plan["sources"]:
        if row["role"] == "RUNTIME_PLATFORM_DEPENDENCY_ONLY":
            continue
        values = {}
        for node in ast.parse(Path(row["source"]["path"]).read_text()).body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    try:
                        value = ast.literal_eval(node.value)
                        json.dumps(value, allow_nan=False)
                    except (ValueError, TypeError, SyntaxError):
                        continue
                    values[target.id] = value
        if values:
            rows.append({"source": row["source"], "constants": values})
    return {"schema_version": "V1_STATIC_PARAMETER_INDEX", "modules": rows,
            "claim_limit": "Static literals only. Computed constants, runtime arguments, calibration and session values remain in signed source/input contracts; no defaults changed."}


def copy_plan(plan: dict, destination: Path):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise RuntimeError("FRESH_BUNDLE_REQUIRED")
    rows = plan["sources"] + plan["metadata"]
    paths = [row["bundle_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise RuntimeError("DUPLICATE_BUNDLE_PATH")
    for row in rows:
        p = Path(row["bundle_path"])
        if p.is_absolute() or ".." in p.parts or not p.parts:
            raise RuntimeError("BUNDLE_PATH_SCOPE_ESCAPE")
        if errors := validate_artifact_ref(row["source"]):
            raise RuntimeError("SOURCE_DRIFT:" + ";".join(errors))
    # Source-copy generation is mechanical packaging; no implementation edits.
    destination.mkdir()
    for row in rows:
        source, target = Path(row["source"]["path"]), destination / row["bundle_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if artifact_ref(source) != row["source"] or artifact_ref(target)["sha256"] != row["source"]["sha256"]:
            raise RuntimeError("SOURCE_CHANGED_DURING_COPY")


def verify_bundle(root: Path, destination: Path) -> dict:
    manifest = load_json(destination / "MANIFEST.json")
    rows = manifest["sources"] + manifest["metadata"]
    paths = [row["bundle_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise RuntimeError("DUPLICATE_BUNDLE_PATH")
    for row in rows:
        path = destination / row["bundle_path"]
        if path.is_symlink() or destination.resolve() not in path.resolve().parents:
            raise RuntimeError("BUNDLE_PATH_SCOPE_ESCAPE")
        ref = artifact_ref(path)
        if ref["sha256"] != row["source"]["sha256"] or ref["bytes"] != row["source"]["bytes"]:
            raise RuntimeError("BUNDLE_SHA_DRIFT:" + row["bundle_path"])
        if row["bundle_path"].endswith(".py"):
            ast.parse(path.read_text())
    if source_plan(root) != {key: manifest[key] for key in ("sources", "metadata", "counts")}:
        raise RuntimeError("SOURCE_CLOSURE_DRIFT")
    from chaoyang.governance.dual_branch_phase_v1 import validate_phase
    if manifest["phase_contract"] != artifact_ref(root / "docs/current/DUAL_BRANCH_BASELINE_V1.json"):
        raise RuntimeError("PHASE_CONTRACT_DRIFT")
    validate_phase(root)
    if manifest["selected_operations"] != SELECTED_OPERATIONS:
        raise RuntimeError("SELECTED_OPERATION_DRIFT")
    if manifest["production_loader_changed"] or manifest["algorithm_changed"]:
        raise RuntimeError("SOURCE_RELEASE_IS_NOT_RUNTIME_MIGRATION")
    for name, ref in manifest["indexes"].items():
        if name not in RELEASE_FILES or ref != artifact_ref(destination / name):
            raise RuntimeError("INDEX_DRIFT")
    return {"status": "PASS", "counts": manifest["counts"], "copied_files_sha_verified": len(rows),
            "selected_operations": SELECTED_OPERATIONS, "production_loader_changed": False,
            "algorithm_changed": False, "claim_limit": "Frozen source consolidation only; not a portable installed environment or fresh model/video replay."}
