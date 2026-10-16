"""Process-local mount compatibility for the frozen Joint26 CAS publisher.

The adopted Inpainting manifest pins governance source bytes. Rather than edit
those sources or old receipts, this opt-in ICT launcher adapts their *local
module bindings*. pathlib, builtins, other processes and source files are not
patched. Original serialized manifests/packets and bytes/SHA remain unchanged.
Only explicitly listed selection/navigation metadata views are normalized for
alias-aware comparison; training manifests, configurations and signed packets
are not.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from .project_mount_paths import PROJECT_ALIASES, resolve_input_path

_INSTALLED_ROOT: Path | None = None
_PATH_CLASS = None
# Explicit consumers reached by validate-governance and the existing publisher.
# This is not an import hook and cannot affect unrelated algorithm modules.
_MODULES = (
    "chaoyang.governance.validate_governance_state",
    "chaoyang.governance.rc1_t5_consistency",
    "chaoyang.governance.current_r3_contracts",
    "chaoyang.governance.current_baseline_v2",
    "chaoyang.governance.register_single_task_packet",
    "chaoyang.governance.v52_contracts",
    "chaoyang.governance.v71_contracts",
    "chaoyang.governance.dual_branch_phase_v1",
    "chaoyang.governance.build_algorithm_baseline_v2_status",
    "chaoyang.governance.build_full_pipeline_v3_status",
    "chaoyang.governance.build_full_pipeline_v4_status",
    "chaoyang.governance.build_human_to_robot_r2_terminal_status",
    "chaoyang.governance.inpainting_v1_selection",
    "chaoyang.governance.project_branch_navigation_v1",
    "chaoyang.inpainting_candidate.publication",
)
_SELECTION_VIEWS = frozenset({
    "inpainting_v1/MANIFEST.json",
    "inpainting_v1/GOVERNANCE_PUBLICATION_R02.json",
    "inpainting_v1/GOVERNANCE_PUBLICATION_R03.json",
    "tasks/receipts/INPAINTING_RAW_TO_ROBOT_V1.json",
    # The existing publisher still checks these historical adoption identities;
    # they remain metadata-only, not an alternate algorithm selection.
    "docs/current/DUAL_BRANCH_BASELINE_V1.json",
    "docs/current/EXACT78_VISUAL_BASELINE_V1.json",
    "docs/current/ROBOT100_HARDWARE_BASELINE_V16.json",
    "tasks/receipts/EXACT78_VISUAL_BASELINE_V1.json",
    # The shared publisher compares this immutable navigation snapshot with a
    # freshly rebuilt branch map. Normalize only its in-memory selection refs.
    "docs/current/PROJECT_BRANCH_MAP.json",
})


def _local_known_path(value, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute() and any(path.is_relative_to(alias) for alias in PROJECT_ALIASES):
        return resolve_input_path(path, root, exists=False)
    return path


def _selection_view(value, root: Path):
    """Return new local metadata, never mutate the read JSON or any file."""
    if isinstance(value, dict):
        return {key: _selection_view(child, root) for key, child in value.items()}
    if isinstance(value, list):
        return [_selection_view(child, root) for child in value]
    if isinstance(value, str) and value.startswith("/"):
        return str(_local_known_path(value, root))
    return value


def bind_publisher_module(module: ModuleType) -> ModuleType:
    """Bind one explicitly loaded frozen infra module, not global pathlib."""
    if _PATH_CLASS is not None:
        module.Path = _PATH_CLASS
    return module


def install_governance_mount_compat(root: Path) -> dict:
    global _INSTALLED_ROOT, _PATH_CLASS
    root = Path(root)
    # The regular /mnt launcher retains its original governance behavior.
    if root == PROJECT_ALIASES[0]:
        return {"enabled": False, "runtime_root": str(root)}
    if root not in PROJECT_ALIASES:
        raise ValueError("UNREGISTERED_ICT_PROJECT_MOUNT")
    resolve_input_path(root, root)  # Validate the physical mount identity.
    if _INSTALLED_ROOT is not None:
        if _INSTALLED_ROOT != root:
            raise ValueError("GOVERNANCE_MOUNT_BINDING_CONFLICT")
        return {"enabled": True, "runtime_root": str(root)}

    class MountPath(type(Path())):
        def __new__(cls, *parts):
            local = _local_known_path(Path(*parts), root)
            return super().__new__(cls, local)

    common = importlib.import_module("chaoyang.governance.common")
    original_load = common.load_json
    original_sha = common.sha256_file
    original_artifact = common.artifact_ref

    def load_local_json(path):
        local = _local_known_path(path, root)
        value = original_load(local)
        relative = str(local.relative_to(root)) if local.is_relative_to(root) else ""
        return _selection_view(value, root) if relative in _SELECTION_VIEWS else value

    def sha_local_file(path):
        return original_sha(_local_known_path(path, root))

    def artifact_local_file(path):
        # Old consumers may compare the exact original descriptor, including
        # its logical path. Read the local bytes but preserve an explicitly
        # supplied old spelling; newly created CPFS refs retain CPFS spelling.
        logical = Path(path)
        result = original_artifact(_local_known_path(logical, root))
        if logical.is_absolute() and logical.is_relative_to(PROJECT_ALIASES[0]):
            result["path"] = str(logical)
        return result

    # The frozen validator's Path(...) and artifact_ref's Path(...) now locate
    # the same bytes at the current mount. Its SHA and byte checks are intact.
    common.Path = MountPath
    common.load_json = load_local_json
    common.sha256_file = sha_local_file
    common.artifact_ref = artifact_local_file
    _PATH_CLASS = MountPath
    for name in _MODULES:
        module = importlib.import_module(name)
        if hasattr(module, "Path"):
            module.Path = MountPath
        # Also handle modules already imported by the launcher's bootstrap.
        if getattr(module, "load_json", None) is original_load:
            module.load_json = load_local_json
        if getattr(module, "sha256_file", None) is original_sha:
            module.sha256_file = sha_local_file
        if getattr(module, "artifact_ref", None) is original_artifact:
            module.artifact_ref = artifact_local_file
    # The publisher also rebuilds the old frozen-payload route. Its strictly
    # project-local _ref checks one sealed reader with an original /mnt path.
    # Adapt that read, retaining its logical descriptor, without selecting or
    # executing the old training operation or altering its frozen source.
    frozen = importlib.import_module("chaoyang.humanego_ict.frozen_payload_entry_cli")
    original_frozen_ref = frozen._ref

    def frozen_local_ref(path, reference_root):
        logical = Path(path)
        result = original_frozen_ref(_local_known_path(path, reference_root), reference_root)
        if logical.is_absolute() and logical.is_relative_to(PROJECT_ALIASES[0]):
            result["path"] = str(logical)
        return result

    frozen._ref = frozen_local_ref
    _INSTALLED_ROOT = root
    return {"enabled": True, "runtime_root": str(root),
            "source_files_modified": False, "original_json_bytes_rewritten": False,
            "scope": "JOINT26_PROCESS_LOCAL_GOVERNANCE_BINDINGS"}
