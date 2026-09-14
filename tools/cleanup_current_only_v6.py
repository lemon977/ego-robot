"""Evidence-first current-only cleanup for the Chaoyang repository.

The default command is read-only.  Mutating commands require an explicit
``--commit-delete`` and refuse current governance references, active process
references, the live Wave0 Clean tree, and the protected data/NAS roots.  An
unrelated candidate is not globally blocked merely because Wave0 is active.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/mnt/workspace/code/chaoyang")
RUN_ROOT = ROOT / "tasks/control/runs/20260914_chaoyang_cleanup_v6"
GOV = ROOT / "docs/governance"
TASK_STATE_PATH = GOV / "LONG_HORIZON_TASK_STATE.json"
CURRENT_FILES = (
    GOV / "CURRENT_STATUS_RECEIPT.json",
    GOV / "CURRENT_AUTHORITY_INDEX.json",
    GOV / "LONG_HORIZON_TASK_STATE.json",
    GOV / "CURRENT_PROJECT_STATUS_MIN.json",
    GOV / "CURRENT_BASELINE_REGISTRY_V2.json",
    GOV / "CURRENT_FILE_LAYOUT.json",
)
ABSOLUTE_PROTECTED_PREFIXES = (
    Path("/mnt/data/egodata"),
    Path("/nas/chenxianchi/egosteertouch"),
    Path("/nas/chenxianchi/egoverse_piper"),
    Path("/nas/chenxianchi/openpi"),
    Path("/nas/chenxianchi/tactiel_pretrain_outputs"),
)
LIVE_CLEAN_PREFIXES = (
    ROOT / "tasks/control/runs/20260913_exact78_v3_wave_clean_v1",
    ROOT / "tasks/control/runs/20260913_exact78_v52/lane_a_clean_successor_v521",
    ROOT / "tasks/control/runs/20260913_exact78_v52/task_packets/exact78_v52_lane_a_clean_successor_v521",
    ROOT / "third_party/ProPainter",
)
LIVE_CLEAN_FILES = (
    ROOT / "tools/run_exact78_v52_clean_lane_successor_v521.py",
    ROOT / "tools/run_exact78_clean_wave_guardian_v52_1.py",
    ROOT / "tools/prepare_exact78_clean_wave_session_v52.py",
    ROOT / "tools/run_generic_same_session_real_donor_v1.py",
    ROOT / "tools/validate_generic_same_session_real_donor_v1.py",
    ROOT / "tools/launch_clean_synthetic_propainter_once.py",
    ROOT / "tools/run_clean_synthetic_propainter_baseline.py",
    ROOT / "tools/prefetch_exact78_clean_prepare_v521.py",
    ROOT / "_run/GPU_LEASE.json",
    ROOT / "_run/GPU_LEASE.lock",
)
CLEANUP_CONTROL_FILES = (
    ROOT / "tools/__init__.py",
    ROOT / "tools/cleanup_current_only_v6.py",
    ROOT / "tools/cleanup_now_legacy_v6.py",
    ROOT / "tools/migrate_depth_accuracy_package_v6.py",
    ROOT / "tools/build_current_pipeline_document_receipt_v6.py",
    ROOT / "tools/governance/current_baseline_v2.py",
    ROOT / "tools/governance/prune_superseded_evidence_v6.py",
    ROOT / "tools/governance/common.py",
    ROOT / "tools/governance/update_task_state.py",
    ROOT / "tools/governance/validate_governance_state.py",
    ROOT / "tools/governance/create_meeting_snapshot.py",
)
CACHE_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_absolute_paths(value: Any, output: set[Path]) -> None:
    if isinstance(value, dict):
        for child in value.values():
            collect_absolute_paths(child, output)
    elif isinstance(value, list):
        for child in value:
            collect_absolute_paths(child, output)
    elif isinstance(value, str) and value.startswith("/"):
        # Current JSONs may contain thousands of paths on CPFS/OSSFS.  Keep a
        # normalized absolute lexical path here; target-specific deletion gates
        # perform the real filesystem/reference check later.
        candidate = Path(os.path.normpath(value.split("/**", 1)[0]))
        output.add(candidate)


def collect_artifact_ref_paths(value: Any, output: set[Path]) -> None:
    """Collect explicit path/bytes/SHA evidence references from a JSON value."""
    if isinstance(value, dict):
        if (
            isinstance(value.get("path"), str)
            and value["path"].startswith("/")
            and isinstance(value.get("bytes"), int)
            and isinstance(value.get("sha256"), str)
            and len(value["sha256"]) == 64
        ):
            output.add(Path(os.path.normpath(value["path"])))
        for child in value.values():
            collect_artifact_ref_paths(child, output)
    elif isinstance(value, list):
        for child in value:
            collect_artifact_ref_paths(child, output)


def expand_json_reference_closure(source_files: Iterable[Path], protected: set[Path]) -> list[Path]:
    """Follow bounded repo-local JSON references until the path closure is stable."""
    queue = [path.resolve() for path in source_files]
    visited: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in visited or path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        visited.add(path)
        try:
            if not path.is_file() or path.stat().st_size > 50 * 1024 * 1024:
                continue
            if path.suffix.lower() == ".jsonl":
                values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                values = [load(path)]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        before = set(protected)
        for value in values:
            collect_absolute_paths(value, protected)
        for candidate in protected - before:
            try:
                candidate.relative_to(ROOT)
            except ValueError:
                continue
            if candidate.suffix.lower() in {".json", ".jsonl"}:
                queue.append(candidate)
    return sorted(visited)


def git_snapshot() -> dict[str, Any]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() or "DETACHED"
    # Full untracked expansion under run/data trees can exceed 100 MiB and can
    # stall CPFS without adding protection value.  Keep tracked state plus
    # top-level untracked roots, then explicitly enumerate source/document
    # trees where individual user files must be preserved.
    output = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=normal"], cwd=ROOT
    )
    rows = []
    for raw in output.split(b"\0"):
        if len(raw) < 4:
            continue
        rows.append({"xy": raw[:2].decode(errors="replace"), "path": raw[3:].decode(errors="replace")})
    detailed = subprocess.check_output(
        [
            "git", "ls-files", "--others", "--exclude-standard", "-z", "--",
            "tools", "tests", "contracts", "pipeline", "HumanEgo", "docs", "systems",
            "README.md", "AGENTS.md", "THIRD_PARTY_NOTICES.md",
        ],
        cwd=ROOT,
    )
    known = {row["path"] for row in rows}
    for raw in detailed.split(b"\0"):
        if not raw:
            continue
        path = raw.decode(errors="replace")
        if path not in known:
            rows.append({"xy": "??", "path": path})
            known.add(path)
    return {"head": head, "branch": branch, "status": rows}


def process_references(scope: Path = ROOT) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scope_text = str(scope)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        candidates: list[tuple[str, str]] = []
        try:
            candidates.append(("cmdline", (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")))
        except OSError:
            pass
        for name in ("cwd", "exe"):
            try:
                candidates.append((name, os.readlink(proc / name)))
            except OSError:
                pass
        tied = any(scope_text in value for _, value in candidates)
        if tied:
            try:
                for fd in (proc / "fd").iterdir():
                    try:
                        candidates.append((f"fd:{fd.name}", os.readlink(fd)))
                    except OSError:
                        pass
            except OSError:
                pass
        selected = [
            {"source": source, "value": value}
            for source, value in candidates
            if scope_text in value
        ]
        if selected:
            try:
                start_ticks = int((proc / "stat").read_text().split()[21])
            except (OSError, IndexError, ValueError):
                start_ticks = None
            rows.append({"pid": int(proc.name), "start_ticks": start_ticks, "references": selected})
    return rows


def current_task_packets(task_state: dict[str, Any]) -> list[Path]:
    active_ids = {
        item.get("task_id") for item in task_state.get("tasks", [])
        if item.get("status") in {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
    }
    packets = []
    for path in (ROOT / "tasks/control/runs").glob("*/task_packet/TASK_PACKET.json"):
        try:
            if load(path).get("task_id") in active_ids:
                packets.append(path)
        except (OSError, json.JSONDecodeError):
            continue
    for path in (ROOT / "tasks/control/runs/20260913_exact78_v52/task_packets").glob("*/TASK_PACKET.json"):
        try:
            if load(path).get("task_id") in active_ids:
                packets.append(path)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(set(packets))


def active_task_reference_view(task_state: dict[str, Any]) -> dict[str, Any]:
    """Return only live scheduler references, not terminal-task history.

    ``LONG_HORIZON_TASK_STATE`` is itself a protected current artifact, but its
    terminal rows and recent-event history are an audit log, not authority for
    retaining every historical output forever.  Current authority is already
    covered by the authority index/registry.  Cleanup therefore follows only
    live task rows and the scheduler-selected next task from this file.
    """
    live_statuses = {"CLAIMED", "RUNNING", "WAIT_GPU_RESOURCE"}
    active_tasks = [
        item for item in task_state.get("tasks", [])
        if item.get("status") in live_statuses
    ]
    active_ids = {item.get("task_id") for item in active_tasks}
    return {
        "tasks": active_tasks,
        "recent_events": [
            item for item in task_state.get("recent_events", [])
            if item.get("task_id") in active_ids
        ],
        "next_task": task_state.get("next_task"),
    }


def protection_snapshot() -> tuple[dict[str, Any], set[Path]]:
    subprocess.run(["python", "-m", "tools.governance.validate_governance_state"], cwd=ROOT, check=True, capture_output=True)
    protected: set[Path] = set(ABSOLUTE_PROTECTED_PREFIXES) | set(LIVE_CLEAN_PREFIXES) | set(LIVE_CLEAN_FILES)
    source_files = [path for path in CURRENT_FILES if path.is_file()]
    task_state = load(GOV / "LONG_HORIZON_TASK_STATE.json")
    packets = current_task_packets(task_state)
    source_files.extend(packets)
    for path in source_files:
        protected.add(path.absolute())
        if path != TASK_STATE_PATH:
            collect_absolute_paths(load(path), protected)
    collect_absolute_paths(active_task_reference_view(task_state), protected)
    # Preserve one evidence hop below the current authority/registry.  This is
    # needed for manifests such as PREDECESSOR_26_TERMINALS.json whose explicit
    # bytes/SHA rows point at the actual current Depth/Object6D RESULT files.
    # Deliberately do not recurse past that hop: old READMEs and terminal
    # contracts must not create an unbounded retention loop.
    direct_evidence: set[Path] = set()
    for path in (
        GOV / "CURRENT_AUTHORITY_INDEX.json",
        GOV / "CURRENT_BASELINE_REGISTRY_V2.json",
    ):
        if path.is_file():
            collect_artifact_ref_paths(load(path), direct_evidence)
    one_hop_evidence: set[Path] = set()
    for path in direct_evidence:
        if (
            path.is_relative_to(ROOT)
            and path.suffix.lower() == ".json"
            and path.is_file()
            and path.stat().st_size <= 50 * 1024 * 1024
        ):
            try:
                collect_artifact_ref_paths(load(path), one_hop_evidence)
            except (OSError, json.JSONDecodeError):
                continue
    protected.update(direct_evidence)
    protected.update(one_hop_evidence)
    # Current authority protects its directly named immutable evidence.  It
    # must not recursively turn every historical input mentioned by an old
    # RESULT into current authority (the README -> old result -> old data
    # retention loop V6 explicitly removes).  Runtime Task Packets are the
    # exception: their declared read-set is expanded while the task is active.
    expanded_json_files = expand_json_reference_closure(packets, protected)
    processes = process_references()
    for row in processes:
        for reference in row["references"]:
            value = reference["value"]
            if value.startswith(str(ROOT)):
                protected.add(Path(value.split(" (deleted)", 1)[0]).absolute())
    git = git_snapshot()
    for row in git["status"]:
        protected.add((ROOT / row["path"]).absolute())
    receipt = load(GOV / "CURRENT_STATUS_RECEIPT.json")
    snapshot = {
        "schema_version": "chaoyang-cleanup-v6-protection-snapshot-v1",
        "created_at": now_iso(),
        "status": "PASS",
        "host": socket.gethostname(),
        "governance_revision": receipt["governance_revision"],
        "current_files": [artifact(path) for path in source_files],
        "expanded_json_reference_files": [str(path) for path in expanded_json_files],
        "direct_current_evidence_files": sorted(str(path) for path in direct_evidence),
        "one_hop_current_evidence_files": sorted(str(path) for path in one_hop_evidence),
        "git": git,
        "active_processes": processes,
        "protected_paths": sorted(str(path) for path in protected),
        "absolute_protected_prefixes": [str(path) for path in ABSOLUTE_PROTECTED_PREFIXES],
        "live_clean_prefixes": [str(path) for path in LIVE_CLEAN_PREFIXES],
        "claim_limit": "Direct current authority plus active Task-Packet runtime closure only; terminal-history transitive references are not retention authority. Absence is not deletion authority until target-specific reference and live-process checks pass.",
    }
    return snapshot, protected


def module_name(path: Path) -> str | None:
    try:
        relative = path.relative_to(ROOT).with_suffix("")
    except ValueError:
        return None
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def python_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return set()
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
            result.update(f"{node.module}.{alias.name}" for alias in node.names)
    return result


def python_tool_file_literals(path: Path) -> set[Path]:
    """Resolve explicit ``'tool_name.py'`` literals used by dynamic tests."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return set()
    result = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if value.endswith(".py") and "/" not in value and "\\" not in value:
                candidate = ROOT / "tools" / value
                if candidate.is_file():
                    result.add(candidate)
    return result


def import_graph() -> tuple[dict[str, Path], dict[Path, set[Path]]]:
    # Both recursive Path.rglob and a repository-wide rg traversal can descend
    # into large ignored HumanEgo output/model trees on CPFS.  Current source
    # and regression entrypoints live at depth <= 3; vendor/model/output trees
    # are assets, not application candidates.  A bounded non-symlink walk is
    # deterministic and covers the 631 top-level tools plus nested packages.
    files: list[Path] = []
    ignored = {
        ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "vendor",
        "outputs", "output", "checkpoints", "logs", "runs", "wandb",
    }
    for relative in ("tools", "pipeline", "HumanEgo", "tests"):
        start = ROOT / relative
        start_depth = len(start.parts)
        for base, directories, names in os.walk(start, followlinks=False):
            here = Path(base)
            depth = len(here.parts) - start_depth
            directories[:] = [
                name for name in directories
                if name not in ignored
                and not (here / name).is_symlink()
                and depth < 3
            ]
            files.extend(here / name for name in names if name.endswith(".py"))
    modules = {name: path for path in files if (name := module_name(path))}
    # Several retained regression tests intentionally add ``tools/`` to
    # sys.path and import a tool by its bare filename.  Model that supported
    # import form as an alias; otherwise a test can remain current while its
    # direct implementation is incorrectly nominated for retirement.
    for path in files:
        if path.parent == ROOT / "tools":
            modules.setdefault(path.stem, path)
    edges: dict[Path, set[Path]] = {path: set() for path in files}
    for path in files:
        for imported in python_imports(path):
            target = modules.get(imported)
            if target:
                edges[path].add(target)
                continue
            pieces = imported.split(".")
            while len(pieces) > 1:
                pieces.pop()
                if ".".join(pieces) in modules:
                    edges[path].add(modules[".".join(pieces)])
                    break
        edges[path].update(python_tool_file_literals(path))
    return modules, edges


def transitive(start: Iterable[Path], edges: dict[Path, set[Path]]) -> set[Path]:
    seen = {path.resolve() for path in start if path.is_file()}
    stack = list(seen)
    while stack:
        path = stack.pop()
        for target in edges.get(path, set()):
            target = target.resolve()
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


def build_reference_graph(protected: set[Path]) -> dict[str, Any]:
    registry = load(GOV / "CURRENT_BASELINE_REGISTRY_V2.json")
    roots: set[Path] = set()
    for entry in registry.get("entries", []):
        for reference in entry.get("code_closure", []):
            if isinstance(reference, dict) and reference.get("path") and reference.get("bytes") is not None:
                roots.add(Path(reference["path"]).resolve())
    roots.update(path for path in LIVE_CLEAN_FILES if path.suffix == ".py" and path.is_file())
    # Cleanup is an active control-plane task, not one of the 12 scientific
    # pipeline stages.  Its own bounded mutators and the governance publisher
    # must therefore be roots explicitly; otherwise a zero-stage-reference
    # heuristic could nominate the running cleanup implementation for deletion.
    roots.update(path.resolve() for path in CLEANUP_CONTROL_FILES if path.is_file())
    regression_path = GOV / "CURRENT_REGRESSION_MANIFEST.json"
    if regression_path.is_file():
        for row in load(regression_path).get("tests", []):
            candidate = ROOT / row["path"]
            if candidate.is_file():
                roots.add(candidate.resolve())
    _, edges = import_graph()
    current_code = transitive(roots, edges)
    reverse_count: dict[Path, int] = {path.resolve(): 0 for path in edges}
    for targets in edges.values():
        for target in targets:
            reverse_count[target.resolve()] = reverse_count.get(target.resolve(), 0) + 1
    tracked_output = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "tools"], cwd=ROOT
    )
    tracked = {
        (ROOT / raw.decode(errors="replace")).absolute()
        for raw in tracked_output.split(b"\0") if raw
    }
    tool_candidates = []
    for path in sorted((ROOT / "tools").glob("*.py")):
        resolved = path.resolve()
        if resolved in current_code:
            disposition = "KEEP_CURRENT_TRANSITIVE"
        elif reverse_count.get(resolved, 0):
            disposition = "CAPSULE_THEN_DELETE_NONCURRENT_REFERENCE_CLUSTER"
        elif resolved in tracked:
            disposition = "DELETE_CANDIDATE_GIT_RECOVERABLE_ZERO_CURRENT_REFERENCE"
        else:
            disposition = "SOURCE_CAPSULE_THEN_DELETE_UNTRACKED_ZERO_CURRENT_REFERENCE"
        tool_candidates.append({
            "path": str(resolved), "bytes": path.stat().st_size,
            "sha256": sha256(path), "reverse_import_count": reverse_count.get(resolved, 0),
            "git_tracked": resolved in tracked,
            "disposition": disposition,
        })
    return {
        "schema_version": "chaoyang-cleanup-reference-graph-v6",
        "created_at": now_iso(),
        "current_code_roots": sorted(str(path) for path in roots),
        "current_code_transitive": sorted(str(path) for path in current_code),
        "python_edge_count": sum(len(targets) for targets in edges.values()),
        "tool_candidates": tool_candidates,
        "reference_policy": "Only current receipt/authority/task packets, current transitive code, active process references, dirty paths, checkpoints and licenses protect a target.",
    }


def tree_stats(path: Path) -> dict[str, int]:
    if not path.exists() and not path.is_symlink():
        return {"bytes": 0, "files": 0, "directories": 0, "symlinks": 0}
    if path.is_file() or path.is_symlink():
        return {"bytes": path.stat().st_size if path.is_file() else 0, "files": int(path.is_file()), "directories": 0, "symlinks": int(path.is_symlink())}
    result = {"bytes": 0, "files": 0, "directories": 0, "symlinks": 0}
    for base, directories, files in os.walk(path, followlinks=False):
        result["directories"] += len(directories)
        for name in files:
            candidate = Path(base) / name
            try:
                if candidate.is_symlink():
                    result["symlinks"] += 1
                else:
                    result["files"] += 1
                    result["bytes"] += candidate.stat().st_size
            except OSError:
                continue
    return result


def active_target_references(targets: Iterable[Path]) -> list[dict[str, Any]]:
    roots = [str(path.resolve()) for path in targets]
    result = []
    for row in process_references(Path("/")):
        for ref in row["references"]:
            value = ref["value"]
            for root in roots:
                if value == root or value.startswith(root.rstrip("/") + "/") or (ref["source"] == "cmdline" and root in value):
                    result.append({**row, "target": root})
                    break
    return result


def is_within(path: Path, prefix: Path) -> bool:
    try:
        path.resolve().relative_to(prefix.resolve())
        return True
    except ValueError:
        return False


def cache_candidates(protected: set[Path]) -> list[Path]:
    candidates = []
    # Large evidence/data trees are audited as explicit targets in phase two;
    # never recursively walk them merely to find tiny Python caches.
    scan_roots = [
        ROOT / "tools", ROOT / "pipeline", ROOT / "systems",
        ROOT / "HumanEgo", ROOT / "tests", ROOT / "contracts", ROOT / "docs",
    ]
    ignored = {
        ".git", "third_party", "tasks", "data", "archive", "_run", "NOW",
        "vendor", "assets", "models", "outputs", "output", "runs", "wandb",
    }
    # Every member was already normalized while the protection snapshot was
    # assembled.  Resolving every protected path again for every cache
    # directory caused an O(candidates * protected) storm of CPFS/OSSFS
    # metadata calls.  Lexical containment is sufficient here because the
    # cache walker never follows symlinks and only visits the repository root.
    protected_text = {str(path) for path in protected}
    protected_ancestors: set[str] = set()
    for path in protected:
        protected_ancestors.update(str(parent) for parent in (path, *path.parents))
    for direct in CACHE_NAMES:
        candidate = ROOT / direct
        if candidate.exists():
            candidates.append(candidate)
    for start in scan_roots:
        if not start.is_dir():
            continue
        start_depth = len(start.parts)
        for base, directories, _ in os.walk(start, followlinks=False):
            here = Path(base)
            depth = len(here.parts) - start_depth
            selected = [name for name in directories if name in CACHE_NAMES]
            for name in selected:
                candidate = here / name
                candidate_text = str(candidate)
                protected_by_ancestor = any(
                    str(parent) in protected_text for parent in (candidate, *candidate.parents)
                )
                contains_protected_descendant = candidate_text in protected_ancestors
                if not protected_by_ancestor and not contains_protected_descendant:
                    candidates.append(candidate)
            directories[:] = [
                name for name in directories
                if name not in CACHE_NAMES
                and name not in ignored
                and not (here / name).is_symlink()
                and depth < 5
            ]
    return sorted(set(candidates))


def remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def command_snapshot(_: argparse.Namespace) -> int:
    snapshot, protected = protection_snapshot()
    graph = build_reference_graph(protected)
    candidates = {
        "schema_version": "chaoyang-cleanup-candidates-v6",
        "created_at": now_iso(),
        "status": "DRY_RUN_ONLY_WHILE_WAVE0_CLEAN_ACTIVE",
        "cache_candidates": [str(path) for path in cache_candidates(protected)],
        "root_actions": [
            {"path": str(ROOT / "NOW"), "action": "CAPSULE_THEN_DELETE", "gate": "PER_TARGET_ZERO_REFERENCE_AND_ZERO_FD"},
            {"path": str(ROOT / "RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md"), "action": "DELETE_DUPLICATE_AFTER_CONTENT_DIFF"},
            {"path": str(ROOT / "DEPTH_ACCURACY_PACKAGE_20260911"), "action": "MIGRATE_TRANSACTIONALLY"},
            {"path": str(ROOT / "archive/external_nas_legacy_20260914"), "action": "CAPSULE_THEN_COMMIT_DELETE", "gate": "PER_TARGET_ZERO_REFERENCE_AND_ZERO_FD"},
        ],
        "forbidden": [str(path) for path in ABSOLUTE_PROTECTED_PREFIXES],
    }
    atomic_json(RUN_ROOT / "snapshots/CLEANUP_PROTECTION_SNAPSHOT.json", snapshot)
    atomic_json(RUN_ROOT / "CLEANUP_REFERENCE_GRAPH.json", graph)
    atomic_json(RUN_ROOT / "CLEANUP_CANDIDATES.json", candidates)
    print(json.dumps({"status": "PASS", "protected": len(protected), "tools": len(graph["tool_candidates"]), "cache_candidates": len(candidates["cache_candidates"])}, ensure_ascii=False))
    return 0


def command_delete_caches(args: argparse.Namespace) -> int:
    snapshot, protected = protection_snapshot()
    targets = cache_candidates(protected)
    refs = active_target_references(targets)
    if refs:
        raise SystemExit(f"active references block cache deletion: {json.dumps(refs[:10], ensure_ascii=False)}")
    rows = [{"path": str(path), "stats": tree_stats(path)} for path in targets]
    manifest = {
        "schema_version": "chaoyang-cleanup-v6-pre-delete-v1", "created_at": now_iso(),
        "kind": "regenerable_caches", "status": "READY_COMMIT_DELETE" if not refs else "BLOCKED",
        "targets": rows, "git_before": snapshot["git"],
    }
    atomic_json(RUN_ROOT / "receipts/PRE_DELETE_CACHE_MANIFEST.json", manifest)
    if not args.commit_delete:
        print(json.dumps({"status": "DRY_RUN", "targets": len(rows), "bytes": sum(row["stats"]["bytes"] for row in rows)}))
        return 0
    for path in targets:
        remove(path)
    remaining = [str(path) for path in targets if path.exists() or path.is_symlink()]
    after = git_snapshot()
    if {(row["xy"], row["path"]) for row in snapshot["git"]["status"]} != {(row["xy"], row["path"]) for row in after["status"]}:
        raise RuntimeError("git dirty/untracked set changed during cache deletion")
    receipt = {
        "schema_version": "chaoyang-cleanup-v6-deletion-receipt-v1", "completed_at": now_iso(),
        "kind": "regenerable_caches", "status": "PASS_PERMANENT_DELETE" if not remaining else "FAILED_PARTIAL_DELETE",
        "deleted": rows, "remaining": remaining,
        "totals": {key: sum(row["stats"][key] for row in rows) for key in ("bytes", "files", "directories", "symlinks")},
        "git_after": after,
    }
    atomic_json(RUN_ROOT / "receipts/DELETION_CACHE_RECEIPT.json", receipt)
    print(json.dumps({"status": receipt["status"], **receipt["totals"]}, ensure_ascii=False))
    return 0 if not remaining else 2


def command_legacy_manifest(args: argparse.Namespace) -> int:
    target = (ROOT / "archive/external_nas_legacy_20260914").resolve()
    if not target.is_dir():
        print(json.dumps({"status": "ALREADY_ABSENT", "path": str(target)}))
        return 0
    files = []
    root_hash = hashlib.sha256()
    for path in sorted(candidate for candidate in target.rglob("*") if candidate.is_file() and not candidate.is_symlink()):
        row = artifact(path)
        row["relative_path"] = str(path.relative_to(target))
        files.append(row)
        root_hash.update((row["relative_path"] + "\0" + str(row["bytes"]) + "\0" + row["sha256"] + "\n").encode())
    capsule = {
        "schema_version": "chaoyang-legacy-evidence-capsule-v1", "created_at": now_iso(),
        "source_root": str(target), "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files), "merkle_sha256": root_hash.hexdigest(),
        "files": files,
        "retention": "Manifest plus selected source/config/readme/environment and one representative preview per legacy root; bulk data is permanently deleted only after per-target zero-reference and zero-FD proof.",
    }
    atomic_json(RUN_ROOT / "LEGACY_EVIDENCE_CAPSULE.json", capsule)
    print(json.dumps({key: capsule[key] for key in ("file_count", "total_bytes", "merkle_sha256")}, ensure_ascii=False))
    return 0


def _legacy_retention_rows(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_suffixes = {".py", ".sh", ".md", ".rst", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
    named = {
        "README", "README.md", "LICENSE", "LICENSE.txt", "NOTICE", "Dockerfile",
        "requirements.txt", "environment.yml", "environment.yaml", "pyproject.toml",
    }
    retained: dict[str, dict[str, Any]] = {}
    for row in files:
        relative = Path(row["relative_path"])
        if row["bytes"] <= 2 * 1024 * 1024 and (relative.suffix.lower() in source_suffixes or relative.name in named):
            retained[row["relative_path"]] = row
    # Media is represented by a bounded JPEG generated below.  Never put an
    # entire legacy MP4 into the recovery capsule merely because it happens to
    # be the smallest video in that legacy root.
    return [retained[key] for key in sorted(retained)]


def _legacy_preview_sources(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    media_suffixes = {".mp4", ".png", ".jpg", ".jpeg", ".webp"}
    by_root: dict[str, list[dict[str, Any]]] = {}
    for row in files:
        relative = Path(row["relative_path"])
        if relative.suffix.lower() in media_suffixes:
            by_root.setdefault(relative.parts[0], []).append(row)
    return [
        min(candidates, key=lambda row: (row["bytes"], row["relative_path"]))
        for _, candidates in sorted(by_root.items())
    ]


def command_prepare_legacy_delete(args: argparse.Namespace) -> int:
    target = (ROOT / "archive/external_nas_legacy_20260914").resolve()
    full_path = RUN_ROOT / "LEGACY_EVIDENCE_CAPSULE.json"
    # Rebuilding only the bounded preview capsule must not force a second
    # 44-GiB content hash.  Reuse the just-published full Merkle manifest when
    # its target identity plus complete file-count/byte inventory still match.
    reuse_full = False
    if full_path.is_file():
        candidate = load(full_path)
        current_files = sorted(
            path for path in target.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        reuse_full = (
            candidate.get("source_root") == str(target)
            and len(current_files) == candidate.get("file_count")
            and sum(path.stat().st_size for path in current_files) == candidate.get("total_bytes")
        )
    if not reuse_full:
        command_legacy_manifest(args)
    full = load(full_path)
    retained = _legacy_retention_rows(full["files"])
    preview_sources = _legacy_preview_sources(full["files"])
    capsule_dir = ROOT / "docs/history/2026-09-14/external_nas_legacy_v6"
    capsule_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = capsule_dir / "representative_previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True)
    generated_previews: list[dict[str, Any]] = []
    for row in preview_sources:
        source = target / row["relative_path"]
        legacy_root = Path(row["relative_path"]).parts[0]
        output = preview_dir / f"{legacy_root}.jpg"
        command = [
            "ffmpeg", "-v", "error", "-y",
            "-i", str(source), "-frames:v", "1",
            "-vf", "scale='min(640,iw)':-2", "-q:v", "3", str(output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError(
                f"failed to generate bounded legacy preview for {source}: "
                f"{completed.stderr[-500:]}"
            )
        generated_previews.append({
            "source": {key: row[key] for key in ("path", "bytes", "sha256")},
            "preview": artifact(output),
        })
    archive = capsule_dir / "SOURCE_CONFIG_AND_REPRESENTATIVE_PREVIEWS.tar.gz"
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=6) as bundle:
        for row in retained:
            source = target / row["relative_path"]
            info = bundle.gettarinfo(str(source), arcname=row["relative_path"])
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                bundle.addfile(info, stream)
        for row in generated_previews:
            source = Path(row["preview"]["path"])
            info = bundle.gettarinfo(str(source), arcname=f"representative_previews/{source.name}")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                bundle.addfile(info, stream)
    os.chmod(temporary, 0o444)
    os.replace(temporary, archive)
    pre_delete = {
        "schema_version": "chaoyang-external-legacy-pre-delete-v1",
        "created_at": now_iso(),
        "status": "READY_COMMIT_DELETE_AFTER_LIVE_REFERENCE_RECHECK",
        "target": str(target),
        "full_manifest": artifact(RUN_ROOT / "LEGACY_EVIDENCE_CAPSULE.json"),
        "file_count": full["file_count"],
        "total_bytes": full["total_bytes"],
        "merkle_sha256": full["merkle_sha256"],
        "retained_file_count": len(retained),
        "retained_source_bytes": sum(row["bytes"] for row in retained),
        "retained_archive": artifact(archive),
        "retained_files": retained,
        "generated_previews": generated_previews,
        "claim_limit": "Navigation/source/configuration capsule plus one generated bounded JPEG per legacy root containing media; source media identity remains in the full path/bytes/SHA manifest. Removed bulk is not current authority.",
    }
    atomic_json(capsule_dir / "LEGACY_EVIDENCE_CAPSULE.json", pre_delete)
    atomic_json(RUN_ROOT / "receipts/PRE_DELETE_EXTERNAL_LEGACY_MANIFEST.json", pre_delete)
    print(json.dumps({
        "status": pre_delete["status"], "files": pre_delete["file_count"],
        "bytes": pre_delete["total_bytes"], "retained": pre_delete["retained_file_count"],
        "archive": pre_delete["retained_archive"],
    }, ensure_ascii=False))
    return 0


def command_delete_legacy(args: argparse.Namespace) -> int:
    if not args.commit_delete:
        raise SystemExit("delete-legacy requires --commit-delete")
    subprocess.run(
        [sys.executable, "-m", "tools.governance.validate_governance_state"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    target = (ROOT / "archive/external_nas_legacy_20260914").resolve()
    pre_path = RUN_ROOT / "receipts/PRE_DELETE_EXTERNAL_LEGACY_MANIFEST.json"
    if not target.is_dir() or not pre_path.is_file():
        raise SystemExit("BLOCKED: target or pre-delete manifest missing")
    pre = load(pre_path)
    retained_archive = Path(pre["retained_archive"]["path"])
    if artifact(retained_archive) != pre["retained_archive"]:
        raise SystemExit("BLOCKED: retained legacy capsule mismatch")
    if active_target_references([target]):
        raise SystemExit("BLOCKED: active process references legacy target")
    direct_current_refs: list[str] = []
    for current in CURRENT_FILES:
        if current.is_file() and str(target) in current.read_text(encoding="utf-8", errors="replace"):
            direct_current_refs.append(str(current))
    if direct_current_refs:
        raise SystemExit(f"BLOCKED: current governance references target: {direct_current_refs}")
    current_files = sorted(path for path in target.rglob("*") if path.is_file() and not path.is_symlink())
    if len(current_files) != pre["file_count"] or sum(path.stat().st_size for path in current_files) != pre["total_bytes"]:
        raise SystemExit("BLOCKED: legacy target count/bytes changed after manifest")
    shutil.rmtree(target)
    receipt = {
        "schema_version": "chaoyang-cleanup-v6-deletion-receipt-v1",
        "completed_at": now_iso(),
        "kind": "external_nas_legacy_bulk",
        "status": "PASS_PERMANENT_DELETE_WITH_EVIDENCE_CAPSULE" if not target.exists() else "FAILED_PARTIAL_DELETE",
        "deleted_root": str(target),
        "deleted_file_count": pre["file_count"],
        "deleted_bytes": pre["total_bytes"],
        "full_manifest": pre["full_manifest"],
        "retained_archive": pre["retained_archive"],
    }
    atomic_json(RUN_ROOT / "receipts/DELETION_EXTERNAL_LEGACY_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if not target.exists() else 2


def _tool_delete_candidates(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row for row in graph["tool_candidates"]
        if row["disposition"] != "KEEP_CURRENT_TRANSITIVE"
    ]


def command_tools_manifest(_: argparse.Namespace) -> int:
    _, protected = protection_snapshot()
    graph = build_reference_graph(protected)
    rows = _tool_delete_candidates(graph)
    capsule_dir = ROOT / "docs/history/2026-09-14/legacy_tools_v6"
    capsule_dir.mkdir(parents=True, exist_ok=True)
    archive = capsule_dir / "LEGACY_TOOL_SOURCES.tar.gz"
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=9) as bundle:
        for row in rows:
            source = Path(row["path"])
            info = bundle.gettarinfo(str(source), arcname=str(source.relative_to(ROOT)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                bundle.addfile(info, stream)
    os.chmod(temporary, 0o444)
    os.replace(temporary, archive)
    manifest = {
        "schema_version": "chaoyang-legacy-tool-source-capsule-v1",
        "created_at": now_iso(),
        "status": "READY_AFTER_PER_TARGET_REFERENCE_RECHECK",
        "source_root": str(ROOT / "tools"),
        "source_file_count": len(rows),
        "source_bytes": sum(row["bytes"] for row in rows),
        "archive": artifact(archive),
        "files": rows,
        "current_code_transitive": graph["current_code_transitive"],
        "claim_limit": "Source-only recovery capsule; not executable current code and not a replacement for scientific evidence.",
    }
    atomic_json(capsule_dir / "LEGACY_EVIDENCE_CAPSULE.json", manifest)
    atomic_json(RUN_ROOT / "receipts/PRE_DELETE_TOOL_MANIFEST.json", manifest)
    print(json.dumps({
        "status": manifest["status"], "files": len(rows),
        "source_bytes": manifest["source_bytes"], "archive": manifest["archive"],
    }, ensure_ascii=False))
    return 0


def command_delete_tools(args: argparse.Namespace) -> int:
    if not args.commit_delete:
        raise SystemExit("delete-tools requires --commit-delete")
    subprocess.run(
        [sys.executable, "-m", "tools.governance.validate_governance_state"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    manifest_path = RUN_ROOT / "receipts/PRE_DELETE_TOOL_MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit("BLOCKED: PRE_DELETE_TOOL_MANIFEST.json missing")
    manifest = load(manifest_path)
    archive_ref = manifest["archive"]
    archive = Path(archive_ref["path"])
    if not archive.is_file() or artifact(archive) != archive_ref:
        raise SystemExit("BLOCKED: source capsule archive mismatch")
    snapshot, protected = protection_snapshot()
    graph = build_reference_graph(protected)
    rows = _tool_delete_candidates(graph)
    expected = {(row["path"], row["bytes"], row["sha256"]) for row in manifest["files"]}
    observed = {(row["path"], row["bytes"], row["sha256"]) for row in rows}
    if observed != expected:
        raise SystemExit("BLOCKED: current tool candidates differ from frozen manifest")
    targets = [Path(row["path"]) for row in rows]
    refs = active_target_references(targets)
    if refs:
        raise SystemExit(f"BLOCKED: active process references: {json.dumps(refs[:10], ensure_ascii=False)}")
    for row in rows:
        path = Path(row["path"])
        current = artifact(path)
        if current["bytes"] != row["bytes"] or current["sha256"] != row["sha256"]:
            raise SystemExit(f"BLOCKED: source changed after manifest: {path}")
    for path in targets:
        path.unlink()
    remaining = [str(path) for path in targets if path.exists()]
    receipt = {
        "schema_version": "chaoyang-cleanup-v6-deletion-receipt-v1",
        "completed_at": now_iso(),
        "kind": "noncurrent_top_level_tool_sources",
        "status": "PASS_PERMANENT_DELETE_WITH_SOURCE_CAPSULE" if not remaining else "FAILED_PARTIAL_DELETE",
        "deleted_file_count": len(rows) - len(remaining),
        "deleted_bytes": sum(row["bytes"] for row in rows if row["path"] not in remaining),
        "source_capsule": archive_ref,
        "remaining": remaining,
        "git_before": snapshot["git"],
        "git_after": git_snapshot(),
    }
    atomic_json(RUN_ROOT / "receipts/DELETION_TOOLS_RECEIPT.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "deleted_file_count", "deleted_bytes")}, ensure_ascii=False))
    return 0 if not remaining else 2


def _path_is_under(path: Path, root: Path) -> bool:
    # Inputs are normalized absolute paths.  Avoid Path.resolve(): the protected
    # set also includes CPFS/OSSFS evidence, and resolving it inside an N x M
    # partition loop turns a local cleanup preflight into minutes of remote
    # metadata requests.  Symlink safety is enforced separately when targets
    # are inventoried and by the active FD/CWD gate.
    candidate = Path(os.path.normpath(str(path)))
    parent = Path(os.path.normpath(str(root)))
    try:
        candidate.relative_to(parent)
        return True
    except ValueError:
        return False


def _run_root_partition(protected: set[Path], graph: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    runs = ROOT / "tasks/control/runs"
    all_roots = sorted(path.resolve() for path in runs.iterdir() if path.is_dir())
    code_text = "\n".join(
        Path(path).read_text(encoding="utf-8", errors="replace")
        for path in graph["current_code_transitive"] if Path(path).is_file()
    )
    active_targets = {Path(row["target"]).resolve() for row in active_target_references(all_roots)}
    kept: list[Path] = []
    candidates: list[Path] = []
    for run in all_roots:
        relative = str(run.relative_to(ROOT))
        referenced = any(_path_is_under(path, run) for path in protected)
        referenced = referenced or run in active_targets or str(run) in code_text or relative in code_text
        (kept if referenced else candidates).append(run)
    return kept, candidates


def _inventory_roots(roots: list[Path]) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    merkle = hashlib.sha256()
    for root in roots:
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and not candidate.is_symlink()):
            row = artifact(path)
            try:
                row["relative_path"] = str(path.relative_to(ROOT))
            except ValueError:
                row["relative_path"] = str(path).lstrip("/")
            rows.append(row)
            merkle.update((row["relative_path"] + "\0" + str(row["bytes"]) + "\0" + row["sha256"] + "\n").encode())
    return rows, merkle.hexdigest()


def _bounded_retention_rows(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text_suffixes = {".py", ".sh", ".md", ".rst", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".json", ".jsonl", ".txt"}
    retained: dict[str, dict[str, Any]] = {}
    for row in files:
        relative = Path(row["relative_path"])
        if row["bytes"] <= 2 * 1024 * 1024 and relative.suffix.lower() in text_suffixes:
            retained[row["relative_path"]] = row
    media_suffixes = {".mp4", ".png", ".jpg", ".jpeg", ".webp"}
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in files:
        relative = Path(row["relative_path"])
        if relative.suffix.lower() in media_suffixes and len(relative.parts) >= 4:
            by_run.setdefault(relative.parts[3], []).append(row)
    for candidates in by_run.values():
        representative = min(candidates, key=lambda row: (row["bytes"], row["relative_path"]))
        retained[representative["relative_path"]] = representative
    return [retained[key] for key in sorted(retained)]


def command_prepare_runs_delete(_: argparse.Namespace) -> int:
    snapshot, protected = protection_snapshot()
    graph = build_reference_graph(protected)
    kept, candidates = _run_root_partition(protected, graph)
    files, merkle = _inventory_roots(candidates)
    retained = _bounded_retention_rows(files)
    capsule_dir = ROOT / "docs/history/2026-09-14/retired_runs_v6"
    capsule_dir.mkdir(parents=True, exist_ok=True)
    archive = capsule_dir / "RETIRED_RUN_TEXT_AND_REPRESENTATIVE_VISUALS.tar.gz"
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=6) as bundle:
        for row in retained:
            source = ROOT / row["relative_path"]
            info = bundle.gettarinfo(str(source), arcname=row["relative_path"])
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                bundle.addfile(info, stream)
    os.chmod(temporary, 0o444)
    os.replace(temporary, archive)
    value = {
        "schema_version": "chaoyang-retired-runs-pre-delete-v1",
        "created_at": now_iso(),
        "status": "READY_COMMIT_DELETE_AFTER_LIVE_REFERENCE_RECHECK",
        "candidate_roots": [str(path) for path in candidates],
        "kept_roots": [str(path) for path in kept],
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "merkle_sha256": merkle,
        "files": files,
        "retained_file_count": len(retained),
        "retained_archive": artifact(archive),
        "claim_limit": "Retired non-current run evidence capsule; current authority roots and transitive references are excluded.",
    }
    atomic_json(capsule_dir / "LEGACY_EVIDENCE_CAPSULE.json", value)
    atomic_json(RUN_ROOT / "receipts/PRE_DELETE_RUNS_MANIFEST.json", value)
    print(json.dumps({
        "status": value["status"], "candidate_roots": len(candidates),
        "kept_roots": len(kept), "files": len(files), "bytes": value["total_bytes"],
    }, ensure_ascii=False))
    return 0


def command_delete_runs(args: argparse.Namespace) -> int:
    if not args.commit_delete:
        raise SystemExit("delete-runs requires --commit-delete")
    subprocess.run(
        [sys.executable, "-m", "tools.governance.validate_governance_state"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    pre_path = RUN_ROOT / "receipts/PRE_DELETE_RUNS_MANIFEST.json"
    if not pre_path.is_file():
        raise SystemExit("BLOCKED: PRE_DELETE_RUNS_MANIFEST missing")
    pre = load(pre_path)
    snapshot, protected = protection_snapshot()
    graph = build_reference_graph(protected)
    _, candidates = _run_root_partition(protected, graph)
    if [str(path) for path in candidates] != pre["candidate_roots"]:
        raise SystemExit("BLOCKED: run candidate roots changed after manifest")
    if active_target_references(candidates):
        raise SystemExit("BLOCKED: active process references candidate run root")
    files, merkle = _inventory_roots(candidates)
    if len(files) != pre["file_count"] or sum(row["bytes"] for row in files) != pre["total_bytes"] or merkle != pre["merkle_sha256"]:
        raise SystemExit("BLOCKED: candidate run content changed after manifest")
    if artifact(Path(pre["retained_archive"]["path"])) != pre["retained_archive"]:
        raise SystemExit("BLOCKED: retired run capsule mismatch")
    for target in candidates:
        shutil.rmtree(target)
    remaining = [str(path) for path in candidates if path.exists()]
    receipt = {
        "schema_version": "chaoyang-cleanup-v6-deletion-receipt-v1",
        "completed_at": now_iso(),
        "kind": "retired_noncurrent_runs",
        "status": "PASS_PERMANENT_DELETE_WITH_EVIDENCE_CAPSULE" if not remaining else "FAILED_PARTIAL_DELETE",
        "deleted_root_count": len(candidates) - len(remaining),
        "deleted_file_count": pre["file_count"],
        "deleted_bytes": pre["total_bytes"],
        "retained_archive": pre["retained_archive"],
        "remaining": remaining,
    }
    atomic_json(RUN_ROOT / "receipts/DELETION_RUNS_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if not remaining else 2


def _archive_child_partition(protected: set[Path]) -> tuple[list[Path], list[Path]]:
    archive_root = ROOT / "archive"
    kept: list[Path] = []
    candidates: list[Path] = []
    children = sorted(path.resolve() for path in archive_root.iterdir() if path.is_dir())
    active_targets = {Path(row["target"]).resolve() for row in active_target_references(children)}
    for child in children:
        referenced = any(_path_is_under(path, child) for path in protected)
        referenced = referenced or child in active_targets
        (kept if referenced else candidates).append(child)
    return kept, candidates


def _archive_retention_rows(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_suffixes = {".py", ".sh", ".md", ".rst", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
    named = {
        "README", "README.md", "LICENSE", "LICENSE.txt", "NOTICE", "Dockerfile",
        "requirements.txt", "environment.yml", "environment.yaml", "pyproject.toml",
    }
    return [
        row for row in files
        if row["bytes"] <= 2 * 1024 * 1024
        and (Path(row["relative_path"]).suffix.lower() in source_suffixes
             or Path(row["relative_path"]).name in named)
    ]


def _archive_preview_sources(files: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    media_suffixes = {".mp4", ".png", ".jpg", ".jpeg", ".webp"}
    by_root: dict[str, list[dict[str, Any]]] = {}
    for row in files:
        relative = Path(row["relative_path"])
        if relative.suffix.lower() in media_suffixes and len(relative.parts) >= 2:
            by_root.setdefault(relative.parts[1], []).append(row)
    # Prefer still images because historical ``.mp4`` files occasionally contain
    # receipts/placeholders without a video stream.  Keep all candidates ordered
    # so the capsule builder can fall through after a decode failure instead of
    # retaining an entire obsolete archive merely because its smallest MP4 is bad.
    still_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    return {
        root: sorted(
            candidates,
            key=lambda row: (
                Path(row["relative_path"]).suffix.lower() not in still_suffixes,
                row["bytes"],
                row["relative_path"],
            ),
        )
        for root, candidates in sorted(by_root.items())
    }


def command_prepare_archive_delete(_: argparse.Namespace) -> int:
    _, protected = protection_snapshot()
    kept, candidates = _archive_child_partition(protected)
    files, merkle = _inventory_roots(candidates)
    retained = _archive_retention_rows(files)
    preview_sources = _archive_preview_sources(files)
    capsule_dir = ROOT / "docs/history/2026-09-14/internal_archive_legacy_v6"
    capsule_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = capsule_dir / "representative_previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True)
    generated_previews: list[dict[str, Any]] = []
    preview_failures: list[dict[str, Any]] = []
    for archive_child, rows in preview_sources.items():
        output = preview_dir / f"{archive_child}.jpg"
        selected: dict[str, Any] | None = None
        for row in rows:
            source = Path(row["path"])
            completed = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-y", "-i", str(source),
                    "-frames:v", "1", "-vf", "scale='min(640,iw)':-2",
                    "-q:v", "3", str(output),
                ],
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0 and output.is_file() and output.stat().st_size:
                selected = row
                break
            output.unlink(missing_ok=True)
            preview_failures.append({
                "source": {key: row[key] for key in ("path", "bytes", "sha256")},
                "reason": "NO_DECODABLE_VIDEO_OR_IMAGE_STREAM",
                "ffmpeg_stderr_tail": completed.stderr[-500:],
            })
        if selected is not None:
            generated_previews.append({
                "source": {key: selected[key] for key in ("path", "bytes", "sha256")},
                "preview": artifact(output),
            })
    bundle_path = capsule_dir / "SOURCE_DOCS_AND_REPRESENTATIVE_PREVIEWS.tar.gz"
    temporary = bundle_path.with_name(f".{bundle_path.name}.{os.getpid()}.tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=6) as bundle:
        for row in retained:
            source = Path(row["path"])
            info = bundle.gettarinfo(str(source), arcname=row["relative_path"])
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                bundle.addfile(info, stream)
        for row in generated_previews:
            source = Path(row["preview"]["path"])
            info = bundle.gettarinfo(str(source), arcname=f"representative_previews/{source.name}")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                bundle.addfile(info, stream)
    os.chmod(temporary, 0o444)
    os.replace(temporary, bundle_path)
    value = {
        "schema_version": "chaoyang-internal-archive-pre-delete-v1",
        "created_at": now_iso(),
        "status": "READY_COMMIT_DELETE_AFTER_LIVE_REFERENCE_RECHECK",
        "candidate_roots": [str(path) for path in candidates],
        "kept_roots": [str(path) for path in kept],
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "merkle_sha256": merkle,
        "files": files,
        "retained_file_count": len(retained),
        "generated_previews": generated_previews,
        "preview_failures": preview_failures,
        "retained_archive": artifact(bundle_path),
        "claim_limit": "Retired internal archive navigation/source/document capsule plus one bounded JPEG per archived child containing media; no current authority is represented by the removed bulk.",
    }
    atomic_json(capsule_dir / "LEGACY_EVIDENCE_CAPSULE.json", value)
    atomic_json(RUN_ROOT / "receipts/PRE_DELETE_INTERNAL_ARCHIVE_MANIFEST.json", value)
    print(json.dumps({
        "status": value["status"], "candidate_roots": len(candidates),
        "kept_roots": len(kept), "files": len(files), "bytes": value["total_bytes"],
        "capsule_bytes": value["retained_archive"]["bytes"],
    }, ensure_ascii=False))
    return 0


def command_delete_archive(args: argparse.Namespace) -> int:
    if not args.commit_delete:
        raise SystemExit("delete-archive requires --commit-delete")
    subprocess.run(
        [sys.executable, "-m", "tools.governance.validate_governance_state"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    pre_path = RUN_ROOT / "receipts/PRE_DELETE_INTERNAL_ARCHIVE_MANIFEST.json"
    if not pre_path.is_file():
        raise SystemExit("BLOCKED: internal archive pre-delete manifest missing")
    pre = load(pre_path)
    _, protected = protection_snapshot()
    _, candidates = _archive_child_partition(protected)
    if [str(path) for path in candidates] != pre["candidate_roots"]:
        raise SystemExit("BLOCKED: internal archive candidate roots changed")
    if active_target_references(candidates):
        raise SystemExit("BLOCKED: active process references internal archive candidate")
    files, merkle = _inventory_roots(candidates)
    if (
        len(files) != pre["file_count"]
        or sum(row["bytes"] for row in files) != pre["total_bytes"]
        or merkle != pre["merkle_sha256"]
    ):
        raise SystemExit("BLOCKED: internal archive content changed after manifest")
    if artifact(Path(pre["retained_archive"]["path"])) != pre["retained_archive"]:
        raise SystemExit("BLOCKED: internal archive evidence capsule mismatch")
    for target in candidates:
        shutil.rmtree(target)
    remaining = [str(path) for path in candidates if path.exists()]
    receipt = {
        "schema_version": "chaoyang-cleanup-v6-deletion-receipt-v1",
        "completed_at": now_iso(),
        "kind": "retired_internal_archive_bulk",
        "status": "PASS_PERMANENT_DELETE_WITH_EVIDENCE_CAPSULE" if not remaining else "FAILED_PARTIAL_DELETE",
        "deleted_root_count": len(candidates) - len(remaining),
        "deleted_file_count": pre["file_count"],
        "deleted_bytes": pre["total_bytes"],
        "retained_archive": pre["retained_archive"],
        "remaining": remaining,
    }
    atomic_json(RUN_ROOT / "receipts/DELETION_INTERNAL_ARCHIVE_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if not remaining else 2


def _runtime_child_partition(protected: set[Path], graph: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    runtime = ROOT / "_run"
    permanent_names = {"GPU_LEASE.json", "GPU_LEASE.lock", ".gpu_lease.dirlock", ".gitkeep", "README.md"}
    code_text = "\n".join(
        Path(path).read_text(encoding="utf-8", errors="replace")
        for path in graph["current_code_transitive"] if Path(path).is_file()
    )
    kept: list[Path] = []
    candidates: list[Path] = []
    for child in sorted(path.resolve() for path in runtime.iterdir()):
        referenced = child.name in permanent_names
        referenced = referenced or any(_path_is_under(path, child) for path in protected)
        referenced = referenced or str(child) in code_text or str(child.relative_to(ROOT)) in code_text
        (kept if referenced else candidates).append(child)
    return kept, candidates


def command_prepare_runtime_delete(_: argparse.Namespace) -> int:
    snapshot, protected = protection_snapshot()
    graph = build_reference_graph(protected)
    kept, candidates = _runtime_child_partition(protected, graph)
    files, merkle = _inventory_roots([path for path in candidates if path.is_dir()])
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            row = artifact(path)
            row["relative_path"] = str(path.relative_to(ROOT))
            files.append(row)
            merkle = hashlib.sha256((merkle + json.dumps(row, sort_keys=True)).encode()).hexdigest()
    retained = _bounded_retention_rows(files)
    capsule_dir = ROOT / "docs/history/2026-09-14/runtime_legacy_v6"
    capsule_dir.mkdir(parents=True, exist_ok=True)
    archive = capsule_dir / "RUNTIME_TEXT_AND_REPRESENTATIVE_VISUALS.tar.gz"
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=6) as bundle:
        for row in retained:
            source = ROOT / row["relative_path"]
            info = bundle.gettarinfo(str(source), arcname=row["relative_path"])
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                bundle.addfile(info, stream)
    os.chmod(temporary, 0o444)
    os.replace(temporary, archive)
    value = {
        "schema_version": "chaoyang-runtime-pre-delete-v1",
        "created_at": now_iso(),
        "status": "READY_COMMIT_DELETE_AFTER_LIVE_REFERENCE_RECHECK",
        "candidate_children": [str(path) for path in candidates],
        "kept_children": [str(path) for path in kept],
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "content_digest": merkle,
        "files": files,
        "retained_archive": artifact(archive),
        "claim_limit": "Old runtime payload only; permanent lease/lock names and all current references are excluded.",
    }
    atomic_json(capsule_dir / "LEGACY_EVIDENCE_CAPSULE.json", value)
    atomic_json(RUN_ROOT / "receipts/PRE_DELETE_RUNTIME_MANIFEST.json", value)
    print(json.dumps({
        "status": value["status"], "candidates": len(candidates),
        "kept": len(kept), "files": len(files), "bytes": value["total_bytes"],
    }, ensure_ascii=False))
    return 0


def command_delete_runtime(args: argparse.Namespace) -> int:
    if not args.commit_delete:
        raise SystemExit("delete-runtime requires --commit-delete")
    subprocess.run(
        [sys.executable, "-m", "tools.governance.validate_governance_state"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    pre_path = RUN_ROOT / "receipts/PRE_DELETE_RUNTIME_MANIFEST.json"
    if not pre_path.is_file():
        raise SystemExit("BLOCKED: PRE_DELETE_RUNTIME_MANIFEST missing")
    pre = load(pre_path)
    snapshot, protected = protection_snapshot()
    graph = build_reference_graph(protected)
    _, candidates = _runtime_child_partition(protected, graph)
    if [str(path) for path in candidates] != pre["candidate_children"]:
        raise SystemExit("BLOCKED: runtime candidate children changed after manifest")
    if active_target_references(candidates):
        raise SystemExit("BLOCKED: active process references runtime candidate")
    files, _ = _inventory_roots([path for path in candidates if path.is_dir()])
    for path in candidates:
        if path.is_file() and not path.is_symlink():
            row = artifact(path)
            row["relative_path"] = str(path.relative_to(ROOT))
            files.append(row)
    observed = {(row["relative_path"], row["bytes"], row["sha256"]) for row in files}
    expected = {(row["relative_path"], row["bytes"], row["sha256"]) for row in pre["files"]}
    if observed != expected:
        raise SystemExit("BLOCKED: runtime candidate content changed after manifest")
    if artifact(Path(pre["retained_archive"]["path"])) != pre["retained_archive"]:
        raise SystemExit("BLOCKED: runtime capsule mismatch")
    for target in candidates:
        remove(target)
    remaining = [str(path) for path in candidates if path.exists() or path.is_symlink()]
    receipt = {
        "schema_version": "chaoyang-cleanup-v6-deletion-receipt-v1",
        "completed_at": now_iso(),
        "kind": "legacy_runtime_payload",
        "status": "PASS_PERMANENT_DELETE_WITH_EVIDENCE_CAPSULE" if not remaining else "FAILED_PARTIAL_DELETE",
        "deleted_child_count": len(candidates) - len(remaining),
        "deleted_file_count": pre["file_count"],
        "deleted_bytes": pre["total_bytes"],
        "retained_archive": pre["retained_archive"],
        "remaining": remaining,
    }
    atomic_json(RUN_ROOT / "receipts/DELETION_RUNTIME_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if not remaining else 2


EXTERNAL_PROJECT_CLONES = (
    Path("/mnt/workspace/code/HaWoR"),
    Path("/mnt/workspace/code/HumanEgo"),
)


def _git_identity(path: Path) -> dict[str, Any]:
    def output(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

    return {
        "head": output("rev-parse", "HEAD"),
        "remote": output("remote", "get-url", "origin"),
        "status_porcelain": output("status", "--porcelain=v1", "--untracked-files=all").splitlines(),
        "diff": output("diff", "--binary", "HEAD"),
    }


def command_prepare_external_clones(_: argparse.Namespace) -> int:
    roots = [path.resolve() for path in EXTERNAL_PROJECT_CLONES if path.is_dir()]
    if len(roots) != len(EXTERNAL_PROJECT_CLONES):
        raise SystemExit("BLOCKED: expected external project clone missing")
    replacements = {
        str(Path("/mnt/workspace/code/HaWoR")): ROOT / "third_party/HaWoR",
        str(Path("/mnt/workspace/code/HumanEgo")): ROOT / "HumanEgo",
    }
    for source, replacement in replacements.items():
        if not replacement.is_dir():
            raise SystemExit(f"BLOCKED: current in-repository replacement missing for {source}")
    files, merkle = _inventory_roots(roots)
    retained = _bounded_retention_rows(files)
    capsule_dir = ROOT / "docs/history/2026-09-14/external_project_clones_v6"
    capsule_dir.mkdir(parents=True, exist_ok=True)
    archive = capsule_dir / "EXTERNAL_CLONE_SOURCE_CONFIG.tar.gz"
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=6) as bundle:
        for row in retained:
            source = Path(row["path"])
            source_root = next(root for root in roots if _path_is_under(source, root))
            arcname = str(Path(source_root.name) / source.relative_to(source_root))
            info = bundle.gettarinfo(str(source), arcname=arcname)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                bundle.addfile(info, stream)
    os.chmod(temporary, 0o444)
    os.replace(temporary, archive)
    value = {
        "schema_version": "chaoyang-external-project-clones-pre-delete-v1",
        "created_at": now_iso(),
        "status": "READY_COMMIT_DELETE_AFTER_LIVE_REFERENCE_RECHECK",
        "roots": [str(path) for path in roots],
        "replacement_roots": {source: str(target) for source, target in replacements.items()},
        "git_identity": {str(path): _git_identity(path) for path in roots},
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "merkle_sha256": merkle,
        "files": files,
        "retained_archive": artifact(archive),
        "retained_file_count": len(retained),
        "claim_limit": "External duplicate clone recovery capsule; current implementations remain in Chaoyang third_party/HaWoR and HumanEgo.",
    }
    atomic_json(capsule_dir / "LEGACY_EVIDENCE_CAPSULE.json", value)
    atomic_json(RUN_ROOT / "receipts/PRE_DELETE_EXTERNAL_CLONES_MANIFEST.json", value)
    print(json.dumps({
        "status": value["status"], "roots": len(roots),
        "files": len(files), "bytes": value["total_bytes"],
    }, ensure_ascii=False))
    return 0


def command_delete_external_clones(args: argparse.Namespace) -> int:
    if not args.commit_delete:
        raise SystemExit("delete-external-clones requires --commit-delete")
    subprocess.run(
        [sys.executable, "-m", "tools.governance.validate_governance_state"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    pre_path = RUN_ROOT / "receipts/PRE_DELETE_EXTERNAL_CLONES_MANIFEST.json"
    if not pre_path.is_file():
        raise SystemExit("BLOCKED: external clone manifest missing")
    pre = load(pre_path)
    roots = [Path(path) for path in pre["roots"]]
    if active_target_references(roots):
        raise SystemExit("BLOCKED: active process references external clone")
    for current in CURRENT_FILES:
        text = current.read_text(encoding="utf-8", errors="replace")
        if any(str(root) in text for root in roots):
            raise SystemExit(f"BLOCKED: current governance references external clone in {current}")
    files, merkle = _inventory_roots(roots)
    if len(files) != pre["file_count"] or sum(row["bytes"] for row in files) != pre["total_bytes"] or merkle != pre["merkle_sha256"]:
        raise SystemExit("BLOCKED: external clone content changed after manifest")
    if artifact(Path(pre["retained_archive"]["path"])) != pre["retained_archive"]:
        raise SystemExit("BLOCKED: external clone capsule mismatch")
    for target in roots:
        shutil.rmtree(target)
    remaining = [str(path) for path in roots if path.exists()]
    receipt = {
        "schema_version": "chaoyang-cleanup-v6-deletion-receipt-v1",
        "completed_at": now_iso(),
        "kind": "external_duplicate_project_clones",
        "status": "PASS_PERMANENT_DELETE_WITH_SOURCE_CAPSULE" if not remaining else "FAILED_PARTIAL_DELETE",
        "deleted_root_count": len(roots) - len(remaining),
        "deleted_file_count": pre["file_count"],
        "deleted_bytes": pre["total_bytes"],
        "retained_archive": pre["retained_archive"],
        "remaining": remaining,
    }
    atomic_json(RUN_ROOT / "receipts/DELETION_EXTERNAL_CLONES_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if not remaining else 2


def _source_retirement_partition(protected: set[Path], graph: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    current = {Path(path).resolve() for path in graph["current_code_transitive"]}
    semantic_protected: set[Path] = set()
    semantic_sources = [path for path in CURRENT_FILES if path.is_file()]
    task_state = load(GOV / "LONG_HORIZON_TASK_STATE.json")
    semantic_sources.extend(current_task_packets(task_state))
    for source in semantic_sources:
        semantic_protected.add(source.resolve())
        collect_absolute_paths(load(source), semantic_protected)
    expand_json_reference_closure(semantic_sources, semantic_protected)
    git = git_snapshot()
    tracked_dirty = {
        (ROOT / row["path"]).resolve()
        for row in git["status"] if row["xy"] != "??"
    }
    code_text = "\n".join(
        Path(path).read_text(encoding="utf-8", errors="replace")
        for path in graph["current_code_transitive"] if Path(path).is_file()
    )
    candidates: list[Path] = []
    kept: list[Path] = []
    test_files = sorted((ROOT / "tests").glob("test_*.py"))
    contract_files = sorted(path for path in (ROOT / "contracts").rglob("*") if path.is_file())
    docs_files = sorted(path for path in (ROOT / "docs").rglob("*") if path.is_file())
    canonical_docs = {
        ROOT / "docs/README.md",
        ROOT / "docs/pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md",
        ROOT / "docs/pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.receipt.json",
        ROOT / "docs/pipeline/MASQUERADE_CONTACT_OCCLUSION_LEARNINGS_ZH.md",
        ROOT / "docs/pipeline/C2W_ROBOT_COORDINATE_AUDIT.md",
    }
    for path in (*test_files, *contract_files, *docs_files):
        resolved = path.resolve()
        relative = str(resolved.relative_to(ROOT))
        directly_protected = resolved in current or resolved in tracked_dirty
        directly_protected = directly_protected or any(
            resolved == reference or (reference.is_dir() and _path_is_under(resolved, reference))
            for reference in semantic_protected
        )
        referenced_by_code = relative in code_text or str(resolved) in code_text
        permanent_doc = _path_is_under(resolved, ROOT / "docs/governance")
        permanent_doc = permanent_doc or _path_is_under(resolved, ROOT / "docs/history")
        permanent_doc = permanent_doc or _path_is_under(resolved, ROOT / "docs/reports/depth_accuracy/20260911")
        keep = directly_protected or referenced_by_code or resolved in {path.resolve() for path in canonical_docs} or permanent_doc
        (kept if keep else candidates).append(resolved)
    return sorted(set(kept)), sorted(set(candidates))


def command_prepare_source_retirement(_: argparse.Namespace) -> int:
    snapshot, protected = protection_snapshot()
    graph = build_reference_graph(protected)
    kept, candidates = _source_retirement_partition(protected, graph)
    rows = []
    for path in candidates:
        row = artifact(path)
        row["relative_path"] = str(path.relative_to(ROOT))
        rows.append(row)
    capsule_dir = ROOT / "docs/history/2026-09-14/retired_tests_contracts_docs_v6"
    capsule_dir.mkdir(parents=True, exist_ok=True)
    archive = capsule_dir / "RETIRED_SOURCE_AND_DOCUMENTS.tar.gz"
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    with tarfile.open(temporary, "w:gz", compresslevel=9) as bundle:
        for row in rows:
            source = Path(row["path"])
            info = bundle.gettarinfo(str(source), arcname=row["relative_path"])
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                bundle.addfile(info, stream)
    os.chmod(temporary, 0o444)
    os.replace(temporary, archive)
    value = {
        "schema_version": "chaoyang-source-retirement-pre-delete-v1",
        "created_at": now_iso(),
        "status": "READY_COMMIT_DELETE_AFTER_LIVE_REFERENCE_RECHECK",
        "candidate_files": rows,
        "kept_files": [str(path) for path in kept],
        "counts": {
            "candidate_total": len(rows),
            "tests": sum("/tests/" in row["path"] for row in rows),
            "contracts": sum("/contracts/" in row["path"] for row in rows),
            "docs": sum("/docs/" in row["path"] for row in rows),
        },
        "candidate_bytes": sum(row["bytes"] for row in rows),
        "retained_archive": artifact(archive),
        "claim_limit": "Source/document navigation capsule; current regression tests, current contracts and canonical/current governance documents are excluded.",
    }
    atomic_json(capsule_dir / "LEGACY_EVIDENCE_CAPSULE.json", value)
    atomic_json(RUN_ROOT / "receipts/PRE_DELETE_SOURCE_RETIREMENT_MANIFEST.json", value)
    print(json.dumps({"status": value["status"], **value["counts"], "bytes": value["candidate_bytes"]}, ensure_ascii=False))
    return 0


def command_delete_source_retirement(args: argparse.Namespace) -> int:
    if not args.commit_delete:
        raise SystemExit("delete-source-retirement requires --commit-delete")
    subprocess.run(
        [sys.executable, "-m", "tools.governance.validate_governance_state"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    pre_path = RUN_ROOT / "receipts/PRE_DELETE_SOURCE_RETIREMENT_MANIFEST.json"
    if not pre_path.is_file():
        raise SystemExit("BLOCKED: source retirement manifest missing")
    pre = load(pre_path)
    snapshot, protected = protection_snapshot()
    graph = build_reference_graph(protected)
    _, candidates = _source_retirement_partition(protected, graph)
    expected_paths = [row["path"] for row in pre["candidate_files"]]
    if [str(path) for path in candidates] != expected_paths:
        raise SystemExit("BLOCKED: source retirement candidate set changed")
    if active_target_references(candidates):
        raise SystemExit("BLOCKED: active process references retiring source")
    for row in pre["candidate_files"]:
        if artifact(Path(row["path"])) != {key: row[key] for key in ("path", "bytes", "sha256")}:
            raise SystemExit(f"BLOCKED: retiring source changed: {row['path']}")
    if artifact(Path(pre["retained_archive"]["path"])) != pre["retained_archive"]:
        raise SystemExit("BLOCKED: source retirement capsule mismatch")
    for path in candidates:
        path.unlink()
    receipt = {
        "schema_version": "chaoyang-cleanup-v6-deletion-receipt-v1",
        "completed_at": now_iso(),
        "kind": "retired_tests_contracts_documents",
        "status": "PASS_PERMANENT_DELETE_WITH_SOURCE_CAPSULE",
        "deleted_file_count": len(candidates),
        "deleted_bytes": pre["candidate_bytes"],
        "deleted_counts": pre["counts"],
        "retained_archive": pre["retained_archive"],
    }
    atomic_json(RUN_ROOT / "receipts/DELETION_SOURCE_RETIREMENT_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("snapshot")
    caches = sub.add_parser("delete-caches")
    caches.add_argument("--commit-delete", action="store_true")
    legacy = sub.add_parser("legacy-manifest")
    legacy.add_argument("--full-sha", action="store_true", help="Compatibility flag; full SHA is always computed.")
    prepare_legacy = sub.add_parser("prepare-legacy-delete")
    prepare_legacy.add_argument("--full-sha", action="store_true", help="Compatibility flag; full SHA is always computed.")
    delete_legacy = sub.add_parser("delete-legacy")
    delete_legacy.add_argument("--commit-delete", action="store_true")
    sub.add_parser("tools-manifest")
    tools = sub.add_parser("delete-tools")
    tools.add_argument("--commit-delete", action="store_true")
    sub.add_parser("prepare-runs-delete")
    runs = sub.add_parser("delete-runs")
    runs.add_argument("--commit-delete", action="store_true")
    sub.add_parser("prepare-archive-delete")
    archive_delete = sub.add_parser("delete-archive")
    archive_delete.add_argument("--commit-delete", action="store_true")
    sub.add_parser("prepare-runtime-delete")
    runtime = sub.add_parser("delete-runtime")
    runtime.add_argument("--commit-delete", action="store_true")
    sub.add_parser("prepare-external-clones")
    external_clones = sub.add_parser("delete-external-clones")
    external_clones.add_argument("--commit-delete", action="store_true")
    sub.add_parser("prepare-source-retirement")
    source_retirement = sub.add_parser("delete-source-retirement")
    source_retirement.add_argument("--commit-delete", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    return {
        "snapshot": command_snapshot,
        "delete-caches": command_delete_caches,
        "legacy-manifest": command_legacy_manifest,
        "prepare-legacy-delete": command_prepare_legacy_delete,
        "delete-legacy": command_delete_legacy,
        "tools-manifest": command_tools_manifest,
        "delete-tools": command_delete_tools,
        "prepare-runs-delete": command_prepare_runs_delete,
        "delete-runs": command_delete_runs,
        "prepare-archive-delete": command_prepare_archive_delete,
        "delete-archive": command_delete_archive,
        "prepare-runtime-delete": command_prepare_runtime_delete,
        "delete-runtime": command_delete_runtime,
        "prepare-external-clones": command_prepare_external_clones,
        "delete-external-clones": command_delete_external_clones,
        "prepare-source-retirement": command_prepare_source_retirement,
        "delete-source-retirement": command_delete_source_retirement,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
