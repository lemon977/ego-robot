"""Read-only inventory of ``_run/current`` task directories and Exact78 code.

Run directories are classified by who still references them; code files by
whether a current algorithm-contract route or only historical packets pin them.
References are ``_run/current/<dir>`` paths and, for task-like names, bare task
ids (current pages cite tasks by id and reach their run directories through
receipts). Run paths cited by the governance packet index are kept, because the
governance validator checks them. Classification is advisory input for a later
cleanup packet.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import time
from typing import Any

RUN_REF = re.compile(r"_run/current/([A-Za-z0-9_.\-]+)")
TOKEN = re.compile(r"[A-Za-z0-9_\-]+")
OTHER_BRANCH = re.compile(r"robot100|0916|sensor0916|egosteer", re.I)
TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".py", ".txt", ".yaml", ".yml"}
BARE_ID_MIN_LENGTH = 12


def bare_id_names(names: list[str]) -> set[str]:
    """Names distinctive enough to count as a reference when they appear as a bare token."""
    return {name for name in names if "_" in name and len(name) >= BARE_ID_MIN_LENGTH}


def referenced_names(text: str, known: set[str], bare: set[str], *, bare_ids: bool = True) -> set[str]:
    found = {name for name in RUN_REF.findall(text) if name in known}
    if bare_ids:
        found |= {token.lower() if token.lower() in bare else token
                  for token in TOKEN.findall(text) if token in bare or token.lower() in bare}
    return found


def _texts(root: Path, pattern: str) -> list[Path]:
    return [path for path in sorted(root.glob(pattern))
            if path.is_file() and path.suffix in TEXT_SUFFIXES and "__pycache__" not in path.parts]


AUTHORITY_FILES = ("ALGORITHM_CONTRACT.json", "CURRENT_BASELINE_REGISTRY_V2.json",
                   "CURRENT_AUTHORITY_INDEX.json", "CURRENT_REGRESSION_MANIFEST.json",
                   "CURRENT_FILE_LAYOUT.json", "DOC_AUTHORITY_MAP.json")


def _corpora(repo: Path) -> dict[str, list[Path]]:
    nav = _texts(repo, "docs/current/**/*")
    governance = _texts(repo, "docs/governance/*")
    run = repo / "_run/current"
    owned = [path for pattern in ("*/attempts/*/RESULT.json", "*/attempts/*/candidate/RESULT.json",
                                  "*/attempts/*/candidate/*/RESULT.json")
             for path in _texts(run, pattern)]
    return {
        "CURRENT_NAV": [path for path in nav if not OTHER_BRANCH.search(path.name)],
        "OTHER_BRANCH_NAV": [path for path in nav if OTHER_BRANCH.search(path.name)],
        "AUTHORITY": [path for path in governance if path.name in AUTHORITY_FILES],
        "LEDGER": [path for path in governance if path.name not in AUTHORITY_FILES],
        "PLAN_DOC": _texts(repo, "docs/plans/**/*"),
        "PACKET": _texts(repo, "tasks/current/*/TASK_PACKET.json"),
        "GOVERNANCE_INDEX": _texts(repo, "tasks/current/INDEX.json"),
        "RECEIPT": _texts(repo, "tasks/receipts/*.json"),
        "RESULT_DEP": owned,
        "REVIEW": _texts(repo, "review/*"),
        "CODE": _texts(repo, "src/chaoyang/**/*.py"),
    }


ROOT_CATEGORIES = (
    ("KEEP_PARENT", {"CURRENT_NAV", "AUTHORITY"}),
    ("KEEP_GOVERNANCE_REF", {"GOVERNANCE_INDEX"}),
    ("KEEP_OTHER_BRANCH", {"OTHER_BRANCH_NAV"}),
    ("KEEP_INPUT", {"CODE"}),
    ("DEV_EVIDENCE", {"REVIEW", "PLAN_DOC"}),
)
EDGE_KINDS = {"PACKET", "RECEIPT", "RESULT_DEP"}
# The packet index lists every task id; only its artifact paths are references.
PATH_ONLY_KINDS = {"GOVERNANCE_INDEX"}
SIZE_BUDGET_SECONDS = 1800


def _owner(repo: Path, path: Path) -> str | None:
    """Task id that owns this corpus file, so self references are ignored."""
    parts = path.relative_to(repo).parts
    if parts[:2] == ("_run", "current") and len(parts) > 2:
        return parts[2]
    if parts[:2] == ("tasks", "current") and len(parts) > 2:
        return parts[2]
    if parts[:2] == ("tasks", "receipts"):
        return path.stem.lower()
    return None


def _review_links(repo: Path) -> set[str]:
    names: set[str] = set()
    review = repo / "review"
    if not review.is_dir():
        return names
    for entry in review.iterdir():
        if entry.is_symlink():
            match = RUN_REF.search(os.readlink(entry))
            if match:
                names.add(match.group(1))
    return names


def dir_size(directory: Path) -> dict[str, int]:
    total = files = 0
    for root, dirs, names in os.walk(directory, followlinks=False):
        for name in names:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
                files += 1
            except OSError:
                continue
    return {"bytes": total, "files": files}


def live_writers(directory: Path) -> list[int]:
    """PIDs from attempt WRITER.json files whose process (same start ticks) is still running."""
    alive = []
    for writer in directory.glob("attempts/*/WRITER.json"):
        try:
            value = json.loads(writer.read_text(encoding="utf-8"))
            pid = int(value["pid"])
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
        if value.get("proc_start_ticks") in (None, int(fields[19])):
            alive.append(pid)
    return alive


def _latest_status(directory: Path) -> tuple[str, str | None]:
    results = sorted(directory.glob("attempts/*/RESULT.json"))
    if not results:
        return "NO_RESULT", None
    try:
        value = json.loads(results[-1].read_text(encoding="utf-8"))
        return str(value.get("status", "UNKNOWN")), results[-1].parent.name
    except (OSError, ValueError):
        return "UNREADABLE_RESULT", results[-1].parent.name


def classify_closure(names: list[str], direct: dict[str, set[str]],
                     edges: dict[str, set[str]]) -> dict[str, str]:
    """Reference-graph closure: a directory inherits the strongest category of any referrer."""
    category: dict[str, str] = {}
    for name in names:
        if name.startswith("exact78_takeover_"):
            category[name] = "TAKEOVER"
        elif OTHER_BRANCH.search(name):
            category[name] = "KEEP_OTHER_BRANCH"
    for label, kinds in ROOT_CATEGORIES:
        frontier = [name for name in names if name not in category and direct.get(name, set()) & kinds]
        if label == "KEEP_INPUT":
            frontier += [name for name in names if category.get(name) == "TAKEOVER"]
        while frontier:
            name = frontier.pop()
            if name not in category:
                category[name] = label
            for target in edges.get(name, ()):
                if target not in category:
                    frontier.append(target)
    for name in names:
        category.setdefault(name, "LEDGER_ONLY_CLEANUP_CANDIDATE")
    return category


def _code_inventory(repo: Path, packets: list[Path]) -> dict[str, Any]:
    contract = json.loads((repo / "docs/governance/ALGORITHM_CONTRACT.json").read_text(encoding="utf-8"))
    routed: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            path = value.get("path")
            if isinstance(path, str) and path.endswith(".py"):
                routed.add(str(Path(path).resolve()))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(contract)
    pinned: dict[str, int] = Counter()
    for packet in packets:
        try:
            value = json.loads(packet.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for ref in value.get("runtime_code", []) or []:
            if isinstance(ref, dict) and isinstance(ref.get("path"), str):
                pinned[str(Path(ref["path"]).resolve())] += 1
    rows = {}
    for directory in ("pipeline", "governance", "ops", "cli"):
        for path in sorted((repo / "src/chaoyang" / directory).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            key = str(path.resolve())
            status = ("ROUTED" if key in routed else
                      "HISTORICAL_PINNED_ONLY" if pinned.get(key) else "UNREFERENCED_BY_ROUTE_OR_PACKET")
            rows[str(path.relative_to(repo))] = {"status": status, "pinned_by_packets": pinned.get(key, 0)}
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    for name, row in rows.items():
        summary[name.split("/")[2]][row["status"]] += 1
    return {"files": rows, "summary": {key: dict(value) for key, value in summary.items()}}


def run(repo: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    run_root = repo / "_run/current"
    names = sorted(entry.name for entry in run_root.iterdir() if entry.is_dir())
    known = set(names)
    references: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    edges: dict[str, set[str]] = defaultdict(set)
    corpora = _corpora(repo)
    receipt_owner = {name.upper(): name for name in names}
    bare = bare_id_names(names)
    for kind, files in corpora.items():
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            owner = _owner(repo, path)
            if kind == "RECEIPT":
                owner = receipt_owner.get(path.stem.upper(), owner)
            for target in referenced_names(text, known, bare, bare_ids=kind not in PATH_ONLY_KINDS):
                if target != owner:
                    references[target][kind].add(str(path.relative_to(repo)))
                    if kind in EDGE_KINDS and owner in known:
                        edges[owner].add(target)
    for target in _review_links(repo):
        if target in known:
            references[target]["REVIEW"].add("review/<symlink>")
    direct = {name: set(refs) for name, refs in references.items()}
    categories = classify_closure(names, direct, edges)
    rows = []
    sizes_complete = True
    for name in names:
        refs = references.get(name, {})
        status, attempt = _latest_status(run_root / name)
        category = categories[name]
        size = None
        if time.monotonic() - started < SIZE_BUDGET_SECONDS:
            size = dir_size(run_root / name)
        else:
            sizes_complete = False
        rows.append({"task_dir": name, "category": category, "latest_status": status,
                     "latest_attempt": attempt, "size": size,
                     "is_task_dir": any((run_root / name).glob("attempts/*/RESULT.json")),
                     "live_writer_pids": live_writers(run_root / name),
                     "referenced_by": {kind: sorted(paths)[:5] for kind, paths in sorted(refs.items())},
                     "reference_counts": {kind: len(paths) for kind, paths in sorted(refs.items())}})
    code = _code_inventory(repo, corpora["PACKET"])
    bytes_by_category: dict[str, int] = Counter()
    for row in rows:
        bytes_by_category[row["category"]] += (row["size"] or {}).get("bytes", 0)
    report = {"schema_version": "exact78-takeover-inventory-v2", "run_dirs": rows,
              "category_counts": dict(Counter(row["category"] for row in rows)),
              "bytes_by_category": dict(bytes_by_category),
              "status_by_category": {category: dict(Counter(row["latest_status"] for row in rows
                                                            if row["category"] == category))
                                     for category in sorted({row["category"] for row in rows})},
              "code": code, "sizes_measured": sizes_complete,
              "elapsed_seconds": round(time.monotonic() - started, 2)}
    (output / "INVENTORY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"run_dirs": len(rows), "category_counts": report["category_counts"],
            "bytes_by_category": report["bytes_by_category"], "sizes_measured": sizes_complete,
            "code_summary": code["summary"], "elapsed_seconds": report["elapsed_seconds"]}
