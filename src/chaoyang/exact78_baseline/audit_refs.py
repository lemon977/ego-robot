"""Read-only evidence reference audit.

Every ``{"path", "sha256"}`` reference found in the scanned JSON roots is checked
against the file on disk; markdown links in current navigation are checked for
existence.  Nothing outside the output directory is written.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Iterator

HASH_SIZE_LIMIT = 2 * 1024 ** 3
MD_LINK = re.compile(r"\]\(([^)\s]+)\)")


class Hasher:
    def __init__(self) -> None:
        self.cache: dict[Path, tuple[int, str | None]] = {}
        self.bytes_hashed = 0

    def get(self, path: Path) -> tuple[int, str | None]:
        if path not in self.cache:
            size = path.stat().st_size
            if size > HASH_SIZE_LIMIT:
                self.cache[path] = (size, None)
            else:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        digest.update(chunk)
                self.bytes_hashed += size
                self.cache[path] = (size, digest.hexdigest())
        return self.cache[path]


def iter_refs(value: Any, trail: tuple[str, ...] = ()) -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        path, sha = value.get("path"), value.get("sha256")
        if isinstance(path, str) and isinstance(sha, str) and len(sha) == 64:
            yield "/".join(trail), value
        for key, child in value.items():
            yield from iter_refs(child, (*trail, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_refs(child, (*trail, str(index)))


def _roots(repo: Path) -> dict[str, list[Path]]:
    run = repo / "_run/current"
    return {
        "current_navigation": sorted((repo / "docs/current").glob("*.json")),
        "exact78_receipts": sorted((repo / "tasks/receipts").glob("EXACT78*.json")),
        "run_results": sorted(run.glob("*/attempts/*/RESULT.json")),
        "task_packets": sorted((repo / "tasks/current").glob("*/TASK_PACKET.json")),
    }


def _classify(repo: Path, hasher: Hasher, trail: str, ref: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raw = Path(ref["path"])
    path = raw if raw.is_absolute() else repo / raw
    is_code = (trail.startswith("runtime_code") or "code" in trail.split("/")[0]
               or path.suffix in {".py", ".sh"})
    detail = {"path": str(path), "trail": trail, "expected_sha256": ref["sha256"],
              "expected_bytes": ref.get("bytes")}
    if not path.is_file():
        return ("CODE_MISSING" if is_code else "MISSING"), detail
    size, sha = hasher.get(path)
    detail["actual_bytes"] = size
    if sha is None:
        ok = ref.get("bytes") in (None, size)
        return ("SIZE_ONLY_OK" if ok else "SIZE_MISMATCH"), detail
    detail["actual_sha256"] = sha
    if sha == ref["sha256"]:
        return "OK", detail
    return ("CODE_SHA_DRIFT" if is_code else "SHA_MISMATCH"), detail


def _markdown_targets(repo: Path) -> list[Path]:
    files = sorted((repo / "docs/current").glob("*.md"))
    files += [repo / "review/README.md", repo / "docs/plans/EXACT78_GENERAL_RAW_TO_ROBOT_V2_20260929/01_P1_STATUS_20260930.md"]
    return [path for path in files if path.is_file()]


def audit_markdown(repo: Path) -> dict[str, Any]:
    broken: list[dict[str, str]] = []
    checked = 0
    for doc in _markdown_targets(repo):
        for target in MD_LINK.findall(doc.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            clean = target.split("#", 1)[0]
            if not clean:
                continue
            path = Path(clean) if clean.startswith("/") else (doc.parent / clean)
            checked += 1
            if not path.exists():
                broken.append({"document": str(doc.relative_to(repo)), "target": target})
    return {"links_checked": checked, "broken": broken}


def run(repo: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    hasher = Hasher()
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    problems: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unreadable: list[str] = []
    files_scanned: Counter[str] = Counter()
    for kind, files in _roots(repo).items():
        for source in files:
            try:
                value = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                unreadable.append(str(source))
                continue
            files_scanned[kind] += 1
            for trail, ref in iter_refs(value):
                status, detail = _classify(repo, hasher, trail, ref)
                summary[kind][status] += 1
                if status not in {"OK", "SIZE_ONLY_OK"}:
                    problems[kind].append({"source": str(source.relative_to(repo)),
                                           "status": status, **detail})
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for kind in ("current_navigation", "exact78_receipts", "run_results"):
        for row in problems[kind]:
            by_source[row["source"]][row["status"]] += 1
    markdown = audit_markdown(repo)
    report = {
        "schema_version": "exact78-takeover-reference-audit-v1",
        "scanned_files": dict(files_scanned),
        "status_counts": {kind: dict(counter) for kind, counter in summary.items()},
        "evidence_problem_sources": {key: dict(value) for key, value in sorted(by_source.items())},
        "problems": problems,
        "unreadable_json": unreadable,
        "markdown": markdown,
        "hash_size_limit_bytes": HASH_SIZE_LIMIT,
        "bytes_hashed": hasher.bytes_hashed,
        "unique_files_checked": len(hasher.cache),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    (output / "REFERENCE_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence_bad = {kind: {k: v for k, v in counter.items() if k in {"MISSING", "SHA_MISMATCH", "SIZE_MISMATCH"}}
                    for kind, counter in summary.items()}
    return {"scanned_files": dict(files_scanned), "evidence_problems": evidence_bad,
            "code_drift": {kind: counter.get("CODE_SHA_DRIFT", 0) + counter.get("CODE_MISSING", 0)
                           for kind, counter in summary.items()},
            "broken_markdown_links": len(markdown["broken"]),
            "bytes_hashed": hasher.bytes_hashed, "elapsed_seconds": report["elapsed_seconds"]}
