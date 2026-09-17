#!/usr/bin/env python3
"""Validate the clean-baseline directory, ignore and portability contracts."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

ALLOWED_ROOTS = {
    ".git", ".gitignore", ".gitattributes", "AGENTS.md", "README.md",
    "THIRD_PARTY_NOTICES.md", "pyproject.toml", "src", "configs", "contracts",
    "manifests", "scripts", "tests", "docs", "tasks", "assets", "vendor",
    "archive", "_run",
}
REQUIRED_ROOTS = {
    "src", "configs", "contracts", "manifests", "scripts", "tests", "docs",
    "tasks", "assets", "vendor", "archive", "_run",
}
FORBIDDEN_ROOTS = {"pipeline", "HumanEgo", "tools", "systems", "data", "third_party"}
IGNORED_ALLOWLIST = ("archive/", "_run/current/", "assets/models/")
CODE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".sh", ".cmd", ".ps1"}
LEGACY_IMPORT = re.compile(r"^\s*(?:from|import)\s+(?:pipeline|HumanEgo|tools)(?:[.\s]|$)", re.MULTILINE)
HARDCODED_ROOT = "/mnt/workspace/code/" + "chaoyang"
DEPRECATED_SOURCE_PATH = "NOW/daemon/" + "tools"


def git_lines(root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    )
    return [line for line in result.stdout.splitlines() if line]


def ignored_paths(root: Path) -> list[str]:
    result = subprocess.run(
        [
            "git", "status", "--porcelain=v1", "-z", "--ignored=matching",
            "--untracked-files=normal",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    paths = []
    for record in result.stdout.split(b"\0"):
        if record.startswith(b"!! "):
            paths.append(record[3:].decode("utf-8", errors="surrogateescape"))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    actual = {path.name for path in root.iterdir()}
    unknown = sorted(actual - ALLOWED_ROOTS)
    if unknown:
        errors.append("unexpected repository roots: " + ", ".join(unknown))
    missing = sorted(REQUIRED_ROOTS - actual)
    if missing:
        errors.append("required repository roots missing: " + ", ".join(missing))
    present_forbidden = sorted(actual & FORBIDDEN_ROOTS)
    if present_forbidden:
        errors.append("legacy repository roots remain: " + ", ".join(present_forbidden))

    ignored = ignored_paths(root)
    bad_ignored = [path for path in ignored if not path.startswith(IGNORED_ALLOWLIST)]
    if bad_ignored:
        errors.append(
            f"ignored files outside allowlist: {len(bad_ignored)} "
            f"(first: {bad_ignored[:10]})"
        )

    for relative in git_lines(root, "ls-files"):
        if not relative.startswith(("src/", "scripts/", "configs/")):
            continue
        path = root / relative
        if not path.is_file() or path.suffix.lower() not in CODE_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if HARDCODED_ROOT in text:
            errors.append(f"hard-coded repository root: {relative}")
        if DEPRECATED_SOURCE_PATH in text:
            errors.append(f"deprecated source path: {relative}")
        if path.suffix.lower() == ".py" and LEGACY_IMPORT.search(text):
            errors.append(f"legacy top-level import: {relative}")

    current_index = root / "tasks/current/INDEX.json"
    if not current_index.is_file():
        errors.append("tasks/current/INDEX.json is missing")
    if not (root / "docs/current/README_ZH.md").is_file():
        errors.append("docs/current/README_ZH.md is missing")

    if not args.allow_dirty:
        dirty = git_lines(root, "status", "--porcelain=v1", "--untracked-files=all")
        if dirty:
            errors.append(f"git worktree is dirty: {len(dirty)} entries")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("PASS: clean-baseline structure is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
