#!/usr/bin/env python3
"""Rewrite repository references after the clean-baseline directory migration."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

TEXT_SUFFIXES = {
    ".py", ".md", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".txt",
    ".sh", ".cmd", ".ps1", ".ini", ".cfg",
}
SKIP_ROOTS = {"archive", "vendor", "_run"}
BASELINE = "archive/baseline-20260917-0aa69e9"
PATH_REPLACEMENTS = (
    ("docs/current/visuals/", "docs/current/visuals/"),
    ("docs/guides/acquisition/", "docs/guides/acquisition/"),
    ("docs/guides/newtask/", "docs/guides/newtask/"),
    ("docs/guides/reproducibility/", "docs/guides/reproducibility/"),
    ("docs/reference/architecture/", "docs/reference/architecture/"),
    ("docs/reference/contracts/", "docs/reference/contracts/"),
    ("docs/reference/data/", "docs/reference/data/"),
    ("docs/reference/pipeline/", "docs/reference/pipeline/"),
    ("docs/reference/robot/", "docs/reference/robot/"),
    ("docs/research/current/reports/", "docs/research/current/reports/"),
    ("archive/baseline-20260917-0aa69e9/content/history/docs/history/", f"{BASELINE}/content/history/archive/baseline-20260917-0aa69e9/content/history/docs/history/"),
    ("archive/baseline-20260917-0aa69e9/content/history/docs/organization/", f"{BASELINE}/content/history/archive/baseline-20260917-0aa69e9/content/history/docs/organization/"),
    ("archive/baseline-20260917-0aa69e9/content/history/docs/collaboration/", f"{BASELINE}/content/history/archive/baseline-20260917-0aa69e9/content/history/docs/collaboration/"),
    ("archive/baseline-20260917-0aa69e9/content/history/tasks/control/runs/", f"{BASELINE}/content/history/tasks/control/runs/"),
    ("archive/baseline-20260917-0aa69e9/content/history/tasks/", f"{BASELINE}/content/history/tasks/"),
    ("src/chaoyang/governance/", "src/chaoyang/governance/"),
    ("src/chaoyang/apps/offline_mask_annotator/", "src/chaoyang/apps/offline_mask_annotator/"),
    ("vendor/glvnd/", "vendor/glvnd/"),
    ("src/chaoyang/ops/", "src/chaoyang/ops/"),
    ("src/chaoyang/pipeline/", "src/chaoyang/pipeline/"),
    ("configs/systems/", "configs/systems/"),
    ("vendor/", "vendor/"),
    ("configs/human_ego/cfg/", "configs/human_ego/cfg/"),
    ("configs/human_ego/config/", "configs/human_ego/config/"),
    ("manifests/human_ego/", "manifests/human_ego/"),
    ("assets/models/humanego/frozen_checkpoints/", "assets/models/humanego/frozen_checkpoints/"),
    ("assets/models/humanego/pretrained_humanego/", "assets/models/humanego/pretrained_humanego/"),
    ("assets/models/humanego/retarget_ab_checkpoints/", "assets/models/humanego/retarget_ab_checkpoints/"),
    ("archive/baseline-20260917-0aa69e9/content/history/HumanEgo/artifacts/newtask_robot_bundles/", f"{BASELINE}/content/history/archive/baseline-20260917-0aa69e9/content/history/HumanEgo/artifacts/newtask_robot_bundles/"),
    ("archive/baseline-20260917-0aa69e9/content/history/HumanEgo/artifacts/newtask_robot_runs/", f"{BASELINE}/content/history/archive/baseline-20260917-0aa69e9/content/history/HumanEgo/artifacts/newtask_robot_runs/"),
    ("archive/baseline-20260917-0aa69e9/content/history/HumanEgo/outputs/", f"{BASELINE}/content/history/archive/baseline-20260917-0aa69e9/content/history/HumanEgo/outputs/"),
    ("archive/baseline-20260917-0aa69e9/content/history/HumanEgo/_run/", f"{BASELINE}/content/history/archive/baseline-20260917-0aa69e9/content/history/HumanEgo/_run/"),
    ("src/chaoyang/human_ego/", "src/chaoyang/human_ego/"),
    ("archive/baseline-20260917-0aa69e9/content/history/data/experiments/", f"{BASELINE}/content/history/archive/baseline-20260917-0aa69e9/content/history/data/experiments/"),
    ("archive/baseline-20260917-0aa69e9/content/history/data/processed/", f"{BASELINE}/content/history/archive/baseline-20260917-0aa69e9/content/history/data/processed/"),
    ("archive/baseline-20260917-0aa69e9/content/history/data/benchmarks/hand/production_runs/", f"{BASELINE}/content/history/archive/baseline-20260917-0aa69e9/content/history/data/benchmarks/hand/production_runs/"),
    ("manifests/legacy_data/", "manifests/legacy_data/"),
)
IMPORT_REPLACEMENTS = (
    ("from chaoyang.governance", "from chaoyang.governance"),
    ("import chaoyang.governance", "import chaoyang.governance"),
    ("from chaoyang.ops", "from chaoyang.ops"),
    ("import chaoyang.ops.", "import chaoyang.ops."),
    ("from chaoyang.pipeline", "from chaoyang.pipeline"),
    ("import chaoyang.pipeline.", "import chaoyang.pipeline."),
    ("from chaoyang.human_ego", "from chaoyang.human_ego"),
    ("import chaoyang.human_ego.", "import chaoyang.human_ego."),
)


def active_files(root: Path):
    for current, directories, files in os.walk(root, topdown=True):
        relative = Path(current).relative_to(root)
        if relative == Path("."):
            directories[:] = [name for name in directories if name not in SKIP_ROOTS and name != ".git"]
        directories[:] = [name for name in directories if name not in {"__pycache__", "archive/baseline-20260917-0aa69e9/content/regenerable/cache/.pytest_cache", "archive/baseline-20260917-0aa69e9/content/regenerable/cache/.ruff_cache", ".mypy_cache"}]
        for name in files:
            path = Path(current) / name
            if path.suffix.lower() in TEXT_SUFFIXES:
                yield path


def rewrite_text(relative: Path, text: str) -> str:
    updated = text.replace("", "")
    tokens: list[tuple[str, str]] = []
    for index, (old, new) in enumerate(PATH_REPLACEMENTS):
        token = f"__CHAOYANG_PATH_REDIRECT_{index}__"
        updated = re.sub(r"(?<![A-Za-z0-9_/])" + re.escape(old), token, updated)
        tokens.append((token, new))
    for token, new in tokens:
        updated = updated.replace(token, new)
    if relative.suffix == ".py":
        for old, new in IMPORT_REPLACEMENTS:
            updated = updated.replace(old, new)
        value = relative.as_posix()
        if value.startswith("src/chaoyang/ops/") and len(relative.parts) == 4:
            updated = updated.replace("Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parents[3]")
            updated = updated.replace("Path(__file__).absolute().parents[1]", "Path(__file__).absolute().parents[3]")
        elif value.startswith("src/chaoyang/governance/"):
            updated = updated.replace("Path(__file__).resolve().parents[2]", "Path(__file__).resolve().parents[3]")
        elif value.startswith("src/chaoyang/pipeline/") and len(relative.parts) == 4:
            updated = updated.replace("Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parents[3]")
            updated = updated.replace("Path(__file__).absolute().parents[1]", "Path(__file__).absolute().parents[3]")
        elif value.startswith("src/chaoyang/human_ego/"):
            updated = updated.replace("Path(__file__).resolve().parents[2]", "Path(__file__).resolve().parents[4]")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    changed: list[str] = []
    for path in active_files(root):
        relative = path.relative_to(root)
        try:
            original = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        updated = rewrite_text(relative, original)
        if updated == original:
            continue
        mode = path.stat().st_mode
        temporary = path.with_name(f".{path.name}.rewrite.tmp")
        temporary.write_text(updated, encoding="utf-8")
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        changed.append(relative.as_posix())
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps({"schema_version": 1, "changed_count": len(changed), "changed": changed}, indent=2) + "\n")
    print(json.dumps({"changed_count": len(changed)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
