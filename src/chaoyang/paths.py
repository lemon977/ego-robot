"""Canonical project paths with explicit, portable overrides."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

DATA_ROOT_DEFAULT = Path("/mnt/data/egodata/datasets/ego")


def _discover_repo_root() -> Path:
    configured = os.environ.get("CHAOYANG_REPO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src/chaoyang").is_dir():
            return parent
    raise RuntimeError("cannot discover Chaoyang repository root")


def _path_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    data_root: Path
    processed_root: Path
    run_root: Path
    model_root: Path

    def ensure_runtime(self) -> None:
        """Create only repository-local runtime directories."""
        for path in (self.run_root, self.run_root / "cache", self.run_root / "locks"):
            path.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, str]:
        return {
            "repo_root": str(self.repo_root),
            "data_root": str(self.data_root),
            "processed_root": str(self.processed_root),
            "run_root": str(self.run_root),
            "model_root": str(self.model_root),
        }


def get_paths() -> ProjectPaths:
    repo = _discover_repo_root()
    data = _path_env("EGO_DATA_ROOT", DATA_ROOT_DEFAULT)
    processed = _path_env("EGO_PROCESSED_ROOT", data / "processed")
    run = _path_env("CHAOYANG_RUN_ROOT", repo / "_run/current")
    models = _path_env("CHAOYANG_MODEL_ROOT", repo / "assets/models")
    for external in (data, processed):
        if external == repo or repo in external.parents:
            raise ValueError(f"dataset path must remain outside the repository: {external}")
    for local in (run, models):
        try:
            local.relative_to(repo)
        except ValueError as exc:
            raise ValueError(f"runtime/model path must stay inside repository: {local}") from exc
    return ProjectPaths(repo, data, processed, run, models)
