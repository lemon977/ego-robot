"""Configuration loading with a small, explicit user-facing surface."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


class ConfigError(RuntimeError):
    pass


SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    input_mp4: Path
    output_dir: Path
    session_id: str
    frame_count: int
    timestamps_csv: Path | None
    host: str
    port: int
    open_browser: bool
    require_complete: bool
    required_nonempty_classes: tuple[str, ...]
    raw: dict[str, Any]

    @property
    def identity_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "frame_count": self.frame_count,
            "timestamps_csv": str(self.timestamps_csv) if self.timestamps_csv else "",
        }

    @property
    def identity_sha256(self) -> str:
        payload = json.dumps(self.identity_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a JSON object")
    return value


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: str | Path, *, require_input: bool = True) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"config not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid config JSON: {config_path}: {exc}") from exc
    raw = _object(raw, "config")
    base = config_path.parent

    for key in ("input_mp4", "output_dir", "session_id"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip() or "CHANGE_ME" in value:
            raise ConfigError(f"edit {key} in {config_path}")
    input_mp4 = _resolve(base, raw["input_mp4"])
    output_dir = _resolve(base, raw["output_dir"])
    session_id = raw["session_id"].strip()
    if not SESSION_RE.fullmatch(session_id):
        raise ConfigError("session_id may contain only letters, digits, dot, underscore and dash")
    if require_input and (not input_mp4.is_file() or input_mp4.suffix.lower() != ".mp4"):
        raise ConfigError(f"input_mp4 must be an existing .mp4 file: {input_mp4}")
    if output_dir == input_mp4 or output_dir in input_mp4.parents:
        raise ConfigError("output_dir cannot be the input file or one of its parents")

    sampling = _object(raw.get("sampling", {}), "sampling")
    frame_count = sampling.get("frame_count", 24)
    if not isinstance(frame_count, int) or not 1 <= frame_count <= 500:
        raise ConfigError("sampling.frame_count must be an integer in [1, 500]")
    csv_value = sampling.get("timestamps_csv", "")
    if not isinstance(csv_value, str):
        raise ConfigError("sampling.timestamps_csv must be a string")
    timestamps_csv = _resolve(base, csv_value) if csv_value.strip() else None
    if timestamps_csv is not None and not timestamps_csv.is_file():
        raise ConfigError(f"timestamps CSV not found: {timestamps_csv}")

    server = _object(raw.get("server", {}), "server")
    host = server.get("host", "127.0.0.1")
    port = server.get("port", 8765)
    open_browser = server.get("open_browser", True)
    if host not in {"127.0.0.1", "localhost"}:
        raise ConfigError("server.host must stay local: 127.0.0.1 or localhost")
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ConfigError("server.port must be an integer in [1024, 65535]")
    if not isinstance(open_browser, bool):
        raise ConfigError("server.open_browser must be boolean")

    validation = _object(raw.get("validation", {}), "validation")
    require_complete = validation.get("require_complete", True)
    required = validation.get("required_nonempty_classes", [])
    if not isinstance(require_complete, bool):
        raise ConfigError("validation.require_complete must be boolean")
    if not isinstance(required, list) or any(not isinstance(v, str) for v in required):
        raise ConfigError("validation.required_nonempty_classes must be a string list")

    return AppConfig(
        config_path=config_path,
        input_mp4=input_mp4,
        output_dir=output_dir,
        session_id=session_id,
        frame_count=frame_count,
        timestamps_csv=timestamps_csv,
        host=host,
        port=port,
        open_browser=open_browser,
        require_complete=require_complete,
        required_nonempty_classes=tuple(required),
        raw=raw,
    )
