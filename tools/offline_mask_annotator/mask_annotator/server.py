"""Local-only HTTP server for the shipped browser annotation UI."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import threading
import webbrowser
from urllib.parse import urlparse

from .config import AppConfig
from .project import (
    ProjectError,
    export_project,
    frame_by_key,
    load_annotation,
    load_project,
    save_annotation,
)
from .validation import validate_project


STATIC_ROOT = Path(__file__).resolve().parent.parent / "web"
FRAME_RE = re.compile(r"^/api/frame/(frame_[0-9]{4})/(image|annotation)$")
MAX_JSON_BYTES = 64 * 1024 * 1024


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def make_handler(config: AppConfig):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OfflineMaskAnnotator/1.0"

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status: int, value) -> None:
            self._send(status, _json_bytes(value), "application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._send_json(status, {"ok": False, "error": message})

        def _manifest(self):
            return load_project(config)

        def do_GET(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/api/project":
                    manifest = self._manifest()
                    frames = []
                    for frame in manifest["frames"]:
                        annotation = load_annotation(config, frame)
                        frames.append({
                            **frame,
                            "complete": bool(annotation["complete"]),
                            "revision": annotation["revision"],
                            "image_url": f"/api/frame/{frame['frame_key']}/image",
                            "annotation_url": f"/api/frame/{frame['frame_key']}/annotation",
                        })
                    self._send_json(HTTPStatus.OK, {
                        "ok": True,
                        "session_id": manifest["identity"]["session_id"],
                        "sampling": manifest["sampling"],
                        "classes": manifest["classes"],
                        "frames": frames,
                    })
                    return
                match = FRAME_RE.fullmatch(parsed.path)
                if match:
                    frame = frame_by_key(self._manifest(), match.group(1))
                    if match.group(2) == "annotation":
                        self._send_json(HTTPStatus.OK, load_annotation(config, frame))
                    else:
                        path = config.output_dir / frame["image_relpath"]
                        self._send(HTTPStatus.OK, path.read_bytes(), "image/png")
                    return
                static_name = "index.html" if parsed.path in {"", "/"} else parsed.path.lstrip("/")
                if static_name not in {"index.html", "app.js", "styles.css"}:
                    self._error(HTTPStatus.NOT_FOUND, "not found")
                    return
                path = STATIC_ROOT / static_name
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                if content_type.startswith("text/") or content_type == "application/javascript":
                    content_type += "; charset=utf-8"
                self._send(HTTPStatus.OK, path.read_bytes(), content_type)
            except Exception as exc:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

        def _read_json(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ProjectError("invalid Content-Length") from exc
            if not 0 < length <= MAX_JSON_BYTES:
                raise ProjectError("JSON request body is empty or too large")
            try:
                value = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProjectError(f"invalid JSON request: {exc}") from exc
            if not isinstance(value, dict):
                raise ProjectError("JSON request root must be an object")
            return value

        def do_PUT(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                match = FRAME_RE.fullmatch(parsed.path)
                if not match or match.group(2) != "annotation":
                    self._error(HTTPStatus.NOT_FOUND, "not found")
                    return
                frame = frame_by_key(self._manifest(), match.group(1))
                saved = save_annotation(config, frame, self._read_json())
                self._send_json(HTTPStatus.OK, {"ok": True, "annotation": saved})
            except ProjectError as exc:
                self._error(HTTPStatus.CONFLICT, str(exc))
            except Exception as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))

        def do_POST(self) -> None:  # noqa: N802
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/api/export":
                    result = export_project(config)
                    self._send_json(HTTPStatus.OK, {"ok": True, "result": result})
                    return
                if parsed.path == "/api/validate":
                    result = validate_project(config)
                    self._send_json(HTTPStatus.OK, {"ok": True, "result": result})
                    return
                self._error(HTTPStatus.NOT_FOUND, "not found")
            except Exception as exc:
                self._error(HTTPStatus.BAD_REQUEST, str(exc))

        def log_message(self, fmt: str, *args) -> None:
            print(f"[annotator] {self.address_string()} {fmt % args}")

    return Handler


def serve(config: AppConfig) -> None:
    load_project(config)
    server = ThreadingHTTPServer((config.host, config.port), make_handler(config))
    url = f"http://{config.host}:{config.port}/"
    print(f"Offline Mask Annotator: {url}")
    print("Press Ctrl+C to stop. All traffic stays on this computer.")
    if config.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
