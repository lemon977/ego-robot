"""Explicit compatibility adapter for the pinned SAM 3.1 multiplex predictor.

The pinned official wrapper passes ``offload_state_to_cpu`` to a multiplex
``init_state`` method that does not accept it.  This adapter is deliberately
small and fail-closed: it validates the exact pinned signatures, accepts that
option only when false, and maps the remaining options explicitly.  It never
edits or monkeypatches upstream classes and has no fallback implementation.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping


ADAPTER_ID = "sam31_compat_adapter_v1"
OFFICIAL_CODE_COMMIT = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
OFFICIAL_MODEL_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
MIRROR_REVISION = "99cb53e9cb67e5660e6d19de43cec2ea14279401"
CHECKPOINT_BYTES = 3_502_755_717
CHECKPOINT_SHA256 = (
    "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
)
USE_ROPE_REAL = False
STRICT_STATE_DICT_LOAD = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CODE_RELATIVE = Path("third_party/SAM3")
CANONICAL_RUNTIME_CONTRACT_RELATIVE = Path(
    "systems/mask/configs/sam31_runtime_source_v1.json"
)
CANONICAL_RUNTIME_CONTRACT_SHA256 = (
    "ce100807728eda716c4d1535fee3264c24078238751622518af8442eb48927e4"
)
CANONICAL_SWITCH_RECEIPT_RELATIVE = Path(
    "docs/organization/2026-09-03/SAM31_STABLE_SWITCH_EXECUTION.json"
)
CANONICAL_SWITCH_RECEIPT_SHA256 = (
    "16ffa9a30cc79d02005daa7f3fcd4677a50578093085f7c6a3cc68272050f621"
)
OFFICIAL_GIT_TREE = "6bc2384dbbfb370fe28096d23955df9b7b0bcdd9"
OFFICIAL_TRACKED_REGULAR_FILES = 525
OFFICIAL_TRACKED_REGULAR_BYTES = 73_169_481

EXPECTED_PREDICTOR_FQN = (
    "sam3.model.sam3_multiplex_video_predictor.Sam3MultiplexVideoPredictor"
)
EXPECTED_MODEL_FQN = (
    "sam3.model.sam3_multiplex_tracking."
    "Sam3MultiplexTrackingWithInteractivity"
)


class Sam31CompatContractError(RuntimeError):
    """Raised when a pinned identity, API signature, or request drifts."""


def official_source_bundle_sha256(payloads: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative, payload in sorted(payloads.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


class _HeldOfficialLoader(importlib.abc.Loader):
    def __init__(
        self,
        fullname: str,
        relative: str,
        payload: bytes,
        snapshot_root: Path,
        bundle_sha256: str,
        is_package: bool,
    ) -> None:
        self.fullname = fullname
        self.relative = relative
        self.payload = payload
        self.snapshot_root = snapshot_root
        self.bundle_sha256 = bundle_sha256
        self.is_package = is_package

    def create_module(self, spec: Any) -> None:
        del spec
        return None

    def exec_module(self, module: Any) -> None:
        logical_path = self.snapshot_root / self.relative
        module.__file__ = str(logical_path)
        if self.is_package:
            module.__path__ = [str(logical_path.parent)]
        exec(compile(self.payload, str(logical_path), "exec"), module.__dict__)
        module.__held_source_sha256__ = hashlib.sha256(self.payload).hexdigest()
        module.__held_source_bytes__ = len(self.payload)
        module.__held_official_bundle_sha256__ = self.bundle_sha256


class HeldOfficialSourceFinder(importlib.abc.MetaPathFinder):
    """Import ``sam3`` modules only from an already-captured byte mapping."""

    def __init__(
        self,
        payloads: Mapping[str, bytes],
        snapshot_root: Path,
        bundle_sha256: str,
    ) -> None:
        self.payloads = dict(payloads)
        self.snapshot_root = snapshot_root
        self.bundle_sha256 = bundle_sha256

    def find_spec(
        self, fullname: str, path: Any = None, target: Any = None
    ) -> Any:
        del path, target
        if fullname != "sam3" and not fullname.startswith("sam3."):
            return None
        stem = fullname.replace(".", "/")
        package_relative = f"{stem}/__init__.py"
        module_relative = f"{stem}.py"
        if package_relative in self.payloads:
            relative = package_relative
            is_package = True
        elif module_relative in self.payloads:
            relative = module_relative
            is_package = False
        else:
            return None
        loader = _HeldOfficialLoader(
            fullname,
            relative,
            self.payloads[relative],
            self.snapshot_root,
            self.bundle_sha256,
            is_package,
        )
        return importlib.machinery.ModuleSpec(
            fullname,
            loader,
            origin=str(self.snapshot_root / relative),
            is_package=is_package,
        )


@dataclass(frozen=True)
class Sam31PinnedIdentity:
    official_code_commit: str = OFFICIAL_CODE_COMMIT
    official_model_revision: str = OFFICIAL_MODEL_REVISION
    mirror_revision: str = MIRROR_REVISION
    checkpoint_bytes: int = CHECKPOINT_BYTES
    checkpoint_sha256: str = CHECKPOINT_SHA256
    use_rope_real: bool = USE_ROPE_REAL
    strict_state_dict_load: bool = STRICT_STATE_DICT_LOAD


def _parameter_shape(method: Any) -> tuple[tuple[str, str, Any], ...]:
    shape = []
    for name, parameter in inspect.signature(method).parameters.items():
        default = (
            "__REQUIRED__"
            if parameter.default is inspect.Parameter.empty
            else parameter.default
        )
        shape.append((name, parameter.kind.name, default))
    return tuple(shape)


_PK = "POSITIONAL_OR_KEYWORD"
EXPECTED_INIT_STATE_SIGNATURE = (
    ("resource_path", _PK, "__REQUIRED__"),
    ("offload_video_to_cpu", _PK, False),
    ("async_loading_frames", _PK, False),
    ("use_torchcodec", _PK, False),
    ("use_cv2", _PK, False),
    ("input_is_mp4", _PK, False),
)
EXPECTED_ADD_PROMPT_SIGNATURE = (
    ("inference_state", _PK, "__REQUIRED__"),
    ("frame_idx", _PK, "__REQUIRED__"),
    ("text_str", _PK, None),
    ("clear_old_points", _PK, True),
    ("points", _PK, None),
    ("point_labels", _PK, None),
    ("boxes_xywh", _PK, None),
    ("box_labels", _PK, None),
    ("clear_old_boxes", _PK, True),
    ("output_prob_thresh", _PK, 0.5),
    ("obj_id", _PK, None),
    ("rel_coordinates", _PK, True),
)
EXPECTED_PROPAGATE_SIGNATURE = (
    ("inference_state", _PK, "__REQUIRED__"),
    ("start_frame_idx", _PK, None),
    ("max_frame_num_to_track", _PK, None),
    ("reverse", _PK, False),
    ("output_prob_thresh", _PK, 0.5),
    ("compute_stability_score", _PK, False),
    ("is_instance_processing", _PK, False),
    ("is_last_batch", _PK, False),
)


def _fqn(instance: Any) -> str:
    cls = type(instance)
    return f"{cls.__module__}.{cls.__name__}"


def validate_pinned_signatures(predictor: Any) -> dict[str, Any]:
    """Require the exact public method shapes of the pinned official commit."""
    if _fqn(predictor) != EXPECTED_PREDICTOR_FQN:
        raise Sam31CompatContractError(
            f"predictor identity drift: {_fqn(predictor)!r}"
        )
    model = getattr(predictor, "model", None)
    if model is None or _fqn(model) != EXPECTED_MODEL_FQN:
        raise Sam31CompatContractError(
            f"model identity drift: {_fqn(model) if model is not None else None!r}"
        )
    observed = {
        "init_state": _parameter_shape(model.init_state),
        "add_prompt": _parameter_shape(model.add_prompt),
        "propagate_in_video": _parameter_shape(model.propagate_in_video),
    }
    expected = {
        "init_state": EXPECTED_INIT_STATE_SIGNATURE,
        "add_prompt": EXPECTED_ADD_PROMPT_SIGNATURE,
        "propagate_in_video": EXPECTED_PROPAGATE_SIGNATURE,
    }
    for name in expected:
        if observed[name] != expected[name]:
            raise Sam31CompatContractError(
                f"{name} signature drift: expected={expected[name]!r}, "
                f"observed={observed[name]!r}"
            )
    return {
        "predictor_fqn": _fqn(predictor),
        "model_fqn": _fqn(model),
        "signatures_match": True,
        "offload_state_to_cpu_supported_by_model": False,
    }


def _validate_identity(identity: Sam31PinnedIdentity) -> None:
    expected = Sam31PinnedIdentity()
    if identity != expected:
        raise Sam31CompatContractError(
            f"pinned identity drift: expected={expected!r}, observed={identity!r}"
        )
    if identity.use_rope_real is not False:
        raise Sam31CompatContractError("use_rope_real must remain false")
    if identity.strict_state_dict_load is not True:
        raise Sam31CompatContractError("strict state-dict load must remain true")


def _validate_request_keys(
    request: Mapping[str, Any], required: set[str], optional: set[str]
) -> None:
    keys = set(request)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise Sam31CompatContractError(f"request missing keys: {sorted(missing)}")
    if unknown:
        raise Sam31CompatContractError(f"request has unknown keys: {sorted(unknown)}")


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise Sam31CompatContractError(f"{name} must be bool")
    return value


class Sam31CompatAdapterV1:
    """Finite request router over the pinned official multiplex model."""

    def __init__(
        self,
        predictor: Any,
        *,
        identity: Sam31PinnedIdentity = Sam31PinnedIdentity(),
    ) -> None:
        _validate_identity(identity)
        self.predictor = predictor
        self.model = predictor.model
        self.identity = identity
        self.signature_evidence = validate_pinned_signatures(predictor)
        self._sessions: dict[str, dict[str, Any]] = {}

    def handle_request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        request_type = request.get("type")
        if request_type == "start_session":
            _validate_request_keys(
                request,
                {"type", "resource_path"},
                {
                    "session_id",
                    "offload_video_to_cpu",
                    "offload_state_to_cpu",
                    "async_loading_frames",
                },
            )
            resource_path = request["resource_path"]
            if not isinstance(resource_path, str) or not resource_path:
                raise Sam31CompatContractError("resource_path must be non-empty str")
            offload_video = _bool(
                request.get("offload_video_to_cpu", False),
                "offload_video_to_cpu",
            )
            offload_state = _bool(
                request.get("offload_state_to_cpu", False),
                "offload_state_to_cpu",
            )
            async_loading = _bool(
                request.get("async_loading_frames", False),
                "async_loading_frames",
            )
            if offload_state:
                raise Sam31CompatContractError(
                    "offload_state_to_cpu=true has no pinned multiplex equivalent"
                )
            session_id = request.get("session_id") or str(uuid.uuid4())
            if not isinstance(session_id, str) or not session_id:
                raise Sam31CompatContractError("session_id must be non-empty str")
            if session_id in self._sessions:
                raise Sam31CompatContractError("session_id already exists")
            state = self.model.init_state(
                resource_path=resource_path,
                offload_video_to_cpu=offload_video,
                async_loading_frames=async_loading,
            )
            self._sessions[session_id] = state
            return {
                "session_id": session_id,
                "adapter_id": ADAPTER_ID,
                "parameter_mapping": {
                    "resource_path": "resource_path",
                    "offload_video_to_cpu": "offload_video_to_cpu",
                    "async_loading_frames": "async_loading_frames",
                    "offload_state_to_cpu": "validated_false_no_model_argument",
                },
            }
        if request_type == "add_prompt":
            _validate_request_keys(
                request,
                {"type", "session_id", "frame_index", "text"},
                {"output_prob_thresh"},
            )
            state = self._session(request["session_id"])
            frame_index = request["frame_index"]
            text = request["text"]
            threshold = request.get("output_prob_thresh", 0.5)
            if not isinstance(frame_index, int) or frame_index < 0:
                raise Sam31CompatContractError("frame_index must be non-negative int")
            if not isinstance(text, str) or not text:
                raise Sam31CompatContractError("text must be non-empty str")
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                raise Sam31CompatContractError(
                    "output_prob_thresh must be numeric in [0, 1]"
                )
            frame_index, outputs = self.model.add_prompt(
                inference_state=state,
                frame_idx=frame_index,
                text_str=text,
                output_prob_thresh=float(threshold),
            )
            return {"frame_index": frame_index, "outputs": outputs}
        if request_type == "close_session":
            _validate_request_keys(
                request, {"type", "session_id"}, set()
            )
            session_id = request["session_id"]
            state = self._sessions.pop(session_id, None)
            if state is None:
                raise Sam31CompatContractError("unknown session_id")
            if isinstance(state, dict):
                state.clear()
            return {"is_success": True}
        raise Sam31CompatContractError(f"unsupported request type: {request_type!r}")

    def handle_stream_request(
        self, request: Mapping[str, Any]
    ) -> Iterator[dict[str, Any]]:
        _validate_request_keys(
            request,
            {"type", "session_id"},
            {
                "propagation_direction",
                "start_frame_index",
                "max_frame_num_to_track",
                "output_prob_thresh",
            },
        )
        if request["type"] != "propagate_in_video":
            raise Sam31CompatContractError(
                f"unsupported stream request type: {request['type']!r}"
            )
        state = self._session(request["session_id"])
        direction = request.get("propagation_direction", "forward")
        if direction not in {"forward", "backward"}:
            raise Sam31CompatContractError(
                "propagation_direction must be forward or backward"
            )
        start = request.get("start_frame_index")
        maximum = request.get("max_frame_num_to_track")
        threshold = request.get("output_prob_thresh", 0.5)
        if start is not None and (not isinstance(start, int) or start < 0):
            raise Sam31CompatContractError(
                "start_frame_index must be non-negative int or null"
            )
        if maximum is not None and (not isinstance(maximum, int) or maximum <= 0):
            raise Sam31CompatContractError(
                "max_frame_num_to_track must be positive int or null"
            )
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise Sam31CompatContractError(
                "output_prob_thresh must be numeric in [0, 1]"
            )
        for frame_index, outputs in self.model.propagate_in_video(
            inference_state=state,
            start_frame_idx=start,
            max_frame_num_to_track=maximum,
            reverse=direction == "backward",
            output_prob_thresh=float(threshold),
        ):
            yield {"frame_index": frame_index, "outputs": outputs}

    def _session(self, session_id: Any) -> dict[str, Any]:
        if not isinstance(session_id, str) or session_id not in self._sessions:
            raise Sam31CompatContractError("unknown session_id")
        return self._sessions[session_id]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_fd(descriptor: int, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, chunk_size, offset)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Sam31CompatContractError(f"invalid canonical provenance JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Sam31CompatContractError(f"canonical provenance is not an object: {path}")
    return value


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    digest = hashlib.sha1()  # noqa: S324 - Git object identity is defined as SHA-1.
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.hexdigest()


def _parse_stage_zero_regular_files(raw: bytes) -> list[tuple[str, str, Path]]:
    rows: list[tuple[str, str, Path]] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        try:
            metadata, raw_relative = entry.split(b"\t", 1)
            mode, blob, stage = metadata.decode("ascii").split(" ")
            relative = Path(os.fsdecode(raw_relative))
        except (ValueError, UnicodeError) as exc:
            raise Sam31CompatContractError("invalid git ls-files stage record") from exc
        if stage != "0" or mode not in {"100644", "100755"}:
            raise Sam31CompatContractError(
                f"unsupported tracked entry mode/stage: {mode} {stage} {relative}"
            )
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise Sam31CompatContractError(f"unsafe tracked path: {relative}")
        rows.append((mode, blob, relative))
    return rows


def verify_canonical_source_identity(
    code_root: Path,
    project_root: Path,
    *,
    expected_runtime_contract_sha256: str = CANONICAL_RUNTIME_CONTRACT_SHA256,
    expected_switch_receipt_sha256: str = CANONICAL_SWITCH_RECEIPT_SHA256,
    expected_commit: str = OFFICIAL_CODE_COMMIT,
    expected_tree: str = OFFICIAL_GIT_TREE,
    expected_tracked_files: int = OFFICIAL_TRACKED_REGULAR_FILES,
    expected_tracked_bytes: int = OFFICIAL_TRACKED_REGULAR_BYTES,
) -> dict[str, Any]:
    """Verify a git-metadata-free canonical tree against its retained source.

    The canonical deployment intentionally excludes ``.git``.  Therefore a
    ``git -C canonical rev-parse`` would silently resolve the enclosing project
    repository and is never an acceptable SAM identity.  Instead, this gate
    validates the immutable activation receipt and runtime contract, then
    proves every tracked regular file is the exact Git blob and the same inode
    as the retained clean official checkout.
    """

    project = project_root.resolve(strict=True)
    canonical = code_root.resolve(strict=True)
    if canonical != (project / CANONICAL_CODE_RELATIVE).resolve(strict=True):
        raise Sam31CompatContractError("canonical SAM code root identity mismatch")
    if code_root.is_symlink() or (canonical / ".git").exists():
        raise Sam31CompatContractError(
            "canonical SAM tree must be ordinary and intentionally git-metadata-free"
        )

    runtime_path = project / CANONICAL_RUNTIME_CONTRACT_RELATIVE
    receipt_path = project / CANONICAL_SWITCH_RECEIPT_RELATIVE
    for path, expected_sha in (
        (runtime_path, expected_runtime_contract_sha256),
        (receipt_path, expected_switch_receipt_sha256),
    ):
        metadata = path.stat(follow_symlinks=False)
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise Sam31CompatContractError(f"canonical provenance is not regular: {path}")
        observed_sha = sha256_file(path)
        if observed_sha != expected_sha:
            raise Sam31CompatContractError(
                f"canonical provenance SHA mismatch: {path} {observed_sha}"
            )
    runtime = _json_object(runtime_path)
    receipt = _json_object(receipt_path)
    official = runtime.get("official_code", {})
    operation = receipt.get("operations", {}).get("canonical_code", {})
    switch_qa = receipt.get("switch_qa", {}).get("official_code", {})
    expected_identity = {
        "canonical_root": CANONICAL_CODE_RELATIVE.as_posix(),
        "commit": expected_commit,
        "git_tree": expected_tree,
        "source_root": official.get("source_root"),
        "tracked_regular_files": expected_tracked_files,
        "tracked_regular_bytes": expected_tracked_bytes,
    }
    if (
        runtime.get("status") != "CANONICAL_SOURCE_CONTRACT"
        or official.get("canonical_root") != expected_identity["canonical_root"]
        or official.get("commit") != expected_commit
        or official.get("git_tree") != expected_tree
        or official.get("tracked_regular_files") != expected_tracked_files
        or official.get("tracked_regular_bytes") != expected_tracked_bytes
        or official.get("nested_git_directory_included") is not False
        or official.get("materialization")
        != "SAME_FILESYSTEM_HARDLINK_EACH_GIT_TRACKED_REGULAR_FILE"
    ):
        raise Sam31CompatContractError("canonical runtime source contract mismatch")
    if (
        receipt.get("status") != "COMPLETE_QA_PASS"
        or receipt.get("runtime_candidate_sha256")
        != expected_runtime_contract_sha256
        or operation.get("target") != expected_identity["canonical_root"]
        or operation.get("source") != expected_identity["source_root"]
        or operation.get("commit") != expected_commit
        or operation.get("git_tree") != expected_tree
        or operation.get("files") != expected_tracked_files
        or operation.get("bytes") != expected_tracked_bytes
        or operation.get("materialization")
        != "SAME_FILESYSTEM_HARDLINK_EACH_GIT_TRACKED_REGULAR_FILE"
        or operation.get("source_moved_or_modified") is not False
        or switch_qa.get("canonical_root") != expected_identity["canonical_root"]
        or switch_qa.get("source_root") != expected_identity["source_root"]
        or switch_qa.get("commit") != expected_commit
        or switch_qa.get("git_tree") != expected_tree
        or switch_qa.get("tracked_regular_files") != expected_tracked_files
        or switch_qa.get("tracked_regular_bytes") != expected_tracked_bytes
        or switch_qa.get("git_status_clean") is not True
        or switch_qa.get("canonical_hardlink_mismatches") != []
    ):
        raise Sam31CompatContractError("canonical FINAL switch receipt mismatch")

    source_relative = Path(expected_identity["source_root"])
    if source_relative.is_absolute() or ".." in source_relative.parts:
        raise Sam31CompatContractError("unsafe retained official source path")
    source = project / source_relative
    source_resolved = source.resolve(strict=True)
    if source.is_symlink() or not source_resolved.is_dir() or not (source_resolved / ".git").exists():
        raise Sam31CompatContractError("retained official source checkout is invalid")
    head = subprocess.check_output(
        ["git", "-C", str(source_resolved), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(source_resolved), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    if head != expected_commit or tree != expected_tree:
        raise Sam31CompatContractError(
            f"retained official git identity mismatch: commit={head} tree={tree}"
        )
    tracked_raw = subprocess.check_output(
        ["git", "-C", str(source_resolved), "ls-files", "--stage", "-z"]
    )
    rows = _parse_stage_zero_regular_files(tracked_raw)
    if len(rows) != expected_tracked_files:
        raise Sam31CompatContractError(
            f"retained official tracked-file count mismatch: {len(rows)}"
        )
    total_bytes = 0
    mode_counts: dict[str, int] = {}
    for mode, blob, relative in rows:
        source_file = source_resolved / relative
        canonical_file = canonical / relative
        source_stat = source_file.stat(follow_symlinks=False)
        canonical_stat = canonical_file.stat(follow_symlinks=False)
        if not stat.S_ISREG(source_stat.st_mode) or not stat.S_ISREG(canonical_stat.st_mode):
            raise Sam31CompatContractError(f"tracked path is not regular: {relative}")
        if (source_stat.st_dev, source_stat.st_ino) != (
            canonical_stat.st_dev,
            canonical_stat.st_ino,
        ):
            raise Sam31CompatContractError(f"canonical hardlink identity mismatch: {relative}")
        executable = bool(source_stat.st_mode & 0o111)
        if executable != (mode == "100755"):
            raise Sam31CompatContractError(f"tracked executable mode mismatch: {relative}")
        if _git_blob_sha1(source_file) != blob:
            raise Sam31CompatContractError(f"retained official Git blob mismatch: {relative}")
        total_bytes += source_stat.st_size
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    if total_bytes != expected_tracked_bytes:
        raise Sam31CompatContractError(
            f"retained official tracked-byte count mismatch: {total_bytes}"
        )
    return {
        "method": "FINAL_RECEIPT_GIT_BLOB_AND_HARDLINK_IDENTITY",
        "official_code_commit": head,
        "official_git_tree": tree,
        "runtime_contract_path": str(runtime_path),
        "runtime_contract_sha256": expected_runtime_contract_sha256,
        "switch_receipt_path": str(receipt_path),
        "switch_receipt_sha256": expected_switch_receipt_sha256,
        "source_root": str(source_resolved),
        "canonical_root": str(canonical),
        "tracked_regular_files": len(rows),
        "tracked_regular_bytes": total_bytes,
        "mode_counts": mode_counts,
        "all_tracked_git_blobs_verified": True,
        "all_canonical_files_same_source_inode": True,
        "enclosing_project_git_head_used": False,
    }


def verify_official_code_identity(code_root: Path) -> dict[str, Any]:
    canonical = PROJECT_ROOT / CANONICAL_CODE_RELATIVE
    if code_root == canonical.resolve(strict=True):
        return verify_canonical_source_identity(code_root, PROJECT_ROOT)
    head = subprocess.check_output(
        ["git", "-C", str(code_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != OFFICIAL_CODE_COMMIT:
        raise Sam31CompatContractError(f"official code commit mismatch: {head}")
    return {
        "method": "NESTED_GIT_HEAD",
        "official_code_commit": head,
        "enclosing_project_git_head_used": False,
    }


def build_pinned_adapter(
    *,
    official_code_root: Path,
    checkpoint_path: Path,
    checkpoint_fd: int | None = None,
    official_source_payloads: Mapping[str, bytes] | None = None,
    official_snapshot_root: Path | None = None,
    expected_official_source_bundle_sha256: str | None = None,
    official_bpe_fd: int | None = None,
) -> tuple[Sam31CompatAdapterV1, dict[str, Any]]:
    """Build, strictly reload, and wrap only the immutable pinned identity."""
    code_root = official_code_root.resolve(strict=True)
    if official_code_root.is_symlink() or not code_root.is_dir():
        raise Sam31CompatContractError("official_code_root must be ordinary directory")
    held_before = None
    if checkpoint_fd is None:
        checkpoint = checkpoint_path.resolve(strict=True)
        if checkpoint_path.is_symlink() or not checkpoint.is_file():
            raise Sam31CompatContractError(
                "checkpoint must be ordinary non-symlink file"
            )
        checkpoint_size = checkpoint.stat().st_size
        checkpoint_for_loader: str | Path = checkpoint
    else:
        held_before = os.fstat(checkpoint_fd)
        if not stat.S_ISREG(held_before.st_mode):
            raise Sam31CompatContractError("held checkpoint FD must be regular file")
        checkpoint_size = held_before.st_size
        checkpoint_for_loader = f"/proc/self/fd/{checkpoint_fd}"
    code_identity = verify_official_code_identity(code_root)
    head = code_identity["official_code_commit"]
    if checkpoint_size != CHECKPOINT_BYTES:
        raise Sam31CompatContractError("checkpoint byte count mismatch")
    checksum_started = time.perf_counter()
    checkpoint_sha = (
        sha256_file(checkpoint)
        if checkpoint_fd is None
        else sha256_fd(checkpoint_fd)
    )
    checksum_seconds = time.perf_counter() - checksum_started
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise Sam31CompatContractError("checkpoint SHA256 mismatch")

    held_finder = None
    source_bundle_sha = None
    if official_source_payloads is not None or official_snapshot_root is not None:
        if official_source_payloads is None or official_snapshot_root is None:
            raise Sam31CompatContractError("official held source inputs are incomplete")
        if any(
            name == "sam3" or name.startswith("sam3.") for name in sys.modules
        ):
            raise Sam31CompatContractError("sam3 was imported before held source install")
        snapshot_root = official_snapshot_root.resolve(strict=True)
        if official_snapshot_root.is_symlink() or not snapshot_root.is_dir():
            raise Sam31CompatContractError("official snapshot root is invalid")
        source_bundle_sha = official_source_bundle_sha256(official_source_payloads)
        if (
            expected_official_source_bundle_sha256 is None
            or source_bundle_sha != expected_official_source_bundle_sha256
        ):
            raise Sam31CompatContractError("official held source bundle mismatch")
        if "sam3/model_builder.py" not in official_source_payloads:
            raise Sam31CompatContractError("held official model builder is missing")
        held_finder = HeldOfficialSourceFinder(
            official_source_payloads, snapshot_root, source_bundle_sha
        )
        sys.meta_path.insert(0, held_finder)

    torch = importlib.import_module("torch")
    builder_module = importlib.import_module("sam3.model_builder")
    builder_file = Path(inspect.getfile(builder_module)).resolve(strict=True)
    expected_builder_root = (
        official_snapshot_root.resolve(strict=True)
        if held_finder is not None and official_snapshot_root is not None
        else code_root
    )
    try:
        builder_file.relative_to(expected_builder_root)
    except ValueError as error:
        raise Sam31CompatContractError(
            f"sam3 builder was not imported from pinned source root: {builder_file}"
        ) from error
    if held_finder is not None and (
        getattr(builder_module, "__held_source_sha256__", None)
        != hashlib.sha256(official_source_payloads["sam3/model_builder.py"]).hexdigest()
        or getattr(builder_module, "__held_official_bundle_sha256__", None)
        != source_bundle_sha
    ):
        raise Sam31CompatContractError("official builder code object is not held-byte bound")

    bpe_descriptor = None
    close_bpe_descriptor = False
    bpe_path = None
    if held_finder is not None:
        bpe_relative = "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        if bpe_relative not in official_source_payloads:
            raise Sam31CompatContractError("held official BPE payload is missing")
        bpe_file = expected_builder_root / bpe_relative
        if official_bpe_fd is None:
            bpe_descriptor = os.open(
                bpe_file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            close_bpe_descriptor = True
        else:
            bpe_descriptor = official_bpe_fd
        if sha256_fd(bpe_descriptor) != hashlib.sha256(
            official_source_payloads[bpe_relative]
        ).hexdigest():
            if close_bpe_descriptor:
                os.close(bpe_descriptor)
            raise Sam31CompatContractError("held official BPE snapshot mismatch")
        bpe_path = f"/proc/self/fd/{bpe_descriptor}"

    build_started = time.perf_counter()
    try:
        predictor = builder_module.build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint_for_loader),
            bpe_path=bpe_path,
            max_num_objects=16,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=False,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
        )
    finally:
        if bpe_descriptor is not None and close_bpe_descriptor:
            os.close(bpe_descriptor)
    torch.cuda.synchronize()
    build_seconds = time.perf_counter() - build_started

    load_started = time.perf_counter()
    payload = torch.load(
        checkpoint_for_loader, map_location="cpu", weights_only=True
    )
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict) or not state:
        raise Sam31CompatContractError("checkpoint state dict is empty or invalid")
    if any(
        key.startswith("sam3_model.") or key.startswith("sam2_predictor.")
        for key in state
    ):
        raise Sam31CompatContractError("legacy checkpoint key remapping is forbidden")
    incompatible = predictor.model.load_state_dict(state, strict=True)
    torch.cuda.synchronize()
    strict_seconds = time.perf_counter() - load_started
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise Sam31CompatContractError(
            f"strict state dict incompatibility: {incompatible}"
        )
    checkpoint_numel = sum(
        value.numel() for value in state.values() if isinstance(value, torch.Tensor)
    )
    del state, payload

    if checkpoint_fd is not None:
        held_after = os.fstat(checkpoint_fd)
        assert held_before is not None
        if (
            held_after.st_dev,
            held_after.st_ino,
            held_after.st_size,
            held_after.st_nlink,
            held_after.st_mtime_ns,
            held_after.st_ctime_ns,
        ) != (
            held_before.st_dev,
            held_before.st_ino,
            held_before.st_size,
            held_before.st_nlink,
            held_before.st_mtime_ns,
            held_before.st_ctime_ns,
        ):
            raise Sam31CompatContractError("held checkpoint changed during model load")

    adapter = Sam31CompatAdapterV1(predictor)
    evidence = {
        "adapter_id": ADAPTER_ID,
        "official_code_commit": head,
        "official_source_execution": {
            "method": (
                "HELD_PAYLOAD_META_PATH_LOADER"
                if held_finder is not None
                else code_identity["method"]
            ),
            "source_bundle_sha256": source_bundle_sha,
            "snapshot_root": (
                str(expected_builder_root) if held_finder is not None else None
            ),
            "code_identity": code_identity,
        },
        "checkpoint_bytes": checkpoint_size,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_same_held_fd_consumed": checkpoint_fd is not None,
        "checkpoint_checksum_seconds": checksum_seconds,
        "use_rope_real": False,
        "strict_state_dict_load": True,
        "strict_missing_keys": [],
        "strict_unexpected_keys": [],
        "build_seconds": build_seconds,
        "strict_load_seconds": strict_seconds,
        "checkpoint_tensor_numel": checkpoint_numel,
        "parameter_numel": sum(
            parameter.numel() for parameter in predictor.model.parameters()
        ),
        "signature_evidence": adapter.signature_evidence,
    }
    return adapter, evidence
