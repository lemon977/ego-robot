"""Strict proof chain for the isolated pick/place candidate stage receipts.

A completed stage is consumed only after its producer terminal and governance
ledger pin an INDEX that pins this exact receipt. An intact different binding is
a cache miss. A malformed, partial, forged or unsealed proof is an explicit
rejection. This does not relax the selected parent stage-reuse contract.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from chaoyang.exact78_baseline.stage_reuse import ref, verify

SCHEMA = "pick_place_1008_sealed_stage_v1"
TERMINAL_STATUSES = frozenset({"PASSED", "FAILED_RUNTIME_FINAL"})
STAGES = frozenset({"source", "motion", "gpu", "render"})

class StageReuseDenied(RuntimeError):
    pass


def _deny(reason: str):
    raise StageReuseDenied("PICK_PLACE_STAGE_REUSE_DENIED:" + reason)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        _deny("JSON_OBJECT_REQUIRED")
    return value


def _canonical_hash(value) -> str:
    # Identical canonicalization to the producer pipeline; no changed wire rule.
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def _plain(path: Path, repo: Path) -> Path:
    """Require a lexical project path with no symlink component below repo."""
    absolute = path.absolute()
    if not absolute.is_relative_to(repo):
        _deny("PATH_OUTSIDE_PROJECT:" + str(path))
    for item in (absolute, *absolute.parents):
        if item == repo:
            break
        if item.is_symlink():
            _deny("SYMLINK_PATH:" + str(item))
    return absolute.resolve(strict=True)


def _pin(pin: dict, expected_path: Path | None = None, repo: Path | None = None):
    if not isinstance(pin, dict) or set(pin) != {"path", "bytes", "sha256"}:
        _deny("ARTIFACT_REFERENCE_SHAPE")
    if expected_path is not None:
        if pin["path"] != str(expected_path):
            _deny("ARTIFACT_PATH_MISMATCH:" + str(pin.get("path")))
        if repo is not None:
            _plain(expected_path, repo)
    verify(pin)


def _validate(repo: Path, root: Path, sid: str, stage: str, expected: dict,
              required_files: tuple[str, ...]):
    folder = root / "sessions" / sid
    path = folder / ("STAGE_" + stage + ".json")
    if not path.exists() and not path.is_symlink():
        return None, None
    _plain(path, repo)
    receipt = _read(path)
    if receipt.get("schema") != SCHEMA or receipt.get("status") != "COMPLETE":
        _deny("IDENTITY_SCHEMA_OR_INCOMPLETE_STAGE")
    binding = receipt.get("binding")
    if not isinstance(binding, dict):
        _deny("BINDING_MISSING")
    declared = receipt.get("binding_sha256")
    if _canonical_hash(binding) != declared:
        _deny("BINDING_DIGEST_TAMPERED")
    if (binding.get("stage") != stage or binding.get("session_id") != sid
            or type(binding.get("raw_frame_count")) is not int
            or binding["raw_frame_count"] <= 0):
        _deny("STAGE_SESSION_OR_FRAME_IDENTITY_MISMATCH")
    if _canonical_hash(expected) != declared:
        return None, None
    files = receipt.get("files")
    if not isinstance(files, dict) or set(files) != set(required_files):
        _deny("MANDATORY_OUTPUT_SET_MISMATCH")
    for name in required_files:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            _deny("MANDATORY_OUTPUT_NAME_ESCAPE")
        _pin(files[name], folder / relative, repo)
    code = binding.get("code_and_dependencies")
    if not isinstance(code, list) or not code:
        _deny("CODE_DEPENDENCY_CLOSURE_MISSING")
    for pin in code:
        _pin(pin)
    raw = binding.get("raw_source")
    raw = list(raw.values()) if isinstance(raw, dict) else raw
    if not isinstance(raw, list) or not raw:
        _deny("RAW_SOURCE_CLOSURE_MISSING")
    for pin in raw:
        _pin(pin)
    _pin(binding["camera_config"])
    upstream = binding.get("upstream")
    if not isinstance(upstream, dict):
        _deny("UPSTREAM_CLOSURE_MISSING")
    for pin in upstream.values():
        _pin(pin)
    producer = receipt.get("producer")
    if not isinstance(producer, dict):
        _deny("PRODUCER_MISSING")
    task_id = producer.get("task_id")
    if (not isinstance(task_id, str) or not task_id.startswith("exact78_pick_place_")
            or Path(task_id).name != task_id or "/" in task_id or "\\" in task_id):
        _deny("PRODUCER_TASK_IDENTITY")
    packet_path = repo / "tasks/current" / task_id / "TASK_PACKET.json"
    _pin(producer["task_packet"], packet_path, repo)
    packet = _read(packet_path)
    if (packet.get("task_id") != task_id or packet.get("run_signature") != producer.get("run_signature")
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"):
        _deny("PRODUCER_PACKET_IDENTITY_OR_SIGNATURE")
    result_path = repo / "_run/current" / task_id / "attempts/attempt_0001/RESULT.json"
    if producer.get("result_path") != str(result_path):
        _deny("PRODUCER_TERMINAL_PATH_MISMATCH")
    if root != result_path.parent / "candidate" or folder != result_path.parent / "candidate/sessions" / sid:
        _deny("STAGE_NOT_OWNED_BY_PRODUCER")
    ledger = _read(repo / "docs/governance/LONG_HORIZON_TASK_STATE.json")
    rows = [row for row in ledger.get("tasks", []) if row.get("task_id") == task_id]
    if len(rows) != 1 or rows[0].get("status") not in TERMINAL_STATUSES:
        _deny("PRODUCER_NOT_SEALED_TERMINAL")
    row = rows[0]
    if row.get("pid") is not None or row.get("proc_start_ticks") is not None:
        _deny("PRODUCER_TERMINAL_WRITER_PRESENT")
    _pin(row["result"], result_path, repo)
    terminal = _read(result_path)
    if (terminal.get("status") != row["status"] or terminal.get("task_id") != task_id
            or terminal.get("run_signature") != producer.get("run_signature")
            or terminal.get("task_packet") != producer["task_packet"]):
        _deny("PRODUCER_TERMINAL_IDENTITY_OR_SIGNATURE")
    index_path = result_path.parent / "candidate/INDEX.json"
    index_pin = terminal.get("required_outputs", {}).get("candidate/INDEX.json")
    if not isinstance(index_pin, dict):
        _deny("TERMINAL_INDEX_REFERENCE_MISSING")
    _pin(index_pin, index_path, repo)
    index = _read(index_path)
    rows = [row for row in index.get("sessions", []) if row.get("session_id") == sid]
    if len(rows) != 1:
        _deny("PRODUCER_INDEX_SESSION_IDENTITY")
    row = rows[0]
    direct = row.get("stage_receipts", {}).get(stage)
    nested = row.get("stages", {}).get(stage, {}).get("receipt")
    if direct is not None and nested is not None and direct != nested:
        _deny("PRODUCER_INDEX_STAGE_REFERENCE_CONFLICT")
    bound = direct if direct is not None else nested
    current = ref(path)
    if bound != current:
        _deny("PRODUCER_INDEX_DOES_NOT_BIND_STAGE")
    _pin(bound, path, repo)
    return folder, current


def find_reuse(repo, roots, sid, stage, expected, required_files):
    """Return one legally sealed match; reject bad evidence explicitly."""
    repo = Path(repo).resolve(strict=True)
    if (stage not in STAGES or not isinstance(sid, str) or not sid or sid in {".", ".."}
            or Path(sid).name != sid or "\\" in sid):
        _deny("LOOKUP_STAGE_OR_SESSION")
    required_files = tuple(required_files)
    if not required_files or len(required_files) != len(set(required_files)):
        _deny("MANDATORY_OUTPUT_CONTRACT_INVALID")
    for item in roots:
        root = Path(item).absolute()
        if not root.is_relative_to(repo):
            _deny("REUSE_ROOT_OUTSIDE_PROJECT")
        if not root.exists():
            continue
        try:
            root = _plain(root, repo)
            folder, receipt = _validate(repo, root, sid, stage, expected, required_files)
            if folder is not None:
                return folder, receipt
        except StageReuseDenied:
            raise
        except (OSError, KeyError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
            _deny("INVALID_OR_UNSEALED_RECEIPT:" + str(exc))
    return None, None
