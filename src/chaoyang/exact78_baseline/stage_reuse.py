"""Evidence-bound stage sealing and reuse; no solver or model imports.

Legacy OK files are not a reuse contract. New receipts are eligible only after
their producer seals the index that binds the receipt and session outputs.
"""
from __future__ import annotations

from functools import lru_cache
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SCHEMA = "EXACT78_SEALED_STAGE_V1"
FILES = {"motion": ("motion.npz", "motion.json"),
         "gpu": ("clean.mp4", "masks.npz", "masks.json"),
         "render": ("final.mp4", "review.mp4", "SESSION.json")}
ROOT_MODULES = {
    "motion": ("exact78_baseline/pico_source.py", "exact78_baseline/retarget.py", "exact78_baseline/robot_model.py"),
    "gpu": ("exact78_baseline/mask.py", "exact78_baseline/clean.py", "pipeline/sam31_compat_adapter_v1.py"),
    "render": ("exact78_baseline/render.py", "pipeline/exact78_visual_compositor_v1.py"),
}


class StageReuseDenied(RuntimeError):
    pass


def _deny(reason):
    raise StageReuseDenied("STAGE_REUSE_DENIED:" + reason)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


@lru_cache(maxsize=4096)
def _hash_file(path, device, inode, size, mtime_ns, ctime_ns):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    current = Path(path).stat()
    # ctime invalidates the cache (including restored-mtime writes), but this
    # filesystem may update it on first read after atomic rename. It is not
    # itself evidence of a content mutation during the read.
    if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (device, inode, size, mtime_ns):
        _deny("FILE_CHANGED_WHILE_HASHING:" + path)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size,
            "sha256": _hash_file(str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)}


def verify(row):
    try:
        if ref(row["path"]) != row:
            _deny("ARTIFACT_SHA_OR_SIZE_DRIFT:" + str(row.get("path")))
    except (OSError, KeyError, TypeError) as exc:
        _deny("ARTIFACT_MISSING_OR_INVALID:" + str(exc))


def receipt_path(folder, stage):
    return Path(folder) / ("STAGE_" + stage + ".json")


def input_fingerprint(session):
    arrays = {}
    for name in ("K", "c2w", "c2w_editor", "hands", "hand_observed", "hand_source", "body", "body_observed", "gravity"):
        value = np.ascontiguousarray(getattr(session, name))
        arrays[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                        "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    return {"session": session.session_id, "raw_dir": str(Path(session.raw_dir).resolve()),
            "video": ref(session.video), "frame_count": session.frame_count, "fps": session.fps,
            "width": session.width, "height": session.height, "calibration": copy.deepcopy(session.calibration),
            "semantic_arrays": arrays}


def dependencies(root, stage):
    from chaoyang.governance.source_snapshot_closure_v1 import local_import_closure
    seeds = [root / "src/chaoyang" / name for name in ROOT_MODULES[stage]]
    seeds += [root / "src/chaoyang/exact78_baseline/pipeline.py", Path(__file__).resolve()]
    paths = set(local_import_closure(root, seeds))
    if stage in ("motion", "render"):
        paths.update(root / name for name in (
            "assets/robot/ROBOT_ASSET_PIN.json",
            "assets/robot/hardware_handoff/kaihand_flange_adapter_v1/excluded_reference_only/MOUNT_VISUAL_PROXY_CONTRACT.json",
            "assets/robot/V2相机固定法兰PRO版本.STL"))
        # The pin document alone is not proof that its referenced asset bytes
        # remain intact, especially on reuse paths that never construct Robot.
        pin = json.loads((root / "assets/robot/ROBOT_ASSET_PIN.json").read_text())
        for row in pin["files"]:
            path = (root / row["path"]).resolve()
            if Path(root).resolve() not in path.parents:
                _deny("PINNED_ASSET_ESCAPES_PROJECT")
            verify({"path": str(path), "bytes": row["bytes"], "sha256": row["sha256"]})
            paths.add(path)
    if stage == "gpu":
        paths.add(root / "assets/models/checkpoints/sam3.1/sam3.1_multiplex.pt")
        paths.update(root / "vendor/ProPainter/weights" / name for name in
                     ("raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth"))
        paths.update(path for path in (root / "vendor/ProPainter").rglob("*.py") if "__pycache__" not in path.parts)
        paths.update(path for path in (root / "vendor/SAM3/sam3").rglob("*.py") if "__pycache__" not in path.parts)
    return [ref(path) for path in sorted(paths)]


def binding(root, stage, session, *, objects=True, propainter_settings=None, upstream=None):
    return {"stage": stage, "input": input_fingerprint(session), "dependencies": dependencies(Path(root), stage),
            "config": {"width": 640, "height": 480, **({"objects": bool(objects),
                       "propainter_settings": propainter_settings or {}} if stage == "gpu" else {})},
            "upstream": {name: ref(path) for name, path in sorted((upstream or {}).items())}}


def signature(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def current_producer(root):
    from chaoyang.governance.exact78_operation_gate import _command_tail, authorize_exact78_operation
    index = json.loads((root / "tasks/current/INDEX.json").read_text())
    rows = [row for row in index["task_packets"] if row.get("execution_allowed") is True]
    if len(rows) != 1:
        _deny("NO_UNIQUE_PRODUCER")
    row = rows[0]
    packet_path = root / row["packet_path"]
    packet = json.loads(packet_path.read_text())
    if (ref(packet_path)["sha256"] != row["packet_sha256"]
            or packet.get("execution_mode") != "EXACT78_CANDIDATE"):
        _deny("PRODUCER_PACKET_UNFENCED")
    tail = _command_tail(packet["commands"][0])
    authorize_exact78_operation(tail[0], tail[1:], repo_root=root, caller_pid=os.getpid())
    return {"task_id": packet["task_id"], "task_packet": ref(packet_path),
            "run_signature": packet["run_signature"],
            "result_path": str(root / "_run/current" / packet["task_id"] / "attempts/attempt_0001/RESULT.json")}


def seal(root, folder, stage, session, *, objects=True, propainter_settings=None, upstream=None, before=None):
    folder, root = Path(folder), Path(root)
    producer = current_producer(root)
    if folder.resolve() != Path(producer["result_path"]).parent / "candidate/sessions" / session.session_id:
        _deny("STAGE_OUTPUT_NOT_OWNED_BY_PRODUCER")
    contract = binding(root, stage, session, objects=objects, propainter_settings=propainter_settings, upstream=upstream)
    if before is None or signature(contract) != signature(before):
        _deny("STAGE_INPUT_OR_DEPENDENCY_CHANGED_DURING_EXECUTION")
    metadata = json.loads((folder / FILES[stage][-1]).read_text())
    if (metadata.get("status") != "OK" or metadata.get("session") != session.session_id
            or metadata.get("frame_count") != session.frame_count):
        _deny("STAGE_OUTPUT_IDENTITY_OR_COUNT_MISMATCH")
    for name in FILES[stage]:
        path = folder / name
        if path.is_symlink() or path.resolve().parent != folder.resolve():
            _deny("STAGE_OUTPUT_LINK_ESCAPES_FOLDER")
    if stage == "motion":
        with np.load(folder / "motion.npz", allow_pickle=False) as arrays:
            for name, shape in (("q_arm", (session.frame_count, 2, 7)), ("q_hand", (session.frame_count, 2, 22))):
                if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                    _deny("STAGE_OUTPUT_MOTION_SHAPE_OR_NONFINITE")
    value = {"schema_version": SCHEMA, "status": "COMPLETE_STAGE_AWAITING_PRODUCER_TERMINAL",
             "session": session.session_id, "frame_count": session.frame_count, "stage": stage,
             "folder": str(folder.resolve()), "binding": contract, "stage_signature": signature(contract),
             "producer": producer, "outputs": {name: ref(folder / name) for name in FILES[stage]},
             "claim_limit": "Integrity/compatibility only; no independent motion, visual, physical or training qualification."}
    from chaoyang.governance.common import atomic_json
    path = receipt_path(folder, stage)
    if path.exists():
        if json.loads(path.read_text()) != value:
            _deny("REFUSE_RECEIPT_OVERWRITE")
    else:
        atomic_json(path, value)
    return ref(path)


def validate(folder, stage, expected, root):
    """Return false for unsealed legacy stages; reject a claimed seal on any drift."""
    path = receipt_path(folder, stage)
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
        sid = expected["input"]["session"]
        if (value["schema_version"] != SCHEMA or value["stage"] != stage
                or value["status"] != "COMPLETE_STAGE_AWAITING_PRODUCER_TERMINAL"
                or value["session"] != sid or value["frame_count"] != expected["input"]["frame_count"]
                or Path(value["folder"]).resolve() != Path(folder).resolve()):
            _deny("IDENTITY_OR_STAGE_MISMATCH")
        if value["stage_signature"] != signature(value["binding"]):
            _deny("STAGE_SIGNATURE_TAMPERED")
        if set(value["outputs"]) != set(FILES[stage]):
            _deny("OUTPUT_CLOSURE_INCOMPLETE")
        for name, output in value["outputs"].items():
            if ((Path(folder) / name).is_symlink() or Path(output["path"]).resolve().parent != Path(folder).resolve()
                    or Path(output["path"]).resolve() != (Path(folder) / name).resolve()):
                _deny("OUTPUT_PATH_MISMATCH")
            verify(output)
        producer = value["producer"]
        verify(producer["task_packet"])
        packet = json.loads(Path(producer["task_packet"]["path"]).read_text())
        if Path(producer["task_packet"]["path"]).resolve() != Path(root).resolve() / "tasks/current" / producer["task_id"] / "TASK_PACKET.json":
            _deny("PRODUCER_PACKET_PATH_MISMATCH")
        if packet["task_id"] != producer["task_id"] or packet["run_signature"] != producer["run_signature"]:
            _deny("PRODUCER_SIGNATURE_MISMATCH")
        result_path = Path(producer["result_path"])
        root = Path(root).resolve()
        if result_path.resolve() != root / "_run/current" / producer["task_id"] / "attempts/attempt_0001/RESULT.json":
            _deny("PRODUCER_RESULT_PATH_MISMATCH")
        ledger = json.loads((root / "docs/governance/LONG_HORIZON_TASK_STATE.json").read_text())
        matches = [row for row in ledger["tasks"] if row.get("task_id") == producer["task_id"]]
        if len(matches) != 1 or matches[0]["status"] != "PASSED":
            _deny("PRODUCER_NOT_SEALED_PASSED")
        verify(matches[0]["result"])
        if Path(matches[0]["result"]["path"]).resolve() != result_path.resolve():
            _deny("LEDGER_RESULT_PATH_MISMATCH")
        result = json.loads(result_path.read_text())
        if result["status"] != "PASSED" or result["task_packet"] != producer["task_packet"]:
            _deny("PRODUCER_TERMINAL_MISMATCH")
        index_ref = result["required_outputs"]["candidate/INDEX.json"]
        verify(index_ref)
        index = json.loads(Path(index_ref["path"]).read_text())
        rows = [row for row in index["sessions"] if row["session"] == sid]
        if (len(rows) != 1 or rows[0]["status"] != "OK" or rows[0]["frames"] != value["frame_count"]
                or rows[0]["stage_receipts"][stage] != ref(path)):
            _deny("PRODUCER_INDEX_DOES_NOT_BIND_STAGE")
        # An intact seal with different code/config/upstream is a cache miss,
        # not permission to consume it and not a reason to block fresh work.
        return value["stage_signature"] == signature(expected)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        _deny("INVALID_OR_UNSEALED_RECEIPT:" + str(exc))
