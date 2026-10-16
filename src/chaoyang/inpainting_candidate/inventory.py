"""Finite raw-byte refresh, without thousands of serial NAS realpath calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import selectors
import stat
import subprocess
import sys
import time

from chaoyang.governance.common import REPO_ROOT, artifact_ref, atomic_json, load_json, now_iso

DEV = ("001", "003", "008", "100", "114", "050")
CHALLENGE = ("039", "007", "199")
RAW_ROOT_0918 = Path("/path/to/private_data/egodata/datasets/ego/chips_cards_handle_highview_0918")
WORKER_MODULE = "chaoyang.inpainting_candidate.inventory"
REQUIRED_0918 = ("manifest.json", "raw/vst.h264", "raw/vst.qpc.ts.jsonl", "raw/vst.ts.jsonl",
                 "raw/pico.jsonl", "raw/manus.jsonl", "raw/camera_params.json", "raw/camera_params.meta.json")
OPTIONAL_0918 = ("raw/manus.meta.json", "dataset.hdf5")


class RootBoundReader:
    """Hash regular files through cached, root-anchored O_NOFOLLOW descriptors."""

    def __init__(self, declared_root, resolved_root):
        self.declared = Path(declared_root)
        if not self.declared.is_absolute() or ".." in self.declared.parts:
            raise ValueError("RAW_ROOT_NOT_ABSOLUTE_OR_TRAVERSAL")
        self._fds = {(): os.open(resolved_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)}

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def relative(self, path):
        candidate = Path(path)
        if not candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("RAW_REFERENCE_TRAVERSAL")
        try:
            parts = candidate.relative_to(self.declared).parts
        except ValueError as exc:
            raise ValueError("RAW_REFERENCE_OUTSIDE_FROZEN_SOURCE_ROOT") from exc
        if not parts:
            raise ValueError("RAW_REFERENCE_NOT_FILE")
        return parts

    def directory(self, parts):
        for depth in range(1, len(parts) + 1):
            prefix = parts[:depth]
            if prefix not in self._fds:
                self._fds[prefix] = os.open(parts[depth - 1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                            dir_fd=self._fds[prefix[:-1]])
        return self._fds[parts]

    def open_file(self, path):
        parts = self.relative(path)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=self.directory(parts[:-1]))
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValueError("RAW_REFERENCE_NOT_REGULAR_FILE")
        return fd

    def read_bytes(self, path):
        with os.fdopen(self.open_file(path), "rb") as stream:
            return stream.read()

    def reference(self, path, expected=None):
        with os.fdopen(self.open_file(path), "rb") as stream:
            before = os.fstat(stream.fileno())
            if expected is not None and before.st_size != expected["bytes"]:
                raise ValueError("RAW_INPUT_BYTES_DRIFT:" + str(path))
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(stream.fileno())
        stable = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
        if stable(before) != stable(after):
            raise ValueError("RAW_INPUT_CHANGED_DURING_HASH:" + str(path))
        actual = {"path": str(Path(path)), "bytes": after.st_size, "sha256": digest.hexdigest()}
        if expected is not None and actual != expected:
            raise ValueError("RAW_INPUT_SHA_DRIFT:" + str(path))
        return actual


def ordered_refs(raw):
    rows = {row["session_id"]: row for row in raw["sessions"]}
    priority = tuple(dict.fromkeys(DEV + CHALLENGE))
    order = [sid for sid in priority if sid in rows] + sorted(set(rows) - set(priority))
    refs = {}
    for sid in order:
        for ref in rows[sid]["source_refs"].values():
            previous = refs.get(ref["path"])
            if previous is not None and previous != ref:
                raise ValueError("FROZEN_RAW_REFERENCE_CONFLICT:" + ref["path"])
            refs.setdefault(ref["path"], ref)
    return list(refs.values())


def select_0918(root, resolved_root, maximum=8):
    """Only identity, original bytes and completeness; no pixels or old output."""
    root = Path(root)
    selected, rejected, groups = [], [], set()
    with RootBoundReader(root, resolved_root) as reader:
        tasks = sorted(entry.name for entry in os.scandir(reader.directory(())) if entry.is_dir(follow_symlinks=False))
        for task in tasks:
            sessions = sorted(entry.name for entry in os.scandir(reader.directory((task,))) if entry.is_dir(follow_symlinks=False))
            for sid in sessions:
                session = root / task / sid
                missing, originals = [], {}
                for name in REQUIRED_0918 + OPTIONAL_0918:
                    try:
                        originals[name] = reader.reference(session / name)
                    except FileNotFoundError:
                        if name in REQUIRED_0918:
                            missing.append(name)
                    except (OSError, ValueError) as exc:
                        missing.append(name + ":" + type(exc).__name__)
                if missing:
                    rejected.append({"path": str(session), "status": "INPUT_INCOMPLETE_OR_UNSAFE", "missing": missing})
                    continue
                identity_bytes = reader.read_bytes(session / "manifest.json")
                if hashlib.sha256(identity_bytes).hexdigest() != originals["manifest.json"]["sha256"]:
                    raise ValueError("RAW_SOURCE_IDENTITY_CHANGED_DURING_SELECTION:" + str(session))
                identity = json.loads(identity_bytes.decode("utf-8-sig"))
                if not isinstance(identity, dict):
                    raise ValueError("RAW_SOURCE_MANIFEST_NOT_OBJECT:" + str(session))
                group = next((str(identity[key]) for key in ("original_acquisition_group", "original_recording_id", "capture_group_id") if identity.get(key)), None)
                if group is not None and group in groups:
                    rejected.append({"path": str(session), "status": "REPEATED_EXPLICIT_SOURCE_GROUP", "original_group": group})
                    continue
                if group is not None:
                    groups.add(group)
                selected.append({"session_dir": str(session), "identity": identity.get("session"),
                                 "source_manifest": originals["manifest.json"], "source_refs": originals,
                                 "original_group": group,
                                 "group_independence": "EXPLICIT_METADATA_REQUIRES_VERIFICATION" if group else "SPLIT_NOT_ESTABLISHED",
                                 "selection": "RAW_PATH_ORDER_IDENTITY_AND_FILE_COMPLETENESS_ONLY",
                                 "visual_reviewed_for_selection": False, "blind_test": False, "hdf5_datasets_read": False})
                if len(selected) >= maximum:
                    return {"status": "METADATA_SELECTION_COMPLETE", "selected": selected, "rejected": rejected}
    return {"status": "METADATA_SELECTION_COMPLETE", "selected": selected, "rejected": rejected}


def _worker():
    readers = {}
    try:
        for line in sys.stdin:
            job = json.loads(line)
            try:
                if job["kind"] == "select_0918":
                    data = select_0918(job["root"], job["resolved_root"], job["maximum"])
                else:
                    key = (job["root"], job["resolved_root"])
                    if key not in readers:
                        readers[key] = RootBoundReader(*key)
                    expected = job["expected"]
                    actual = readers[key].reference(expected["path"], expected)
                    data = {"expected": expected, "actual": actual, "status": "FULL_SHA_PASS"}
                response = {"job_id": job["job_id"], "ok": True, "result": data}
            except Exception as exc:
                response = {"job_id": job["job_id"], "ok": False, "failure_code": str(exc), "exception_type": type(exc).__name__}
            print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
    finally:
        for reader in readers.values():
            reader.close()


def refresh_jobs(jobs, output, deadline, total_raw, *, io_wait_seconds=300):
    """Two owned subprocesses; partial and kernel-I/O waits never count as PASS.

    io_wait_seconds is a runtime timeout, not a quality gate. A process which
    remains alive after its own terminate request is truthfully recorded.
    """
    started = time.monotonic()
    verified, selection, failures, pending = [], None, [], iter(jobs)
    selector, workers, busy = selectors.DefaultSelector(), [], {}

    def progress(force=False):
        current = time.monotonic()
        if force or current - progress.last >= 10:
            atomic_json(output / "INPUT_PROGRESS.json", {"status": progress.status if not failures else "BLOCKED",
                        "verified_files": len(verified), "total_raw_files": total_raw,
                        "cross_domain_status": selection["status"] if selection else "PENDING_INDEPENDENT_LANE",
                        "active_jobs": [{"pid": process.pid, "job_id": state[0]["job_id"],
                                         "path": state[0].get("expected", {}).get("path", state[0].get("root")),
                                         "waiting_seconds": current - state[1]} for process, state in busy.items()],
                        "failures": failures, "elapsed_seconds": current - started, "updated_at": now_iso()})
            progress.last = current
    progress.last = 0.0
    progress.status = "RUNNING"
    progress(True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    try:
        for _ in range(min(2, len(jobs))):
            process = subprocess.Popen([sys.executable, "-B", "-m", WORKER_MODULE, "--worker"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                       cwd=REPO_ROOT, env=environment, bufsize=1)
            workers.append(process)
            selector.register(process.stdout, selectors.EVENT_READ, process)

        def dispatch(process):
            job = next(pending, None)
            if job is not None:
                process.stdin.write(json.dumps(job, ensure_ascii=False) + "\n")
                process.stdin.flush()
                busy[process] = (job, time.monotonic())
            elif not process.stdin.closed:
                process.stdin.close()

        for process in workers:
            dispatch(process)
        while busy and not failures:
            if datetime.now(timezone.utc) >= deadline:
                failures.append({"failure_code": "FINITE_G0_DEADLINE"})
                break
            for process, (job, assigned) in list(busy.items()):
                if time.monotonic() - assigned > io_wait_seconds:
                    failures.append({"failure_code": "RAW_IO_WAIT_TIMEOUT_NOT_VERIFIED", "job_id": job["job_id"], "pid": process.pid})
                    break
            if failures:
                break
            for key, _ in selector.select(timeout=1):
                process = key.data
                line = process.stdout.readline()
                if not line:
                    selector.unregister(process.stdout)
                    if process in busy:
                        failures.append({"failure_code": "OWN_INPUT_WORKER_EXITED_WITHOUT_RESULT", "pid": process.pid})
                    continue
                result = json.loads(line)
                job, _ = busy.pop(process)
                if result["job_id"] != job["job_id"]:
                    failures.append({"failure_code": "OWN_INPUT_WORKER_JOB_MISMATCH", "pid": process.pid})
                elif not result["ok"]:
                    problem = {"job_id": job["job_id"], "failure_code": result["failure_code"], "exception_type": result["exception_type"]}
                    if job["kind"] == "select_0918":
                        selection = {"status": "BLOCKED_0918_METADATA", "selected": [], "rejected": [], "failures": [problem]}
                        atomic_json(output / "CROSS_DOMAIN_SELECTION.json", selection)
                    else:
                        failures.append(problem)
                elif job["kind"] == "select_0918":
                    selection = result["result"]
                    atomic_json(output / "CROSS_DOMAIN_SELECTION.json", selection)
                else:
                    verified.append(result["result"])
                progress(len(verified) == 1 or len(verified) % 25 == 0 or job["kind"] == "select_0918")
                if failures:
                    break
                dispatch(process)
            progress()
    finally:
        selector.close()
        for process in workers:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.stdout:
                process.stdout.close()
    if not failures and len(verified) != total_raw:
        failures.append({"failure_code": "INCOMPLETE_RAW_VERIFICATION"})
    progress.status = "COMPLETED_ACTUAL_INPUT_SHA" if not failures else "BLOCKED"
    progress(True)
    return {"verified": verified, "selection": selection, "failures": failures,
            "worker_states": [{"pid": process.pid, "exit_code": process.poll(), "still_alive": process.poll() is None} for process in workers],
            "elapsed_seconds": time.monotonic() - started}


def freeze_inputs(packet: dict, output: Path, *, cross_root: Path = RAW_ROOT_0918) -> dict:
    output = output.resolve()
    if not output.is_relative_to(Path(packet["output_dir"]).resolve()) or output.exists():
        raise ValueError("FRESH_G0_OUTPUT_REQUIRED")
    output.mkdir(parents=True)
    parent = load_json(REPO_ROOT / "inpainting_v1/MANIFEST.json")
    raw_path = Path(parent["inputs"]["raw_manifest"]["path"])
    if artifact_ref(raw_path) != parent["inputs"]["raw_manifest"]:
        raise ValueError("PARENT_RAW_MANIFEST_SHA_DRIFT")
    raw = load_json(raw_path)
    refs = ordered_refs(raw)
    atomic_json(output / "INPUT_PROGRESS.json", {"status": "VALIDATING_AUTHORISED_ROOT_ONCE", "verified_files": 0,
                "total_raw_files": len(refs), "updated_at": now_iso()})
    declared_root = Path(raw["source_root"])
    allowed = declared_root.resolve(strict=True)
    for ref in refs:
        # Lexical only: no source metadata RPC or traversal per file.
        path = Path(ref["path"])
        if ".." in path.parts or not path.is_relative_to(declared_root):
            raise ValueError("RAW_REFERENCE_OUTSIDE_FROZEN_SOURCE_ROOT")
    deadline = datetime.fromisoformat(packet["deadline_at"])
    jobs = [{"kind": "hash", "job_id": str(index), "root": str(declared_root),
             "resolved_root": str(allowed), "expected": ref} for index, ref in enumerate(refs)]
    if cross_root.is_dir():
        jobs.insert(min(1, len(jobs)), {"kind": "select_0918", "job_id": "0918_SELECTION", "root": str(cross_root),
                    "resolved_root": str(cross_root.resolve(strict=True)), "maximum": 8})
    outcome = refresh_jobs(jobs, output, deadline, len(refs))
    selection = outcome["selection"] or {"status": "NOT_EVALUATED_OR_RAW_ROOT_ABSENT", "selected": [], "rejected": []}
    rows = {row["session_id"]: row for row in raw["sessions"]}
    verified_paths = {entry["actual"]["path"] for entry in outcome["verified"]}
    ready = [sid for sid, row in rows.items() if all(ref["path"] in verified_paths for ref in row["source_refs"].values())]
    complete = not outcome["failures"] and len(outcome["verified"]) == len(refs)
    result = {"schema": "inpainting-g0-source-freeze-v2", "task_id": packet["task_id"], "run_signature": packet["run_signature"],
              "parent_raw_manifest": artifact_ref(raw_path), "regression_sessions": len(rows),
              "regression_frames": sum(row["raw_frame_count"] for row in rows.values()), "unique_raw_files": len(refs),
              "verified_files": len(outcome["verified"]), "verified_bytes": sum(entry["actual"]["bytes"] for entry in outcome["verified"]),
              "verification": "FULL_SHA_ALL_ACTUAL_RAW_FILES" if complete else "BLOCKED_ACTUAL_RAW_REFRESH_INCOMPLETE",
              "fully_verified_sessions": ready, "elapsed_seconds": outcome["elapsed_seconds"], "raw_file_checks": outcome["verified"],
              "failures": outcome["failures"], "worker_states": outcome["worker_states"],
              "development": [{"session_id": sid, "frames": rows[sid]["raw_frame_count"], "raw_source_sha_status": "PASS" if sid in ready else "UNVERIFIED"} for sid in DEV],
              "known_challenges_not_blind": [{"session_id": sid, "frames": rows[sid]["raw_frame_count"], "raw_source_sha_status": "PASS" if sid in ready else "UNVERIFIED"} for sid in CHALLENGE],
              "cross_domain_root": str(cross_root), "cross_domain_selection_status": selection["status"],
              "cross_domain_selected": selection["selected"], "cross_domain_rejected_completeness": selection["rejected"],
              "cross_domain_execution": "NOT_EXECUTED", "directory_read_policy": "ROOT_ANCHORED_NOFOLLOW_CACHED_DIRECTORY_FDS",
              "production_read_policy": "EVALUATION_FREEZE_EVIDENCE_NOT_A_FRESH_PRODUCER_DERIVED_INPUT",
              "model_calls": 0, "training_steps": 0, "external_writes": 0, "files_deleted": 0, "created_at": now_iso()}
    atomic_json(output / "INPUT_FREEZE.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker()
        return 0
    if not args.packet or not args.output:
        parser.error("--packet and --output are required")
    packet = load_json(args.packet.resolve())
    if packet.get("task_id") != "inpainting_generalization_g0_integration_20261013" or packet.get("algorithm_execution_allowed") is not False:
        raise ValueError("CURRENT_G0_CPU_PACKET_REQUIRED")
    result = freeze_inputs(packet, args.output)
    print(json.dumps({key: result[key] for key in ("verification", "regression_sessions", "regression_frames", "unique_raw_files", "verified_files", "elapsed_seconds")}, ensure_ascii=False), flush=True)
    print(json.dumps({"cross_domain_selected": len(result["cross_domain_selected"]), "cross_domain_selection_status": result["cross_domain_selection_status"], "failures": result["failures"]}, ensure_ascii=False), flush=True)
    return 0 if result["verification"] == "FULL_SHA_ALL_ACTUAL_RAW_FILES" else 2


if __name__ == "__main__":
    raise SystemExit(main())
