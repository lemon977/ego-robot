"""Explicit, bounded CPU test profiles; not a task dispatcher or model runner.

Portable tests use synthetic inputs. Server tests require the historical,
SHA-bound evidence and an independently checked current read-only view.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("project_handoff_check_v1", Path(__file__).with_name("project_handoff_check_v1.py"))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)
PROFILE_HELPERS = ("tests/helpers/__init__.py", "tests/helpers/handoff_fixtures_v1.py")
ENV_PACKAGES = ("pytest", "jsonschema")


def environment_snapshot():
    result = CHECK.environment_report(require_pytest=True)
    result["interpreter"] = sys.executable
    result["software_versions"] = {}
    for package in ENV_PACKAGES:
        try:
            result["software_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["software_versions"][package] = None
    result["claim_limit"] = "Local metadata snapshot, not GPU health, model readiness or complete installation proof."
    return result


def bounded_timeout(value):
    number = int(value)
    if not 1 <= number <= 300:
        raise CHECK.CheckError("CPU_TIMEOUT_OUT_OF_BOUNDS", value)
    return number


def output_parent(root, value=None):
    return CHECK.project_path(root, value or "_run/maintenance/project_checks_v1")


def pytest_environment(root, attempt):
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PYTHONPATH": str(Path(root) / "src"), "CUDA_VISIBLE_DEVICES": "",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                "TMPDIR": str(attempt / "tmp"), "TMP": str(attempt / "tmp"), "TEMP": str(attempt / "tmp"),
                "XDG_CACHE_HOME": str(attempt / "env_cache"), "MPLCONFIGDIR": str(attempt / "env_cache/mpl"),
                "NUMBA_CACHE_DIR": str(attempt / "env_cache/numba"),
                "PYTHONPYCACHEPREFIX": str(attempt / "env_cache/pycache"),
                "PYTEST_ADDOPTS": "", "PYTEST_PLUGINS": ""})
    return env


def pytest_command(root, profile, attempt):
    return [sys.executable, "-B", "-m", "pytest", "-q", "--disable-warnings",
            "--basetemp", str(attempt / "pytest_tmp"), "-o", "cache_dir=" + str(attempt / "pytest_cache"),
            "-o", "addopts=", "-o", "pythonpath=src"] + [str(Path(root) / name) for name in CHECK.profile_files(profile)]


def profile_inputs(root, profile):
    paths = list(CHECK.profile_files(profile)) + ["scripts/maintenance/project_handoff_check_v1.py",
                                                "scripts/maintenance/run_project_checks_v1.py"]
    if profile == "portable":
        paths += list(PROFILE_HELPERS)
    paths += [name for name in ("pyproject.toml", "tests/conftest.py") if (Path(root) / name).is_file()]
    return [CHECK.file_ref(CHECK.project_path(root, name)) for name in paths]


def cpu_limit():
    """Limit this explicit test subprocess, never other project workers."""
    if hasattr(os, "sched_getaffinity"):
        allowed = os.sched_getaffinity(0)
        if not allowed:
            raise CHECK.CheckError("CPU_AFFINITY_UNAVAILABLE")
        cpu = min(allowed)
        return lambda: os.sched_setaffinity(0, {cpu}), [cpu]
    raise CHECK.CheckError("CPU_AFFINITY_UNAVAILABLE", "Explicit one-CPU profile requires Linux affinity support.")


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def signature(value):
    return hashlib.sha256(CHECK.canonical_json(value).encode()).hexdigest()


def test_counts(stdout):
    return {name: int(found.group(1)) if (found := re.search(r"\b(\d+) " + name + r"\b", stdout)) else 0
            for name in ("passed", "failed", "skipped", "errors")}


def run_profile(root, profile, output_root=None, timeout_seconds=180):
    root = Path(root).resolve()
    timeout = bounded_timeout(timeout_seconds)
    tests = CHECK.profile_files(profile)
    environment = environment_snapshot()
    if environment["errors"]:
        raise CHECK.CheckError(environment["errors"][0], CHECK.canonical_json(environment))
    view = None
    if profile == "server":
        CHECK.require_server_evidence(root)
        view = CHECK.inspect_project(root)
    inputs = profile_inputs(root, profile)
    parent = output_parent(root, output_root)
    parent.mkdir(parents=True, exist_ok=True)
    CHECK.project_path(root, parent)
    now = datetime.now(timezone.utc)
    attempt = parent / (profile + "_" + now.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:12])
    attempt.mkdir(exist_ok=False)
    for directory in ("tmp", "env_cache", "env_cache/mpl", "env_cache/numba", "env_cache/pycache"):
        (attempt / directory).mkdir(parents=True, exist_ok=True)
    command = pytest_command(root, profile, attempt)
    affinity, cpus = cpu_limit()
    try:
        raw_stat = Path("/proc/self/stat").read_text()
        startticks = int(raw_stat[raw_stat.rfind(")") + 2:].split()[19])
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except (OSError, ValueError, IndexError):
        startticks, boot_id = None, None
    packet = {"schema": "finite-local-cpu-test-profile-v1", "profile": profile,
              "registered_at": now.isoformat(), "deadline_at": (now + timedelta(seconds=timeout)).isoformat(),
              "writer": {"pid": os.getpid(), "startticks": startticks, "boot_id": boot_id,
                         "local_attempt_token": attempt.name},
              "inputs": inputs, "command": command, "tests": list(tests), "environment": environment,
              "limits": {"wall_seconds": timeout, "cpu_threads": 1, "cpu_affinity": cpus,
                         "gpu": 0, "models": 0, "training": 0},
              "shared_governance_write": False, "production_dispatch": False,
              "evidence_profile": "SERVER_BOUND_NOT_PORTABLE" if profile == "server" else "SYNTHETIC_PORTABLE"}
    packet["run_signature"] = signature(packet)
    save_json(attempt / "RUN_PACKET.json", packet)
    if view is not None:
        save_json(attempt / "READONLY_HANDOFF_BEFORE.json", view)
    try:
        process = subprocess.run(command, cwd=root, env=pytest_environment(root, attempt),
                                 capture_output=True, text=True, timeout=timeout, preexec_fn=affinity)
        stdout, stderr, returncode = process.stdout, process.stderr, process.returncode
        status = "PASSED" if returncode == 0 else "FAILED_TESTS"
        failure = None if returncode == 0 else "PYTEST_RETURN_NONZERO"
    except subprocess.TimeoutExpired as error:
        stdout = (error.stdout or b"").decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = (error.stderr or b"").decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        returncode, status, failure = None, "FAILED_TIMEOUT", "CPU_PROFILE_DEADLINE"
    for name, content in (("PYTEST_STDOUT.txt", stdout), ("PYTEST_STDERR.txt", stderr)):
        with (attempt / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
    counts = test_counts(stdout)
    if status == "PASSED" and (counts["passed"] == 0 or counts["failed"] or counts["errors"]):
        status, failure = "FAILED_TESTS", "CPU_PROFILE_NO_VALID_PASS_COUNT"
    try:
        for ref in inputs:
            CHECK.read_ref(root, ref, "CPU_PROFILE_INPUT_SHA_DRIFT")
        if profile == "server":
            CHECK.require_server_evidence(root)
            view_after = CHECK.inspect_project(root)
            save_json(attempt / "READONLY_HANDOFF_AFTER.json", view_after)
    except (CHECK.CheckError, OSError, KeyError, TypeError) as error:
        status, failure = "FAILED_INPUT_BINDING", getattr(error, "code", "CPU_PROFILE_INPUT_UNREADABLE")
    result = {"schema": "finite-local-cpu-test-result-v1", "status": status, "failure_code": failure,
              "profile": profile, "run_signature": packet["run_signature"], "returncode": returncode,
              "finished_at": datetime.now(timezone.utc).isoformat(), "test_counts": counts,
              "environment": environment, "models_run": 0, "production_dispatched": False,
              "profile_claim": packet["evidence_profile"], "attempt_directory": str(attempt),
              "artifacts": [CHECK.file_ref(attempt / name) for name in ("RUN_PACKET.json", "PYTEST_STDOUT.txt", "PYTEST_STDERR.txt")],
              "claim_limit": "CPU profile only; no algorithm, visual, training, geometry or production qualification."}
    save_json(attempt / "RESULT.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=("portable", "server"))
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args(argv)
    try:
        result = run_profile(args.project_root, args.profile, args.output_root, args.timeout_seconds)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "PASSED" else 2
    except (CHECK.CheckError, OSError, KeyError, TypeError) as error:
        code = getattr(error, "code", "CPU_PROFILE_INPUT_UNREADABLE")
        print(json.dumps({"status": "FAILED_PREFLIGHT", "failure_code": code,
                          "detail": str(error)[:500], "models_run": 0, "production_dispatched": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
