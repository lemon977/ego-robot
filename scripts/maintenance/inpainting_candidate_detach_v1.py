"""Start one finite existing executor independently of a terminal lifetime.

No queue, lease or publisher here: the child executor still registers and fences
its own exact Task Packet and owns its existing finite deadline.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from chaoyang.governance.common import REPO_ROOT, artifact_ref, now_iso
from chaoyang.inpainting_candidate.resources import proc_start_ticks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", choices=("model", "model-recovery", "cpu", "geometry", "composite"), required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--launch-id", required=True)
    args = parser.parse_args(argv)
    if not args.launch_id.startswith("inpainting_") or not args.launch_id.replace("_", "").isalnum():
        raise ValueError("FINITE_PROJECT_LAUNCH_ID_REQUIRED")
    relative = {"model": "scripts/maintenance/inpainting_candidate_model_task_v1.py",
                "model-recovery": "scripts/maintenance/inpainting_candidate_model_recover_v1.py",
                "geometry": "scripts/maintenance/inpainting_candidate_geometry_task_v1.py",
                "composite": "scripts/maintenance/inpainting_candidate_composite_task_v1.py",
                "cpu": "scripts/maintenance/inpainting_optimization_task_v1.py"}[args.executor]
    command = [sys.executable, "-B", str(REPO_ROOT / relative)]
    inputs = [artifact_ref(REPO_ROOT / relative)]
    if args.executor in {"model", "cpu", "geometry", "composite"}:
        spec = args.spec.resolve(strict=True) if args.spec is not None else None
        if spec is None or not spec.is_relative_to(REPO_ROOT / "inpainting_optimization"):
            raise ValueError("FROZEN_PROJECT_FINITE_SPEC_REQUIRED")
        command += ["--spec", str(spec)]
        inputs.append(artifact_ref(spec))
    elif args.spec is not None:
        raise ValueError("RECOVERY_HAS_FIXED_PARENT_NO_ARBITRARY_SPEC")
    output = REPO_ROOT / "_run" / args.launch_id
    output.mkdir(exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO_ROOT / "src"),
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    with (output / "EXECUTOR.log").open("xb") as log:
        child = subprocess.Popen(command, cwd=REPO_ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    value = {"schema": "inpainting-finite-detached-launch-v1", "pid": child.pid,
        "proc_start_ticks": proc_start_ticks(child.pid), "command": command,
        "executor_inputs": inputs, "started_at": now_iso(),
        "claim_limit": "Launch receipt only; executor must register its own finite packet before any algorithm."}
    with (output / "LAUNCH.json").open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
