"""Takeover baseline v1: raw PICO sessions -> robot videos (chaoyang.exact78_baseline.pipeline).

Writes only under --output-root (stage outputs) and the project cache directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from chaoyang.governance.exact78_operation_gate import authorize_operation_entry
    authorize_operation_entry("run_exact78_takeover_baseline_v1", argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", default="", help="comma-separated session ids (run first with --all)")
    parser.add_argument("--all", action="store_true", help="every session under the two raw roots")
    parser.add_argument("--raw-input-manifest", type=Path,
                        help="explicit SHA-pinned native session inputs; authorized source adaptation only")
    parser.add_argument("--workers", type=int, default=8, help="CPU processes for motion and render")
    parser.add_argument("--reuse-from", action="append", default=[], type=Path,
                        help="earlier candidate directory whose finished stages are referenced, not recomputed")
    parser.add_argument("--reuse-stages", default="motion",
                        help="comma-separated stages (motion,gpu,render) that may be taken from --reuse-from; "
                             "only list stages whose code is unchanged since that candidate")
    parser.add_argument("--recompute", default="",
                        help="comma-separated sessions whose gpu and render stages are never taken from --reuse-from")
    parser.add_argument("--no-objects", action="store_true", help="skip the held-object SAM pass")
    parser.add_argument("--raft-iter", type=int, default=20)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)

    from chaoyang.paths import get_paths

    repo = get_paths().repo_root.resolve()
    output = args.output_root.resolve()
    if repo not in output.parents or output.exists():
        raise SystemExit("output root must be a fresh path inside the project")
    reuse = [path.resolve() for path in args.reuse_from]
    for path in reuse:
        if repo not in path.parents or not (path / "sessions").is_dir():
            raise SystemExit(f"reuse directory must be an earlier candidate inside the project: {path}")
    cache = repo / "_run/caches/takeover_baseline_v1"
    for name, sub in (("NUMBA_CACHE_DIR", "numba"), ("TORCH_HOME", "torch"), ("XDG_CACHE_HOME", "xdg"),
                      ("HF_HOME", "hf"), ("MPLCONFIGDIR", "mpl")):
        os.environ[name] = str(cache / sub)
        (cache / sub).mkdir(parents=True, exist_ok=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "2"

    from chaoyang.exact78_baseline import pico_source, pipeline

    reuse_stages = tuple(item for item in args.reuse_stages.split(",") if item)
    if not set(reuse_stages) <= set(pipeline.STAGE_FILES):
        raise SystemExit(f"unknown reuse stage in {reuse_stages}")
    sessions = [item for item in args.sessions.split(",") if item]
    if args.all:
        sessions += [item for item in pico_source.list_sessions() if item not in sessions]
    if not sessions or len(set(sessions)) != len(sessions):
        raise SystemExit("non-empty, duplicate-free session list required")
    if args.raw_input_manifest:
        if args.all or reuse:
            raise SystemExit("native input review forbids --all and historical stage reuse")
        raw_manifest = json.loads(args.raw_input_manifest.read_text())
        if sessions != [row["session_id"] for row in raw_manifest["sessions"]]:
            raise SystemExit("native input manifest/session identity mismatch")
        from chaoyang.exact78_baseline import native_input
        output.mkdir(parents=True)
        # Do not initialize video/BLAS thread pools in the process that the
        # unchanged V1 pipeline later forks. Source preparation is an isolated
        # CPU-only, packet-fenced descendant; all its arrays are validated.
        import subprocess
        import sys
        subprocess.run([sys.executable, "-B", "-m", "chaoyang.exact78_baseline.native_input",
                        "--manifest", str(args.raw_input_manifest.resolve()), "--output", str(output)],
                       check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        prepared = output / "PREPARED_INPUTS.json"
        os.environ[native_input.MANIFEST_ENV] = str(prepared)
    else:
        os.environ.pop("CHAOYANG_EXACT78_PREPARED_INPUTS", None)
    for session in sessions:
        pico_source.find_session(session)
    recompute = tuple(item for item in args.recompute.split(",") if item)
    if not set(recompute) <= set(sessions):
        raise SystemExit("--recompute sessions must be in the session list")
    if not args.raw_input_manifest:
        output.mkdir(parents=True)
    import secrets
    import signal
    import sys

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    from chaoyang.pipeline.causal_modal_mask_gpu_adapter_v71 import FileGpuLease

    task_id = output.relative_to(repo / "_run/current").parts[0]
    lease = FileGpuLease(lease_path=repo / "_run/current/GPU_LEASE.json",
                         lock_path=repo / "_run/current/GPU_LEASE.lock", task_id=task_id,
                         attempt_id="attempt_0001", gpu_id=0, executor_epoch=1,
                         fencing_token=secrets.token_hex(16), min_free_mib=40960)
    with lease:
        summary = pipeline.run(repo, output, sessions, workers=args.workers, reuse=reuse,
                               objects=not args.no_objects,
                               propainter_settings={"raft_iter": args.raft_iter}, reuse_stages=reuse_stages,
                               recompute=recompute)
    ok = summary["counts"].get("OK", 0)
    result = {"schema_version": "exact78-takeover-baseline-stage-v1",
              "status": "COMPLETE" if ok == len(sessions) else ("PARTIAL" if ok else "FAILED"),
              "sessions_requested": len(sessions), "sessions_ok": ok, "frames_ok": summary["frames_ok"],
              "counts": summary["counts"], "elapsed_s": summary["elapsed_s"],
              "index": str(output / "INDEX.json"), "index_page": str(output / "index.html"),
              "reuse_from": [str(path) for path in reuse], "reuse_stages": list(reuse_stages),
              "recomputed": list(recompute),
              "claim_limit": "Visual baseline candidates for human review; no physical-occlusion, control, "
                             "training or production authority.",
              "files_written_outside_output_root": "project cache only (_run/caches/takeover_baseline_v1)"}
    (output / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "sessions_ok", "frames_ok", "elapsed_s")}))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
