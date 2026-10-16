#!/usr/bin/env python
"""Takeover autopilot: run the queued takeover steps on this host without an operator.

Every cycle it re-reads AUTOPILOT_PLAN.json (steps may be appended or put on hold while it runs),
keeps per-step progress in AUTOPILOT_STATE.json and rewrites AUTOPILOT_STATUS_ZH.md, all in
docs/plans/EXACT78_TAKEOVER_20261004. Tasks are launched only through launch_detached.sh, so the
executor still enforces governance, one active task and code pinning. It never deletes; it writes
only inside the project (its own files are in _run/caches/takeover_autopilot).

Step types:
  run_task   launch specs/<task>.json unless already launched, wait for the seal. kind=full adds a
             stall check (worker killed when PROGRESS.json is unchanged for stall_minutes) and, when
             sessions were left unfinished, a resume task <base>r<n> that reuses this run's finished
             sessions and recomputes the rest (at most max_resumes).
  post_full  summary JSON, 40_VIDEO_INDEX_ZH.md, evidence entry, STATE / PROGRESS update.
  cleanup    cleanup spec from a sealed inventory, guard, review window, run, 30_CLEANUP_ZH.md.
  apply_code install files staged in _run/caches/takeover_autopilot/staged (sha256 pinned in the step) once
             no task is active, run the unit tests (undo on failure), commit, --publish-route.
  commit     snapshot commit onto takeover-20261004.

Start:  bash scripts/takeover/autopilot_start.sh   (no-op while it runs; restart after a host restart)
Check:  /usr/local/bin/python -B scripts/takeover/autopilot.py --dry-run --once
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path("/path/to/chaoyang")
PLANS = ROOT / "docs/plans/EXACT78_TAKEOVER_20261004"
SPECS = PLANS / "specs"
PLAN = PLANS / "AUTOPILOT_PLAN.json"
STATE = PLANS / "AUTOPILOT_STATE.json"
STATUS = PLANS / "AUTOPILOT_STATUS_ZH.md"
WORK = ROOT / "_run/caches/takeover_autopilot"
LAUNCH_LOGS = ROOT / "_run/caches/takeover_launch"
PY = "/usr/local/bin/python"
CYCLE_SECONDS = 60
COMMIT_EVERY_SECONDS = 1800
DEAD_CHECKS_BEFORE_RECOVER = 3
PROTECTED = re.compile(r"robot100|0916|sensor0916|egosteer|exact78_takeover_", re.I)
TESTS = ["tests/test_exact78_takeover_baseline_v1.py", "tests/test_exact78_takeover_cleanup_v1.py"]

sys.path.insert(0, str(ROOT / "src"))
from chaoyang.governance.common import process_identity  # noqa: E402

DRY = False


class StepError(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def hm() -> str:
    return datetime.now().astimezone().strftime("%H:%M")


def log(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with (WORK / "autopilot.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def load(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def save(path: Path, value) -> None:
    if DRY:
        return
    tmp = path.with_name(path.name + ".autopilot.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    if DRY:
        return
    tmp = path.with_name(path.name + ".autopilot.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env() -> dict:
    return {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(WORK / "tmp"), "XDG_CACHE_HOME": str(WORK / "cache")}


def sh(args: list[str], timeout: int = 900, check: bool = True) -> str:
    proc = subprocess.run(args, cwd=ROOT, env=env(), capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise StepError(f"{' '.join(args)[:200]} rc={proc.returncode}: {(proc.stdout + proc.stderr)[-600:]}")
    return proc.stdout


def event(state: dict, text: str) -> None:
    log(text)
    state.setdefault("events", []).append(f"{now()} {text}")
    state["events"] = state["events"][-60:]


def progress_line(text: str) -> None:
    if DRY:
        return
    with (PLANS / "PROGRESS_ZH.md").open("a", encoding="utf-8") as stream:
        stream.write(f"- {hm()} [autopilot] {text}\n")


def commit(message: str, state: dict) -> None:
    if DRY:
        return
    out = sh(["bash", "scripts/takeover/commit_snapshot.sh", message], timeout=600).strip()
    state["last_commit_at"] = time.time()
    if out.startswith("COMMITTED"):
        state["last_commit"] = out.split()[1]
        log(out)


def attempt_dir(task: str) -> Path:
    return ROOT / "_run/current" / task / "attempts/attempt_0001"


def seal_of(task: str) -> dict | None:
    path = attempt_dir(task) / "RESULT.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return {"status": value["status"], "failure_code": value.get("failure_code"), "sha256": sha256(path),
            "path": str(path.relative_to(ROOT))}


def writer_alive(task: str) -> bool | None:
    writer = load(attempt_dir(task) / "WRITER.json")
    if writer is None:
        return None
    try:
        identity = process_identity(int(writer["pid"]))
    except (OSError, KeyError, ValueError):
        return False
    return bool(identity.get("alive")) and identity.get("start_ticks") == writer.get("proc_start_ticks")


def children(pid: int) -> list[int]:
    try:
        return [int(x) for x in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except OSError:
        return []


def launch_result(task: str) -> dict:
    """Last JSON line the executor printed into the launch log (status, governance_revision)."""
    path = LAUNCH_LOGS / f"{task}.log"
    if not path.is_file():
        return {}
    for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {}


def progress_of(task: str) -> dict | None:
    return load(attempt_dir(task) / "candidate/PROGRESS.json")


def spec_args(spec: dict, flag: str) -> list[str]:
    args = spec["args"]
    return [args[i + 1] for i, x in enumerate(args) if x == flag]


def spec_sessions(spec: dict) -> list[str]:
    """Run order as the baseline entry builds it: --sessions first, then (with --all) every other raw session."""
    sessions = [s for s in ",".join(spec_args(spec, "--sessions")).split(",") if s]
    if "--all" in spec["args"]:
        from chaoyang.exact78_baseline import pico_source
        sessions += [s for s in pico_source.list_sessions() if s not in sessions]
    return sessions


# --------------------------------------------------------------------------- launching

def release_stale_lease(state: dict) -> None:
    """The GPU lease never reclaims itself: release it when its takeover task is sealed and the holder is gone."""
    with (ROOT / "_run/current/GPU_LEASE.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        lease_path = ROOT / "_run/current/GPU_LEASE.json"
        current = load(lease_path)
        if (not current or current.get("status") != "ACQUIRED"
                or not str(current.get("task_id", "")).startswith("exact78_takeover_")
                or seal_of(current["task_id"]) is None):
            return
        pid = int(current["pid"])
        identity = process_identity(pid) if Path(f"/proc/{pid}").exists() else {"alive": False}
        if identity.get("alive") and identity.get("start_ticks") == current.get("process_startticks"):
            return
        if DRY:
            log(f"DRY would release the stale GPU lease of {current['task_id']}")
            return
        current.update(status="RELEASED", released_at=now(),
                       release_reason="OWNER_GONE_TASK_SEALED_RELEASED_BY_TAKEOVER_AUTOPILOT")
        tmp = lease_path.with_name(lease_path.name + ".tmp")
        tmp.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, lease_path)
    event(state, f"released the stale GPU lease of {current['task_id']} (holder pid {pid} gone, task sealed)")


def check_only(spec_path: Path) -> str:
    out = sh([PY, "-B", "-m", "chaoyang.governance.execute_exact78_takeover_v1", "--check-only", str(spec_path)],
             timeout=900)
    return json.loads(out.strip().splitlines()[-1])["status"]


def guard_full_reuse(spec: dict) -> int:
    """Every session outside --recompute must have a finished render in the first reuse root holding one."""
    if "render" not in ",".join(spec_args(spec, "--reuse-stages")).split(","):
        return 0
    sessions = spec_sessions(spec)
    recompute = set(",".join(spec_args(spec, "--recompute")).split(",")) - {""}
    roots = [ROOT / p for p in spec_args(spec, "--reuse-from")]
    if not recompute <= set(sessions):
        raise StepError(f"recompute not a subset of sessions: {sorted(recompute - set(sessions))[:5]}")
    ok_by_root = {}
    for root in roots:
        progress = load(root / "PROGRESS.json") or {}
        ok_by_root[root] = {r["session"] for r in progress.get("sessions", []) if r["status"] == "OK"}
    bad = []
    for sid in sessions:
        if sid in recompute:
            continue
        holder = next((r for r in roots if (r / "sessions" / sid / "SESSION.json").is_file()), None)
        if holder is None or sid not in ok_by_root[holder]:
            bad.append(sid)
    if bad:
        raise StepError(f"sessions outside --recompute without an OK render in the reuse roots: {bad[:8]}")
    return len(sessions) - len(recompute)


def launch(task: str, step: dict, state: dict) -> bool:
    """Launch specs/<task>.json; False while another task is active."""
    spec_path = SPECS / f"{task}.json"
    spec = load(spec_path)
    if spec is None:
        raise StepError(f"spec missing: {spec_path}")
    if spec["operation"] == "run_exact78_takeover_baseline_v1":
        guard_full_reuse(spec)
    if int(spec["budgets"].get("gpu_count", 0)):
        release_stale_lease(state)
    status = check_only(spec_path)
    if "ACTIVE_TASK_CONFLICT" in status or "ACTIVE_ROUTE_CONFLICT" in status:
        return False
    if "GOVERNANCE_NOT_PASS" in status and "code_closure" in status:
        if DRY:
            log(f"DRY would publish-route ({status[:160]})")
            return False
        event(state, f"{task}: governance needs the code closure republished; running --publish-route")
        sh([PY, "-B", "-m", "chaoyang.governance.execute_exact78_takeover_v1", "--publish-route"], timeout=1800)
        status = check_only(spec_path)
    if status != "DRY_RUN_READY":
        raise StepError(f"{task} not routable: {status[:400]}")
    if step.get("tests", spec["operation"] == "run_exact78_takeover_baseline_v1"):
        sh([PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS], timeout=900)
    if DRY:
        log(f"DRY would launch {task}")
        return False
    commit(f"takeover autopilot: before launching {task}", state)
    out = sh(["bash", "scripts/takeover/launch_detached.sh", str(spec_path.relative_to(ROOT))], timeout=120)
    event(state, f"launched {task}: {out.strip()}")
    progress_line(f"启动任务 `{task}`（提交 {state.get('last_commit', '?')[:7]}）。")
    return True


def resume_spec(task: str, number: int, state: dict) -> str | None:
    """Successor spec that keeps this run's OK sessions and recomputes its unfinished ones."""
    spec = load(SPECS / f"{task}.json")
    progress = progress_of(task) or {"sessions": []}
    ok = {r["session"] for r in progress["sessions"] if r["status"] == "OK"}
    sessions = spec_sessions(spec)
    recompute = [s for s in ",".join(spec_args(spec, "--recompute")).split(",") if s]
    unfinished = [s for s in sessions if s not in ok]
    if not unfinished:
        return None
    base = re.sub(r"r\d+$", "", task)
    successor = f"{base}r{number}"
    cand = str(attempt_dir(task).relative_to(ROOT) / "candidate")
    roots = [cand] + [r for r in spec_args(spec, "--reuse-from") if r != cand]
    args = ["--sessions", ",".join(unfinished + [s for s in sessions if s in ok])]
    flags = spec["args"]
    i = 0
    while i < len(flags):
        if flags[i] in ("--reuse-from", "--recompute", "--sessions", "--reuse-stages"):
            i += 2
            continue
        args.append(flags[i])
        i += 1
    args += ["--reuse-stages", "motion,gpu,render"]
    for root in roots:
        args += ["--reuse-from", root]
    # Unfinished sessions take only motion from the roots: a partial gpu / render from older code is never mixed in.
    args += ["--recompute", ",".join(unfinished)]
    seal = seal_of(task) or {}
    new = {**spec, "task_id": successor, "args": args,
           "inputs": [str(attempt_dir(task).relative_to(ROOT) / "RESULT.json")],
           "objective": (f"Autopilot resume of {task} ({seal.get('status')} / {seal.get('failure_code')}): "
                         f"its {len(ok)} OK sessions are reused as they are; its {len(unfinished)} unfinished sessions "
                         f"({len([s for s in unfinished if s in recompute])} of them on its --recompute list) are "
                         f"recomputed on the current code. Original objective: {spec['objective']}")}
    path = SPECS / f"{successor}.json"
    if path.exists():
        raise StepError(f"resume spec already exists: {path}")
    if DRY:
        log(f"DRY would write {path.name}: recompute={len(unfinished)} reuse={len(ok)}")
        return None
    path.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    event(state, f"wrote resume spec {successor}: recompute {len(unfinished)}, reuse {len(ok)}")
    return successor


# --------------------------------------------------------------------------- steps

def step_run_task(step: dict, st: dict, state: dict) -> str:
    task = st.setdefault("task", step["task"])
    st.setdefault("tasks", [task])
    if not (LAUNCH_LOGS / f"{task}.log").exists() and not attempt_dir(task).exists():
        if not launch(task, step, state):
            st["note"] = "waiting for the active task to finish"
            return "WAIT"
        st["launched_at"] = now()
        st["code_commit"] = state.get("last_commit")
        return "WAIT"
    seal = seal_of(task)
    if seal:
        st.setdefault("seals", {})[task] = {**seal, **{k: v for k, v in launch_result(task).items()
                                                        if k == "governance_revision"}}
        progress = progress_of(task)
        counts = (progress or {}).get("counts")
        event(state, f"{task} sealed {seal['status']} {seal.get('failure_code') or ''} counts={counts}")
        progress_line(f"`{task}` 封存：{seal['status']} {seal.get('failure_code') or ''}，"
                      f"RESULT `{seal['sha256'][:8]}…`，会话计数 {counts}。")
        st["final_task"] = task
        if seal["status"] == "PASSED":
            return "DONE"
        if step.get("kind") == "full" and len(st["tasks"]) <= int(step.get("max_resumes", 2)):
            successor = resume_spec(task, len(st["tasks"]), state)
            if successor:
                st["task"] = successor
                st["tasks"].append(successor)
                for key in ("dead_checks", "stall_term_at", "stall_workers"):
                    st.pop(key, None)
                return "WAIT"
        return "FAILED" if step.get("halt_on_fail", True) else "DONE_WITH_FAILURE"
    alive = writer_alive(task)
    if alive is None:
        return "WAIT"
    if not alive:
        st["dead_checks"] = st.get("dead_checks", 0) + 1
        if st["dead_checks"] >= DEAD_CHECKS_BEFORE_RECOVER and not DRY:
            event(state, f"{task}: writer gone without a seal; running --recover")
            sh([PY, "-B", "-m", "chaoyang.governance.execute_exact78_takeover_v1", "--recover", task], timeout=900)
        return "WAIT"
    st["dead_checks"] = 0
    if step.get("kind") == "full":
        path = attempt_dir(task) / "candidate/PROGRESS.json"
        limit = 60 * int(step.get("stall_minutes", 60))
        if path.is_file() and time.time() - path.stat().st_mtime > limit:
            # SIGTERM first; the worker's main thread can block on pool shutdown, so SIGKILL the group after 90 s.
            if "stall_term_at" not in st:
                workers = children(int(load(attempt_dir(task) / "WRITER.json")["pid"]))
                event(state, f"{task}: PROGRESS.json unchanged for {limit // 60} min; SIGTERM worker group {workers}")
                sig, st["stall_workers"], st["stall_term_at"] = signal.SIGTERM, workers, time.time()
            elif time.time() - st["stall_term_at"] > 90:
                sig = signal.SIGKILL
                event(state, f"{task}: worker group {st['stall_workers']} still running 90 s after SIGTERM; SIGKILL")
            else:
                return "WAIT"
            if not DRY:
                for pid in st["stall_workers"]:
                    try:
                        os.killpg(pid, sig)
                    except ProcessLookupError:
                        pass
    return "WAIT"


def step_post_full(step: dict, st: dict, state: dict) -> str:
    source = state["steps"][step["from_step"]]
    task = source["final_task"]
    seal = source["seals"][task]
    summaries = PLANS / "summaries"
    summary_path = summaries / f"{task}.json"
    text = sh([PY, "-B", "scripts/takeover/baseline_summary.py", task], timeout=1800)
    summary = json.loads(text)
    if DRY:
        log(f"DRY summary ok={summary['ok_sessions']} counts={summary['progress_counts']}")
        return "WAIT"
    summaries.mkdir(exist_ok=True)
    summary_path.write_text(text, encoding="utf-8")
    seal_path = WORK / f"seal_{task}.json"
    seal_path.write_text(json.dumps({"status": seal["status"], "sha256": seal["sha256"]}), encoding="utf-8")
    sh([PY, "-B", "scripts/takeover/video_index.py", str(summary_path), str(seal_path),
        str(PLANS / "40_VIDEO_INDEX_ZH.md"), str(PLANS / "visual_checks.json")], timeout=300)
    ledger = load(PLANS / "EVIDENCE_LEDGER.json")
    number = 1 + max(int(e["id"][1:]) for e in ledger["entries"])
    failed = summary.get("failed") or {}
    ledger["entries"].append({
        "id": f"E{number:04d}", "recorded_at": now(),
        "claim": (f"Full run {task} sealed {seal['status']}"
                  + (f" ({seal['failure_code']})" if seal.get("failure_code") else "")
                  + f": {summary['ok_sessions']} sessions OK ({summary['frames']} frames), {len(failed)} failed, "
                  f"run counts {summary['progress_counts']}; {len(summary.get('recomputed') or [])} recomputed on "
                  f"this code, {summary.get('folders_outside_candidate', 0)} referenced from earlier candidates; "
                  f"forearm re-prompt fired in {len(summary.get('forearm_retries') or {})} sessions"),
        "result_path": seal["path"], "sha256": seal["sha256"],
        "summary_path": str(summary_path.relative_to(ROOT)), "summary_sha256": sha256(summary_path),
        "code_commit": source.get("code_commit") or state.get("last_commit"),
        "governance_revision": seal.get("governance_revision"), "status": seal["status"],
        "verified_by_takeover": False, "recorded_by": "autopilot"})
    save(PLANS / "EVIDENCE_LEDGER.json", ledger)
    status = load(PLANS / "STATE.json")
    status["updated_at"] = now()
    status["latest_pointers"]["baseline_full"] = {
        "task_id": task, "status": seal["status"], "result_sha256": seal["sha256"],
        "evidence": f"E{number:04d}", "video_index": "docs/plans/EXACT78_TAKEOVER_20261004/40_VIDEO_INDEX_ZH.md",
        "summary": str(summary_path.relative_to(ROOT)),
        "index_page": str(attempt_dir(task).relative_to(ROOT) / "candidate/index.html")
        if summary.get("has_index") else None,
        "runs": source["tasks"]}
    status["active_jobs"] = [job for job in status.get("active_jobs", []) if job.get("task_id") not in source["tasks"]]
    status["steps"]["m9-full"] = (f"{seal['status']} {task} (autopilot {now()}): {summary['ok_sessions']}/234 OK, "
                                  f"{len(failed)} failed; video index 40_VIDEO_INDEX_ZH.md, evidence E{number:04d}")
    save(PLANS / "STATE.json", status)
    progress_line(f"全量 `{task}` 汇总：{summary['ok_sessions']} 条 OK、{len(failed)} 条失败，"
                  f"40_VIDEO_INDEX_ZH.md 已重写，证据 E{number:04d}。")
    commit(f"takeover autopilot: {task} sealed {seal['status']}; video index, evidence E{number:04d}", state)
    return "DONE"


def step_cleanup(step: dict, st: dict, state: dict) -> str:
    task = step["task"]
    spec_path = SPECS / f"{task}.json"
    if "planned_at" not in st:
        source = state["steps"][step["from_step"]]
        inv_task = source["final_task"]
        if source["seals"][inv_task]["status"] != "PASSED":
            raise StepError(f"inventory {inv_task} not PASSED")
        inventory = attempt_dir(inv_task) / "candidate/INVENTORY.json"
        rows = load(inventory)["run_dirs"]
        movable = [r for r in rows if r["category"] == "LEDGER_ONLY_CLEANUP_CANDIDATE" and r.get("is_task_dir")
                   and not PROTECTED.search(r["task_dir"]) and not r.get("live_writer_pids")]
        st["movable"] = [[r["task_dir"], (r.get("size") or {}).get("bytes", 0), r.get("latest_status")]
                         for r in movable]
        st["movable_bytes"] = sum(row[1] for row in st["movable"])
        if not movable:
            event(state, f"{inv_task}: nothing to move")
            return "DONE"
        spec = {
            "task_id": task, "operation": "run_exact78_takeover_cleanup_v1",
            "args": ["--inventory", str(inventory.relative_to(ROOT)), "--inventory-sha256", sha256(inventory)],
            "objective": ("Reversible cleanup of _run/current: move the task directories that the sealed inventory "
                          f"{inv_task} classifies LEDGER_ONLY_CLEANUP_CANDIDATE (referenced only by ledgers / receipts, "
                          "no live writer, re-checked against the packet index right before the move) into this task's "
                          "archive; governance is validated after the move and every move is undone if it fails. "
                          "Takeover run directories (exact78_takeover_*), other branches (0916 / Robot100 / egosteer) "
                          "and anything referenced by current navigation, authority files, the packet index or code "
                          "are kept. Nothing is deleted; --restore with MOVES.json moves everything back."),
            "phase": "EXACT78_TAKEOVER_C1_CLEANUP", "session": None,
            "read_set": ["AGENTS.md", "tasks/current/INDEX.json", "docs/governance"],
            "inputs": [str(inventory.relative_to(ROOT))],
            "budgets": {"workers_max": 1, "cpu_threads_total": 2, "gpu_count": 0, "model_calls": 0,
                        "render_calls": 0, "training_steps": 0, "wall_seconds_max": 3600, "session_attempts_max": 1},
            "required_outputs": ["RESULT.json", "MOVES.json"],
            "claim_limit": "Reversible move inside the project; no deletion, repair or promotion.",
            "stop_conditions": ["ANY_WRITE_OUTSIDE_PROJECT"]}
        if DRY:
            log(f"DRY cleanup would move {len(movable)} dirs ({st['movable_bytes'] / 1e9:.1f} GB)")
            return "WAIT"
        if not spec_path.exists():
            spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        st["planned_at"] = time.time()
        event(state, f"cleanup planned: {len(movable)} dirs, {st['movable_bytes'] / 1e9:.2f} GB; "
                     f"launch after a {step.get('review_minutes', 20)} min review window (set hold=true to stop)")
        return "WAIT"
    if step.get("hold"):
        st["note"] = "on hold (plan hold=true)"
        return "WAIT"
    if time.time() - st["planned_at"] < 60 * int(step.get("review_minutes", 20)):
        st["note"] = "review window"
        return "WAIT"
    outcome = step_run_task({"task": task, "kind": "plain"}, st, state)
    if outcome not in ("DONE", "FAILED", "DONE_WITH_FAILURE"):
        return outcome
    moves = load(attempt_dir(task) / "candidate/MOVES.json") or {}
    lines = ["# 接手清理 C1：`_run/current` 可逆移动（自动生成）", "",
             f"任务 `{task}`，封存 {st['seals'][task]['status']}，RESULT `{st['seals'][task]['sha256']}`。",
             f"清点来源：`{moves.get('inventory', {}).get('path')}`（sha256 `{moves.get('inventory', {}).get('sha256')}`）。",
             f"移动后治理校验：{'PASS' if moves.get('governance_pass_after_move') else '未通过，已全部移回'}；"
             f"移动 {len(moves.get('moves', []))} 个目录，共 {moves.get('bytes_moved', 0) / 1e9:.2f} GB；"
             f"跳过 {len(moves.get('skipped', []))} 个。",
             f"归档位置：`{attempt_dir(task).relative_to(ROOT)}/candidate/archive/`。没有删除任何文件。",
             "", "恢复（全部移回原处）：新建一个 cleanup 任务包，参数 `--restore <MOVES.json> --restore-sha256 <sha>`。",
             "", "| 目录 | 大小 (GB) | 最后状态 |", "|------|-----------|----------|"]
    lines += [f"| `{m['task_dir']}` | {((m.get('size') or {}).get('bytes', 0)) / 1e9:.2f} | {m.get('latest_status')} |"
              for m in moves.get("moves", [])]
    if moves.get("skipped"):
        lines += ["", "跳过：", ""] + [f"- `{s['task_dir']}`：{s['reason']}" for s in moves["skipped"]]
    write_text(PLANS / "30_CLEANUP_ZH.md", "\n".join(lines) + "\n")
    progress_line(f"清理 `{task}`：{st['seals'][task]['status']}，移动 {len(moves.get('moves', []))} 个目录 "
                  f"{moves.get('bytes_moved', 0) / 1e9:.2f} GB，见 30_CLEANUP_ZH.md。")
    commit(f"takeover autopilot: cleanup {task} {st['seals'][task]['status']}", state)
    return outcome


def step_commit(step: dict, st: dict, state: dict) -> str:
    commit(step["message"], state)
    return "DONE"


def no_active_task() -> bool:
    from chaoyang.governance.common import TASK_STATE_PATH
    tasks = load(TASK_STATE_PATH)
    return not tasks.get("next_task") and not any(
        row["status"] in ("RUNNING", "CLAIMED", "WAIT_GPU_RESOURCE") for row in tasks["tasks"])


def step_apply_code(step: dict, st: dict, state: dict) -> str:
    """Install staged files (sha256-pinned in the plan) once no task is active; tests must pass or all is undone."""
    if not no_active_task():
        st["note"] = "waiting until no task is active (the executor hashes package code at the seal)"
        return "WAIT"
    staged, backup = WORK / "staged", WORK / "backup" / step["id"]
    changed = []
    for rel, digest in step["files"].items():
        if sha256(staged / rel) != digest:
            raise StepError(f"staged {rel} does not match the pinned sha256")
        if not ((ROOT / rel).is_file() and sha256(ROOT / rel) == digest):
            changed.append(rel)
    if DRY:
        log(f"DRY would install {changed}")
        return "WAIT"
    for rel in changed:
        target = ROOT / rel
        if target.is_file():
            (backup / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup / rel)
        tmp = target.with_name(target.name + ".autopilot.tmp")
        shutil.copy2(staged / rel, tmp)
        os.replace(tmp, target)
    try:
        sh([PY, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS], timeout=900)
    except StepError:
        for rel in changed:
            if (backup / rel).is_file():
                shutil.copy2(backup / rel, ROOT / rel)
        event(state, f"{step['id']}: unit tests failed with the staged files; restored {changed}")
        raise
    commit(step["message"], state)
    route = sh([PY, "-B", "-m", "chaoyang.governance.execute_exact78_takeover_v1", "--publish-route"],
               timeout=1800, check=False).strip()
    st["publish_route"] = route[-400:]
    event(state, f"{step['id']}: installed {changed}; tests pass; publish-route {route[-200:]}")
    progress_line(f"安装暂存代码 {', '.join(f'`{rel}`' for rel in changed)}（单测通过，提交 "
                  f"{state.get('last_commit', '?')[:7]}，publish-route：{route[-120:]}）。")
    return "DONE"


HANDLERS = {"run_task": step_run_task, "post_full": step_post_full, "cleanup": step_cleanup,
            "commit": step_commit, "apply_code": step_apply_code}
TERMINAL = ("DONE", "FAILED", "DONE_WITH_FAILURE", "SKIPPED")


# --------------------------------------------------------------------------- status page

def task_line(task: str) -> str:
    progress = progress_of(task)
    lease = load(ROOT / "_run/current/GPU_LEASE.json") or {}
    alive = writer_alive(task)
    parts = [f"任务 `{task}`", f"写进程{'存活' if alive else ('未启动' if alive is None else '已退出')}"]
    if progress:
        counts = progress["counts"]
        parts.append(f"会话计数 {counts}，已运行 {progress['elapsed_s'] / 60:.0f} min，PROGRESS 更新于 {progress['updated_at']}")
        spec = load(SPECS / f"{task}.json") or {"args": []}
        recompute = [s for s in ",".join(spec_args(spec, "--recompute")).split(",") if s]
        if recompute:
            done = sum(1 for r in progress["sessions"] if r["session"] in recompute and r["status"] == "OK")
            parts.append(f"重算 {done}/{len(recompute)} 条完成")
            if done >= 2:
                eta = progress["elapsed_s"] / done * (len(recompute) - done)
                finish = datetime.fromtimestamp(time.time() + eta).astimezone().strftime("%m-%d %H:%M")
                parts.append(f"按当前速度约 {finish} 跑完")
    parts.append(f"GPU 租约 {lease.get('task_id')} / {lease.get('status')}")
    return "；".join(parts) + "。"


def write_status(plan: dict, state: dict, current: str | None) -> None:
    lines = ["# 接手自动驾驶状态（autopilot 每分钟重写，不要手改）", "",
             f"- 更新：{now()}；autopilot pid {os.getpid()}，启动于 {state.get('started_at')}。",
             "- 计划：`AUTOPILOT_PLAN.json`（可追加步骤或给步骤加 `\"hold\": true`）；状态：`AUTOPILOT_STATE.json`；"
             "日志：`_run/caches/takeover_autopilot/autopilot.log`。",
             f"- 最近一次快照提交：{state.get('last_commit', '—')}。", ""]
    if current:
        st = state["steps"].get(current, {})
        lines += ["## 当前步骤", "", f"`{current}`：{st.get('status')}" + (f"（{st['note']}）" if st.get("note") else "")
                  + (f"；最近错误：{st['error'][-300:]}" if st.get("error") else "")]
        if st.get("task"):
            lines += ["", task_line(st["task"])]
        lines.append("")
    elif plan["steps"]:
        lines += ["## 当前步骤", "", "计划里的步骤都已结束。", ""]
    lines += ["## 步骤", "", "| 步骤 | 类型 | 状态 | 任务 | 结束于 |", "|------|------|------|------|--------|"]
    for step in plan["steps"]:
        st = state["steps"].get(step["id"], {})
        tasks = " → ".join(f"`{t}`" for t in st.get("tasks", [])) or (f"`{step['task']}`" if step.get("task") else "")
        lines.append(f"| `{step['id']}` | {step['type']} | {st.get('status', 'PENDING')}"
                     f"{' (hold)' if step.get('hold') else ''} | {tasks} | {st.get('finished_at', '')} |")
    lines += ["", "## 最近事件", ""] + [f"- {e}" for e in reversed(state.get("events", [])[-25:])] + [""]
    if DRY:
        print("\n".join(lines))
    write_text(STATUS, "\n".join(lines))


# --------------------------------------------------------------------------- main loop

def cycle(state: dict) -> str | None:
    plan = load(PLAN)
    current = None
    for step in plan["steps"]:
        st = state["steps"].setdefault(step["id"], {"status": "PENDING"})
        if st["status"] in TERMINAL:
            if st["status"] == "FAILED" and step.get("halt_on_fail", True):
                current = step["id"]
                break
            continue
        current = step["id"]
        if step.get("hold") and step["type"] != "cleanup":
            st["note"] = "on hold (plan hold=true)"
            break
        if st["status"] == "PENDING":
            st.update(status="RUNNING", started_at=now())
            event(state, f"step {step['id']} started")
        try:
            outcome = HANDLERS[step["type"]](step, st, state)
            st.pop("error", None)
            st["errors"] = 0
        except Exception as exc:  # noqa: BLE001 - keep the loop alive, count and surface the error
            st["error"] = f"{type(exc).__name__}: {exc}"
            st["errors"] = st.get("errors", 0) + 1
            log(traceback.format_exc())
            outcome = "FAILED" if st["errors"] >= int(step.get("max_errors", 3)) else "WAIT"
            event(state, f"step {step['id']} error {st['errors']}: {st['error'][:300]}")
        if outcome in TERMINAL:
            st.update(status=outcome, finished_at=now())
            st.pop("note", None)
            event(state, f"step {step['id']} {outcome}")
            if outcome == "FAILED" and step.get("halt_on_fail", True):
                break
            continue
        break
    if not DRY:
        save(STATE, state)
    write_status(plan, state, current)
    return current


def main() -> int:
    print("LEGACY_AUTOPILOT_RETIRED: use docs/current/DUAL_BRANCH_BASELINE_V1_ZH.md and a new finite Task Packet", file=sys.stderr)
    return 64


def historical_main_not_an_entrypoint() -> int:
    global DRY
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="decide and print, change nothing")
    args = parser.parse_args()
    DRY = args.dry_run
    for name in ("tmp", "cache"):
        (WORK / name).mkdir(parents=True, exist_ok=True)
    lock = (WORK / "autopilot.lock").open("a")
    if not DRY:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another autopilot holds the lock", file=sys.stderr)
            return 1
        (WORK / "autopilot.pid").write_text(f"{os.getpid()}\n")
    state = load(STATE) or {"schema_version": "exact78-takeover-autopilot-state-v1", "steps": {}}
    state["started_at"] = now()
    if not DRY:
        event(state, f"autopilot started pid {os.getpid()}")
    while True:
        cycle(state)
        if args.once:
            return 0
        if time.time() - state.get("last_commit_at", 0) > COMMIT_EVERY_SECONDS:
            try:
                commit("takeover autopilot: periodic snapshot (docs, status)", state)
            except Exception as exc:  # noqa: BLE001
                event(state, f"periodic commit failed: {exc}")
        time.sleep(CYCLE_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
