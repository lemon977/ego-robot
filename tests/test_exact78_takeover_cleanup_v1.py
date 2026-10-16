import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from chaoyang.exact78_baseline import audit_inventory
from chaoyang.ops import run_exact78_takeover_cleanup_v1 as cleanup


def _task(repo: Path, name: str) -> None:
    attempt = repo / "_run/current" / name / "attempts/attempt_0001"
    attempt.mkdir(parents=True)
    (attempt / "RESULT.json").write_text(json.dumps({"status": "PASSED"}))
    (attempt / "payload.bin").write_bytes(b"x" * 100)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for name in ("alpha_task_v1_0929", "beta_task_v1_0929", "gamma_task_v1_0929", "delta_task_v1_0929"):
        _task(repo, name)
    (repo / "_run/current/locks").mkdir()
    (repo / "docs/current").mkdir(parents=True)
    (repo / "docs/current/PAGE.md").write_text("Visual parent: alpha_task_v1_0929 (see receipt).\n")
    (repo / "tasks/receipts").mkdir(parents=True)
    (repo / "tasks/receipts/ALPHA_TASK_V1_0929.json").write_text(
        json.dumps({"inputs": ["/x/_run/current/beta_task_v1_0929/attempts/attempt_0001/RESULT.json"]}))
    (repo / "tasks/current").mkdir(parents=True)
    (repo / "tasks/current/INDEX.json").write_text(json.dumps({"task_packets": [
        {"task_id": "gamma_task_v1_0929"},
        {"task_id": "delta_task_v1_0929",
         "development_result": {"path": "/x/_run/current/delta_task_v1_0929/attempts/attempt_0001/RESULT.json"}}]}))
    (repo / "docs/governance").mkdir(parents=True)
    (repo / "docs/governance/ALGORITHM_CONTRACT.json").write_text("{}")
    (repo / "docs/governance/TASK_STATE.json").write_text(json.dumps({"tasks": [
        {"task_id": "gamma_task_v1_0929", "result": "_run/current/gamma_task_v1_0929/attempts/attempt_0001"}]}))
    return repo


def test_inventory_follows_bare_ids_and_receipts_and_keeps_index_refs(tmp_path):
    repo = _repo(tmp_path)
    output = tmp_path / "inventory"
    output.mkdir()
    audit_inventory.run(repo, output)
    rows = {row["task_dir"]: row for row in json.loads((output / "INVENTORY.json").read_text())["run_dirs"]}
    assert rows["alpha_task_v1_0929"]["category"] == "KEEP_PARENT"
    assert rows["beta_task_v1_0929"]["category"] == "KEEP_PARENT"
    assert rows["delta_task_v1_0929"]["category"] == "KEEP_GOVERNANCE_REF"
    assert rows["gamma_task_v1_0929"]["category"] == "LEDGER_ONLY_CLEANUP_CANDIDATE"
    assert rows["gamma_task_v1_0929"]["is_task_dir"] and rows["gamma_task_v1_0929"]["size"]["files"] == 2
    assert not rows["locks"]["is_task_dir"]


def _inventory(tmp_path: Path, repo: Path) -> tuple[Path, str]:
    output = tmp_path / "inventory"
    output.mkdir()
    audit_inventory.run(repo, output)
    path = output / "INVENTORY.json"
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_cleanup_moves_only_ledger_only_task_dirs_and_restores(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    inventory, sha = _inventory(tmp_path, repo)
    monkeypatch.setattr(cleanup, "get_paths", lambda: SimpleNamespace(repo_root=repo))
    monkeypatch.setattr(cleanup, "_governance_ok", lambda _repo: (True, ""))
    out = repo / "_run/current/exact78_takeover_c1_cleanup_v1/attempts/attempt_0001/candidate"
    assert cleanup.main(["--inventory", str(inventory), "--inventory-sha256", sha, "--output-root", str(out)]) == 0
    assert (out / "archive/gamma_task_v1_0929/attempts/attempt_0001/RESULT.json").is_file()
    assert not (repo / "_run/current/gamma_task_v1_0929").exists()
    assert (repo / "_run/current/locks").is_dir() and (repo / "_run/current/alpha_task_v1_0929").is_dir()
    moves = out / "MOVES.json"
    back = repo / "_run/current/exact78_takeover_c1_restore_v1/attempts/attempt_0001/candidate"
    assert cleanup.main(["--restore", str(moves), "--restore-sha256", hashlib.sha256(moves.read_bytes()).hexdigest(),
                         "--output-root", str(back)]) == 0
    assert (repo / "_run/current/gamma_task_v1_0929/attempts/attempt_0001/RESULT.json").is_file()


def test_cleanup_rolls_back_when_governance_fails(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    inventory, sha = _inventory(tmp_path, repo)
    monkeypatch.setattr(cleanup, "get_paths", lambda: SimpleNamespace(repo_root=repo))
    monkeypatch.setattr(cleanup, "_governance_ok", lambda _repo: (False, "artifact missing"))
    out = repo / "_run/current/exact78_takeover_c1_cleanup_v1/attempts/attempt_0001/candidate"
    assert cleanup.main(["--inventory", str(inventory), "--inventory-sha256", sha, "--output-root", str(out)]) == 1
    assert (repo / "_run/current/gamma_task_v1_0929").is_dir()
    assert json.loads((out / "RESULT.json").read_text())["status"] == "ROLLED_BACK_GOVERNANCE_FAILED"
