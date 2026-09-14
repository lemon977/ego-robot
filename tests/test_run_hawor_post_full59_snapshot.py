from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run_hawor_post_full59_snapshot.py"
CONTRACT = ROOT / "systems/hawor/post_full59_snapshot_execution_contract.json"


def load_tool():
    spec = importlib.util.spec_from_file_location("hawor_post_full59_snapshot", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checked_in_contract_and_all_bound_argv_are_valid(monkeypatch):
    tool = load_tool()
    monkeypatch.chdir(ROOT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    blockers = tool.validate_static_contract(contract)
    authority = json.loads(
        (ROOT / "systems/hawor/environment_authority.json").read_text(encoding="utf-8")
    )
    authority_pin = contract["static_authorities"]["environment_authority"][
        "sha256"
    ]
    if tool.sha256_file(ROOT / "systems/hawor/environment_authority.json") == authority_pin:
        assert blockers == []
    else:
        # The contract is single-use.  After successful promotion, its frozen
        # pre-promotion authority pin must be retained as transition evidence;
        # rerunning the destructive create path must stay fail-closed.
        assert authority["status"] == "PASS_LOCAL_ENVIRONMENT_PROMOTED"
        assert authority["promotion_evidence"]["pre_promotion_authority_sha256"] == authority_pin
        assert len(blockers) == 1
        assert blockers[0].startswith("static_authorities.environment_authority SHA-256 drift:")
    for record in contract["commands"].values():
        assert tool.argv_sha256(record["argv"]) == record["argv_sha256"]


def test_full59_requires_complete_exact_59_unique_identity_closure():
    tool = load_tool()
    rows = [
        {"task_id": "chips" if index < 16 else "poker", "session_id": f"session-{index:02d}"}
        for index in range(59)
    ]
    plan = {"rows": rows}
    checkpoint = {
        "status": "COMPLETE",
        "plan_sha256": "plan-sha",
        "completed": copy.deepcopy(rows),
        "failed": [],
        "hold_budget": [],
        "remaining": [],
    }
    raw = json.dumps(checkpoint, sort_keys=True).encode()
    blockers, evidence = tool.validate_full59(plan, [raw, raw], "plan-sha")
    assert blockers == []
    assert evidence["completed_unique_pairs"] == 59

    checkpoint["status"] = "COMPLETE_WITH_FAILURES"
    checkpoint["completed"][-1] = copy.deepcopy(checkpoint["completed"][0])
    changed = json.dumps(checkpoint, sort_keys=True).encode()
    blockers, _ = tool.validate_full59(plan, [changed, changed], "plan-sha")
    assert any("not COMPLETE" in item for item in blockers)
    assert any("59 unique" in item for item in blockers)
    assert any("exactly cover" in item for item in blockers)


def test_validate_only_never_enters_execute(monkeypatch, capsys):
    tool = load_tool()
    monkeypatch.setattr(
        tool,
        "live_readiness",
        lambda _contract, query_gpu=True: {
            "status": "BLOCKED",
            "ready": False,
            "read_only": True,
            "blockers": ["full59 is still running"],
        },
    )
    monkeypatch.setattr(
        tool,
        "execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("execute called")),
    )
    assert tool.main(["--validate-only"]) == 2
    assert json.loads(capsys.readouterr().out)["read_only"] is True


def test_execute_requires_both_exact_operator_confirmations(monkeypatch):
    tool = load_tool()
    monkeypatch.setattr(
        tool,
        "execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("execute called")),
    )
    assert tool.main(["--execute"]) == 2
    assert tool.main(
        [
            "--execute",
            "--confirm-contract-sha256",
            tool.sha256_file(CONTRACT),
            "--confirm-post-full59-state",
            "WRONG",
        ]
    ) == 2
