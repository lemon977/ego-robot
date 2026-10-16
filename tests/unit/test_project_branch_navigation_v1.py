"""Synthetic CPU regression for navigation, source labels and historical routing."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chaoyang.exact78_baseline.review_labels import motion_review_title
from chaoyang.governance import project_branch_navigation_v1 as nav
from chaoyang.governance.exact78_operation_gate import Exact78GateDenied, authorize_exact78_operation


@pytest.mark.parametrize("status,active", [
    ("RUNNING", True), ("CLAIMED", True), ("WAIT_GPU_RESOURCE", True),
    ("PENDING", False), ("READY", False), ("PASSED", False), ("CANCELLED", False),
])
def test_live_tasks_do_not_treat_pending_as_running(status, active):
    assert nav.live_tasks({"tasks": [{"task_id": "example", "status": status}]}) == (["example"] if active else [])


@pytest.mark.parametrize("source,name", [
    (None, "PICO"), ("PICO_OPENXR26_SAME_SESSION", "PICO"),
    ("PICO_CONTROLLER_PLUS_MANUS_INSTALLATION_PRIOR", "CONTROLLER + MANUS PRIOR"),
    ("HAWOR_RGB_PREDICTION", "HAWOR PREDICTION"), ("unexpected", "UNKNOWN MOTION SOURCE"),
])
def test_motion_title_uses_actual_source(source, name):
    session = SimpleNamespace(calibration={} if source is None else {"actual_motion_source": source})
    assert motion_review_title(session, 12, 171) == f"RAW + {name} 12/171"


def test_live_status_scopes_adoption_and_keeps_history_separate(tmp_path, monkeypatch):
    board = {"selected_visual_route_count": 2, "branches": {"exact78": {}, "manus": {}},
             "history_four_stream": {"products_adopted": "0/4", "status": "CLOSED"}}
    p = tmp_path / "docs/current/PROJECT_BRANCH_MAP.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(board))
    monkeypatch.setattr(nav, "build_branch_map", lambda root: board)
    state = {"governance_revision": 1, "next_task": None,
             "tasks": [{"task_id": "old_pending", "status": "PENDING"}]}
    got = nav.build_live_status(tmp_path, state, "fixed-time")
    assert got["active_tasks"] == []
    assert got["status"] == "QUIESCENT_NO_ACTIVE_TASKS"
    assert got["counts"] == {"selected_visual_routes": 2}
    assert got["historical_four_stream"]["products_adopted"] == "0/4"
    assert got["training_eligible"] is False


def test_map_drift_is_not_silently_reselected(tmp_path, monkeypatch):
    p = tmp_path / "docs/current/PROJECT_BRANCH_MAP.json"
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    monkeypatch.setattr(nav, "build_branch_map", lambda root: {"changed": True})
    with pytest.raises(RuntimeError, match="SOURCE_DRIFT"):
        nav.build_live_status(tmp_path, {}, "fixed-time")


def test_missing_selection_does_not_override_legacy_projection(tmp_path):
    assert nav.visual_selection_present(tmp_path) is False


def test_source_receipt_drift_fails_closed(tmp_path):
    pointer = tmp_path / "docs/current/EXACT78_VISUAL_BASELINE_V1.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(json.dumps({"baseline_id": "EXACT78_VISUAL_BASELINE_V1"}))
    receipt = tmp_path / "tasks/receipts/EXACT78_VISUAL_BASELINE_V1.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"status": "PASSED", "baseline_pointer": {"sha256": "wrong"}}))
    with pytest.raises(RuntimeError, match="RECEIPT_DRIFT"):
        nav.visual_selection_present(tmp_path)


@pytest.mark.parametrize("wrong_source", [False, True])
def test_source_labels_bind_actual_motion_and_immutable_media(tmp_path, wrong_source):
    base = tmp_path / "_run/current/exact78_takeover_v1_two_source_review_v3_20261008/attempts/attempt_0001/candidate/sessions"
    for identity, source, count in [
        ("play_cards_0915_001", "PICO_OPENXR26_SAME_SESSION", 150),
        ("get_potato_chips_0918_001", "PICO_CONTROLLER_PLUS_MANUS_INSTALLATION_PRIOR", 289),
    ]:
        folder = base / identity
        folder.mkdir(parents=True)
        outputs = {}
        for name in ("motion", "review", "final", "clean"):
            path = folder / name
            path.write_bytes(b"synthetic-test-fixture")
            outputs[name] = nav.artifact_ref(path)
        row = {"session": identity, "frame_count": count,
               "motion": {"calibration": {"actual_motion_source": "WRONG" if wrong_source else source}},
               "outputs": outputs}
        (folder / "SESSION.json").write_text(json.dumps(row))
    if wrong_source:
        with pytest.raises(RuntimeError, match="REVIEW_SOURCE_MISMATCH"):
            nav.build_source_labels(tmp_path)
    else:
        got = nav.build_source_labels(tmp_path)
        assert sum(row["frame_count"] for row in got["sessions"]) == 439
        assert all(row["existing_media_rewritten"] is False for row in got["sessions"])
        assert got["sessions"][1]["future_review_title_example"] == "RAW + CONTROLLER + MANUS PRIOR 0/289"


def test_pipeline_numerical_stages_unchanged_except_review_title_and_integrity_seals():
    root = Path(__file__).resolve().parents[2]
    relative = "src/chaoyang/exact78_baseline/pipeline.py"
    before = root / "_run/current/project_branch_navigation_convergence_v1_20261008/attempts/attempt_0001/before" / relative
    if not before.exists():
        pytest.skip("Point-in-time parent is server-only; pure source-label tests still apply")
    old = ast.parse(before.read_text())
    new = ast.parse((root / relative).read_text())
    new.body = [n for n in new.body if not (isinstance(n, ast.ImportFrom) and n.module == "chaoyang.exact78_baseline.review_labels")]

    class RestoreTitle(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(target, ast.Name) and target.id in {"stage_before", "stage_upstream"} for target in node.targets):
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "stage_reuse"):
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if "stage_before" in ast.unparse(node.test):
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "motion_review_title":
                return ast.parse('f"RAW + PICO {frame}/{count}"', mode="eval").body
            return node
    new = RestoreTitle().visit(new)
    stages = {"motion_stage", "gpu_stage", "render_stage"}
    old_functions = {n.name: ast.dump(n, include_attributes=False) for n in old.body
                     if isinstance(n, ast.FunctionDef) and n.name in stages}
    new_functions = {n.name: ast.dump(n, include_attributes=False) for n in new.body
                     if isinstance(n, ast.FunctionDef) and n.name in stages}
    assert old_functions == new_functions


def test_production_does_not_read_old_v2_route_when_v1_selected(tmp_path, monkeypatch):
    # Reuse the existing synthetic gated-writer fixture, not a real algorithm.
    import importlib.util
    path = Path(__file__).with_name("test_exact78_operation_gate.py")
    spec = importlib.util.spec_from_file_location("gate_fixture_for_navigation", path)
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    fixtures._fixture(tmp_path, mode="EXACT78_PRODUCTION")
    contract = tmp_path / "docs/governance/ALGORITHM_CONTRACT.json"
    contract.write_text(json.dumps({"special_status_contracts": {"exact78_visual_baseline_v1": {
        "execution_policy": {"enabled": True}}}}))
    # If old JSON were still consumed, this invalid file would trigger a parse error.
    (tmp_path / "docs/current/EXACT78_EXECUTION_ROUTE_V1.json").write_text("invalid old route")
    with pytest.raises(Exact78GateDenied, match="old V2/null routes cannot select"):
        fixtures._authorize(tmp_path)
