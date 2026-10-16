"""Meaningful CPU cases for explicit, byte-bound, non-mutating resume plans."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chaoyang.inpainting_candidate import automation as auto, v1_resume_plan as resume
from chaoyang.inpainting_candidate.contracts import STAGES, file_ref
from tests.test_inpainting_v1_automation_flow import (
    authority, packet, project, request, save, synthetic_builders,
)
from tests.unit.test_inpainting_repair_rounds_entry import specs


@pytest.fixture
def case(project):
    req = request(project, "050")
    counter = 0

    def parent(*, failed=False, reuse=None, different_clean=False):
        nonlocal counter
        counter += 1
        value = packet(project, req, "parent_%02d" % counter)
        value.update(mode="Automation", task_id="synthetic_parent_%02d" % counter,
                     inputs={"request": file_ref(value["request_path"])})
        runtime = project.root / ("outer_wrapper_%02d.py" % counter)
        runtime.write_text("# operation wrapper; no image algorithm\n")
        value["runtime_code"] = [file_ref(runtime)]
        value["reuse_stages"] = reuse or {}
        builders = synthetic_builders([])
        if different_clean:
            original = builders["Clean"]

            def clean(**kwargs):
                original(**kwargs)
                root = Path(kwargs["output_dir"])
                (root / "synthetic.txt").write_text("different valid Clean bytes")
                receipt_path = root / "RECEIPT.json"
                receipt = json.loads(receipt_path.read_text())
                receipt["outputs"] = [file_ref(root / "synthetic.txt")]
                save(receipt_path, receipt)

            builders["Clean"] = clean
        if failed:
            def timeout(**kwargs):
                root = Path(kwargs["output_dir"])
                root.mkdir()
                (root / "unfinished.txt").write_text("partial stays private")
                raise RuntimeError("SYNTHETIC_TIMEOUT")

            builders["Clean"] = timeout
            with pytest.raises(RuntimeError, match="SYNTHETIC_TIMEOUT"):
                auto.execute_automation(value, authorization=authority(value), builders=builders)
            refs = {stage: file_ref(Path(value["output_dir"]) / "sessions/050" / stage / "RECEIPT.json")
                    for stage in ("Source", "Motion", "Mask")}
            model_ref = None
        else:
            model = auto.execute_automation(value, authorization=authority(value), builders=builders)
            refs = model["sessions"][0]["stage_receipts"]
            model_ref = save(Path(value["output_dir"]) / "MODEL_RESULT.json", model)
        packet_ref = save(project.root / "tasks" / value["task_id"] / "TASK_PACKET.json", value)
        terminal = {"task_id": value["task_id"], "run_signature": value["run_signature"],
                    "status": "FAILED_RUNTIME_FINAL" if failed else "PASSED"}
        if model_ref:
            terminal.update(model_result=model_ref, model_result_consumable=True)
        result_ref = save(Path(value["output_dir"]) / "RESULT.json", terminal)
        candidate = {"packet": packet_ref, "result": result_ref}
        if failed:
            candidate["stage_receipts"] = {"050": refs}
        return SimpleNamespace(packet=value, candidate=candidate, refs=refs, runtime=runtime)

    def plan(parents=None, config=None, root=None):
        # Pure target signature derivation consumes the actual frozen fixture
        # closure, not signatures hand-adjusted to make a candidate pass.
        code = {stage: [file_ref(project.root / "code" / (stage + ".py"))] for stage in STAGES}
        return resume.plan_resume(project_root=root or project.root, request=req, component_code=code,
            component_config=config, candidates=parents)

    return SimpleNamespace(project=project, request=req, parent=parent, plan=plan)


def test_all_six_stages_reuse_in_new_fenced_task_with_zero_builders(case):
    parent = case.parent()
    plan = case.plan([parent.candidate])
    assert list(plan["reuse_stages"]["050"]) == list(STAGES)
    next_packet = packet(case.project, case.request, "new_finite_task")
    next_packet["reuse_stages"] = plan["reuse_stages"]
    calls = []
    result = auto.execute_automation(next_packet, authorization=authority(next_packet), builders=synthetic_builders(calls))
    assert calls == [] and result["sessions"][0]["reused_stages"] == list(STAGES)
    assert result["model_calls"] == result["render_calls"] == 0
    assert plan["session_execution_status"] == "NOT_EXECUTED_BY_PLANNER"
    assert not plan["candidate_adopted"] and plan["human_review"] == "PENDING"


def test_failed_parent_sealed_prefix_continues_in_new_directory(case):
    parent = case.parent(failed=True)
    unfinished = Path(parent.packet["output_dir"]) / "sessions/050/Clean/unfinished.txt"
    old_bytes = unfinished.read_bytes()
    plan = case.plan([parent.candidate])
    assert list(plan["reuse_stages"]["050"]) == ["Source", "Motion", "Mask"]
    child = packet(case.project, case.request, "after_timeout")
    child["reuse_stages"] = plan["reuse_stages"]
    calls = []
    result = auto.execute_automation(child, authorization=authority(child), builders=synthetic_builders(calls))
    assert calls == [("050", stage) for stage in ("Clean", "Render", "Composite")]
    assert result["status"] == "SEALED_COMPLETE" and unfinished.read_bytes() == old_bytes
    assert json.loads(Path(parent.candidate["result"]["path"]).read_text())["status"] == "FAILED_RUNTIME_FINAL"
    assert plan["parents"][0]["parent_status"] == "FAILED_RUNTIME_FINAL"
    assert plan["parent_outcomes_changed"] is False


def test_partial_receipt_is_rejected_even_from_terminal_parent(case):
    parent = case.parent(failed=True)
    root = Path(parent.packet["output_dir"]) / "sessions/050/Clean"
    ref = save(root / "RECEIPT.json", {"schema": "inpainting-candidate-component-v1", "status": "PARTIAL",
        "stage": "Clean", "session_id": "050", "native_frames": 3,
        "component_signature": parent.packet["component_signatures"]["050"]["Clean"], "outputs": []})
    parent.candidate["stage_receipts"]["050"]["Clean"] = ref
    with pytest.raises(resume.ResumePlanError, match="PARTIAL_NOT_CONSUMABLE"):
        case.plan([parent.candidate])


def _replace_receipt(parent, stage, **changes):
    path = Path(parent.refs[stage]["path"])
    body = json.loads(path.read_text())
    body.update(changes)
    new_ref = save(path, body)
    model_path = Path(parent.candidate["result"]["path"]).parent / "MODEL_RESULT.json"
    model = json.loads(model_path.read_text())
    model["sessions"][0]["stage_receipts"][stage] = new_ref
    model_ref = save(model_path, model)
    result_path = Path(parent.candidate["result"]["path"])
    terminal = json.loads(result_path.read_text())
    terminal["model_result"] = model_ref
    parent.candidate["result"] = save(result_path, terminal)


@pytest.mark.parametrize("change,match", [
    ({"session_id": "007"}, "NATIVE_IDENTITY_DRIFT"),
    ({"native_frames": 4}, "NATIVE_IDENTITY_DRIFT"),
    ({"component_signature": "d" * 64}, "COMPONENT_SIGNATURE_MISMATCH"),
])
def test_stage_session_native_and_signature_drift_rejected(case, change, match):
    parent = case.parent()
    _replace_receipt(parent, "Mask", **change)
    with pytest.raises(resume.ResumePlanError, match=match):
        case.plan([parent.candidate])


def test_signed_result_sha_drift_rejected(case):
    parent = case.parent()
    Path(parent.candidate["result"]["path"]).write_text("{}")
    with pytest.raises(resume.ResumePlanError, match="SHA_DRIFT"):
        case.plan([parent.candidate])


def test_output_sha_drift_rejected(case):
    parent = case.parent()
    receipt = json.loads(Path(parent.refs["Clean"]["path"]).read_text())
    Path(receipt["outputs"][0]["path"]).write_text("changed after sealing")
    with pytest.raises(resume.ResumePlanError, match="SHA_DRIFT"):
        case.plan([parent.candidate])


def test_parent_component_signature_not_trusted_without_derivation(case):
    parent = case.parent()
    parent.packet["component_signatures"]["050"]["Mask"] = "e" * 64
    parent.candidate["packet"] = save(Path(parent.candidate["packet"]["path"]), parent.packet)
    with pytest.raises(resume.ResumePlanError, match="PARENT_COMPONENT_SIGNATURE_DRIFT"):
        case.plan([parent.candidate])


@pytest.mark.parametrize("field,value,match", [
    ("task_id", "another_parent", "TASK_IDENTITY_DRIFT"),
    ("run_signature", "b" * 64, "TASK_IDENTITY_DRIFT"),
    ("status", "RUNNING", "NOT_TERMINAL"),
])
def test_parent_task_binding_and_terminal_status_required(case, field, value, match):
    parent = case.parent()
    path = Path(parent.candidate["result"]["path"])
    body = json.loads(path.read_text())
    body[field] = value
    parent.candidate["result"] = save(path, body)
    with pytest.raises(resume.ResumePlanError, match=match):
        case.plan([parent.candidate])


def test_stage_cannot_be_borrowed_from_undeclared_parent(case):
    one, two = case.parent(), case.parent()
    candidate = deepcopy(one.candidate)
    # This preserves the same algorithm signature but violates parent output
    # ownership and is absent from its signed reuse_stages.
    model_path = Path(candidate["result"]["path"]).parent / "MODEL_RESULT.json"
    model = json.loads(model_path.read_text())
    model["sessions"][0]["stage_receipts"]["Clean"] = two.refs["Clean"]
    model_ref = save(model_path, model)
    terminal = json.loads(Path(candidate["result"]["path"]).read_text())
    terminal["model_result"] = model_ref
    candidate["result"] = save(Path(candidate["result"]["path"]), terminal)
    with pytest.raises(resume.ResumePlanError, match="NOT_BOUND_TO_PARENT"):
        case.plan([candidate])


def _failed_catalog(parent, stages):
    body = json.loads(Path(parent.candidate["result"]["path"]).read_text())
    body["status"] = "FAILED_RUNTIME_FINAL"
    return {"packet": parent.candidate["packet"],
            "result": save(Path(parent.candidate["result"]["path"]), body),
            "stage_receipts": {"050": {stage: parent.refs[stage] for stage in stages}}}


def test_same_component_signature_cannot_mix_different_clean_byte_ancestors(case):
    first = case.parent()
    reusable = {stage: first.refs[stage] for stage in ("Source", "Motion", "Mask", "Render")}
    second = case.parent(reuse={"050": reusable}, different_clean=True)
    assert first.packet["component_signatures"] == second.packet["component_signatures"]
    assert first.refs["Clean"]["sha256"] != second.refs["Clean"]["sha256"]
    candidates = [_failed_catalog(second, ("Source", "Motion", "Mask", "Clean", "Render")),
                  _failed_catalog(first, ("Composite",))]
    plan = case.plan(candidates)
    assert plan["reuse_stages"]["050"]["Clean"] == second.refs["Clean"]
    assert "Composite" not in plan["reuse_stages"]["050"]
    decision = next(row for row in plan["decisions"] if row["stage"] == "Composite")
    assert decision["reason"] == "UPSTREAM_BYTE_REFERENCE_MISMATCH"


def test_planner_finds_complete_coherent_parent_after_earlier_partial_prefix(case):
    partial, complete = case.parent(), case.parent()
    prefix = _failed_catalog(partial, ("Source",))
    plan = case.plan([prefix, complete.candidate])
    assert plan["reuse_stages"]["050"] == complete.refs
    assert all(row.get("candidate_index") == 1 for row in plan["decisions"])


def test_only_changed_clean_and_its_consumer_miss(case):
    parent = case.parent()
    plan = case.plan([parent.candidate], config={"Clean": {"algorithm_change": "new_candidate"}})
    assert set(plan["reuse_stages"]["050"]) == {"Source", "Motion", "Mask", "Render"}
    misses = {row["stage"] for row in plan["decisions"] if row["decision"] == "BUILD_IN_NEW_TASK"}
    assert misses == {"Clean", "Composite"}


def test_outer_wrapper_change_does_not_rehash_historical_wrapper_as_algorithm(case):
    parent = case.parent()
    before = parent.candidate["packet"]["sha256"]
    parent.runtime.write_text("# new operation wrapper; sealed algorithms unchanged\n")
    plan = case.plan([parent.candidate])
    assert list(plan["reuse_stages"]["050"]) == list(STAGES)
    assert plan["parents"][0]["packet"]["sha256"] == before
    assert plan["parents"][0]["historical_wrapper_bytes_required_to_equal_current_wrapper"] is False


def test_missing_runtime_signature_is_not_silently_accepted(case):
    parent = case.parent()
    del parent.packet["runtime_code"]
    parent.candidate["packet"] = save(Path(parent.candidate["packet"]["path"]), parent.packet)
    with pytest.raises(resume.ResumePlanError, match="RUNTIME_SIGNATURE_INCOMPLETE"):
        case.plan([parent.candidate])


def test_opaque_algorithm_closure_is_rejected_before_trusting_its_signature(case):
    parent = case.parent()
    parent.packet["component_code"]["Clean"] = [{"not_a_signed_file": True}]
    parent.candidate["packet"] = save(Path(parent.candidate["packet"]["path"]), parent.packet)
    with pytest.raises(resume.ResumePlanError, match="ALGORITHM_SIGNATURE_INCOMPLETE"):
        case.plan([parent.candidate])


def test_model_claiming_sealed_complete_cannot_supply_partial_stage_catalog(case):
    parent = case.parent()
    model_path = Path(parent.candidate["result"]["path"]).parent / "MODEL_RESULT.json"
    model = json.loads(model_path.read_text())
    del model["sessions"][0]["stage_receipts"]["Composite"]
    model_ref = save(model_path, model)
    terminal_path = Path(parent.candidate["result"]["path"])
    terminal = json.loads(terminal_path.read_text())
    terminal["model_result"] = model_ref
    parent.candidate["result"] = save(terminal_path, terminal)
    with pytest.raises(resume.ResumePlanError, match="MODEL_STAGE_SET_INCOMPLETE"):
        case.plan([parent.candidate])


@pytest.mark.parametrize("candidates", [None, [], {"schema": resume.MANIFEST_SCHEMA, "parents": []}])
def test_no_candidates_means_no_scan_or_filesystem_input_read(case, monkeypatch, candidates):
    parent = case.parent()
    monkeypatch.setattr(resume, "_read", lambda *args: pytest.fail("undeclared file read"))
    monkeypatch.setattr(resume, "verify_v1_request", lambda *args, **kwargs: pytest.fail("no-candidate filesystem probe"))
    plan = case.plan(candidates, root=Path("/does/not/exist/must/not/be/scanned"))
    assert plan["reuse_stages"] == {} and plan["parents"] == []
    assert all(row["reason"] == "NO_EXPLICIT_CANDIDATE" for row in plan["decisions"])


def test_explicit_manifest_and_parent_objects_are_not_mutated(case):
    parent = case.parent()
    manifest = {"schema": resume.MANIFEST_SCHEMA, "parents": [parent.candidate]}
    before = deepcopy(manifest)
    plan = case.plan(manifest)
    assert manifest == before and len(plan["reuse_stages"]["050"]) == 6
    assert plan["files_written"] == plan["model_calls"] == plan["render_calls"] == 0


def test_wrapper_change_creates_new_task_without_changing_component_signatures(project,specs):
    req=auto.build_v1_request(project_root=project.root,session=project.raw/'050',repair_rounds=2)
    ref=save(project.root/'REQUEST.json',req);context=specs.temporal_execution_context(project.root,wall_seconds=120)
    first=specs.compatible_spec(project.root,req,ref,execution_context=context)
    (project.root/'scripts/maintenance/inpainting_v1_automation_specs.py').write_text('# new navigation only\n')
    second=specs.compatible_spec(project.root,req,ref,execution_context=context)
    assert first['task_id']!=second['task_id']
    assert first['component_signatures']==second['component_signatures']


def test_explicit_resume_navigation_is_task_input_not_clean_recipe(project,specs):
    req=auto.build_v1_request(project_root=project.root,session=project.raw/'050',repair_rounds=2)
    ref=save(project.root/'REQUEST.json',req);context=specs.temporal_execution_context(project.root,wall_seconds=120)
    first=specs.compatible_spec(project.root,req,ref,execution_context=context)
    manifest=save(project.root/'RESUME.json',{'schema':resume.MANIFEST_SCHEMA,'parents':[]})
    second=specs.compatible_spec(project.root,req,ref,execution_context=context,resume_manifest=manifest)
    assert first['task_id']!=second['task_id']
    assert first['component_signatures']==second['component_signatures']
    assert second['inputs']['resume_manifest']==manifest and second['reuse_stages']=={}
    assert 'resume_manifest'not in second['component_config']['Clean']


def test_cli_resume_is_explicit_v1_only(project,monkeypatch):
    from chaoyang.cli import inpainting_submit
    from chaoyang import paths
    from chaoyang.governance import inpainting_candidate_registration as registration
    monkeypatch.setattr(paths,'get_paths',lambda:SimpleNamespace(repo_root=project.root));seen=[]
    monkeypatch.setattr(registration,'submit_v1_compatible_request',lambda req,**kw:seen.append(kw)or{'status':'SYNTHETIC'})
    manifest=project.root/'RESUME.json'
    inpainting_submit.main(['--session',str(project.raw/'050'),'--mode','v1-compatible','--resume-from',str(manifest),'--preview'])
    assert seen[0]['resume_manifest_path']==manifest
    with pytest.raises(SystemExit):
        inpainting_submit.main(['--session','/not/read','--resume-from',str(manifest)])
