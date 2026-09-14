from __future__ import annotations

import importlib.util
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]


def load_script(name: str):
    path=ROOT/"tools"/name
    spec=importlib.util.spec_from_file_location(name.removesuffix(".py"),path)
    assert spec and spec.loader
    module=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_explicit_mapping_does_not_guess_task_from_number():
    batch=load_script("batch_convert_handle_egodex_to_tracker.py")
    chips=batch.mapping("001",task="potato_chips",date_tag="0910")
    cards=batch.mapping("001",task="playing_cards",date_tag="0910")
    assert chips["target_name"]=="get_potato_chips_0910_001"
    assert cards["target_name"]=="play_cards_0910_001"
    assert chips["policy_excluded"] is False
    assert cards["policy_excluded"] is False


def test_migration_is_resumable_and_preserves_content(tmp_path: Path):
    migration=load_script("migrate_handle_source_to_task_layout.py")
    migration.task_for=lambda session_id: (
        "potato_chips" if int(session_id)<=102 else "playing_cards"
    )
    # The production tool freezes 001..104, so exercise it through the CLI with
    # lightweight one-file sessions rather than weakening the production range.
    source=tmp_path/"source"
    source.mkdir()
    for number in range(1,105):
        session=source/f"{number:03d}"
        session.mkdir()
        (session/"identity.txt").write_text(f"session-{number:03d}\n")
    command=[sys.executable,str(ROOT/"tools"/"migrate_handle_source_to_task_layout.py"),
             "--source-root",str(source),"--execute"]
    subprocess.run(command,check=True,capture_output=True,text=True)
    assert (source/"potato_chips"/"001"/"identity.txt").read_text()=="session-001\n"
    assert (source/"playing_cards"/"104"/"identity.txt").read_text()=="session-104\n"
    assert not (source/"001").exists()
    second=subprocess.run(command,check=True,capture_output=True,text=True)
    assert "ALREADY_COMMITTED" in second.stdout


def test_split_task_dry_run_supports_overlapping_ids_and_explicit_exclusion(
    tmp_path: Path,
):
    chips=tmp_path/"chips"
    cards=tmp_path/"cards"
    for root,names in ((chips,("001","002")),(cards,("001",))):
        root.mkdir()
        for name in names:
            (root/name).mkdir()
    command=[
        sys.executable,str(ROOT/"tools"/"batch_convert_handle_egodex_to_tracker.py"),
        "--task-source",f"potato_chips={chips}",
        "--task-source",f"playing_cards={cards}",
        "--task-sessions","potato_chips=001",
        "--task-sessions","playing_cards=001",
        "--policy-exclude","potato_chips=001",
        "--date-tag","0910",
        "--dataset-id","test_0910",
        "--target-root",str(tmp_path/"output"),
        "--dry-run",
    ]
    done=subprocess.run(command,check=True,capture_output=True,text=True)
    payload=json.loads(done.stdout)
    assert payload["session_count"]==2
    mappings=payload["mapping"]["sessions"]
    assert mappings["potato_chips/001"]["policy_excluded"] is True
    assert mappings["playing_cards/001"]["policy_excluded"] is False


def test_committed_rejected_migration_updates_dataset_indexes(tmp_path: Path):
    root=tmp_path/"processed"
    rejected=root/"rejected"
    rejected.mkdir(parents=True)
    sessions=[]
    for session_id in ("001","103"):
        final=rejected/session_id
        final.mkdir()
        for name in ("REJECTED.json","CONTENT_MANIFEST.json","RESULT.json"):
            (final/name).write_text('{"ok": true}\n')
        sessions.append({"session_id":session_id,"classification":"REJECTED",
                         "final":str(final)})
    (root/"DATASET_RESULT.json").write_text(
        json.dumps({"state":"COMMITTED","sessions":sessions})+"\n"
    )
    fields=["session_id","classification","final"]
    with (root/"BATCH_STATUS.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for item in sessions:
            writer.writerow(item)
    command=[
        sys.executable,
        str(ROOT/"tools"/"migrate_committed_rejected_to_task_layout.py"),
        "--dataset-root",str(root),"--execute",
    ]
    subprocess.run(command,check=True,capture_output=True,text=True)
    assert (rejected/"potato_chips"/"001"/"REJECTED.json").is_file()
    assert (rejected/"playing_cards"/"103"/"REJECTED.json").is_file()
    result=json.loads((root/"DATASET_RESULT.json").read_text())
    assert result["layout_migration"]["status"]=="COMMITTED_VERIFIED"
    assert result["sessions"][0]["final"].endswith("rejected/potato_chips/001")
