"""T0 successor: preserve raw/legacy frame-count disagreement as evidence."""

from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path

from chaoyang.governance import execute_exact78_raw_t0_v1 as base
from chaoyang.governance.common import REPO_ROOT, artifact_ref


TASK = "exact78_raw_to_robot_t0_input_freeze_retry1_20260930"
SCRIPT = Path(__file__)
OUT = REPO_ROOT / "_run/current" / TASK / "attempts/attempt_0001"
PACKET = REPO_ROOT / "tasks/current" / TASK / "TASK_PACKET.json"
RECEIPT = REPO_ROOT / "tasks/receipts/EXACT78_RAW_TO_ROBOT_T0_RETRY1.json"
PRIOR = REPO_ROOT / "_run/current/exact78_raw_to_robot_t0_input_freeze_v1_20260930/attempts/attempt_0001/RESULT.json"
RECOVERY = PRIOR.parent / "TERMINALIZATION_RECOVERY.json"


def install_successor() -> None:
    """Keep v1 immutable; bind both modules into this successor's signature."""
    original_ref = base._ref
    original_inventory = base.inventory
    original_selected_inputs = base.selected_inputs
    original_frame_time = base.frame_time
    original_seal = base.seal
    original_register = base.register
    original_atomic_json = base.atomic_json
    base.TASK, base.OUT, base.PACKET, base.RECEIPT, base.SCRIPT = TASK, OUT, PACKET, RECEIPT, SCRIPT
    original_campaign_deadline = base.load_json(PRIOR)["campaign_deadline_at"]
    if datetime.fromisoformat(original_campaign_deadline) <= datetime.now().astimezone():
        raise RuntimeError("ORIGINAL_CAMPAIGN_DEADLINE_EXPIRED")

    def code_ref(path: Path):
        ref = original_ref(path)
        if path == SCRIPT and isinstance(ref, dict):
            return {**ref, "imported_helper": artifact_ref(Path(base.__file__))}
        return ref

    def inventory():
        cohorts, selected = original_inventory()
        if not PRIOR.is_file() or not RECOVERY.is_file():
            raise RuntimeError("PRIOR_TERMINAL_EVIDENCE_ABSENT")
        for source in ("0901", "0902"):
            formal = [r for r in cohorts[source]["rows"]
                      if Path(r["raw_path"]).parent.name in ("playing_cards", "potato_chips")]
            formal_ids = {r["session_id"] for r in formal}
            for name in list(selected):
                if source in name and name not in base.DEV[source] and name not in formal_ids:
                    del selected[name]
            dev_groups = {r["group_key"] for r in formal if r["session_id"] in base.DEV[source]}
            holdout = next((r for r in formal if r["complete"] and r["group_key"] not in dev_groups), None)
            if holdout:
                selected[holdout["session_id"]] = {**holdout, "role": "holdout"}
            cohorts[source].update(rows=formal,
                distinct_group_keys=sorted({r["group_key"] for r in formal}),
                holdout_session_id=holdout["session_id"] if holdout else None,
                holdout_status="FROZEN_INDEPENDENT_GROUP" if holdout else "SPLIT_NOT_ESTABLISHED",
                excluded_nonformal_subtrees=[".pico_editor_trash"])
        for name, row in selected.items():
            if "0915" not in name:
                continue
            raw = Path(row["raw_path"]) / "raw"
            qpc_count = len(base._ticks(raw / "vst.qpc.ts.jsonl"))
            utc_count = len(base._ticks(raw / "vst.ts.jsonl"))
            row["legacy_manifest_frame_count"] = row.get("frame_count")
            row["raw_qpc_line_count"] = qpc_count
            row["raw_timestamp_line_count"] = utc_count
            row["prior_attempt_result"] = artifact_ref(PRIOR)
            row["prior_terminal_recovery"] = artifact_ref(RECOVERY)
            row["frame_count_provenance"] = (
                "RAW_SIDECARS_AGREE_LEGACY_MANIFEST_DIFFERS"
                if qpc_count == utc_count and row.get("frame_count") != qpc_count
                else "RAW_SIDECARS_MATCH_LEGACY_MANIFEST" if qpc_count == utc_count
                else "RAW_SIDECARS_DISAGREE")
        for row in cohorts["0915"]["rows"]:
            selected_row = selected.get(row["session_id"])
            if selected_row:
                row.update({key: selected_row[key] for key in (
                    "legacy_manifest_frame_count", "raw_qpc_line_count", "raw_timestamp_line_count",
                    "frame_count_provenance")})
        return cohorts, selected

    def selected_inputs(selected):
        inputs = original_selected_inputs(selected)
        for name, row in selected.items():
            if "0915" in name:
                inputs[name]["frame_count_provenance"] = {
                    "legacy_manifest": row["legacy_manifest_frame_count"],
                    "raw_qpc_lines": row["raw_qpc_line_count"],
                    "raw_timestamp_lines": row["raw_timestamp_line_count"],
                    "status": row["frame_count_provenance"],
                    "prior_attempt_result": row["prior_attempt_result"],
                    "prior_terminal_recovery": row["prior_terminal_recovery"],
                }
        return inputs

    def frame_time(selected, packet):
        adjusted = copy.deepcopy(selected)
        for name, row in adjusted.items():
            if "0915" in name:
                if row["raw_qpc_line_count"] != row["raw_timestamp_line_count"]:
                    raise RuntimeError(f"RAW_TIMESTAMP_LINE_COUNT_MISMATCH:{name}")
                row["frame_count"] = row["raw_qpc_line_count"]
        mapping = original_frame_time(adjusted, packet)
        for name, session in mapping["sessions"].items():
            row = selected[name]
            if "0915" in name:
                count = session["decoded_frames"]
                if count != row["raw_qpc_line_count"] or count != row["raw_timestamp_line_count"]:
                    raise RuntimeError(f"RAW_DECODE_SIDECAR_COUNT_MISMATCH:{name}")
                session["frame_count_provenance"] = {
                    "decoded_raw": count,
                    "qpc_raw_lines": row["raw_qpc_line_count"],
                    "timestamp_raw_lines": row["raw_timestamp_line_count"],
                    "legacy_processed_group_manifest": row["legacy_manifest_frame_count"],
                    "legacy_mismatch": row["legacy_manifest_frame_count"] != count,
                    "status": row["frame_count_provenance"],
                }
        return mapping

    def seal(packet, status, outputs, reason=None):
        return original_seal(packet, "FAILED_RUNTIME_FINAL" if status == "FAILED_RUNTIME" else status,
                             outputs, reason)

    def atomic_json(path, value, mode=0o444):
        if path == PACKET:
            value = {**value, "campaign_deadline_at": original_campaign_deadline,
                     "successor_of": artifact_ref(PRIOR)}
        return original_atomic_json(path, value, mode)

    def register(cohorts, selected):
        packet = original_register(cohorts, selected)
        packet["campaign_deadline_at"] = original_campaign_deadline
        packet["successor_of"] = artifact_ref(PRIOR)
        if base.load_json(PACKET) != packet:
            raise RuntimeError("SUCCESSOR_PACKET_MISMATCH")
        return packet

    base._ref = code_ref
    base.inventory = inventory
    base.selected_inputs = selected_inputs
    base.frame_time = frame_time
    base.seal = seal
    base.atomic_json = atomic_json
    base.register = register


def main() -> int:
    install_successor()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
