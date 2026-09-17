#!/usr/bin/env python3
"""Build exact78 Wave0 Contact HYPOTHESIS_ONLY readiness from immutable receipts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def digest(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def checked(ref: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(ref["path"]).resolve(strict=True)
    actual = digest(path)
    if actual["bytes"] != int(ref["bytes"]) or actual["sha256"] != ref["sha256"]:
        raise ValueError(f"immutable reference mismatch: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True)
    selection_path = args.selection.resolve(strict=True)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if len(selection.get("sessions", [])) != 58:
        raise ValueError("frozen Wave0 denominator is not 58")

    rows: list[dict[str, Any]] = []
    for selected in selection["sessions"]:
        session = str(selected["session_id"])
        task = str(selected["task"])
        expected_frames = int(selected["frame_count"])
        values: dict[str, dict[str, Any]] = {}
        refs: dict[str, dict[str, Any]] = {}
        for stage in ("hawor", "role_mask", "task_object_mask", "depth", "object6d"):
            path, value = checked(selected["upstream"][stage])
            values[stage] = value
            refs[stage] = digest(path)
            identity = value.get("session_id") or value.get("session")
            if identity != session:
                raise ValueError(f"{session}: {stage} identity mismatch")

        hawor = values["hawor"]
        role = values["role_mask"]
        object_mask = values["task_object_mask"]
        depth = values["depth"]
        object6d = values["object6d"]
        expected_instances = 3 if task == "chips" else 1
        observed_frames_total = 0
        instance_gate = True
        instance_results: list[dict[str, Any]] = []
        instances = object6d.get("instances", {})
        legacy_single_direct = not instances and expected_instances == 1 and (
            object6d.get("schema_version") == "visual-fixed-instance-object6d-result-v2"
            and object6d.get("status") == "PASS_VISUAL_OBJECT6D_OBSERVED_ONLY_BASELINE"
            and object6d.get("unobserved_pose_policy") == "KEEP_INVALID"
            and int(object6d.get("propagated_frames", -1)) == 0
            and int(object6d.get("observed_frames", 0)) > 0
            and int(object6d.get("instance_count", -1)) == 1
            and object6d.get("forbidden_inputs_absent") is True
        )
        if legacy_single_direct:
            observed_frames_total = int(object6d["observed_frames"])
            instance_results.append({
                "physical_instance_id": 0,
                "direct_observed_only": True,
                "observed_frames": observed_frames_total,
                "result": refs["object6d"],
                "legacy_single_instance_receipt": True,
            })
        elif len(instances) != expected_instances:
            instance_gate = False
        for raw_id, instance in sorted(instances.items(), key=lambda item: int(item[0])):
            try:
                result_path, instance_result = checked(instance["result"])
            except (KeyError, ValueError, FileNotFoundError):
                instance_gate = False
                continue
            direct = (
                instance_result.get("unobserved_pose_policy") == "KEEP_INVALID"
                and int(instance_result.get("propagated_frames", -1)) == 0
                and int(instance_result.get("observed_frames", 0)) > 0
            )
            instance_gate &= direct
            observed_frames_total += int(instance_result.get("observed_frames", 0))
            instance_results.append({
                "physical_instance_id": int(raw_id),
                "direct_observed_only": direct,
                "observed_frames": int(instance_result.get("observed_frames", 0)),
                "result": digest(result_path),
            })

        modern_input_closure = (
            depth.get("input_closure_sha256") is not None
            and depth.get("input_closure_sha256") == object6d.get("input_closure_sha256")
        )
        legacy_depth_ref = object6d.get("inputs", {}).get("depth_result", {})
        legacy_input_closure = (
            legacy_single_direct
            and legacy_depth_ref.get("path") == refs["depth"]["path"]
            and legacy_depth_ref.get("sha256") == refs["depth"]["sha256"]
            and int(object6d.get("frame_count", -1)) == expected_frames
        )

        gates = {
            "hawor_numeric": hawor.get("status") == "PASS_NUMERIC_NEEDS_HUMAN_REVIEW" and hawor.get("numeric_gate_pass") is True,
            "role_mask": role.get("grade") == "B" and role.get("downstream_authorized") is True,
            "object_mask": (
                object_mask.get("grade") == "B" and object_mask.get("downstream_authorized") is True
                and int(object_mask.get("instance_count", -1)) == expected_instances
                and sum(int(value) for value in object_mask.get("observed_counts", {}).values()) > 0
            ),
            "depth": (
                depth.get("status") == "PASS_CORRECTED_DENSE_METRIC_DEPTH_FULLSESSION"
                and depth.get("consumption_authorized") is True
                and "VISUAL_OBJECT6D_CANDIDATE_INPUT" in depth.get("authorized_scopes", [])
            ),
            "object6d": (
                (
                    object6d.get("status") == "PASS_OBSERVED_ONLY_OBJECT6D_GRADE_B"
                    and object6d.get("downstream_authorized") is True
                    and object6d.get("multi_instance_union_used") is False
                    and int(object6d.get("physical_instance_count", -1)) == expected_instances
                    or legacy_single_direct and object6d.get("consumption_authorized") is True
                )
                and object6d.get("unobserved_pose_policy") == "KEEP_INVALID"
                and instance_gate and observed_frames_total > 0
            ),
            "frame_identity": (
                int(role.get("frame_count", -1)) == expected_frames
                and int(object_mask.get("frame_count", -1)) == expected_frames
                and (modern_input_closure or legacy_input_closure)
            ),
        }
        order = [
            ("hawor_numeric", "HAWOR_NUMERIC_NOT_READY"),
            ("role_mask", "ROLE_MASK_NOT_READY"),
            ("object_mask", "OBJECT_MASK_NOT_READY"),
            ("depth", "DEPTH_NOT_READY"),
            ("object6d", "OBJECT6D_NOT_DIRECT_OBSERVED_ONLY"),
            ("frame_identity", "FRAME_OR_INPUT_CLOSURE_MISMATCH"),
        ]
        first_blocker = next((reason for gate, reason in order if not gates[gate]), "NONE_FOR_HYPOTHESIS_ONLY")
        ready = first_blocker == "NONE_FOR_HYPOTHESIS_ONLY"
        rows.append({
            "position": int(selected["position"]), "task": task, "session": session,
            "frame_count": expected_frames, "expected_instances": expected_instances,
            **gates, "direct_object6d_observed_frames_total": observed_frames_total,
            "can_generate_hypothesis_only": ready, "first_blocker": first_blocker,
            "claim_status": "HYPOTHESIS_ONLY" if ready else "BLOCKED_PREREQ",
            "external_accuracy": "UNKNOWN", "contact_authority": False,
            "attachment_used": False, "attachment_may_support_contact": False,
            "input_receipts": refs, "object6d_instances": instance_results,
        })

    counts = {
        "wave0_sessions": len(rows),
        "hypothesis_only_ready": sum(row["can_generate_hypothesis_only"] for row in rows),
        "blocked_prereq": sum(not row["can_generate_hypothesis_only"] for row in rows),
        "chips_ready": sum(row["task"] == "chips" and row["can_generate_hypothesis_only"] for row in rows),
        "poker_ready": sum(row["task"] == "poker" and row["can_generate_hypothesis_only"] for row in rows),
    }
    matrix = {
        "schema_version": "CONTACT_READINESS_MATRIX_R22_V1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection": digest(selection_path), "counts": counts, "rows": rows,
        "evidence_policy": "DIRECT_OBJECT6D_ONLY; HAND_OBJECT_ATTACHMENT_FORBIDDEN_AS_CONTACT_SUPPORT",
        "authority_promoted": False,
        "claim_limit": "Input readiness for development HYPOTHESIS_ONLY Contact; not Contact truth or accuracy.",
    }
    write_json(output / "CONTACT_READINESS_MATRIX.json", matrix)
    fields = [
        "position", "task", "session", "frame_count", "expected_instances", "hawor_numeric",
        "role_mask", "object_mask", "depth", "object6d", "frame_identity",
        "direct_object6d_observed_frames_total", "can_generate_hypothesis_only", "first_blocker",
        "claim_status", "external_accuracy", "contact_authority", "attachment_used",
    ]
    with (output / "CONTACT_READINESS_MATRIX.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    write_json(output / "METRICS.json", {"schema_version": "CONTACT_READINESS_METRICS_R22_V1", **counts})
    blockers: dict[str, int] = {}
    for row in rows:
        blockers[row["first_blocker"]] = blockers.get(row["first_blocker"], 0) + 1
    (output / "DECISION.md").write_text(
        "# exact78 Contact readiness决定\n\n"
        f"冻结Wave0共58条；{counts['hypothesis_only_ready']}条满足开发态HYPOTHESIS_ONLY输入门，"
        f"{counts['blocked_prereq']}条存在前置阻塞。该结论只表示能运行接触假设，"
        "不表示Contact正确或具有外部真值。\n\n"
        "所有Object6D只接受DIRECT_OBSERVED_ONLY/KEEP_INVALID/零传播实例证据。"
        "HAND_OBJECT_ATTACHMENT未使用，也不得反向证明Contact。\n",
        encoding="utf-8",
    )
    write_json(output / "NEXT_ACTION.json", {
        "schema_version": "CONTACT_READINESS_NEXT_R22_V1", "status": "PASSED",
        "next_task_id": "RUN_BOUNDED_CONTACT10_ON_SELECTED_READY_SESSIONS",
        "automatic_model_run": False,
    })
    write_json(output / "RUN_RECEIPT.json", {
        "schema_version": "CONTACT_READINESS_RECEIPT_R22_V1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection": digest(selection_path), "producer": digest(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_calls": 0, "attachment_used": False, "authority_promoted": False,
    })
    write_json(output / "ARTIFACT_MANIFEST.json", {
        "schema_version": "CONTACT_READINESS_ARTIFACTS_R22_V1",
        "artifacts": [
            digest(output / "CONTACT_READINESS_MATRIX.json"), digest(output / "CONTACT_READINESS_MATRIX.csv"),
            digest(output / "METRICS.json"), digest(output / "DECISION.md"), digest(output / "NEXT_ACTION.json"),
        ],
    })
    result = {
        "schema_version": "CONTACT_READINESS_RESULT_R22_V1", "terminal_status": "PASSED",
        "status": "PASS_DEVELOPMENT_CONTACT_READINESS_MATRIX", "counts": counts,
        "first_blocker_counts": blockers, "matrix": digest(output / "CONTACT_READINESS_MATRIX.json"),
        "authority_promoted": False, "claim_limit": matrix["claim_limit"],
    }
    write_json(output / "RESULT.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
