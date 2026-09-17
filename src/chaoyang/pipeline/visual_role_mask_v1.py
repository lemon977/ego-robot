"""Strict role vocabulary and manifest validation for 0915 SAM3.1 masks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import json

import jsonschema


SAM31_IDENTITY = "SAM3.1"
ROLE_NAMES = frozenset({
    "left_human_skin_forearm",
    "right_human_skin_forearm",
    "left_finger_sleeve_attachment",
    "right_finger_sleeve_attachment",
    "left_cable",
    "right_cable",
    "task_object",
})
CLEAN_REMOVAL_ROLES = frozenset({
    "left_human_skin_forearm",
    "right_human_skin_forearm",
    "left_finger_sleeve_attachment",
    "right_finger_sleeve_attachment",
    "left_cable",
    "right_cable",
})
FORBIDDEN_ROLE_TOKENS = frozenset({"tracker", "controller", "pico", "pico26"})
SCHEMA_PATH = Path(__file__).resolve().parents[3] / "contracts" / "visual_role_mask_v1.schema.json"


class VisualRoleContractError(ValueError):
    pass


def validate_manifest(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(value)
    roles = set(value["roles"])
    if roles - ROLE_NAMES:
        raise VisualRoleContractError("unknown visual role")
    if value["model"]["identity"] != SAM31_IDENTITY:
        raise VisualRoleContractError("0915 role mask model must be SAM3.1")
    instance_ids: set[str] = set()
    for instance in value["instances"]:
        instance_id = str(instance["instance_id"])
        if instance_id in instance_ids:
            raise VisualRoleContractError(f"duplicate instance_id: {instance_id}")
        instance_ids.add(instance_id)
        lowered = instance_id.lower()
        if any(token in lowered for token in FORBIDDEN_ROLE_TOKENS):
            raise VisualRoleContractError(
                f"forbidden tracker/controller/PICO identity: {instance_id}"
            )
        if instance["role"] not in roles:
            raise VisualRoleContractError(
                f"instance role is not declared: {instance['role']}"
            )
        if root is not None:
            directory = (root / instance["mask_directory"]).resolve()
            if not directory.is_relative_to(root.resolve()):
                raise VisualRoleContractError("mask directory escapes artifact root")
    task_objects = [row for row in value["instances"]
                    if row["role"] == "task_object"]
    if len({row["instance_id"] for row in task_objects}) != len(task_objects):
        raise VisualRoleContractError("task objects must remain separate instances")


def clean_removal_roles() -> tuple[str, ...]:
    """Return explicit visual-removal roles; task objects are never included."""
    return tuple(sorted(CLEAN_REMOVAL_ROLES))
