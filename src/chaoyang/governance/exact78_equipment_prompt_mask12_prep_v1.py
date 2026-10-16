"""Read-only CPU preflight for a proposed four-development-scene Mask experiment.

This is deliberately not a registered operation or a SAM runner.  It checks
the proposed one-variable contract against the sealed Mask12 anchor packet.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MASK12_PACKET = ROOT / "tasks/current/exact78_joint_hand_mask12_four_dev_v1_20261001/TASK_PACKET.json"
EXPECTED_PACKET_SHA = "bb60bb1dbe5b232cee71fbb3d91477df445eb444233468643e3822456399dc4f"
ADDED_PROMPT = "a white device attached to a hand or wrist"
EXPECTED_ANCHORS = {
    "play_cards_0902_042": (0, 85, 170),
    "get_potato_chips_0902_103": (0, 142, 283),
    "get_potato_chips_0901_013": (0, 435, 869),
    "play_cards_0915_001": (0, 75, 149),
}
EXPECTED_CHECKPOINT_SHA = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"


def proposal(packet: dict) -> dict:
    """Describe the proposed role calls without changing the sealed packet."""
    config = packet["config"]
    return {
        "sessions": [
            {"session_id": scene["session_id"],
             "source_family": scene["source_family"],
             "task": scene["task"],
             "anchors": [anchor["frame"] for anchor in scene["anchors"]]}
            for scene in packet["inputs"]["sessions"]
        ],
        "human_prompt": config["new_human_prompt"],
        "object_prompts": {"poker": config["other_role_prompts"]["poker_object"],
                           "chips": config["other_role_prompts"]["chips_object"]},
        "equipment_prompts_original": {
            "TRACKER_0902": config["other_role_prompts"]["tracker_equipment"],
            "TRACKER_0901": config["other_role_prompts"]["tracker_equipment"],
            "VST_0915": config["other_role_prompts"]["vst_equipment"],
        },
        "equipment_prompt_added": ADDED_PROMPT,
        "vst_extra_prompts": {
            "finger_sleeve": config["other_role_prompts"]["vst_finger_sleeve"],
            "yellow_cable": config["other_role_prompts"]["vst_yellow_cable"],
        },
        "output_prob_thresh": config["output_prob_thresh"],
        "instance_score_gate": config["instance_score_gate"],
        "sam_checkpoint_sha256": packet["inputs"]["sam_checkpoint"]["sha256"],
        "model_calls_max": 54,
        "clean_calls_authorized": 0,
        "holdout_calls_authorized": 0,
        "review": "PENDING_HUMAN",
    }


def validate(packet: dict, candidate: dict) -> None:
    if packet.get("task_id") != "exact78_joint_hand_mask12_four_dev_v1_20261001":
        raise ValueError("SEALED_MASK12_PACKET_ID_DRIFT")
    if packet["inputs"]["sam_checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA:
        raise ValueError("SAM_CHECKPOINT_DRIFT")
    if candidate != proposal(packet):
        raise ValueError("SINGLE_VARIABLE_OR_SOURCE_DRIFT")
    anchors = {row["session_id"]: tuple(row["anchors"])
               for row in candidate["sessions"]}
    if anchors != EXPECTED_ANCHORS or len(candidate["sessions"]) != 4:
        raise ValueError("FOUR_DEV_TWELVE_ANCHORS_REQUIRED")
    if candidate["human_prompt"] != "a person's complete hand and forearm, including anything worn on or attached to them":
        raise ValueError("SEALED_MASK12_HUMAN_PROMPT_REQUIRED")
    if candidate["output_prob_thresh"] != 0.5 or candidate["instance_score_gate"] != 0.5:
        raise ValueError("SAM_THRESHOLD_DRIFT")
    if candidate["model_calls_max"] != 54:
        raise ValueError("EXACT_ROLE_CALL_BUDGET_REQUIRED")
    if candidate["clean_calls_authorized"] or candidate["holdout_calls_authorized"]:
        raise ValueError("CLEAN_OR_HOLDOUT_NOT_AUTHORIZED")


def self_test(packet: dict) -> int:
    base = proposal(packet)
    validate(packet, base)
    changes = (
        ("human", lambda x: x.update(human_prompt=packet["config"]["old_human_prompt"])),
        ("threshold", lambda x: x.update(instance_score_gate=0.49)),
        ("holdout", lambda x: x["sessions"].append({"session_id": "play_cards_0901_001", "anchors": [0]})),
        ("equipment", lambda x: x.update(equipment_prompt_added="a wristband")),
        ("clean", lambda x: x.update(clean_calls_authorized=1)),
    )
    for name, change in changes:
        altered = copy.deepcopy(base)
        change(altered)
        try:
            validate(packet, altered)
        except ValueError:
            continue
        raise AssertionError(f"NEGATIVE_GATE_FAILED:{name}")
    return len(changes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.check_only and not args.self_test:
        parser.error("--check-only or --self-test required")
    data = MASK12_PACKET.read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_PACKET_SHA:
        raise RuntimeError("SEALED_MASK12_PACKET_SHA_DRIFT")
    packet = json.loads(data)
    candidate = proposal(packet)
    validate(packet, candidate)
    negatives = self_test(packet) if args.self_test else 0
    print(json.dumps({"status": "CHECK_ONLY_PASS", "development_sessions": 4,
                      "anchors": 12, "model_calls_budget": 54,
                      "negative_cases_passed": negatives, "model_called": False,
                      "clean_called": False, "route_registered": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
