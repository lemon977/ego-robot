"""Fixed-grid same-session table-only donor probe for 007 frame 184."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from chaoyang.governance.common import REPO_ROOT, artifact_ref, load_json
from chaoyang.ops.run_human_to_robot_007_device_donor_probe import (
    RAW, HUMAN, ROLE, REBOUND, fit_heldout_homography, table_mask,
)

TASK = "human_to_robot_007_table_donor_expansion_20260923"
ATTEMPT = REPO_ROOT / f"_run/current/{TASK}/attempts/attempt_0001"
DEST = ATTEMPT / "lanes/scene/table_donor_expansion_v1"
TARGET = 184
DONORS = tuple(range(8, 369, 8))


def image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    value = cv2.imread(str(path), flags)
    if value is None:
        raise FileNotFoundError(path)
    return value


def choose_best(rows: list[dict]) -> int | None:
    """Only geometry-qualified donors with actual table support are candidates."""
    eligible = [row for row in rows if row["geometry_pass"] and row["table_write_pixels"] > 0]
    return min(eligible, key=lambda row: (-row["table_write_pixels"], row["donor_frame"]))["donor_frame"] if eligible else None


def main() -> int:
    index = load_json(REPO_ROOT / "tasks/current/INDEX.json")
    packets = index.get("task_packets", [])
    if len(packets) != 1 or packets[0].get("task_id") != TASK or not packets[0].get("execution_allowed"):
        raise RuntimeError("TASK_NOT_ROUTABLE")
    if DEST.exists():
        raise FileExistsError(DEST)
    cv2.setNumThreads(2)
    raw_path = RAW / f"{TARGET:06d}.png"
    raw = image(raw_path)
    write_path = REBOUND / "write/000003.png"
    write = image(write_path, cv2.IMREAD_GRAYSCALE) > 0
    role_path = ROLE / f"{TARGET:06d}.png"
    role = image(role_path, cv2.IMREAD_GRAYSCALE) > 0
    if raw.shape != (960, 1280, 3) or write.shape != (960, 1280) or role.shape != write.shape:
        raise RuntimeError("TARGET_DOMAIN_MISMATCH")
    target_table = table_mask(raw, write | role)
    rows: list[dict] = []
    homographies: dict[int, np.ndarray] = {}
    for frame in DONORS:
        donor_path = RAW / f"{frame:06d}.png"
        donor = image(donor_path)
        human_path = HUMAN / f"{frame:06d}.png"
        human = image(human_path, cv2.IMREAD_GRAYSCALE) > 0
        donor_role_path = ROLE / f"{frame:06d}.png"
        donor_role = image(donor_role_path, cv2.IMREAD_GRAYSCALE) > 0
        if donor.shape != raw.shape or human.shape != write.shape or donor_role.shape != write.shape:
            raise RuntimeError(f"DONOR_DOMAIN_MISMATCH:{frame}")
        donor_table = table_mask(donor, human | donor_role, donor_frame=frame)
        h, metric = fit_heldout_homography(donor, raw, donor_table, target_table)
        table_write_pixels = 0
        if h is not None:
            warped_table = cv2.warpPerspective(donor_table, h, (1280, 960), flags=cv2.INTER_NEAREST) > 0
            table_write_pixels = int((warped_table & write).sum())
            homographies[frame] = h
        rows.append({"donor_frame": frame,
                     "raw": artifact_ref(donor_path),
                     "human": artifact_ref(human_path),
                     "capture_device": artifact_ref(donor_role_path),
                     **metric, "table_write_pixels": table_write_pixels})
    best_frame = choose_best(rows)
    DEST.mkdir(parents=True)
    candidate = None
    if best_frame is not None:
        donor = image(RAW / f"{best_frame:06d}.png")
        human = image(HUMAN / f"{best_frame:06d}.png", cv2.IMREAD_GRAYSCALE) > 0
        donor_role = image(ROLE / f"{best_frame:06d}.png", cv2.IMREAD_GRAYSCALE) > 0
        donor_table = table_mask(donor, human | donor_role, donor_frame=best_frame)
        h = homographies[best_frame]
        warped = cv2.warpPerspective(donor, h, (1280, 960))
        warped_table = cv2.warpPerspective(donor_table, h, (1280, 960), flags=cv2.INTER_NEAREST) > 0
        support = warped_table & write
        composite = raw.copy()
        composite[support] = warped[support]
        support_path = DEST / "007_FRAME184_TABLE_ONLY_SUPPORT.png"
        review_path = DEST / "007_FRAME184_TABLE_ONLY_DONOR_REVIEW.png"
        if not cv2.imwrite(str(support_path), support.astype(np.uint8) * 255):
            raise RuntimeError("SUPPORT_WRITE_FAILED")
        if not cv2.imwrite(str(review_path), np.concatenate([raw, composite], axis=1)):
            raise RuntimeError("REVIEW_WRITE_FAILED")
        candidate = {"donor_frame": best_frame, "support": artifact_ref(support_path),
                     "review": artifact_ref(review_path), "replaced_table_pixels": int(support.sum()),
                     "claim_limit": "Geometry-qualified table-only diagnostic; plate/chips and occluded surfaces remain unknown."}
    result = {
        "schema_version": "HUMAN_TO_ROBOT_007_TABLE_DONOR_EXPANSION_V1",
        "task_id": TASK, "session_id": "get_potato_chips_0915_007",
        "target_frame": TARGET, "donor_frames": list(DONORS), "donor_count": len(DONORS),
        "prior_four_donors_excluded": [0, 100, 250, 377],
        "execution": "EXECUTED", "structure": "PASS",
        "quality": "TABLE_ONLY_CANDIDATE" if candidate else "NO_QUALIFIED_TABLE_DONOR",
        "adoption": "NOT_ADOPTED", "target_raw": artifact_ref(raw_path),
        "target_write": artifact_ref(write_path), "target_capture_device": artifact_ref(role_path),
        "rows": rows, "geometry_pass_count": sum(bool(row["geometry_pass"]) for row in rows),
        "candidate": candidate,
        "claim_limit": "Fixed-grid same-session table-only donor diagnostic, not complete Clean or product quality.",
        "training_eligible": False, "control_ground_truth": False,
        "physical_deployable": False, "external_metric_authority": False,
    }
    (DEST / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"geometry_pass_count": result["geometry_pass_count"],
                      "best_frame": best_frame, "result": str(DEST / "RESULT.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
