#!/usr/bin/env python3
"""Build small, shallow 0916 cleaning review artifacts after final commit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT = Path(
    "/mnt/data/egodata/datasets/ego/processed/"
    "chips_cards_handle_highview_0916/DATASET_RESULT.json"
)
DEFAULT_OUTPUT = ROOT / "docs/current/visuals/0916_CLEANING_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    if (
        result.get("state") != "COMMITTED"
        or result.get("session_count") != 240
        or result.get("failed") != 0
        or len(result.get("results", [])) != 240
    ):
        raise ValueError("0916 dataset is not a 240-session zero-runtime-failure commit")
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    reasons: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for row in result["results"]:
        classification = str(row["classification"])
        totals[classification] += 1
        by_task[str(row["task"])][classification] += 1
        if classification == "REJECTED":
            reasons[str(row.get("reason") or "UNSPECIFIED")] += 1
    expected = {
        "CLEANED": int(result["cleaned"]),
        "REJECTED": int(result["rejected"]),
        "FAILED": int(result["failed"]),
    }
    if any(totals[key] != value for key, value in expected.items()):
        raise ValueError("0916 published counts are not derived from session rows")
    return {
        "schema_version": "0916-cleaning-review-summary-v1",
        "state": result["state"],
        "session_count": result["session_count"],
        "cleaned": totals["CLEANED"],
        "rejected": totals["REJECTED"],
        "failed": totals["FAILED"],
        "by_task": {task: dict(sorted(counts.items())) for task, counts in sorted(by_task.items())},
        "rejection_reasons": dict(sorted(reasons.items())),
        "claim_limit": "0916 content-cleaning review only; no downstream model authority.",
    }


def selected_clean_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    tasks = sorted({str(row["task"]) for row in result["results"]})
    for task in tasks:
        rows = sorted(
            (row for row in result["results"]
             if row["task"] == task and row["classification"] == "CLEANED"),
            key=lambda row: str(row["session_id"]),
        )
        if not rows:
            continue
        for index in sorted({0, len(rows) // 2, len(rows) - 1}):
            selected.append(rows[index])
    return selected


def render_counts(summary: dict[str, Any], output: Path) -> None:
    canvas = np.full((620, 1200, 3), 248, np.uint8)
    cv2.putText(canvas, "0916 CLEANING V1 - FIXED DENOMINATOR 240", (45, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.15, (35, 35, 35), 2, cv2.LINE_AA)
    rows = [
        ("CLEANED", int(summary["cleaned"]), (64, 160, 72)),
        ("REJECTED_QUALITY", int(summary["rejected"]), (40, 130, 220)),
        ("FAILED_RUNTIME", int(summary["failed"]), (30, 30, 190)),
    ]
    maximum = max(value for _label, value, _color in rows) or 1
    for index, (label, value, color) in enumerate(rows):
        y = 160 + index * 130
        cv2.putText(canvas, label, (45, y + 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (50, 50, 50), 2, cv2.LINE_AA)
        width = int(760 * value / maximum)
        cv2.rectangle(canvas, (330, y), (330 + width, y + 64), color, -1)
        cv2.putText(canvas, str(value), (350 + width, y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Quality rejection is terminal and is not a runtime failure.",
                (45, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (70, 70, 70), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"failed to write {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset_result = args.dataset_result.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"fresh shallow review directory required: {output}")
    value = json.loads(dataset_result.read_text(encoding="utf-8"))
    summary = summarize(value)
    output.mkdir(parents=True)
    atomic_json(output / "SUMMARY.json", summary)
    render_counts(summary, output / "COUNTS.png")

    representatives = []
    for row in selected_clean_rows(value):
        source = Path(row["target"]) / "review/manus_projection_eight_frame.jpg"
        if not source.is_file():
            raise RuntimeError(f"representative review missing: {source}")
        name = f"{row['task']}__{row['session_id']}__review.jpg"
        target = output / name
        shutil.copyfile(source, target)
        representatives.append({
            "task": row["task"], "session_id": row["session_id"],
            "source": ref(source), "published": ref(target),
        })
    rejected = [{
        "task": row["task"], "session_id": row["session_id"],
        "reason": row.get("reason"), "source": row.get("source"),
        "receipt": ref(Path(row["target"]) / "RESULT.json"),
    } for row in value["results"] if row["classification"] == "REJECTED"]
    manifest = {
        "schema_version": "0916-cleaning-visual-manifest-v1",
        "dataset_result": ref(dataset_result),
        "summary": ref(output / "SUMMARY.json"),
        "counts_visual": ref(output / "COUNTS.png"),
        "representatives": representatives,
        "rejected": rejected,
        "claim_limit": summary["claim_limit"],
    }
    atomic_json(output / "MANIFEST.json", manifest)
    markdown = [
        "# 0916 清洗 V1 审阅", "",
        f"- 固定分母：{summary['session_count']}",
        f"- CLEANED：{summary['cleaned']}",
        f"- REJECTED：{summary['rejected']}",
        f"- FAILED：{summary['failed']}",
        "- 计数图：[`COUNTS.png`](COUNTS.png)", "",
        "质量拒绝是有效终态，不计为运行失败。该目录只证明 0916 内容清洗与抽样审阅，"
        "不授予 HaWoR、Mask、Depth、Contact 或 Robot 权威。", "",
        "## 代表性审阅图", "",
    ]
    markdown.extend(
        f"- `{row['task']}/{row['session_id']}`：[`{Path(row['published']['path']).name}`]({Path(row['published']['path']).name})"
        for row in representatives
    )
    markdown.extend(["", "## 拒绝终态", ""])
    markdown.extend(
        f"- `{row['task']}/{row['session_id']}`：`{row['reason']}`"
        for row in rejected
    )
    (output / "README_ZH.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
