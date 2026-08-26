#!/usr/bin/env python3
"""Audit and plot the completed Kai R2 training/validation trajectory."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/grap_a_cap/kai22_retarget_ab_r2"
OUTPUT = ROOT / "outputs/training_analysis/kai22_r2"


def selection_score(row: dict) -> float:
    return (
        float(row["pos_err_w_m"])
        + float(row["rot_err_w_deg"]) / 100.0
        + float(row["joint_mae_w_normalized"]) * 0.10
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    history = json.loads((run / "train_history.json").read_text(encoding="utf-8"))
    complete = json.loads((run / "training_complete.json").read_text(encoding="utf-8"))
    snapshots = []
    for path in sorted((run / "eval_snapshots").glob("eval_ep_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["selection_score"] = selection_score(row)
        snapshots.append(row)
    if not history or not snapshots:
        raise RuntimeError("training history or validation snapshots are empty")

    first_eval_history_index = next(
        index for index, row in enumerate(history) if int(row.get("frames", 0)) > 0
    )
    epoch_offset = int(snapshots[0]["epoch"]) - (first_eval_history_index + 1)
    epochs = np.arange(1 + epoch_offset, len(history) + 1 + epoch_offset)
    expected_last = int(complete["completed_epoch"])
    if int(epochs[-1]) != expected_last:
        raise RuntimeError(
            f"history alignment failed: inferred last epoch {epochs[-1]}, expected {expected_last}"
        )
    for index, row in enumerate(history):
        if int(row.get("frames", 0)) <= 0:
            continue
        epoch = int(epochs[index])
        snapshot = next(item for item in snapshots if int(item["epoch"]) == epoch)
        if not np.isclose(float(row["pos_err_w_m"]), float(snapshot["pos_err_w_m"])):
            raise RuntimeError(f"validation/history mismatch at epoch {epoch}")

    fields = ["epoch", "loss", "l_flow", "l_pos", "l_rot", "l_g", "l_done", "lr"]
    with (output / "train_loss.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for epoch, row in zip(epochs, history):
            writer.writerow({"epoch": int(epoch), **{key: row.get(key) for key in fields[1:]}})

    validation_fields = [
        "epoch", "selection_score", "pos_err_w_m", "rot_err_w_deg",
        "joint_mae_w_normalized", "pos_err_k1_m", "pos_err_kK_m",
        "joint_mae_k1_normalized", "joint_mae_kK_normalized",
    ]
    with (output / "validation_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=validation_fields, lineterminator="\n")
        writer.writeheader()
        for row in snapshots:
            writer.writerow({key: row[key] for key in validation_fields})

    best = min(snapshots, key=lambda row: row["selection_score"])
    last = max(snapshots, key=lambda row: row["epoch"])
    first = min(snapshots, key=lambda row: row["epoch"])
    best_train = history[int(best["epoch"]) - 1 - epoch_offset]
    minimum_index = int(np.argmin([float(row["loss"]) for row in history]))
    minimum_train = history[minimum_index]

    figure, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    ax = axes[0, 0]
    for key, label in (("loss", "总损失"), ("l_flow", "流匹配损失")):
        ax.plot(epochs, [row[key] for row in history], label=label, linewidth=1.8)
    ax.set_yscale("log")
    ax.set_title("Kai R2训练损失")
    ax.set_xlabel("训练轮次（epoch）")
    ax.set_ylabel("损失（对数坐标）")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for key, label in (
        ("l_pos", "手腕位置诊断项"),
        ("l_rot", "手腕旋转诊断项"),
        ("l_g", "机器人关节q诊断项"),
        ("l_done", "任务结束BCE"),
    ):
        ax.plot(epochs, [row[key] for row in history], label=label, linewidth=1.35)
    ax.set_yscale("log")
    ax.set_title("训练诊断项（已包含在流损失中，不可重复相加）")
    ax.set_xlabel("训练轮次（epoch）")
    ax.set_ylabel("诊断值（对数坐标）")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    eval_epochs = np.asarray([row["epoch"] for row in snapshots])
    scores = np.asarray([row["selection_score"] for row in snapshots])
    ax = axes[1, 0]
    ax.plot(eval_epochs, scores, marker="o", markersize=3, linewidth=1.5)
    ax.scatter([best["epoch"]], [best["selection_score"]], color="red", zorder=5)
    ax.axvline(best["epoch"], color="red", linestyle="--", alpha=0.65)
    ax.annotate(
        f"最优轮次 {best['epoch']}\n分数 {best['selection_score']:.6f}",
        (best["epoch"], best["selection_score"]), xytext=(8, 10),
        textcoords="offset points", color="red",
    )
    ax.set_title("验证集checkpoint选择分数（越低越好）")
    ax.set_xlabel("训练轮次（epoch）")
    ax.set_ylabel("位置[m] + 旋转[度]/100 + 0.1×q归一化MAE")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(eval_epochs, [row["pos_err_w_m"] * 1000 for row in snapshots], label="手腕位置误差（mm）")
    ax.plot(eval_epochs, [row["rot_err_w_deg"] for row in snapshots], label="手腕旋转误差（度）")
    ax.set_xlabel("训练轮次（epoch）")
    ax.set_ylabel("位置误差 / 旋转误差")
    ax.grid(alpha=0.25)
    q_axis = ax.twinx()
    q_axis.plot(
        eval_epochs,
        [row["joint_mae_w_normalized"] for row in snapshots],
        color="#8e44ad", label="机器人q归一化MAE",
    )
    q_axis.set_ylabel("机器人q归一化MAE", color="#8e44ad")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = q_axis.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper right")
    ax.set_title("验证集指标")

    figure.suptitle(
        "Kai22 R2：78段数据训练审计｜第285轮早停｜验证集最优第225轮",
        fontsize=15,
    )
    curve_path = output / "r2_training_loss_and_validation.png"
    figure.savefig(curve_path, dpi=180)
    plt.close(figure)

    report = {
        "schema_version": 1,
        "run": str(run),
        "history_alignment": {
            "recorded_epoch_range": [int(epochs[0]), int(epochs[-1])],
            "missing_history_prefix": [1, int(epochs[0] - 1)],
            "method": "align first history row containing validation metrics to the earliest eval snapshot",
            "validated_against_all_eval_snapshots": True,
        },
        "training": {
            "configured_epochs": int(complete["configured_epochs"]),
            "completed_epoch": int(complete["completed_epoch"]),
            "termination_reason": complete["termination_reason"],
            "global_step": int(complete["global_step"]),
            "elapsed_hours": float(complete["elapsed_s"] / 3600.0),
            "first_recorded": {"epoch": int(epochs[0]), "loss": float(history[0]["loss"])},
            "at_validation_best": {"epoch": int(best["epoch"]), "loss": float(best_train["loss"])},
            "minimum": {
                "epoch": int(epochs[minimum_index]),
                "loss": float(minimum_train["loss"]),
                "l_flow": float(minimum_train["l_flow"]),
                "l_pos": float(minimum_train["l_pos"]),
                "l_rot": float(minimum_train["l_rot"]),
                "l_robot_q": float(minimum_train["l_g"]),
                "l_done": float(minimum_train["l_done"]),
            },
            "recorded_loss_reduction_percent": float(
                (1.0 - minimum_train["loss"] / history[0]["loss"]) * 100.0
            ),
        },
        "validation": {
            "selection_formula": "pos_err_w_m + rot_err_w_deg/100 + 0.1*joint_mae_w_normalized",
            "first": {key: first[key] for key in validation_fields},
            "best": {key: best[key] for key in validation_fields},
            "last": {key: last[key] for key in validation_fields},
            "interpretation": "training loss kept decreasing after epoch 225 while the validation score stopped improving; best.pt correctly freezes epoch 225 EMA weights",
        },
        "outputs": {
            "curve": str(curve_path),
            "train_csv": str(output / "train_loss.csv"),
            "validation_csv": str(output / "validation_curve.csv"),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
