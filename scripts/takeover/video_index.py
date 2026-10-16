"""Render 40_VIDEO_INDEX_ZH.md from a baseline_summary.py JSON.

Usage: video_index.py <summary.json> <seal.json or ""> <out.md> [<visual_checks.json>]
"""
import json
import sys

summary = json.load(open(sys.argv[1]))
seal = json.load(open(sys.argv[2])) if len(sys.argv) > 2 and sys.argv[2] else {}
checks = json.load(open(sys.argv[4])) if len(sys.argv) > 4 and sys.argv[4] else {}
task = summary["task"]
cand = f"_run/current/{task}/attempts/attempt_0001/candidate"
root = "/path/to/chaoyang/" + cand


def pct(value):
    return f"{100 * value:.1f}%"


def stat_line(name, s, fmt=pct):
    return f"| {name} | {fmt(s['min'])} | {fmt(s['median'])} | {fmt(s['p90'])} | {fmt(s['max'])} |"


groups = summary["by_group"]
lines = [
    "# Exact78 接手基线 v1：全量视频索引",
    "",
    f"全量运行 `{task}` 的视频位置与汇总统计。做法与限制见 `20_BASELINE_V1_ZH.md`。",
    "这些视频是供人工评审的视觉基线候选，不是物理遮挡结论，也不能作为训练或控制依据。",
    "",
    "## 1. 从哪里看",
    "",
    (f"- 总索引页：`{root}/index.html`（每条会话一行，带 final / review 链接）。" if summary.get("has_index", True)
     else "- 这次运行没有跑完，没有总索引页（`index.html` / `INDEX.json` 只在运行结束时写出），直接按会话目录看。"),
    f"- 每条会话：`{root}/sessions/<会话>/`"
    + (f"（其中 {summary['folders_outside_candidate']} 条在被续跑的上一轮 candidate 里，以 `INDEX.json` 每行的 `folder` 为准）"
       if summary.get("folders_outside_candidate") else ""),
    "  - `final.mp4`：最终机器人视频（640×480）。",
    "  - `review.mp4`：6 宫格评审视频（原片 + PICO 骨架 / 移除掩码与手持物体 / Clean / 自我视角机器人 / 外部视角整机 / 最终）。",
    "  - `SESSION.json`：该会话的指标、标签和各产物的 SHA-256。",
    (f"- 机器可读汇总：`{cand}/INDEX.json`；运行进度：`{cand}/PROGRESS.json`。" if summary.get("has_index", True)
     else f"- 运行进度（每条会话的状态和目录）：`{cand}/PROGRESS.json`。"),
]
if seal:
    lines += [f"- 封存结果：`_run/current/{task}/attempts/attempt_0001/RESULT.json`，状态 {seal['status']}，"
              f"sha256 `{seal['sha256']}`。"]
lines += [
    "",
    "## 2. 覆盖",
    "",
    f"进入 OK 的会话 {summary['ok_sessions']} 条，共 {summary['frames']:,} 帧；运行状态计数：{summary['progress_counts']}。",
    "",
    "| 分组 | 会话 | 帧 |",
    "|------|------|----|",
]
names = {"0901/playing_cards": "0901 扑克", "0901/potato_chips": "0901 薯片",
         "0902/playing_cards": "0902 扑克", "0902/potato_chips": "0902 薯片"}
for key, value in groups.items():
    lines.append(f"| {names.get(key, key)} | {value['sessions']} | {value['frames']:,} |")
failed = summary.get("failed") or {}
lines += ["", ("失败会话：" + "、".join(f"`{k}`" for k in failed)) if failed else "失败会话：无。", ""]
if summary.get("recomputed"):
    retries = summary.get("forearm_retries") or {}
    lines += [
        f"本轮重算了 {len(summary['recomputed'])} 条会话（gpu / render 在本轮的代码上重新计算），其余会话直接引用上一轮 candidate 的目录；"
        f"用新代码（前后夹住填充）算出的会话共 {summary.get('sessions_with_bracket_code', 0)} 条。"
        f"其中 {len(retries)} 条触发了前臂重试提示：" + ("、".join(
            f"`{name}`（{'、'.join(row['side'] + ' 锚定 ' + str(row['anchor_frame']) + '，重试前命中率 ' + str(row['hit_rate_before']) for row in rows)}）"
            for name, rows in retries.items()) or "无") + "。",
        "",
        "重算的会话：" + "、".join(f"`{name}`" for name in summary["recomputed"]) + "。",
        "",
    ]
lines += [
    "## 3. 指标分布（按会话统计）",
    "",
    "| 指标 | 最小 | 中位 | p90 | 最大 |",
    "|------|------|------|-----|------|",
    stat_line("左手 PICO 观测帧占比", summary["observed_left"]),
    stat_line("右手 PICO 观测帧占比", summary["observed_right"]),
    stat_line("臂跟踪通过率（观测帧，取较差一侧）", summary["arm_pass_min_side"]),
    stat_line("手根位置误差 p95（mm，取较差一侧）", summary["pos_p95_mm"], lambda v: f"{v:.2f}"),
    stat_line("手指骨方向误差中位数（°，取较差一侧）", summary["hand_bone_median_deg"], lambda v: f"{v:.1f}"),
    stat_line("掩码 PICO 兜底帧占比", summary["mask_fallback_fraction"]),
    stat_line("其中配准到 SAM 或沿用平移的兜底帧", summary["mask_fallback_registered_fraction"]),
    stat_line("SAM 前臂探针命中率（取较差一侧）", summary["sam_forearm_hit_rate_min_side"]),
    stat_line("移除掩码占画面比例", summary["removal_fraction"]),
    stat_line("移除区域中未被机器人盖住的比例", summary["removal_uncovered_fraction"]),
    "",
    f"未观测而补出的手部帧（插值 / 缓动 / 首尾保持，已在 `motion.npz` 的 `hand_source` 中标注）："
    f"左 {summary['filled_frames_total']['left']:,} 帧，右 {summary['filled_frames_total']['right']:,} 帧；"
    f"完全没有手部数据（`NONE`）：左 {summary['none_frames_total']['left']:,}，右 {summary['none_frames_total']['right']:,}。",
    "",
    f"用时：GPU 阶段合计 {summary['gpu_seconds_total'] / 3600:.1f} h，render 合计 {summary['render_seconds_total'] / 3600:.1f} worker·h；"
    f"final + review 视频共 {summary['video_bytes_total'] / 1e9:.1f} GB。",
    "",
    "## 4. 建议优先看的会话",
    "",
    "按单项指标挑出的最差几条，便于抽检：",
    "",
]
labels = {"fallback": "掩码兜底帧最多", "uncovered": "未被机器人盖住的修补区最多（ProPainter 模糊最显眼）",
          "arm_pass": "臂跟踪通过率最低", "observed": "手部观测帧占比最低（补帧最多）", "bone_med": "手指方向误差最大",
          "forearm": "SAM 前臂探针命中率最低（前臂可能没被完整移除）"}
for key, label in labels.items():
    items = "，".join(f"`{name}`（{value}）" for name, value in summary["worst"][key])
    lines.append(f"- {label}：{items}")
verdicts = {"OK": "正常", "LIMITATION": "已知限制", "RECOMPUTED": "有缺陷，下一轮重算",
            "FIXED": "上一轮的缺陷已修好", "STILL_DEFECT": "仍有缺陷"}
if checks:
    lines += ["", "## 5. 目检记录", "",
              "接手方逐条看过 review.mp4 的会话（原片 / 移除掩码 / Clean / 最终四个面板）。", ""]
    for run in sorted((k for k in checks if k != "note"), key=lambda k: int(k.lstrip("v") or 0), reverse=True):
        rows = checks.get(run) or {}
        if not rows:
            continue
        lines += [f"{run}：", "", "| 会话 | 结论 | 说明 |", "|------|------|------|"]
        lines += [f"| `{name}` | {verdicts.get(v, v)} | {note} |" for name, (v, note) in sorted(rows.items())]
        lines.append("")
lines += [
    "",
    f"## {6 if checks else 5}. 读视频时要知道的",
    "",
    "- 机器人臂比人前臂粗，手臂靠近镜头时上臂会占掉画面边缘一大块，这是几何上的真实结果。",
    "- 移除区域里没被机器人盖住的部分由 ProPainter 补出，大面积时会模糊、拖影；手臂整段静止、背景从没露出过的区域"
    "（例如 0902 扑克里一直放在画面左下角的左前臂）会补成灰色的纹理块，机器人臂外侧可能露出一部分。",
    "- 腕部 tracker（腕带 + 白色圆盘）依赖 PICO 腕带体积 + 颜色生长去除，PICO 腕点漂移大时偶尔还会留下少量残影。",
    "- 手持物体（扑克牌、薯片）按 2D 距离归属决定画在机器人前面还是后面，不代表真实的物理遮挡。",
    "",
]
open(sys.argv[3] if len(sys.argv) > 3 else "40_VIDEO_INDEX_ZH.md", "w").write("\n".join(lines))
