#!/usr/bin/env python3
"""Build Chinese meeting diagrams for depth semantics and accuracy evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(path: Path) -> dict[str, Any]:
    value = path.resolve(strict=True)
    return {"path": str(value), "bytes": value.stat().st_size, "sha256": sha256(value)}


def wrapped(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], text_font: ImageFont.FreeTypeFont, fill: str, spacing: int = 8) -> None:
    left, top, right, _ = box
    max_width = right - left
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for token in paragraph:
            candidate = current + token
            if current and draw.textlength(candidate, font=text_font) > max_width:
                lines.append(current)
                current = token
            else:
                current = candidate
        lines.append(current)
    draw.multiline_text((left, top), "\n".join(lines), font=text_font, fill=fill, spacing=spacing)


def box(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], title: str, body: str, color: str) -> None:
    draw.rounded_rectangle(rect, radius=24, fill="#ffffff", outline=color, width=5)
    x0, y0, x1, _ = rect
    draw.rounded_rectangle((x0, y0, x1, y0 + 78), radius=22, fill=color)
    draw.rectangle((x0, y0 + 52, x1, y0 + 78), fill=color)
    draw.text((x0 + 22, y0 + 15), title, font=font(30), fill="white")
    wrapped(draw, body, (x0 + 22, y0 + 98, x1 - 22, rect[3] - 16), font(23), "#17202a", 9)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], label: str = "") -> None:
    draw.line((start, end), fill="#456779", width=7)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        sign = 1 if ex > sx else -1
        draw.polygon(((ex, ey), (ex - sign * 22, ey - 13), (ex - sign * 22, ey + 13)), fill="#456779")
    else:
        sign = 1 if ey > sy else -1
        draw.polygon(((ex, ey), (ex - 13, ey - sign * 22), (ex + 13, ey - sign * 22)), fill="#456779")
    if label:
        mx, my = (sx + ex) // 2, (sy + ey) // 2
        draw.rounded_rectangle((mx - 90, my - 24, mx + 90, my + 24), radius=10, fill="#eef5f8")
        draw.text((mx - 78, my - 17), label, font=font(18), fill="#294b5a")


def depth_types(path: Path) -> None:
    image = Image.new("RGB", (2400, 1600), "#f5f8fb")
    draw = ImageDraw.Draw(image)
    draw.text((70, 45), "项目里五种不同的“深度 / 距离”", font=font(58), fill="#102a43")
    draw.text((70, 120), "它们来源、坐标、物理含义和授权范围都不同，不能统称为同一个 Depth", font=font(30), fill="#b42318")

    box(draw, (70, 220, 730, 590), "1  HaWoR Z", "输入：单目 selected-left RGB（1280×960）\n方法：学习模型预测 MANO pose / shape / root translation，再由 MANO forward 得到 Mesh 与21个关节\n含义：估计的手部 camera optical-Z；world版本只是逐帧乘 c2w\n边界：不是深度传感器，也没有 wrist/tip 外部真值", "#6f42c1")
    box(draw, (870, 220, 1530, 590), "2  FoundationStereo Z", "输入：同步左右鱼眼 → 去畸变/极线校正 → 640×480\n方法：预测视差 d，使用 Z=fB/d；当前 f=320 px，B=63.7717 mm\n含义：左相机当前可见表面的公制 optical-Z\n边界：遮挡后背景不可见；细边、反光、低纹理、模糊可能局部错误", "#007f8b")
    box(draw, (1670, 220, 2330, 590), "3  Object6D Z / near-far", "输入：原始RGB + 物体身份Mask + registered Stereo Depth + K/c2w\n方法：物体Mask内有效点云的稳健中心、主轴/几何约束与时序门\n含义：可见帧物体空间位置及近/远表面范围\n边界：不是另一台深度网络；遮挡/身份不确定帧 KEEP_INVALID", "#2e7d32")

    box(draw, (470, 815, 1130, 1190), "4  Robot depth", "输入：Robot CAD/URDF、关节状态和虚拟相机\n方法：渲染器 z-buffer\n含义：数字机器人模型在虚拟相机中的可见表面深度\n边界：它是合成几何；不能验证现实 Robot、TCP 或安装误差", "#c05a00")
    box(draw, (1270, 815, 1930, 1190), "5  Contact signed distance", "输入：KaiHand fingertip pad CAD + Object6D几何 + 同一坐标链\n方法：点/面或SDF signed distance与碰撞计算\n含义：数字模型中 pad 到物体表面的带符号距离\n边界：不是图像Depth；没有触觉/力/外部tracker时不是现实接触真值", "#b42318")

    arrow(draw, (730, 405), (870, 405), "共同camera域")
    arrow(draw, (1530, 405), (1670, 405), "Depth+Mask")
    arrow(draw, (1995, 590), (1640, 815), "物体几何")
    arrow(draw, (1190, 590), (800, 815), "手部动作")
    arrow(draw, (1130, 1000), (1270, 1000), "同坐标链")

    draw.rounded_rectangle((120, 1320, 2280, 1510), radius=30, fill="#fff4df", outline="#dd8b00", width=4)
    draw.text((165, 1350), "当前授权结论", font=font(34), fill="#8a4b00")
    wrapped(draw, "FoundationStereo 当前只授权 VISUAL_OBJECT6D_CANDIDATE_INPUT。Clean 中 ProPainter 合成的像素没有真实深度；Robot z-buffer 和 contact signed distance 也不能反向证明 Stereo 或 HaWoR 的现实精度。", (165, 1400, 2230, 1490), font(26), "#543000", 8)
    image.save(path)


def accuracy_boundary(path: Path) -> None:
    image = Image.new("RGB", (2400, 1700), "#f7f9fb")
    draw = ImageDraw.Draw(image)
    draw.text((70, 45), "当前精度证据：内部一致性 ≠ 外部真实精度", font=font(57), fill="#102a43")
    draw.text((70, 120), "只有独立物理真值才能回答“真实误差是多少毫米”", font=font(31), fill="#b42318")

    draw.rounded_rectangle((70, 210, 1160, 1460), radius=28, fill="#edf8f1", outline="#2e7d32", width=5)
    draw.text((115, 245), "已经测得：内部闭环 / 跨系统代理", font=font(38), fill="#1e6a36")
    internal = [
        ("Depth公式闭合", "冻结 disparity 重算 Z 的差约 2.5×10⁻⁷ m\n证明实现一致，不是测距误差"),
        ("Depth→RGB registration", "Chips034 median 0.131–0.149 px，P90≤0.522 px\nPoker042 median 0.130–0.174 px，P90≤0.608 px\n是图像特征内部对齐，不是外部控制点真值"),
        ("HaWoR↔Stereo表面", "Chips034右手 abs MAE 59.15 mm / P95 100.76 mm\n是两估计器差异，不能指定谁错"),
        ("Object6D内部几何", "平面/点云 residual、投影和时序限速\n证明拟合与约束工作，不是pose GT error"),
        ("Robot数字残差", "IK/FK 对自身target的mm/deg residual\n证明数字求解闭合，不是现实TCP或接触误差"),
    ]
    y = 330
    for title, body in internal:
        draw.ellipse((115, y + 7, 143, y + 35), fill="#2e7d32")
        draw.text((160, y), title, font=font(28), fill="#153f24")
        wrapped(draw, body, (160, y + 48, 1100, y + 180), font(23), "#26362b", 7)
        y += 215

    draw.rounded_rectangle((1240, 210, 2330, 1460), radius=28, fill="#fff0ef", outline="#b42318", width=5)
    draw.text((1285, 245), "尚未测得：外部物理真值", font=font(38), fill="#9e1c14")
    external = [
        ("FoundationStereo绝对Z", "30 / 50 / 70 / 100 cm 已知平面的 bias、MAE、P95、时序std：NOT MEASURED"),
        ("HaWoR手部3D", "wrist / fingertip camera-Z 与三维位置相对 MoCap/治具真值：NOT MEASURED"),
        ("Depth registration真误差", "独立标定板控制点上的pixel MAE/P95：NOT MEASURED"),
        ("Object6D pose", "相对AprilTag/MoCap/测量治具的平移与旋转误差：NOT MEASURED"),
        ("Robot实体链", "camera/world→base、flange→KaiHand、FK endpoint、重复安装分布：NOT MEASURED"),
        ("现实接触", "触觉/力/导电事件与pad→object signed distance同步误差：NOT MEASURED"),
    ]
    y = 330
    for title, body in external:
        draw.ellipse((1285, y + 7, 1313, y + 35), fill="#b42318")
        draw.text((1330, y), title, font=font(28), fill="#71150f")
        wrapped(draw, body, (1330, y + 48, 2265, y + 150), font(23), "#442521", 7)
        y += 178

    draw.rounded_rectangle((120, 1520, 2280, 1640), radius=24, fill="#172b3a")
    draw.text((165, 1545), "禁止：把 estimator disagreement、registration px、Object6D limiter、IK residual 和 mount proxy 直接相加或做 RSS，伪造成最终接触精度。", font=font(27), fill="white")
    image.save(path)


def run(output: Path) -> dict[str, Any]:
    destination = output.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"output is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    depth_path = destination / "DEPTH_TYPES_OVERVIEW_ZH.png"
    accuracy_path = destination / "INTERNAL_VS_EXTERNAL_ACCURACY_ZH.png"
    depth_types(depth_path)
    accuracy_boundary(accuracy_path)
    result = {
        "schema_version": "depth-accuracy-meeting-diagrams-result-v1",
        "status": "PASS",
        "artifacts": {path.name: evidence(path) for path in (depth_path, accuracy_path)},
        "claim_limit": "Diagrams summarize current evidence boundaries; no external ground truth or authority is introduced.",
    }
    result_path = destination / "RESULT.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args().output), ensure_ascii=False, indent=2, sort_keys=True))
