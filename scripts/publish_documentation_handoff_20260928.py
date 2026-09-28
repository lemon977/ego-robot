"""One-shot, documentation-only consolidation; never starts an algorithm task."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from chaoyang.governance.common import (
    REPO_ROOT, RECEIPT_PATH, TASK_STATE_PATH, AUTHORITY_PATH,
    load_json, artifact_ref, atomic_json, atomic_write, publish_bundle, now_iso,
)

EXPECTED_TASK = "human_to_robot_shared_hand_delivery_20260924"
ARCHIVE_REL = "docs/archive/navigation_20260928"
DOCUMENTS = {}
DOCUMENTS["docs/current/README_ZH.md"] = """# Chaoyang 当前工作总入口

文档整理日期：2026-09-28。最新算法终态仍为2026-09-24的共享手交付任务；本次仅整理导航和提交工程快照，不启动算法、不提升质量。

1. [当前状态与授权边界](PLAN.md) · [机器状态](STATUS.json)
2. [支线1 Scene/Clean](V5_SCENE.md) · [支线2 Sensor](V5_SENSOR.md) · [支线3 Motion/Robot](V5_MOTION.md) · [支线4 HuRo](V5_HURO.md)
3. [最新不可变结果与视频导航](SHARED_HAND_DELIVERY_RESULT_ZH.md)
4. [后续任务建议，尚未登记执行](NEXT_ACTIONS_ZH.md)
5. [历史计划关系与全部任务索引](../plans/INDEX_ZH.md)
6. [AI接手步骤](AI_WORK_ENTRY_ZH.md) · [Git快照与复现边界](GIT_DELIVERY_ZH.md)

四产品结构4/4，技术质量0/4，采用0/4；合格Kai22 H50窗口0。四产品指007、031、0902_103、0902_042，不是四条支线完成率。当前没有算法活动任务。

旧 `AI1/AI2/AI4/EXACT78` 名称只保留跳转；`V5_*.md` 文件名为兼容链接保留，正文已汇总至最新终态。不要根据文件名、mtime或目录中任务包的数量判断当前任务。

原始数据清洗与Kai22标签质量是两项工作：[0911/0914/0915清洗基线](DATA_CLEANING_0911_0915_ZH.md)已经封账，不重启处理队列。
"""
DOCUMENTS["docs/current/PLAN.md"] = """# 当前终态与后续工作边界

最新算法任务：`human_to_robot_shared_hand_delivery_20260924`，2026-09-24终态 `REJECTED_QUALITY`。本页不是新的算法执行授权。

- 当前活动算法任务：无；执行依据仍为[任务索引](../../tasks/current/INDEX.json)。
- 四产品：结构4/4、技术质量0/4、采用0/4。
- 开发Kai22合格H50窗口：0。
- 训练、控制真值、物理部署和外部公制精度资格均未取得。
- 最新三条增量是Robot固定窗、Sensor097局部手映射和Clean实现定位；HuRo沿用已有冻结比较。

[最新结果](SHARED_HAND_DELIVERY_RESULT_ZH.md)记录实物和失败；[四线用途及验收边界](HUMAN_TO_ROBOT_BASELINE_V1_ZH.md)记录长期目标；[下一任务建议](NEXT_ACTIONS_ZH.md)只有经过新的有限任务登记才能执行。

2026-09-28获授权的工作仅为文档整理和GitHub提交。没有恢复旧终态任务，没有改动模型、质量门、输入、processed或历史RESULT。
"""
DOCUMENTS["docs/current/AI_WORK_ENTRY_ZH.md"] = """# 后续AI接手步骤

1. 在项目内设置TMP、缓存与输出；运行 `PYTHONPATH=src /usr/local/bin/python -B -m chaoyang.cli validate-governance`。
2. 读取[总入口](README_ZH.md)、[机器状态](STATUS.json)、[当前任务索引](../../tasks/current/INDEX.json)和receipt绑定的[权威登记](../governance/DOC_AUTHORITY_MAP.json)。
3. 仅阅读本线短卡和其中明确的最新RESULT/STATE。需要追溯再查[历史索引](../plans/INDEX_ZH.md)，不要从旧计划恢复命令。
4. 当前无活动算法任务。[后续建议](NEXT_ACTIONS_ZH.md)不等于执行授权；获得授权后登记有限任务、冻结输入/代码/环境/门槛并取得独立写入范围。
5. 发布前核对PID/startticks/epoch/fencing；不得热改其他执行者、复活历史包或覆盖封存结果。
6. 报告执行、结构、质量、改善、审阅与采用六个独立状态；视频存在与CPU测试PASS不是产品PASS。

旧AI1=Sensor，AI2=HaWoR/Motion，AI4=HuRo；支线1现为共享Scene/Clean。CPU维护B与原始数据清洗不是额外算法支线。

仅在原H20上能直接解析历史绝对路径；GitHub克隆不包含模型、视频、processed、环境和运行证据实体。见[Git交付说明](GIT_DELIVERY_ZH.md)。
"""
DOCUMENTS["docs/current/HUMAN_TO_ROBOT_BASELINE_V1_ZH.md"] = """# Human→Robot 目标、基线与能力边界

本页仅定义共同路线；最新状态见[总入口](README_ZH.md)，不再累积逐轮运行日志。

| 支线 | 最终目标 | 可复用的当前成果 | 未完成的质量条件 |
|---|---|---|---|
| Scene/Clean | 原场景去人、保护任务物体、合法遮挡背景 | 四会话真实模型候选，031开发级深度/可见几何 | 合格Clean、全会话可靠遮挡与严格Contact |
| Sensor | PICO腕与MANUS骨架贴合，并导出合格局部手动作 | 三会话466帧后端与回放；097方向改善 | 腕贴合独立验证、夹合、碰撞和标签资格 |
| Motion/Robot | 实际q/FK驱动机械手、连接件、机械臂替换原人手 | 四会话正式候选、连接件颜色/深度/部件ID、签名resume | 输入异常、解跳变、腕跟随、最终原场景质量 |
| HuRo | 同输入、同资产与同场景公平比较 | 官方核心适配、全片求解与独立FK评价 | 原q限位、旋转、碰撞与产品质量 |

长期产品四会话仍为0915_007、0915_031、0902_103、0902_042。后来的代表会话Chips0915_042是另一会话，不能替换007的验收分母。Sensor为0916_097/098/101；102/103不在本路线读取范围。

Clean生成像素仅用于离线视觉，不能作为Depth/Object6D/Contact真值。Attachment不能反向证明身份或接触。可见表面与完整Object6D、开发安装与真实标定、运动学自洽与真实贴合必须分开。

CAD已存在并进入渲染；实测安装、TCP及相机到机器人基座的物理标定仍不能由视觉模型替代。031的inferred表示模型预测，不自动等于时序补帧，也不是独立观测真值。

旧逐轮正文按原字节保留于[文档归档](../archive/navigation_20260928/README_ZH.md)。旧不可变RESULT、Task Packet、视频和SHA引用原位置均不移动。
"""
DOCUMENTS["docs/current/V5_SCENE.md"] = """# 支线1：Scene / Clean / 几何

当前：无活动任务；没有合格Clean产品。本卡汇总至2026-09-24共享手终态，文件名V5仅为链接兼容。

## 已完成

- R2真实执行007/031/0902_103/0902_042全片ProPainter候选；不是未调用模型。
- S1完成031全149帧FoundationStereo及三张牌的可见表面/中心/平面方向，不等于隐藏完整位姿。
- 后续核验写入区、保护区、模型内部、传播与回贴；局部保护修复不等于视觉改善。
- 最新Poker76–91帧完成LaMa V2/V3真实推理，定位到输出域适配错误。

## 最新失败与下一步

已处于0..255的LaMa输出被再次乘255，出现饱和白洞。这是实现无效，LaMa方法质量尚未评估；不是已经证明该方法质量失败。现任务修复预算已耗尽，未扩171帧。

新授权后仅修输出×255一项，冻结其他条件，重跑同16帧并与Raw/旧候选同帧比较。必须核验mask外逐像素不变、物体保护、残留与时序；未过固定窗不得扩全片。007残留与供体覆盖问题另立任务，不与Poker混算改善。

## 权威证据

- [最新Clean终态](../../_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/clean/POKER_076_091_LAMA_RESIDUAL_V3/TERMINAL_RESULT.json)
- [最新状态](../../_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/clean/STATE.json)
- [031几何与Contact历史分会话终态](../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/geometry_contact/SESSION_TERMINALS.json)
- [视频与结果](SHARED_HAND_DELIVERY_RESULT_ZH.md) · [历史轮次](../plans/INDEX_ZH.md)

严格Contact/R1未取得可用成果；同会话不同图像域和短窗/全片结论不得互相继承。新实验不得阻断其他无此依赖的支线。
"""
DOCUMENTS["docs/current/V5_SENSOR.md"] = """# 支线2：PICO / MANUS / 局部手动作

当前：无活动任务。三会话回放属于既有结果，最新共享手修复只执行097；未扩098/101。

## 实际成果

097/098/101共466帧已经进入共同target-builder、solver和Kai22 FK，并有等比例回放。最新097共165帧、330侧帧，局部方向P50约118.58°降至13.48°，左侧方向165/165通过。

## 未通过

最新夹合门0/330、声明非相邻手部自碰撞门0/330；合格局部q22标签与H50窗口均0，interface_pass=false。旧任务的接口前向成功、旧碰撞结论不得移植给新映射。

方向改善不证明PICO腕已经贴合真实视频；3D有效、出画面与源无效分开。无独立腕中心证据时不得报告真实标定精度。

## 后续任务边界

先基于最新失败数组和冻结局部目标区分MANUS节点语义、方向映射、夹合目标与碰撞约束，再设计一个可证伪候选；不可只调平滑或缩放。097通过全部预定门后才讨论098/101回归；不拟合098、不调101，不读取102/103。腕图像贴合与局部手标签采用分别验收。

- [最新结果](../../_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/hand_data/RESULT_REPAIR2.json)
- [最新状态](../../_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/hand_data/STATE.json)
- [既有三会话显示修正结果](../../_run/current/human_to_robot_sensor_display_correction_20260923/attempts/attempt_0001/RESULT.json)
- [后续建议](NEXT_ACTIONS_ZH.md) · [历史](../plans/INDEX_ZH.md)
"""
DOCUMENTS["docs/current/V5_MOTION.md"] = """# 支线3：HaWoR / Robot / 原场景产品

当前：无活动任务，质量未通过。原始q/FK与历史失败保持不变。

## 已完成

四会话原场景候选、连接件颜色/深度/部件ID、正式入口与同签名resume已经执行。007双侧378帧；031全149帧时间轴、右102有效、左无合法输出。后续Chips0915_042和Poker0902_042有363/171帧第三人称回放，但不是新的合格原场景产品。

## 需要分开处理的问题

- 031第47→48帧约1.286m跳变在HaWoR源腕中已存在；第47帧贴底22×12px ROI。不能归咎renderer或删帧掩盖；没有证明物理不可达。
- Poker与后续Chips还有求解分支/初始化导致的跳变。旧平滑和连续性修复曾恶化腕目标门，未采用。
- 最新Poker80–111帧联合轨迹四块中三块可行，右93–95仍超过20mm/15°门；搜索未找到合格窗，不证明不存在可行解；没有扩全171帧或把三块拼成成功全片。

## 后续任务边界

Poker冻结目标、装配与评价器，只针对失败块的约束求解及边界连续性提出新候选。031作为独立上游证据任务，不能混入Poker的求解改善率。通过固定窗后才能扩片，并同时检查幅度、延迟、限位、碰撞和原场景合成。

- [最新Robot状态及失败帧](../../_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/robot/STATE.json)
- [最新固定窗数值](../../_run/current/human_to_robot_shared_hand_delivery_20260924/attempts/attempt_0001/lanes/robot/POKER_080_111_JOINT_TRAJECTORY_FIX1.json)
- [最新视频导航](SHARED_HAND_DELIVERY_RESULT_ZH.md)
- [四产品正式候选历史](visuals/HUMAN_TO_ROBOT_S2/INDEX_ZH.md)
"""
DOCUMENTS["docs/current/V5_HURO.md"] = """# 支线4：HuRo核心公平比较

当前：无活动任务；最新共享手任务未新跑HuRo。下述冻结比较不能用于后来更换输入的代表会话。

## 已执行与未采用

官方核心适配和腕目标测试、007/031全片求解已经执行。后续用同一固定URDF独立重算Local/HuRo FK；不是首次接入任务。

原HuRo硬限位失败007为230/756侧帧、031为84/102；共同运动学有效526与18侧帧用于残差统计，原分母不得删去。007该交集内HuRo完整旋转P50约32°，仍不合格。

固定16帧窗口派生有界q消除了该窗口的限位问题，但属于后处理研究，不替换原HuRo，不代表全片、碰撞、旋转或产品通过。没有方法胜者。

## 后续任务边界

新授权后在官方核心适配层检查原始求解限位与完整腕旋转，保存原q和派生结果。固定输入、资产、安装、相机、时间轴、掩码和评价器；优先失败短窗，不直接重跑全片。Local新增输入不得与旧HuRo直接比较。

- [同输入独立FK结果](../../_run/current/human_to_robot_10h_delivery_20260924/attempts/attempt_0001/lanes/compare/independent_fk_v1_runtime_fix1/RESULT.json)
- [窗口派生修复](../../_run/current/human_to_robot_four_lane_quality_increment_20260924/attempts/attempt_0001/lanes/compare/huro_limit_repair_window_v1/RESULT.json)
- [历史与视频索引](../plans/INDEX_ZH.md)
"""
DOCUMENTS["docs/current/NEXT_ACTIONS_ZH.md"] = """# 下一阶段任务建议（未授权、未登记、未执行）

本次用户授权仅为文档整理和GitHub提交。以下是后续算法任务拆分，不自动创建任务或占用GPU。

| 优先 | 独立任务 | 固定起点 | 验收/停止线 |
|---|---|---|---|
| P0 | Clean输出域最小修复 | Poker76–91、最新LaMa V3输入/模型/mask | 仅移除输出×255；数值域、mask外不变、物体保护、真实16帧审阅。固定窗失败不得扩171 |
| P1 | Robot约束轨迹 | Poker80–111、最新FIX1失败右93–95 | 20mm/15°等现行门不变；边界连续、幅度/延迟、限位与碰撞共同评估；不拼成功块冒充全片 |
| P1 | Sensor局部映射 | 097共享手REPAIR2与冻结MANUS节点映射 | 定位夹合与声明碰撞全失败；方向改善不得单独采用。通过后再决定098/101回归 |
| P1 | 031上游腕证据 | 第47/48帧ROI及全149时间轴、右102有效 | 对源输出异常提出可证伪原因；保留缺侧与异常帧，不用下游强拉腕替代证据 |
| P2 | HuRo约束核心 | 007/031冻结共同输入及原q | 先修短窗原始限位/旋转；后处理与核心分名；共同输入变化时两方法同版本重评 |

每项启动前必须取得新授权并登记：owner/write-set、实际base commit、输入/代码/环境/资产签名、候选与运行次数预算、时间截止、回归帧和有限停止线。具体资源预算由新任务确定，不能继承已过期8小时/12小时授权。

Scene、Sensor、运动和HuRo的独立前置分开；无关质量失败不阻止合法独立工作。共享实现只由唯一publisher在安全点集成；执行中不得热换current引用。

交付最小集合：RESULT/STATE、逐项质量指标、失败帧、真实视频（注明新跑/复用、短窗/全片）、复现命令、下一步。不得通过放宽门或改分母宣布成功。
"""
DOCUMENTS["docs/current/GIT_DELIVERY_ZH.md"] = """# Git快照、运行证据与复现边界

本次整理保存截至2026-09-28的既有工程改动和文档，不代表算法采用，不训练模型。

## 提交范围

源码、测试、配置/合同、文档、任务包及小型治理收据受Git管理。模型权重、环境、缓存、原始/processed数据、运行视频及完整attempt实体不提交；现有大型第三方资产仍按既有规则排除。当前分支不强推，不自动合并main。

GitHub上的代码与计划可读；历史收据引用的绝对路径指向H20，不能据此声称任意新克隆能离线重放所有实验。远端已有资产由清单/收据确认，不能因缺文件自动联网补模型。

## 状态解释

HEAD/提交记录用于代码版本；治理revision用于发布事实；Task Packet revision用于冻结任务。它们不要求数值相等。提交后的HEAD变化不会改写旧收据中的执行时commit。

当前算法事实从[STATUS](STATUS.json)和[最新结果](SHARED_HAND_DELIVERY_RESULT_ZH.md)读取。只备份Git不等于备份运行数据；忽略的archive和_run需要单独受控保管。

## 恢复文档

[归档说明](../archive/navigation_20260928/README_ZH.md)与MANIFEST记录整理前原字节及SHA。恢复正文后必须重新经过publisher发布新revision，不能倒写旧receipt。历史Task Packet/RESULT不搬迁、不改写。
"""
DOCUMENTS["docs/current/visuals/README_ZH.md"] = """# 可视化入口与用途

先看[最新共享手结果与视频](../SHARED_HAND_DELIVERY_RESULT_ZH.md)：最新Robot只有固定窗数值，171帧视频为旧结果复用；Clean是16帧实现无效对照，不是合格效果。

- [四产品原场景候选](HUMAN_TO_ROBOT_S2/INDEX_ZH.md)：有连接件与正式合成，质量未通过。
- [15槽位全链审阅](HUMAN_TO_ROBOT_10H_DELIVERY/INDEX_ZH.md)：阶段/对照/复用视频，不代表15项质量通过。
- [代表会话全机械臂](HUMAN_TO_ROBOT_REPRESENTATIVE_BASELINE/INDEX_ZH.md)：第三人称运动诊断，不替代原场景Robot。
- [历史计划索引](../../plans/INDEX_ZH.md)：查旧视频对应任务和当时结论。

画面完整解码、来源SHA和视觉/算法质量独立验收。AI固定样本审阅不等于用户全片确认。所有真实媒体在H20项目内；GitHub不包含视频本体。
"""

def ensure_idle(status, state, index):
    if status.get("latest_task") != EXPECTED_TASK:
        raise RuntimeError("LATEST_TASK_CHANGED")
    if status.get("active_tasks") or state.get("next_task") is not None or index.get("task_packets"):
        raise RuntimeError("ACTIVE_TASK_OR_ROUTING_EXISTS")
    if any(t.get("pid") or t.get("status") in {"RUNNING", "READY", "PENDING", "WAIT_GPU_RESOURCE"} for t in state["tasks"]):
        raise RuntimeError("WRITER_OR_PENDING_TASK_EXISTS")

def relative_ref(ref):
    if not ref or not ref.get("path"):
        return None
    p = Path(ref["path"])
    if not p.is_absolute():
        p = REPO_ROOT / p
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True, type=int)
    args = parser.parse_args()
    run = REPO_ROOT / "_run/current/documentation_handoff_20260928/attempts/attempt_0001"
    run.mkdir(parents=True, exist_ok=True)
    for parent in [run, *run.parents]:
        if parent.is_symlink():
            raise RuntimeError("SYMLINK_OUTPUT_REFUSED")
    with (run / "publisher.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = load_json(RECEIPT_PATH)
        if receipt["governance_revision"] != args.expected_revision:
            raise RuntimeError("GOVERNANCE_CAS_MISMATCH")
        status = load_json(REPO_ROOT / "docs/current/STATUS.json")
        state = load_json(TASK_STATE_PATH)
        ensure_idle(status, state, load_json(REPO_ROOT / "tasks/current/INDEX.json"))
        original_result = artifact_ref(Path(status["result"]["path"]))
        if original_result != status["result"]:
            raise RuntimeError("LATEST_RESULT_REF_MISMATCH")
        archive = REPO_ROOT / ARCHIVE_REL
        if archive.exists() or (run / "PUBLICATION.json").exists():
            raise RuntimeError("IMMUTABLE_DOCUMENTATION_SNAPSHOT_ALREADY_EXISTS")
        archive.mkdir(parents=True)
        writer = {"pid": os.getpid(),
                  "proc_start_ticks": int(Path(f"/proc/{os.getpid()}/stat").read_text().split(") ", 1)[1].split()[19]),
                  "executor_epoch": 1, "fencing_token": uuid.uuid4().hex,
                  "expected_revision": args.expected_revision,
                  "code": artifact_ref(Path(__file__)),
                  "scope": "DOCUMENTATION_ONLY_NO_ALGORITHM_ROUTING",
                  "started_at": now_iso()}
        atomic_json(run / "WRITER.json", writer)
        snapshots = []
        for rel in DOCUMENTS:
            src = REPO_ROOT / rel
            if src.exists():
                dst = archive / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                snapshots.append({"original_path": rel, "archive_path": dst.relative_to(REPO_ROOT).as_posix(),
                                  "bytes": dst.stat().st_size, "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})
        # Preserve immutable source snapshots for the generated historical task inventory.
        atomic_json(archive / "TASK_LEDGER_SNAPSHOT.json", state)
        atomic_json(archive / "MANIFEST.json", {
            "schema_version": "DOCUMENTATION_SNAPSHOT_V1", "source_revision": args.expected_revision,
            "created_at": now_iso(), "files": snapshots,
            "restore": "Copy only selected archived text to original_path, then publish a new governance revision; never overwrite immutable receipts.",
            "historical_links": "Archived text is byte-exact. Relative links retain their original source-file context; do not treat this folder as executable navigation."
        })
        atomic_write(archive / "README_ZH.md", (
            "# 2026-09-28 文档整理前快照\n\n"
            "本目录仅为历史证据，不授权运行。MANIFEST.json绑定原始路径、归档路径、bytes/SHA；正文逐字节保留。"
            "旧正文内相对链接按原始路径解释，不是归档目录下的新执行链接。\n\n"
            "恢复仅复制明确原始文件，再发布新的治理revision；不可覆盖历史RESULT或receipt。"
            "Task Packet和运行产物未移动。返回[当前入口](../../current/README_ZH.md)。\n"
        ).encode())
        for rel, body in DOCUMENTS.items():
            atomic_write(REPO_ROOT / rel, body.encode())
        history = []
        for t in state["tasks"]:
            history.append({k: t.get(k) for k in ("task_id", "status", "updated_at", "task_packet", "result")})
        atomic_json(REPO_ROOT / "docs/plans/TASK_HISTORY_20260928.json", {
            "schema_version": "READ_ONLY_TASK_HISTORY_V1",
            "source_revision": args.expected_revision,
            "claim_limit": "Historical snapshot; only tasks/current/INDEX.json authorizes execution.",
            "source": artifact_ref(archive / "TASK_LEDGER_SNAPSHOT.json"),
            "tasks": history,
        })
        intro = """# 计划关系与历史任务索引

整理日期2026-09-28；来源为治理revision14167的冻结任务账本，不按mtime猜最新。
这是一份历史导航，不是执行队列。当前执行权只来自[任务索引](../../tasks/current/INDEX.json)。

## 读法

- V2/V3：早期工程、原场景候选与投影诊断。
- V4 → Recovery → Takeover：运行故障恢复/接管；后被共享路线替代，不能续跑。
- V5：共享Scene、Motion、Sensor、HuRo路线起点；四短卡保留V5文件名用于链接兼容。
- R1 → R2 → S1 → S2：完成真实模型、局部几何、正式装配合成及证据链，但未取得产品采用。
- convergence / product_first / source_coverage及007/031专项：有限诊断和局部候选；PASSED只适用于声明范围。
- 20260924的10h / increment / representative / quality_acceptance / result_breakthrough：不同范围的后继，不是一份任务不断延长。
- 最新算法终态shared_hand_delivery：Robot、局部手标签、Clean三条增量；HuRo未新增求解。

docs/plans下旧00_EXECUTION等正文是对应任务的冻结设计，不因存在文件而可执行。
旧任务包仍在tasks/current物理容器以保护引用；目录名称不授予执行权。
下表列出账本全部任务，旧等待/阻塞状态也不代表当前队列；机器快照见[TASK_HISTORY_20260928.json](TASK_HISTORY_20260928.json)。

返回[当前总入口](../current/README_ZH.md) · [下一任务建议（未执行）](../current/NEXT_ACTIONS_ZH.md)。

| 历史任务 | 记录状态 | 最后记录 | 任务包 | 结果 |
|---|---|---|---|---|
"""
        rows = []
        for t in history:
            refs = []
            for key in ("task_packet", "result"):
                rel = relative_ref(t.get(key))
                refs.append(f"[{key}](../../{rel})" if rel else "未绑定")
            rows.append(f"| {t['task_id']} | {t['status']} | {t.get('updated_at') or '未记录'} | {refs[0]} | {refs[1]} |")
        atomic_write(REPO_ROOT / "docs/plans/INDEX_ZH.md", (intro + "\n".join(rows) + "\n").encode())
        # Keep old entry filenames, remove stale S1-only summaries.
        for name, target, label in [("AI1.md", "V5_SENSOR.md", "旧AI1 / Sensor"),
                                     ("AI2.md", "V5_MOTION.md", "旧AI2 / Motion"),
                                     ("AI4_HURO.md", "V5_HURO.md", "旧AI4 / HuRo"),
                                     ("EXACT78.md", "V5_SCENE.md", "旧Exact78 / 共享Scene")]:
            # Old texts are additionally snapshotted before replacement.
            src = REPO_ROOT / "docs/current" / name
            dst = archive / "docs/current" / name
            shutil.copyfile(src, dst)
            snapshots.append({"original_path": src.relative_to(REPO_ROOT).as_posix(),
                              "archive_path": dst.relative_to(REPO_ROOT).as_posix(),
                              "bytes": dst.stat().st_size, "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})
            atomic_write(src, f"# 历史入口：{label}\n\n当前阅读[{label}]({target})及[总入口](README_ZH.md)。旧训练计划与命令不授权执行；历史见[索引](../plans/INDEX_ZH.md)。\n".encode())
        manifest = load_json(archive / "MANIFEST.json")
        manifest["files"] = snapshots
        atomic_json(archive / "MANIFEST.json", manifest)
        receipt = publish_bundle(load_json(AUTHORITY_PATH), state,
                                 event_type="DOCUMENTATION_HANDOFF_CONSOLIDATED_NO_ALGORITHM_CHANGE",
                                 expected_revision=args.expected_revision, generator_path=Path(__file__))
        after = load_json(REPO_ROOT / "docs/current/STATUS.json")
        if after["counts"] != status["counts"] or after["result"] != status["result"]:
            raise RuntimeError("UNEXPECTED_ALGORITHM_STATE_CHANGE")
        atomic_json(run / "PUBLICATION.json", {
            "schema_version": "DOCUMENTATION_HANDOFF_PUBLICATION_V1",
            "governance_revision": receipt["governance_revision"],
            "archive": artifact_ref(archive / "MANIFEST.json"),
            "previous_algorithm_result": original_result,
            "source_data_modified": False, "algorithm_executed": False, "gpu_used": False,
            "git_push": "NOT_YET_EXECUTED", "writer": writer,
        })
        print(json.dumps({"status": "DOCUMENTS_PUBLISHED", "revision": receipt["governance_revision"],
                          "historical_tasks": len(history), "archived_texts": len(snapshots)}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
