"""Generate concise closeout navigation; preserve historical evidence/packets."""
from pathlib import Path
from chaoyang.governance.common import REPO_ROOT, atomic_write as write_bytes


def atomic_write(path, text):
    write_bytes(path, text.encode('utf-8'))

BASE = '../../_run/current/four_stream_completion_20260928/attempts/attempt_0001'
FINAL = BASE + '/final_closeout_20260928'
NOTICE = '本轮任务已结束，部分质量目标未达成。仅交付已有成果，不再自动续跑；重新研究须新授权与新任务。'
LINKS = f'''
## 当前证据与观看

- [最终结束清单]({FINAL}/RESULT.json)：任务关闭与质量采用分开；质量采用0，人工审阅未完成。
- [逐会话交付、取消与文件校验]({FINAL}/DELIVERY_INVENTORY.json)
- [观看说明]({FINAL}/先看这里.md)
- [工程回归]({FINAL}/ENGINEERING_CHECKS.json)
- [历史检查点8]({BASE}/checkpoints/PROGRESS_0008.json)：历史事实保留，其后续研究建议已停止。
- [生成状态](STATUS.json)、[支线1/Clean](V5_SCENE.md)、[支线2/PICO＋MANUS](V5_SENSOR.md)、[支线3/HaWoR](V5_MOTION.md)、[HuRo终态](V5_HURO.md)。
'''
SCOPE = '''
## 指定范围与终态

| 支线 | 会话 | 结论 |
|---|---|---|
| Exact78/Clean | 0902 Chips103、Cards042 | 数值适配工程修复保留；语义残留/伪影仍失败，不采用 |
| PICO＋MANUS | 0916 Cards097、098、101 | 保留语义/坐标诊断；真实腕骨架贴合未验证，用户结束研究 |
| HaWoR→Robot | 0915 Chips007、Cards031 | 失效传播修复和全片账本保留；无合格新原场景基线 |
| HuRo | 007、031历史研究 | 保持REJECTED_QUALITY，不采用，不再求解 |

七会话现有视频仅作历史候选/诊断。未生成的新质量全片、阶段视频、同步对照已取消，不用旧片或原图补数。
合格Robot替换覆盖未获认证；007/031最新资格账本全帧回退，不等于成功替换。没有新增Depth/Object6D/Contact质量采用。
用户结束研究的指令不代表人工视觉验收通过，也不证明物理精度、训练资格或上机安全。
'''
CARDS = {
    'V5_SCENE.md': '''# 支线1／共享Clean：本轮结束，不采用质量候选

C01数值范围适配已修复并接入实际生产重放；保留工程修复，不宣称Clean视觉质量通过。
C02三候选及方法分析均保留，残留/伪影问题未解决；本轮REJECTED_QUALITY，无第四候选。
固定会话：get_potato_chips_0902_103、play_cards_0902_042；共享Clean亦服务0915的007/031。
Poker76–91诊断为16帧、固定4FPS短窗，不能称全片或采集时长。
旧Robot视频是历史未采用候选，不含本轮修复。新合格全片和完整阶段展示取消。
不得从视觉生成背景推导几何/Contact真值；Attachment不能重新证明身份或接触。
''',
    'V5_SENSOR.md': '''# 支线2／PICO＋MANUS：诊断交付，贴合未达成

固定会话：play_cards_0916_097、098、101；不新增读取102/103。
保留坐标链、侧别、MANUS节点和同名点可观测性审计及097语义图片。
controller原点、可见外壳点、估计腕点、MANUS虚拟根不是同一物理点。
controller_to_wrist_v1现有数值只是未独立验证的采集先验；没有冻结可信的物理腕贴合配置。
历史466帧显示修正版仅证明显示结构，不证明图像腕/骨架对齐。夹合/碰撞历史失败不被改写为通过。
本轮用户取消后续研究，保留UNVERIFIED/NOT_ADOPTED；无需新增采集才能办理本轮结束。
''',
    'V5_MOTION.md': '''# 支线3／HaWoR→Robot：工程修复保留，产品质量未采用

固定会话：get_potato_chips_0915_007(378帧)、play_cards_0915_031(149帧)。
031失效传播、坏目标拒绝与原图回退已有生产证据；不能用中性位、保持、对侧复制制造动作。
两条全片资格账本已由登记的operation实际执行。独立资格证据不足导致全帧原图回退，product_completed=false。
回退覆盖不等于Robot替换成功；历史原场景候选仅作失败证据。本轮REJECTED_QUALITY，新合格全片/阶段视频取消。
既有20mm/15°门不降低，失败不能据此简单称物理不可达；数字资产碰撞仍不能作为物理安全认证。
连接件CAD/网格已存在，不能再写成没有CAD；安装/投影/物理精度未因此自动验证。
''',
    'V5_HURO.md': '''# 支线4／HuRo：终止，不采用

官方核心、独立环境、007/031研究及固定失败窗证据已存在；不再重建环境、求解、调参或做新对比视频。
本轮之前已经REJECTED_QUALITY / NOT_ADOPTED，原终态收据保持不变。
旋转目标冲突与约束未满足，不能将内部目标下降或posthoc clip称作质量改进。
未调用的腕局部适配器仅有测试证据，不代表真实序列修复。
没有符合基本约束的新公平比较，不能宣称HuRo优于或劣于本地产品基线。
''',
}


def main():
    root = REPO_ROOT / 'docs/current'
    overview = '# Chaoyang 本轮最终收尾\n\n' + NOTICE + '\n' + SCOPE + LINKS
    for name, title in [('README_ZH.md', ''), ('PLAN.md', ''), ('COMPLETION_20260928_ZH.md', '')]:
        atomic_write(root / name, overview)
    for name, content in CARDS.items():
        atomic_write(root / name, content + '\n' + NOTICE + '\n' + LINKS)
    atomic_write(root / 'NEXT_ACTIONS_ZH.md', '''# 非活动研究建议，不是续跑入口

本轮已由用户结束。无本轮可执行下一任务，不得复活旧任务包、检查点或旧worker。
未来如重新授权，应独立登记：Clean语义证据、物理同名点/安装可观测性、运动消费者资格与装配遮挡证据。
以上仅研究方向，没有自动执行权；历史候选次数保留，不能改名重置。HuRo不在后续默认范围。
''' + LINKS)
    atomic_write(root / 'AI_WORK_ENTRY_ZH.md', '''# AI工作入口：先确认本轮已结束

1. 阅读AGENTS与本目录README，核验生成STATUS与当前receipt。
2. 本轮completion任务全部终态，当前任务包索引不提供其执行路由；不得沿历史next_action续跑。
3. 结果事实以最终结束清单及不可变前驱为准，不把任务关闭等同质量通过。
4. 其他AI任务、维护目录和数据原件保持不动。新任务必须有新授权、范围、writer与登记。
''' + LINKS)
    atomic_write(root / 'GIT_DELIVERY_ZH.md', '''# Git交付

功能分支：`task/four-stream-completion-20260928`；目标仓库：lemon977/ego-robot。
检查点9d2e2ba已从Windows可信SSH身份推送并回读确认。main未修改，未force push。
最终收尾提交、Windows文件校验与最终push的实际结果记录在最终目录POST_PUBLICATION_DELIVERY.json（发布后生成）。
该记录未出现前，不能据检查点推送声称最终关闭提交已推送。
视频、权重、环境、缓存不进入Git；其他AI的维护目录及新增天机文档不纳入本轮提交。
''' + LINKS)


if __name__ == '__main__':
    main()
