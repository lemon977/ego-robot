# V1唯一基线验收记录

本支线唯一采用为 **INPAINTING_RAW_TO_ROBOT_V1**：pick_place_1008 **原版199会话V9生产链，29,322原生帧**。不拿后续优化片替换原199，也不把最高版本号当当前。

## 唯一入口与全链路

[README](README.md) → [MANIFEST](MANIFEST.json) → [算法与阶段](PIPELINE_ZH.md) → [新任务运行说明](RUN_ZH.md)。

Raw VST RGB／原生时间映射 → 同会话Controller/Head＋MANUS → Tianji双臂＋KaiHand22；并行Raw → SAM3.1去除与保护Mask → ProPainter Clean；再完整Robot渲染、原场景遮挡合成和封存发布。实际operation仍保留兼容名称，执行必须新建有限Task Packet。

这是**硬件辅助离线批量视觉基线**，不是纯RGB／HaWoR，也不是陌生数据域无需种子的已验证通用入口。

## 实际核对

| 项目 | 结果 |
|---|---|
| 当前本支线基线选择 | 1条 |
| 原版核心算法 | 19/19 SHA一致，未改算法 |
| 当前运行源码 | 94/94 SHA一致；唯一选用包装修复有独立补充收据 |
| 原199会话种子 | 199/199 SHA一致 |
| 阶段依赖并集 | 421项；420项完整SHA核验；1个大权重沿原封存SHA＋当前文件大小，未重哈希 |
| 受保护引用 | 196项全部一致 |
| 历史隔离文件 | 13项原字节与SHA全部保留 |
| CPU回归 | 105项＋9个subtests通过；0失败、0跳过 |
| `make handoff` | PASS，默认只显示这条V1，19核心与94运行源码实际验SHA |
| 旧takeover命令 | 3条实际返回RETIRED_OPERATION，导入算法前拒绝 |
| 无有效生产Task的V1命令 | 实际返回EXACT78_GATE_DENIED |
| 两ICT原Bash预检 | Raw、Robotized均exit0；没有导入Torch／启动CUDA训练 |
| ICT新Task签名闭包 | 两路各50项，与当前投影一致 |
| 本次模型／渲染／训练／删除／移动／外部写入 | 全部0 |

ICT Raw缓存为618训练窗口／75验证窗口，Robotized为458／57；各179训练采集组＋20验证组，交集0。仅确认现有研究数据和命令未被整理破坏，不据此授予实机控制或新checkpoint资格。

## 历史如何处理

旧v13／V2／P1／0915／0918及优化候选退出本支线默认；[归档索引](../archive/inpainting_baselines/README.md)记录历史身份和保留原因。采用**逻辑隔离，不移动或删除**。原Task、原结果、SHA及V1现用共享依赖保持原路径，不制造第二份可变算法源码。

部分旧边界审计仍是协议要求的参考导航；正文中的“当前”属于历史时点，不能据此选旧基线。旧冻结JSON仍保留当年的两个分支计数，不等于今天本支线有两个默认。当前选择从receipt绑定的新V1合同和根MANIFEST读取。

0916 MANUS由另一维护者负责，不在本轮归档范围。ICT原数据、模型、训练代码与Bash均保留，旧已签名任务不热改，未来新任务正常重签治理包装。

## 质量边界与修复记录

**完成渲染、用户采用、视觉合格、完整产品质量、可训练标签、实机控制资格分别判断。** 本轮没有重渲染或重新解码199条全片；原数值FAIL、失败审阅、Clean残留、遮挡问题和隐藏区域UNKNOWN全部保留，不授予新的视觉合格。

沙箱进程身份、测试路径和首次新投影的必需审计导航检查中断均留下真实失败记录，没有改数值门或把失败改成PASS。导航发布兼容修复仅改选用包装，根MANIFEST保持原字节；[补充收据](GOVERNANCE_PUBLICATION_R02.json)绑定原／现包装SHA，不是第二条基线。

详细机器证据：[EVIDENCE](../_run/current/inpainting_v1_navigation_final_20261012/attempts/attempt_0001/EVIDENCE.json)；实际测试：[JUnit](../_run/maintenance/inpainting_v1_adoption_20261012/attempt_0001/FINAL_TEST_RESULTS.xml)。最终状态以[治理收据](../docs/governance/CURRENT_STATUS_RECEIPT.json)及[有限任务](../tasks/current/inpainting_v1_navigation_final_20261012/TASK_PACKET.json)输出为准。

## 后续

仅以本V1为父版本，为明确失败簇另开有限任务；每轮单项优化、保留A/B、通过后显式采用。不得自动选择旧命令、最新retry或旧媒体作新生产输入。
