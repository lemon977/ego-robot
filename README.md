# Chaoyang Ego-to-Robot Pipeline

本仓库实现第一视角数据的 Raw → HaWoR / Mask → Stereo Depth → Object6D → Clean → Contact → Robot Visual → HumanEgo 流水线。

## 当前唯一入口

执行或回答状态问题前，按顺序读取：

1. [当前状态 receipt](docs/governance/CURRENT_STATUS_RECEIPT.json)：绑定同一 revision 的全部 current 文件。
2. [最小事实页](docs/governance/CURRENT_PROJECT_STATUS_MIN.json)：计数、活动任务与下一任务。
3. [当前基线注册表 V2](docs/governance/CURRENT_BASELINE_REGISTRY_V2.json)：12 阶段算法、代码、权重、schema、质量门、authority 和限制。
4. [当前阶段说明](docs/governance/CURRENT_STAGE_BASELINES_ZH.md)：给人看的算法与整改需求。
5. [端到端复现文档](docs/pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md)：稳定技术接口与复现顺序。

旧 README、聊天、目录名中的 `current/latest/final`、未绑定 SHA 的视频都不是当前事实。

## 目录职责

```text
chaoyang/
├── assets/       # 权重、Robot URDF/CAD与固定资产
├── contracts/    # 当前机器合同和schema
├── data/         # 小型manifest、fixture和必要项目内派生数据
├── docs/         # current治理、技术文档、报告和历史胶囊
├── HumanEgo/     # 视觉辅助/策略模型代码
├── pipeline/     # 可复用阶段实现
├── systems/      # 注册表驱动的阶段入口说明
├── tasks/        # current与不可变证据run
├── tests/        # 当前回归
├── third_party/  # 固定第三方实现及许可证
├── tools/        # current入口与治理/清理工具
├── archive/      # 尚未完成删除证明的历史胶囊
└── _run/         # GPU lease、lock和当前executor控制面
```

V6 已移除重复端到端文档和顶层深度专题包。清理采用逐目标门：与活动 Clean 无传递依赖、无 current 引用、无活动 FD/CWD且胶囊闭合的旧资产可立即删除；只有活动依赖本身需要等待。`NOW/` 已完成证据胶囊，剩余硬编码引用正在与对应旧工具一起退休。历史路径通过 `docs/governance/PATH_REDIRECTS.json` 解析。

## 不可破坏规则

- `/mnt/data/egodata` 的原始数据、0909/0910 release 与数据侧可视化永久保护；`/nas/chenxianchi` 的其他项目不触碰。
- current authority 只由 receipt 绑定的机器文件解释；生成的 current Markdown 禁止手改。
- 已有 final 不覆盖；不同输入、代码、权重、标定或 schema 签名不得静默复用。
- HaWoR 单目三维、Stereo/Object6D 内部残差、Robot 数字接触距离和视觉审核都不是物理真实精度。
- `visual_robot_trajectory_sidecar` 不是 `real_robot_action_sidecar`；Visual Aux checkpoint 不是最终 Policy checkpoint。
- Clean 合成像素不得反喂 Depth、Object6D、Contact 几何或动作真值。
- 永久删除前必须有 dry-run 清单、零 current 引用、零活动 FD/CWD、Git保护快照和删除收据。

第三方版本与许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

KaiHand—Tianji 法兰连接件的当前硬件参考包见 [assets/robot/hardware_handoff/kaihand_flange_adapter_v1/README_ZH.md](assets/robot/hardware_handoff/kaihand_flange_adapter_v1/README_ZH.md)。该包只有经 SHA 核验的参考几何与测量清单，当前没有可直接打印的连接件 STEP/STL。
