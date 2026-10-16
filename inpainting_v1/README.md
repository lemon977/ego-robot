# Inpainting Raw→Robot V1

本支线唯一采用基线为 **INPAINTING_RAW_TO_ROBOT_V1**，来源是 pick_place_1008 原版 V9 的 **199 会话、29,322 原生帧**生产链。用户于2026-10-12指定该版本作为后续优化起点；这是采用选择，不是重新渲染或全片质量晋升。

接手只读四项：[机器清单](MANIFEST.json)、[全链路与算法](PIPELINE_ZH.md)、[运行与恢复](RUN_ZH.md)、[历史隔离索引](../archive/inpainting_baselines/README.md)。完整SHA集中在机器清单和原收据，人类说明不另维护重复哈希。

算法清单保持原字节；[治理发布补充收据](GOVERNANCE_PUBLICATION_R02.json)只记录选用包装的兼容修复，不是第二条基线，也没有更换算法。最终核对见[验收记录](VERIFICATION_ZH.md)。

## 从原始到最终

原始VST RGB及时间映射 → 同会话Controller/Head＋MANUS动作 → Tianji双臂＋KaiHand22求解；原始RGB → SAM3.1去除／保护遮罩 → ProPainter Clean；最后两路进入完整Robot渲染和物体支持遮挡合成。

原始数据：`/path/to/private_storage/datasets/ego/pick_place_1008`。

原版199交付：[冻结说明](../review/pick_place_1008/ALL199_ORIGINAL_EXECUTION_OUTPUTS_UNIFIED_DELIVERY_V1/FROZEN_V9_BASELINE_FOR_TRAINING_HANDOFF_V1.json)、[逐会话文件清单](../review/pick_place_1008/ALL199_ORIGINAL_EXECUTION_OUTPUTS_UNIFIED_DELIVERY_V1/MANIFEST_ALL199_ORIGINAL_EXECUTION_OUTPUTS_V1.json)。数据已重定位到NAS；使用既有只读适配和清单导航，不写入或重新复制NAS数据。审阅从[统一导航](../review/NAVIGATION_ZH.md)进入。

## 已确定与尚有限制

- 已确定：原版199生产者身份、19份核心源码、原始清单、相机／安装配方、逐会话种子、模型、Robot资产及历史运行签名；运行源码仍唯一位于原`src/chaoyang/`路径，不新建可变副本。
- 这是离线、硬件辅助的视觉批量基线，不是纯RGB→HaWoR。原始逐会话种子和视觉拟合配置仍是依赖；新输入域不能无验证地套用。
- 199条完成渲染不等于199条视觉通过。原NUM、失败审阅、残留、遮挡缺陷、隐藏区域UNKNOWN和人工待审保持原样。
- Robot成片不是实机控制标签。Clean只用于离线视觉，不进入Depth、Object6D、Contact或控制真值；V1没有把旧几何算法纳入生产阶段。

## 接下来怎样优化

只以本清单为父版本，新建有限任务和新attempt，针对明确失败簇改一项并比较。候选不能自动替换V1；升级需显式采用收据。旧v13/V2/P1/0915/0918及C3/QP等不再作为本支线默认选择，不从最高版本号或latest选入口。

旧文档原字节为证据而保留，其正文中的“当前”只属于当时。治理协议仍保留的旧复现／Clean边界审计只用于解释历史限制，不能据此选择旧算法或增加本版生产阶段；当前选择以本清单及receipt绑定的新V1合同为准。

另一维护者的0916 MANUS v16、HumanEgo ICT两条训练及其旧checkpoint属于独立工作域，原数据、算法和命令不改。`baseline_v1/`是旧冻结依赖包，不是本轮唯一基线入口，也不设为PYTHONPATH。
