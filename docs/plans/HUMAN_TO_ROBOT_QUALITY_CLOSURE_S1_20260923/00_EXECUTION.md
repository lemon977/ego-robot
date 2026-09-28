# Human→Robot 质量闭环 S1

本任务是已封存 R2 的有限 successor，不修改或复活 R2。它只推进 R2 中明确未完成的质量项，
不以候选存在替代质量通过。

## 固定范围

- Scene/Clean：`play_cards_0915_031`、`get_potato_chips_0915_007`、
  `get_potato_chips_0902_103`、`play_cards_0902_042`。
- Sensor：只消费已封存的 097/098/101 `KINEMATIC_ONLY` 结果，不重新拟合。
- Compare：只消费同一冻结 HandMotion、机器人资产、相机域和有效帧。
- 原始、processed、archive、sealed 与 R2 产物只读。

## 执行阶段

1. 在固定困难窗上独立核对设备/线缆像素证据、任务物体直接可见区域和现有 SAM/Mask
   依赖；输出逐像素来源与 UNKNOWN。没有新证据则终态化该分支，不重跑 R2 同签名候选。
2. 只有输入证据、配置或代码形成新签名且反例测试通过，才允许一套保守 Scene 修复；
   先小窗，再按同配方逐会话运行和质量审阅。
3. 盘点同会话现有 Depth/Object6D/有限表面资产。只在图像域、时间、K/baseline、尺度和
   产物 SHA 闭合时接入实片遮挡与 Contact；缺失保持 UNKNOWN。
4. Robot R1 仅消费合格局部窗口，固定腕、机械臂、相机、物体和安装，只允许合同列出的
   手指自由度；硬门退化即整段回 R0。
5. 转接环候选网格可以用于审阅；实测 mount/TCP/camera-world-base 缺失时完整装配采用继续阻塞。
6. Local/HuRo 只做同目标、同分母复核；无独立真值不宣布胜者。

## 停止与成功语义

- 每个故障包最多两轮有新证据的修复；同签名质量失败不重试。
- 各分支分别记录 execution、structure、quality、adoption。
- 全部分支终态化即可结束；质量失败和证据不足是合法终态。
- 默认并保持 `training_eligible=false`、`control_ground_truth=false`、
  `physical_deployable=false`、`external_metric_authority=false`。
