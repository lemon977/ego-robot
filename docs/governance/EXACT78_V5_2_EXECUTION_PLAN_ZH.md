# exact78 V5.2 当前执行规范

> 状态：用户已批准，`plan_revision=exact78-v5.2`。本文只保存当前合同；运行计数由治理账本自动生成。

## 交付边界

- Chips 78 + Poker 78 必须形成 156 行唯一终态矩阵。
- 冻结 Wave0 58 条必须各有 Clean 唯一终态；选择文件 SHA 固定为
  `10ee9e3668aa4928f0087c961dbb5d909202af576f859adf7dbb02604f5b3bd1`。
- Robot 只消费 `clean_join_ready`；其余行必须是质量 C、运行失败或明确阻塞。
- 四支训练产物是 `visual_aux`，不是控制策略；必须附 loss、ADE、FDE、PCK 和固定验证视频。
- 数字 Robot 轨迹仅称 `visual_robot_trajectory_sidecar`，内部指标不等于物理真值。

## 固定阶段

1. `governance_recovery_v5`：校验 receipt 的 revision/bytes/SHA、PID/startticks、GPU lease/进程并按实际 finals 重计。
2. 建立最小状态、任务队列、Task Packet、executor fencing、统一 run signature 和故障注入测试。
3. Lane A 完成 Wave0 Clean；Lane B 有界修复上游 C；Lane C 使用冻结数字代理完成 Robot 可视化/数值终态，无外部真值的 contact 固定为 UNKNOWN；Lane D 只以合法数字视觉标签训练四支 visual aux；Lane E/F 同步整理当前文档与可恢复清理。
4. 生成 exact78、Robot、checkpoint、baseline 与 cleanup 五类最终索引，最后原子封账。

## Clean 合同

- 预检每条的帧数、Raw/RGB/Role/Object mask 路径/bytes/SHA、三 Chips 实例独立性、同 session donor、ProPainter 代码/权重与磁盘预算。
- GPU gate 连续检查三次，间隔 10 秒；等待 30 分钟后进入 `BLOCKED_RESOURCE`，GPU 等待不计 attempt。
- `0902_107` 只按完全相同签名采用既有 Grade B；`0903_052` 先核对 prepare 再续跑。
- PASS 必须有全帧 Clean、master MP4、中文全片复核、source map、manifest、RESULT、AGENT_REVIEW，且媒体和对象保护硬门通过。

## Contact / Robot 合同

- 固定几何 fixtures 覆盖牌面上下、接触、穿透、边缘、非 pad link 与 Chips 三实例；全部通过前禁止真实视频 canary。
- Object6D 保持 `DIRECT_OBSERVED_ONLY/KEEP_INVALID`。接触假设分为 DIRECT、双向刚体、手物 attachment 和 UNKNOWN。
- 单帧预算 30 秒，按 4 关键帧→24帧→全片晋升；最多两轮新方法 successor。
- Chips107/Poker243 goldset review pack 保留为证据，但用户明确不提供人工标注；其状态固定为 `EXPLICITLY_UNAVAILABLE_BY_USER_DECISION`，不得伪造 known accuracy/coverage。
- 用户同样不提供 adapter CAD/TCP/安装或 world→base 实物标定；数字流程按冻结代理参数继续，物理部署永久保持 `BLOCKED_EXTERNAL`，不将此作为待等任务。

## Visual Aux 合同

- 每任务 train≥16、val≥3、train windows≥256、val windows≥48；不足只按固定 split/session ID 补最少 Visual Clean Wave1。
- H50 标签为原图像素 2D endpoint、normalized 2D、valid mask 和 RGB training valid mask；Raw/Robotized 唯一变量只能是 RGB。
- 顺序：chips raw、chips robotized、poker raw、poker robotized；先完成四支真实 epoch-0。
- seed=7，主指标 validation_ADE_2D，每5 epoch验证，patience=12，最多180 epoch/每支12 GPU小时。

## 状态、签名与清理

- Session 状态与 Attempt 状态分离；Attempt 最大3次，quality-C不重试。
- run signature 绑定输入清单、代码闭包、配置、权重、标定或明确 ABSENT、schema。
- final 只允许原子写一次；同 SHA 幂等，不同 SHA 禁止覆盖。
- 清理先生成保护快照和引用图；缓存类可在零引用/零FD/CWD后删除，其他候选同盘隔离7天后仅人工 `--commit-delete`。
- 当前文档分 `CURRENT_GENERATED`、`CURRENT_TECHNICAL`、`HISTORICAL_EVIDENCE`、`OBSOLETE`；旧文档不得参与当前计数。

## 完成定义

只有 receipt FRESH、Wave0 58 Clean 唯一终态、156 全阶段和 Robot 矩阵无遗漏、四支 checkpoint 有成功或预算终态、goldset/外部标定的显式不可用声明与 claim boundary 守界、基线/清理/README/AGENTS 同一事实时，才可称 `EXACT78_END_TO_END_COMPLETE`。
