# Chaoyang 4小时集成优化与清理计划 R2.2

> 状态：`CURRENT_BOUNDED_EXECUTION_PLAN`。本文件定义本轮执行边界，不是完成证明。实时状态仅由 `CURRENT_STATUS_RECEIPT.json` 绑定的机器账本给出。

## 冻结原则

- G0发布 `RUN_START_SNAPSHOT.json`，绑定启动receipt、generation、revision、Task Packet Index、算法合同、Robot hard/soft索引、选择集、Git状态和GPU/PID证据。
- Worker只消费冻结输入并写自己的immutable attempt；current账本只由aggregator CAS更新。
- v77只恢复旧算法的单会话完整终态，禁止采用v76 partial；v78另用新revision研究前伸/动作幅度。
- SAM3.1保持当前Mask基线；Clean、Contact、Occlusion、Robot候选不自动晋升authority。
- 四支Visual Aux本轮只做资格预检，不训练。

## 四小时最低闭环

1. `V77-ADOPT-12`闭合既有单会话证据，并至少完成一个3条新batch。
2. Poker/Chips各运行一条独立evaluation reference的SAM3.1 Mask canary。
3. Poker245/Chips039闭合semantic donor、lossless source-map和Poker causal atlas前置；不满足时发布`BLOCKED_PREREQ`。
4. 完成新传感器rectification诊断；未达到vertical epipolar P90与覆盖门时不运行FoundationStereo。
5. 至少运行一条真实`CONTACT-10` canary，保持`HYPOTHESIS_ONLY`。
6. 至少生成一条基础几何Occlusion全片；Contact缺失只使接触窄带为UNKNOWN。
7. 清理只删除普通可再生缓存并生成旧资产dry-run；旧run、历史文档、工具/测试/合同留给后续显式commit任务。

## 时间与终态

- 3:00停止启动新GPU任务，3:15停止启动全部新计算。
- 3:15--3:45生成全片视频、实验矩阵和恢复包，3:45--4:00发布receipt与状态。
- 每个子任务最多两个运行时attempt；质量C不自动重试；GPU等待30分钟后转`BLOCKED_RESOURCE`。
- 四小时父任务必须终结，未启动工作写入`NEXT_ACTION.json`，不得留下无heartbeat的`RUNNING`。

## 清理保护边界

永久保护current receipt/authority、CURRENT文档、checkpoint、权重、许可证、Clean R7_0、v77 adopt闭合前的v74/v75/v76证据、`docs/current_visuals`、未分类Git变化、`/mnt/data/egodata`和`/nas/chenxianchi`。

本轮所有视觉产物默认标记：`DEVELOPMENT`、`control_ground_truth=false`；Contact无独立真值时保持`external_accuracy=UNKNOWN`。
