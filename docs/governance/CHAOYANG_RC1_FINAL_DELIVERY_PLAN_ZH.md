# Chaoyang 有限交付计划 RC1-FINAL

状态：`CURRENT_DELIVERY_PLAN`（仅在 `PLAN_MIGRATION_RECEIPT.json` 与 current receipt 绑定后生效）。

## 1. 交付目标

RC1 固定处理 exact78 的 156 条会话，采用：

```text
SAM3.1 Mask
→ 因果 Clean
→ v77 求解器按前缀重算 Robot
→ 基础 Z-buffer Occlusion
→ Raw / Robotized 严格配对
→ Chips、Poker 四支 H50 Visual Aux checkpoint
```

手套、Controller、MANUS、PICO 新传感器线独立记录，不进入本轮训练分母。Contact、数字碰撞和内部深度残差均不构成物理真值。所有视觉轨迹固定：

```text
control_ground_truth=false
physical_deployment_authorized=false
```

RC1 允许质量拒绝、运行失败、前置阻塞和预算不足，但每个任务必须进入唯一终态。预算耗尽时发布 `INCOMPLETE`，禁止无限调参。

## 2. 当前消费合同

历史 Robot authority、Contact authority 和完整 Object Atlas 不作为 Visual Aux 全局前置。正式 loader 只消费 receipt 绑定的：

```text
consumer_eligibility.visual_aux_rc1.eligible=true
input_mode=CAUSAL_TRAINING_INPUT
contract_sha256与current release一致
paired manifest / label / valid / neutralization / pixel source闭合
```

状态必须分开保存：

```text
execution_status
visual_quality
train_eligible
```

旧质量 C 和旧 authority 保持不可变；RC1 只追加新的消费资格 sidecar。

## 3. 数据与因果规则

- H50 endpoint 为左右 Robot `T_actual_hand_root_world` 平移；0 为物理解剖左手，1 为右手。
- 标签为 `t+1...t+50`，使用每个未来帧自己的 `inv(c2w[s])` 与 `K[s]`。
- bundle 坐标域为 640×480：`x/639`、`y/479`；模型可将 RGB 缩放到 320×240。
- 输入只有当前 RGB 与共享 neutralization mask；当前输入的全部依赖必须 `source_frame<=t`。
- 未来标签允许使用未来，因果测试只比较 `input_payload_sha`。
- scheduled starts 在质量筛选前固定为 `15,20,25,...,T-51`。
- H50 采用 `ANY_ENDPOINT_TIMESTEP_40_OF_50`：至少40个时间步有任一侧有效，双侧同时无效最长不超过5步，逐侧 valid 原样保留。

Clean 使用 `[t-15,t]` 历史窗口，只发布末帧。双向全片结果只能离线复核。`M_remove/M_flow/M_write/M_composite` 分离，`M_write` 外在编码前必须 byte-exact。

保守对象区域由当前可见实例与过去合法同实例证据构建。无法证明为背景、又可能属于任务物体的洞必须是 `UNKNOWN`，不能用桌面纹理冒充。分别保存 `clean_unknown_mask` 与 `final_input_unknown_mask`；UNKNOWN 门作用于 neutralization 前最终编码器输入，全图不超过1%，冻结交互 ROI 不超过5%。通过门后两支使用相同 mask，并以 RGB `[127,127,127]` 中和。

v77 只复用 URDF、资产、目标定义和 solver。`ARM_BIDIRECTIONAL_STATES.npz`、reverse lookahead、全片 placement 与其他未来优化轨迹不得进入训练输入；当前 Robot 必须按 prefix 重算。已有双向轨迹只能作为离线诊断或未来 target。

## 4. 固定任务图

```text
RC1-MIGRATE-00
→ RC1-T0 容量、source group、split与分母冻结
→ RC1-T1 SAM3.1两轮有界修复
→ RC1-T2 因果Clean与UNKNOWN
→ RC1-T3 v77因果Robot与局部窗口回收
→ RC1-T4 Occlusion、paired bundle、200步smoke
→ RC1-T5 固定预算批量生产和转化率
→ RC1-T6 Chips/Poker两个独立checkpoint pair
→ RC1-T7 2×2评价、release与会议快照
```

每任务至少需要16个train source groups、3个validation source groups、256个train starts、48个validation starts。source group 来自原始采集 UUID/manifest/资产 SHA；无法证明则不计数。同组不得跨 split，开发组不得进入 validation。没有独立 test 时结论范围为 `VALIDATION_ONLY`。

每个 pair 使用同一初始化 SHA、相同样本顺序和相同有效更新数：

```text
seed=7
batch_size=16
AdamW(lr=3e-4, weight_decay=1e-4)
scheduler=ABSENT
AMP=ABSENT
N=min(100*ceil(train_windows/16),10000)
```

取消分支独立 early-stop。`last.pt` 用于严格同预算比较，`best.pt` 用于补充模型选择比较。每任务执行 Raw/Robotized 模型 × Raw/Robotized validation 的 2×2 评价。

## 5. 有限退出与成功定义

```text
集成修订最多2轮
有效工程修复累计24小时
同一run signature最多2次runtime attempt
GPU单次等待30分钟
canary最长30分钟
单阶段单会话最长2小时
单checkpoint最多12 GPU小时
四支训练总上限48 GPU小时
生产批量总上限96 GPU小时
```

`PIPELINE_OPERATIONALLY_CLOSED` 除所有任务有终态外，还要求 Chips 与 Poker 各至少一条真实路径完成：因果输入→paired bundle→200步smoke→保存/重载→中断恢复。

`FOUR_CHECKPOINTS_TRAINED` 仅在两任务四支均达到固定更新数并发布 best/last 时为真。Robotized 比 Raw 改善至少5%记 `POSITIVE`，退化至少5%记 `NEGATIVE`，其余或样本不足为 `INCONCLUSIVE`；这是单 seed 工程观察，不是统计显著性或 Policy authority。

## 6. AI 执行约束

Worker 不默认读取本长文。固定顺序：校验 current receipt→读取 `CURRENT_RC1_STATUS_MIN.json`→读取当前 `TASK_PACKET.json`→只读取 `read_set`→执行 preflight→读取 `RESULT_SUMMARY.json`。每包初始文件不超过8个、搜索结果不超过20条、日志尾部不超过80行、目录深度不超过3、`CONTEXT_CARD.md` 不超过150行。

历史聊天、旧 README、目录名、mtime、partial、孤立视频和旧固定计数均不能作为当前事实。`STALE`、`STATUS_CONFLICT` 或 SHA 不一致时立即停止。Worker 只写 immutable attempt；current 状态只能由 aggregator CAS 发布。
