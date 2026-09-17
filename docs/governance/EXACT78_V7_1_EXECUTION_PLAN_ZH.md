# Chaoyang 执行路线 V7.1

本文件是 V7.1 的冻结执行合同。实时数量只从
`CURRENT_STATUS_RECEIPT.json` 绑定的事实账本读取，不在本文手工维护。

## 完成语义

阶段完成表示分母内每项均进入唯一终态，不表示全部质量通过。允许终态为
`PASSED`、`FAILED_QUALITY_C`、`FAILED_RUNTIME_FINAL`、`BLOCKED_PREREQ`、
`BLOCKED_RESOURCE`、`BLOCKED_EXTERNAL`、`BLOCKED_REFERENCE_PROOF`、
`UNKNOWN_VERIFICATION_REQUIRED` 或 `CANCELLED`。

里程碑严格区分：

- `EXACT78_TERMINAL_COMPLETE`：156 条跨阶段终态完整。
- `EXACT78_WAVE0_CLEAN_TERMINAL_COMPLETE`：冻结 58 条均有 Clean 终态。
- `EXACT78_END_TO_END_COMPLETE`：Robotized、训练及真实 Policy 均完成；本轮不承诺。

## 治理与版本

- `G0_CORE_GOVERNANCE` 是唯一全局硬门，只覆盖死亡任务、GPU lease、单一
  ledger writer、current index、Task Packet 引用和 receipt/revision 闭包。
- 权重 SHA、历史胶囊、旧运行复现和外部 CAD/TCP/真实动作属于局部
  `G0_PROVENANCE_DEBT`，只限制引用它们的 authority。
- Clean 固定消费 `R7_0`；successor 只产生 `R7_1+`。产物不可原地覆盖；旧结果
  可保持 `VALID_FOR_PINNED_REVISION`，并可同时成为
  `STALE_FOR_LATEST_REVISION`，但不得自动级联重跑。
- Worker 只写 immutable receipt；current ledger 只由单一 aggregator 通过 CAS 更新。

## 单向证据和视觉模式

正式 Object6D 保持 `DIRECT_OBSERVED_ONLY/KEEP_INVALID`。Object/Contact 证据只允许：

```text
DIRECT_OBJECT6D / BIDIRECTIONAL_TRACKED
        -> CONTACT_SEED
        -> stable contact hypothesis
        -> HAND_OBJECT_ATTACHMENT
        -> occluded pose continuation
```

Attachment 结果不得反向证明 Contact、Object6D、tactile-supported contact 或 gold
accuracy。每次发布前必须验证 evidence DAG 无环和祖先类型合法。

遮挡输出分为：

- `VISUAL_OCCLUSION_SILVER_READY`：只检查 provenance、coverage、unknown、时序、
  z-buffer 和 byte-exact，不报告 accuracy。
- Gold：只在冻结的独立人工审计集上报告 accuracy；缺 goldset 时以
  `BLOCKED_PREREQ_GOLDSET` 终结，不阻塞 Silver 或 Robot Geometry。

正式训练输入必须为 `CAUSAL_TRAINING_INPUT`，第 t 帧只能消费不晚于 t 的观测。
双向跟踪、未来 donor 和双向补图只允许标记为
`OFFLINE_BIDIRECTIONAL_VISUALIZATION`。

## Robot 和训练拆分

Robot 分为：

1. `ROBOT_GEOMETRY`：手/腕、URDF、相机和可选几何，输出 q、link transforms 和
   collision QA；不依赖 Clean。
2. `METRIC_CONTACT_EVALUATION`：只消费正式 Object6D 和合法 Contact evidence。
3. `ROBOTIZED_COMPOSITOR`：消费 Clean、Robot Geometry、合法对象外观和统一
   z-buffer。

所有左右 arm、palm、finger 和对象使用同一深度比较，保存 depth、instance、part、
link、triangle ID。z-buffer 只处理可见性；非相邻 link、自指穿掌和左右机器人穿透由
独立 self-collision QA 处理。

Visual Aux 新增 `VISUAL_AUX_ELIGIBLE`，不要求 metric calibration 或正式 Contact。
四支训练拆为 Chips 与 Poker 两个独立 pair；每个 pair 的 Raw/Robotized 必须共享
session、frame、label、split、seed 和 valid mask。监督仅为 H50 future-2D，数字轨迹
始终 `control_ground_truth=false`。

## 新传感器线

手套、Controller、MANUS、PICO 数据采用独立 cohort：

```text
H0 Admission
|- H1 Canonical Hand
|- H2 Tactile
|- H3 Stereo
`- H4 Sensor Role/Object Mask
```

H1 使用 Controller-to-Wrist 与 MANUS25，发布
`HAND21_FROM_MANUS_CONTROLLER`，不冒充 HaWoR/MANO 真值。H2 只提供接触事件
时序证据。H3 使用同 session 的 SBS 与标定，Stereo 表面深度不直接覆盖 wrist。
H4 独立标记左右手套/前臂、Controller 和任务对象实例。

## 资源、止损与清理

- GPU heartbeat 30 秒、TTL 120 秒、等待上限 30 分钟。
- stale lease 仅在 heartbeat 过期、PID/startticks 不匹配且 GPU 无对应进程时回收。
- GPU 优先级：短 canary、regression、full batch、checkpoint training。
- 每失败簇最多两轮，每轮 1 条 canary、2 条冻结 regression、4 小时墙钟和
  2 GPU 小时。
- Cleanup 不等待 Clean 全部完成，只避开 current authority、活动 PID/FD/CWD、
  checkpoint、许可证、`/mnt/data/egodata` 数据及数据侧可视化。
- 证明不足的删除候选必须进入 `BLOCKED_REFERENCE_PROOF`。

## 最终交付

必须形成 exact78 终态矩阵、转换原因账本、Object/Contact 证据索引、Robot 终态
矩阵、Occlusion Silver 报告、Goldset 或阻塞收据、Visual Aux checkpoint 索引与曲线、
新传感器终态矩阵、清理收据和治理会议快照。

任何研究任务失败都必须在预算内形成明确诊断或负面终态，禁止无限运行、无限调参或
用内部残差升级物理精度。
