# 新数据 HaWoR → Mask → Clean → Robot 分层质量门 V1

日期：2026-09-02  
适用任务：`poker`、`chips` 及后续相同 PICO ego 任务  
机器实现：`pipeline/session_stage_admission.py`、`tools/evaluate_session_stage_admission.py`

## 1. 先纠正旧批次数字的含义

当前磁盘有 157 个源 session 目录、78 个 `final_v3` HaWoR/重建产物。旧审计把 78 称为
`availability_eligible`；这不等于已经证明其余约 79 条全是 HaWoR 算法质量失败，其中还混有
未处理、输入不完整、未进入冻结集合等情况。历史“约 48 条可用于 Robot”也不是新的物理门
结果：旧 Robot 使用过错误的 MANO 点索引和接触语义，必须按本规范重新计算，不能继承旧数字。

以后每条 session 必须保留原始分母，并分别报告：

```text
CAPTURE_READY
HAWOR_MASK_SEED_READY
HAWOR_ROBOT_SEED_READY
MASK_READY
CLEAN_READY
ROBOT_READY
FINAL_COMPOSITE_READY
```

## 2. S0：采集与文件门

源视频完整解码；RGB、timestamps、c2w 数量严格一致；K 必须逐帧存在，或明确登记为整个
session 唯一常量；时间戳严格递增，`reset_count=0`。连续录制必须有九段：

1. `EMPTY_TABLE_STATIC`
2. `EMPTY_TABLE_VIEW_SWEEP`
3. `OBJECT_CONTACT_FIT`
4. `OBJECT_LIFT_NEGATIVE`
5. `OBJECT_CONTACT_EVAL`
6. `SLEEVE_FOREGROUND_CALIBRATION`
7. `NORMAL_TASK_FIT`
8. `NORMAL_TASK_EVAL`
9. `END_EMPTY_TABLE`

同一 PICO、同一曝光、同一场景姿态的空桌段是 Clean authority；尺子/卡尺与物体同框照片、
外围非共线标志点是 Object/Robot authority。缺少它们时 Mask 仍可走人工提示路线，但 Clean
或 Robot 必须归因到 `CAPTURE`，不能归咎 Mask。

## 3. S1：HaWoR 双用途门

HaWoR 对 Mask 只是左右手自动提示种子，对 Robot 则是三维腕/掌/手指运动 authority；因此
必须分成两个门，不能用“Mask 能跟住”证明“Robot 能用”。

### 3.1 所有用途共同硬门

- 有效比例和最长缺失只在采集前声明的 `EXPECTED_ACTIVE` 区间计算；空桌段不得进入分母。
  每侧必须保存区间帧数、来源及人工/采集协议核验，禁止用 HaWoR 自己的检测结果定义分母。
- 关节顺序必须显式为 MANO21：wrist=`0`，tips=`[4,8,12,16,20]`，
  MCP=`[2,5,9,13,17]`；HumanEgo wrist=`5` 只允许在显式 reorder 后使用。
- 左右手各自具名并保持同一 identity；重复 track 帧数和 identity switch 都为 0。
- 双手同时出现时，两条轨迹必须形成经核验的一一对应；双手 collapse 帧数必须为 0。
- root rotation 有限、右手系，正交误差 ≤ `1e-4`；有效点深度为正的比例 ≥ `99.5%`。
- 2D 重投影误差 p95：Mask 自动种子 ≤`20px`，Robot 自动种子 ≤`12px`；单侧骨长
  时序最大 CV ≤`8%`。20px 不能替代 identity/collapse 门。
- 可见关节落在手轮廓内或距轮廓 ≤12px；该覆盖比例 p05：Mask ≥85%，Robot ≥90%。
  轮廓覆盖必须由预注册人工 canary 或独立视觉证据核验，单纯“投影仍在画内”不能替代。
- 腕点略出画不直接淘汰：若 ≥19/21 关节仍由真实手/前臂 mask 支撑且入画边连续，可标
  `BOUNDARY_SUPPORTED`；禁止只凭 wrist y 超出 1–37px 淘汰整条视频。

### 3.2 Mask 自动种子门

- required side 有效帧比例 ≥95%；最长无效间隙 ≤8 帧；
- confidence median ≥0.65，p05 ≥0.45；
- 未过时仍可进入 `ASSISTED_KEYFRAME`，不能假装成 HaWoR 自动 PASS。

### 3.3 Robot 自动种子门

- required side 有效帧比例 ≥98%；最长无效间隙 ≤3 帧；
- confidence median ≥0.70，p05 ≥0.50；
- 直接观测而非插值的比例 ≥90%；接触 canary 必须 100% 直接观测；
- 接触 canary 必须至少1帧，且由任务段/物体状态或人工标签独立定义，不能由 HaWoR 自己猜；
- wrist step p99 ≤80mm/frame、max ≤150mm/frame（30fps；其他 fps 按时间换算）；
- 任何接触窗口中的 identity swap、中长插值或退化掌面都直接归因 `HAWOR`。

### 3.4 HaWoR 下一轮优化优先级

先改可诊断性和时序稳定性，再考虑重新训练大模型：

1. 左右手独立 track/state/置信度；单侧短暂失败不能让另一侧一起失效。
2. 输出显式数组 schema 与逐帧 provenance，固定 MANO21 顺序；任何 reorder 都落 manifest。
3. 对画面边缘采用前臂/19个以上关节支撑的 `BOUNDARY_SUPPORTED`，替代 wrist 单点越界淘汰。
4. 把直接观测、插值、fallback 分开记账；接触窗口禁止用插值冒充三维 authority。
5. 输出 identity switch、重复轨迹、骨长漂移、掌面退化、重投影和腕速度的 reason code，
   先按失败桶统计，再决定是否做 task-specific 微调。
6. FIT 数据可用少量人工核验校准 confidence；EVAL 阈值和权重冻结，禁止逐视频调参。

只有当新批次主要失败集中在“可见但 confidence/identity/姿态系统性错误”时，才值得追加
扑克/薯片域微调；若失败来自 reset、缺 clean plate、遮挡或 tracker 标注，则应修采集/Mask，
不应让 HaWoR 背锅。

## 4. S3：Mask 门

- 24 帧预注册 canary 中，双手/前臂、袖口、腕带、tracker 的 required-role recall=100%；
- `HUMAN_CORE ∩ OBJECT_CORE = 0`；fixture 误入 HUMAN 的面积比 <0.2%；
- 同一 object/tracker identity 不交换；光流对齐 mask IoU median ≥0.85；
- 全片帧覆盖=100%，union 必须逐像素等于各人体 role 的 OR 再扣除保护对象；
- contact 争议只进入 `CONTACT_UNCERTAIN`，不得用颜色阈值、全局膨胀或 corridor 填充。

HaWoR 门通过而本门失败，责任才是 `MASK`。当前手机诊断中 chips 通过、poker tracker
未通过，正属于这一类。

## 5. S4：Clean 门

- donor 必须来自同一 session、同一相机/曝光；跨机位空桌只能用于诊断；
- 授权编辑域外、保护物体内改动像素均为 0；
- `SOURCE_MAP` 已知覆盖 ≥99.9%，`RESIDUAL` 面积 ≤0.1%；
- 人、袖口、tracker、影子和 seam 人工复核均不可见；桌垫纹理相位连续；
- 开头/结尾空桌经冻结曝光模型后闭合。

当前提供的手机空桌确实是有效场景参考，但与动作 MOV 的视角/布局不闭合；已经出现
37–85px 对应残差、透视拉伸和物体重复，所以不能升为正式 Clean。新 PICO 同一次连续录制
里的 `EMPTY_TABLE_STATIC/VIEW_SWEEP/END_EMPTY_TABLE` 才是直接可用的 clean plate。

## 6. S5：Robot 门与计算策略

每条视频都要做验证，但不应每条都从零做数小时全局搜索。

项目/FIT 阶段只做一次：冻结机器人 CAD、NaturalV2 mount、MANO→Kai 映射、对象几何、
抓取接触模板、碰撞 BVH/SDF 和多分支 retarget profile。EVAL session 只做：

1. 由本 session 标志点/桌面确定唯一 session-constant base；
2. 从冻结抓取模板做 warm-start IK 与短窗口多分支 DP；
3. 全片批量 coarse SDF/broadphase；
4. 只对 near-contact/疑似碰撞帧做 dense 采样与 exact triangle SAT；
5. 所有数值门通过后才渲染。

硬门：required pose ≤10mm/5°；30fps 下关节速度 ≤0.12rad/frame、加速度
≤0.06rad/frame²；joint limits PASS；非接触 dense signed distance ≥-1mm；具名接触指腹
距离 0–3mm；distinct-finger exact SAT 自碰=0；ObjectDepth 必须在真实前后关系下遮挡指节。

这套 coarse→exact 架构仍需要每条视频跑 CPU/GPU 验证，但避免每帧从零全局优化。当前 004
的 3–6 小时估计来自纠错期 Python/SciPy 稠密审计，不应成为批量生产的固定成本。工程目标是
把 per-session CPU 缩到几十分钟，渲染另行串行；在完成向量化与缓存前不承诺具体速度。

## 7. 失败归因表

| 最早失败门 | 责任层 | 后续动作 |
|---|---|---|
| S0 文件/九段/reset/clean plate | CAPTURE | 当场补拍，不启动下游调参 |
| S1 identity/关节顺序/重投影/三维跳变 | HAWOR | 重跑或人工三维修正；Mask 可单独走 assisted |
| S1 PASS、S3 FAIL | MASK | 追加预注册关键帧标注/传播或训练 tracker 类 |
| S3 PASS、S4 FAIL | CLEAN | donor、相机注册或 clean plate 问题 |
| S4 PASS、S5 FAIL | ROBOT | base、IK、retarget、接触/碰撞问题 |
| S5 PASS、最终遮挡 FAIL | COMPOSITOR | ObjectDepth/RobotDepth/抗锯齿问题 |

机器报告必须保存每层 PASS/FAIL/NOT_EVALUATED，禁止把未运行写成 PASS。

## 8. 旧数据回放校准结果

已对 `grap_a_cap_0812` 做 CPU-only 只读回放：157 条源 session 中，78 条有 `final_v3`，
旧 reconstruction gate 将这 78 条全部判为 PASS；但其中只有 68 条有后续 identity 审计，
分类为 CLEAN 35、MINOR 18、DEFECT 15，另有 10 条未进入该审计。其余 79 条缺少 final
产物和执行失败日志，只能记为 `NOT_PROCESSED_OR_NO_FINAL_ARTIFACT`，不能归咎 HaWoR。

旧数据还缺独立 `EXPECTED_ACTIVE`、手轮廓 canary 与 contact intent，因此正式新门仍为
`NOT_EVALUATED`。在可回放数值项和旧 identity 结果上，Mask 候选为 12/78，Robot 候选为
5/78；三帧可视校准支持 Mask 20px / Robot 12px 的分门设计，同时证明 identity、zero-collapse
和独立轮廓 canary 不能省略。

- [批量回放报告](../../archive/audits/hawor_gate_legacy_0812_v2/REPORT.md)
- [逐 session 表](../../archive/audits/hawor_gate_legacy_0812_v2/SESSIONS.csv)
- [候选与负对照三帧可视校准](../../archive/audits/hawor_gate_legacy_0812_visual_review_v1/REVIEW.md)
