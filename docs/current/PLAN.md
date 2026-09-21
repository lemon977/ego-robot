# Chaoyang 四支线算法基线优化与交付 V2

状态：CURRENT EXECUTION PLAN；标识 `FOUR_STREAM_ALGORITHM_BASELINE_V2`。
任务：`four_stream_algorithm_baseline_v2`；实际起点 `bf5e4e743dea90b72beb1d7d88ffb305ae60208c`。
执行窗口：2026-09-21 11:41:59 至 19:41:59（北京时间）。

本轮推进可复现、可验证、可继续优化的算法和工程基线，**不训练 checkpoint**。本页替代 V3.2 的当前执行路由，不否定历史 hand-only 的原用途，不重启历史训练。

## 权威、隔离与实时状态

先读 [AI 工作入口](AI_WORK_ENTRY_ZH.md)、[状态快照](STATUS.json)、[当前 receipt](../governance/CURRENT_STATUS_RECEIPT.json)、[算法合同](../governance/ALGORITHM_CONTRACT.json)、[当前任务索引](../../tasks/current/INDEX.json)。只有实际登记且 execution_allowed 的操作可运行。

唯一 publisher 串行登记、集成和发布；四线使用隔离 worktree、lane 输出和独立 writer。旧父任务失效进程已受控退役，历史收据不改。父任务 FRESH 只说明活动进程心跳，不表示算法或文档全部通过。任务完成、工程修复、算法采用、视觉审阅、训练资格分别记录。

本轮产物统一在 [attempt_0001](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/)，原始数据、processed、archive、sealed、旧结果只读。不得停止其他 AI，不 reset、push、prune、gc 或修改现有模型环境。

CPU 总预算最多8线程，每 worker 最多2线程；GPU最多一个治理租约，JAX 首次导入也受此限制。当前实现优先CPU，利用率不是成功指标。

用户补充授权：仅 HuRo 可以在本项目自己的 lane 中新建独立环境，联网获取必要 Python 依赖和官方依赖源码；不下载模型、不升级系统或其他活动环境。缓存、安装临时文件、下载和源码均留在该 lane。

## 四线固定范围

| 支线 | 固定输入 | 本轮交付 |
|---|---|---|
| [Exact78](EXACT78.md) | chips0902_103，284帧；cards0902_042，171帧 | 单一受限Clean候选、455帧像素和来源检查、完整虚拟Robot与两条全片视频 |
| [PICO/MANUS](AI1.md) | cards0916_097/098/101，165/179/122帧；不读102/103 | 真实时钟、世界/相机/腕手账本、三条全片及完整Robot；101不是盲测 |
| [HaWoR/Kai22](AI2.md) | chips0915_007，378帧；cards0915_031，149帧 | 逐阶段失败归因、原始/裁剪口径分离、固定腕臂+手指、两条全片 |
| [HuRo 核心](AI4_HURO.md) | 与上一线相同的冻结 RAW/R0、同资产和placement | 官方固定版本核心真实调用、完整腕臂/手指、共同目标公平评价、两条全片 |

支线3/4共同输入已提前冻结为原R0和对应RAW，不替换上游。精确文件和 SHA 见 [共同输入冻结](../../_run/current/four_stream_algorithm_baseline_v2/attempts/attempt_0001/SHARED_AI2_AI4_INPUT_FREEZE.json)。R0 历史上经过clip；在限位内不等于原始优化器未越界。

## 几何、时间和数据语义

- `T_A_B` 把B点变到A；米、右手系、矩阵方向及变换来源必须明确。
- 腕位置、腕旋转、手指、表面、左右手分别有效；observed/inferred/invalid 不互换。未知不能用零或单位阵变成有效观测。
- 原始 frame_id、int64 timestamp_ns、视频映射保留。真实 dt 评价，不跨缺口、重复时钟、tracking reset；展示30FPS不是采集时钟。
- 固定底座、一次确定的虚拟安装和比较视角；增益1，不逐帧缩放或移动底座。图像域外参未经验证，不把诊断叠加称真实贴合。
- `T_flange_hand` 与 `T_tool_hand` 不是同一变换；使用 `T_tool_hand = inverse(T_flange_tool) @ T_flange_hand`，并验证FK等价。
- 机器人 physical left/right 与 anatomical left/right 映射为 `[1,0]`，只转换一次。蓝色是解剖左、红色是解剖右；图例不替代机器映射。
- 表面点必须标明资产近似或真实标定。URDF link origin 不是物理指腹接触点。无物体/环境几何不声称做了对应碰撞。
- SELF_COLLISION 必须显式启用，再按声明规则排除直接邻接；手内、臂手、双手和未知覆盖分别报告，不通过关闭碰撞对获取PASS。

## 每线算法边界

Exact78：身份unknown不能解释成对象不存在；M_write外编码前像素严格不变。旧donor继承不等于重新认证，Attachment不能证明身份/Contact。旧Depth/Object6D不消费；无有效几何不借用其他session。

PICO：使用controller位姿与已声明的固定腕prior组合MANUS局部骨架，不引入HaWoR拟合或缺手补齐，不采用398/481mm历史拟合。不做无约束外参/时移搜索。相机编码域和native外参分别审计。

HaWoR/Kai22：按检测→跟踪→侧别→有效性→映射→渲染定位首次失败。缺手不复制另一手，不隐藏困难帧；同时报告覆盖、幅度、真实dt动态和夹合。不可达/未收敛腕目标保存为失败。

HuRo：只复用 pinned upstream 的真实 `solve_retargeting`，不运行会插值/边界保持/搜索placement的完整Stage8。组合URDF与原资产FK一致；未知自由度的先验不升格观测。官方索引正则若未做dt归一化必须明示，不能冒充真实dt优化。用同一冻结RAW构造的共同目标比较，不拿R0的FK自证新解精度。

## 有限执行与验收

第3小时冻结共同输入；第5小时后不扩算法范围；第7小时后只验证、发布和传输。每线一个最终候选签名，第二次只允许修明确运行错误，不能质量失败后调参。小型接口自测单列，不冒充全片。

必须实测几何/时间/mask/来源负例、FK与关节映射、碰撞正负例、跨CWD入口、传递SHA、全片解码。语义数组比较不使用压缩容器字节作为确定性标准。原始/processed不得写入。

最终按能力发布：可复现参考、已采用工程修复、已采用算法改进、候选未证实、拒绝或缺项。视频是证据，不是独立成功标准；没有用户审阅就保留待审阅。未完成的原因必须真实记录，继续排障不能改写成PASS。

九条主视频目标为2+3+2+2，Windows交付到 `D:\本地测试ego\Chaoyang_四支线算法基线_20260921\`。逐项核对bytes/SHA，跳板机不留中间文件。缺项单列，不用旧视频补数。

历史当前文档原件仅在本attempt的 `predecessor_current_docs` 保留字节与恢复清单；它们不是执行入口。本轮不搬迁大产物、不全仓归档。后续任务必须写明输入、入口、资源、门槛和停止线。
