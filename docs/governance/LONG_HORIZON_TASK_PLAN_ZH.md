# exact78 接触遮挡、Robot、训练与治理长程任务 V3

## 冻结目标

```text
Raw / HaWoR / 双语义Mask
→ Stereo Depth / observed-only Object6D
→ Clean
→ Human Contact Hypothesis
→ Contact-aware Robot Retarget
→ Robot Render
→ Occlusion Compositor
→ HumanEgo Raw vs Robotized
```

## 不可打破的边界

- Wave0固定58条metric-ready，已有Clean 4、待54；后续修复只发布Wave delta。
- 正式Object6D只保存直接观测，遮挡期接触pose必须使用独立hypothesis namespace。
- FoundationStereo没有原生confidence；只能发布明确命名的质量证据。
- Contact hypothesis、Robot retarget、Occlusion compositor是三个单向依赖模块。
- Clean/SYNTHETIC像素不反喂Depth、Object6D或控制动作。
- Robot视觉训练authority与物理部署authority分开。
- 人类视频只提供未来2D辅助监督；最终policy action必须来自同步真实Robot数据。

## 执行顺序

1. 事实账本、Wave、磁盘和调度合同。
2. Wave0 Clean与上游C修复并行。
3. Chips034/Poker042执行Contact/Robot的4帧、24帧、全片逐级canary。
4. 通过后扩exact78视觉Robot，并构建严格配对Raw/Robotized bundle。
5. 完成四个视觉辅助checkpoint；真实Robot action通过schema后才启动policy训练。
6. exact78冻结后复用到0909/0910 release。

## 冻结分母与 Wave

- exact78 Raw 固定为156条；Raw routing口径中59/156缺标定。
- HaWoR、Role Mask、Object Mask三路共同A/B为101条，其中43/101缺标定。
- `EXACT78_WAVE0_SELECTION.json` 固定58条metric-ready（Chips39、Poker19），已有Clean B 4条、待54条。Wave0运行后不因上游修复改变分母或ETA。
- 上游新增结果只能进入 `EXACT78_WAVE1_DELTA.json`、`EXACT78_WAVE2_DELTA.json`，不得回写Wave0。

## 状态机、重试与调度

任务只允许以下状态：

```text
PENDING / CLAIMED / RUNNING / PASSED
FAILED_QUALITY_C / FAILED_RUNTIME_RETRYABLE / FAILED_RUNTIME_FINAL
BLOCKED_PREREQ / BLOCKED_RESOURCE / BLOCKED_EXTERNAL / CANCELLED
```

每个会话使用 `attempts/attempt_NNNN/` 与只读 `final/` 分离。运行时最多初次运行加两次相同输入、代码签名的重试；质量C不自动重试。已有final且SHA一致则skip，SHA不一致时拒绝覆盖。调度器短时claim后立即释放queue lock，再等待GPU lease；等待GPU不得持有queue/session锁。lease必须记录host、PID/startticks、task/attempt、GPU、输入/代码SHA、heartbeat和过期时间。回收前同时核验heartbeat、进程身份和GPU实际进程。

## 接触遮挡三模块

### Human Contact Hypothesis V1

只消费HaWoR、物体身份Mask、正式observed-only Object6D和动作时序，输出独立假设，不消费Robot render，也不修改正式Object6D。pose source为 `DIRECT_OBJECT6D`、`BIDIRECTIONAL_RIGID_HYPOTHESIS`、`HAND_OBJECT_ATTACHMENT_HYPOTHESIS` 或 `UNKNOWN`。遮挡gap最多15帧/0.5秒；前后端点差不超过15mm/10°；attachment起点数字距离不超过10mm；手物相对漂移不超过15mm/10°；缺后端观测、实例冲突或超限即UNKNOWN。Chips三个实例始终独立。

### Contact-aware Robot Retarget V1

依次执行腕/臂IK、人手姿态retarget、接触手指refinement、碰撞清理、时序优化和终审。每阶段最多2个初始化、200次迭代、单帧30秒，单session canary最多30分钟。硬约束包含有限值、proper rotation、左右解剖身份、结构闭包、限位/可达性、gross penetration、非接触指持续穿透和坐标/帧身份。数字门：Poker contact 0–2mm、penetration P95≤1.5mm、gross max≤3mm；Chips分别0–3mm、≤3mm、≤6mm；非接触指穿透>1mm次数为0。这些只表示数字模型残差。

### Occlusion Compositor V1

仅在Robot解算与渲染后决定 `BACKGROUND/HUMAN_FRONT/OBJECT_FRONT/ROBOT_FRONT/TIE_UNKNOWN`，不得反向控制solver。Depth输入为 `depth_m`、`depth_valid`、`depth_confidence_present=false` 和明确命名的质量证据，不伪造FoundationStereo confidence。物体像素按当前Raw、验证过的时序donor、可信纹理renderer顺序取用；只有几何没有外观时必须UNKNOWN，不得用Clean桌面冒充物体。

QA必须同时报告known accuracy与coverage：known accuracy≥95%、known coverage≥70%、unknown pixel≤30%、unknown contact frames≤20%、最长UNKNOWN≤5帧、错误前后关系连续≤2帧；接触窄带外保持byte-exact，应在Robot前方的物体像素条件保留率≥99%。单帧接触区UNKNOWN>30%时Raw/Robotized同时排除。

三个模块分别过门：Clean缺失只能阻挡compositor，不能反向阻挡Human Contact或Retarget；正式Object6D可作为明确的 `HYPOTHESIS_ONLY` 输入，但不能因此升级成接触真值。

## 上游C有界修复

独立失败簇为HaWoR 12C、HaWoR已过但Role Mask失败的Chips19/Poker1、Poker Object Identity额外23C、以及43条三路A/B但缺标定。每簇每轮只跑1条代表canary和2条旧A/B regression，最多2轮、4小时工程墙钟、2 GPU小时；canary通过且回归不退化才扩批，超预算冻结C并继续Wave0。标定只接受同设备、同采集合同且会话身份可验证的结果。

## Wave0 Clean

首条全片canary必须给出每帧字节、临时/最终倍率、总帧数和峰值空间；可用空间须达到预计峰值的1.5倍。保留现有4条Poker B，对54条fresh/no-clobber执行；通过后删除可再生staging，只保留master、中文review、source map和receipt。最后生成覆盖156条的混合终态索引，未满足上游门者明确为C。

## Robot与训练authority

Robot分 `VISUAL_TRAINING_AUTHORITY` 与 `PHYSICAL_DEPLOYMENT_AUTHORITY`。视觉链仍必须world-first、完整Tianji CAD、确认的NaturalV2法兰、真实KaiHand URDF，不能用黄色柱或未知adapter冒充零件；检查root闭包、掌面、虎口、拇指、五指覆盖、法兰端口和左右手性，并按4关键帧→24帧→全片逐级晋升。缺adapter CAD、TCP、安装或camera/world→base实测时，物理authority保持外部阻塞。

HumanEgo future标签固定为 `[T,50,2,2]` original/normalized坐标和 `[T,50,2]` valid；homography失败位置无效。human auxiliary与real Robot batch固定1:1，初始 `lambda_policy=10`。Raw/Robotized只允许视觉RGB不同，其余session/frame/label/valid/split/seed/语言/配置完全一致。辅助checkpoint与最终policy分名；缺同步真实Robot action时只可准备schema/bundle/epoch0/视觉辅助，policy保持 `BLOCKED_EXTERNAL`。

## 清理合同

可永久删除的范围仅限零开放FD的Python/pytest/ruff可再生缓存，以及具有正式终态收据的孤立staging。`_run`、`NOW`、旧runs和脚本先建立引用图、PID/FD、checkpoint与SHA审计，进入quarantine至少7天并完成一次当前回归后才考虑永久删除。禁止 `git clean -fdx`，禁止删除authority、receipt、checkpoint、评审视频、当前Git修改或仍被绝对路径引用的文件。

## 事实账本

唯一读取顺序为 `CURRENT_STATUS_RECEIPT.json` → `CURRENT_AUTHORITY_INDEX.json` → `LONG_HORIZON_TASK_STATE.json` → 自动生成的 `CURRENT_PROJECT_STATUS_ZH.md`。发布使用governance lock、CAS、临时bundle、schema/路径/bytes/SHA校验、fsync、原子rename，最后发布receipt并追加hash-chain changelog。活动任务每30秒heartbeat，语义变化立即重建Markdown；receipt不一致、过期或幽灵PID时显示冲突并停止authority晋升。已确认事实、假设和冻结决定必须分开；内部残差不得升级为外部真实精度。

## 止损

- 运行时最多初次+2次重试；质量C不自动重试。
- 每个上游失败簇最多2轮successor、4小时墙钟和2 GPU小时。
- Contact/Robot一天内必须给4帧Go/No-Go，但不保证获得authority。
- 任一研究路线失败都必须发布诊断、失败门和下一外部阻塞，不得无限RUNNING。

精确实时状态、计数、PID和下一任务只读 [CURRENT_PROJECT_STATUS_ZH.md](CURRENT_PROJECT_STATUS_ZH.md)。
