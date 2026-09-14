# exact78 V3 实现与接手续读

本文记录“实现在哪里、如何验证、失败时从哪里恢复”，不维护实时数量。当前数量、PID、阻塞和 next task 必须从 [CURRENT_PROJECT_STATUS_ZH.md](CURRENT_PROJECT_STATUS_ZH.md) 读取，并先校验 `CURRENT_STATUS_RECEIPT.json`。

## 1. 治理事实账本

- 机器事实：`CURRENT_AUTHORITY_INDEX.json`、`LONG_HORIZON_TASK_STATE.json`。
- 原子绑定：`CURRENT_STATUS_RECEIPT.json`；任一 revision、generation、bytes 或 SHA 不一致即 `STATUS_CONFLICT`。
- 人类入口：自动生成的 `CURRENT_PROJECT_STATUS_ZH.md`，禁止手改关键计数。
- 追加历史：`STATE_CHANGELOG.jsonl`。
- 工具：`tools/governance/`；claim 统一通过 `register_claim.py`，stage 通过 `register_stage_authority.py`，任务通过 `update_task_state.py`。
- 会议冻结：`python -m tools.governance.create_meeting_snapshot --timestamp <YYYYMMDD_HHMM>`。

## 2. Wave0 Clean

- 执行根：`tasks/control/runs/20260913_exact78_v3_wave_clean_v1/`。
- 冻结选择：`EXACT78_WAVE0_SELECTION.json`；后续新增只能进入 Wave1/2 delta。
- 当前唯一可执行闭包：`EXECUTION_AUTHORITY_V3.json`。V2 已在GPU执行前撤回，见 `EXECUTION_AUTHORITY_V2_WITHDRAWN.json`。
- Chips107 CPU handoff 已按 Raw PNG、Role manifest、对象manifest和三个独立物体实例逐帧闭合；见 `preparation_receipts/get_potato_chips_0902_107.json`。
- 运行目录固定为 `sessions/<session>/attempts/attempt_NNNN/` 与 `final/RESULT_REF.json`；质量C不重试，运行时最多三次，final SHA冲突拒绝覆盖。
- 资源释放瞬间若别的合法任务先取得lease，guardian最多等待30分钟且不持queue/session锁，不会把lease竞争当成三次CUDA失败。
- 等待器和恢复命令记录在 `WAIT_GUARDIAN_PID.json`、`MACHINE_INTERFACE.json`、`TASK_PROGRESS.md`。

## 3. 上游 C 有界修复

- 执行根：`tasks/control/runs/20260913_exact78_v3_lane_c_successors_v1/`。
- 冻结分簇与代表样本：`REMEDIATION_SELECTION.json`。
- 执行闭包：`EXECUTION_AUTHORITY.json`。
- 每簇最多两轮、四小时工程墙钟、两GPU小时；失败canary与两条旧A/B regression必须一起过门才扩批。
- 等待器曾因事实账本正确拒绝变更后的证据SHA而退出；无GPU/data任务启动。恢复证据为 `WAIT_GUARDIAN_RUNTIME_RECOVERY_20260913T0222.json`，后继继续 no-clobber 等待。

## 4. 接触、Robot与遮挡三模块

- `pipeline/human_contact_hypothesis_v1.py`：只消费 HaWoR、对象身份Mask、正式observed-only Object6D和时序，绝不消费Robot render或修改正式Object6D。
- `pipeline/contact_aware_robot_retarget_v1.py`：六阶段、任务特定接触/穿透门、预算和最佳诊断解合同。
- `pipeline/occlusion_compositor_v1.py`：只处理ownership和合法像素来源；没有物体外观时输出UNKNOWN，不以Clean桌面冒充物体。
- schemas 位于 `contracts/*_v1.schema.json`。
- Poker042全片假设结果：`tasks/control/runs/20260913_contact_robot_v1_canary_v3/poker/play_cards_0902_042/RESULT.json`。
- 中文四帧诊断：`tasks/control/runs/20260913_contact_robot_v1_canary_v3_visual_v1/poker/play_cards_0902_042/`。
- Robot solver没有运行：缺真实NaturalV2→KaiHand装配闭包、adapter CAD/安装测量、world→Tianji base、TCP/安装标定。不得用旧黄色proxy或视觉placement补齐。

## 5. HumanEgo

- future-2D与real Robot action使用不同schema：`contracts/edited_human_aux_manifest_v1.schema.json`、`contracts/real_robot_policy_manifest_v1.schema.json`。
- 代码：`HumanEgo/utils/visual_aux_contract.py`。
- readiness：`tasks/control/runs/20260913_humanego_visual_aux_prepare_v1/READINESS.json`。
- 缺Robotized视觉authority时不能构建正式paired bundle；缺同步真实Robot action时只能做辅助准备，最终policy保持 `BLOCKED_EXTERNAL`。

## 6. 清理

- 第一轮只删除可再生cache，证据根：`tasks/control/runs/20260913_root_cleanup_v3_lane_a/`。
- `_run`、`NOW`和旧runs因仍有current/文档绝对路径引用未移动。必须先闭合引用图、PID/FD、checkpoint和SHA，再进入至少7天quarantine；禁止直接删除。

## 7. 回归与故障处理

核心回归覆盖治理冲突/篡改/幽灵PID、Clean retry/no-clobber/GPU lease竞态、接触gap与三实例、UNKNOWN覆盖率、Depth confidence边界、旧Robot contact/compositor行为、HumanEgo标签schema。

恢复原则：先读current receipt并验证，再核PID+startticks、物理GPU进程与中央lease；不得仅因lease显示RELEASED就抢卡。任何运行中断先写 `FAILED_RUNTIME_RETRYABLE` 收据，再恢复为新的attempt；质量C不得当运行时错误重跑。
