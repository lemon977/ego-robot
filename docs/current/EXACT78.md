# Exact78 V3.2 执行入口

总计划：[PLAN.md](PLAN.md)；机器状态：[STATUS.json](STATUS.json)。

固定 cohort 为 156 个历史成员，不补位：train=0901 两个 source group（59），validation=0902（77），
development final=0903（20）。0903 已暴露，只是冻结后的 development final，不是首次盲测。

当前顺序：身份与 lineage 重绑定 → 真实 Raw/Robotized 像素和 H50 标签 → 两次隔离生产的
suffix-invariance → loader/loss/backward/参数变化/save/reload → 四模型共同 update 训练。Robotized 四小时
仍被可复现错误阻塞时，可单独建立 `RAW_ONLY_DEV`，但不得冒充成对 A/B 完成。

四模型是 Chips/Poker × Raw/Robotized 的 H50 future-2D Visual Aux，不是 Robot policy。第一资源周期
四模型合计 12 GPU 小时；任何临时 A/B 只比较共同 checkpoint，并同时报告 stay-put 与 constant-velocity。
Stereo canary 独立终态化，不阻塞 pair 或训练。

## 当前执行状态（2026-09-20）

- 固定 156 成员 cohort 已恢复并冻结，当前正式 Raw/Robotized pair 仍未闭合。
- `play_cards_0901_042` 已完成当前会话的 Raw、HaWoR、SAM3.1 human mask、hand-only q22 与完整 FK：
  196 帧、383 个有效 side-frame、9 个 `UNKNOWN`，不做 forward-fill。
- 审阅视频已完整解码 196/196 帧，SHA-256 为
  `48e82c3846dc05159234ad97183b13dea421afebeb86611abd3f9b98ee3781ba`。
- 该结果明确是 `OFFLINE_NONCAUSAL / HAND_ONLY / ARM_BLOCKED`，
  `robotized_training_input=false`、`training_eligible=false`。未观测到 robot base、camera→base 与
  tool→hand mount，因此不得用单位阵或跨会话模板补造完整 Robotized 输入。
- 下一冻结动作：到 T+4 检查点仍无法闭合 Robotized 时，按计划启动 `EXACT78_RAW_ONLY_DEV_V1`，
  只证明 Raw 训练闭环，不宣称四模型 A/B 完成。
