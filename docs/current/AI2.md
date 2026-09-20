# AI2 V3.2 执行入口

总计划：[PLAN.md](PLAN.md)；机器状态：[STATUS.json](STATUS.json)。

首批固定 `play_cards_0915_031` 与 `get_potato_chips_0915_007`。物理左目 `sourceIndex=1`、encoded
resize-only；不读 0916/PICO/controller 作为手输入。

在读取被评 HaWoR 前冻结 16 个等时间分桶审阅帧和最多两段、每段最多 30 帧的独立 RGB 可评估窗口。
text-only SAM3.1 只是 hand-region proxy，不能证明逐指可见、左右身份或 anatomical wrist。raw 与现有
`HOLD_NUMERIC_GATES` bounded 候选只在同一独立集合比较，不再先开第二套平滑路线。

所有有效帧生成 `KAI22_FULL_FK_SIDECAR_V1`。`KINEMATIC_ONLY`、`DEVELOPMENT_R0` 与
`H50_READY` 分层发布；bounded 拒绝不阻塞 raw 的独立运动学结果，Contact/Object 也不是 R0 总开关。
证据不足时停在正确层级，不为获得 R0 标签临时创造阈值。
