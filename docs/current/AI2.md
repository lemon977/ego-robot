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

## 当前执行状态（2026-09-20）

- 两个首批会话均已生成独立 RGB 可观测性证据、raw/bounded 对照、全帧 q22/FK 与回放。
- `play_cards_0915_031`：149 帧，98 个有效 side-frame；90 帧存在真实 clip，最大 0.197328 rad。
- `get_potato_chips_0915_007`：378 帧，378 个有效 side-frame；372 帧存在真实 clip，最大 0.106450 rad。
- 两段视频均完整解码，SHA-256 分别为
  `8bb146bcf9b681e3cf48769f43024420ffc7e0558643063819240ad4dfbcbf4b` 与
  `26f241a56b8b01d2b29e313e4a38b85af1a81edee939c0f632aae5b33d4ec121`。
- 当前最高层级严格保持 `KINEMATIC_ONLY`，不是 `DEVELOPMENT_R0`。历史 A6 thumb 候选的拒绝仍有效，
  本轮没有再创建第二套 thumb/smoothing 候选。
