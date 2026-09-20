# AI4 / HuRo V3.2 执行入口

总计划：[PLAN.md](PLAN.md)；机器状态：[STATUS.json](STATUS.json)。

固定上游：HuRo commit `033197778fcc30edc3631dddf3343a967683da09`。本轮仅执行：

```text
HURO_DERIVED_HAND_ONLY_RETARGET
DEVELOPMENT_ONLY
NOT_OFFICIAL_FULL_HURO_REPRODUCTION
```

共同输入是 physical-left resize-only 域对应的冻结 MANO21/wrist/timestamp/valid。先运行
`get_potato_chips_0915_097` 的 Kai22 fixed-wrist/root-relative hand-only 对照，再按 044、007、031
顺序扩展。两方法必须使用同一 target、有效帧、KaiHand URDF/mesh/限位、FK 评估器和 renderer。

HuRo 原生视觉 Stages 1–7、本机不具备条件的 Stage 9 RT 渲染与 Stage 10 均不在当前执行闭环。
当前无 Wuji20 URDF/mesh/joint order/limits，因此 Wuji 固定 `BLOCKED_ROBOT_ASSET`，不能把 Kai q 改名。

## 当前执行状态（2026-09-20）

- 四个冻结会话 097、044、007、031 均已生成 HuRo-derived hand-only q22/FK 与共同输入数值对照。
- 007：378 个共同有效帧，q RMS 差异 37.1204°；031：98 个共同有效帧，q RMS 差异 41.7909°。
  这些是两种 retarget 的差异，不是外部精度或优劣真值。
- 四片均已有同帧 root-relative 对照视频并通过完整解码。新增三片视频 SHA-256：
  044 `c8274584c7ae841b7f87fc984aa46a69b9bfbeb184670615d5c6029960cdeac1`、
  007 `ab7fd378ef307aa5d820a5f8fa3927b9fac76fd8d765a3ebdcbe032fa5bef6cd`、
  031 `b03ab43b16581f42d408fe6f9056fc601c64ff8bf3dca8297a127a1ce1291d8c`。
  视频显示的是差异而非精度；无共同有效侧时明确显示 `NONE; no fill`。
- Stage 9 保持 `BLOCKED_HARDWARE_LICENSE`，Wuji 保持 `BLOCKED_ROBOT_ASSET`；均不影响已完成的
  Kai22 development hand-only 比较。

Canonical 状态和四片结果入口：

```text
_run/current/four_stream_pretraining_baseline_v32/attempts/attempt_0001/lanes/ai4_huro/STATE.json
_run/current/four_stream_pretraining_baseline_v32/attempts/attempt_0001/lanes/ai4_huro/chips097_hand_only_v1/RESULT.json
_run/current/four_stream_pretraining_baseline_v32/attempts/attempt_0001/lanes/ai4_huro/poker044_hand_only_v1/RESULT.json
_run/current/four_stream_pretraining_baseline_v32/attempts/attempt_0001/lanes/ai4_huro/chips007_hand_only_v1/RESULT.json
_run/current/four_stream_pretraining_baseline_v32/attempts/attempt_0001/lanes/ai4_huro/poker031_hand_only_v1/RESULT.json
```

统一回放由 `src/chaoyang/ops/render_huro_hand_only_common_review_v1.py` 从保存的 q/FK 生成；视频不含
仅供显示的姿态偏移。
