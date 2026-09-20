# AI1 V3.2 执行入口

总计划：[PLAN.md](PLAN.md)；机器状态：[STATUS.json](STATUS.json)。

097/098 是 development，101 是已暴露 regression，102/103 只允许在候选冻结后首次打开。先交付
MANUS25、PICO/controller、M0 和完整整手回放，再诊断约 398/481 mm 的 M1 修正；旧 `selected=M1`
不代表 adopted。

诊断顺序固定为图像域、`T_A_B`/单位、MANUS root/轴、timestamp/有限 lag、controller-local residual、
HaWoR 2D/相对手形/absolute-depth 分解。不同 wrist 语义不得直接组成误差；不得穷举坐标和时移后挑
视觉最贴者。M2 只在独立旋转证据充分时打开。

无合格静态候选时，正确终态为 `NO_ADMISSIBLE_STATIC_CANDIDATE`，但原生整手、M0/M1 数值和失败
证据仍须交付。任何通过最多是 position-only development adapter，不是实测安装标定或外部毫米精度。

## 当前执行状态（2026-09-20）

- 097/098/101 共 466 帧的 MANUS25、PICO/controller、M0/M1 数值与完整回放已生成。
- M1 复现出约 397.9 mm（左）和 480.8 mm（右）的 controller-local translation correction；
  这只是模型拟合结果，不是实测安装量。
- 冻结证据未支持采用 M1，M0 继续为默认消费，终态为
  `NO_ADMISSIBLE_STATIC_CANDIDATE`。
- 102/103 仍未打开；没有为通过而拟合 M2、逐帧 wrist 或 Contact 对齐。

Canonical 结果与完整回放：

```text
_run/current/four_stream_pretraining_baseline_v32/attempts/attempt_0001/lanes/ai1/STATE.json
_run/current/four_stream_pretraining_baseline_v32/attempts/attempt_0001/lanes/ai1/static_wrist_candidate_v32/RESULT.json
_run/current/four_stream_pretraining_baseline_v32/attempts/attempt_0001/lanes/ai1/static_wrist_candidate_v32/AI1_097_098_101_M0_M1_FULL_REVIEW.mp4
```

数值、输入和 producer SHA 以 `RESULT.json` 为恢复入口；不能从视频反推或覆盖矩阵。
