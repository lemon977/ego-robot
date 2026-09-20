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
