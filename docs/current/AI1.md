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
