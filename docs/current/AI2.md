# AI2 当前执行入口

计划：[PLAN.md](PLAN.md)  
机器状态：[STATUS.json](STATUS.json)

## 固定范围

- 只消费 0915 processed 数据和固定只读资产。
- 物理左目 `sourceIndex=1`、encoded VST resize-only，禁止重复去畸变。
- 不读 0916，不使用 PICO/controller 作为手输入。

## 当前任务

1. 建立 wrist/palm/五指部位级独立可观测性和正确分母。
2. visible-but-no-HaWoR 必须计失败；离屏、遮挡与 UNKNOWN 分开。
3. 审计 HaWoR 及所有训练输入的 causal/noncausal 权限。
4. 在同一冻结可见帧集比较 raw/bounded 的抖动、reprojection 和 latency。
5. 核对 Kai22 joint order/axis/sign/unit/zero/palm/thumb4，并分层发布 R0。

质量门修复与 HaWoR 数值改善分别报告；局部窗口不得反写 session strict PASS。

