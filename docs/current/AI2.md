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

## V3.1 实际终态

- `poker` 是逻辑任务名，当前资产目录明确映射为 `playing_cards`；该路径错误已修复。
- W0/A1/A2 共 8 个会话全部完成 current-only 只读审计，8/8 输入路径与 SHA 绑定成功。
- 终态为 8 `BLOCKED_EVIDENCE`、0 `REJECTED`、0 `PASS`。阻塞集中在独立部位可观测集、独立
  reprojection、suffix pair materialization 和 R0 local-quality；A1/A2 还缺 bounded candidate。
- 没有运行模型或 GPU，也没有因为路径修复而宣称 HaWoR 数值改善或 Kai22 R0 通过。

下一步应先生成与待评估 HaWoR 解耦的部位级可观测/重投影证据和 suffix pairs，再在冻结的相同
帧集上评价 bounded candidate；不能用当前 HaWoR 输出反向定义自己的可见性分母。
