# 0915 Robot 15 小时执行状态

状态：`COMPLETED_WITH_FINAL_AUDIT_REJECTED_QUALITY`

运行标识：`robot15h-0915-20260919T000959+0800`

时间边界：T0 `2026-09-19 00:09:59+08:00`；算法候选最晚冻结 H9
`09:09:59`；H13.5 后不启动新会话；H14 后 GPU/写入排空；H15
`15:09:59` 前完成最终审计。最终 audit 于 `14:19:38` 完成并诚实终态化为
`REJECTED_QUALITY`。本页只汇总当前正式执行事实；聊天、导出文件存在或任务顶层
`PASSED` 都不能替代逐能力质量门。

## 固定范围

- 唯一数据批次：`/mnt/data/egodata/datasets/ego/processed/chips_cards_hands_0915`。
- 220 个编号会话按用户确认分别录制；来源 manifest 中 220 个 capture ID/source group 也互异。
- `play_cards_0915_001` 仅作 development，不进入 final holdout。
- 0916 highview 仍只有清洗 authority，不进入本轮 HaWoR、Depth、Contact 或 Robot。
- 图像域固定为 SBS 中 `sourceIndex=1` 物理左目，crop/resize-only；禁止 lens
  undistortion。FoundationStereo 只允许显式的双眼同步水平反射视差符号 adapter，并将
  输出反射回原物理左目像素域。
- 所有 Robot 结果最多为 `DEVELOPMENT_RELATIVE / NON_CONTROL / NON_DEPLOYABLE`。

## W0 实际结果

W0 固定 4 个独立录制、1,058 帧：Poker031、Poker119、Chips007、Chips042。

| 能力 | 实际运行 | 质量准入 | 当前结论 |
|---|---:|---:|---|
| HaWoR recovery | 4 | 0 | 运行闭合，但 4/4 未通过严格手部质量门；不得扩为 W1 Human authority |
| Kai22 R0 | 4 | 0 | 4 个数值导出、1,320 个有效 side-frame；strict R0 仍为 0 |
| FoundationStereo recovery | 4 | 4（限开发 Depth scope） | encoded-domain 内部门通过；外部毫米精度仍 `UNVERIFIED` |
| SAM3.1 hand proxy | 4 | 按 hand/role 局部准入 | 不是 session-wide Mask PASS；弱角色和 task object 不继承该准入 |
| SAM3.1 task object | 4 | Poker 2；Chips 0 | Poker 发布 6 个稳定 card 实例；Chips 两会话质量拒绝 |
| Object6D | 2 尝试、2 上游阻塞 | Poker 2 | 6 个 card 实例、824 个可观测 instance-frame；只表示有限可见表面 |
| Interaction | 1 尝试、3 上游阻塞 | 1 个开发诊断 | Poker119 有 161 个 hand-associated visible-surface samples、251 个 metric pair row；有限 patch 内命中为 0 |
| strict Contact / R1-E | 1 个诊断会话 | 0 | 严格 Contact window 为 0；R1-E 0 个导出、0 个采用 |
| R1-H | 1 | 0 | 16 个 hand-only counterfactual frame；因缺 object-independent metric wrist placement 拒绝采用 |
| Virtual R2 | 4 | 0 | 4 个虚拟安装数值候选；继承 R0 质量上限，完整物体/环境碰撞未验证 |

R2 的“4 个导出”不是 4 个 Robot 成功。人工复核也看到虚拟手臂大量位于视域边缘；自动
门与人工结论一致，`r2_quality_admitted=0`。

浅层入口：

- [Poker119 四栏总览](visuals/0915_ROBOT15H_W0_SUMMARY_V1/README_ZH.md)
- [尺度原因审计](visuals/0915_ROBOT15H_SCALE_CAUSE_AUDIT_V1/README_ZH.md)
- [W0 HaWoR recovery](visuals/0915_ROBOT15H_W0_HAWOR_RECOVERY_V1/README_ZH.md)
- [W0 Kai22 R0](visuals/0915_ROBOT15H_W0_KAI22_R0_V1/README_ZH.md)
- [W0 FoundationStereo recovery](visuals/0915_ROBOT15H_W0_FOUNDATIONSTEREO_RECOVERY_V1/get_potato_chips_0915_042_FOUNDATIONSTEREO_REVIEW.mp4)
- [W0 Object6D](visuals/0915_ROBOT15H_W0_OBJECT6D_V1/INDEX.json)
- [W0 Interaction](visuals/0915_ROBOT15H_W0_INTERACTION_V1/INDEX.json)
- [W0 Contact](visuals/0915_ROBOT15H_W0_CONTACT_V1/INDEX.json)
- [W0 R1](visuals/0915_ROBOT15H_W0_R1_V1/INDEX.json)
- [W0 Virtual R2](visuals/0915_ROBOT15H_W0_VIRTUAL_R2_V1/README_ZH.md)

## H9 冻结与 W1

H9 release candidate 已在截止前冻结，20 个 capability/task/scope row 中：13 个实际执行、
11 个导出、4 个窄 scope 质量准入。获准扩展的只有：

- FoundationStereo 的 Poker/Chips `DEVELOPMENT_DEPTH_VISUAL_OBJECT6D_INPUT_ONLY`；
- SAM3.1 的 Poker/Chips `DIRECT_ANCHORED_HAND_INSTANCE_PROXY_ONLY`。

但是 SAM3.1 的冻结算法初始化依赖同会话 HaWoR box/point；HaWoR W0 为 0 个质量准入，
所以 W1 没有合法的新会话 prompt source。不能把算法偷换为 text-only、人工 box 或伪造
anchor。因此 W1 当前实际状态是：

| W1 节点 | 8 会话实际状态 |
|---|---|
| HaWoR | 0 运行，8 阻塞：`W0_HAWOR_HAS_ZERO_QUALITY_ADMITTED_SCOPE` |
| SAM3.1 hand | 0 运行，8 阻塞：`MISSING_RELEASE_ADMITTED_HAWOR_PROMPT_SOURCE` |
| FoundationStereo | 8 实际运行、8 内部门通过、0 质量拒绝、0 运行失败；2,332 帧 |
| Robot | 0 运行、8 阻塞：`NO_RELEASE_ADMITTED_W1_HAWOR_OR_KAI22_R0` |

FoundationStereo W1 在 `2026-09-19 09:09:59+08:00` 达到 H9 后启动，8 个源视频全部
SHA 绑定，CPU preflight 为 8/8，通过一次 GPU 单租约、一次模型加载和 4,664 次推理完成。
8 条审阅视频均完整解码，入口见
[W1 FoundationStereo](visuals/0915_ROBOT15H_W1_FOUNDATIONSTEREO_V1/README_ZH.md)。这些
PASS 只授权开发级 Depth/视觉 Object6D 候选输入，`external_accuracy=UNVERIFIED`、
`strict_metric_contact_authorized=false`。

这一区分很重要：release matrix 中“允许 SAM hand scope 扩展”不等于新会话已经具备运行该
冻结算法的完整输入。缺输入时必须阻塞，不能为了提高运行数改变算法。

## 截至 W1 的机器计数

库存分母为 220；冻结选择 12（W0 4、W1 新增 8），其余 208 未选择、未运行。下面的
“导出”不等于“质量成功”，阻塞也不能计入实际运行数。

| Robot 层 | 分母 | 实际尝试 | 数值导出 | 质量成功 | 质量拒绝 | 局部/上游阻塞 |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 12 | 4 | 4 | 0 | 4 | 8 |
| R1-E | 12 | 0 | 0 | 0 | 0 | 12 |
| R1-H | 12 | 1 | 1 | 0 | 1 | 11 |
| R2 virtual arm | 12 | 4 | 4 | 0 | 4 | 8 |

FoundationStereo 则为 W0 `4/4`、W1 `8/8` 内部门通过；这与 Robot 四层计数分开。最终
H14 审计已把同一口径固化为 `CAMPAIGN_COUNTS.json`，没有把全部终态写成全部成功。

## H14/H15 封账结果

- H13.5 后没有启动新会话；H14 时 GPU lease 已释放、任务队列为空、没有 campaign worker。
- 12 个冻结源视频重新哈希通过，内容漂移 0，且 `0916_consumed=false`。
- 50 条 Robot15h 审阅视频共 12,824 帧完整解码；完整回归为 `1160 passed`，治理、结构和
  diff check 均通过。
- 第一次本地提交为 `c7c9cea`，commit evidence 证明当时工作树干净且未 push。
- 最终 sidecar audit 全部通过；最终 reference audit 检查 282 个引用，其中 281 个有效、
  1 个漂移。唯一漂移是早期 R2 不可变 RESULT 绑定的虚拟安装合同 SHA
  `fb412c90...`（4,438 字节），与封账前通用治理引用刷新后仓库合同 SHA
  `d86b18c...`（5,863 字节）不一致。
- 未修改旧 R2 RESULT、未伪造旧合同、未放宽引用门，因此最终 audit 正确终态为
  `REJECTED_QUALITY`。这不撤销 12/12 Depth 内部门结果，但阻止本轮发布“完整 SHA 闭合”
  或任何 Robot 成功/部署结论。
- 最终治理为 `12362 / gov-012362-8d04bae5bef9`，任务索引重新回到
  `PASS_NO_ACTIVE_TASKS`；全程未 push。

## 已知审计限制

- H9 matrix 的 collector 只识别 `batch_result/worker_batch`，SAM object recovery 使用
  `object_identity_batch`；因此 release matrix 没有把 Poker object 的 W0 通过升级成 W1
  扩批权。封存 matrix 不原地修改；最终审计必须保留该限制。
- inventory 中的初始 `FROZEN_DAG.json` 先于 runtime recovery/quality successor 形成；
  当前治理账本记录实际 successor 链。旧冻结文件只作 T0 调度证据，不覆盖后续不可变终态。
- `visible_surface_center`、hand-associated visible surface 和虚拟 IK 都不是外部几何真值；
  不能据此声明真实接触、完整碰撞通过或真机可部署。
