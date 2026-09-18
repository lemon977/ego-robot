# 0915 Removal Envelope V1 单会话复核

会话：`play_cards_0915_001`

当前结论：运行与证据发布 `PASSED`；用户完整视频复核后将 Removal 质量判为
`REJECTED_QUALITY`。V1 只作为失败实验封存，不调半径/HSV 后重试，不进入 inpaint 或
扩批。本页不把执行成功写成 Clean PASS。

## 直接查看

- [150 帧 2×2 完整审阅视频](0915_REMOVAL_ENVELOPE_REVIEW.mp4)
- [压力帧与自动极值帧接触表](0915_REMOVAL_ENVELOPE_CONTACT_SHEET.jpg)
- [结果收据](../../../../tasks/receipts/0915_REMOVAL_ENVELOPE_SINGLE_SESSION_CANARY_V1_RESULT.json)
- [用户视觉质量裁决](../../../../tasks/receipts/0915_REMOVAL_ENVELOPE_V1_USER_VISUAL_REVIEW.json)
- [Removal Envelope V1 当前合同](../../REMOVAL_ENVELOPE_V1_ZH.md)

视频四个面板依次显示：

1. 封存 SAM semantic evidence；
2. MANO finger capsules 与 palm/wrist/forearm corridors；
3. appearance cable region 与 protected visible-object core；
4. 最终 removal envelope 和 feather band。

## 自动闭合结果

- 150/150 帧完整生成并完整解码；
- HaWoR 两侧合计 293 个直接观测 side-frame，7 个双端验证的
  `BOUNDED_INTERNAL_HOLD`，0 个 `UNKNOWN`；
- 直接观测 MANO21 关节在 removal 内的覆盖为 1.0；
- 150/150 帧 removal 非空；
- source bits 与最终 removal 逐像素闭合；
- protected visible-object core 与最终 removal 的交集为 0；
- removal 面积逐帧相对变化 P95 为 0.1509；
- 输入文件前后 SHA/字节一致；GPU 未使用，SAM 与 inpaint 均未运行。

以上只证明合同、来源和运行闭合，不证明真人手/附件已经视觉擦净，也不证明背景或任务
物体没有误伤。

## 人工复核结论

V1 的失败不是半径尚未调好，而是弱证据越权。MANO 中心线被扩成整根粗 capsule；wrist
方向被直接延伸到图像边界；宽松黄色 HSV、78 px hand anchor 和 70 px previous-track
邻域把多个无身份连通域一起接收。第 60 帧约有 34 个黄色候选连通域，最终 removal 为
108,822 px；全片 150/150 帧均标为 `APPEARANCE_TRACKED`，这不能解释为可靠 cable
tracking。自动覆盖门没有测量背景 spill、面积 inflation 或 repair contribution，因此被
自一致的大 mask 骗过。

V2 只能用 admitted SAM foreground 作主体。MANO 只修补 SAM 邻域内的局部断口；forearm
只从 wrist-connected 的真实 foreground proposal 中选择；cable 只接受带 anchor、细长度、
路径、端点和双向时序验证的 top-1/极小 top-k 实例。无可靠证据时必须保持 `UNKNOWN`。

## 证据边界

封存的弱角色 SAM 运行没有发布 raw candidate 像素，本任务如实记录
`ABSENT_UPSTREAM_NOT_PERSISTED`，没有从 semantic 反造。Semantic 仍保持用户已判定的
`REJECTED_QUALITY_AS_CLEAN_BASELINE`；Removal V1 也已独立判为 `REJECTED_QUALITY`。

`removal_envelope` 和 `feather_alpha` 含几何扩张、appearance 推断与显式 bounded hold，
只能进入 Clean QA 或后续独立的视觉 inpaint；禁止用于 Depth、Object6D、Contact、Robot
geometry 和 control ground truth。
