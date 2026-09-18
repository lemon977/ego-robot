# 0915 Removal Envelope V1 单会话复核

会话：`play_cards_0915_001`

当前结论：运行与证据发布 `PASSED`；Removal 质量
`AWAITING_USER_VISUAL_REVIEW`。本页不把执行成功写成 Clean PASS。

## 直接查看

- [150 帧 2×2 完整审阅视频](0915_REMOVAL_ENVELOPE_REVIEW.mp4)
- [压力帧与自动极值帧接触表](0915_REMOVAL_ENVELOPE_CONTACT_SHEET.jpg)
- [结果收据](../../../../tasks/receipts/0915_REMOVAL_ENVELOPE_SINGLE_SESSION_CANARY_V1_RESULT.json)
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

## 当前人工复核重点

相比 SAM-only 视频，MANO capsule 让手指及皮套附近的 removal 连续得多，前臂 corridor
也不再依赖 SAM 前臂是否逐帧出 mask。当前黄色设备 profile 在 150 帧都产生了 cable
候选；接触表可见少量靠近卡牌黄色图案或小黄点的候选，可能是 appearance 假阳性，必须
重点检查完整视频里的背景损伤与 object preservation。因此尚未批准为 Clean 基线，也未
运行 inpaint。

## 证据边界

封存的弱角色 SAM 运行没有发布 raw candidate 像素，本任务如实记录
`ABSENT_UPSTREAM_NOT_PERSISTED`，没有从 semantic 反造。Semantic 仍保持用户已判定的
`REJECTED_QUALITY_AS_CLEAN_BASELINE`；Removal 单独等待人工结论。

`removal_envelope` 和 `feather_alpha` 含几何扩张、appearance 推断与显式 bounded hold，
只能进入 Clean QA 或后续独立的视觉 inpaint；禁止用于 Depth、Object6D、Contact、Robot
geometry 和 control ground truth。
