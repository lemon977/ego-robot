# V2.1 0915 D1_CLEAN_PREP fail-closed 设计

## 权威闭包

`D1_CLEAN_PREP` 只覆盖 Poker044 与 Chips097。B1R 只修复 Poker044/Hand capability：其终态为
`COMPLETED_WITH_QUALITY_REJECTION`，但 Poker left/right 均 `consumer_allowed=false`，所以
两侧一律 UNKNOWN/null。原 B1 `FAILED_RUNTIME_FINAL` 保留、未修改、不作 aggregate terminal；其
Chips right Hand 子工件显式 `consumer_allowed=true`，因此唯一被规范化的 Hand 是
393/394 frame proxy，frame 58 保留 UNKNOWN。

Object lane 不依赖 Hand terminal。Poker 三个 visible proxy 只作同帧 Raw 写入 veto，物理
身份/隐藏外观/atlas 均未授权。Chips 保留三个有序物理槽，逐帧空 mask +
`UNKNOWN_NOT_ABSENT`，不发布 union mask。

composite 不放宽 terminal/capability gate：上游 path/bytes/SHA、side decision、会话分母或
实例槽任一偏移即 fail closed。

## 像素契约

```text
M_remove = union(admitted per-side Hand) AND NOT visible Task Object protection
M_write  = M_remove                         # frozen write allowance
M_flow   = elliptical_dilate(M_write, 16)  # context only
UNKNOWN  = M_write                         # no fresh source materialized
candidate = Raw                            # full-frame decoded RGB byte-exact
```

硬门是 `M_remove ⊆ M_write ⊆ M_flow`、`M_write ∩ protected_object = ∅`、
`UNKNOWN == M_write`、以及 `M_write` 外 candidate/Raw byte-exact。CPU preparation 不产生
候选像素修改。source map 0=同帧 Raw，2=Poker identity-UNKNOWN protected Raw，10/11/12=Chips
三独立槽，250=UNKNOWN/unwritten。

## 模型边界

旧 donor 路由含未来帧且不能证明语义表面一致，因此不得作权威。本包只注册 fresh
inpainting 输入，不运行 GPU/模型。新候选必须使用新 output root，并重新通过 source
map、UNKNOWN、Object protection 和 byte-exact 审计。
