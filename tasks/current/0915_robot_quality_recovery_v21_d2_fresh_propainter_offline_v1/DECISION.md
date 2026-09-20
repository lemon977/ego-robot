# D2 fresh-fill successor 审计决定

结论：**A — 可以冻结一个可执行的 fresh-fill challenger**，但它只是 `OFFLINE_BIDIRECTIONAL_VISUAL_ONLY`，不是 Clean 真值、因果训练输入或控制权威。本次没有运行 GPU/模型。

## 唯一候选

- 算法 ID：`robot_quality_recovery_v21_d2_fresh_propainter_offline_v1`
- 模型：现有 ProPainter，不换模型、不扫参、不回退到其他方法。
- 处理参数：`960×720, dilation=4, ref_stride=10, neighbor=10, subvideo=80, raft_iter=20, fp16, 30 FPS`。
- Poker044：`M_write=0`，166 帧 Raw 直通，不加载模型；可见牌 Mask 仅保护原像素，物理牌/牌面身份仍为 UNKNOWN。
- Chips097：仅用 Raw + D1 `M_flow` 作模型上下文，只在冻结 `M_write` 中合成模型输出；左手仍 UNKNOWN，只处理已承认的右手写域；三个袋子 slot 继续分离且 UNKNOWN，禁止 union。
- Mask 语义：`input_mask_domain=M_flow`；因冻结 `mask_dilation=4`，模型内部实际 mask 为 `DILATE(RESIZED_M_flow,4)`，可超出 D1 `M_flow`。该扩展仅用于模型内部，最终写域始终严格是原分辨率 `M_write`。
- 模型生成像素来源码为 `SYNTHETIC_PROPAINTER_VISUAL_NOT_SCENE_TRUTH`；即使视觉上已补全，`UNKNOWN` 仍必须精确等于 `M_write`。

## 权威与 provenance

- 当前算法账本已登记 `same_pixel_temporal_donor_then_propainter_v1`，且三个权重 SHA 与历史成功全片收据一致。
- 历史 `get_potato_chips_0902_039` 306 帧和 `play_cards_0902_042` 171 帧证明该 ProPainter 路线曾完成结构 Grade-B 全片；这只证明执行、帧数、来源和视频闭合，不证明隐藏表面真实性。
- 当前 `vendor/ProPainter` 没有嵌套 `.git`；旧 runner 直接 `git -C vendor/ProPainter rev-parse HEAD` 会读到主仓 HEAD，因而会假报 commit mismatch。D2 不改模型，改为同时固定 53 个源文件树 SHA、入口文件 SHA 和三个权重 SHA。历史 upstream commit `e870e793...` 只作 lineage 记录，不伪装成当前嵌套 Git 验证。
- `vendor/ProPainter/weights` 是指向 `assets/models/vendor/propainter` 的软链接。D2 对每个权重分别固定 vendor `lexical_path`、assets `resolved_path`、bytes 和 SHA；词法路径与解析实体任一不符都拒绝，但不再把正常软链接布局误报为越界。
- D1 560 帧、3360 个逐帧文件引用已重算 SHA；Poker `M_write=0`，Chips `M_write=14,909,727 px`、`M_flow=21,667,934 px`。

## 硬拒绝条件

1. D1 terminal/result/frame-manifest 或任一逐帧 Raw/Mask/source-map 的 bytes/SHA 不一致。
2. Poker 任一帧 `M_write>0`，或 Chips 不再是“左手 UNKNOWN + 右手承认 + 三实例分离/no-union”。
3. `M_remove ⊆ M_write ⊆ M_flow`、`D1 UNKNOWN=M_write`、任务物体与 `M_write` 不相交任一失败。
4. ProPainter 源码树、入口或任一权重 SHA 改变。
5. 模型输出 PNG 数不是 394，或运行错误不是“全部 PNG 已完整落盘后的已知 PyAV preview writer 错误”。
6. 编码前候选在 `M_write` 外发生任一像素改变，或任务物体原像素不再 byte-exact。
7. `UNKNOWN` 被生成像素清除，或输出被标为 causal/training/contact/control/deployment authority。
8. 任一全片复核 MP4 解码帧数不是 166/394。
9. 深层生成任意异常后 final 目录可见、sibling staging 未清理，或任一 JSON 内嵌 `.staging.` 路径。
10. 浅层产物是软链接、不是深层 MP4 的 byte-exact 实体副本，或浅层发布不是同盘 sibling staging 加一次 `os.replace`。
11. vendor 输出文件名不是精确 `0000.png..0393.png`，或 D2 标准化结果不是精确 `000000.png..000393.png`；只检查数量不充足。
12. 浅层发布未提供 GPU lease receipt，receipt SHA 未写入索引，或其 status/task/attempt/command/命令中 deep output root 任一不精确匹配。

## 产物合同

真实执行时应生成两条全片四栏复核视频（Raw / D1 domains / D2 candidate / change+UNKNOWN）。所有帧明确标注 `OFFLINE_VISUAL / NOT_FOR_TRAINING / UNKNOWN retained`。候选完成后仍只能写 `COMPLETED_OFFLINE_VISUAL_CANDIDATE_UNKNOWN_RETAINED`，不能写 Clean PASS。

深层和浅层发布都已冻结为事务性合同：同文件系统 sibling staging，任意异常时全清理且 final 不可见，成功时一次 `os.replace`。深层 JSON 在 staging 中计算 bytes/SHA，但 `path` 必须预先投影到 final。浅层发布器复制实体 MP4 并复核 SHA/解码帧数，不创建软链接。精确 V7.1 lease 及浅层发布命令见 `README_ZH.md`。
