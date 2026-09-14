# 深度与三维精度专项任务状态

更新时间：2026-09-11T03:20:00+08:00  
状态：`COMPLETE_VALIDATED`  
原则：本文件是本专项的执行约束。每完成一个产物，必须先登记路径、SHA、验证和结论，再进入下一项；聊天记忆不构成 authority。

## 1. 目标

整理 `Raw → HaWoR / FoundationStereo Depth → Object6D → Robot` 中所有与深度、三维位置和精度有关的来源、坐标链、内部一致性、外部真值边界及当前问题，并生成可用于会议的直观视频、图片、表格和结论。

## 2. 固定声明边界

- FoundationStereo 当前只授权为 `VISUAL_OBJECT6D_CANDIDATE_INPUT`。
- `Z=fB/d` 重算、SIFT registration、点云平面 residual、IK/FK residual 都是内部闭环，不是外部物理真值。
- HaWoR MANO 与 FoundationStereo 都是估计器；二者差异不是任一方的真实 MAE。
- `Stereo surface-depth proxy` 是在 HaWoR 2D joint ray 附近采样到的可见表面深度，不是真值骨架或解剖关节。
- per-frame median-Z correction 只用于诊断整手平移解释力，不回写 HaWoR、Depth、Object6D 或 Robot。
- c2w→Robot 只做代码路径审计，不修改算法或 placement。

## 3. 冻结输入证据

| 输入 | SHA256 | 用途 |
|---|---|---|
| `20260911_quantitative_accuracy_evidence_audit_v1/REPORT_ZH.md` | `110567ee130f18a9a607ed394042bc12e5a95837167a34daa2fc922f8e5973cf` | 外部真值边界 |
| `20260911_hawor_stereo_surface_consistency_qa_v1/RESULT.json` | `53b57297f93de1d0a986cff9030de794266da30c3b2de433a4cd45a8108f47ab` | 四手定量结果 |
| `20260911_hawor_stereo_surface_consistency_qa_v1/REPORT_ZH.md` | `1c8d381c9f74ef46043cffaf35c5b0d3681187f6f1eec85e21c78b0145d112b2` | 固定QA方法 |
| `docs/pipeline/RAW_TO_HUMANEGO_END_TO_END_REPRODUCTION_ZH.md` | `dbb2db2e85f1c0445fa154a52e571be84148a123b0d104a8decd4f9a0131b39d` | Depth输入输出、坐标与流程位置 |
| `tools/run_hawor_stereo_spatial_relation_video.py` | `46358b2da685b5f64d514d8a44e005b78a8b6f898fd49ad59fb6491408cf9a00` | 空间关系与surface-depth proxy可视化生成器 |

正式产物必须继续绑定逐帧 HaWoR NPZ、MANO surface cache、Depth frames、registration authority、role mask 与 selected-left RGB 的精确路径/SHA；以每个 `RESULT.json` 为准。

## 4. 工作清单

| ID | 产物 | 状态 | 完成/验证摘要 |
|---|---|---|---|
| D01 | `Chips034_Right_3D_Depth_Comparison.mp4` | `COMPLETE` | 293/293，H.264 1920×720@30；SHA `fa25f760...` |
| D02 | `Chips034_Right_Before_After_Z_Alignment.mp4` | `COMPLETE` | 原始/median-Z平移同帧对照；SHA `0539db5d...` |
| D03 | `Chips034_Right_Proxy_Skeleton.mp4/png` | `COMPLETE` | surface-depth proxy独立视频/关键帧；SHA `dd7e4058...` / `9925e359...` |
| D04 | 三时间窗、手掌/五指 residual | `COMPLETE` | JSON SHA `8a64ac34...`；区域为MANO最近关节表面proxy，非解剖GT |
| D05 | `HAWOR_STEREO_DIFFERENCE_TABLE.csv/png` | `COMPLETE` | CSV SHA `142201a8...`；PNG SHA `726e2161...` |
| D06 | `DEPTH_TYPES_OVERVIEW_ZH.png` | `COMPLETE` | 五类量严格区分；SHA `7294edd0...` |
| D07 | `INTERNAL_VS_EXTERNAL_ACCURACY_ZH.png` | `COMPLETE` | 内部闭环与外部真值边界；SHA `ff475f50...` |
| D08 | `DEPTH_ACCURACY_CURRENT_STATUS_ZH.md` | `COMPLETE` | Depth来源、分辨率、生产公式、QA与边界；SHA `b9ad9fb5...` |
| D09 | `EXTERNAL_DEPTH_VALIDATION_PLAN_ZH.md` | `COMPLETE` | 12位置、真值定义、固定ROI和统计合同；SHA `5632c4c7...` |
| D10 | `C2W_ROBOT_COORDINATE_AUDIT.md` | `COMPLETE` | 当前六会话链确认world-first，物理精度仍未标定；SHA `75e64fcd...` |
| D11 | `MEETING_DEPTH_SUMMARY_ZH.md` | `COMPLETE` | 已确认/未确认/禁止结论/下一步一页摘要；SHA `785fdcf2...` |
| D12 | 全产物SHA、视频解码、图像/CSV/Markdown验证 | `COMPLETE` | 19产物、3视频全解码、4图、4文档、36行CSV；收据SHA `a442f3a5...` |
| D13 | 项目根目录独立深度汇报包 | `COMPLETE` | 复制而非移动；按视频/图片/表格/文档/凭证/脚本分类并可独立验SHA |

## 5. 已完成的前置诊断

初版293帧空间视频已经完成并通过H.264/yuv420p、1920×720、30 fps、293/293帧全解码验证：

- 原始MANO与Stereo侧视：`20260911_hawor_stereo_surface_consistency_qa_v1/chips034_right_spatial_relation_v1/CHIPS034_RIGHT_HAND_原始空间错位_侧视3D.mp4`
- median-Z对齐后：`20260911_hawor_stereo_surface_consistency_qa_v1/chips034_right_spatial_relation_v1/CHIPS034_RIGHT_HAND_减去中位Z后_侧视3D.mp4`
- 初版结果：`20260911_hawor_stereo_surface_consistency_qa_v1/chips034_right_spatial_relation_v1/RESULT.json`，SHA `1379347ce6a0f0091d73095bedb45729e864795e16e1cfa487e8b4d457618bd6`

初步定量结论：

- 原始frame-balanced surface MAE：`59.1476 mm`。
- 每帧只减去surface median-Z后，frame-balanced residual MAE：`13.9859 mm`。
- MAE降低约`76.4%`，因此整手absolute-Z placement是主要解释因素。
- 对齐后frame-balanced residual P95仍为`43.4329 mm`，说明手指/手形/局部Stereo表面仍有明显结构残差，不能归结为纯刚体Z平移。

## 6. 更新日志

- `2026-09-11T03:08:52+08:00`：建立任务文档与机读状态；登记冻结证据和初版空间视频结论。
- `2026-09-11T03:15:10+08:00`：D01完成。发布293帧RGB+MANO+Stereo表面点云X-Z/Y-Z侧视+逐帧数值/曲线视频；全片解码PASS。
- `2026-09-11T03:15:10+08:00`：D02完成。发布同帧“原始 vs 仅施加median-Z平移”视频；MAE `59.15→13.99 mm`，降低约`76.4%`，但residual P95仍`43.43 mm`。
- `2026-09-11T03:15:10+08:00`：D03完成。发布HaWoR joint-center与Stereo surface-depth proxy连线视频/关键帧图；画面内明确标注proxy不是Stereo关节或外部GT。统一媒体凭证为 `media_v1/MEDIA_DELIVERY_RECEIPT.json`。
- `2026-09-11T03:20:00+08:00`：D04完成。三个重点窗逐帧移除整手median-Z后：020–060 residual MAE/P95=`7.81/24.07 mm`；090–120=`18.46/60.20 mm`；180–210=`16.89/57.04 mm`。后两窗中指/无名指等局部残差仍大；180–210小指没有满足定义的可见支持，诚实记为N/A/coverage 0，不插值。
- `2026-09-11T03:20:00+08:00`：D05完成。四手表、Chips右手288负偏帧、270帧`<-30 mm`、204帧`<-50 mm`、最长连续39–213共175帧、每秒bias和分区表已写入 `tables_v1/HAWOR_STEREO_DIFFERENCE_TABLE.csv/png`；机读区域结果为 `tables_v1/CHIPS034_RIGHT_THREE_WINDOW_REGIONAL_METRICS.json`。
- `2026-09-11T03:23:00+08:00`：D06完成。发布 `diagrams_v1/DEPTH_TYPES_OVERVIEW_ZH.png`，明确区分 HaWoR MANO Z、FoundationStereo optical-Z、Object6D空间位置/near-far、Robot CAD z-buffer 与 contact signed distance，禁止将五者统称为同一种“深度”。
- `2026-09-11T03:23:00+08:00`：D07完成。发布 `diagrams_v1/INTERNAL_VS_EXTERNAL_ACCURACY_ZH.png`，把公式重算、registration、点云拟合、IK/FK residual归入内部一致性；把真实距离、手部3D、物体pose、Robot TCP/安装/接触归入尚未完成的外部真值。两图统一结果 `diagrams_v1/RESULT.json`，SHA `e1c0e1e2...`，已目视检查。
- `2026-09-11T03:27:00+08:00`：D08完成。发布 `docs/pipeline/DEPTH_ACCURACY_CURRENT_STATUS_ZH.md`，详细记录六种分辨率域、FoundationStereo的rectification/视差/有效域/registration链、四手QA、Chips右手三窗、Object6D范围和允许/禁止结论；SHA `b9ad9fb5f12c1d0ac86dec8a1c1bd6b769bd7e35303d1110d039c307dd5cf787`。
- `2026-09-11T03:31:00+08:00`：D09完成。发布 `docs/pipeline/EXTERNAL_DEPTH_VALIDATION_PLAN_ZH.md`；冻结30/50/70/100 cm×中心/左/右共12位置、5–10秒采集、真实optical-Z与ROI定义、bias/MAE/P95/temporal std/valid rate输出、Raw与标定SHA合同；状态明确为计划完成、真值采集尚未执行。SHA `5632c4c7c39a7c07322943bbe9405903089d5d66604c1cd9ed015680dc874929`。
- `2026-09-11T03:37:00+08:00`：D10完成。只读审计确认当前六会话development producer使用`joints_3d_world`与固定task-level `T_world_base`做world-first IK，渲染时才计算`inv(c2w(t))@T_world_base`；旧Poker单帧seed存在camera-first来源，但没有被直接扩为固定camera全片target。物理c2w/安装精度、Robot authority与action sidecar仍未成立；未修改任何Robot代码。文档SHA `75e64fcd9a1a060d61629a48f31d9b50f2a9a55dc384ff46ec0f763fa86a9923`。
- `2026-09-11T03:40:00+08:00`：D11完成。发布 `docs/pipeline/MEETING_DEPTH_SUMMARY_ZH.md`，用一页结构整理已确认、尚未确认、禁止写入汇报的结论、六个优先观看产物与后续实验；SHA `785fdcf2404a8c42735f7fa7882a221e6c99ce0ff7afe11b42b50a898661bbc9`。
- `2026-09-11T03:43:00+08:00`：D12完成。`tools/finalize_depth_accuracy_spatial_diagnostic.py`（SHA `fb934a14...`）完成最终机器复核：19项产物SHA、三条293帧H.264/yuv420p视频全片`ffmpeg -xerror`解码、四张图片解码、36行CSV、四份文档链接、四份机读JSON全部PASS。最终收据 `FINAL_RECEIPT.json` SHA `a442f3a5936f67af9fcaf83306d8ffedcae3be6ebab99a42491fb5c63dac00f5`；`SHA256SUMS.txt` SHA `870a64943453cd97dd42046bb49362b13d730d7538cf7329f95956685a2b5574`。专项终态为`COMPLETE_VALIDATED`，不改变任何上游authority。
- `2026-09-11T03:55:00+08:00`：D13完成。在项目根目录创建 `DEPTH_ACCURACY_PACKAGE_20260911/`；采用复制而非移动，原authority与正式产物路径均未改变。包内按`videos/images/tables/documents/evidence/receipts/scripts`分类，入口为`README_ZH.md`，`PACKAGE_MANIFEST.json`与`scripts/verify_package.py`支持逐文件字节数/SHA复验。
