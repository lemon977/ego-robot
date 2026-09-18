# 0915 单样本全链基线复核 V1

> **已撤销基线资格。** 本目录的 RGB 来自一次显式
> `equiDis62 → 1280×960 FOV90 pinhole` 重映射。用户观察到画面弯曲，VST 编码
> 像素域尚未确认；以下内容只用于复现问题，不能评价正确输入域上的任何模型。
> 见 [`../../VST_IMAGE_DOMAIN_HOLD_0915_ZH.md`](../../VST_IMAGE_DOMAIN_HOLD_0915_ZH.md)。

本目录固定复核 `playing_cards/play_cards_0915_001`。输入来自 0915 processed-only
物理左目，150 帧，1280×960；未使用 PICO26/controller 手部结果，未创建 tracker
角色，源数据未修改。

## 直接观看

- `play_cards_0915_001_FULL_FUNNEL_REVIEW.mp4`：150 帧四格视频，依次为
  Raw+HaWoR、SAM3.1、FoundationStereo optical-Z、Clean 预览。
- `play_cards_0915_001_FULL_FUNNEL_CONTACT_SHEET.jpg`：同一视频的 12 帧抽样。
- `play_cards_0915_001_SAM31_BASELINE_REVIEW.mp4`：Raw+HaWoR 与 SAM3.1
  的双格基线视频。
- `play_cards_0915_001_SAM31_BASELINE_CONTACT_SHEET.jpg`：SAM3.1 基线 12 帧抽样。
- `RESULT.json`、`SAM31_BASELINE_RESULT.json`：视频解码、帧数和 SHA 收据。

## 本次终态

| 阶段 | 终态 | 证据摘要 |
| --- | --- | --- |
| Raw | `PASS_DEVELOPMENT_CANARY_INPUT` | 物理左目 `sourceIndex=1`，150/150 帧 |
| HaWoR | `OBSERVED_ONLY_DEVELOPMENT_CANARY` | 左 94、右 95、双手同时 39 帧；缺失不补造；严格 expected-active 门因独立可见性分母缺失而未评估 |
| SAM3.1 | `REJECTED_QUALITY` | 牌对象 150/150，但三张牌、承托板及周边被合为一个大实例；左右手各只保留 1/150 帧 |
| FoundationStereo | `PASS` | 150/150 推理；中位有效覆盖 58.45%；held-out 极线误差中位 0.595 px、P90 1.325 px |
| Object6D | `BLOCKED_UPSTREAM` | Mask 未通过，禁止消费失败掩码 |
| Clean | `BLOCKED_UPSTREAM` | 视频中的 Clean 仅为复核预览，不是发布结果 |
| Contact | `BLOCKED_UPSTREAM` | Mask/Object6D 未通过 |
| Robot Visual | `REJECTED_QUALITY` | 开发级相对运动求解运行完成，但速度/加速度质量门未通过 |
| Contact-aware Robot | `BLOCKED_UPSTREAM` | 上游未通过且实测 TCP/安装/世界到底座标定仍缺失 |

Depth 只声明同会话内部一致的 optical-Z，不声明外部毫米精度。Robot sidecar 固定
`control_ground_truth=false`、`physical_deployment_authorized=false`、
`calibration_authority=DEVELOPMENT_ONLY`。

## 结论

此前“没有显示整套相机参数用错”的结论已撤销。held-out 极线误差只检验当前重映射
内部的一致性，不能证明 VST 编码像素需要该重映射。由于图像域前提不成立，当前
HaWoR、SAM3.1、Depth 与 Robot 现象都不能用于定位首要算法瓶颈；Object6D/Contact
继续 fail-closed，整个 0915 链停止。

完整逐帧产物与 GPU 收据位于：
`_run/current/0915_one_session_full_funnel_canary_v1/attempts/attempt_0001/`。
