# 四支线 V5 实际可视化索引（有界终态）

当前路线：[Human→Robot Baseline v1](../../HUMAN_TO_ROBOT_BASELINE_V1_ZH.md)；机器任务包与源域冻结见正文。此目录只放**实际 MP4 文件**，无符号链接。视频能播放不等于质量通过。所有文件都是 `OFFLINE_VISUAL / NOT_FOR_TRAINING / NOT_CONTROL_GROUND_TRUTH`。没有去人 Clean 的 Raw 叠加、Mask 审阅、传感器回放均不是 `robot.mp4` 产品。

## 已完成并核对 SHA 的视频

| 支线 | 视频 | 帧数 | 结论 |
|---|---|---:|---|
| Scene | [Chips007 角色 Mask 全片](get_potato_chips_0915_007_ROLE_MASK_FAILED_REVIEW.mp4) | 378 | `FAILED_QUALITY_C`：设备Mask 371帧为空；SHA `e78ebffe…a453f` |
| Scene | [Poker031 角色 Mask 全片](play_cards_0915_031_ROLE_MASK_FAILED_REVIEW.mp4) | 149 | `FAILED_QUALITY_C`：设备Mask 142帧为空；SHA `6d43b2ee…e71d` |
| Scene | [Chips103 角色 Mask 全片](get_potato_chips_0902_103_ROLE_MASK_FAILED_REVIEW.mp4) | 284 | `FAILED_QUALITY_C`：设备Mask 全片为空；SHA `a88c8ea1…d8c8a` |
| Scene | [Poker042 角色 Mask 全片](play_cards_0902_042_ROLE_MASK_FAILED_REVIEW.mp4) | 171 | `FAILED_QUALITY_C`：设备Mask 112帧为空；SHA `5bd77487…1c3e2` |
| Sensor | [Poker097 Controller＋MANUS](play_cards_0916_097_SENSOR_REVIEW.mp4) | 165 | 全片解码，贴合质量待审；SHA `01e2090c…7901` |
| Sensor | [Poker098 Controller＋MANUS](play_cards_0916_098_SENSOR_REVIEW.mp4) | 179 | 全片解码，贴合质量待审；SHA `2e3d09ec…9705` |
| Sensor | [Poker101 Controller＋MANUS](play_cards_0916_101_SENSOR_REVIEW.mp4) | 122 | 全片解码，贴合质量待审；SHA `2546205f…fd8b` |
| Motion | [Chips103 R0／原片同域诊断](get_potato_chips_0902_103_R0_RAW_DIAGNOSTIC.mp4) | 284 | `C_ARM_TOLERANCE_0_OF_VALID`，非产品；SHA `46443b59…9bda` |
| Motion | [Poker042 R0／原片同域诊断](play_cards_0902_042_R0_RAW_DIAGNOSTIC.mp4) | 171 | R0数值质量C，非产品；SHA `63f47542…ac17` |
| HuRo | [Chips007 原片／本地R0／HuRo核心全片对照](get_potato_chips_0915_007_R0_VS_HURO_RAW_DIAGNOSTIC.mp4) | 378 | 同输入同渲染；HuRo关节限位失败，非产品；SHA `1ea789f1…38a5` |
| HuRo | [Poker031 原片／本地R0／HuRo核心全片对照](play_cards_0915_031_R0_VS_HURO_RAW_DIAGNOSTIC.mp4) | 149 | 031左侧无合法输入；HuRo关节限位失败，非产品；SHA `9045c53a…cc71` |

Scene 的独立失败收据：[007](../../../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/review_007_v1/RESULT.json) · [031](../../../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/review_031_v1/RESULT.json) · [103](../../../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/review_103_v1/RESULT.json) · [042](../../../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/review_042_v1/RESULT.json)。传感器来源、视频SHA、HandMotion与466帧核对见[支线2结果](../../../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/sensor/run_0001/RESULT.json)。

新视频帧数、输入与SHA见[Chips103 R0](../../../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/motion/review_0902_103_v1/RESULT.json)、[Poker042 R0](../../../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/motion/review_0902_042_v1/RESULT.json)、[Chips007 HuRo对照](../../../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/huro/review_get_potato_chips_0915_007/RESULT.json)、[Poker031 HuRo对照](../../../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/huro/review_play_cards_0915_031/RESULT.json)。HuRo 的[关节限位拒绝](../../../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/huro/HURO_TERMINAL.json)是硬门，不因视频可播放而晋升。

## 四张问题关键帧

| 会话 | 图像 | 重点 |
|---|---|---|
| Chips007 | [设备与物体](ISSUE_007_DEVICE_OBJECT.jpg) | 白色指套/线缆漏检；写入区碰薯片候选。SHA `cc22bcc9…b168f` |
| Poker031 | [设备与物体](ISSUE_031_DEVICE_OBJECT.jpg) | 单侧有效性及设备漏检；牌区保护未获独立验证。SHA `bba927a0…ed52` |
| Chips103 | [设备与物体](ISSUE_103_DEVICE_OBJECT.jpg) | 灰色腕带可见而设备 Mask 全片为空。SHA `c2ce3f22…36ba9` |
| Poker042 | [设备与物体](ISSUE_042_DEVICE_OBJECT.jpg) | 灰色腕带漏检，部分物体帧无 Mask。SHA `91273103…73a49` |

四会话同一配方的[Scene 终态评估](../../../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/lanes/scene/SCENE_FINAL_ASSESSMENT.json)记录了空帧、写入区交叠、未启动 ProPainter 的原因及最小解阻条件。

## 尚未交付的产品与原因

四会话产品 Clean 输入门均未过，因此**目前没有可称去人产品的 `robot.mp4`**。四条单配方 Mask 全片审阅均为质量C，设备漏检、物体保护来源不足；ProPainter 未启动。四会话 R0 已完成来源绑定，但数值质量未通过；031 左侧没有合法独立模型输入。HuRo 核心腕目标可运行，但已触发关节限位硬门；两条全片同原片对照均为失败诊断，不能当产品。

现有 Depth 只覆盖007 4/378帧、103 4/284帧、042 3/171帧，031无同域Depth；007标定 `REJECTED_HELDOUT`。所以接触R1与深度遮挡均未获证实，ContactHints为UNKNOWN，不用合成图像反推几何。

本轮实际为 **11/11条审阅视频、4/4张问题拼图、0/4条去人产品**；[任务终态收据](../../../../_run/current/four_stream_visual_delivery_v5/attempts/attempt_0001/RESULT.json)记录质量阻塞。计划目标没有被偷换为完成质量；不能用历史视频填补纯产品缺口。
