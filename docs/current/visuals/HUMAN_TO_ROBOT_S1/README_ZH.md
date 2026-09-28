# S1最新可视化：局部Clean、031几何与Contact阻塞

更新于2026-09-23。S1已终态 `REJECTED_QUALITY / TERMINAL_WITH_QUALITY_GAPS`，当前无运行任务。本页只链接原始实体。[最终结果](../../../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/RESULT.json)及[15条视频完整解码/SHA清单](../../../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/FINAL_VALIDATION.json)为证据。下载交接包是明确标记的静态副本，不是新的current产物。

| 优先看 | 视频 | 能回答什么 |
| --- | --- | --- |
| 1 | [031附件→Clean，16帧](../../../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/attachment_clean_canary_v1/play_cards_0915_031/ATTACHMENT_CLEAN_REVIEW.mp4) | 66–81帧；附件支持被消费，但手—牌边界和补图仍失败 |
| 2 | [007附件→Clean，16帧](../../../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/attachment_clean_canary_v1/get_potato_chips_0915_007/ATTACHMENT_CLEAN_REVIEW.mp4) | 181–196帧；指端附件改善，长黄线残留与背景污迹仍在 |
| 3 | [031全片Depth，149帧](../../../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/geometry_contact/depth_full_v1/play_cards_0915_031/DEPTH_CANARY_REVIEW.mp4) | encoded物理左目域的内部开发级深度；文件名含canary，实际为149帧全片 |
| 4 | [031三张牌可见几何，149帧](../../../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/geometry_contact/object6d_visible_031_v1/OBJECT6D_VISIBLE_REVIEW.mp4) | 可见表面中心/平面/轴，各自可观测性；无隐藏完整牌体 |
| 5 | [031 Contact阻塞，149帧](../../../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/geometry_contact/interaction_contact_031_v1/INTERACTION_CONTACT_BLOCKER_REVIEW.mp4) | 橙色为inferred投影；没有合格接触窗口，不能当接触成功回放 |
| 6 | [007 Stereo preflight，12采样帧](../../../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/geometry_contact/encoded_stereo_preflight_v1/get_potato_chips_0915_007/ENCODED_STEREO_PREFLIGHT_REVIEW.mp4) | 对应点/极线检查；有效匹配1021低于1500，尚未运行Depth |
| 7 | [042附件追踪，94帧局部窗](../../../../_run/current/human_to_robot_quality_closure_s1_20260923/attempts/attempt_0001/lanes/scene_evidence/attachment_track_042_canary_v1/play_cards_0902_042/ATTACHMENT_TRACK_REVIEW.mp4) | 77–170帧，不是171帧全时间轴全部有附件证据 |

Sensor三会话、四条Robot候选、Local/HuRo两条同背景诊断请读[R2完整索引](../HUMAN_TO_ROBOT_R2/README_ZH.md)。这些为继承结果，不能列为S1新增执行。所有结果保持OFFLINE_VISUAL，训练/控制/部署/外部公制权限均为false。
