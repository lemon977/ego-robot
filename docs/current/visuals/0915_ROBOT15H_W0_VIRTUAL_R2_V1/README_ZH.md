# 0915 Robot15h W0 Virtual R2 审阅

这些视频是固定虚拟底座中的独立 Robot-space 审阅，不是 RGB 叠加。动态输入只有 R0 `q22_init`、相对 wrist、有效位和时间戳。

固定安装关系仅为 `ASSUMED_DEVELOPMENT_PRIOR` visual proxy；实测安装、TCP、camera/world→base 都不存在。q22 原样传递，无效帧保持 NaN，相对腕部位移没有缩放。

- [get_potato_chips_0915_042](get_potato_chips_0915_042_KAI22_R2_VIRTUAL_REVIEW.mp4)：numeric candidate `True`，质量准入 `False`，numeric pass side-frame `535/710`。
- [get_potato_chips_0915_007](get_potato_chips_0915_007_KAI22_R2_VIRTUAL_REVIEW.mp4)：numeric candidate `True`，质量准入 `False`，numeric pass side-frame `85/378`。
- [play_cards_0915_031](play_cards_0915_031_KAI22_R2_VIRTUAL_REVIEW.mp4)：numeric candidate `True`，质量准入 `False`，numeric pass side-frame `2/98`。
- [play_cards_0915_119](play_cards_0915_119_KAI22_R2_VIRTUAL_REVIEW.mp4)：numeric candidate `True`，质量准入 `False`，numeric pass side-frame `124/134`。

由于 W0 的 R0 strict quality 全部拒绝，本轮 `r2_quality_admitted=0`；可视化和 IK 数值候选不能升级为 Robot 成功。碰撞仅为双侧有效帧上的 proxy-space Robot 自碰撞诊断，单侧帧不检查且不可质量准入；完整物体与环境均为 `UNVERIFIED`。
