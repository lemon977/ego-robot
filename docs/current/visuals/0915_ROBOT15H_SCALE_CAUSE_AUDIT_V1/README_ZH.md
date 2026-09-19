# 0915 Robot15h 尺度原因审计 V1

`0.783730` 被复现，但它只是在 001 的冻结非接触样本上，将 MANO 前可见表面 optical-Z 映射到 Stereo 可见表面 optical-Z 的仿射斜率；不是骨长、世界尺度或机器人尺度。

- 左/右手分层斜率：`0.263` / `0.752`。
- 30 帧块斜率范围：`0.475–0.823`。
- 当前 encoded Depth 仍使用缩放 factory K 与 camera-centre baseline；encoded-domain P_L/P_R 外部精度未验证。
- 结论：`UNRESOLVED_NON_IDENTIFIABLE_CONFOUNDED_FIT`；不采用统一尺度修正，不改 5 mm Contact 门，不授权公制 wrist-object 平移。

下一步使用相互独立的 W0 录制取得新的 encoded-domain Depth 与直接可见手表面证据，不能继续调 001 自证。
