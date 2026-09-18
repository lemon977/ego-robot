# 0915 FoundationStereo 单会话质量复核

状态：`REJECTED_QUALITY / WITHDRAWN_WRONG_IMAGE_DOMAIN`

会话：`play_cards_0915_001`

> **不要把本视频当作正确 VST 画面或 Depth 基线。** 用户已确认所有 VST 编码视频本来
> 没有畸变；本任务却再次应用了 `equiDis62` lens-undistortion，所以左栏出现弯曲。

这是错误图像域上的完整 150 帧 bounded canary。模型单次加载，完成 300 次推理及整段
SBS/审阅视频解码；运行、GPU 租约、writer fence 和发布均正常。不可变执行终态仍保留
为 `REJECTED_QUALITY`，但其算法证据 authority 已撤销为
`WITHDRAWN_WRONG_IMAGE_DOMAIN`。

- [完整 150 帧深度审阅视频](0915_FOUNDATIONSTEREO_DEPTH_REVIEW.mp4)
- [Git 内结果收据](../../../../tasks/receipts/0915_FOUNDATIONSTEREO_SINGLE_SESSION_CANARY_V1_RESULT.json)
- [运行期质量汇总](../../../../_run/current/0915_foundationstereo_single_session_canary_v1/attempts/attempt_0001/DEPTH_SUMMARY.json)

## 质量门结果

通过：

- 几何有效覆盖中位数：`0.5845`，门限 `>= 0.40`
- 左右一致性可测试覆盖中位数：`0.5840`，门限 `>= 0.45`
- 最终有效覆盖中位数：`0.4336`，门限 `>= 0.25`
- 93.33% 帧的最终有效覆盖不低于 0.20，门限 `>= 90%`
- RGB 支持的视差边缘比例中位数：`0.6991`，门限 `>= 0.15`
- optical-Z 公式复算最大误差：`0 m`

失败：

- 左右一致性残差的跨帧 P90：`17.8653 px`，门限 `<= 5 px`
- 帧级深度中位数步长 P90：`0.4290 m`，门限 `<= 0.35 m`

此外，本运行的 lens-undistortion 违反当前 VST 编码域合同。因此无论数值门结果如何，
`consumption_authorized=false`、授权 scope 为空，结果不能作为
`VISUAL_OBJECT6D_CANDIDATE_INPUT`。Planar Object6D 没有启动，也没有通过降低门限或
补造 confidence 绕过拒绝。视频只保留为重复去畸变失败证据。
