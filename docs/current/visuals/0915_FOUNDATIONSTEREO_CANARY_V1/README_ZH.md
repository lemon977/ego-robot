# 0915 FoundationStereo 单会话质量复核

状态：`REJECTED_QUALITY`

会话：`play_cards_0915_001`

这是物理双目、同会话标定和冻结 FoundationStereo checkpoint 上的完整 150 帧
bounded canary。模型单次加载，完成 300 次推理及整段 SBS/审阅视频解码；运行、GPU
租约、writer fence 和发布均正常。拒绝原因是深度质量门，而不是环境或运行失败。

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

因此 `consumption_authorized=false`、授权 scope 为空，结果不能作为
`VISUAL_OBJECT6D_CANDIDATE_INPUT`。Planar Object6D 没有启动，也没有通过降低门限或
补造 confidence 绕过拒绝。当前深度只声明 rectified-left optical-Z 内部量纲一致，外部
毫米精度仍为 `UNVERIFIED`。

