# Removal Envelope V2 保守修补合同

状态：核心实现与合同测试 `PASS`；真实视频 Canary 尚未运行，质量 authority 为
`NOT_EVALUATED`。V1 已被人工复核判为 `REJECTED_QUALITY`，V2 是独立后继，不修改或覆盖
V1 证据。

V2 固定采用：

```text
Removal V2 = admitted SAM foreground + validated local repairs
```

SAM semantic mask 是唯一主体。MANO、foreground proposal 与 appearance 只能修补已经有
局部支持的缺失；证据不足时输出 `UNKNOWN` 或空 repair，不能主动生成大片擦除区。

## 四类有限来源

1. MANO 只连接 SAM 邻域内、两端都有 SAM 支持且长度受限的短缺口；禁止完整手指 capsule。
2. Sleeve 是紧贴 hand boundary、靠近 MANO 方向并具备时序共动的小块 repair；不是自由目标。
3. Forearm 只能从真实 foreground proposal 中选择 wrist-connected、形状和时序均合格的
   连通分量；禁止 wrist→图像边界生成。
4. Cable 每帧最多接受配置允许的 top-k 实例，评分绑定 anchor、细长度、路径、端点和双向
   时序；黄色只是当前设备 appearance profile，不是 cable 类别定义。

## 独立质量门

V2 逐帧发布 semantic base、各 repair、最终 removal、source bits 和 feather alpha，并检查：

- background spill；
- removal/semantic area inflation；
- temporal area derivative；
- repair contribution ratio；
- 每类 repair 的来源与状态。

缺 stable-background hook、真实 foreground proposal 或 cable reverse-pass support 时，相应门
保持 `UNKNOWN_FAIL_CLOSED`，不得宣称质量通过或启动扩批。

## 权限边界

`removal_envelope_v2` 与 `feather_alpha_v2` 只允许进入 `CLEAN_MASK_QA` 和
`VISUAL_INPAINT`。它们禁止进入 Depth、Object6D、Contact、Robot geometry 或 control
ground truth。FoundationStereo 与 Object6D 从 Raw/Stereo、相机参数和 admitted semantic
object evidence 独立推进，不依赖 Clean。

实现入口：

- `src/chaoyang/pipeline/removal_envelope_v2.py`
- `contracts/removal_envelope_v2.schema.json`
- `configs/systems/clean/removal_envelope_0915_play_cards_001_v2.json`
- `tests/pipeline/test_removal_envelope_v2.py`

当前只有合成合同测试证据，不存在 V2 真实视频或视觉 PASS 结论。下一步必须另建有限
runner 和新鲜 task packet，接入真实 proposal/background/reverse-pass 后再运行单样本 Canary。
