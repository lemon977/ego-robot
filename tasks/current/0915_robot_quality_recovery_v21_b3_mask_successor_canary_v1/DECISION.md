# B3 Mask successor 审计结论

结论选择 **A：单一冻结 CPU/GPU canary 包**，但范围仅限“纯视觉、几何候选诊断”，不等于 Mask 修复或质量通过。

## 审计结论

现有 B1/B1R 不能解决缺失能力：

- Poker044 左手仍是 `REJECTED_DIRECT_OBSERVED_ADMISSION`；右手是 `REJECTED_TRACKER_DIRECTION_INCOMPLETE`；两侧都不可消费。
- Chips097 左手没有 HaWoR 直接锚点；右手仅有 393/394 帧通过开发代理门。这不是像素精度证据。
- B1 和 B1R 的终态及 SHA 均固定为不可变父证据，本包不覆盖、不收养它们的 Hand 输出。

项目里没有一条已经具备消费权威的“独立 Hand + 物理 Tracker 身份”路线。已有 Hand 路线依赖 HaWoR、MANUS 或 Controller 做播种、左右分配或质量自证；native semantic-full 的独立证据只覆盖 Poker015 物体，不覆盖手或佩戴设备。

但已有两个可复用且真实存在的基础能力：固定 SAM3.1 checkpoint/adapter 的纯文本检测，以及 native full propagation。因此冻结一个单因素 canary 是可执行的：

```text
同一 resize-only RGB
├─ fresh SAM state: text="hand"
└─ fresh SAM state: text="wrist-worn device"
```

两路结果只形成 `HAND_CANDIDATE_UNASSIGNED` 和 `WORN_EQUIPMENT_CANDIDATE_UNASSIGNED`。不使用 HaWoR/MANO/PICO/旧 Hand Mask；不分左右；64 帧分块之间不拼身份；空输出是 UNKNOWN；任何两类 Mask 像素重叠都硬拒绝，禁止相减、合并或调阈值掩盖。

## 为什么仍不具备消费资格

纯文本输出只能回答“SAM 是否给出几何候选”，不能独立证明：

1. 候选确实是人手而不是前臂、物体或背景；
2. `wrist-worn device` 是同一物理 Tracker；
3. Hand Mask 没有吞入 Tracker；
4. 候选属于左手还是右手；
5. 无输出代表真实不存在。

所以即使 canary 所有几何门通过，也固定 `consumer_allowed=false`、`mask_accuracy_claimed=false`、`physical_tracker_identity_claimed=false`。只有后续获得独立的已版本化 Hand/佩戴设备身份来源或人工 Gold，才可另立晋升任务；不得用本 canary 自证。

## 冻结执行范围

- Poker044：166 帧，分成 3 个连续窗口。
- Chips097：394 帧，分成 7 个连续窗口。
- 模型、checkpoint、adapter、RGB、B1/B1R 父终态全部按 SHA 固定。
- 单 checkpoint、单提示合同、无参数扫描、无模型替换。
- 不读取 Object/Clean，不修改任何 Object/Clean 产物。
- 输出 packed 候选、逐窗口 ledger、逐帧分离门和全片 review MP4；仅供离线复核。
- CPU `--preflight-only` 在不加载模型的情况下验证全部代码/权重/父收据/RGB pin，并完整解码两条输入视频。
- 预检和正式模型输出都使用 final 同目录 staging；异常清理，成功时才以单次 `os.replace` 暴露 final。NPZ、视频和收据中的路径均预先投影为 final path。
- GPU 仅允许通过已固定 SHA 的 V7.1 lease wrapper 启动一次：等待 1800 秒、wall 3600 秒、最低空闲 49152 MiB。
- 浅层视频由独立 CPU publisher 从已提交 deep terminal 复制；源和目标均 full-decode 后原子提交 fresh 目录，GPU worker 不直接修改浅层目录。
- 每帧、每窗口、每会话分别记录 `PASS_GEOMETRY_ONLY`、`REJECTED_OVERLAP` 或 `UNKNOWN`；模型运行完成不会被写成 Mask pass。

本地单元测试 13/13 通过；模型没有执行，远端没有写入。
