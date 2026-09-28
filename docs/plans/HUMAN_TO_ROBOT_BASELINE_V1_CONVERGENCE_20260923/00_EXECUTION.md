# Human→Robot Baseline v1 四支线收敛执行合同

本合同承接已经终态化的 `human_to_robot_evidence_unlock_s2_20260923`，不修改或复活 S2。

## 固定范围

- 12 小时墙钟预算；H3/H6/H9 是进度检查点，不是等待点。
- 四条 lane：`scene`、`sensor`、`motion_product`、`compare`。
- 主产品：`get_potato_chips_0915_007`、`play_cards_0915_031`。
- 回归：`get_potato_chips_0902_103`、`play_cards_0902_042`。
- Sensor：`play_cards_0916_097/098/101`。
- 只使用现有模型、权重、数据、机器人资产与运行器；不训练、不下载新模型、不操作真机。
- 所有新增文件、缓存和临时文件必须位于 `/mnt/workspace/code/chaoyang/`。

## 冻结算法门

- 031 腕位置：同定义手根欧氏残差 `<= 20 mm`。
- 获准支持方向：逐方向夹角 `<= 15 deg`。
- 完整旋转：SO(3) 测地误差 `<= 15 deg`，与部分方向门独立。
- `5 mm` 仅为旧求解损失尺度，不是质量门。
- 方向资格在读取新 IK 输出前冻结；无资格记 `NOT_EVALUATED`。
- 固定相机/base/placement、68.4 mm 完整开发安装合同、机器人资产、关节限位及手指 q。

031 先运行 66–81 帧右手小窗。只有小窗所有有方向资格的输入逐帧通过位置、支持方向、原限位及已声明碰撞范围，并且有交付余量，才允许对 149 帧时间轴中的 102 个原有效右手输入运行完全相同配方。左手和原无效帧不补值；全片失败不返回调参。

## 四线工作

1. Motion/Product：方向资格、硬位置 IK、小窗和条件全片；独立检查左手 RGB→ROI→模型输入；按会话产品会合及正式入口/resume。
2. Scene：007 持续 top-1 线缆实例及一次有证据的小窗 Clean；031 牌边独立问题；时序同表面遮挡；Contact 三层漏斗和条件 R1；连接件碰撞能力核验。
3. Sensor：复用三会话数组和共同后端，修合法线段裁切、等比例世界视图、朝向和真实 FK 面板。
4. Compare：同目标、同安装、同评价集合的有限 Local/HuRo 数值结论；目标合同不同必须披露，不能宣布总体赢家。

缺失项只阻塞真实消费者。007 与 031 不设全局 AND 门。初期最多两个 CPU 重任务，CPU 总软上限 8，GPU 单租约。

## 证据和交付

所有结果分别发布 execution、structure、quality、improvement、adoption。Contact 分列 `qualification_checked`、`geometry_screened`、`r1_executed`。

固定 15 个交付槽位：4 条 Scene/Clean、3 条 Sensor、2 条 Motion/Product、2 条 Local/HuRo、4 条纯 Robot 产品。槽位可为 NEW、REUSED、REJECTED 或 NOT_EVALUATED；不为凑数量重算或复制改名。

全部结果保持：

```text
OFFLINE_VISUAL
training_eligible=false
control_ground_truth=false
physical_deployable=false
external_metric_authority=false
```

