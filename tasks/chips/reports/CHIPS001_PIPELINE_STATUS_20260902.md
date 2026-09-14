# 薯片 001：HaWoR → Mask → Clean → Robot

会话：`get_potato_chips_0901_001`，799 帧，1280×960，25 fps，31.96 秒。

## 可直接查看

- [完整 Robot 视频](../runs/robot/20260902_chips001_clean_robot_full799_v1/CHIPS001_ROBOT_ON_CLEAN_FULL799_VISUAL.mp4)
- [Robot 12 帧总览](../runs/robot/20260902_chips001_clean_robot_full799_v1/CHIPS001_ROBOT_ON_CLEAN_FULL799_12FRAME.png)
- [完整 Clean 视频](../../../data/processed/chips/get_potato_chips_0901_001/clean/20260902_v3/CHIPS001_CLEAN_PLATE_FULL799.mp4)
- [Clean 前后 12 帧](../../../data/processed/chips/get_potato_chips_0901_001/clean/20260902_v3/RAW_VS_CLEAN_12FRAME.png)
- [Mask 12 帧](../runs/mask/20260902_get_potato_chips_0901_001_assisted_v8/MASK_ASSISTED_CANARY12.png)

## 阶段结论

1. HaWoR：PASS。仓内代码与权重完成全 799 帧 MANO21；左 799 帧直接观测、右 797 帧直接观测，
   f238/f746 由显式 infiller 补齐，且逐帧保留 provenance。输出 SHA-256：
   `bf101a2d51ecd8d42ab091e6941eaaf41ad9eb36e4665272489a42fe95ef38e3`。
2. Mask：PASS（assisted）。799 帧双手/前臂、双 tracker 与腕带上下文、可见薯片/碗保护均完整；
   `consumption_authorized=true`，发布 human/object overlap 最大值为 0。
3. Clean：PASS（visual assisted）。`base_001` 的完整 stereo 视频成员和相机参数已在工程内固定；
   单一任务级桌面标定、每帧自动注册，799 帧无 homography fallback。手区覆盖 min/median=1/1，
   保护物体与桌外区域改动均为 0。完整视频已全解码。
4. Robot：PASS（visual full timeline）。每 10 帧加末帧共 81 个锚点，左右 162 个实际 FK 位姿均过
   10 mm/5°。两次凸时域滤波后，臂速度/加速度最大 0.04841/0.02804 rad/帧，手最大
   0.10749/0.04857 rad/帧，均过 0.12/0.06。完整视频 799 帧全解码。

## 权威边界

- `base_001.zip` 整包仍在上传时，只消费了已完整写入、可独立解码且 CRC/哈希固定的 stereo
  视频和相机参数成员；未把半包当完整 archive。
- Mask 是共享 SAM3.1 基座加任务配置；不是薯片专用训练权重。
- Clean 是同场景视觉替换，不是 metric ground truth。
- Robot 使用一个全会话刚体基座、81 个实际 FK 锚点和全片平滑插值；没有逐帧 base patch。
- 当前数据没有三片薯片与碗的 metric Object6D，所以视频只证明完整视觉链和连续运动，
  不证明逐帧机器人接触、非穿透或可执行抓取。

机器可读聚合清单见
[`manifest.json`](../runs/pipeline/20260902_chips001_completion_first_v1/manifest.json)；终端验收见
[`QA.json`](../runs/pipeline/20260902_chips001_completion_first_v1/QA.json)。
