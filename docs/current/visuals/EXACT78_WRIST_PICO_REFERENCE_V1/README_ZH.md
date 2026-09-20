# exact78 Chips023：PICO 腕点作为用户指定参考

- [420 帧完整视频](CHIPS023_PICO_ASSUMED_REFERENCE_FULL_REVIEW.mp4)
- [6 帧接触表](CONTACT_SHEET.jpg)
- [2D/3D 数值](METRICS.json)

本页按用户明确授权，把 PICO/OpenXR wrist 当作该工程比较的参考值；这不等于实验室测量或 MoCap authority。视频顶部直接给出 HaWoR 相对 PICO 的全片均值。

| HaWoR − PICO | 左手 | 右手 |
|---|---:|---:|
| 2D 像素误差均值 | 237.7 px | 175.8 px |
| 相机 3D 欧氏误差均值 | 90.3 mm | 105.4 mm |
| signed mean ΔZ | -54.1 mm | -83.1 mm |

2D 是同一 selected-camera 图像域内两个腕点的像素欧氏距离；3D 是同一 camera-optical
XYZ 中 `HaWoR − PICO` 的欧氏距离；`ΔZ` 只是其中的光轴深度分量。比较直接使用
`T_wrist_to_camera` 与 HaWoR MANO wrist，没有做事后贴合，也没有消费已撤权 Stereo 深度。

确定结论：该 exact78 样本的 HaWoR 绝对 wrist 定位不能作为毫米级 Robot/Contact 标签。时序平滑不等于绝对位置正确。

视频中 Stereo 栏仅为历史谱系，已因旧 exact78 Depth 重复 lens-undistortion 而撤权，不能用于深度精度结论。完整视频解码 420/420 帧，SHA-256 `f156f4d37e45cd3c50d3f56c115c3dd90e39ecea1e24f755fa613aa1677e8e20`。
