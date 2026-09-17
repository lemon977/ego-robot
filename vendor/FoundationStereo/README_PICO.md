# PICO 双目深度：FoundationStereo

本目录是 NVIDIA 官方 `FoundationStereo`，另外增加了 PICO 双目入口。输入是
`pico_data_pipeline` 裁剪结果中的原始双目鱼眼视频，输出米制深度、视差和点云。

## 已配置的位置

- 项目：`/cpfs_infra/user/chenxianchi/code/FoundationStereo`
- Conda 环境：`/cpfs_infra/user/chenxianchi/miniconda3/envs/foundation_stereo`
- 依赖缓存：`/mnt/workspace/code/chaoyang/_cache/foundationstereo/uv`
- 本地 GPU wheels：`/mnt/workspace/code/chaoyang/_cache/foundationstereo/wheelhouse`
- timm / DINO 运行缓存：`/mnt/workspace/code/chaoyang/_cache/foundationstereo/model_cache`
- ViT-Large 权重：`pretrained_models/23-51-11/model_best_bp2.pth`
- PICO 入口：`scripts/pico_stereo_depth.py`
- 双目标定缓存：`calibrations/`

脚本读取：

```text
<clip>/
├── camera_params.json
└── source_stereo/
    └── CameraRecord_*_stereo.mp4
```

## 第一次安装

当前服务器已经配置过；重建环境或权重丢失时才需要执行：

```bash
cd /cpfs_infra/user/chenxianchi/code/FoundationStereo
./setup_foundation_stereo.sh
```

Conda 环境安装在 Miniconda 的标准 `envs` 目录；依赖下载缓存和 3.3 GB
权重保留在 NAS，`pretrained_models` 是项目内指向权重目录的软链接。
安装脚本会续传权重并校验 SHA-256，未下载完整的文件不会被当成可用模型。
安装使用并发下载并保留大型 wheel；如果安装过程被中断，重新执行同一命令即可。

安装后可执行以下命令检查 GPU：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/envs/foundation_stereo/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 先检查双目校正

```bash
cd /cpfs_infra/user/chenxianchi/code/FoundationStereo

/cpfs_infra/user/chenxianchi/miniconda3/envs/foundation_stereo/bin/python \
  scripts/pico_stereo_depth.py \
  --clip-dir /mnt/workspace/code/chaoyang/archive/external_nas_legacy_20260914/pico_new/pick_place_005 \
  --frame 150 \
  --output-dir /mnt/workspace/code/chaoyang/archive/external_nas_legacy_20260914/foundation_stereo_outputs/pick_place_005_frame_000150_prepare \
  --prepare-only
```

查看 `rectification_check.png`：同一个物体在左右图中应落在同一条黄色水平线上。
`manifest.json` 的 `median_vertical_error_px` 越接近 0 越好。

## 运行深度

推荐先用半分辨率和 16 次迭代确认效果：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/envs/foundation_stereo/bin/python \
  scripts/pico_stereo_depth.py \
  --clip-dir /mnt/workspace/code/chaoyang/archive/external_nas_legacy_20260914/pico_new/pick_place_005 \
  --frame 150 \
  --output-dir /mnt/workspace/code/chaoyang/archive/external_nas_legacy_20260914/foundation_stereo_outputs/pick_place_005_frame_000150 \
  --scale 0.5 \
  --valid-iters 16 \
  --z-near 0.1 \
  --z-far 3.0
```

输出：

```text
left_rectified.png       # 去畸变、极线校正后的左目
right_rectified.png      # 去畸变、极线校正后的右目
rectification_check.png  # 带水平线的双目检查图
disparity_px.npy         # 像素视差
disparity_vis.png        # 视差可视化
depth_meter.npy          # 米制深度，无效处为 NaN
depth_vis.png            # 深度可视化
cloud.ply                # 左相机光学坐标系点云
manifest.json            # 输入、内参、基线、质量和运行参数
```

点云坐标系为校正后左相机光学坐标系：`+x` 向右、`+y` 向下、`+z`
向前，单位为米。转到片段固定世界坐标系时，使用对应帧左相机的 `c2w`：

```text
p_world = c2w_left(frame) @ p_left_camera
```

## 高质量模式

```bash
/cpfs_infra/user/chenxianchi/miniconda3/envs/foundation_stereo/bin/python \
  scripts/pico_stereo_depth.py \
  --clip-dir /mnt/workspace/code/chaoyang/archive/external_nas_legacy_20260914/pico_new/pick_place_005 \
  --frame 150 \
  --output-dir /mnt/workspace/code/chaoyang/archive/external_nas_legacy_20260914/foundation_stereo_outputs/pick_place_005_frame_000150_full \
  --scale 1.0 \
  --valid-iters 32 \
  --z-far 3.0
```

## 标定缓存与注意事项

入口会用若干同步双目帧估计固定的左右相机旋转，从 `camera_params.json` 读取
米制基线，再联合执行 `equiDis62` 去畸变和极线校正。结果按标定文件内容哈希
缓存到 `calibrations/`。同一套相机后续直接复用；若设备或标定变化，加入
`--recalibrate`。

- 左右输入不能交换，否则视差方向和深度会错误。
- 深度关系为 `Z = fx_rectified × baseline / disparity`。
- `--scale 0.5` 适合批量预检；最终物体点云建议用 `--scale 1.0`。
- SAM2 应先得到左目 mask，再用 mask 从 `depth_meter.npy` 取物体点云。
- 构造网络时使用仓库内的 DINOv2 源码；完整 checkpoint 会覆盖初始化权重，
  所以日常推理不依赖 GitHub 或 Hugging Face 网络。
- 官方开源许可证限制商业使用；商业用途需使用 NVIDIA TAO 版本或确认许可。

## 本机验收结果

已用 `pick_place_005` 第 150 帧在 NVIDIA H20 上实际跑通。半分辨率结果为
`640×480`，有效深度像素占 `94.85%`，深度中位数 `0.721 m`；双目校正后
垂直误差中位数 `0.374 px`、P90 为 `1.618 px`，共输出 `72780` 个点。
离线构造网络后重新推理，与首次结果逐像素完全一致。验收输出位于：

```text
/mnt/workspace/code/chaoyang/archive/external_nas_legacy_20260914/foundation_stereo_outputs/pick_place_005_frame_000150_offline
```
