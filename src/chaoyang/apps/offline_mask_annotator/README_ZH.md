# 离线 Mask 标注器

这是一个可以整目录拷到 Windows 或 Linux 电脑上运行的本地标注工具。界面在本机浏览器打开，服务默认只监听 `127.0.0.1`，不会上传视频或标注。

## 你只需要改三项

打开 [config.json](config.json)，修改顶部三个值：

```json
{
  "input_mp4": "D:/videos/task01.mp4",
  "output_dir": "D:/annotations/task01",
  "session_id": "task01_take01"
}
```

Linux 路径例如 `/data/videos/task01.mp4`。普通视频不需要改下面的高级配置，默认固定抽取 24 帧。

## 环境准备

- Python 3.10 或更高版本；
- FFmpeg，且命令行能运行 `ffmpeg -version` 和 `ffprobe -version`；
- 第一次启动时能从 Python 包源安装锁定版本的 Pillow。以后可以完全离线运行。

Windows 安装 FFmpeg 后，需要把含 `ffmpeg.exe`、`ffprobe.exe` 的 `bin` 目录加入 PATH。Ubuntu/Debian 可运行 `sudo apt install ffmpeg python3-venv`。

## 一键启动

Linux：

```bash
cd offline_mask_annotator
chmod +x start_linux.sh
./start_linux.sh
```

Windows：双击 `start_windows.bat`。

脚本会建立本目录下的 `.venv`、安装锁定依赖、读取 MP4、固定抽帧并打开浏览器。再次启动会校验原视频、session 和抽帧身份，然后从已有 JSON 继续，不会覆盖不同项目。

也可手工运行：

```bash
python -m pip install -r requirements-lock.txt
python annotator.py run --config config.json
```

## MP4 什么时候已经足够

正常 CFR（固定帧率）、能被 ffprobe 完整读取的 MP4，只提供 MP4 就够。工具会读取每个解码帧的真实 PTS，再按解码帧序列固定抽 24 帧，不使用 `帧号 ÷ fps` 猜时间。

下列情况建议额外提供 `timestamps.csv`：

- VFR 视频需要按真实时间均匀覆盖，而不是按解码帧数量覆盖；
- 采集过程中掉帧、暂停或时间轴跳变；
- 需要与传感器、动作事件或另一台相机的指定时间对齐；
- 已经由采集协议预先规定了必须标注的时刻。

CSV 格式：

```csv
timestamp_seconds
0.500
1.250
2.000
```

时间必须严格递增，行数必须等于 `sampling.frame_count`。把 CSV 路径填入 `sampling.timestamps_csv`；工具会把每个时间解析到最近且不重复的解码帧 PTS，并将对应关系写入 manifest。

## 标注操作

- 多边形：`P`，单击添加点，双击或 Enter 闭合，Esc 取消；
- 画笔：`B`，按住鼠标涂画，`[` / `]` 调整半径；
- 撤销/重做：Ctrl+Z / Ctrl+Y；
- 左侧类别：`Q` 皮肤、`W` 袖口、`E` tracker；
- 右侧类别：`A` 皮肤、`S` 袖口、`D` tracker；
- 任务物体：`O`；不确定边界：`U`；背景/橡皮：`X`；
- 数字 `0`–`8` 也可直接选类别；
- 左右方向键切帧，`C` 标记本帧已完整检查，Tab 临时只看原图；
- Ctrl+S 保存。每次完成绘制也会短延时自动保存。

标可见表面，不补画被物体挡住的手或衣袖。tracker 的外壳和腕带归 tracker；衣服袖口归 sleeve；圆柱、扑克牌等正在操作的物体统一归 `O_TASK_OBJECT`。模糊到确实无法判断的窄边界才归 `U_UNCERTAIN`。

画面里只要出现 tracker，就必须标到对应的 `L_TRACKER` / `R_TRACKER`；如果这个新任务确实没有佩戴 tracker，相应类别为空是正常的，校验只给 warning，不会 FAIL。`U_UNCERTAIN` 也允许为空。

## 九类定义

| id | 类别 |
|---:|---|
| 0 | `B_BACKGROUND` |
| 1 / 2 | `L_SKIN` / `R_SKIN` |
| 3 / 4 | `L_SLEEVE` / `R_SLEEVE` |
| 5 / 6 | `L_TRACKER` / `R_TRACKER` |
| 7 | `O_TASK_OBJECT` |
| 8 | `U_UNCERTAIN` |

画布默认全是背景，后画的区域覆盖先画的类别，因此每个像素最终只有一个 class id。请不要把“默认背景”当成自动识别；完成一帧前仍要人工检查有没有漏掉细指、袖口和 tracker。

## 导出与校验

界面右上角可以“导出”和“校验”，也可以关闭界面后运行：

```bash
python annotator.py export --config config.json
python annotator.py validate --config config.json
```

`validate` 检查：原图身份与尺寸、标注是否保存并勾选完成、操作坐标、类别 id、导出 class-id 一致性、每类 binary 是否严格 0/255、九类是否互斥且覆盖全图、整个项目是否存在空类。它只能证明结构正确，不能自动发现“人眼漏标了一块袖口”；仍需要人工双人复核。

## 输出目录

```text
output_dir/
  FRAME_MANIFEST.json       # MP4哈希、ffprobe信息、抽帧index/PTS/PNG哈希
  frames/
    frame_0000.png ...
  annotations/
    frame_0000.json ...     # 可续标polygon/brush操作、complete、revision
  draft_class_id/
    frame_0000.png ...      # 每次保存同步更新，可续标的当前0..8草稿PNG
  export/
    class_id/
      frame_0000.png ...    # 0..8的8-bit PNG
    binary/
      B_BACKGROUND/ ...     # 每类0/255 PNG
      L_SKIN/ ...
      ...
    overlay/
      frame_0000.png ...
  EXPORT_MANIFEST.json      # 每个导出文件的PTS、尺寸与SHA-256
  VALIDATION.json           # PASS/FAIL、错误、警告与每类像素统计
```

## 自测

不会读取任何项目数据，只用 FFmpeg 生成 3 秒测试图视频：

```bash
python -m unittest discover -s tests -v
python annotator.py self-test
```

成功时最后输出 `"status": "PASS"`、6 个不同 PTS、九个 class id 和 `validation_status: PASS`。

## 拷贝注意事项

只拷贝 `offline_mask_annotator/` 代码目录。不要把真实 MP4 或真实 `output_dir` 放进这个代码目录再分发。每个任务使用代码目录外独立的输出路径。若换视频、session 或抽帧设置，请换一个新的 `output_dir`；工具会拒绝在旧项目上静默覆盖。
