# Artifact index

## 最短入口

1. 修改 `config.json` 顶部的 `input_mp4`、`output_dir`、`session_id`。
2. Linux 运行 `./start_linux.sh`；Windows 双击 `start_windows.bat`。
3. 标完后在界面点击“导出”，再点击“校验”。

## 用户文件

- `README_ZH.md`：完整中文安装、标注、MP4/CSV选择和导出说明。
- `config.json`：唯一用户配置；正常 CFR MP4 只改顶部三项。
- `start_linux.sh`：Linux 一键建环境并启动。
- `start_windows.bat`：Windows 一键建环境并启动。
- `requirements-lock.txt`：唯一 Python 依赖的固定版本。

## 程序文件

- `annotator.py`：`prepare/serve/run/export/validate/self-test` 命令入口。
- `mask_annotator/media.py`：ffprobe frame PTS 清单、固定抽帧与 timestamps CSV 解析。
- `mask_annotator/project.py`：项目身份、原子 JSON/草稿 PNG、导出。
- `mask_annotator/raster.py`：polygon/brush 到互斥 class-id mask。
- `mask_annotator/server.py`：仅监听本地的浏览器 API。
- `mask_annotator/validation.py`：结构、尺寸、哈希、空类和导出一致性检查。
- `web/`：无外部 CDN 的浏览器 GUI。

## 审计文件

- `PROTOCOL.md`：隐私、采样、标签、续标、导出协议。
- `SMOKE_TEST_RESULT.json`：合成视频、PTS、9类、导出、校验与本地 API 测试结果。
- `tests/test_core.py`：零项目数据的核心单元测试。

## 可移植包

发布 zip 与本目录同级，名称为 `offline_mask_annotator_portable_v1.zip`。压缩包只含上述代码、文档和静态网页，不含 MP4、项目帧、真实标注、缓存或虚拟环境。
