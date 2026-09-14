# `<task_id>` 任务

本目录只保存这个任务的配置、输入只读引用、标注、运行和报告。算法与权重统一从
`../../systems/` 引用，禁止把 checkpoint 复制进本目录。

## 开始前

1. 编辑 `task.manifest.json` 的 task ID、物体、几何、可见 role、Clean 保留对象和输入
   SHA；不要沿用另一个任务的 anchor。
2. 正常 MP4 的离线 Mask 标注默认 24 帧；小 tracker 或已有失败区间按证据增加预注册
   关键帧。标注帧必须在看最终结果前冻结。
3. Clean 最好在同一次录制、同相机/姿态/曝光下增加 2–3 秒无手 clean plate。
4. 运行 `python tools/validate_task_registry.py` 后，才写 fresh
   `tasks/<task_id>/runs/<system>/<run_id>`。端到端 `runs/pipeline/<id>/manifest.json`
   只能引用各 stage run，不复制产物。

每次结果在本页只链接，不把运行视频和逐帧文件复制到任务根目录。
