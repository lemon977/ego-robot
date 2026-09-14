# 通用 Robot + Clean compositor V1

## 状态

`PENDING_GENERAL_OBJECT6D_COMPOSITOR` 的**接口与 CPU compositor 实现**已经关闭。当前关闭的是“通用程序是否存在、是否去掉 chips001/799 硬编码、是否能严格验证所有上游 authority”这一项，不等于真实扑克/薯片 Robot 视频已经可正式发布。

真实视频仍缺 measured Object6D、正式 Mask/Clean、正式 functional-retarget/nonpenetration 和对应 renderer buffers。缺任一项都在 preflight 阶段 fail closed，不会先渲染再补证据。

## 唯一入口

```bash
python tools/compose_robot_clean_general.py \
  --input-manifest <immutable_COMPOSITOR_INPUT.json> \
  --output-root tasks/<task>/runs/robot/<new_run_id>
```

只做检查、不写视频：

```bash
python tools/compose_robot_clean_general.py \
  --input-manifest <immutable_COMPOSITOR_INPUT.json> \
  --validate-only
```

CLI 故意没有 `--task-id`、`--session-id`、`--start-frame`、`--end-frame`、颜色、深度阈值或物体 override。所有身份和帧范围只能来自一个 SHA 固定的 manifest，防止在运行时把 poker/chips、不同 session 或局部帧拼错。

主 schema 是 `contracts/robot_clean_compositor_input_v1.schema.json`，填写入口是 `tasks/templates/ROBOT_CLEAN_COMPOSITOR_INPUT.template.json`。

## 必须同时满足的输入

| 输入 | 关键硬门 |
|---|---|
| canonical task manifest | 必须是 `tasks/<task_id>/task.manifest.json`；Robot system manifest pin 当前 SHA 必须一致 |
| session manifest | task/session/frame IDs/timestamps/fps/resolution/object IDs 全冻结；formal 与 synthetic 不得混用 |
| Object6D NPZ | 每帧、每物体有效；`T_object_to_camera` 和 `T_world_object` 都是右手 SE(3)；与 c2w 满足 `T_world_object=T_camera_to_world@T_object_to_camera` |
| camera NPZ | 每帧 `K`、`T_camera_to_world`，frame ID 和 hardware timestamp 与 Object6D/trajectory 完全一致 |
| formal Clean master | 只接收单视频流、无音频、固定 fps/尺寸/长度的 FFV1；Clean authority、Mask authority、pixel provenance 三者 SHA 串联 |
| formal Mask/Object-depth authority | object ID/order 与 task 完全一致；2x optical-Z near/far、object index、independent visible mask 逐帧 bundle；near/far/index support 必须相同 |
| Robot trajectory | `q_arm[N,2,D]`、`q_hand[N,2,D]`，frame ID/timestamp 全对齐 |
| Robot render | 2x sRGB BGR、Euclidean camera Range、straight alpha；逐帧 bundle 与 trajectory/camera/PBR/nonpenetration SHA 串联 |
| PBR | 只能引用共享 `robot.pbr_palette.004ref.v1`，task/session/frame RGB override 被拒绝 |
| nonpenetration result | 必须覆盖同一组 frame IDs；trajectory/Object6D/geometry SHA 一致；dense mesh、named-pad SDF、self-SAT、joint limits、true-dt 全通过 |
| cross-stage admission | formal 时 `mask_formal_ready`、`clean_formal_ready`、`robot_sidecar_ready`、`robot_composite_execution_allowed` 全为 true |

所有 artifact 必须是 `chaoyang` 内的 project-relative、普通且非 symlink 文件，并同时匹配 bytes 与 SHA-256。逐帧 bundle 在 preflight 和实际消费时各重验；Clean 和顶层引用在结束前再次重验。

## 遮挡、alpha 与 codec

- Object depth 是相机 optical-axis Z（米），Robot 输入是相机 Euclidean Range（米），程序先按每帧 K 转换 Range→Z；两者禁止静默混用。
- 固定规则是 `object_near_z <= robot_z + 0.003 m` 时物体在前。3 mm 是冻结系统常量，不可从 task/session/CLI 改写。
- Mask visible support 用于独立一致性门，Object6D amodal depth 用于真正的前后排序；物体在前时从正式 Clean master 恢复物体像素。
- straight alpha 在由 sRGB 解码出的 linear Rec.709 中合成，然后 2×2 下采样并编码回 sRGB，避免直接在 gamma 编码值上混色。
- 输出同时包含 FFV1 lossless master 和 H264/yuv420p review，均重新 ffprobe 并完整 decode 校验帧数、尺寸、fps、codec 和无音频。
- formal 输入也只生成 `FORMAL_CANDIDATE...`，`consumption_authorized=false`；还要对新输出做人审并单独 promotion，不能由 compositor 自我授权。

## 资源边界

程序 CPU-only，显式清空 `CUDA_VISIBLE_DEVICES`；两个 encoder 各最多 2 threads。Raster 输入是逐帧 NPZ bundle，内存上界为一帧 Clean + 一帧 Object-depth bundle + 一帧 Robot bundle，不会把完整视频/depth timeline 载入内存。尺寸上限 4096×2160、总帧上限 200000，单帧 NPZ 上限 768 MiB。

## Synthetic 证据

构建器：

```bash
python tools/build_robot_clean_compositor_synthetic_fixture.py \
  --fixture-root tasks/control/runs/<run_id>/fixture \
  --task-id chips
```

Synthetic 使用非零且非连续 source frame IDs，chips 与 poker 都走同一 runner；扑克的三张牌 fixture 还共享一个 geometry SHA，用来验证合法共享模板不会被误拒。它只用于验证接口、遮挡、alpha 和 codec：

- 每帧视频上方和下方都烧录 `DEV_SYNTHETIC`；
- authority、结果和 cross-stage route 全部不可 formal promotion；
- 不读取上传 ZIP，不调用 GPU，不运行真实 Robot renderer；
- 不提供真实 contact、Object6D、Mask、Clean 或 Robot 成功证据。

历史 `tools/compose_robot_clean_full.py` 继续保留为 chips001/799 visual-only 证据，不是本接口的 fallback，正式运行绝不调用它。

