# V1运行、批量和恢复

唯一基线为[MANIFEST](MANIFEST.json)中的`INPAINTING_RAW_TO_ROBOT_V1`。唯一成片operation保留原兼容名称`run_exact78_pick_place_raw_to_robot_candidate_v1`；名字含candidate是旧接口标识，不是重新选一条候选，也不是无任务授权。

## 合法入口

先执行：

```bash
cd /path/to/chaoyang
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B -m chaoyang.cli validate-governance
```

新生产须使用现有`chaoyang.governance.execute_pick_place_1008_v1`登记全新有限Task Packet。该执行器为兼容旧接口，任务spec仍须放在`docs/plans/PICK_PLACE_1008_RAW_TO_ROBOT_V1/specs/`，task_id以`exact78_pick_place_`开头；这是固定登记路径，不是旧计划恢复授权。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B \
  -m chaoyang.governance.execute_pick_place_1008_v1 --check-only <本次全新spec绝对路径>
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B \
  -m chaoyang.governance.execute_pick_place_1008_v1 --execute <同一新spec绝对路径>
```

以上是接口说明，尖括号不是可直接执行的命令。本轮不登记新GPU重跑、不复活原199生产Task。

新spec的成片参数固定为：`--stage full --manifest <V1 inputs.raw_manifest> --camera-config <V1 inputs.camera> --mask-seeds <V1 inputs.mask_seeds> --sessions <明确会话ID>`；CPU池和预算沿父配方声明，输出由执行器生成新attempt。`full`会完成原始读取、动作、SAM、Clean、Robot和合成，不需要先读旧动作数组。原批次曾复用已封存source；如本次要求从原始重跑，则不传`--reuse-from`。未经核验的partial不能消费。

所有运行代码、权重、配置、原始／种子leaf输入和schema重新冻结。新包装SHA与历史包装不同已在V1清单记录；原算法字节不变不等于旧Task可继续执行。当前没有合法活动Task时，operation在导入算法之前拒绝。

## 批量范围

先支持冻结199的同输入域和既有配置。新会话需要合法原始清单、同类时间／设备字段、可审的遮罩种子及相机／安装适配，不宣称陌生会话已全自动。扩展输入域或改算法另开有限验证任务，不借“V1批量”跳过来源说明。

单GPU租约，CPU池在已冻结预算内。仍保存数值FAIL与逐帧UNKNOWN；Clean质量优化不是ICT研究训练的新硬门。每会话完整解码、帧身份、输出SHA和原stage收据独立验证后才计完成。

## 断线后

1. 从治理收据、当前INDEX和live PID/startticks确定真实状态；不要按旧PID文件或latest判断。
2. 同签名完整封存阶段可复用；未封存partial不可用。新代码或配置签名只使受影响阶段失效，不热切正在运行的任务。
3. 旧生产Task已封存，不能重启。需要恢复时登记新的有限successor，并引用明确的封存阶段。
4. 沙箱的`/proc`可能虚拟化进程标识；本轮首个维护尝试因此失败，后继publisher使用实际host上下文。不要把沙箱`ps`空结果当作host进程已死亡，也不能为方便忽略writer门。

旧`run_exact78_takeover_*`命令已从本支线入口退役；旧v13/V2/P1等参见[归档](../archive/inpainting_baselines/README.md)。0916 MANUS和ICT保留原入口，不能用旧视觉默认覆盖本V1。
