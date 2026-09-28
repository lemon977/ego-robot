# 使用说明

## 1. 这是任务补充，不是远端运行成果

主文件是00_继续执行与最终验收.md。将它作为已有唯一操作入口的本轮执行章节，保留旧任务历史，不建立第二个current。
01合并进实际生效的项目AI规则，不覆盖用户原AGENTS。02–05分别是四条线的短卡；06是待接入项目的算法验收和故障注入要求。

建议交给执行AI的启动文字：

> 按00执行本轮补充。先加载实际适用规则，将01合并为本轮约束，冻结DELIVERY_CONTRACT及评价门。只核验当前消费的文件，不全库审计。按各线短卡从失败复现开始，生成实际候选和消费者结果，不只补文档。禁止替换007/031、Raw冒充Clean、代理冒充CAD、手填PASS。每次修复有失败测试、差异、复跑和签名。结构合法允许候选集成，质量门决定采用；R1/Depth不能阻断Clean和R0。实际任务按已有治理登记新执行实例，不改旧预算/收据。运行结束由独立复核和现有publisher发布，没完成就列缺项，不能把终态叫成功。

默认执行者只读01＋本线卡＋最新摘要。主协调者另读00、06和交付清单。不要让每个worker每次读全包。

## 2. 文件分工

- 00：唯一主计划，定义事实锁、最终章节、边界和时序。
- 01：短执行规则，合并到项目实际规则机制。
- 02–05：四线任务卡。
- 06：项目算法验收矩阵，需接入已有测试入口；不是已经运行过的测试。
- DELIVERY_CONTRACT.json：固定期望15个视频产物，即4纯产品和11审阅；不是状态账本。
- DELIVERY_TEMPLATE_NOT_RUN.json：空模板，故意不能通过检查，不含虚构成功结果。
- check_delivery_inventory.py：只读文件/身份/哈希/帧数清单检查。
- tests_inventory.py / CHECKER_TEST_RESULTS.txt：该检查器自身的测试和本容器执行记录。

## 3. 把检查器接到现有publisher，而非新建调度框架

当前publisher应从真实生产者/消费者结果导出一个最小清单视图。不要由执行AI手填成功字段，也不要把该清单变成另一份authority。

交付JSON在模板基础上填入实际run_id和artifacts。每个artifact含：

```json
{
  "artifact_id": "product:get_potato_chips_0915_007",
  "session_id": "get_potato_chips_0915_007",
  "kind": "robot_product",
  "file": {"path": "实际相对或绝对MP4路径", "sha256": "实际64位SHA"},
  "receipt": {"path": "实际最小渲染收据JSON路径", "sha256": "实际64位SHA"}
}
```

最小渲染收据视图含：artifact_id、session_id、kind、media_sha256、frame_count、output_frame_ids(0..N-1)、source_frame_ids、timestamps_ns、source_ref(path/SHA)。
source_ref指向由真实Session导出的JSON视图，含session_id、frame_count、source_frame_ids、timestamps_ns；与渲染收据逐项相同。

纯robot_product还须有background.kind=CLEAN_SYNTHETIC和background.result_ref(path/SHA)，指向同会话完整Clean来源收据视图。后者含session_id、frame_count、execution_state=COMPLETED、model_invocations>=1。复用时指向有真实模型执行证据的原始Clean来源收据，而不是捏造一次新调用；REUSED/FRESH仍在原项目记录。

这里对“Clean消费/执行”的检查依赖导出的真实收据。它不能靠JSON字段反推出模型是否诚实运行，因此真实调用链验证、冻结producer签名和独立复核不能省。

所有相对path都相对实际交付JSON的所在目录解析，包括嵌套source/clean引用。MP4不允许符号链接。

## 4. 实际可运行的检查命令

环境：Python 3.10+，标准库；完整媒体检查另外使用现有ffprobe/ffmpeg，不安装模型、不访问网络。

先在本目录测试检查器自身：

```bash
python -m unittest -v tests_inventory.py
```

在正式执行开始前，由合同owner记录DELIVERY_CONTRACT.json的SHA到既有冻结运行记录。发布时用这一枚已冻结SHA，而不是修改清单后重算一枚让自己通过。

```bash
python check_delivery_inventory.py \
  --contract DELIVERY_CONTRACT.json \
  --contract-sha256 '<执行前冻结的64位SHA>' \
  --delivery '<publisher导出的实际交付JSON>' \
  --media
```

上面两个尖括号参数必须来自实际记录，不是已经存在的项目路径。命令返回0只代表该清单技术检查通过；非0打印具体错误。省略--media只做清单/来源/哈希，不会声称视频已完整解码。--media串行检查视频，解码器限制1线程。

检查器不写文件、不修改authority。需要保存输出时，由现有运行器保存stdout。

## 5. 明确不能用它证明的事情

它不验证Tracker语义、真实Mask召回、ProPainter视觉质量、原像素回贴、NPZ关节含义、FK、碰撞、接触、图形深度转换、真实source视频像素身份或算法优劣。source_ref校验的是既有Session导出视图和消费者收据一致，不是重新审计全部原数据。

它也不能防止所有人合谋伪造JSON，或阻止拥有相同文件权限的AI故意改所有证据。它面向偶发错会话、漏交、过期/错SHA、空Clean、时间轴错误、错误数量与媒体损坏。实际防误判仍需：冻结合同、不同职责、运行器产生记录、真实功能测试、独立可视复核。

## 6. 本包测试记录的含义

CHECKER_TEST_RESULTS.txt记录本容器对检查器运行20项测试通过，含真实生成的两帧视频解码，以及“全部元数据声称三帧但实际只有两帧”的拒绝测试。
这些夹具在临时目录中产生并清除，不属于chaoyang项目，不是007/031产品，也不是远端验收。远端尚需接入既有publisher并运行自己的实际交付与算法验收。
