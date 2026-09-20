# 验收测试与实片对照矩阵（V2.1）

这是待本地实现/运行的测试要求，不是“测试已通过”声明。复用现有测试体系，不为追求数量拆出大量低价值用例。原合同阈值除可证实的单位/语义错误外保持不变。

## 1. 证据与运行

| ID | 测试 | 必须证明 |
|---|---|---|
| E01 | 原R2技术合同恢复 | 4,438 bytes仅辅助定位；必须exact SHA一致；旧RESULT不改 |
| E02 | 治理刷新 | 改gov元数据后技术snapshot字节不变；真实安装值变化必须新hash |
| E03 | tombstone闭包 | 缺失资产可正式撤权闭包，旧R2仍不可采用；不等于恢复质量 |
| E04 | 运行许可 | 正确RGB的W1诊断推理不因W0 strict=0自动拒绝；错误域/权重仍拒绝 |
| E05 | 消费许可 | 同帧2D提示可用不授予3D/Contact权限；局部窗口不授予整片PASS |
| E06 | 租约与writer | 两个GPU任务无法同时持有；不同lane不写共享合同；只publisher发布 |
| E07 | 缓存 | 后处理变化不失效无关Depth；输入/域/模型/技术合同变化能正确失效 |
| E08 | 终态与预算 | 合法独立任务不因另一支失败整体终止；无预算不新启；只停本轮进程 |
| E09 | 数据访问围栏 | DIAG可决定候选但不能采用；ADOPTION打开前冻结且结果不能触发修改；H9/EXTRA FINAL未到门不得读取 |
| E10 | EXTRA FINAL固定 | 盐重算严格得到030/085/017/095；`sealed=true`、不可替换；扩批门失败保持未打开 |
| E11 | provenance纠正 | V2.1只声称metadata验证，无用户逐会话人工确认；旧manifest字节不改 |
| E12 | P0局部阻塞 | R2合同找不到只阻塞新R2，A/B/C lane可独立继续 |

## 2. HaWoR与R0

| ID | 测试 | 必须证明 |
|---|---|---|
| H01 | 坐标回映射 | crop/resize/pad/镜像点坐标与原RGB闭环；无旧去畸变 |
| H02 | 左右身份 | 交叉/丢失保留unknown，不能用最近邻强制反转 |
| H03 | MANO长度 | 固定beta的3D原生骨段与投影长度区分，不把透视变化当骨长漂移 |
| H04 | 观测层 | short_gap/noncausal恢复不增加observed；原raw字段保持 |
| H05 | 时间门 | 用实际dt；不跨长gap产生伪速度；度弧度分开；相机与世界运动分开 |
| H06 | 聚合门 | 一只手某帧缺失不把另一只手所有合法窗口归零；仍保留整会话原strict失败 |
| R01 | joint/FK | pinned左右Kai22顺序、轴、限位、单位正确；FK有限只是基础检查 |
| R02 | 碰撞与边界 | 检查非邻接自碰撞；原生robot几何不缩放；gap边缘显式invalid |
| R03 | 同样本对照 | 原raw、原bounded、新candidate、Robot四者固定帧对照；coverage不偷换 |
| R04 | 整片视频 | 输出帧数/时间轴与输入一致；invalid有标识，不能冻结上一姿态冒连续 |

## 3. SAM与Object

| ID | 测试 | 必须证明 |
|---|---|---|
| M01 | 无HaWoR物体启动 | 合法RGB+text/visual prompt能调用pinned SAM物体流程，不要求全片3D通过 |
| M02 | optional hand提示 | 合法同帧2D候选与SAM自身候选有明确来源，身份未知不硬分左右 |
| M03 | state生命周期 | 初始化、传播、reseed后实际state更新；旧错误缓存不继续冒名输出 |
| M04 | raw/admitted | 传播yield、raw非空、semantic通过是不同计数；unknown不写absent |
| M05 | 双向身份 | 正反向一致仅为检查；无法辨别相似牌时保留歧义 |
| M06 | 局部分母 | 固定可见评估集不随候选改；标更多unknown不能自动提升质量 |
| O01 | 模块独立 | object mask+depth可用时，不被hand/forearm/Clean失败拦截 |
| O02 | 可见中心 | 遮挡改变visible centroid不自动当物体真实移动或全局跳变 |
| O03 | 任务路由 | Poker有限平面；Chips不强制完整刚体；未知尺寸/隐藏面不补真值 |

## 4. Stereo/尺度/接触

| ID | 测试 | 必须证明 |
|---|---|---|
| G01 | 深度类型 | 已知合成点的optical-Z与ray length区别明确 |
| G02 | 投影矩阵 | crop/resize与K/P同变换；左右主点项及镜像回域不丢；不暗改3D手性 |
| G03 | 对应表面 | MANO背面/关节中心不能与可见Stereo表面误配；边缘/附件分层 |
| G04 | 旧门不变 | 不通过放小尺度下界或增加Contact距离容差求旧holdout通过 |
| C01 | 布尔条件 | 人体采样排除object后又要求同一uv在object内的错误若存在应被检出 |
| C02 | 正交足点 | 正交投影到平面与原始uv不同；有限patch检查使用正确坐标 |
| C03 | 真接触被遮挡 | 严格可见证据分支允许unknown；不得断言无接触或伪造hidden point |
| C04 | 非接触反例 | 近但分离、无限平面外、错物体/错finger、过大不确定度仍不能过严格门 |
| C05 | 唯一采样 | 同surface重复配三个object不能当三次独立观测 |
| C06 | 时间窗 | 固定hand/finger/object，真实连续frame_id和dt；缺帧不能压缩成连续 |
| C07 | tactile | 无合法时间/手指/物体关联不得授予具体接触点；event score非概率 |
| C08 | 遮挡回放 | 被隐藏数据及已消费其信息的时序缓存不泄漏；无法证明则不能标独立验证 |
| C09 | 双层公制资格 | `local_stereo_metric_dev` 与 `external_metric_authority` 严格分离，前者不能晋升后者 |
| C10 | 非同域5mm拒绝 | 任一同帧/同Depth/physical-left/K/P/baseline/identity/patch/LR/registration/uncertainty条件失败时，5mm不能授权R1-E |
| C11 | 防循环自证 | Contact窗口和阈值不得用于拟合alignment，再由同一距离证明alignment正确 |

## 5. R1与虚拟R2

| ID | 测试 | 必须证明 |
|---|---|---|
| R11 | E/H隔离 | R1-H不写严格Contact/R1-E，training_eligible=false；无新原因不重复旧失败 |
| R12 | 修正边界 | q/wrist幅度与真实dt运动门保持；transition不增加Contact标签 |
| R13 | 公平比较 | 固定样本、覆盖/碰撞/切向逃逸共同评价；不能只看被优化loss |
| R21 | 安装链 | hand-root与flange的目标正确互换；FK闭环，SE(3) det=+1 |
| R22 | R2依赖 | 合格R0窗口可不等R1；不合格R0不能盲目整片IK |
| R23 | 虚拟base | 会话固定；整个场景同变换，不逐帧移动或仅移动物体 |
| R24 | 几何范围 | 开放patch只做其合法距离/相交检查；隐藏厚度/环境不报全碰撞PASS |
| R25 | 质量与闭包 | 合同修复不算IK改善，IK改善不豁免合同缺失 |

## 6. 实片结果必须保留三种对照

- W0原strict vs 本轮同口径strict，判定原质量是否真的恢复。
- 同固定样本原consumer范围 vs 经审计新consumer范围，判定是能力拆分还是数值改进。
- 未参与拟合的W1/额外验证组结果，独立性不明时标限制，不误称泛化。

所有统计同时报告全时间轴、有效观测、合法窗口、拒绝/未知，不只报通过的子集。量化没有实测真值时明确代理指标，人工验收仍PENDING。

最终算法分级的验收必须由实片结果决定：W0至少2条同口径严格恢复、W1-ADOPTION冻结后成立、H9 holdout保持及多会话R0质量通过才可写 `TARGET_MET`；Contact/R1仍为零但HaWoR/R0跨会话恢复且主瓶颈明确时最多为 `PARTIAL_MATERIAL_PROGRESS`；strict Robot仍为零且主因未定位/候选未通过adoption时为 `REJECTED_NO_RECOVERY`；真实资源或运行故障未完成协议时为 `INCONCLUSIVE_RUNTIME`。
