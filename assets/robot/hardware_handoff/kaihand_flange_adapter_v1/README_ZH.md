# KaiHand—Tianji 法兰连接件硬件交接包 V1

状态：`REFERENCE_GEOMETRY_ONLY_NOT_READY_TO_PRINT`

## 结论

项目中已有 Tianji 末端和 KaiHand 手根的 URDF/STL 参考几何，但没有经过核实的 KaiHand—Tianji 连接件 CAD，也没有两侧安装面的孔位图、实测装配变换或 TCP 标定。因此本目录可以交给硬件同事作为建模起点，不能直接拿现有圆环代理去打印并装机。

`reference_geometry/` 内是当前项目实际使用的参考模型：

- `tianji/Link7_L.STL`、`Link7_R.STL`：左右末端外形参考。
- `tianji/marvin_CCS_m6.urdf`：名义机器人链；其中 flange→tool 的 145 mm 是 URDF 名义工具偏置，不是连接件实测长度。
- `kaihand_left/hand_l_base_link.STL`、`kaihand_right/hand_r_base_link.STL`：左右 KaiHand 手根外形参考。
- 左右 KaiHand URDF：坐标系、关节树和 mesh 来源参考。

`excluded_reference_only/` 中的文件不得当作打印件：

- `FLANGE_RING_VISUAL_PROXY_CONTRACT.json` 明确声明 `real_adapter_cad=false`、`measured_mount=false`；内径是视觉设计值。
- `NOT_KAIHAND_ADAPTER_V2_CAMERA_FLANGE_PRO.STL` 是另一用途的相机固定法兰参考，不是 KaiHand 连接件。

## 硬件同事还需要补齐的输入

1. Tianji 法兰安装面的 STEP/STP 或带尺寸工程图：孔数、孔径/螺纹、PCD、沉头/沉孔、中心定位台阶、键位/零度方向。
2. KaiHand 左右手底座安装面的 STEP/STP 或带尺寸工程图：同样的孔位、定位面、螺纹深度和禁入区域。
3. 明确左/右手是否共用一个件、是否允许镜像；左右分别给出掌面朝向和拇指朝向。
4. 法兰面到 KaiHand root 的目标轴向距离和绕轴 clocking；电源/通信接头、线缆弯曲半径及拆装工具空间。
5. 紧固件和结构要求：螺钉规格、螺纹啮合长度、螺母/热熔嵌件、材料、打印工艺、载荷/力矩、安全系数、允许质量。
6. 制造公差：定位配合、孔径补偿、同轴度/垂直度；建议先做低成本薄片孔位样件，再打印承力件。
7. 装配后实测 `T_flange_kaihand_root` 和 TCP；未测之前只能做视觉仿真，不能获得实体部署 authority。

可直接使用 `MEASUREMENT_AND_DESIGN_INPUT_FORM_ZH.md` 回填这些信息。

## 硬件最终应回交的文件

- 可编辑 `STEP/STP`（首选）和打印用 `STL/3MF`，单位明确为 mm。
- 带基准、尺寸、公差和螺纹说明的二维 `PDF/DXF`。
- 左/右件料号、版本、材料、打印方向、层高/壁厚/填充率和后处理说明。
- BOM、紧固件清单、装配爆炸图及线缆走向图。
- 实物尺寸复测表、试装照片、干涉检查和承载测试记录。
- 文件 bytes/SHA256；任何改版使用新版本号，不覆盖已交付版本。

## 建模建议

不要从 STL 三角面反推孔位作为唯一依据。应以厂家原生 CAD/工程图和卡尺/CMM 实测共同冻结接口；本包 STL 只用于包络与初步干涉检查。连接件完成后先做 CAD 装配和低成本孔位样件，再做承力打印。

