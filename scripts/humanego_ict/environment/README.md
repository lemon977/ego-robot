# Joint26 GPU环境锁

这是新版Joint26 GPU训练及共享治理发布所需的精简依赖闭包，不是系统`pip freeze`。`runtime-requirements.txt`锁定13个包，`torch-cu128-requirements.txt`锁定Torch及24个传递依赖；后者明确使用官方CUDA12.8轮子。Conda基础为Python3.11.11、pip23.3.2，两份requirements由项目setup入口显式安装。不得省略传递依赖或仅补装jsonschema；安装后必须运行`python -m pip check`及项目环境检查。

版本来自本机已运行的`/usr/local/bin/python`。SciPy1.17.0和networkx3.6.1要求Python≥3.11，不能把这套精确锁装进Python3.10的`openwam_alpha`。Linux x86_64的Torch轮子标签是`cp311-cp311-manylinux_2_28_x86_64`；迁移目标需匹配Python ABI、架构及glibc≥2.28。当前本机glibc为2.35。CUDA12.8用户态库已列入锁，但NVIDIA驱动和GPU／NCCL通信须由目标DLC容器实际检查，环境包不包含宿主机驱动。

只安装`opencv-python-headless==4.11.0.86`，不同时安装GUI OpenCV、contrib或另一份headless。系统实际cv2为GUI4.11.0.86；这次选择同版本headless以减少GUI系统库依赖，仍需新环境导入检查。系统另残留headless4.13.0.90元数据，其NumPy≥2要求与NumPy1.26.4冲突，不能照抄系统freeze。

共享publisher的合同聚合会导入cv2、SciPy、trimesh、Numba和PyYAML，所以GPU训练环境仍需这些包；不会因此执行渲染。Numba导入可能创建JIT缓存，`PYTHONDONTWRITEBYTECODE=1`不能阻止它。setup／检查／训练应把`TMPDIR`、`NUMBA_CACHE_DIR`及其他编译缓存显式放项目内；Conda打包迁移后按所用工具完成路径修复，并由项目入口控制缓存位置。导入与`pip check`成功后再验证目标GPU；八卡NCCL须在同事机器上另做启动检查。

不包含未用于这条GPU训练入口的MuJoCo、Mink、qpsolvers、daqp、模型下载SDK、torchvision、torchaudio及trimesh可选extras，不改变CPU数学环境。只读合同构建屏蔽Cython、colorama、idna、lxml和xxhash仍通过165个合同；这些可选包不属于最小锁。迁移时应重建或使用经验证的Conda打包工具，不直接复制系统site-packages或旧venv。
