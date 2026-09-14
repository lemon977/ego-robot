# Clean

当前算法为同会话真实temporal/stereo donor优先，剩余UNKNOWN区域使用ProPainter。入口、权重、Wave0 authority和质量门见 [当前注册表](../../docs/governance/CURRENT_BASELINE_REGISTRY_V2.json)。

固定边界：保护物体像素byte-exact；逐像素source map；synthetic只用于视觉，不反喂几何或动作真值。
