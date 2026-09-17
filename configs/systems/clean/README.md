# Clean

当前已发布 exact78 结果实际使用“同会话、相同整数像素坐标的 temporal donor；两帧 RGB 一致后复制”，剩余删除区使用 ProPainter。当前 donor 没有做 Stereo 重投影或场景几何对齐；旧结果名中的 `temporal/stereo donor` 不能据此解释为已经使用 Stereo。入口、权重、Wave0 authority 和质量门见 [当前注册表](../../docs/governance/CURRENT_BASELINE_REGISTRY_V2.json)。

固定边界：`byte-exact` 只保护当前帧可见 object mask 内的像素，不保护被指尖遮住的隐含物体表面；人手固定膨胀 18–24 px、Tracker 膨胀 60 px 可能过度删除边缘；同坐标 temporal donor 也可能从别帧带入盘子等错误语义。当前 Grade B 只代表帧数、来源、可见物体像素和解码合同闭合，不代表接触边界视觉正确。synthetic 只用于视觉，不反喂几何或动作真值。
