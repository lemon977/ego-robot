# Chips 当前推进状态

- PHONE_DEV Mask：双手/前臂+双 tracker 360/360 PASS，`consumption_authorized=true`。
- PHONE_DEV Clean：跨机位空桌已尝试局部 donor、graph-cut、PatchMatch、quilting 与整桌
  变换，均因纹理块、拉伸或物体重复停在 frame0 可视门，未冒充全片 PASS。
- 新 PICO 到位后：共享同一 Mask 算法/权重重跑24帧 canary；正式 Clean 直接使用该连续
  session 的空桌/视角扫描，不继承手机跨机位 donor。
- Robot 几何仍需用真实测量值替换当前 illustrative `CHIP_SADDLE/BOWL_REVOLVE`。

