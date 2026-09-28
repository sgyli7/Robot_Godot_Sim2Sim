# v18AJ 九个原始 ONNX：FixedStepRuntime 重新截帧

日期：2026-09-28。本次在已冻结的 v18AJ 科学站上，使用开发诊断入口通过 `FixedStepRuntime` 运行九个原始 ONNX 的真实 60 Hz Rapier 积分和 60 Hz 原生 ORT 推理，并从**新驱动报告**重新捕获九段 Bevy GPU 视频。旧的 [AJ 录像与报告](legacy_nine_onnx_aj_v7_rerecord.md)保留，本次媒体和收据位于独立 `.scratch/legacy_nine_native_probe/fixed_step_driver_video_v1/`。本批仍采用简化 P-only 力矩，不能当作 BAM、完整接触或技能验收。

九份新源报告在 [固定步回归](fixed_step_runtime_probe.md)中各重复运行两次且字节一致，均带 `fixed_step_driver="FixedStepRuntime"`、60 Hz `final_clock`、零待处理 Tick，合计 **2,400 次真实积分和 2,400 次 ORT 推理**。新驱动报告相对于旧 AJ 报告只增加驱动与时钟收据；原有逐 Tick 观测、动作、力矩、接触计数和机器人姿态完全相同。本次 GPU 捕获重新读取九份新报告，独立生成 1,209 张 PNG 和九个 MP4，绝非重命名旧媒体。每段收据都把新物理报告 SHA、模型身份、取帧序号与姿态 SHA 绑定起来。

| 原始 ONNX | 新源报告 SHA-256 前 12 位 | 推理 / 积分 | 视频帧 |
| --- | --- | ---: | ---: |
| `alpha_walking` | `dccea058c69a` | 360 / 360 | 181 |
| `alpha_stand` | `9405ce220d97` | 240 / 240 | 121 |
| `alpha_sitstand` | `a9b4dd643c9d` | 480 / 480 | 241 |
| `alpha_ground_pick` | `b01d0b9f0013` | 240 / 240 | 121 |
| `ball_kick_left` | `1f950f584939` | 180 / 180 | 91 |
| `ball_kick_right` | `400781cfafa3` | 180 / 180 | 91 |
| `roller` | `ee0481c28f3d` | 360 / 360 | 181 |
| `roller_crouch` | `fdfd49db0ed0` | 180 / 180 | 91 |
| `roulade` | `57cd5ffcc0d8` | 180 / 180 | 91 |

新旧画面相同程度也被单独审计：**1,209/1,209 个采样姿态 SHA 相同，1,202/1,209 张 PNG 文件逐字节相同**。其余七帧均仅有 1–2 个像素变化，最大单通道差值 10；九个逐段 MP4 的文件 SHA 全部与旧 AJ 逐段 MP4 相同。这与新旧物理轨迹一致相符，也说明不能用视觉相同来冒称新技能成果。新合辑额外明确标注 `FixedStepRuntime 60/60`，因此其 SHA 与旧 AJ 合辑不同。

新合辑 `nine_original_onnx_aj_fixed_step_60_60_p_only_diagnostic.mp4` 的 SHA-256 为 `4bcc81c41f8783617f6642cf0987096489223e70423c19ac8911d0beb3b79735`。独立解码探测得到 **1920×1080、30 fps、40.30 秒、1,209 帧**；从最终合辑另行抽取九段中点画面并亲审机器人姿态及来源字幕，见本地 `combined_review/combined_midpoints.png`。九段的可见动作与[先前 AJ 观察](legacy_nine_onnx_aj_v7_rerecord.md)相同：步行、站立、坐站、双踢和轮行均出现失稳或翻倒；地面拾取没有实体目标交互，轮足下蹲与翻滚只有动作展示而无技能判定。

证据入口：

- `input_freeze.json` 冻结 v18AJ 完整 10 文件 `assets` 树、机器人模型与外观、视频二进制和九个新源报告；最终 live 资产复核完全一致。
- `video_delivery_manifest.json`，SHA-256 `abd43319c079087d576563baa14408d9bd6e0ac769477d2df8e0a56ff8105405`，记录全部逐段来源报告、原 ONNX、ORT、模型、视频和最终时钟身份。
- `old_new_media_comparison.json` 与 `old_new_png_pixel_difference_audit.json` 分别记录全 1,209 帧姿态/PNG 文件对照和七帧像素微差；`final_identity_audit.json`，SHA-256 `abea8712eeb7447e6b3712d6263f35f82a8551926a504db20c0b4245651ff220`，记录最终 live 哈希和媒体检查。
- `gpu/<name>_frames/`、`gpu/<name>.mp4`、`gpu/<name>_receipt.json` 保留九段独立 GPU 捕获；旧 AJ 证据目录没有改写。

这条链路仍是开发诊断工具。正式 `src/main.rs` 的游戏循环、BAM 外载与延迟模型、完整场景接触和九项技能验收没有因此闭合。
