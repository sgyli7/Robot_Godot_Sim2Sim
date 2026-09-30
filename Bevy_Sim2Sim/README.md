# Bevy_Sim2Sim

使用 Bevy、Rapier 和 CPU ONNX 推理实现 MuJoCo → Bevy 的 MicroDuck Sim2Sim。当前目标是完整 Pollen/mjlab 本体、BAM 执行器、真实 60 Hz 物理与策略、九技能和 Sprint，以及重新设计的科学站。

**当前机器人行为不可交付。** [九个旧 ONNX 的 60/60 录像](docs/legacy_nine_onnx_aj_fixed_step_video.md)是失败诊断：`alpha_stand` 在 0.8 秒左右剧烈摆动、约 2 秒倒地，行走、坐站、双踢和轮行也失稳。旧策略原按 200 Hz 物理／50 Hz 推理运行，这里仅用简化 P-only 力矩在 Rapier 上零样本重放；九段视频不能当作九项技能展示或验收。

工程接线已有单世界 60 Hz 固定步生命周期驱动；九个旧 ONNX 在冻结 v18AJ 科学站完成真实推理、积分与 GPU 截帧。这只证明诊断链路能执行。开发窗口 v3 已将推理／物理移到独立 60 Hz 工作线程、默认使用 FIFO，并接入 Bevy/Rapier 自带性能诊断；静态标牌烘焙、不可变站体质量属性准备和 CCD 固定场景缓存维护已接入，**开发测量不授予正式游戏或技能资格**。[源端 BAM 站立训练](docs/source_rl_multirun_60hz.md)已完成 200 次真实 PPO 更新，但最终 checkpoint 仅 1/5 个独立种子通过 600 Tick Idle 门槛，不可晋升。目标端 BAM 外载、源接触等价、稳定的 60/60 策略、场景交互和正式游戏循环均未验收。[固定步与实时接线收据](docs/fixed_step_runtime_probe.md)、[性能诊断说明](docs/render_performance_triage.md)。

- [开发交接入口](docs/handoff.md)
- [完整实施计划](docs/implementation_plan.md)
- [历史 Godot 调研](docs/research.md)
- [唯一工程规范](bevy_engineering_rules.md)

在本目录运行，使用 Rust 1.98：

```bash
cargo run -- --scene science_station_preview --robot none
cargo run --features dev_tools -- --headless --verify --scene foundation --robot none --output .scratch/foundation
cargo test -p common_minigame -p simulation_minigame -p robot_minigame
```

基础验证运行 600 次物理积分和 20 次冷重置，明确记录 `inference_count=0`；它只验证工程基础。暂未实现或验收的机器人模式会返回非零退出码。

依赖由实际生成的 Cargo.lock 固定，Cargo 会自动获取 Rust 包。需要手动下载时使用：

| 依赖 | 固定版本 | 下载 |
| --- | --- | --- |
| Bevy | 0.19.1 | [官方 crate](https://crates.io/api/v1/crates/bevy/0.19.1/download) |
| Rapier 核心 | 0.35.3 | [官方 crate](https://crates.io/api/v1/crates/rapier3d/0.35.3/download) |
| ORT Rust | 2.0.0-rc.13 | [官方 crate](https://crates.io/api/v1/crates/ort/2.0.0-rc.13/download) |
| ONNX Runtime CPU | 1.30.0 Linux aarch64 | [官方原生运行库](https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-aarch64-1.30.0.tgz) |

原生运行库归档 SHA256 为 `e16a27a8ed330bbc698df7330b0cf56e722f354e3bcc92118682c74ef3c3e3da`；解压后的 `libonnxruntime.so.1.30.0` SHA256 为 `64e903a43a041240fd6bcffe0ac6d4fea47ef87bf24b9d097801bd00a9612a4b`。采用直接 Rapier 桥接，版本选择及兼容性边界见实施计划。`third_party/rapier3d` 基于官方 `0.35.3` 发布源码，包含默认关闭的只读 Sim2Sim 观测功能，以及 CCD 固定场景缓存维护与回归测试；观测功能当前仍未提供完整控制外载。

开发工具与训练 Python 保留在非默认开发模块，最终游戏不依赖 Python。一次性进程文件使用 `/tmp`；历史模型、权重、录像和实验结果保留在 `/home/ethan/ProjectBackups/<date>/Sai_Lab/`。实验原始轨迹、截图、日志和审查报告随对应实验归档，不入库。

机器人初始化 GPU 预览工具位于开发模块，需启用非默认 `rendering_preview` feature；其参数显式绑定模型、外观与源 qpos 文件的 SHA256。腿式与轮式四组实际 1080p 截图已完成，全原视觉顶点对照通过。当前两种零 UV 模型另按 MuJoCo 3.10 的角点显示法线规则渲染，三组真实模型累计 3,760,632 个角点与独立 C oracle 核对，八张固定机位 GPU 对照已检查；`raw_normals` 仅用于开发诊断。它只显示唯一 Rapier 世界的零积分初始化快照；入口和接线说明见 [渲染模块说明](crates/modules/rendering/README.md)。完整控制、运动与性能验收继续推进。

当前 live 科学站为 v18AJ／`windpass_compound_v7`：西侧样本作业带、东侧服务廊、错位双塔和中央标定庭院形成整体空间布局。六个固定视角的真实 GPU 画面已复核；同世界零步预检装入 2,553 个站体静态碰撞体与腿式机器人，共 16 个 body／2,564 个 collider／14 个 joint，真实 GPU 首帧显示机器人与站体同场。20 个可移动道具碰撞体未导入；地形仍有两个保留的零面积源三角。机器人通行、完整站体接触与目标物理尚未合格。后续地编在 scratch 并行推进，当前视频绑定的 live 资产保持冻结；证据边界见 [整体地编报告](docs/station_layout_redesign.md)和[交接文档](docs/handoff.md)。

截图和限帧开发接线位于 `dev_tools_minigame::visual_preview`；根入口的
`--capture`／`--frames` 须显式启用 `--features dev_tools`。普通窗口由根入口装配
`StationVisualPlugin`，业务渲染 crate 不包含截图、录像收据或诊断窗口生命周期。
