# Bevy_Sim2Sim

使用 Bevy、Rapier 和 CPU ONNX 推理实现 MuJoCo → Bevy 的 MicroDuck Sim2Sim。当前目标是完整 Pollen/mjlab 本体、BAM 执行器、真实 60 Hz 物理与策略、九技能和 Sprint，以及重新设计的科学站。

当前为开发中的工程：固定步与 Rapier 基础验证、原生策略边界、BAM 公式核对、完整机器人结构初始化和科学站 GPU 预览已经可运行。完整机器人控制、技能、场景交互与 1080p／60 FPS 尚未验收。

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

原生运行库归档 SHA256 为 `e16a27a8ed330bbc698df7330b0cf56e722f354e3bcc92118682c74ef3c3e3da`；解压后的 `libonnxruntime.so.1.30.0` SHA256 为 `64e903a43a041240fd6bcffe0ac6d4fea47ef87bf24b9d097801bd00a9612a4b`。采用直接 Rapier 桥接，版本选择及兼容性边界见实施计划。`third_party/rapier3d` 保留官方 `0.35.3` 发布源码身份，仅加入默认关闭的只读 Sim2Sim 观测功能；该功能当前仍未提供完整控制外载。

开发工具与训练 Python 保留在非默认开发模块，最终游戏不依赖 Python。临时日志、原始轨迹、截图、录像和审查证据使用 `.scratch/`。

机器人初始化 GPU 预览工具位于开发模块，需启用非默认 `rendering_preview` feature；其参数显式绑定模型、外观与源 qpos 文件的 SHA256。腿式与轮式四组实际 1080p 截图已完成，全原视觉顶点对照通过。当前两种零 UV 模型另按 MuJoCo 3.10 的角点显示法线规则渲染，三组真实模型累计 3,760,632 个角点与独立 C oracle 核对，八张固定机位 GPU 对照已检查；`raw_normals` 仅用于开发诊断。它只显示唯一 Rapier 世界的零积分初始化快照；入口和接线说明见 [渲染模块说明](crates/modules/rendering/README.md)。完整控制、运动与性能验收继续推进。

科学站当前场景在四个功能区和宽敞环路的基础上，增加一处西侧实体岩脊与样本、维修等区域的贴地色标。六个固定视角的真实 GPU 画面已复核；中庭仍留作机器人活动。`robot_initialization_preview` 的 `--zero-step-only` 开发模式可在同一个 Rapier 世界中装入当前科学站 2,478 个静态碰撞体和腿式机器人，核对初始位姿与对象清单后退出，不开启 GPU 或积分；同输入的真实 GPU 首帧显示机器人与站体同场景。此预检中的 20 个可移动道具只作视觉对象；地形有两个保留的零面积源三角。岩脊小球接触仅作局部诊断，机器人通行、完整站体接触与目标物理仍未合格，证据边界见 [交接文档](docs/handoff.md)。
