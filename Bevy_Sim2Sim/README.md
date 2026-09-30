# Bevy_Sim2Sim

以 **Rust + Bevy + Rapier + CPU ONNX** 实现 MuJoCo → Bevy 的 MicroDuck Sim2Sim。目标包括完整 Pollen/mjlab 本体、BAM 执行器、60 Hz 物理与策略、九技能与 Sprint，以及科学站场景和交互。

当前已建立单世界固定步运行时、科学站渲染、腿式／轮式机器人显示和实时 ONNX 诊断，并完成受控性能分析。**完整机器人行为尚未通过验收**：旧 200/50 ONNX 在 60/60 P-only 诊断中仍会失稳；完整 BAM、源接触等价、新策略和正式游戏循环继续推进。

[仓库主页](../README.md) · [开发交接](docs/handoff.md) · [实施计划](docs/implementation_plan.md) · [工程规范](bevy_engineering_rules.md) · [CPU CI](https://github.com/sgyli7/Sai_Lab/actions/workflows/bevy_sim2sim_cpu.yml)

## 当前功能与验收

| 模块 | 已实现 | 当前边界 |
| --- | --- | --- |
| 固定步运行时 | 唯一 Rapier 世界、60 Hz 时钟、事件队列、逐 Tick 推理／积分／发布、失败停机 | 完整 BAM 与合格机器人控制器尚未接入正式游戏循环 |
| 科学站 | v18AJ／`windpass_compound_v7` 场景、固定机位与跟随相机、描边与静态标牌烘焙 | 20 个可移动道具碰撞体尚未导入，完整机器人通行与接触待验收 |
| 机器人显示 | 同源编译模型与外观、原生位姿帧、腿式与轮式初始化 GPU 预览 | 初始化几何与显示通过不等于动态控制通过 |
| 实时诊断 | 独立 60 Hz 物理／CPU ONNX 工作线程、顺序发布、最终窗口截图、逐帧收据 | P-only 诊断控制器；旧 ONNX 重放未取得技能资格 |
| 性能工具 | Bevy 时间线、Rapier 阶段计时、线程 CPU 时钟、GPU 时间戳与额外主线程负载探针 | 非默认开发功能；业务负载需按实际内容复测 |
| 源端训练 | 60/60 MuJoCo Warp＋BAM 探索训练与独立评估流程 | 最终策略未通过多种子 Idle，未晋升生产 candidate |

固定步运行时强制每个到期 Tick 执行“事件／场景同步 → 一次推理 → 力矩 → 一次积分 → 时钟提交 → 状态发布”，不隐藏高频时间子步。显示可以跳过中间姿态，工作线程仍连续执行每个物理和策略 Tick。接口与验收边界见 [固定步与实时接线](docs/fixed_step_runtime_probe.md)。

当前科学站已有六个真实 GPU 机位复核。同世界腿式初始化包含 2,553 个站体静态碰撞体，共 16 个 body／2,564 个 collider／14 个 joint；地形保留的两个零面积源三角及道具未导入项仍需后续处理，详见 [科学站说明](docs/station_layout_redesign.md)。

## 性能进展

以下为 **2026-09-30 开发窗口测量**：NVIDIA GB10、Linux / Vulkan、解锁 X11 桌面、1920×1080、8×MSAA、原阴影配置、真实 CPU ONNX，每轮 600 个物理 Tick。无额外负载的吞吐使用不含新增 CPU／GPU 探针、Chrome 时间线和 Rapier 计时的构建；分段成本与额外负载另用测量构建采集。

| 项目 | 实测结果 |
| --- | --- |
| 默认 FIFO 主循环吞吐 | 两轮约 60.07 FPS |
| 不锁帧主循环吞吐 | 三轮约 150–157 FPS |
| 不锁帧，额外 10 ms 主线程 CPU 负载 | 两轮约 75 FPS |
| 主线程 CPU | P50 约 1.43 ms，P95 约 2.66 ms |
| 根渲染图 GPU | P50 约 4.56 ms，P95 约 5.32 ms |
| FIFO 下新增主线程纯计算预算 | 建议先按 10 ms 规划；受控容量约 12 ms，14 ms 时吞吐开始下降 |

物理与推理在上述各档位均保持 60 Hz，工作线程未观察到误期。FPS 表示活跃主循环吞吐，不等于显示器或远程视频的显示频率。不锁帧使用 `AutoNoVsync` 请求，驱动最终呈现模式未单独采集。

10–12 ms 额外负载下平均吞吐仍约 60 FPS，但帧间隔 P95 增至约 23 ms；因此 10 ms 是当前场景的规划起点，不能保证每帧准时。真实业务的内存访问、锁、同步推理或共享 GPU 推理需另行测量。代码对照保持逐 Tick 状态和最终 RGBA 画面一致，保留原画质；这些开发结果尚未授予正式游戏性能或技能资格。

测量开关、计时边界与复现方法见 [实时窗口性能诊断](docs/render_performance_triage.md)。原始 trace、火焰图、收据、日志与二进制随所属实验归档，不入库。

## 快速开始

在本目录运行。需要 **Rust 1.98**、可用的图形桌面与 Vulkan 驱动；当前 Bevy 配置启用 Linux X11。Linux 构建库清单见 [CPU CI 配置](../.github/workflows/bevy_sim2sim_cpu.yml)。科学站预览使用仓库内的场景、shader 和字体。

```bash
# 科学站窗口；目前正式入口仅开放 robot none
cargo run --locked --release -- --scene science_station_preview --robot none

# 基础无头验证：600 次积分、20 次冷重置
cargo run --locked --features dev_tools -- \
  --headless --verify --scene foundation --robot none --output .scratch/foundation

# CPU workspace 测试；需要外部原生 fixture 的用例默认忽略
cargo test --locked --workspace
```

基础验证明确记录 `inference_count=0`，只验证工程基础。尚未实现或验收的机器人模式返回非零退出码。

Tab 切换机位；跟随视角用鼠标右键环视、滚轮调整距离。截图和限帧属于非默认开发功能：

```bash
cargo run --locked --release --features dev_tools -- \
  --scene science_station_preview --robot none --view overview \
  --frames 120 --capture /tmp/bevy_station_overview.png
```

`BEVY_SIM2SIM_ASSETS` 可指定资产根。必需字体／shader 缺失、资产加载或 GPU 管线失败会返回错误。

### 真实机器人开发入口

```bash
cargo build --locked --release -p dev_tools_minigame --features rendering_preview \
  --bin robot_initialization_preview \
  --bin station_robot_60hz_diagnostic \
  --bin station_robot_live_preview \
  --bin robot_pose_sequence_video
```

这些入口需要另行准备 SHA256 绑定的编译本体、外观、源 qpos、ONNX 策略及原生 ONNX Runtime；克隆仓库不会自动取得全部诊断输入。初始化工具只读取唯一世界的零积分快照；实时窗口和录像工具复用真实诊断状态，不创建第二套物理或 FK。参数见 [渲染与捕获接线](crates/modules/rendering/README.md)、[ONNX 诊断](docs/legacy_nine_onnx_aj_fixed_step_video.md)及 [性能诊断](docs/render_performance_triage.md)。

旧九模型录像目前用于失败定位：`alpha_stand` 起步摆动、约两秒倒地，多个其他策略也失稳。源端训练与评估方法见 [PPO 探索流程](docs/source_rl_exploratory_smoke.md)和 [多种子 Idle 评估](docs/source_rl_multirun_60hz.md)。

## 工程与依赖

工程按公共层 → 业务层 → 开发层组织：`common_minigame` 提供业务无关基础，`robot_minigame` 管理本体与位姿合同，`simulation_minigame` 持有唯一物理世界，`rendering_minigame` 提供运行时显示，`dev_tools_minigame` 管理诊断窗口、截图、训练与测量。根包负责启动装配；发布构建默认不含开发工具，正式资源包须排除开发场景。最终游戏不依赖 Python。

依赖由 [Cargo.lock](Cargo.lock) 固定，Cargo 自动获取 Rust 包。采用直接 Rapier 桥接，兼容性边界见实施计划。

| 依赖 | 固定版本 | 来源 |
| --- | --- | --- |
| Bevy | 0.19.1 | [官方 crate](https://crates.io/api/v1/crates/bevy/0.19.1/download) |
| Rapier 核心 | 0.35.3 | [官方 crate](https://crates.io/api/v1/crates/rapier3d/0.35.3/download) |
| ORT Rust | 2.0.0-rc.13 | [官方 crate](https://crates.io/api/v1/crates/ort/2.0.0-rc.13/download) |
| ONNX Runtime CPU | 1.30.0 Linux aarch64 | [官方原生运行库](https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-aarch64-1.30.0.tgz) |

原生运行库归档 SHA256 为 `e16a27a8ed330bbc698df7330b0cf56e722f354e3bcc92118682c74ef3c3e3da`；解压后的 `libonnxruntime.so.1.30.0` SHA256 为 `64e903a43a041240fd6bcffe0ac6d4fea47ef87bf24b9d097801bd00a9612a4b`。

本地 `third_party/rapier3d` 基于官方 0.35.3 发布源码，含默认关闭的 Sim2Sim 观测／诊断功能及固定场景 CCD 缓存维护。具体补丁与上游身份见 [本地变更](third_party/rapier3d/LOCAL_CHANGES.md)和 [原始来源](third_party/rapier3d/UPSTREAM_SOURCE.json)；当前观测仍未提供完整 BAM 控制外载。

## 来源与许可

自研 Rust 包声明 Apache-2.0。第三方源码、字体与模型分别遵循原许可：Rapier 保留 [许可证](third_party/rapier3d/LICENSE)，Noto 字体保留 [许可声明](assets/third_party/fonts/license.debian)。MicroDuck 上游对 3D 模型另有 Creative Commons BY-SA-NC 声明，模型及衍生网格须保留来源和非商业限制，不能归入自研代码许可证。

开发输入、模型、权重、录像和实验结果保留在所属仓库的历史备份目录 `/home/ethan/ProjectBackups/<date>/Sai_Lab/`；一次性进程文件使用 `/tmp`。更早的 Godot 调研保留在 [历史研究](docs/research.md)，当前实现以 Bevy 实施计划和运行说明为准。
