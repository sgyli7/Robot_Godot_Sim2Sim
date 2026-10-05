# Sai_Lab

Sai_Lab 汇集三个独立的机器人 Sim2Sim 工程，覆盖 MicroDuck、Unitree G1、轮式变体和 Sai Robot 的仿真、控制与训练。

| 目录 | 工程 | 说明 |
| --- | --- | --- |
| [Bevy_Sim2Sim](Bevy_Sim2Sim/README.md) | MuJoCo → Bevy / Rapier | Rust 工程、单世界 50 Hz 原生物理、科学站、G1 本地视觉任务与 MicroDuck 诊断。完整行为仍在验收。 |
| [Godot_Sim2Sim](Godot_Sim2Sim/README.md) | MuJoCo ↔ Godot/Jolt | MicroDuck、轮滑版和 Sai Robot 的场景、控制与训练。 |
| [Unity_Sim2Sim](Unity_Sim2Sim/README.md) | Unity/团结引擎 + MuJoCo 原生 | MicroDuck 的原生物理运行、ONNX 策略与场景实现。 |

三个工程分别维护依赖、运行命令和验收。克隆后进入对应目录，按各自 README 操作。

## Bevy 当前进展

![G1 中文操作入口重置后真实搬箱并松手，12 秒两段原速开发回归](Bevy_Sim2Sim/docs/media/g1_science_station_controls_box_release_12s.gif)

2026-10-05 新片：旧 Qwen 请求在途时停止、重置，新回合实际持箱行走、放入容器并松手；独立检查为 1.99 米搬运、2.76 秒放稳。这里节选 5 秒行走和 7 秒释放，两段原速、中间一次直接切镜。此开发回归不替代下面的冻结成绩。

![G1 科学站视觉搬运并松手，12 秒两段原速开发诊断](Bevy_Sim2Sim/docs/media/g1_science_station_box_release_12s.gif)

![G1 科学站持箱行走，12 秒原速开发诊断](Bevy_Sim2Sim/docs/media/g1_science_station_box_carry_12s.gif)

![G1 在源任务近似场景搬箱、行走和释放，12 秒原速开发诊断](Bevy_Sim2Sim/docs/media/g1_source_box_carry_12s.gif)

主页录像均已按 URI 风格重新录制。G1 保留官方银灰外壳和深色关节，科学站使用蓝／黄橙／暖白配色；风格只作用于主窗口，VLA 相机和控制合同保持原样。实现与验证见 [G1 展示渲染](Bevy_Sim2Sim/docs/g1_uri_presentation.md)。

G1 已有科学站持箱行走约 **1.99 米**、放入容器并松手的实际样例，主页 12 秒 GIF 为原速节选。2026-10-05 完整重跑冻结的五组位置 × 两个种子：**T1 3/10、T2 2/10，均未达到 8/10**。T1 执行 5,580 次积分和 10 次原配 N1.7；T2 执行 12,925 次积分、40 次原配 N1.6 和 18 次本地 Qwen。全部二十段原速录像、成功与失败轨迹已分别审计，旧 1/10 测试保留为历史记录。固定任务中文入口、启动／停止／重置与会话资源回收已实现，使用范围和实际证据见 [阶段记录](Bevy_Sim2Sim/docs/g1_milestone.md#2026-10-05-冻结十例与操作入口) 与 [中文操作入口](Bevy_Sim2Sim/docs/g1_station_session.md)。**变化恢复、地形、连续全程 1× 与完整许可链仍有缺项，整体尚未验收。**

- **工程基础**：一个 Rapier 世界，默认 50 Hz、每 Tick 一次积分；G1 与 MicroDuck 保留独立合同；实时开发窗口使用独立工作线程执行物理和 CPU ONNX 推理。
- **科学站与显示**：当前场景为 `windpass_compound_v7`，支持固定机位、跟随相机、静态文字烘焙，以及腿式／轮式机器人的原生位姿显示。
- **性能**：2026-09-30 在 NVIDIA GB10、Linux / Vulkan、1080p 原画质的开发诊断中，FIFO 平均约 60 FPS，不锁帧约 150–157 FPS；新增主线程纯计算预算可先按 10 ms 规划。完整条件和帧节奏限制见 [Bevy 性能进展](Bevy_Sim2Sim/README.md#性能进展)。
- **验收状态**：工程、渲染与诊断链路已合入主干。旧 ONNX 的 60/60 重放仍会失稳；G1 诊断链路按阶段合入；正式成功率、恢复与地形验收仍缺，MicroDuck 完整 BAM、源接触等价、新技能与正式游戏循环继续推进。

[Bevy 运行说明](Bevy_Sim2Sim/README.md#快速开始) · [实施计划](Bevy_Sim2Sim/docs/implementation_plan.md) · [开发交接](Bevy_Sim2Sim/docs/handoff.md) · [CPU CI](https://github.com/sgyli7/Sai_Lab/actions/workflows/bevy_sim2sim_cpu.yml)

## 快速进入工程

```bash
git clone https://github.com/sgyli7/Sai_Lab.git
cd Sai_Lab/Bevy_Sim2Sim
# 也可选择 Sai_Lab/Godot_Sim2Sim 或 Sai_Lab/Unity_Sim2Sim

# Bevy 科学站预览；需要 Rust 1.98 和可用的图形桌面
cargo run --locked --release -- --scene science_station_preview --robot none
```

## 来源与许可

Godot 工程保留原 `Robot_Godot_Sim2Sim` 的提交历史、[LICENSE](Godot_Sim2Sim/LICENSE) 和 [NOTICE](Godot_Sim2Sim/NOTICE)。Unity 工程从 [MicroDuck-Unity-Sim2Sim](https://github.com/sgyli7/MicroDuck-Unity-Sim2Sim) 导入并保留其提交历史；其 [NOTICE](Unity_Sim2Sim/NOTICE) 说明 MicroDuck 3D 模型及衍生网格的非商业许可限制。使用或分发前请分别查看各目录的许可文件。

Bevy 自研 Rust 包在 [Cargo.toml](Bevy_Sim2Sim/Cargo.toml) 中声明 Apache-2.0；Rapier 源码与字体保留各自的第三方许可。MicroDuck 模型仍须遵守上游模型许可，详见 [Bevy 来源与许可](Bevy_Sim2Sim/README.md#来源与许可)。

## 迁移期间的开发

已有任务如果还在旧的 `Robot_Godot_Sim2Sim` 本地检出目录工作，请先把自己的改动提交到独立分支，再同步新的 `main`。不要在有未提交改动时直接切换到迁移后的 `main`；文件路径已经整体移动，Git 可能需要人工处理冲突。新的 Bevy、Godot、Unity 改动分别放在 `Bevy_Sim2Sim/`、`Godot_Sim2Sim/`、`Unity_Sim2Sim/`。原 GitHub 仓库地址会重定向到 `Sai_Lab`，但建议将本地 `origin` 更新为新地址。
