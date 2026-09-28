# Bevy_Sim2Sim：60 Hz MicroDuck 与 URI 风格科学站

## 目标与边界

实现 **MuJoCo 源端训练 → Bevy/Rapier 目标端部署**。游戏内机器人、物件和场景共用一个 Rapier 世界，物理与策略均为 **60 Hz**，每个 Tick 单步积分、一次推理。

本轮交付九技能、Sprint、腿式／轮式切换和完整科学站。按用户 2026-09-28 的新授权，重新设计科学站整体结构与布局：沿用原资产，改善主广场、环形行走路线、功能分区、地标和环境层次，丰富场景并保持连续的可行走面积。向 [UriWallpaper Exploration Collection 006](https://uriwallpaper.gumroad.com/l/exploration006) 的线条、色板、明暗和细节密度深化实时画面。验收平台为 DGX Spark，目标 **1080p／60 FPS**。

嘴部真实抓放、SaiRobot、维修站及其他平台真机验收后置。实施已在隔离工作树开始；当前通过的构建、零积分本体、静态画面和部分有界物理机制检查，均不能代替完整控制、技能和性能门禁。

## 工程与统一契约

- 从正式 Sai_Lab 最新 main 创建 `codex/bevy_microduck60` 分支，工作树放入 `TempWorktree/Sai_Lab/bevy_microduck60`；保留现有工程及旧实验。
- 当前推荐验证基线锁定 Bevy `=0.19.1`、Rapier 内核 `rapier3d =0.35.3`、Rust 包 `ort =2.0.0-rc.13` 和 ONNX Runtime `1.30.0` CPU，提交依赖锁文件。`ort` 关闭默认特性，显式启用 `std`、`load-dynamic`、`api-28`，加载固定路径原生库。使用当前 Rust `1.98.0` 工具链。
- `simulation_minigame` 直接管理唯一的 Rapier `PhysicsWorld`，机器人、物件和场景共用该世界；维护 Bevy 实体与后端句柄映射，位姿通过显式 `f32` 数组转换。替换原先依赖已发布 `bevy_rapier3d 0.36.0` 的接入路线：该插件严格锁定 `rapier3d =0.35.0-glamx0.2`，未包含后续关键修复，不能通过额外添加新版 Rapier 升级它的内部世界。
- `robot_minigame` 管理本体定义、观测、执行器、技能状态和 ONNX；`simulation_minigame` 管理 Rapier、输入、地图和渲染；`dev_tools_minigame` 管理训练适配、回放、审查和测量。Python 训练工具归入开发层，发布游戏独立运行。
- 固定完整 Pollen/mjlab 任务与训练链。腿式技能统一使用完整碰撞本体，轮式使用对应完整轮式本体；相对上游 Walking 模型的差异明确登记。保存 BAM 处理后的有效质量、惯量、armature、碰撞与执行器配置。
- 建立新的版本化模型契约：61 维观测、14 维动作、关节映射、HOME、命令语义、归一化、历史、时延、执行器及重置规则。观测采用原源端语义；BAM 的负载反馈、摩擦和静摩擦求解是核心迁移工作。
- MuJoCo 先验证单步 60 Hz。需要更小积分步长时，单独记录实际步长、控制更新与执行器时序，通过 Rapier 真正单步 60 Hz 验证后采用。

## 实施顺序

1. **先建立可执行 Workflow。** 完整枚举注册任务、配置、编译本体、动态随机化字段和测试覆盖，生成差异及验证责任；接通候选身份、阶段准入、预算和资格失效机制。
2. **贯通完整本体与控制。** 验证坐标、惯性框架、关节锚点、限位、被动轮、BAM、接触及非零姿态初始化，再完成站立、行走、转向和停止闭环。
3. **实现固定步生命周期。** 在自有固定步入口依次执行场景变更同步、观测／推理、后端力矩更新、单次积分和状态写回。显式设置 `dt=1/60`、`num_solver_iterations=1`、每个刚体的 `additional_solver_iterations=0`、`max_ccd_substeps=1`，约束收敛只调整同一时间步内的 PGS 迭代；记录实际积分次数，排除默认四次 solver 时间子步。第零步先推理；暂停冻结仿真时钟；掉帧保留欠账，每显示帧最多追赶八步。重置和形态切换冷重建、清空历史，并保留规定的物件状态。
4. **完成十项技能训练与迁移。** 沿用完整上游任务、随机化、课程、保存／恢复和导出链；独立建立 Sprint 任务。阶段、滤波和延迟均按真实时间定义。每次长训最多两小时，每技能累计最多十二 GPU 小时、至少三个独立种子。
5. **并行建设科学站与画风。** 用原地形、建筑、设施和物件重组整站：明确主广场与主建筑锚点，连通宽敞环形路线，设置停泊、观测、样品互动及外围设备区，保留九技能、踢球和物件互动所需的安全空间。允许成组移动、转向、增减资产，补足路标与前中远景层次。场景布局以不可变配置表达，显示和碰撞使用同一份几何，场景身份变化使下游评价和录像失效并重跑。完善墨线、内部结构线、排线、有限色板、镜片和运动抗锯齿，使用六个固定机位样片确认视觉方向。

预算耗尽仍未达标时，关闭该技能追加长训，保存失败证据并重新讨论路线；继续推进其他独立工作，保留未完成状态。

## 品质 Workflow 与 GPT 独立审查

**完整上游采用 → 本体及接口验证 → 有界学习 → 独立选模 → 导出 → Rapier 验收 → GPT 审查 → 整包发布。** 每阶段消费当前证据，最终技能成绩属于最终验收阶段。

- **身份与重建：** 同一候选绑定源码、完整组装 MJCF／场景、资源、实际配置、依赖、checkpoint、ONNX、评估器、轨迹和录像。相关变更使下游资格失效，正式入口重新检查。
- **四维原始记录：** 保存完整首回合、终局及重置前数据，包括全部刚体三维位姿、速度、关节状态、动作、目标、实际力矩、接触位置与冲量／力、物件状态、输入和阶段事件。源端保留 MJCF 内部状态，目标端保留对应 Rapier 状态及映射；缺失量明确标记。
- **独立行为裁决：** 从原始状态重新计算姿态、跟踪、停稳、滑移、穿透、技能顺序和终局。评估器独立于奖励；失败、截尾和异常均计入分母。持续回归原有能力，并主动检验奖励与任务目标的矛盾。
- **根 GPT 亲自 Review：** 每个待晋级候选审查完整时序分析，重点检查起步、切换、接触、终局、全部异常及预先固定的抽样回合；同时检查对应真实录像。记录时间位置、数据依据、观察方法和处置。抽帧审查明确标注覆盖；审查发现能否决选模、继续训练或发布。
- **正常链与反例：** 跑通一个真实候选的完整正常链；验证换权重、漏终局、错映射、空测试覆盖、过期审查和行为退化会在相应入口被拒绝。训练、恢复、选模、导出和发布统一接入此机制。

## 验收与交付

- **数值与生命周期：** 同输入观测／动作转换容差 `1e-7`，ONNX 容差 `1e-5`；另验执行器和跨引擎行为。六百步对应六百次推理、控制和积分；覆盖不同显示帧率、500 ms 停顿、暂停、二十次重置、坏模型、缺资产及 shader 错误。
- **技能：** 每项标称场景成功率至少 95%；二百个未见初态至少 90%。Rapier 相对同一新策略在 MuJoCo 的成功率差，其 95% 置信区间下界不低于 −5 个百分点，无严重安全事件。Sprint 另验速度收益、转向和停止；各技能采用对应阶段判据。
- **场景与视觉：** 两种形态完成主环路、缓坡、塔底和舱室路线；检查安全点、切换、复位、物件和相机。固定机位及运动录像共同验收 URI 画风。
- **性能：** 发布配置预热三十秒、采样一百二十秒、重复五次；正常游玩达到 1080p／60 FPS，另完成十分钟组合游玩。记录帧时间、物理／推理耗时、实时比、内存和失败；录制开销另测。
- **交付：** 提供独立运行包、完整技能策略及契约，启动接口支持场景、形态、回放、无窗口验证和输出目录。更新 README 与交接文档；原始日志、轨迹、录像及审查记录留在 `.scratch/`。全部本轮门禁和 GPT 审查通过后再进入代码交付与发布阶段。

## 版本二次核验（2026-09-28）

- Bevy `0.19.1` 是最新正式发布版，MSRV `1.95`；`0.20.0-rc.1` 仍是候选版。[正式发布](https://github.com/bevyengine/bevy/releases/tag/v0.19.1)、[候选发布](https://github.com/bevyengine/bevy/releases/tag/v0.20.0-rc.1)。
- Rapier 最新正式版已是 `0.36.0`（9 月 25 日发布）。本项目先选择 `0.35.3`：它已包含近奇异 multibody 能量保护、motor／limit CFM、接触冲量累计与约束式 frictionloss 修复；`0.36.0` 新增软体并改变 pipeline、事件、contact 和快照接口，当前机器人没有相应需求。最新源码内的 Bevy binding 依赖 Bevy git main，不能当作已发布插件与 Bevy `0.19.1` 的兼容证明。[变更记录](https://github.com/dimforge/rapier/blob/v0.36.0/CHANGELOG.md)、[已发布插件实际依赖](https://docs.rs/crate/bevy_rapier3d/0.36.0/source/Cargo.toml)、[新 binding](https://github.com/dimforge/rapier/blob/v0.36.0/bindings/bevy_rapier/bevy_rapier3d/Cargo.toml)。
- `ort` 是 Rust 接口包；ONNX Runtime 是微软原生推理库，二者有不同版本号。`ort rc.13` 仍是最新发布；ONNX Runtime 最新正式版为 `1.30.0`，原计划的 `1.29.0` 已有后续修补和正式版本。[ort 发布](https://github.com/pykeio/ort/releases/tag/v2.0.0-rc.13)、[原生库发布](https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0)。
- 已在 DGX Spark 的 Rust `1.98.0` 上通过隔离验证：Bevy `0.19.1` 与 Rapier `0.35.3` 编译、同一物理世界的简单刚体六百步和 `f32` 位姿转换；`ort rc.13` 通过 API 28 加载官方 ARM64 ONNX Runtime `1.30.0`，执行三组输入的 `61→32/ReLU→14` FP32 模型，输出与 NumPy 参考在打印精度内一致。原生库归档 SHA256 与官方 release asset digest 一致。这些仅为接入与推理烟测，真实 MicroDuck、完整渲染、性能和技能门禁仍未通过；最终版本依据这些实际门禁固定，不能称已证明全局最优。CFM 与 frictionloss 修复改变实际求解行为，需重新验证关节刚度并避免重复施加摩擦。

| 依赖 | 固定下载地址 |
| --- | --- |
| Bevy `0.19.1` | [完整源码 ZIP](https://github.com/bevyengine/bevy/archive/refs/tags/v0.19.1.zip) · [Cargo 包](https://static.crates.io/crates/bevy/bevy-0.19.1.crate) |
| Rapier `0.35.3` | [完整源码 ZIP](https://github.com/dimforge/rapier/archive/refs/tags/v0.35.3.zip) · [Cargo 包](https://static.crates.io/crates/rapier3d/rapier3d-0.35.3.crate) |
| Rust `ort 2.0.0-rc.13` | [Cargo 包](https://static.crates.io/crates/ort/ort-2.0.0-rc.13.crate) |
| ONNX Runtime `1.30.0` Linux ARM64 CPU | [原生库 TGZ](https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-aarch64-1.30.0.tgz) |

ONNX Runtime 官方归档 SHA256：`e16a27a8ed330bbc698df7330b0cf56e722f354e3bcc92118682c74ef3c3e3da`。Rust 包正常由 Cargo 与锁文件自动下载；Bevy／Rapier 源码 ZIP 是完整仓库源码，不是独立编辑器或游戏运行包。
