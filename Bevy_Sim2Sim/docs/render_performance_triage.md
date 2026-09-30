# 实时窗口性能诊断

本页说明开发入口 `station_robot_live_preview` 的测量方法和报告字段。该入口使用一个 Rapier 世界、独立的 60 Hz 物理／推理线程与 P-only 诊断控制器。它不能授予正式游戏、BAM 控制器或机器人技能资格。

实验报告、原始 trace、截图、逐帧查看器和审查记录保存在所属仓库的 `/home/ethan/ProjectBackups/<date>/Sai_Lab/<run_name>/`，不得作为 Agent 临时产出物入库。一次运行应保存构建 feature、完整参数、输入文件 SHA256、环境变量和桌面呈现状态，以便复现。

## 构建与开关

在 Bevy 工程根运行。普通开发窗口不启用 Rapier 计时：

```bash
cargo build -p dev_tools_minigame --release \
  --bin station_robot_live_preview --features rendering_preview
```

需要逐 Tick 原生计数器时，显式启用非默认开发 feature：

```bash
cargo build -p dev_tools_minigame --release \
  --bin station_robot_live_preview \
  --features rendering_preview,live_physics_profile
```

需要 Chrome 时间线时，另加 `bevy/trace_chrome`：

```bash
cargo build -p dev_tools_minigame --release \
  --bin station_robot_live_preview \
  --features rendering_preview,live_physics_profile,bevy/trace_chrome
```

开启时间线时，`TRACE_CHROME` 指向归档内的新 JSON 文件；`RUST_LOG` 必须包含 `info`，才能记录 `sim2sim_live_frame` 和 `sim2sim_worker_tick` 标记。例如 `info,bevy_ecs::query=warn,bevy_ecs::schedule::executor=warn,wgpu=warn`。追踪会增加运行开销；实际吞吐应另用不含 Chrome 追踪的构建测量。

需要分开测量主线程的 CPU 与墙钟时间时，显式添加 `live_render_profile` feature，并设置 `SAI_LAB_LIVE_PROFILE=1`。该开关可与上述构建组合，也可不启用 Chrome 追踪。Linux 使用线程 CPU 时钟；其他平台的 CPU 值为不可用，不用零值代替。

| 环境变量 | 合法值 | 默认值 |
| --- | --- | --- |
| `SAI_LAB_LIVE_PROFILE` | `1`；不设置则关闭 | 关闭 |
| `SAI_LAB_LIVE_PROFILE_RESOLUTION` | `1920x1080`、`960x540` | `1920x1080` |
| `SAI_LAB_LIVE_PROFILE_PRESENT_MODE` | `fifo`、`auto_vsync`、`auto_no_vsync`、`mailbox` | `fifo` |
| `SAI_LAB_LIVE_PROFILE_ABLATION` | `none`、`signage_cameras`、`msaa`、`shadows` | `none` |
| `SAI_LAB_LIVE_PROFILE_MAIN_BUDGET_MS` | `0` 至 `16` 的有限毫秒值；需要 `live_render_profile` 和 Linux 线程 CPU 时钟 | `0` |

设置其余测量变量时必须同时设置 `SAI_LAB_LIVE_PROFILE=1`。普通窗口也使用显式 FIFO。受控测量冻结相机输入；每次对照只改变一个变量，并保持模型、qpos、策略、外观、场景、Tick 数和原生 ORT 库一致。

入口参数顺序：

```text
station_robot_live_preview MODEL MODEL_SHA QPOS QPOS_SHA ASSETS ONNX
  POLICY_CONTRACT|--legacy-original NATIVE_ORT_LIB MIN_TICKS
  APPEARANCE APPEARANCE_SHA NEW_REPORT.json [--final-png NEW_IMAGE.png]
```

模型、qpos 和外观参数使用对应输入文件的 SHA256。输出路径必须是尚不存在的新文件；最终 PNG 保持实际测量相机。启动阶段等待机器人外观、GPU 管线和静态标牌烘焙就绪，随后才启动计时工作线程。

## 报告字段与时间边界

报告 schema 为 `station_robot_live_preview_v3`。`SAI_LAB_LIVE_PROFILE=1` 时挂载 Bevy 自带的帧时间、实体数量、渲染和网格分配诊断，并写入 `live_profile`。

| 字段 | 含义 |
| --- | --- |
| `passed` | 连续 Tick、推理、积分和姿态发布账本闭合 |
| `worker_60hz_deadline_met` | 工作线程无误期，且在请求 Tick 数的 60 Hz 时间预算内结束 |
| `performance_qualified` | 开发入口固定为 `false` |
| `frame_samples` | 每个活跃主循环起点的帧编号、运行相对时间、上次间隔、已显示步号和渲染通道样本 |
| `worker_tick_samples` | 每个物理 Tick 的 `advance_cpu_ns` 和 `deadline_lag_ns` |
| `rapier_counters_per_tick` | 启用 `live_physics_profile` 后的 Rapier 原生阶段计时 |
| `station_label_cameras_baked` | 烘焙完成后停止重复绘制的静态标牌相机数 |
| `station_mass_properties_prepared` | 加载时准备质量属性的不可变站体碰撞体数 |

帧间隔从主线程 `advance_live` 的相邻起点计算，排除启动与最终写盘；它表示窗口主循环吞吐，不等于显示器扫描频率。`start_since_run_ns` 的时间原点是运行初始化，工作线程的截止时间原点则是线程启动，二者不能直接当作同一时钟坐标。

`sim2sim_live_frame` 仅覆盖该次开发更新系统，整帧还包含其他主调度、渲染与窗口等待。分析整帧时，应按相邻标记划分区间，并保留与区间相交的跨帧跨度。`sim2sim_worker_tick` 标记包含该 Tick 的推进和发布准备；`advance_cpu_ns` 是兼容保留的旧字段名，实际记录 `session.advance_frame` 调用的墙钟时间，包含线程停调度，不含其后的姿态提取与工作消息发送。`worker_advance_timing_semantics` 明确该语义；CPU 收据 v2 输出 `worker_tick_wall` 与 `advance_frame_wall`，这些值不能当作线程 CPU 时间。

渲染表保存主线程帧开始时最新已完成的 Bevy 通道样本，可能落后当前帧。单通道 CPU／GPU 时间不能当作整帧 GPU 总耗时；嵌套跨度和并行线程时间也不能直接相加或相减作为整帧墙钟。

### Rapier 原生计时

`ccd_toi_ns` 对应 `ccd.toi_computation_time`，记录实际 TOI／运动夹紧阶段。旧 `ccd_ns` 字段为兼容保留，当前管线不填充其 `stages.ccd_time`；零值不能推导 CCD 没有成本。`update_ns` 包含子步质量属性维护，`collision_detection_ns` 记录碰撞检测阶段。

其他阶段包括广相、窄相、岛构建、约束维护、求解和用户变更处理。字段为零可能表示阶段没有执行、计时器未使用或计时功能不可用；应结合编译 feature 和实际调用路径解读。接触数量从物理快照读取，不依赖没有接入管线的计数 setter。

### 渲染交接与主线程 CPU

`live_profile.render_thread_cpu` 仅在 `live_render_profile` 测量构建中出现。它包装 Bevy 原有提取回调，保留调度与通道交接顺序。每条样本包含阶段、活跃帧编号、相对运行起点、墙钟跨度和线程 CPU 纳秒值：

- `handoff`：主线程等待渲染世界返回、执行提取并交回渲染线程；包含等待期间在主线程执行的协作渲染准备任务。
- `extract`：渲染世界返回后执行原有提取回调；包含在 `handoff` 内，不可再累加。
- `main_frame_interval`：相邻活跃更新起点之间的整段主线程 CPU 与墙钟时间；包含游戏更新和渲染相关任务。
- `render_schedule`：渲染调度调用线程的 CPU 与墙钟时间；不包含并行工作线程的 CPU 总和。
- `main_budget_probe`：显式启用后在主调度 `Last` 阶段加入的纯计算负载，以线程 CPU 时钟计量；只在活跃测量帧执行。

测量 feature 同时记录 `render/live_frame_gpu/elapsed_gpu`，用硬件时间戳包围根渲染图产生的命令缓冲。起止标记在图执行前后按提交顺序插入，结束标记先于诊断查询解析；该跨度包含渲染图全部 GPU 命令及其间隙，排除呈现节奏等待、渲染图外的截图与读回。硬件时间戳不支持时该 GPU 字段不可用。不要再将其与内部通道相加。

主线程预算须同时记录已有 CPU 消耗、新增负载、帧间隔尾部、物理工作线程误期及交接等待。纯计算负载只估计当前场景的计算余量；真实业务的内存访问、锁、主线程同步推理或共享 GPU 推理需单独复测。不要将 FIFO 等待全部视作可无条件使用的预算，也不要在基准采集时同时编译或运行其他负载。

墙钟减线程 CPU 的差值包括等待及被操作系统停调度的时间，不能仅靠差值证明某个特定阻塞函数。应结合 Bevy 时间线与驱动级记录定位。例如 `prepare_windows` 包含取得交换链图像的全部路径，其内部可能在 Vulkan fence 上等待；不能将外层跨度直接称为提交计算。最终截屏及关闭窗口的阶段不进入这些活跃样本。

需要驱动级对照时可用 Nsight Systems 的 `vulkan,osrt` 追踪，并以 `--vulkan-gpu-workload=batch` 采集提交批次，减少逐 draw 追踪的额外开销。保留原始记录，区分初始化、活跃帧和关闭阶段，避免把所有嵌套 GPU 范围相加作为 GPU 忙碌时间。

## 静态内容生命周期

`StationVisualPlugin` 统一管理静态标牌的烘焙状态。字体、字形和渲染管线连续就绪后，保留文字纹理与 3D 标牌，关闭其离屏相机；`StationLabelBakeStatus` 保存独立运行时状态，不改写已加载的 `StationScene` 配置。

诊断加载器通过同一个 Rapier 原生方法准备不可变站体的默认密度质量属性，保持碰撞体插入顺序，不提前推进碰撞或动力学步骤。CCD 固定场景缓存不会因纯力／力矩或唤醒更新而失效，并在碰撞体、位置、类型、启用状态或移除发生变化时失效；无活跃 CCD 的步骤也必须记录几何失效。

正式发布构建按工程规范排除开发 feature 和开发场景。静态渲染生命周期属于运行时渲染模块；测量、实验接线和收据工具归属开发模块。

固定镜头及重复机器人姿态仅在实际值发生变化时改写 ECS 组件，避免无效的 `Changed` 标记。相机切换、跟随、真正的新姿态和可见性恢复仍必须立即更新；姿态绑定、覆盖检查及错误隐藏逻辑继续完整执行。

## 分析与对照流程

1. 记录显示连接、锁屏、窗口可见性、呈现模式和其他 GPU 活动。远程锁屏可能改变 FIFO 等待；先用相同旧构建在相同桌面状态下对照，避免把环境恢复全部归因于代码。
2. 冻结全部输入和画质参数，保存每个版本的构建与运行参数。分别采集无追踪吞吐和带追踪的热点时间线。
3. 按帧查看主调度、渲染、窗口等待与物理线程；确认标记配对完整，区分计算与等待，保留最长帧及跨帧 Tick 证据。
4. 比较所有动作、逐 Tick 位姿、初末快照、实体数量和场景标识，确认优化符合既定状态演进约束。改变轨迹的试验必须保留拒绝原因和证据。
5. 报告帧间隔及 Tick 墙钟耗时的 P50／P95／P99／峰值、工作线程误期次数和测量范围，避免用单次均值推导精确加速倍数。

原始 Chrome JSON 可在 Chrome tracing 或 Perfetto 中查看；若采用逐帧查看器，其图表应明确主线程、渲染线程与物理线程的共享时间轴，以及 GPU 通道样本的滞后语义。

CPU 收据工具在工程根运行：

```bash
PYTHONPATH=crates/dev_tools/python/src \
  python3 -m bevy_microduck_tools.live_perf_receipt \
  /path/to/archive/run_report.json /path/to/archive/new_receipt.json
```

该工具严格核对连续 Tick、真实误期计数、帧起点与间隔、样本与汇总的一致性，并输出最近秩分位数；旧 v3 报告缺少新增逐帧字段时相应分布保持不可用。输出拒绝覆盖既有文件，始终保持 `performance_qualified=false` 与 `skill_qualified=false`。

完整控制器、源接触等价和技能验收另行执行，入口边界见 [固定步与实时接线](fixed_step_runtime_probe.md)。
