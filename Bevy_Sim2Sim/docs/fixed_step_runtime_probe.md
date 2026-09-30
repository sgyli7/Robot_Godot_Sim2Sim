# 60 Hz 固定步生命周期：真实 ONNX 诊断接线

2026-09-29 更新：实时开发窗口 v3 已把每 Tick 推理与物理推进移至独立 60 Hz 工作线程，显示主线程只接收最新姿态；每 Tick 仍按连续编号核对。默认窗口呈现改为显式 FIFO，`SAI_LAB_LIVE_PROFILE=1` 可采样 Rapier 内建阶段计数器。下文所有已测性能数字、64 Tick 超跑和显示帧欠账均是**此前 v2 主线程追帧架构**的历史基线，不能当作 v3 的测量结果。v3 的 600 Tick／FPS／最终截图还待独占 GPU 复测，未获实时资格。

更新日期：2026-09-28。`simulation_minigame::fixed_step_runtime::FixedStepRuntime` 持有唯一 `SimulationWorld`、`FixedStepClock(60 Hz, 每显示帧最多 8 步)` 与全局有序事件队列。每个到期边界依次排出事件、同步场景、检查同步阶段没有额外积分、验证恰好一次推理、提交力矩并只积分一次、提交时钟、核对物理／时钟计数、发布完成快照。任一生命周期错误使驱动永久停机，不重放已消费事件或部分物理步。暂停保留时间欠账；恢复后的输入时间从恢复时刻计算。

业务库的真实 Rapier fixture 覆盖首 Tick 顺序、600 Tick 各 600 次推理计数／力矩更新／积分／发布、500 ms 掉帧后的 `8/8/8/6` 步追赶、暂停与同 Tick 事件顺序，以及同步阶段偷积分、推理少／多次、非法力矩、发布失败等停机路径。基础库测试为 16 通过、10 项要求封闭源端 fixture 的测试按原门禁忽略；开启只读物理观测特性时为 19 通过、10 忽略。工程结构检查 420 项通过。

现有 `station_robot_60hz_diagnostic` 已改为通过该驱动执行真实策略，不再直接在自身循环里调用 `step_with_torques`。它用精确的整数纳秒序列逐个送入 60 Hz 时间边界；现有 P-only 诊断控制律、模型、场景碰撞与状态读写保持不变。对 [v18AJ 九模型视频](legacy_nine_onnx_aj_v7_rerecord.md)所用九个原始 ONNX 和冻结资产，逐个从初始状态重放，累计 **2,400 次推理、2,400 次积分**。每个模型独立重放两遍，输出报告逐字节相同；剔除新增 `fixed_step_driver` 和 `final_clock` 两字段后，**九份新报告与原视频物理报告的所有字段逐项相同**，包括每 Tick 观测、动作、实际 P-only 力矩、接触计数、完整姿态序列与最终物理快照。所有新时钟收据均为 60 Hz、零欠账，步数与原报告一致。两遍原始报告及含全部 SHA 的比较收据在本机 `.scratch/legacy_nine_native_probe/fixed_step_driver_v1/`，汇总 SHA256 `144233ce92e80a60dc4a8c219f9ad8bab90fb6c2fc5b46be2175da8121f3e7db`。旧视频与旧报告保留原始身份，并未重新标记为新驱动录制。

随后已从上述**新驱动报告**在同一冻结 v18AJ 资产上重新 GPU 截取九段，生成带 `FixedStepRuntime 60/60` 来源字幕的[新合辑和独立收据](legacy_nine_onnx_aj_fixed_step_video.md)：1,209 帧、1920×1080、30 fps、40.30 秒，视频 SHA256 `4bcc81c41f8783617f6642cf0987096489223e70423c19ac8911d0beb3b79735`。新旧 1,209 个姿态 SHA 一致，1,202 张 PNG 逐字节一致，其余七帧仅有 1–2 个像素微差；九段单片 SHA 均与旧单片相同。新合辑来源身份与字幕不同，旧证据未覆盖。此复录并没有改善这些旧策略的站立或行走表现。

开发工具现另有 `station_robot_live_preview` **实时窗口接线探针**。它与离线诊断复用同一套模型／场景装配和 P-only 控制器，在一个 Rapier 世界中按真实显示帧耗时推进 `FixedStepRuntime`，每个完成 Tick 都保留推理、积分和位姿账本；Bevy 当前显示帧取最后一张完成位姿，`RobotVisualInput` 在同帧 `PostUpdate` 验证并写回模型。无截图运行只有在最终位姿的 `RobotVisualStatus` 为正确 Tick、且该发布后渲染调度又进入一次 `RenderSystems::Cleanup` 时才成功退出；这个计数不是 GPU 绘制完成、像素呈现或耗时证明。追加可选 `--final-png NEW_IMAGE.png` 时，程序冻结最终物理步、核对 Ready 状态，等待后续渲染并实际读回窗口 PNG；文件、尺寸、SHA256 和对应 Tick 都通过才写成功收据。正式 `src/main.rs` 仍拒绝机器人模式，不把诊断控制器装进正式游戏。

用原始 `alpha_stand.onnx`、冻结 AJ 资产和 `MIN_TICKS=60` 的真实窗口实跑：因一次显示帧最多追 8 Tick，**实际完成 64 Tick**，超出下界 4 Tick；64 次推理、64 次积分、64 次位姿发布，报告 64 条逐 Tick trace 与含零步的 65 张位姿，视觉停在第 64 Tick，最终发布后记录到 1 次渲染调度 Cleanup。收据 `.scratch/live_runtime_dev/alpha_stand_min60_v3.json` 的 SHA256 为 `ca1226ad7e790146d804b9ad5bf5f05c6558b8a371d05fa0aaedafca2dc85689`。调试构建仅 9 个显示帧完成本次窗口探针时尚欠 416 Tick（约 6.93 秒），因此**没有达到实时性能资格**；`passed=true` 只表示开发接线的计数和 Bevy 位姿状态核对通过，不是像素回读。抽出共享控制器后，另在全新输出目录重跑九份原 ONNX 离线报告，全部与冻结 `run_a` 报告逐字节相同，累计仍为 2,400 Tick；独立复核收据 `.scratch/live_runtime_dev/offline_refactor_recheck/comparison.json` 的 SHA256 为 `a3254ba5eeed136e882087feb27e97dd049640c918dab0c9abdc39d64aa417be`。实时入口的连续步、追帧取末帧与零步显示边界测试通过。

同一原始 `alpha_stand` 的无截图发布构建窗口再分别跑最少 60 与 600 Tick：实际完成 `64/600` Tick，推理、积分、发布均逐项同数；结束时分别积欠 `415` 与 `3,899` Tick（约 `6.93/64.99 s`），进程墙钟约 `19.93/85.87 s`，其中包括场景／GPU 启动和最终报告落盘。后者的运行时钟总需求约 75 秒而只完成了 10 秒仿真，故持续吞吐约 8 Tick/s，**不能称为实时 60 Hz**。收据分别在 `.scratch/live_runtime_dev/alpha_stand_release_min60.json`（SHA `dc433bca…`）与 `alpha_stand_release_min600.json`（SHA `57bb9feb…`）。相同模型、场景和 600 Tick 在无窗口的**发布构建**中总墙钟约 `5.14 s`（调试构建约 `6.11 s`），逐 Tick trace、位姿、最终物理状态与窗口版完全相同；发布离线收据 `.scratch/live_runtime_dev/alpha_stand_offline_release_min600.json` 的 SHA 为 `3477950e…`。这把额外耗时定位到窗口相关路径，但尚未区分 Bevy 位姿应用、渲染、呈现等待及调度，不能把时钟欠账直接称为 GPU 耗时。

独立的**截图证据**使用原始 `alpha_stand` 与同一冻结 AJ 资产、最少 60 Tick：真实窗口完成 64 次推理、64 次积分和 64 次发布，`RobotVisualStatus::Ready`、最终发布位姿与读回 PNG 收据均绑定 `global_step=64`。1920×1080 PNG 的 SHA256 为 `b279ca8c45fbad2900a58ab6066c992350e729bc62d78f848f3724b46d2754b6`，路径 `.scratch/live_runtime_dev/alpha_stand_min60_capture_v2.png`；JSON 收据 SHA 为 `4ba706abf3e48275b1cf572cf403b76478ed7a01e42269dd9a518fe8f9856587`。截图的 Follow 镜头中机器人可见；故意令 PNG 保存失败的实跑以非零码退出，报告 `passed=false`、无 PNG 收据。截图后等待与 GPU 读回不用于衡量 60 Hz 吞吐；性能数字以上述**无截图**发布构建为准。此 PNG 只证明一个诊断时刻实际出图，`live_visual_qualified`、`performance_qualified`、BAM／接触／技能资格仍为 `false`。

为定位窗口欠账，`dev_tools` 窗口另有**显式开启**的测量模式：设置 `SAI_LAB_LIVE_PROFILE=1`，可选 `SAI_LAB_LIVE_PROFILE_RESOLUTION=1920x1080|960x540` 和呈现模式环境变量；不开启时维持原 1920×1080 `AutoVsync` 行为，报告不新增 `live_profile`。测量报告记录请求配置、实际图形适配器名称／后端、显示帧起点间隔和 `FixedStepRuntime::advance_frame` 主线程 CPU 耗时的次数／总和／最大值。两组时间可能与 Bevy 的并行渲染重叠，不能相减后把余量称作 GPU 耗时；报告的 `requested_present_mode` 只记录请求，具体配置另用底层日志核对。

同一发布二进制、`alpha_stand`、冻结 AJ 资产、最少 600 Tick、无截图的受控实跑如下；四种配置的**前 600 条推理／物理 trace 和 601 张位姿**均与上述发布版离线报告逐项相同，适配器均为 NVIDIA GB10／Vulkan。帧起点时间不含启动前等待与退出后的报告落盘；墙钟包含这些成本。

| 请求窗口 | 实际完成 Tick／显示帧 | 末尾欠账 Tick | 帧起点间累计 | 固定步推进 CPU 累计 | 进程墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1920×1080，AutoVsync | 600／76 | 3,899 | 75.00 s | 10.89 s | 86.55 s |
| 960×540，AutoVsync | 600／76 | 3,899 | 74.99 s | 10.17 s | 86.23 s |
| 1920×1080，AutoNoVsync | 601／77 | 252 | 14.23 s | 11.27 s | 19.98 s |
| 1920×1080，AutoNoVsync 复跑 | 600／79 | 83 | 11.39 s | 8.87 s | 18.25 s |
| 960×540，AutoNoVsync | 603／78 | 354 | 15.96 s | 13.25 s | 23.21 s |

这些数据说明在本机此窗口里，请求 AutoNoVsync 明显减少帧间等待和时钟欠账，而分辨率减半没有稳定改善；尚不能区分交换链／呈现等待、渲染线程及系统负载的各自份额，也不代表 AutoNoVsync 已达到实时资格。独立短跑以 `RUST_LOG=wgpu_core::device::resource=debug` 读取实际提交给 wgpu 的 `SurfaceConfiguration`：本机 1920×1080 下 `AutoVsync → FifoRelaxed`，`AutoNoVsync → Immediate`，两次短跑均成功退出。该配置在 Bevy 选择受支持的 surface mode 后写入，进一步说明 A/B 确实切换了交换链模式；它不是 Vulkan 驱动完成呈现或 GPU 时长的证据。日志分别在 `.scratch/live_runtime_dev/present_{vsync,novsync}_debug.log`。五组完整报告在 `.scratch/live_runtime_dev/alpha_stand_release_profile_{1080_vsync,540_vsync,1080_novsync,1080_novsync_repeat,540_novsync}_min600.json`；相应 SHA256 依次为 `5412d556…`、`3391aecf…`、`5b423931…`、`50f57c09…`、`758afa96…`。下一步要分段测 Bevy 渲染／呈现与固定步内部的 ORT、Rapier、快照和发布。

**资格边界：** 这些真实 ONNX 回放依旧是旧 200/50 策略在 60/60 的 P-only 诊断；它不包含合格 BAM 上一步外载、源接触等价、道具交互或技能成功。实时窗口探针仍在 `dev_tools`，20 个可移动道具碰撞体尚未导入；`src/main.rs` 当前只提供无机器人视觉预览，正式游戏还没有合格的机器人控制及场景交互。此项接线回归不能替代完整游戏、接触修正、十技能训练与 1080p／60 FPS 验收。
