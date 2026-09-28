# 60 Hz 固定步生命周期：真实 ONNX 诊断接线

更新日期：2026-09-28。`simulation_minigame::fixed_step_runtime::FixedStepRuntime` 持有唯一 `SimulationWorld`、`FixedStepClock(60 Hz, 每显示帧最多 8 步)` 与全局有序事件队列。每个到期边界依次排出事件、同步场景、检查同步阶段没有额外积分、验证恰好一次推理、提交力矩并只积分一次、提交时钟、核对物理／时钟计数、发布完成快照。任一生命周期错误使驱动永久停机，不重放已消费事件或部分物理步。暂停保留时间欠账；恢复后的输入时间从恢复时刻计算。

业务库的真实 Rapier fixture 覆盖首 Tick 顺序、600 Tick 各 600 次推理计数／力矩更新／积分／发布、500 ms 掉帧后的 `8/8/8/6` 步追赶、暂停与同 Tick 事件顺序，以及同步阶段偷积分、推理少／多次、非法力矩、发布失败等停机路径。基础库测试为 16 通过、10 项要求封闭源端 fixture 的测试按原门禁忽略；开启只读物理观测特性时为 19 通过、10 忽略。工程结构检查 414 项通过。

现有 `station_robot_60hz_diagnostic` 已改为通过该驱动执行真实策略，不再直接在自身循环里调用 `step_with_torques`。它用精确的整数纳秒序列逐个送入 60 Hz 时间边界；现有 P-only 诊断控制律、模型、场景碰撞与状态读写保持不变。对 [v18AJ 九模型视频](legacy_nine_onnx_aj_v7_rerecord.md)所用九个原始 ONNX 和冻结资产，逐个从初始状态重放，累计 **2,400 次推理、2,400 次积分**。每个模型独立重放两遍，输出报告逐字节相同；剔除新增 `fixed_step_driver` 和 `final_clock` 两字段后，**九份新报告与原视频物理报告的所有字段逐项相同**，包括每 Tick 观测、动作、实际 P-only 力矩、接触计数、完整姿态序列与最终物理快照。所有新时钟收据均为 60 Hz、零欠账，步数与原报告一致。两遍原始报告及含全部 SHA 的比较收据在本机 `.scratch/legacy_nine_native_probe/fixed_step_driver_v1/`，汇总 SHA256 `144233ce92e80a60dc4a8c219f9ad8bab90fb6c2fc5b46be2175da8121f3e7db`。旧视频与旧报告保留原始身份，并未重新标记为新驱动录制。

随后已从上述**新驱动报告**在同一冻结 v18AJ 资产上重新 GPU 截取九段，生成带 `FixedStepRuntime 60/60` 来源字幕的[新合辑和独立收据](legacy_nine_onnx_aj_fixed_step_video.md)：1,209 帧、1920×1080、30 fps、40.30 秒，视频 SHA256 `4bcc81c41f8783617f6642cf0987096489223e70423c19ac8911d0beb3b79735`。新旧 1,209 个姿态 SHA 一致，1,202 张 PNG 逐字节一致，其余七帧仅有 1–2 个像素微差；九段单片 SHA 均与旧单片相同。新合辑来源身份与字幕不同，旧证据未覆盖。此复录并没有改善这些旧策略的站立或行走表现。

开发工具现另有 `station_robot_live_preview` **实时窗口接线探针**。它与离线诊断复用同一套模型／场景装配和 P-only 控制器，在一个 Rapier 世界中按真实显示帧耗时推进 `FixedStepRuntime`，每个完成 Tick 都保留推理、积分和位姿账本；Bevy 当前显示帧取最后一张完成位姿，`RobotVisualInput` 在同帧 `PostUpdate` 验证并写回模型。程序只有在最终位姿的 `RobotVisualStatus` 为正确 Tick、且该发布后又完成 GPU 渲染流程时才成功退出。正式 `src/main.rs` 仍拒绝机器人模式，不把诊断控制器装进正式游戏。

用原始 `alpha_stand.onnx`、冻结 AJ 资产和 `MIN_TICKS=60` 的真实窗口实跑：因一次显示帧最多追 8 Tick，**实际完成 64 Tick**，超出下界 4 Tick；64 次推理、64 次积分、64 次位姿发布，报告 64 条逐 Tick trace 与含零步的 65 张位姿，视觉停在第 64 Tick，最终发布后有 1 次 GPU 渲染流程。收据 `.scratch/live_runtime_dev/alpha_stand_min60_v3.json` 的 SHA256 为 `ca1226ad7e790146d804b9ad5bf5f05c6558b8a371d05fa0aaedafca2dc85689`。调试构建仅 9 个显示帧完成本次窗口探针时尚欠 416 Tick（约 6.93 秒），因此**没有达到实时性能资格**；`passed=true` 只表示这条开发接线的计数和最终画面核对通过。抽出共享控制器后，另在全新输出目录重跑九份原 ONNX 离线报告，全部与冻结 `run_a` 报告逐字节相同，累计仍为 2,400 Tick；独立复核收据 `.scratch/live_runtime_dev/offline_refactor_recheck/comparison.json` 的 SHA256 为 `a3254ba5eeed136e882087feb27e97dd049640c918dab0c9abdc39d64aa417be`。实时入口的连续步、追帧取末帧与零步显示边界测试通过。

**资格边界：** 这些真实 ONNX 回放依旧是旧 200/50 策略在 60/60 的 P-only 诊断；它不包含合格 BAM 上一步外载、源接触等价、道具交互或技能成功。实时窗口探针仍在 `dev_tools`，20 个可移动道具碰撞体尚未导入；`src/main.rs` 当前只提供无机器人视觉预览，正式游戏还没有合格的机器人控制及场景交互。此项接线回归不能替代完整游戏、接触修正、十技能训练与 1080p／60 FPS 验收。
