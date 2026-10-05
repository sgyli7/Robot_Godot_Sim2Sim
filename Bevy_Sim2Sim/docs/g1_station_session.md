# 科学站 G1 中文操作入口

这是有限的开发实验入口，支持固定苹果取放和固定棕箱到蓝容器搬运。T1 使用 N1.7／AGILE，T2 使用本地 Qwen、N1.6／Homie v2；后续几何控制与导航单独披露。它不开放任意目标，也不代表两项已达到 8/10。完整结果见 [阶段记录](g1_milestone.md)，权重、身体与未闭合的许可来源见 [制品清单](g1_artifact_inventory.md)。

需要本机已经准备好的 `/home/ethan/models/unitree_g1` 缓存、原生 ORT、两套隔离 Python 环境、X11/Vulkan 桌面以及配置所绑定的科学站、纹理和校准文件。移动任务还需要既有的 localhost Qwen 服务脚本与镜像。启动器不安装权重、不改服务配置；任何 SHA256 不匹配由原加载器拒绝。

在 `Bevy_Sim2Sim` 中构建。此共享主机使用已有构建锁与两项编译并发：

```bash
flock /tmp/sai-g1-cargo.lock env CARGO_BUILD_JOBS=2 cargo build --locked \
  --bin bevy_sim2sim --features \
  dev_tools,dev_tools_minigame/g1_constraint_diagnostic,dev_tools_minigame/g1_source_lighting
export DISPLAY=:1 XDG_RUNTIME_DIR=/run/user/1000
```

运行时使用构建命令实际生成的二进制路径；本机共享 target 是 `/home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target`。下面的配置来自已保存的真实运行，保留身体、相机、动作时序和物理参数；准备程序生成新的 episode、服务端口和操作入口配置。episode 默认取当前 Unix 毫秒，也可用 `--episode-id` 指定新的正整数。

静态入口：

```bash
python3 crates/dev_tools/python/scripts/unitree_g1_station_session.py \
  --profile static \
  --config /home/ethan/ProjectBackups/2026-10-05/Sai_Lab/g1_fixed_t1_and_loaded_turn_failure_001/evidence/t1_observed_frozen_suite_20261005_0524/position_1_seed_0/config.json \
  --binary /home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target/debug/bevy_sim2sim \
  --output .scratch/g1_static_window_001 --maximum-episodes 4 --record
```

移动入口：

```bash
python3 crates/dev_tools/python/scripts/unitree_g1_station_session.py \
  --profile mobile \
  --config /home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_both_thumb_fix_milestone_001/evidence/t2_both_thumb_serial_load_20261004_0456/position_2_seed_0/config.json \
  --binary /home/ethan/Projects/Sai_Lab/Bevy_Sim2Sim/target/debug/bevy_sim2sim \
  --output .scratch/g1_mobile_window_001 --maximum-episodes 4 --record
```

输出目录必须是新目录。首次执行可增加 `--prepare-only`，使用另一个输出目录，检查配置准备；该选项不启动进程、不执行推理或物理。服务分别独占 localhost 5557／5558，移动任务的 Qwen 为 8002。先加载 N1.6，再按既有隔离配置启动自己的 Qwen。启动器在加载前持有跨 profile／工作树的 G1 会话文件锁；另一个启动器会立即拒绝，不并发加载。占用中的端口或运行中的 G1 Qwen 同样使启动器退出。关闭时核对原容器 ID 和启动时间，拒绝停止同名但已重启的服务。该锁不抢占其他应用，进程退出后由内核释放，不删除仍可能被持有的锁文件。

静态入口可另加 `--with-local-qwen`：先加载 N1.7，再启动同一受管的本地 Qwen，由零 Tick 的真实原配相机图像准入固定苹果任务。它使用独立的 `scientific_station_static_from_instruction_v1` 合同；原配 AGILE、60 Tick 站立准备、一次 40 帧 N1.7、2 秒动作图像时限及后续实际双视图定位保持原样。该接入只有“开始、再观察、停止”的启动决策，没有移动任务的箱子／容器目标合同、结束模型反馈或自动重试。2026-10-05 实际回归已核验两个零 Tick 原配 RGB／Qwen 请求、重置后不同中文措辞进入实际模型输入，以及一次 N1.7／1044 Tick／放稳 6.26 秒；该开发样例不替代冻结十例成绩。

初始化后，输入框保留当前固定中文任务。静态支持“把苹果放到盘子里”“把苹果放到盘中”“将苹果放入盘子”；移动支持默认句或“把 A 区的棕色箱子搬到 B 区的蓝色容器里”。通过任务检查的原输入进入该回合的 Qwen 请求；加载后的配置不变，重置清空输入覆盖与旧请求。点击启动或 Ctrl+Enter 开始；点击停止或 Esc 暂停物理并使旧命令失效。停止或完成后必须先重置，再启动下一回合。重置创建新的 episode，清空旧图像、请求、动作、控制历史和决策状态；同一模型服务继续驻留。全景／抓取细节按钮只改变展示相机。

关闭窗口或在启动器终端按 Ctrl+C 即结束。默认会话上限 1,800 秒，回合上限 4；可设置 `--session-timeout`（最多 7,200 秒）与 `--maximum-episodes`（2–10）。结束时回收自己的应用、录像和策略进程，并只停止自己启动且 owner 标签一致的 Qwen。若清理失败，`session_receipt.json` 明确记录失败，不声称资源已释放。`--record` 录制实际所属窗口的前 150 秒，长会话不能把这段录像当作全程录像。

每回合仍最多 1 次 N1.7／4 次 N1.6，最多 1,100／3,300 个物理 Tick；正式物理保持 50 Hz，每 Tick 一次 20 ms 积分。图像和模型边界有明确暂停。暂停中的墙钟时间单独记录，不能据此宣称全程连续 1×。移动 UI 在回合之间使用同一模型随机流；冻结十例验收另用原配 `--episode-seed-reset` 服务和种子清单，两者成绩不能混用。

检查入口与清理回归：

```bash
python3 -m unittest discover -s crates/dev_tools/python/tests \
  -p test_unitree_g1_station_session.py -v
```

`session_receipt.json` 保存实际模型与应用命令、PID、初末健康计数、配置和二进制 SHA256、清理结果。`native/station_controls_receipt.json` 保存按钮事件、停止区间、回合身份和当前产物目录；根 `native/owner_steps.jsonl` 记录全会话的实际积分，不在重置时覆盖。新回合的图像与放置报告位于 `native/episode_<id>/`。按钮测试成功与任务放置成功是分别核验的结果。

有限按钮回归可额外使用 `--maximum-episodes 2 --smoke-stop-reset`：走同一按钮消息路径，运行至 60 Tick、暂停 30 个实际渲染帧、重置并执行新回合。移动入口的 `--smoke-reset-during-qwen` 是单独故障测试：在零 Tick 的真实初始 Qwen 请求尚未返回时停止并重置；若错过该边界则测试失败，不冒充在途重置。新世界等待旧 HTTP 结束并丢弃结果，之后重新取图。它不改变模型或物理参数，不计入冻结十例成绩。

2026-10-05 的实际移动在途重置回归：旧请求在零 Tick 被停止，结果随后丢弃；新 episode 完成 2527 Tick、四次 N1.6、三次本回合 Qwen、1.99 米搬运和 2.76 秒放稳。完整原片保留，开发回归不计入冻结 T2 的 2/10。重置恢复初始原配相机和标记 gate 状态；定位 CPU 子进程使用一个 BLAS 线程，保留 3 秒时限，物理及模型服务配置没有变化。服务 timeout／断连的零 Tick 原生实验分别保留为失败，不计算任务成功。
