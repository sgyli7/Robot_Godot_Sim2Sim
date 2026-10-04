# G1 阶段运行与录像

2026-10-04 将当前 G1 开发成果按用户要求合入主干，并在 Bevy 主页展示真实运行录像。完整任务目标继续保留；本页中的诊断成功不授予正式任务能力。

## 已录制的真实运行

| 项目 | 科学站取放 | 源任务近似场景搬箱 |
| --- | --- | --- |
| 新 episode | 20619 | 20620 |
| 原生积分／身体控制更新 | 1,046／1,046 | 2,134／2,134 |
| 新任务策略调用 | N1.7：1 次，原配 40 帧 | N1.6：4 次，原配每块 50 帧 |
| 本片 Qwen 调用 | 0 | 0 |
| 身体与执行器 | 原配 AGILE／native force based | 原配 Homie v2／native force based |
| 明确启用的诊断配置 | 预测关节限制，16 PGS | 4 PGS |
| 严格独立放置检查 | 6.02 秒 | 2.52 秒 |
| 活跃仿真／墙钟时间比 | 0.998699 | 0.998795 |
| 待处理 Tick／控制误期 | 0／7 | 0／0 |
| 环境 | 科学站的 2,553 个真实静态碰撞体，同源渲染 | 原配任务近似仓储场景；科学站搬箱待接入 |

两轮都保持 50 Hz、每 Tick 仅一次 20 ms 积分，没有隐藏时间子步、物体附着约束、动画驱动物理或验收真值输入。独立检查使用原始物体全部碰撞顶点、当前支撑接触、手分离、站立和冻结速度限制；不将检查结果作为控制观察。模型和图像边界的显式暂停仍然存在。

搬箱完整运行的箱子水平位移为 1.990227 m（三维位移 2.003680 m），机器人净水平位移为 1.322950 m。初始抓取使用原配 VLA；后续运动由公开地图、实际 RGB 标记定位、自身状态和传统几何控制执行。本片没有再次调用 Qwen。另一次 episode 20618 使用当前新 RGB 验证目标集合，让本地 Qwen 选择搬运，再执行同一路径并观察新图像；两次新 Qwen 请求均及时收到。录制轮与该轮的 2,134 个身体／物体物理步骤完全一致，区别是 episode 和模型边界等待时间。

科学站录像与先前成功 episode 20504 的全部 1,046 个身体／物体物理步骤也一致。本次录像脚本在窗口自行关闭时曾误报“录像先结束”；应用、编码器均正常退出，637 帧完整可解码，独立放置检查通过。保留原始错误和另外的审计说明，没有为修正录像脚本再跑一次物理。两段原片各有一帧由编码器为恒定帧率重复采样，未改变时长。

## 主页媒体与身份

两段 GIF 都是连续 **12.00 秒**墙钟原速节选、150 帧（12.5 fps），没有时长缩放。科学站片段取原片 7–19 秒并裁切任务区；搬箱片段取原片 58–70 秒并裁切机器人和容器。裁切只用于媒体导出；原始运行使用 1920×1080、8×MSAA，机器人相机和模型输入保持原设置。选段排除了窗口首次绘制之前的桌面占位内容。

录像源代码身份：`4805489f39ef0d6e4f2f0f2ab7ca7d66d12b4821`。实际应用二进制 SHA256：`ee5829f381b2d7bc9fd53355f56b33f2901fa6a627f3473e02a5b58f29605381`。

- 科学站完整原片 SHA256：`678aa629470c07c65f4a4bc9fb56b51a430c13810d9d70635317a1f1fe083f52`。
- 搬箱完整原片 SHA256：`f6fd8573df929561568464bf823ee135a876938c8e0fdc5021980f595fc927d6`。
- 本体、模型、原配预处理与许可：[G1 制品清单](g1_artifact_inventory.md)。任务策略分别为 T1 revision `7f78bebf1a90131e7304beacfcd47eb27bad16ab` 和 T2 revision `dfe74af855007f26093f362cd2d7a2f404b64b93`，不混接合同。

外部证据保存到 `/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_milestone_delivery_001/`；权重、原片、日志、临时配置和审计报告不进入普通 Git。GIF 是用户要求发布的主页媒体。

## 本机准备与运行

在 `Bevy_Sim2Sim` 中运行。需要已准备的 `/home/ethan/models/unitree_g1/` 本体／权重缓存、两套隔离 Python 环境、原生 ORT、可用 X11/Vulkan 桌面和冻结配置引用的校准／模板文件。克隆仓库不会下载这些外部制品。配置拒收 SHA256 不匹配的内容；详细准备入口见 [T1](g1_t1_source_task.md)、[T2](g1_t2_source_task.md)、[本体与控制](g1_agile_native_runner.md)。

```bash
cargo build --locked --bin bevy_sim2sim --features \
  dev_tools,dev_tools_minigame/g1_constraint_diagnostic,dev_tools_minigame/g1_source_lighting
```

科学站录制轮使用如下服务和应用命令；模型服务前台运行，另一终端启动应用。服务监听本机 5557，不应与其他拥有者共享该端口。冻结配置是外部证据中 `g1_station_milestone_recording_20261004_0419/config.json` 的原始副本。

```bash
/home/ethan/models/unitree_g1/envs/policy_onnx_cu13/bin/python \
  crates/dev_tools/python/scripts/unitree_g1_static_server.py \
  --receipt /home/ethan/Projects/Sai_Lab/.scratch/unitree_g1/policy/static_apple_files.json \
  --device cuda --port 5557 --seed-offset 0

./target/debug/bevy_sim2sim --scene g1_static_observed_place_diagnostic --robot g1 \
  --g1-config /home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_milestone_delivery_001/evidence/g1_station_milestone_recording_20261004_0419/config.json \
  --g1-ticks 1100 --output .scratch/g1_station_new_run
```

搬箱服务和应用使用另一套 N1.6 环境；两套重模型串行运行。冻结配置为外部证据中 `g1_mobile_milestone_recording_20261004_0420/config.json`。

```bash
/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python \
  crates/dev_tools/python/scripts/unitree_g1_mobile_server.py \
  --gr00t-source /home/ethan/Projects/Sai_Lab/upstream/unitree_g1/isaac_gr00t_n16 \
  --model-root /home/ethan/models/unitree_g1/mobile_box/dfe74af855007f26093f362cd2d7a2f404b64b93 \
  --receipt /home/ethan/Projects/Sai_Lab/.scratch/unitree_g1/policy/mobile_box_files.json \
  --runtime-env /home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13 \
  --port 5558 --seed 42

./target/debug/bevy_sim2sim --scene g1_mobile_auxiliary_release_diagnostic --robot g1 \
  --g1-config /home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_milestone_delivery_001/evidence/g1_mobile_milestone_recording_20261004_0420/config.json \
  --g1-ticks 3150 --output .scratch/g1_mobile_new_run
```

应用达到有限预算或反馈阶段会自动退出。手动停止使用应用窗口关闭／对应终端 Ctrl+C；之后对自己前台启动的模型服务 Ctrl+C。录制驱动另保存于外部证据，负责识别自己应用的 `_NET_WM_PID`、只录对应窗口、有限超时及回收自己启动的子进程。不会结束其他实验。

独立检查新运行：

```bash
/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python \
  crates/dev_tools/python/scripts/unitree_g1_static_placement_audit.py \
  --definition /home/ethan/models/unitree_g1/task_assets/20261001_frozen/native_task_objects_t1_60_diagnostic_v2.json \
  --trace .scratch/g1_station_new_run/owner_steps.jsonl --output .scratch/g1_station_audit.json

/home/ethan/models/unitree_g1/envs/policy_gr00t_n16_cu13/bin/python \
  crates/dev_tools/python/scripts/unitree_g1_mobile_placement_audit.py \
  --definition /home/ethan/models/unitree_g1/task_assets/20261001_frozen/native_task_objects_t1_60_diagnostic_v2.json \
  --trace .scratch/g1_mobile_new_run/owner_steps.jsonl --output .scratch/g1_mobile_audit.json
```

## 科学站 T2 抓箱迁移

同日 episode 20622 在原生科学站完成了单次抓箱诊断：四次真实本体相机图片和原配 N1.6 新推理、200 次单独 20 ms 积分及身体控制更新，末帧箱子中心抬高 0.143950 m。独立日志确认正接触来自左右手、箱子已脱离货架，最低站立余弦为 0.993949。末段有 42 Tick 的抬高／手接触样本；这不等于已验证长期持物或搬运。活跃仿真／墙钟比为 0.995166，控制误期与积压均为零，模型边界暂停继续存在。本轮 Qwen 调用为零，不能当作科学站自然语言闭环或正式 T2 成功。

前置预检执行 100 次单独 50 Hz 积分，机器人站立，箱子／容器位移分别约 0.09／0.47 毫米，并有货架／台车的正支撑冲量。新场景只保留原任务货架和台车的六个碰撞体及对应六个网格，源几何、质量和材质保留；省略原仓库其余几何和电钻。旧视觉导出遗漏的台车桌面已从同一冻结 USD 补回，其源材质明确设置纹理亮度为零和颜色偏移 0.11。其余 122 个原视觉网格、材质和 UV 经逐项比较一致。原版完整仓库入口仍拒绝该科学站场景。

本轮总世界为 58 bodies／2,613 colliders／52 multibody joint handles，含 2,553 个同源科学站碰撞体；原宽平面在零 Tick 时移除。原 Homie v2／N1.6、动作映射、增益、质量、步长和图像有效期均保持，4 PGS 是独立的开发诊断，未增加时间子步或正式执行能力。

源码：`76707ce5266e147e83f4698126193ef2b01c53cf`；应用 SHA256：`c90662a0aba6c8eb489ac53c14d05e93773e3ea5029b5675a446f1a9ff42130e`。原片 SHA256：`bb0329881b3d3005aceb5d7de96a840eed2b84265c95e63ee550cc0eb8dee8f8`，327 帧／13.08 秒，恒定帧率采样重复一帧。首次绘制前显示了窗口占位时钟；公开片段仅使用原片 5.88–13.08 秒的实际应用画面。新增 GIF 为 960×540、90 帧、7.20 秒原速节选，完整窗口只做尺寸缩小和采样，没有补造或延长动作；两段 12 秒主页 GIF 保持。

新视觉导出 SHA256：`69129a51ae7fd53ddf9a00cce7170bc780e4d0f4fceff5c720cbe359db7eee9e`，仍绑定原背景物理 `0ecc6d502967d7cca82bd208b2338dedf86505c781bf5d47551b186fa73b86e6`。外部证据位于 `/home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_t2_fixture_milestone_001/`，包含冻结配置、独立审计、原速录像与源资产。复用上述 N1.6 服务启动命令后运行：

```bash
./target/debug/bevy_sim2sim --scene g1_camera_diagnostic --robot g1 \
  --g1-config /home/ethan/ProjectBackups/2026-10-04/Sai_Lab/g1_station_t2_fixture_milestone_001/evidence/g1_station_t2_grasp_live_20261004_0422/config.json \
  --g1-ticks 200 --output .scratch/g1_station_t2_new_grasp
```

该独立入口仅接受零 Tick 相机初始化或 200 Tick／四块新动作的抓取诊断；不开放科学站搬运、预取或 Qwen 调度。模型／本体的准备和停止沿用上述不同 profile 的命令；新视觉资产的源导出工具是 `crates/dev_tools/python/scripts/unitree_g1_background_visual_export.py` 的 `--restore-station-tabletop` 选项，不能覆盖原仓库视觉制品。

## 仍需完成

T1/T2 正式合同与控制配置、各 5 组初始位置 × 2 个种子的至少 8/10、科学站搬箱、从初始指令开始的 Qwen 调度、全程连续 1×、故障恢复、重置与过期隔离、低坡／门槛、共存性能及完整回归仍需验收。中文 UI 目前只开放已验证观察和停止，正式执行／导航按钮不随阶段合入而解锁。主干中的成功样例也不能替代这些要求。
