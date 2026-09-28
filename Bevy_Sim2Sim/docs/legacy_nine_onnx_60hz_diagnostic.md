# 九个原始 ONNX 在科学站的 60 Hz 零样本诊断

更新日期：2026-09-28。此记录是运行中的 **Sim2Sim 诊断**，不是九项技能或目标物理的验收结果。原策略在旧 Unity/Godot 路线中以 200 Hz 物理、50 Hz 策略运行；本次直接把其原始 ONNX 放到单一 Rapier 科学站世界，逐个按 60 Hz 做一次 CPU ONNX Runtime 推理、一次力矩写入和一次真实物理积分。没有为 60/60 重新训练。

## 已实际贯通

- 九个原始 `61→14` ONNX 的输入/输出名称、文件 SHA 和 CPU 推理已核对；每个模型两组输入共 18 次输出与 Python ONNX Runtime 1.29.0 按 f32 位串相同。Rust 入口使用 `ort 2.0.0-rc.13` 加载固定的 ONNX Runtime 1.30.0 ARM64 CPU 库，原生库 SHA256 为 `64e903a43a041240fd6bcffe0ac6d4fea47ef87bf24b9d097801bd00a9612a4b`。
- 两种完整机器人本体在同一个 Rapier 世界导入；站体静态碰撞体也导入同一世界。14 路关节位置/速度来自原生铰链，14 路力矩按原生铰链轴转换为父/子刚体上的等反世界力矩。每步读回唯一世界的机器人位姿，无第二套运动学或手工动画。
- 每个模型完整跑其原场景命令时长。九次独立运行共 **2,400 次真实 ONNX 推理、2,400 次 Rapier 积分**，每次运行的世界数为 1。GPU 从同一轨迹导出 30 FPS、1920×1080 视频，共 1,209 帧；组合视频为 40.3 秒，编码后重新逐帧解码计数为 1,209。
- 运行入口是 `crates/dev_tools/src/bin/station_robot_60hz_diagnostic.rs`；视频入口是 `crates/dev_tools/src/bin/robot_pose_sequence_video.rs`。两者均为开发工具，不是游戏运行时技能系统。每个视频收据核对模型/场景身份、策略/物理步数、轨迹与根位姿结构、采样帧及视频 SHA。不能单凭这种结构核对证明报告生产者可信，须结合原始运行收据。

| 原始模型 | 物理步 / 推理 | 30 FPS 画面 | 本次可见结果 |
| --- | ---: | ---: | --- |
| `alpha_walking` | 360 / 360 | 181 | 约第 119 步明显倾斜，后续倒地；未走成稳定路线 |
| `alpha_stand` | 240 / 240 | 121 | 约第 97 步倾倒，末段部分恢复；非合格站立 |
| `alpha_sitstand` | 480 / 480 | 241 | 完整 8 秒坐起命令段，约第 318 步明显倾斜；非合格坐起 |
| `alpha_ground_pick` | 240 / 240 | 121 | 约第 76 步倾倒；没有可抓取物体的物理交互 |
| `ball_kick_left` | 180 / 180 | 91 | 约第 30 步倾倒；没有实体球 |
| `ball_kick_right` | 180 / 180 | 91 | 约第 25 步倾倒；没有实体球 |
| `roller` | 360 / 360 | 181 | 轮式本体移动后约第 100 步倾倒 |
| `roller_crouch` | 180 / 180 | 91 | 根体全段大致保持正向；尚无下蹲目标与接触验收 |
| `roulade` | 180 / 180 | 91 | 明显翻滚；该动作本就要求旋转，不能以正向姿态单指标裁定成功 |

表中的“明显倾斜”是根体上方向与世界上方向点积首次低于 0.5 的诊断标记，不是预定义的技能判据。九份报告里的 `passed=true` **仅表示请求的推理与积分数完整执行**；所有报告均为 `skill_qualified=false`。

## 运行边界

目前执行器是公开标明的诊断 P-only 法则：`τ = clamp(0.55 × (target − q), ±0.6405236)`，`Kd=0`。旧命令相位、动作幅度和头/腿目标滤波按旧场景语义移植到实验性的 60 Hz 时钟，但 BAM 的前次求解负载、动态摩擦、延迟队列与限幅后历史尚未进入此闭环。因此视频不能验证约定的完整物理/推理框架，更不能说明旧 200/50 模型可直接作为新 60/60 策略。站体接触等价和源机器人自碰撞过滤均未通过资格门禁；20 个动态道具碰撞体延后导入，踢球两段无实体球。静态站场画面、模型和场景 SHA 均绑定在原始报告中；地编资产变化后须重新运行并录制。

BAM 源码当前读取上一步 `-qfrc_bias + qfrc_constraint - own_dof_friction_force` 作为外载，并另保存**实际上一步施加的** `qfrc_actuator`。Rapier 可观测的重力/惯性广义力与 joint/contact 冲量只能构成待验证候选，不能直接声称等价。带观测 epoch、真实步长、关节 DOF 映射及实例/episode 身份的上一求解帧已作为诊断实现；冻结本体的第一批 MuJoCo/Rapier 分项对照也已完成，结果仍不足以打开生产 BAM 入口。后续须解决限位语义、源自碰撞过滤、站体和道具接触及实际上一步 actuator torque，再重训或适配 60/60 策略，并按实施计划逐技能验收。

后续受力对照已记录在 [MuJoCo / Rapier 逐关节受力对照](paired_force_comparison.md)：无约束的重力/惯性样本接近，但故意触发关节上限时出现约 `0.03511 N·m` 的缺失约束力和不同的步末状态。因此 BAM 入口继续关闭；这份视频的诊断性质不变。

本地原始证据与媒体不入 Git：`.scratch/legacy_nine_native_probe/report.json`、`nine_full_v2/`、`nine_full_video_v2/video_delivery_manifest.json`、`nine_full_video_v2/nine_onnx_rapier_60hz_zero_shot_diagnostic.mp4`。视频收据包含每段原 ONNX SHA、运行报告 SHA、视频 SHA、原生库 SHA、物理/推理/采样计数及场景身份。MP4 上方逐段显示模型与频率，下方持续注明 P-only、BAM/接触/技能未验收与踢球无实体球。

## 复现接口

运行命令须提供已校验的编译模型、初始 qpos、科学站资产、原始 ONNX 和本机 ARM64 ORT 库；二者的 SHA 参数均为对应文件完整字节的 SHA256。输出必须是新的路径：

```text
cargo run --locked -p dev_tools_minigame --features rendering_preview \
  --bin station_robot_60hz_diagnostic -- \
  MODEL MODEL_SHA QPOS QPOS_SHA ASSETS ORIGINAL.onnx --legacy-original \
  LIBONNXRUNTIME.so TICKS NEW_REPORT.json

cargo run --locked -p dev_tools_minigame --features rendering_preview \
  --bin robot_pose_sequence_video -- \
  MODEL MODEL_SHA APPEARANCE APPEARANCE_SHA ASSETS NEW_REPORT.json \
  NEW_FRAME_DIRECTORY 30 NEW_VIDEO_RECEIPT.json NEW_VIDEO.mp4
```

`TICKS` 分别为表中 360/240/480/240/180/180/360/180/180。复现时应先核对本地源文件身份和安装库；不能把开发命令或这些输出计数当作最终验收入口。
