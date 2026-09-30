# `alpha_stand` 倒地：首轮隔离证据

更新：2026-09-28。**当前 Bevy 录像中的 Idle 判定失败，不能交付。** 原始 `alpha_stand.onnx` 在 v18AJ 科学站的 60 Hz Rapier／60 Hz ORT／P-only 诊断中，起步即明显摆动；第 114 Tick（1.90 s）根体直立点积首次低于旧场景的 0.5 门槛，第 129 Tick（2.15 s）低于 0；第 240 Tick（4.0 s）水平漂移约 0.282 m，超过旧场景 0.2 m 门槛。最低直立点积约 −0.845。逐帧画面见[九模型失败诊断录像](legacy_nine_onnx_aj_fixed_step_video.md)，本地分析输入为 `.scratch/legacy_nine_native_probe/fixed_step_driver_video_v1/cpu/alpha_stand.json`（SHA256 `9405ce220d973488c3d133776b6c308a00023269293c6dde7d7fc3306467ea`）。

为隔离“仅仅把 200/50 改成 60/60 就会倒”这个解释，用同一原始 ONNX、同一旧 MuJoCo `scene.xml` 本体、零命令、HOME、动作幅度和头／腿目标滤波，分别运行完整 4 秒。只改变 MuJoCo 物理步长与策略 decimation；均不调用 BAM。结果：

| 同一旧 MuJoCo 本体 | 最低直立点积 | 4 秒水平漂移 | 旧 Idle 门槛 |
| --- | ---: | ---: | --- |
| 200 Hz 物理／50 Hz 策略 | 0.999666 | 3.506 mm | 通过 |
| 60 Hz 物理／60 Hz 策略 | 0.999605 | 3.440 mm | 通过 |
| Bevy 冻结 BAM 本体／Rapier／60/60 P-only | −0.845471 | 281.932 mm | **失败** |

源端 60/60 与 Bevy 首个策略 Tick 的 61 维观测、14 维动作、原始／滤波目标及关节位置／速度逐元素一致；源端实际 actuator force 与 Bevy 命令力矩最大差 `3.73e−9 N·m`。**首个分叉发生在第一次物理积分，而不是第一次 ONNX 推理。** 首步源端 `ncon=0`，Bevy 报告有 9 个 active contact pairs；这两个计数来自不同引擎，不能直接解释为九对多余的等价接触。首步根体高度相差约 0.063 mm；第二次推理前的关节角速度最大差 `0.25257 rad/s`，随后动作最大差 `0.02718`。

这个对照只排除“频率变化单独造成 Idle 倒地”，**没有证明单一根因是接触**：旧源端使用 plain XML 本体（`scene.xml` SHA `0d2f58…4ef1`，active joint damping `0.053`、frictionloss `0.0048`，位置执行器 gain `0.55`），Bevy 视频使用 SHA `54439af…09dc` 的冻结 BAM 编译本体（active joint damping／frictionloss 均为 0），且该 P-only 诊断没有调用 BAM。`RobotAssembly` 确实导入编译定义中的 damping／frictionloss／armature；本次冻结定义的零值和未接入 BAM 才是具体缺口。站体地面、接触法则、控制 plant 与 MuJoCo plain XML 也并非同一条件。

第二轮已使用视频所用**同一冻结 BAM 编译本体**（原 `.mjb` SHA `832e1f08…bdc47c`；有效 XML／38 个 SHA 核定资源重编，仅加一个 `z=0` 平面）在 MuJoCo 中，以相同初态、原 ONNX、60/60、同一 `τ=clamp(0.55×(target−q), ±0.6405236)`、**无 BAM** 运行 4 秒。MuJoCo 的 Idle 也失败：1.883 秒首次低于 0.5，最低直立点积 −0.953，最终直立点积 0.034、水平漂移 0.630 m；Bevy 分别是 1.900 秒、−0.845、0.715。首 Tick 的观测、动作、目标、关节状态和力矩逐元素相同，第二 Tick 观测最大差仅 0.00263。因平面和站体碰撞网格不同、有效 XML 重编也不是 `.mjb` 逐字节复刻，不能据此宣布 Rapier 接触等价；但**保留 MuJoCo 求解器仍倒地**，说明不能优先把这段 Idle 崩溃归罪于 Rapier。完整前五步和模型字段对照在本地 `bam_compiled_p_only_source_isolation_v1.json`，SHA256 `5812579c1d62771cb47a41430a912d2c909f693515002672ecd5acb53f1f22ea`。

再在同一 BAM 编译本体、同一 MuJoCo 平面和 P-only 闭环里，只改 14 路关节被动黏性阻尼／干摩擦，做因果反事实：

| 仅此隔离实验的参数 | 首次低于直立门槛 | 4 秒最低直立点积 | 4 秒水平漂移 |
| --- | ---: | ---: | ---: |
| 冻结值 `0 / 0` | 1.883 s | −0.953 | 630 mm |
| 只填 BAM 参数中的黏性数值 `0.005359668 / 0` | 1.283 s | −0.963 | 1,068 mm |
| 只恢复旧 plain XML 黏性 `0.053 / 0` | 未跌破 | 0.999617 | 3.455 mm |
| 只恢复旧干摩擦 `0 / 0.0048` | 2.000 s | 0.362798 | 411 mm |
| 恢复旧 plain XML 两项 `0.053 / 0.0048` | 未跌破 | 0.999623 | 3.441 mm |

这证明**缺少足够的速度相关被动／电机反馈是本隔离里使 Idle 崩溃的关键因素**；不表示把旧 `0.053` 硬写入冻结 BAM 本体就是正式修复。BAM 的黏性数值更小，实际控制还包括电机反电势、电压、负载、干摩擦与延迟；目标端须实现并对照完整执行器链，同时保持源接触和目标求解的独立门禁。五格原始前五步与四秒结果在本地 `bam_passive_counterfactual_v1.json`，SHA256 `e6c936b294404ddf999b98c15978905410de755e260a9d66d68ea0e9d4de0b57`。

新增独立 [Idle 行为门禁工具](../crates/dev_tools/python/src/bevy_microduck_tools/legacy_idle_gate.py)，从整段原始位姿重新计算旧 4 秒门槛。原视频报告虽然 `passed=true`（仅指诊断执行成功），独立门禁明确输出 `behavior_passed=false`，第一次跌破门槛为第 114 Tick、末段漂移 0.282 m；结果在本地 `.scratch/legacy_nine_native_probe/alpha_stand_behavior_gate_v1.json`，SHA256 `372937a01fb5eeaa6e489acb412a7e7f7a932a94a0d0f4ce91973ef558a4e2e7`。工具对完整稳定轨迹、翻倒、截短与错位步号有四项测试；通过旧门槛也不授予新 60/60 技能资格。

三个隔离实验均有同名 `.py` 和 `.json` 保存在 `.scratch/legacy_nine_native_probe/`：`source_idle_frequency_isolation_v1`、`bam_compiled_p_only_source_isolation_v1`、`bam_passive_counterfactual_v1`。所用原始 ONNX SHA、源模型和 Bevy 报告身份都在各自输出；三项均由根 Agent 独立复跑且输出逐字节相同。所有 scratch 实验均不修改正式物理和策略，也不给 BAM 或九技能授予资格。
