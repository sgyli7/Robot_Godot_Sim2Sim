# 同力矩移动误差的质量矩阵因果探针

更新日期：2026-09-28。用同一冻结腿式和轮式 `.mjb`、编译定义、`qpos`、四种 `qvel`、14 路力矩与 `1/60 s` 时间步，在直连 `SimulationWorld` 中比较 Rapier 原始自由加速度解和一条**默认关闭的诊断路径**。后者只把 `acc_inv_augmented_mass` 的自由加速度求解换成 Rapier 已构造的 `inv_augmented_mass`，不改广义力、关节约束、接触、时间步或生产控制入口。它关闭此步隐式 Coriolis 能量守卫，因为该守卫只用于原始隐式解。`sim2sim_plain_mass_probe` feature 和探针的 `--plain-mass-diagnostic` 开关必须**同时**启用才会选择诊断矩阵；任一未启用均保持原始物理路径。

比较指标是每个引擎内部的力矩增量 `Δv = v⁺(原14路力矩) − v⁺(零力矩)`，再逐驱动 DOF 比较两引擎的 `Δv`。因此相同初态的重力、速度偏置和无力矩步误差被扣除。全部 32 组单步（两本体 × 四速度状态 × 两矩阵 × 两力矩条件）的冻结输入字节和 SHA 经探针及 MuJoCo 脚本校验。

| 本体 | 初始速度 | 原隐式矩阵最大 `Δv` 差 (rad/s) | 普通矩阵最大 `Δv` 差 (rad/s) | 改善倍数 |
| --- | --- | ---: | ---: | ---: |
| 腿式 | 零速 | `1.288e-6` | `1.288e-6` | `1.0` |
| 腿式 | 仅关节运动 | `0.001185402` | `1.472e-6` | `805` |
| 腿式 | 仅根体运动 | `0.000944384` | `1.496e-6` | `631` |
| 腿式 | 根体及关节运动 | `0.002130061` | `1.416e-6` | `1504` |
| 轮式 | 零速 | `1.533e-6` | `1.533e-6` | `1.0` |
| 轮式 | 仅关节运动 | `0.001198274` | `1.344e-6` | `891` |
| 轮式 | 仅根体运动 | `0.001003576` | `1.457e-6` | `689` |
| 轮式 | 根体及关节运动 | `0.000919636` | `1.262e-6` | `728` |

零速状态两矩阵结果逐项相同。其余六个速度状态下，原始矩阵的零力矩和非零力矩报告均为 `energy_guard_evaluated=true`、`energy_guard_fallback=false`、`energy_guard_acceleration_cleared=false`，实际选用 `implicit_gyro_coriolis_mass`。诊断矩阵全部报告为 `plain_mass_diagnostic`，守卫三个标志均为 `false`。特性开启但未指定开关的八个已有腿/轮非零力矩报告，在移除新增的四个状态字段后与之前 JSON **逐项相同**。这说明诊断没有暗中改变原路径。

本组两个 MuJoCo 源模型均选 Euler 积分器；关节 damping、frictionloss、stiffness 最大值为零，armature 最大值约 `0.0018077433`。32 组报告的初态位置最大差 `1.23e-8 rad`、驱动速度最大差 `6.23e-9 rad/s`、根体速度最大差 `5.96e-9`。实际 `qfrc_actuator` 与所请求力矩逐路完全相同；Rapier 14 路原生 `Jᵀ` 力矩投影最大差 `1.675e-7 N·m`。所有源 `qfrc_constraint`、驱动 DOF Rapier 通用关节冲量及接触冲量均为零。直连世界仍出现五对无冲量的自接触候选，不代表源碰撞过滤已经合格。

因此，本次**单步、无约束碰撞冲量的移动状态**中，主要误差来自 Rapier 隐式陀螺/Coriolis 增强质量矩阵的自由加速度解，而不是力矩映射或关节限幅。换成普通矩阵后剩余约 `1.3–1.5e-6 rad/s` 的差异仍待解释；该诊断不验证多步稳定性、限位、摩擦、站体接触、BAM 外载或九个 ONNX 的可迁移性，不应直接成为生产求解设置。

原始 32 对报告及 `manifest.json` 位于 `.scratch/paired_mass_causal_v1/`；清单 SHA256 为 `fb41b875f2b2d852847c660b27fd572ab21fc6a03251b0f1ecf18711133c8ed8`。`*_rapier.json` 含实际矩阵与守卫标志，`*_comparison.json` 含同源 MuJoCo 对照。原模型、姿态、速度和力矩的 SHA 写在每份报告中。复现时在同一已审计的冻结输入上各运行一次普通探针和开关探针，再用现有比较脚本核验：

```text
cargo build --locked -p dev_tools_minigame --features sim2sim_plain_mass_probe --bin paired_force_probe
target/debug/paired_force_probe [--plain-mass-diagnostic] \
  MODEL MODEL_SHA QPOS QPOS_SHA QVEL QVEL_SHA \
  [TORQUES TORQUES_SHA] RAPIER_REPORT.json
python crates/dev_tools/python/scripts/compare_paired_force.py \
  SOURCE.mjb MJB_SHA QPOS QPOS_SHA RAPIER_REPORT.json COMPARISON.json \
  --qvel QVEL --qvel-sha256 QVEL_SHA \
  [--torques TORQUES --torques-sha256 TORQUES_SHA]
```

`paired_force_probe` 不经过 `SourceCollisionWorld`，没有消耗其受保护的积分预算。所有报告继续标记 `bam_external_load_qualified=false`。
