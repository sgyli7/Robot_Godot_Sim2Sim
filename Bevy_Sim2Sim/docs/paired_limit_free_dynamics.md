# head_roll 自由动力学与源软限位预测

更新日期：2026-09-28。此诊断只用冻结的腿式和轮式 MicroDuck、原始编译 MuJoCo `.mjb`，检查 `head_roll` 上限附近的**约束前自由动力学**。每种本体有 7 个上限深度 × 3 个初始关节速度，共 42 个源状态；每个状态在 Rapier 直连本体世界分别运行默认隐式质量路径与默认关闭的纯质量诊断路径，共 84 次单步。所有案例为零 actuator torque、`1/60 s`、无非零接触冲量。没有调用 `SourceCollisionWorld`，没有消耗其受保护的 128 步预算。

`paired_force_probe --head-roll-free-acceleration-diagnostic` 在唯一一次实际积分后，只读 Rapier 保留的 `generalized_acceleration()` 和 `inv_augmented_mass()`。前者是该步约束求解前计算的自由加速度；后者对 `head_roll` 单位广义力求解并读取该 DOF，得到当前姿态的有效逆惯量。探针核对观测 epoch、一次 `1/60 s` 积分、完整本体接触冲量为零、有限且正的逆惯量，以及输入和初始 `head_roll` 位置/速度身份；开关默认关闭，不改生产物理。两种模式各选一例，移除新增诊断字段后与同一二进制不带该只读开关的完整 JSON **逐项相同**，包含步前/步末 q/v。

源侧独立读取 SHA 锁定的 `.mjb` 和 [源限位矩阵](limit_law_investigation.md)，用 `mj_forward` 的 `qacc_smooth`、当前质量矩阵、编译的 `jnt_solref`、`jnt_solimp`、`dof_invweight0` 重建单行公式 `f=max(0,(a_ref-a₀)/(A+R))`，其中 `a₀=Jqacc_smooth`、`A=JM⁻¹Jᵀ`、`J=-e_head_roll`。源端 30 个激活行的重建力与先前源求解报告误差小于 `1e-10 N·m`。随后只替换公式中的 `a₀` 和 `A` 为目标读数；`a_ref` 与 `R` 仍取源模型参数。这一步是**假设将源软限位法则接入目标**时的预测，不是 Rapier 当前限位的实际广义力。

| 比较指标：42 个状态最大绝对误差 | 默认隐式质量 | 纯质量诊断 |
| --- | ---: | ---: |
| 约束前行自由加速度 `a₀` | `7.283e-7 rad/s²` | `7.283e-7 rad/s²` |
| 当前有效逆惯量 `A` | `7.297e-5 rad/s²/N·m` | `7.297e-5 rad/s²/N·m` |
| 按源软限位法则预测的行力 | `8.601e-9 N·m` | `8.616e-9 N·m` |
| **目标实际**步末关节位置与源之差，激活行 | **`0.0150002 rad`** | **`0.0150002 rad`** |
| **目标实际**步末关节速度与源之差，激活行 | **`0.299982 rad/s`** | **`0.299982 rad/s`** |

这组数字把阻断点收窄到限位法则及求解/积分接法：冻结状态下，自由加速度和单行有效逆惯量已足以把**预测**源限位力做到接近源值，但现有 Rapier 硬限位仍把位置钳到上限、速度清零。两条质量路径在这 42 个只动一个关节且零力矩的状态几乎一致；不能以此取代先前在全本体运动、非零 actuator torque 下观察到的质量路径差异，也不能据此开启 BAM 外载。

复现命令（从 `Bevy_Sim2Sim` 根目录执行；Python 须有 MuJoCo 3.10.0）：

```sh
cargo build --locked -p dev_tools_minigame --features sim2sim_plain_mass_probe --bin paired_force_probe
python crates/dev_tools/python/scripts/compare_head_roll_free_dynamics.py
```

脚本核对两个 `.mjb`、两个原生碰撞凸包定义、两个源 `qpos` 和两个源矩阵的 SHA，再为每个案例生成 SHA 绑定输入和原始目标报告；最终本机报告为 `.scratch/paired_limit_target_v1/comparison.json`，SHA256 `7c6e74ebd3250633b8b82b75e0e3ec3191ca570e1418f125a9a9b213b4ba20f0`。报告包含 84 个逐案例输入、读数、预测、误差和目标原始 JSON SHA。`target_limit_qualified=false`、`multi_row_or_contact_qualified=false`、`bam_external_load_qualified=false`：源软限位尚未在同一个真实目标求解器中落地；多约束、接触、持续轨迹和 BAM 历史均未验收。
