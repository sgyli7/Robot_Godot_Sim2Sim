# 非零力矩与诊断软限位单步对照

更新日期：2026-09-28。此项在[单行软限位诊断](source_limit_row_probe.md)之后，检验实际执行器广义力与软限位行同时参与一次 `1/60 s` 求解时的结果。它只在默认关闭的 `sim2sim_source_limit_probe` 特性及显式 CLI 开关下运行；没有改动正式机器人定义或生产求解路径。

脚本 `crates/dev_tools/python/scripts/compare_source_limit_with_torque.py` SHA 核对腿式／轮式原编译 `.mjb`、两份带原生凸包与源限位参数的 scratch 定义、上限 +`0.02 rad` 且零初速的 `qpos/qvel`，然后在两侧分别执行相同的一步。每种本体测三个输入：既有 14 路非零力矩向量 `[0.12,-0.08,0.05,-0.11,0.16,-0.09,0.07,-0.13,0.04,0.15,-0.06,0.10,-0.14,0.03] N·m`，以及只有 `head_roll` 为 `+0.12` 或 `−0.12 N·m`。源端使用 MuJoCo 3.10.0 的真实 `mj_step`；目标端通过 `actuator_body_torques` 实际写入力矩，并从完成的 Rapier 行读取最终冲量。源端每次均为单条限位约束、零接触；目标端仍有既知的自碰撞窄相候选，但这六步所有接触冲量为零。这里的“零接触”只指**求解冲量**，不表示接触拓扑等价。

| 形态 | 14 路向量：限位力差 | 单关节正力矩：限位力差 | 单关节反力矩：限位力差 | 三组最大步末角度差 | 三组最大步末角速度差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 腿式 | `6.247e−8 N·m` | `1.728e−7 N·m` | `0` | `4.231e−8 rad` | `1.524e−6 rad/s` |
| 轮式 | `4.483e−8 N·m` | `1.702e−7 N·m` | `0` | `4.932e−8 rad` | `1.706e−6 rad/s` |

完整 14 路输入时，源限位力分别是腿式 `−0.0747785022 N·m`、轮式 `−0.0748592303 N·m`；反向单关节力矩使活动限位行的求解力归零，两侧均为零。六组源执行器读回与输入完全一致；目标原生 `Jᵀ` 投影最大差小于 `1e−6 N·m`。14 路的总外载候选最大差不超过 `3.611e−7 N·m`，但这仍不能认定 BAM 外载等价。六组最大步末角速度差（所有驱动关节）为 `1.706e−6 rad/s`。

脚本逐组拒绝输入 SHA 漂移、错误开关、非单步、非零接触冲量、源接触、错误 actuator 投影，以及限位力／终态超出预设门槛。两次完整重跑的汇总文件逐字节相同，`.scratch/source_limit_actuated_v1/summary.json` SHA256 为 `d9cbb4f2d835e0840c3073d1408b2cfe4dc06184bf2aaef77c00c0dba854b55a`；同目录留存六组目标原始报告和 MuJoCo/Rapier 对照报告。复现命令（Python 须提供 MuJoCo 3.10.0）：

```sh
cargo build --locked -p dev_tools_minigame --features sim2sim_source_limit_probe,sim2sim_plain_mass_probe --bin paired_force_probe
python crates/dev_tools/python/scripts/compare_source_limit_with_torque.py
```

这只覆盖冻结初态下的单条关节软限位。连续多步、多行同时激活、接触耦合、干摩擦、站体与道具、BAM 上一步外载和九个 ONNX 技能仍须分别验收。正式定义未重导出，生产 `previous_solve_load` 仍为 `Unavailable`。
